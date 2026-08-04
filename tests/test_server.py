"""Protocol and abuse-limit tests for the public evaluation API.

The server is imported with SOKOBAN_LEVELS pointed at a small generated set
and SOKOBAN_DB at a temp file, so no test touches the real hidden levels or
the live leaderboard. Levels load at import time, which is why the fixture
sets the environment before importing server.app.
"""

import base64
import importlib
import json

import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi", reason="server extras not installed")
from fastapi.testclient import TestClient  # noqa: E402

from latent_sokoban.env import Level, SokobanEnv  # noqa: E402
from latent_sokoban.levels import generate_level  # noqa: E402


N_LEVELS = 3


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("server")
    rng = np.random.default_rng(1)
    levels, solutions = [], []
    for _ in range(N_LEVELS):
        level, solution = generate_level(rng, size=8, n_boxes=1,
                                         wall_density=0.10,
                                         min_solution_len=3, max_solution_len=10)
        levels.append({"ascii": level.to_ascii(), "optimal_len": len(solution),
                       "n_crates": 1, "max_steps": 3 * len(solution)})
        solutions.append(solution)
    (tmp / "levels.json").write_text(json.dumps({"levels": levels}))

    import os
    os.environ["SOKOBAN_LEVELS"] = str(tmp / "levels.json")
    os.environ["SOKOBAN_DB"] = str(tmp / "test.db")
    import server.app
    app_module = importlib.reload(server.app)
    yield app_module, TestClient(app_module.app), levels, solutions


@pytest.fixture
def client(server):
    app_module, client, _, _ = server
    app_module._sessions.clear()
    app_module._scorecards.clear()
    return client


_ips = iter(f"203.0.113.{i}" for i in range(1, 255))


def register(client, name):
    """Mint a key from a fresh client IP, so the per-IP cap (covered on its
    own below) does not bound how many keys the other tests may create."""
    r = client.post("/api/keys", json={"name": name},
                    headers={"X-Forwarded-For": next(_ips)})
    assert r.status_code == 200, r.text
    return {"X-API-Key": r.json()["api_key"]}


def play(client, headers, actions_for):
    """Run a full session; actions_for(episode_index, step) picks each action."""
    sid = client.post("/api/scorecards", headers=headers).json()["scorecard_id"]
    frame = client.post(f"/api/scorecards/{sid}/games/standard/start",
                        headers=headers).json()
    gid = frame["session_id"]
    step = 0
    while frame["state"] != "GAME_OVER":
        ep = frame["episode"]["index"]
        frame = client.post(f"/api/sessions/{gid}/act", headers=headers,
                            json={"action": actions_for(ep, step)}).json()
        step = 0 if (frame.get("last") or {}).get("episode_done") else step + 1
    return sid, gid, client.post(f"/api/scorecards/{sid}/close",
                                 headers=headers).json()


# ------------------------------------------------------------------ protocol --

def test_key_is_required(client):
    assert client.post("/api/scorecards").status_code == 401
    assert client.post("/api/scorecards",
                       headers={"X-API-Key": "lsk-nope"}).status_code == 401


def test_duplicate_name_rejected(client):
    register(client, "duplicate-probe")
    ip = {"X-Forwarded-For": "198.51.100.7"}
    assert client.post("/api/keys", json={"name": "duplicate-probe"},
                       headers=ip).status_code == 409
    assert client.post("/api/keys", json={"name": "x"},
                       headers=ip).status_code == 422


def test_key_creation_is_capped_per_ip(client, server):
    app_module, _, _, _ = server
    ip = {"X-Forwarded-For": "198.51.100.42"}
    for i in range(app_module.MAX_KEYS_PER_IP_PER_DAY):
        assert client.post("/api/keys", json={"name": f"ip-probe-{i}"},
                           headers=ip).status_code == 200
    assert client.post("/api/keys", json={"name": "ip-probe-over"},
                       headers=ip).status_code == 429


def test_observations_match_the_local_renderer(client, server):
    """The bytes on the wire are exactly what latent_sokoban.render produces."""
    from latent_sokoban.render import render, render_goal

    _, _, levels, _ = server
    headers = register(client, "render-probe")
    sid = client.post("/api/scorecards", headers=headers).json()["scorecard_id"]
    obs = client.post(f"/api/scorecards/{sid}/games/standard/start",
                      headers=headers).json()["observation"]

    level = Level.from_ascii(levels[0]["ascii"])
    decode = lambda b: np.frombuffer(base64.b64decode(b), dtype=np.uint8) \
                         .reshape(64, 64, 3)
    assert np.array_equal(decode(obs["obs"]), render(level))
    assert np.array_equal(decode(obs["goal"]), render_goal(level))


def test_optimal_play_scores_a_perfect_card(client, server):
    _, _, _, solutions = server
    headers = register(client, "oracle")
    _, _, result = play(client, headers,
                        lambda ep, step: solutions[ep][step])
    score = result["games"]["standard"]
    assert score["episodes"] == N_LEVELS and score["solved"] == N_LEVELS
    assert score["success_rate"] == 1.0
    assert score["move_efficiency"] == 1.0  # optimal play, by construction


def test_unplayed_episodes_count_as_unsolved(client, server):
    """Closing early must not let a run keep only the levels it liked."""
    _, _, _, solutions = server
    headers = register(client, "quitter")
    sid = client.post("/api/scorecards", headers=headers).json()["scorecard_id"]
    gid = client.post(f"/api/scorecards/{sid}/games/standard/start",
                      headers=headers).json()["session_id"]
    for action in solutions[0]:  # solve level 1, then walk away
        client.post(f"/api/sessions/{gid}/act", headers=headers,
                    json={"action": action})
    score = client.post(f"/api/scorecards/{sid}/close",
                        headers=headers).json()["games"]["standard"]
    assert score["episodes"] == N_LEVELS
    assert score["solved"] == 1 and score["success_rate"] == round(1 / N_LEVELS, 4)


def test_session_is_bound_to_its_key(client):
    a, b = register(client, "owner-a"), register(client, "owner-b")
    sid = client.post("/api/scorecards", headers=a).json()["scorecard_id"]
    gid = client.post(f"/api/scorecards/{sid}/games/standard/start",
                      headers=a).json()["session_id"]
    assert client.post(f"/api/sessions/{gid}/act", headers=b,
                       json={"action": 0}).status_code == 404
    assert client.post(f"/api/scorecards/{sid}/close",
                       headers=b).status_code == 404


def test_one_session_per_game_per_scorecard(client):
    headers = register(client, "retry-probe")
    sid = client.post("/api/scorecards", headers=headers).json()["scorecard_id"]
    client.post(f"/api/scorecards/{sid}/games/standard/start", headers=headers)
    assert client.post(f"/api/scorecards/{sid}/games/standard/start",
                       headers=headers).status_code == 409


def test_invalid_action_is_rejected(client):
    headers = register(client, "action-probe")
    sid = client.post("/api/scorecards", headers=headers).json()["scorecard_id"]
    gid = client.post(f"/api/scorecards/{sid}/games/standard/start",
                      headers=headers).json()["session_id"]
    assert client.post(f"/api/sessions/{gid}/act", headers=headers,
                       json={"action": 9}).status_code == 422


def test_frames_are_compressed_for_clients_that_ask(client, server):
    """A frame is ~32KB of base64 that gzips to roughly a fiftieth."""
    headers = register(client, "gzip-probe")
    sid = client.post("/api/scorecards", headers=headers).json()["scorecard_id"]
    path = f"/api/scorecards/{sid}/games/standard/start"

    plain = client.post(path, headers={**headers, "Accept-Encoding": "identity"})
    assert "content-encoding" not in plain.headers  # unchanged for old clients

    sid2 = client.post("/api/scorecards", headers=headers).json()["scorecard_id"]
    zipped = client.post(f"/api/scorecards/{sid2}/games/standard/start",
                         headers={**headers, "Accept-Encoding": "gzip"})
    assert zipped.headers.get("content-encoding") == "gzip"
    # httpx decodes .content transparently, so the wire size has to come from
    # the header the middleware set
    on_the_wire = int(zipped.headers["content-length"])
    assert on_the_wire < len(plain.content) / 5, (
        f"{on_the_wire}B compressed vs {len(plain.content)}B plain")
    # and the payload the agent sees is byte-identical either way
    assert zipped.json()["observation"]["obs"] == plain.json()["observation"]["obs"]


def test_goal_image_is_stable_across_an_episode(client, server):
    """It is rendered once per episode now; it must still be the same bytes
    every frame, and must change when the next episode starts."""
    from latent_sokoban.render import render_goal

    _, _, levels, solutions = server
    headers = register(client, "goal-probe")
    sid = client.post("/api/scorecards", headers=headers).json()["scorecard_id"]
    frame = client.post(f"/api/scorecards/{sid}/games/standard/start",
                        headers=headers).json()
    gid = frame["session_id"]
    first = frame["observation"]["goal"]
    assert first == base64.b64encode(
        render_goal(Level.from_ascii(levels[0]["ascii"])).tobytes()).decode()

    for action in solutions[0][:-1]:      # stay inside episode 0
        frame = client.post(f"/api/sessions/{gid}/act", headers=headers,
                            json={"action": action}).json()
        assert frame["observation"]["goal"] == first

    frame = client.post(f"/api/sessions/{gid}/act", headers=headers,
                        json={"action": solutions[0][-1]}).json()
    assert frame["last"]["episode_done"] and frame["episode"]["index"] == 1
    assert frame["observation"]["goal"] != first  # episode 1's own goal
    assert frame["observation"]["goal"] == base64.b64encode(
        render_goal(Level.from_ascii(levels[1]["ascii"])).tobytes()).decode()


def test_spec_describes_the_loaded_levels(client, server):
    _, _, levels, _ = server
    game = client.get("/api/spec").json()["games"]["standard"]
    assert game["episodes"] == len(levels)
    assert game["crates_per_level"] == "1-1"


# -------------------------------------------------------------------- limits --

def test_opening_a_scorecard_counts_against_the_daily_cap(client, server):
    """Abandoning a run must not buy a free retry: opens are what count."""
    app_module, _, _, _ = server
    headers = register(client, "limit-probe")
    cap = app_module.MAX_SCORECARDS_PER_KEY_PER_DAY
    for i in range(cap):
        assert client.post("/api/scorecards", headers=headers).status_code == 200
    assert client.post("/api/scorecards", headers=headers).status_code == 429


def test_active_sessions_are_capped_per_key(client, server):
    app_module, _, _, _ = server
    headers = register(client, "session-probe")
    for _ in range(app_module.MAX_SESSIONS_PER_KEY):
        sid = client.post("/api/scorecards", headers=headers).json()["scorecard_id"]
        assert client.post(f"/api/scorecards/{sid}/games/standard/start",
                           headers=headers).status_code == 200
    sid = client.post("/api/scorecards", headers=headers).json()["scorecard_id"]
    assert client.post(f"/api/scorecards/{sid}/games/standard/start",
                       headers=headers).status_code == 429


def test_replay_actions_are_owner_only(client, server):
    """A published action string would be a replayable perfect score."""
    _, _, _, solutions = server
    owner = register(client, "replay-owner")
    other = register(client, "replay-other")
    _, gid, _ = play(client, owner, lambda ep, step: solutions[ep][step])

    public = client.get(f"/api/replays/{gid}").json()
    assert public["owner_view"] is False
    assert all("actions" not in ep for ep in public["episodes"])
    assert [ep["solved"] for ep in public["episodes"]] == [True] * N_LEVELS

    assert client.get(f"/api/replays/{gid}", headers=other).json()["episodes"] \
        == public["episodes"]

    mine = client.get(f"/api/replays/{gid}", headers=owner).json()
    assert mine["owner_view"] is True
    assert all(ep["actions"] for ep in mine["episodes"])


def test_leaderboard_shows_only_closed_scorecards(client, server):
    _, _, _, solutions = server
    headers = register(client, "board-probe")
    client.post("/api/scorecards", headers=headers)  # opened, never closed
    before = client.get("/api/leaderboard").json()
    assert not any(e["name"] == "board-probe" for e in before["entries"])

    play(client, headers, lambda ep, step: solutions[ep][step])
    entry = next(e for e in client.get("/api/leaderboard").json()["entries"]
                 if e["name"] == "board-probe")
    assert entry["success_rate"] == 1.0
    assert entry["solved"] == N_LEVELS and entry["episodes"] == N_LEVELS


def test_replayed_actions_reproduce_the_score(client, server):
    """The hidden set is fixed and deterministic, which is exactly why the
    action strings above have to stay private."""
    _, _, levels, solutions = server
    for entry, solution in zip(levels, solutions):
        env = SokobanEnv(Level.from_ascii(entry["ascii"]),
                         max_steps=entry["max_steps"])
        env.reset()
        for action in solution:
            env.step(action)
        assert env.solved
