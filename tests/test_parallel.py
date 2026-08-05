"""Tier-gated parallel episodes.

Episodes inside a tier may run concurrently, but a harder tier must not open
while an easier one is unfinished -- otherwise a run is no longer evaluated
in difficulty order. The serial protocol (parallel=1) must be untouched;
tests/test_server.py covers that path and is the backwards-compatibility
guard.
"""

import importlib
import json
import os

import numpy as np
import pytest

pytest.importorskip("fastapi", reason="server extras not installed")
from fastapi.testclient import TestClient  # noqa: E402

from latent_sokoban.levels import generate_level  # noqa: E402

TIER_SIZES = ((1, 4), (2, 3))          # (crates, count) -> tiers of 4 then 3
N_LEVELS = sum(n for _, n in TIER_SIZES)


@pytest.fixture
def server(tmp_path):
    # Function-scoped and reloaded per test on purpose. server.app keeps its
    # levels in module globals, and importlib.reload mutates the module dict
    # in place -- so a module-scoped fixture here would have tests/test_server.py
    # redefine LEVELS underneath this file's client depending on run order.
    tmp = tmp_path
    rng = np.random.default_rng(1)
    levels = []
    for crates, n in TIER_SIZES:
        for _ in range(n):
            lv, sol = generate_level(rng, size=8, n_boxes=crates,
                                     wall_density=0.10, min_solution_len=3,
                                     max_solution_len=12)
            levels.append({"ascii": lv.to_ascii(), "optimal_len": len(sol),
                           "n_crates": crates, "max_steps": 3 * len(sol)})
    (tmp / "levels.json").write_text(json.dumps({"levels": levels}))
    os.environ["SOKOBAN_LEVELS"] = str(tmp / "levels.json")
    os.environ["SOKOBAN_DB"] = str(tmp / "db.sqlite")
    import server.app as app
    importlib.reload(app)
    return app, TestClient(app.app)


def _session(client, parallel):
    key = client.post("/api/keys", json={"name": f"p{parallel}"}).json()["api_key"]
    h = {"X-API-Key": key}
    sid = client.post("/api/scorecards", headers=h).json()["scorecard_id"]
    frame = client.post(
        f"/api/scorecards/{sid}/games/standard/start?parallel={parallel}",
        headers=h).json()
    return h, sid, frame


def test_tiers_follow_crate_counts(server):
    app, _ = server
    assert app.TIERS == [(0, 4), (4, 7)]


def test_whole_tier_opens_at_once(server):
    _, client = server
    _, _, frame = _session(client, 4)
    assert [e["index"] for e in frame["episodes"]] == [0, 1, 2, 3]
    assert frame["tier"] == {"index": 0, "of": 2, "range": [0, 4]}


def test_episode_id_required_when_several_in_flight(server):
    _, client = server
    h, _, frame = _session(client, 3)
    r = client.post(f"/api/sessions/{frame['session_id']}/act",
                    json={"action": 0}, headers=h)
    assert r.status_code == 400
    r = client.post(f"/api/sessions/{frame['session_id']}/act",
                    json={"action": 0, "episode": 99}, headers=h)
    assert r.status_code == 409


def test_later_tier_waits_for_earlier_one(server):
    _, client = server
    h, _, frame = _session(client, 4)
    gid = frame["session_id"]
    rng = np.random.default_rng(0)
    acts = 0
    while frame["state"] != "GAME_OVER" and acts < 5000:
        idxs = [e["index"] for e in frame["episodes"]]
        # the invariant: nothing from tier 2 may be open until all four of
        # tier 1 have been recorded
        if frame["finished"] < 4:
            assert all(i < 4 for i in idxs), "tier 2 opened too early"
        frame = client.post(
            f"/api/sessions/{gid}/act",
            json={"action": int(rng.integers(0, 4)), "episode": idxs[0]},
            headers=h).json()
        acts += 1
    assert frame["state"] == "GAME_OVER"
    assert frame["result"]["episodes"] == N_LEVELS


def test_serial_frames_keep_the_original_shape(server):
    """parallel=1 must produce the pre-existing single-episode frame."""
    _, client = server
    h, _, frame = _session(client, 1)
    assert "episodes" not in frame and "tier" not in frame
    assert frame["episode"]["index"] == 0
    assert frame["episode"]["of"] == N_LEVELS
    assert frame["observation"]["shape"] == [64, 64, 3]
    nxt = client.post(f"/api/sessions/{frame['session_id']}/act",
                      json={"action": 0}, headers=h).json()
    assert nxt["episode"]["steps_used"] == 1
