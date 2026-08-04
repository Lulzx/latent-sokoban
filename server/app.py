"""Public evaluation API + leaderboard for the Latent Sokoban Challenge.

ARC-AGI-3-style agent protocol:

    1. POST /api/keys                     one-time: get an API key (identity)
    2. POST /api/scorecards               open a scorecard (a submission)
    3. POST /api/scorecards/{sid}/games/{game}/start
                                          start a game session -> first frame
    4. POST /api/sessions/{gid}/act       send an action        -> next frame
    5. POST /api/scorecards/{sid}/close   lock it; results hit the leaderboard

Levels never leave the server. Agents receive rendered 64x64 RGB
observations only. Closing a scorecard counts every unplayed or
unfinished episode as unsolved, so partial runs can't cherry-pick.
The exact wire schema lives in docs/API.md; machine-readable OpenAPI
at /api/openapi.json.

Run locally:
    SOKOBAN_SEED=123 uvicorn server.app:app --port 8321

Environment:
    SOKOBAN_DB      sqlite path            (default server/leaderboard.db)
    SOKOBAN_LEVELS  hidden split json      (default server/hidden_levels.json,
                                            generated from SOKOBAN_SEED if absent)
    SOKOBAN_SEED    generation seed for the hidden set (required on first boot)
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import secrets
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latent_sokoban.env import Level, SokobanEnv
from latent_sokoban.render import render, render_goal
from latent_sokoban.solver import is_deadlocked

ROOT = Path(__file__).resolve().parent
DB_PATH = Path(os.environ.get("SOKOBAN_DB", ROOT / "leaderboard.db"))
LEVELS_PATH = Path(os.environ.get("SOKOBAN_LEVELS", ROOT / "hidden_levels.json"))

GAMES = ("standard",)
SESSION_TTL = 3600
MAX_ACTIVE_SESSIONS = 200
MAX_SESSIONS_PER_KEY = 3
MAX_KEYS_PER_IP_PER_DAY = 5
# Counts scorecards *opened*, not closed. Opening is what exposes the hidden
# levels, so an uncapped open is free reconnaissance no matter how few runs
# are ever submitted.
MAX_SCORECARDS_PER_KEY_PER_DAY = 24
NAME_RE = re.compile(r"^[\w .\-]{2,40}$")

app = FastAPI(title="Latent Sokoban Challenge", docs_url="/api/docs",
              openapi_url="/api/openapi.json")

# Every frame carries two base64 observations, about 32 KB, and roughly half
# of that is the goal image, which is identical for the whole episode. The
# observations are synthetic renders of flat tiles in few colours, so they
# compress about fiftyfold: a frame goes from 32 KB to under 1 KB for a
# fraction of a millisecond. Applies only when the client advertises
# Accept-Encoding: gzip, so no existing client changes behaviour.
#
# It also gzips the fonts and sprites, which are already compressed and
# gain nothing (a woff2 goes 40240 -> 40082 bytes). That is about fifteen
# small files on a cold page load against thousands of frames per
# evaluation run, so it is not worth a content-type filter to avoid.
# Range requests are unaffected: a 206 comes back uncompressed with its
# content-range intact.
app.add_middleware(GZipMiddleware, minimum_size=1000)

_lock = threading.Lock()
_sessions: dict[str, dict] = {}       # game sessions in flight
_scorecards: dict[str, dict] = {}     # open scorecards


# ---------------------------------------------------------------- levels --

def _load_levels() -> list[dict]:
    """Load the hidden set, generating it from the seed if absent.

    Generation takes roughly 70s, so the API is unreachable for that long on
    a cold first boot. Run scripts/generate_hidden.py ahead of time when
    that matters; this path exists so a fresh deploy still works unattended.
    """
    if not LEVELS_PATH.exists():
        seed = os.environ.get("SOKOBAN_SEED")
        if not seed:
            raise SystemExit("no hidden level file and SOKOBAN_SEED unset")
        from latent_sokoban.levels import generate_hidden_set
        LEVELS_PATH.write_text(json.dumps(generate_hidden_set(int(seed))))
    return json.loads(LEVELS_PATH.read_text())["levels"]


LEVELS = _load_levels()


def _db() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    with conn:
        conn.execute("""CREATE TABLE IF NOT EXISTS keys (
            key_hash TEXT PRIMARY KEY, name TEXT UNIQUE,
            created REAL, ip TEXT)""")
        # A row is written when a scorecard is *opened*, with closed NULL and
        # no metrics; close fills the rest in. Only rows with closed set are
        # results; the rest are attempts, which is what the daily cap counts.
        conn.execute("""CREATE TABLE IF NOT EXISTS scorecards (
            id TEXT PRIMARY KEY, name TEXT, opened REAL, closed REAL,
            episodes INTEGER, solved INTEGER, success_rate REAL,
            move_efficiency REAL, avg_steps_solved REAL, deadlock_rate REAL,
            total_actions INTEGER)""")
        conn.execute("""CREATE INDEX IF NOT EXISTS scorecards_name_opened
            ON scorecards (name, opened)""")
        conn.execute("""CREATE TABLE IF NOT EXISTS replays (
            session_id TEXT PRIMARY KEY, scorecard_id TEXT, name TEXT,
            game TEXT, finished REAL, episodes_json TEXT)""")
    return conn


# ---------------------------------------------------------------- helpers --

def _hash(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def _client_ip(request: Request) -> str:
    fwd = request.headers.get("x-forwarded-for")
    return fwd.split(",")[0].strip() if fwd else (request.client.host or "?")


def _name_for_key(x_api_key: str | None) -> str | None:
    """The leaderboard name bound to a key, or None if it is absent/unknown."""
    if not x_api_key:
        return None
    conn = _db()
    row = conn.execute("SELECT name FROM keys WHERE key_hash=?",
                       (_hash(x_api_key),)).fetchone()
    conn.close()
    return row[0] if row else None


def _auth(x_api_key: str | None) -> str:
    """Returns the leaderboard name bound to the key."""
    if not x_api_key:
        raise HTTPException(401, "missing X-API-Key header")
    name = _name_for_key(x_api_key)
    if name is None:
        raise HTTPException(401, "unknown API key")
    return name


def _b64(img: np.ndarray) -> str:
    return base64.b64encode(
        np.ascontiguousarray(img, dtype=np.uint8).tobytes()).decode()


def _frame(session: dict, last: dict | None = None) -> dict:
    level: Level = session["level"]
    env: SokobanEnv = session["env"]
    state = "GAME_OVER" if session["done"] else "IN_PROGRESS"
    frame = {
        "session_id": session["id"],
        "game": session["game"],
        "state": state,
        "episode": {
            "index": min(session["episode"], len(LEVELS) - 1),
            "of": len(LEVELS),
            "steps_used": env.state.steps,
            "max_steps": env.max_steps,
        },
        "observation": None if session["done"] else {
            "obs": _b64(render(level, env.state)),
            # constant for the episode, so rendered and encoded once at its
            # start rather than on every action
            "goal": session["goal_b64"],
            "shape": [64, 64, 3],
            "dtype": "uint8",
            "encoding": "base64-raw",
        },
        "last": last,
    }
    if session["done"]:
        frame["result"] = _game_summary(session["results"])
    return frame


def _start_episode(session: dict) -> None:
    entry = LEVELS[session["episode"]]
    session["level"] = Level.from_ascii(entry["ascii"])
    session["env"] = SokobanEnv(session["level"], max_steps=entry["max_steps"])
    session["env"].reset()
    session["goal_b64"] = _b64(render_goal(session["level"]))
    session["pushed_any"] = False
    session["actions"] = []


def _finish_episode(session: dict) -> None:
    env: SokobanEnv = session["env"]
    entry = LEVELS[session["episode"]]
    solved = env.solved
    # cheap corner test only: the exact BFS check would block the serving
    # lock for seconds. This undercounts, so the metric is a lower bound.
    deadlocked = (not solved and session["pushed_any"]
                  and is_deadlocked(session["level"], env.state.boxes))
    session["results"].append({
        "solved": solved, "steps": env.state.steps,
        "optimal": entry["optimal_len"], "deadlocked": deadlocked,
        "actions": "".join("UDLR"[a] for a in session["actions"])})
    session["episode"] += 1


def _game_summary(results: list[dict], pad_to: int | None = None) -> dict:
    res = list(results)
    if pad_to:  # unplayed episodes count as unsolved
        res += [{"solved": False, "steps": 0, "optimal": 0,
                 "deadlocked": False}] * (pad_to - len(res))
    solved = [r for r in res if r["solved"]]
    return {
        "episodes": len(res),
        "solved": len(solved),
        "success_rate": round(len(solved) / len(res), 4) if res else 0.0,
        "move_efficiency": round(float(np.mean(
            [r["optimal"] / max(r["steps"], 1) for r in solved])), 4) if solved else 0.0,
        "avg_steps_solved": round(float(np.mean(
            [r["steps"] for r in solved])), 2) if solved else 0.0,
        "deadlock_rate": round(float(np.mean(
            [r["deadlocked"] for r in res])), 4) if res else 0.0,
        "total_actions": sum(r["steps"] for r in res),
    }


def _gc(now: float) -> None:
    for k in [k for k, s in _sessions.items() if now - s["touched"] > SESSION_TTL]:
        del _sessions[k]
    for k in [k for k, s in _scorecards.items()
              if now - s["touched"] > SESSION_TTL * 6]:
        del _scorecards[k]


# ------------------------------------------------------------------ API ----

class NewKey(BaseModel):
    name: str = Field(..., description="leaderboard name, 2-40 chars, unique")


class Act(BaseModel):
    action: int = Field(..., ge=0, le=3,
                        description="0 up, 1 down, 2 left, 3 right")


@app.get("/api/spec")
def spec() -> dict:
    # Read off the loaded set rather than restating it: the crate ramp and the
    # per-level step budgets have both changed, and a hardcoded summary drifts.
    crates = sorted({lv["n_crates"] for lv in LEVELS if "n_crates" in lv})
    budgets = [lv["max_steps"] for lv in LEVELS]
    return {
        "games": {"standard": {
            "board": "8x8, hidden layouts, ordered easiest first",
            "episodes": len(LEVELS),
            "crates_per_level": (f"{crates[0]}-{crates[-1]}" if crates
                                 else "unspecified"),
            "max_steps_per_episode": (
                f"{min(budgets)}-{max(budgets)}, three times each level's own "
                "optimal solution"),
        }},
        "actions": {"0": "up", "1": "down", "2": "left", "3": "right"},
        "observation": "64x64x3 uint8, base64-encoded raw bytes",
        "protocol": "https://github.com/Lulzx/latent-sokoban/blob/main/docs/API.md",
        "rules": "https://github.com/Lulzx/latent-sokoban/blob/main/docs/RULES.md",
        "note": "invalid actions are no-ops that consume a step",
    }


@app.post("/api/keys")
def create_key(body: NewKey, request: Request) -> dict:
    name = body.name.strip()
    if not NAME_RE.match(name):
        raise HTTPException(422, "name must be 2-40 word chars/spaces/dots/dashes")
    ip = _client_ip(request)
    conn = _db()
    day_ago = time.time() - 86400
    n_ip = conn.execute("SELECT COUNT(*) FROM keys WHERE ip=? AND created>?",
                        (ip, day_ago)).fetchone()[0]
    if n_ip >= MAX_KEYS_PER_IP_PER_DAY:
        conn.close()
        raise HTTPException(429, "key-creation limit reached; try tomorrow")
    key = "lsk-" + secrets.token_urlsafe(24)
    try:
        with conn:
            conn.execute("INSERT INTO keys VALUES (?,?,?,?)",
                         (_hash(key), name, time.time(), ip))
    except sqlite3.IntegrityError:
        raise HTTPException(409, "name already taken; choose another")
    finally:
        conn.close()
    return {"api_key": key, "name": name,
            "note": "shown once; store it. Pass as X-API-Key on every request."}


@app.post("/api/scorecards")
def open_scorecard(request: Request,
                   x_api_key: str | None = Header(None)) -> dict:
    name = _auth(x_api_key)
    now = time.time()
    conn = _db()
    n_day = conn.execute(
        "SELECT COUNT(*) FROM scorecards WHERE name=? AND opened>?",
        (name, now - 86400)).fetchone()[0]
    if n_day >= MAX_SCORECARDS_PER_KEY_PER_DAY:
        conn.close()
        raise HTTPException(
            429, f"{MAX_SCORECARDS_PER_KEY_PER_DAY} scorecards opened in the "
                 "last 24h; try later")
    sid = "sc-" + secrets.token_urlsafe(16)
    with conn:  # recorded at open, so abandoning a run does not buy a retry
        conn.execute("INSERT INTO scorecards (id, name, opened) VALUES (?,?,?)",
                     (sid, name, now))
    conn.close()
    with _lock:
        _gc(now)
        _scorecards[sid] = {"id": sid, "name": name, "opened": now,
                            "touched": now, "games": {}}
    return {"scorecard_id": sid, "games": list(GAMES),
            "episodes_per_game": len(LEVELS),
            "opened_today": n_day + 1,
            "daily_limit": MAX_SCORECARDS_PER_KEY_PER_DAY}


@app.post("/api/scorecards/{sid}/games/{game}/start")
def start_game(sid: str, game: str,
               x_api_key: str | None = Header(None)) -> dict:
    name = _auth(x_api_key)
    if game not in GAMES:
        raise HTTPException(404, f"unknown game; available: {list(GAMES)}")
    now = time.time()
    with _lock:
        card = _scorecards.get(sid)
        if not card or card["name"] != name:
            raise HTTPException(404, "unknown or expired scorecard")
        if game in card["games"]:
            raise HTTPException(409, "this scorecard already has a session for "
                                     "this game; open a new scorecard to retry")
        _gc(now)
        if len(_sessions) >= MAX_ACTIVE_SESSIONS:
            raise HTTPException(503, "server at capacity; try later")
        # a per-key cap as well, so one entrant cannot take the whole pool
        mine = sum(1 for s in _sessions.values() if s["name"] == name)
        if mine >= MAX_SESSIONS_PER_KEY:
            raise HTTPException(
                429, f"{MAX_SESSIONS_PER_KEY} sessions already in flight for "
                     "this key; finish or abandon one first")
        card["touched"] = now
        gid = "gs-" + secrets.token_urlsafe(16)
        session = {"id": gid, "scorecard": sid, "name": name, "game": game,
                   "episode": 0, "results": [], "done": False,
                   "created": now, "touched": now}
        _start_episode(session)
        _sessions[gid] = session
        card["games"][game] = gid
        return _frame(session)


@app.post("/api/sessions/{gid}/act")
def act(gid: str, body: Act, x_api_key: str | None = Header(None)) -> dict:
    name = _auth(x_api_key)
    with _lock:
        session = _sessions.get(gid)
        if not session or session["name"] != name:
            raise HTTPException(404, "unknown or expired session")
        if session["done"]:
            raise HTTPException(409, "session is over")
        session["touched"] = time.time()
        env: SokobanEnv = session["env"]
        _, done, info = env.step(body.action)
        session["actions"].append(body.action)
        if info.pushed:
            session["pushed_any"] = True
        last = {"action": body.action, "moved": info.moved,
                "solved": bool(env.solved), "episode_done": bool(done)}
        if done:
            _finish_episode(session)
            if session["episode"] >= len(LEVELS):
                session["done"] = True
                conn = _db()
                with conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO replays VALUES (?,?,?,?,?,?)",
                        (gid, session["scorecard"], name, session["game"],
                         time.time(), json.dumps([
                             {k: r[k] for k in
                              ("solved", "steps", "optimal", "deadlocked",
                               "actions")} for r in session["results"]])))
                conn.close()
            else:
                _start_episode(session)
        return _frame(session, last=last)


@app.get("/api/scorecards/{sid}")
def scorecard_state(sid: str, x_api_key: str | None = Header(None)) -> dict:
    name = _auth(x_api_key)
    with _lock:
        card = _scorecards.get(sid)
        if not card or card["name"] != name:
            raise HTTPException(404, "unknown or expired scorecard")
        games = {}
        for game, gid in card["games"].items():
            s = _sessions.get(gid)
            games[game] = (_game_summary(s["results"]) if s
                           else {"status": "expired"})
            if s:
                games[game]["status"] = "done" if s["done"] else "in_progress"
        return {"scorecard_id": sid, "name": name, "open": True, "games": games}


@app.post("/api/scorecards/{sid}/close")
def close_scorecard(sid: str, x_api_key: str | None = Header(None)) -> dict:
    name = _auth(x_api_key)
    with _lock:
        card = _scorecards.pop(sid, None)
        if not card or card["name"] != name:
            raise HTTPException(404, "unknown or expired scorecard")
        gid = card["games"].get("standard")
        session = _sessions.pop(gid, None) if gid else None
        results = session["results"] if session else []
    # unplayed episodes count as unsolved: closing locks the data
    summary = _game_summary(results, pad_to=len(LEVELS))
    conn = _db()
    with conn:
        conn.execute(
            """UPDATE scorecards SET closed=?, episodes=?, solved=?,
                   success_rate=?, move_efficiency=?, avg_steps_solved=?,
                   deadlock_rate=?, total_actions=?
               WHERE id=?""",
            (time.time(), summary["episodes"], summary["solved"],
             summary["success_rate"], summary["move_efficiency"],
             summary["avg_steps_solved"], summary["deadlock_rate"],
             summary["total_actions"], sid))
    conn.close()
    return {"scorecard_id": sid, "name": name, "closed": True,
            "games": {"standard": summary}}


@app.get("/api/replays/{gid}")
def replay(gid: str, x_api_key: str | None = Header(None)) -> dict:
    """Per-episode outcomes for a finished session (never level layouts).

    The action strings are withheld from everyone but the run's owner. The
    hidden set is fixed and deterministic, so a published action string is a
    replayable perfect score: anyone holding it can post the same numbers
    without an agent. Outcomes stay public so results remain inspectable.
    """
    conn = _db()
    row = conn.execute("SELECT name, game, finished, episodes_json "
                       "FROM replays WHERE session_id=?", (gid,)).fetchone()
    conn.close()
    if not row:
        raise HTTPException(404, "no replay for that session")
    owner = _name_for_key(x_api_key) == row[0]
    episodes = json.loads(row[3])
    if not owner:
        episodes = [{k: v for k, v in ep.items() if k != "actions"}
                    for ep in episodes]
    return {"session_id": gid, "name": row[0], "game": row[1],
            "finished": row[2], "owner_view": owner, "episodes": episodes}


@app.get("/api/leaderboard")
def leaderboard() -> JSONResponse:
    conn = _db()
    # Bare columns alongside MAX() come from the row that matched the max:
    # a SQLite-specific guarantee, and it holds because there is one aggregate.
    rows = conn.execute(
        """SELECT name, MAX(success_rate) AS sr, move_efficiency,
                  avg_steps_solved, deadlock_rate, episodes, solved, closed
           FROM scorecards WHERE closed IS NOT NULL GROUP BY name
           ORDER BY sr DESC, move_efficiency DESC, deadlock_rate ASC
           LIMIT 100""").fetchall()
    total = conn.execute(
        "SELECT COUNT(*) FROM scorecards WHERE closed IS NOT NULL").fetchone()[0]
    conn.close()
    return JSONResponse({
        "entries": [
            {"rank": i + 1, "name": r[0], "success_rate": r[1],
             "move_efficiency": r[2], "avg_steps_solved": r[3],
             "deadlock_rate": r[4], "episodes": r[5], "solved": r[6],
             "date": time.strftime("%Y-%m-%d", time.gmtime(r[7]))}
            for i, r in enumerate(rows)],
        "total_scorecards": total,
    })


ICONS = ROOT / "static" / "icons"


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Serve the crate sprite for bare /favicon.ico requests.

    Browsers ask for this path without being told to, and they expect an
    image rather than the SVG this used to answer with. The pages
    themselves link the PNGs under /static/icons directly.
    """
    from fastapi.responses import Response
    return Response((ICONS / "favicon-32.png").read_bytes(),
                    media_type="image/png",
                    headers={"Cache-Control": "public, max-age=86400"})


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")

# The MkDocs build output is generated, not committed, so a checkout that
# has not run `mkdocs build` simply has no /docs rather than failing to boot.
_DOCS_SITE = ROOT.parent / "site"
if _DOCS_SITE.is_dir():
    app.mount("/docs", StaticFiles(directory=_DOCS_SITE, html=True), name="docs")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return (ROOT / "index.html").read_text()


@app.get("/leaderboard", response_class=HTMLResponse)
def leaderboard_page() -> str:
    return (ROOT / "leaderboard.html").read_text()


@app.get("/play", response_class=HTMLResponse)
def play_page() -> str:
    return (ROOT / "play.html").read_text()
