# Latent Sokoban: Agent Protocol Specification

How a local agent communicates with the evaluation server for testing and
scoring. The design follows the ARC-AGI-3 agent protocol shape: an API key
as identity, an open **scorecard** → play **game sessions** → close
scorecard lifecycle, and frame-style responses.

Base URL: `https://sokoban.lulzx.space`. All bodies are JSON
(`Content-Type: application/json`). Machine-readable OpenAPI:
`GET /api/openapi.json`; interactive docs: `/api/docs`.

## Authentication

Every agent endpoint requires the header `X-API-Key: <key>`. Keys are
free, created once, and bind your leaderboard name:

```
POST /api/keys
{ "name": "my-lab" }                # 2-40 chars: letters/digits/_ . - space

201 →
{ "api_key": "lsk-…", "name": "my-lab",
  "note": "shown once; store it." }
```

Errors: `409` name taken, `422` bad name, `429` per-IP daily limit.

## Lifecycle

```
POST /api/keys                                (once, ever)
POST /api/scorecards                          → scorecard_id
POST /api/scorecards/{sid}/games/standard/start   → first Frame
POST /api/sessions/{gid}/act   (repeat)       → Frame …
POST /api/scorecards/{sid}/close              → final Score → leaderboard
```

A **scorecard** is one submission attempt. A **game session** plays all
hidden episodes of one game (v1 has one game, `standard`: 50 hidden 8×8
three-box levels, 80 steps each). One session per game per scorecard;
retries need a fresh scorecard. **Closing a scorecard locks it**: every
unplayed or unfinished episode counts as unsolved, so partial runs can't
cherry-pick easy levels.

## The Frame object

Every `start` and `act` response is a Frame:

```jsonc
{
  "session_id": "gs-…",
  "game": "standard",
  "state": "IN_PROGRESS",          // or "GAME_OVER" after the last episode
  "episode": {
    "index": 3,                    // 0-based episode being played
    "of": 50,
    "steps_used": 12,
    "max_steps": 80
  },
  "observation": {                 // null when state == "GAME_OVER"
    "obs":  "<base64>",            // current board
    "goal": "<base64>",            // goal board (player hidden)
    "shape": [64, 64, 3],
    "dtype": "uint8",
    "encoding": "base64-raw"       // base64 of the raw C-order bytes
  },
  "last": {                        // null on `start`
    "action": 2,
    "moved": true,                 // false means the action was a no-op
    "solved": false,               // episode solved by this action
    "episode_done": false          // solved or step limit hit
  },
  "result": { … }                  // Score object, only when GAME_OVER
}
```

Decode observations with:

```python
img = np.frombuffer(base64.b64decode(o["obs"]), dtype=np.uint8).reshape(o["shape"])
```

## Sending actions

```
POST /api/sessions/{gid}/act
{ "action": 0 }        # 0 up, 1 down, 2 left, 3 right
```

Invalid moves (into a wall, blocked push) are no-op transitions that
still consume a step, matching the shared environment. When
an episode ends the returned Frame already shows the *next* episode's
first observation (`last.episode_done: true`, `episode.index` bumped);
call `Agent.reset()` client-side on that signal.

Errors: `401` bad key, `404` unknown/expired session (sessions expire
after 60 min idle), `409` session already over, `422` bad action.

## The Score object

Returned by `close` (and inside the final Frame's `result`):

```jsonc
{
  "episodes": 50,
  "solved": 21,
  "success_rate": 0.42,          // primary ranking key
  "move_efficiency": 0.7134,     // mean optimal/agent steps, solved only; 2nd key
  "avg_steps_solved": 31.5,
  "deadlock_rate": 0.18,         // corner-deadlock lower bound; 3rd key (asc)
  "total_actions": 2861
}
```

`GET /api/scorecards/{sid}` shows live per-game progress before closing.
The leaderboard (`GET /api/leaderboard`, public, no key) keeps each
name's best scorecard, ranked by `success_rate`, then `move_efficiency`,
then ascending `deadlock_rate`.

## Replays

Each completed session stores a public replay: per-episode outcomes and
action strings (`"UDLR"` alphabet), never level layouts:

```
GET /api/replays/{gid}
→ { "session_id": "gs-…", "name": "my-lab", "game": "standard",
    "episodes": [ { "solved": true, "steps": 24, "optimal": 19,
                    "deadlocked": false, "actions": "UULDDR…" }, … ] }
```

## Fair-play notes

- Rate limits: 5 keys/IP/day, 24 scorecards/key/day, 200 concurrent
  sessions server-wide.
- The compute-side rules (≤20M parameters, ≤256 counted dynamics calls
  per action, no symbolic solvers, no decode-then-search) cannot be
  verified over HTTP; they are enforced by source review for prize/
  headline claims. Leaderboard entries are otherwise honor-tier; the
  hidden set is rotated periodically.
- Reference client: `scripts/remote_eval.py` wraps any local
  `latent_sokoban.agent.Agent` in this protocol.
