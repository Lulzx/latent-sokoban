# Latent Sokoban: Agent Protocol Specification

How a local agent communicates with the evaluation server for testing and
scoring. The design follows the ARC-AGI-3 agent protocol shape: an API key
as identity, an open **scorecard** → play **game sessions** → close
scorecard lifecycle, and frame-style responses.

Base URL: `https://sokoban.lulzx.space`. All bodies are JSON
(`Content-Type: application/json`).

!!! tip "Interactive reference"

    Every endpoint below is also browsable and callable at
    [**/api/docs**](https://sokoban.lulzx.space/api/docs), generated from the
    running server, with the machine-readable schema at
    [/api/openapi.json](https://sokoban.lulzx.space/api/openapi.json). This
    page is the prose companion: it explains the lifecycle and the rules the
    schema cannot state.

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
hidden episodes of one game (v1 has one game, `standard`: 100 hidden 8×8
levels on a 1-to-4 crate ramp, ordered easiest first, each with a budget of
three times its own optimal solution). `GET /api/spec` reports the live
numbers. One session per game per scorecard; retries need a fresh
scorecard. **Closing a scorecard locks it**: every unplayed or unfinished
episode counts as unsolved, so partial runs can't cherry-pick easy levels.

Opening a scorecard is what counts against your daily allowance, not
closing it, so abandoning a run mid-way does not buy a free retry.

## Running episodes concurrently

By default a session plays one episode at a time. That makes a run bound by
network latency rather than by your agent: 100 levels is roughly 6,000
actions, and at a 190 ms round trip that is over an hour, of which the agent
itself accounts for a few milliseconds per action.

Pass `parallel` on `start` to put several episodes of the **current tier**
in flight at once (maximum 32):

```
POST /api/scorecards/{sid}/games/standard/start?parallel=25
```

A tier is a run of levels sharing a crate count — for `standard` that is
levels 1–25, 26–50, 51–80, 81–100. **A harder tier does not open until every
episode of the current one has finished**, so a run is still evaluated in
difficulty order and you cannot skip ahead to sample the whole set.

With `parallel > 1` the Frame carries an `episodes` array instead of a
single `episode`, and every `act` must name which episode it applies to:

```jsonc
{
  "state": "IN_PROGRESS",
  "episodes": [                      // one entry per in-flight episode
    { "index": 3, "of": 100, "steps_used": 7, "max_steps": 30,
      "observation": { "obs": "…", "goal": "…", "shape": [64, 64, 3] } }
  ],
  "tier":     { "index": 0, "of": 4, "range": [0, 25] },
  "finished": 12,                    // episodes recorded so far
  "acted_episode": 3,
  "last":     { "action": 2, "episode": 3, "episode_done": false }
}
```

Two things to get right. Responses within a round arrive concurrently, so
the `episodes` array on one of them can predate another episode finishing —
track which episodes you have seen end and do not act on them again, or the
next `act` returns `409`. And one HTTP connection carries one request at a
time, so genuine concurrency needs a connection per worker rather than one
shared socket.

`parallel=1` is the default and is exactly the protocol described in the
rest of this document: single `episode` object, no `episode` field on `act`.
Existing clients need no changes.

This affects wall-clock only. The levels, step budgets, call budget and
scoring are identical either way; `scripts/remote_eval.py --parallel 25`
took a full 100-level run from ~74 minutes to ~5.

## The Frame object

Every `start` and `act` response is a Frame:

```jsonc
{
  "session_id": "gs-…",
  "game": "standard",
  "state": "IN_PROGRESS",          // or "GAME_OVER" after the last episode
  "episode": {
    "index": 3,                    // 0-based episode being played
    "of": 100,
    "steps_used": 12,
    "max_steps": 57             // this level's own budget, 3x its optimal
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
{ "action": 0 }               # 0 up, 1 down, 2 left, 3 right
{ "action": 0, "episode": 3 } # required when several are in flight
```

Invalid moves (into a wall, blocked push) are no-op transitions that
still consume a step, matching the shared environment. When
an episode ends the returned Frame already shows the *next* episode's
first observation (`last.episode_done: true`, `episode.index` bumped);
call `Agent.reset()` client-side on that signal.

Errors: `401` bad key, `404` unknown/expired session (sessions expire
after 60 min idle), `409` session already over or the named episode is not
in flight, `422` bad action. `400` means several episodes are open and you
did not say which one to act on.

Clients should retry network faults, `429` and `5xx` rather than aborting: a
run is thousands of requests, so a transient reset is likely rather than
exceptional, and dying mid-run leaves the scorecard open — which scores
nothing, however well the run was going. Retrying `close` matters most.
Note that retrying `act` after a lost *response* replays the action, since
the API takes no idempotency key; the returned Frame is still the true state
and replanning from it costs a step rather than the episode.

## The Score object

Returned by `close` (and inside the final Frame's `result`):

```jsonc
{
  "episodes": 100,
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

Each completed session stores a replay of per-episode outcomes, never
level layouts:

```
GET /api/replays/{gid}
→ { "session_id": "gs-…", "name": "my-lab", "game": "standard",
    "owner_view": false,
    "episodes": [ { "solved": true, "steps": 24, "optimal": 19,
                    "deadlocked": false }, … ] }
```

Outcomes are public so results stay inspectable. The action strings
(`"UDLR"` alphabet) are included only when the request carries the key that
produced the run, which the response marks with `owner_view: true`. The
hidden set is fixed and
deterministic, so a published action string is a replayable perfect score:
anyone holding it could post the same numbers without an agent.

## Fair-play notes

- Rate limits: 5 keys/IP/day, 24 scorecards *opened* per key per day, 3
  concurrent sessions per key, 200 server-wide.
- The compute-side rules (≤20M parameters, ≤256 counted dynamics calls
  per action, no symbolic solvers, no decode-then-search) cannot be
  verified over HTTP; they are enforced by source review for prize/
  headline claims. Leaderboard entries are otherwise honor-tier; the
  hidden set is rotated periodically.
- Reference client: `scripts/remote_eval.py` wraps any local
  `latent_sokoban.agent.Agent` in this protocol.
