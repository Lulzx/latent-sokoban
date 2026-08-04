# How it works, end to end

This page follows a single evaluation run from `git clone` to a row on the
leaderboard, naming the code that does each part.

## The shape of the system

```
   your machine                        sokoban.lulzx.space
   ------------                        -------------------
   your Agent                          FastAPI app (server/app.py)
       |                                    |
       | observation (64x64x3 RGB)          | holds the 100 hidden levels
       |<-----------------------------------| never sends layouts
       |                                    |
       | action: up/down/left/right         | steps SokobanEnv
       |----------------------------------->| scores the episode
                                            |
                                       sqlite: scorecards, replays
```

Your agent never receives a grid, a coordinate, or a rule. It receives two
images per frame: the current board, and the goal configuration.

## 1. Levels exist before you arrive

The hidden set is generated once from a secret seed and written to
`hidden_levels.json`, which lives outside the repo on the server. See
[level generation](level-generation.md).

The generator is deterministic in its seed, which is what makes the hidden
set auditable after the fact: publish the seed once the round closes and
anyone can regenerate the exact levels and verify what was scored.

`server/app.py` loads that file at import time into `LEVELS`.

## 2. You claim an identity

```bash
python scripts/remote_eval.py --register "your-name"
```

`POST /api/keys` mints an API key bound to a leaderboard name. The server
stores only a SHA-256 hash of the key, so the plaintext exists solely in
your environment. Keys are rate limited per IP per day.

## 3. You open a scorecard

`POST /api/scorecards` begins one submission attempt. A scorecard is the
unit that lands on the leaderboard: it aggregates every episode played
under it, and it can be opened, played, and closed exactly once.

## 4. You play the game

`POST /api/scorecards/{sid}/games/standard/start` starts a session and
returns the first frame. From there the loop is:

```python
frame = start(...)
while frame["state"] == "IN_PROGRESS":
    obs  = decode(frame["observation"]["obs"])   # 64x64x3 uint8
    goal = decode(frame["observation"]["goal"])  # 64x64x3 uint8
    action = agent.act(obs, goal)
    frame = act(session_id, action)
```

Each `act` call steps `SokobanEnv` by one action. Invalid moves, walking
into a wall or pushing a crate that cannot move, are no-ops that still
consume a step. When a level is solved or its step budget runs out, the
server records the episode and advances to the next level automatically.

The frame carries observations as base64 of the raw bytes, not PNG, so
there is no image decoding in the hot loop. Exact schema in
[the agent protocol](API.md).

## 5. You close the scorecard

`POST /api/scorecards/{sid}/close` locks it and writes it to the
leaderboard. This is the step that makes the benchmark honest: closing
counts **every unplayed episode as unsolved**. Stopping early because the
run is going badly does not help you, because the levels you skipped are
scored as failures anyway.

Metrics are computed at this point. See [scoring](scoring.md).

## 6. The result becomes public

The leaderboard ranks the best scorecard per entrant by success rate, then
by move efficiency, then by deadlock rate. Completed sessions also store a
replay: the action string per episode, so a result can be inspected rather
than taken on faith.

## What runs where

| Component | File | Role |
| --- | --- | --- |
| Environment | `latent_sokoban/env.py` | State, moves, push rules, solved test |
| Generator | `latent_sokoban/levels.py` | Rejection-sampled solvable levels, the difficulty ramp |
| Solver | `latent_sokoban/solver.py` | BFS for optimal length, deadlock detection |
| Renderer | `latent_sokoban/render.py` | State to 64×64×3 pixels |
| Server | `server/app.py` | Keys, scorecards, sessions, scoring, leaderboard |
| Client | `scripts/remote_eval.py` | Reference loop against the live API |
| Baseline | `baseline/` | Reference world model and planner |

The solver exists for the *server's* benefit, to know each level's optimal
length and to detect deadlocks. Using anything like it inside an agent is
[against the rules](RULES.md).
