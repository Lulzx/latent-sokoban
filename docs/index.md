# Latent Sokoban Challenge

A public benchmark for **latent world models**. Your agent sees two 64×64
images, the board and the goal, and nothing else: no coordinates, no rules,
no solver. It has to learn the dynamics from pixels, plan in its own learned
representation, and push every crate home.

[Play it yourself](https://sokoban.lulzx.space/play){ .md-button .md-button--primary }
[Leaderboard](https://sokoban.lulzx.space/leaderboard){ .md-button }

## Why this is hard

Sokoban is trivial to solve symbolically and brutal to solve from pixels.
The difficulty is not search, it is that every mistake is permanent: push a
crate into a corner and the level becomes unsolvable, with no signal saying
so until the step budget runs out. An agent has to learn that consequence
from raw frames.

The rules exist to keep that the actual task:

| Constraint | Why |
| --- | --- |
| 64×64 RGB observations only | No symbolic state, ever |
| No symbolic solvers | No BFS or A\* anywhere in the loop |
| No decode-then-search | Recovering the grid from pixels defeats the point |
| ≤ 20M parameters | Keeps it about representation, not scale |
| ≤ 256 dynamics calls per action | Planning must be guided, not exhaustive |

Full detail in [the rules](RULES.md).

## The benchmark

100 hidden levels on an 8×8 board, ordered easiest first. Crate count rises
with the level number and so does the length of the shortest solution:

| Levels | Crates | Optimal solution |
| --- | --- | --- |
| 1–25 | 1 | 6–18 moves |
| 26–50 | 2 | 10–28 moves |
| 51–80 | 3 | 15–31 moves |
| 81–100 | 4 | 20–42 moves |

Each level's step budget is three times its own optimal solution. The
layouts never leave the server; agents only ever receive rendered frames.
See [level generation](level-generation.md) for how the set is built and
why difficulty rides on crate count.

## Getting started

```bash
git clone https://github.com/Lulzx/latent-sokoban && cd latent-sokoban
pip install -e .

# once: claim your leaderboard name, get an API key
python scripts/remote_eval.py --register "your-name"
export SOKOBAN_API_KEY=lsk-...

# sanity check with the built-in random agent
python scripts/remote_eval.py --agent random

# your model: implement latent_sokoban.agent.Agent, then
python scripts/remote_eval.py --agent my_pkg.agent:MyAgent
```

Then read [how it works](how-it-works.md) for the end-to-end picture, or go
straight to the [agent protocol](API.md) for the wire format.

## Where things live

| What | Where |
| --- | --- |
| Environment, generator, solver | `latent_sokoban/` |
| Reference world-model baseline | `baseline/` |
| Evaluation API and site | `server/` |
| Dataset and level tooling | `scripts/` |

The [code reference](reference/index.md) documents the public API of each
module directly from its docstrings.
