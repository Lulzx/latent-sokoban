# Latent Sokoban Challenge — Shared Infrastructure

Shared environment, dataset generator and evaluation harness for a two-person
research competition: build the best **pixel-based latent world model** for
Sokoban. Agents see only rendered images of the current and goal boards, learn
dynamics from action-labelled trajectories, and plan in latent space.

![Sokoban observations: current state, goal, visual-generalization theme, 7x7 board](docs/preview.png)

*Left to right: current observation, goal observation (player hidden), a
Split-B visual-generalization theme, a 7×7 Split-C board. All observations are
64×64 RGB.*

This repo is the neutral ground both competitors build on: same environment,
same dataset, same levels, same scoring script. Models and planners live in
each competitor's own repo.

## What's here

| Component | Where | Notes |
| --- | --- | --- |
| Environment | `latent_sokoban/env.py` | Deterministic 6×6 Sokoban, 4 actions, invalid actions are no-ops, 40-step limit, ASCII level format |
| Renderer | `latent_sokoban/render.py` | Pure-numpy 64×64 RGB, themeable for visual generalization |
| Level generator | `latent_sokoban/levels.py` | Rejection sampling against a BFS solver; every level is solvable, difficulty controlled by optimal-length band |
| Solver | `latent_sokoban/solver.py` | Optimal BFS + deadlock detection (data generation and eval analysis only — never available to agents) |
| Dataset generator | `latent_sokoban/dataset.py`, `scripts/generate_dataset.py` | 50% random / 30% solver / 20% perturbed-solver trajectories, sharded `.npz` |
| Benchmark splits | `scripts/generate_levels.py` | Splits A–D (+ optional E), including the hidden-test-set protocol |
| Evaluation harness | `latent_sokoban/evaluation.py`, `scripts/evaluate.py` | Deterministic, multi-seed, reports all official metrics |
| Agent interface | `latent_sokoban/agent.py` | The only contract competitors implement |

The only dependency is numpy.

## Quickstart

```bash
pip install -e .

# generate the shared training dataset
python scripts/generate_dataset.py --out data/train --episodes 2000 --seed 13

# generate development benchmark splits A-D
python scripts/generate_levels.py --all --n 100 --seed 1001 --out levels/

# sanity-check the harness with the built-in random agent
python scripts/evaluate.py --agent random --splits levels/split_a.json --seeds 0 1 2
```

Run the test suite with `python -m pytest tests/`.

## The agent contract

Implement `latent_sokoban.agent.Agent` in your own repo:

```python
from latent_sokoban.agent import Agent

class MyAgent(Agent):
    def reset(self):
        ...  # called at the start of every episode

    def act(self, obs, goal, action_history) -> int:
        # obs, goal: (64, 64, 3) uint8 images. Return 0=up 1=down 2=left 3=right.
        ...
```

Then evaluate it:

```bash
python scripts/evaluate.py --agent my_submission.agent:MyAgent \
    --splits levels/split_a.json levels/split_b.json levels/split_c.json levels/split_d.json \
    --seeds 0 1 2 --out results.json
```

Agents receive **only** the current image, the goal image and their own action
history. Symbolic board state, coordinates, transition rules and solver access
are all off-limits at inference time (the solver is allowed for training-data
generation and post-hoc analysis).

## Benchmark splits

| Split | Purpose | Configuration |
| --- | --- | --- |
| A | Core performance | 6×6, 1 box, training visual style, unseen layouts |
| B | Visual generalization | Split-A boards with randomized colours, checker patterns and pixel noise |
| C | Structural generalization | 7×7 boards, longer optimal solutions, denser walls |
| D | Deadlock avoidance | Solvable levels where at least one reachable push is provably irreversible |
| E | Bonus: two boxes | 7×7, 2 boxes, longer horizon (`--split E`) |

The harness reports, per split, averaged over evaluation seeds: **success
rate** (primary metric), **move efficiency** (optimal ÷ agent moves, solved
levels only), **planning time** per action, and **deadlock rate** (fraction of
episodes that entered a provably dead state, checked exactly with a bounded
BFS after every push).

## Dataset format

Each shard is an `.npz` with a `.json` sidecar (counts for data-budget
audits). Episodes are concatenated; `actions[i]` is the action taken *from*
`frames[i]` (`-1` marks the last frame of an episode), so
`(frames[i], actions[i], frames[i+1])` is a training transition whenever
`actions[i] != -1`. Shards also carry per-episode goal images, per-step
`pushed`/`invalid` flags, episode outcomes, trajectory kind (random / solver /
perturbed) and the ASCII levels for reproducibility. See
`latent_sokoban/dataset.py` for the full schema.

## Hidden test set protocol

1. Freeze this repo at an agreed tag; agree on generation constraints.
2. One competitor (or a trusted third party) runs `scripts/generate_levels.py`
   once per split with a **secret seed**.
3. Store the seed in a password-protected archive. Nobody inspects the
   generated files.
4. Reveal the password only after both submissions are frozen, regenerate,
   and run `scripts/evaluate.py` on both submissions on the same machine.

Because generation is fully determined by `(script, constraints, seed)`, the
seed alone is a commitment to the exact test levels.

## Competition rules (summary)

Full rules live in [docs/RULES.md](docs/RULES.md). Defaults: fixed shared
dataset, ≤20M parameters, ≤12 GPU-hours for the final run, ≤500 ms inference
per action, ≤10,000 planning rollouts per action, shared training seeds
{13, 42, 137}, scores averaged over ≥3 evaluation seeds. Final score:
45% standard success, 20% generalization, 10% move efficiency, 10% planning
speed, 10% deadlock avoidance, 5% reproducibility.

## License

MIT
