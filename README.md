# Latent Sokoban Challenge — Shared Infrastructure

Shared environment, dataset generator and evaluation harness for a two-person
research competition: build the best **pixel-based latent world model** for
Sokoban. Agents see only rendered images of the current and goal boards, learn
dynamics from action-labelled trajectories, and plan in latent space.

![Sokoban observations: current state, goal, visual-generalization theme, 10x10 board](docs/preview.png)

*Left to right: current observation (8×8, three boxes), goal observation
(player hidden), a Split-B visual-generalization theme, a 10×10 Split-C
board. All observations are 64×64 RGB.*

This repo is the neutral ground both competitors build on: same environment,
same dataset, same levels, same scoring script. Models and planners live in
each competitor's own repo.

## Anti-brute-force design

The benchmark is deliberately sized so that brute force loses:

- **Official config is 8×8 with 3 boxes** (~250k reachable states), so the
  state graph cannot be exhausted within the planning budget — search must
  be guided by a learned heuristic. A 6×6 one-box warmup (Split W) exists
  for the baseline round only and is never scored.
- **Planning is budgeted in counted dynamics calls, not wall-clock**: at
  most **256 learned-dynamics calls per action** (one call = one predicted
  transition of one candidate state; a batch of B rolled H steps costs
  B×H). Planners tick a shared `CallMeter`; the harness fails any episode
  that exceeds the cap and logs per-action counts into the results. Meter
  honesty is verified by source review of the frozen submission.
- **Decode-then-search is prohibited by rule**: no module may reconstruct
  an exact symbolic board at inference time or run graph search over
  enumerated discrete states, whether the decoder or transition function
  was hand-coded or learned. See [docs/RULES.md](docs/RULES.md).

## What's here

| Component | Where | Notes |
| --- | --- | --- |
| Environment | `latent_sokoban/env.py` | Deterministic Sokoban (official: 8×8, 3 boxes, 80-step limit), 4 actions, invalid actions are no-ops, ASCII level format |
| Renderer | `latent_sokoban/render.py` | Pure-numpy 64×64 RGB, themeable for visual generalization |
| Level generator | `latent_sokoban/levels.py` | Rejection sampling against a BFS solver; every level is solvable, difficulty controlled by optimal-length band |
| Solver | `latent_sokoban/solver.py` | Optimal BFS + deadlock detection (data generation and eval analysis only — never available to agents) |
| Dataset generator | `latent_sokoban/dataset.py`, `scripts/generate_dataset.py` | 50% random / 30% solver / 20% perturbed-solver trajectories, sharded `.npz` |
| Benchmark splits | `scripts/generate_levels.py` | Warmup W + splits A–D (+ optional E), including the hidden-test-set protocol |
| Evaluation harness | `latent_sokoban/evaluation.py`, `scripts/evaluate.py` | Deterministic, multi-seed, enforces the dynamics-call budget, reports all official metrics |
| Agent interface | `latent_sokoban/agent.py` | The only contract competitors implement, including the `CallMeter` |

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
        z_next = self.dynamics(z_batch, actions)   # your world model
        self.call_meter.tick(len(z_batch))         # REQUIRED: count every
        ...                                        # dynamics forward pass
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
| W | Warmup (unscored) | 6×6, 1 box — baseline-round sanity checks only |
| A | Core performance | 8×8, 3 boxes, training visual style, unseen layouts |
| B | Visual generalization | Split-A boards with randomized colours, checker patterns and pixel noise |
| C | Structural generalization | 10×10 boards, 3 boxes, longer optimal solutions |
| D | Deadlock avoidance | 8×8, 3 boxes; at least one reachable push is provably irreversible |
| E | Bonus: five boxes | 10×10, 5 boxes, 160-action horizon (`--split E`) |

The harness reports, per split, averaged over evaluation seeds: **success
rate** (primary metric), **move efficiency** (optimal ÷ agent moves, solved
levels only), **planning time** per action, **model calls** per action
(average, max, and cap violations), and **deadlock rate** (fraction of
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
per action, ≤256 counted dynamics calls per action, shared training seeds
{13, 42, 137}, scores averaged over ≥3 evaluation seeds. Final score:
45% standard success, 20% generalization, 10% move efficiency, 10% planning
speed, 10% deadlock avoidance, 5% reproducibility.

## License

MIT
