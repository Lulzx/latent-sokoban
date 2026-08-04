# Latent Sokoban Challenge

An open benchmark for **pixel-based latent world models**: agents see only
rendered images of the current and goal Sokoban boards, learn dynamics from
action-labelled trajectories, and plan in latent space.

**Public leaderboard & live evaluation API: https://sokoban.lulzx.space** —
anyone can register a key and evaluate an agent against 50 hidden levels
held by the server (agent protocol: [docs/API.md](docs/API.md)):

```bash
python scripts/remote_eval.py --register "your-name"   # once: get an API key
export SOKOBAN_API_KEY=lsk-…
python scripts/remote_eval.py --agent my_pkg.agent:MyAgent
```

Why this is hard: our hidden levels need 10–50 optimal moves, and
[SokoBench (arXiv:2601.20856)](https://arxiv.org/abs/2601.20856) shows even
frontier reasoning models degrade consistently on Sokoban past ~25-move
horizons — this benchmark deliberately lives in that regime, from pixels.

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
| Shared baseline | `baseline/` | The rulebook's required baseline: CNN encoder, 128-d latent, residual MLP dynamics, VICReg-style regularization, beam-search MPC (needs PyTorch) |
| Scoring script | `scripts/score.py` | The official 100-point formula (45S+20G+10M+10P+10D+5R) computed from evaluation results |
| Hidden-test commitment | `scripts/hidden_test.py` | Encrypt-and-commit a secret generation seed; verified reveal after submissions freeze |
| Visualizer | `scripts/visualize.py` | Rollout contact sheets (agent or optimal solver) + per-step metadata for reports and the live final |

The infrastructure's only dependency is numpy; the optional baseline adds
PyTorch.

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

## Shared baseline

`baseline/` implements the required baseline from the rulebook: a small
CNN encoder → 128-d latent, learned action embeddings, a residual MLP
dynamics model (`z' = z + f(z, a)`, 830k parameters total), a one-step
latent prediction loss with VICReg-style variance/covariance
regularization, and beam-search MPC scored by Euclidean latent distance
to the encoded goal image, replanning every action. Requires PyTorch
(`pip install -e ".[baseline]"`).

```bash
python baseline/train.py --data data/warmup --out baseline/checkpoint.pt --steps 5000 --seed 13
python scripts/evaluate.py --agent baseline.agent:BaselineAgent --splits levels/eval_w.json --seeds 0 1 2
```

Measured results (5,000 training steps, ~9 min on an M-series MPS; 50
warmup levels / 20 Split-A levels, 3 evaluation seeds):

| Agent | Split W (6×6, 1 box) | Split A (8×8, 3 boxes) | Plan time | Calls/action |
| --- | --- | --- | --- | --- |
| Random | 4% | 0% | — | 0 |
| Baseline | **12%** | **0%** | 3.6 ms | 84 |

Two findings from instrumenting this baseline, so competitors don't
rediscover them:

- **Plan short, replan often.** One-step prediction is sharp (open-loop
  drift after 1 imagined step ≈ one true transition), but drift reaches
  the *entire* typical start-to-goal distance after ~6 imagined steps —
  long beams score pure noise. The baseline therefore plans at horizon 3
  and relies on MPC replanning. Extending the usable horizon (multi-step
  rollout losses, better dynamics) is the single most obvious research
  direction.
- **Pure greedy oscillates.** The Euclidean distance field has local
  minima (any required detour temporarily increases distance), and a
  deterministic argmin planner parks in them, bouncing left–right
  forever — even *oracle* dynamics with greedy latent distance only
  solves 16% of warmup levels. The baseline adds small seeded Gumbel
  noise (η = 0.2) to plan scores to break loops. Learned value/distance
  functions that understand detours are the second obvious direction.

The 0% on the official 8×8 three-box config is the point of the
competition: the baseline verifies the pipeline end-to-end (training
doesn't collapse — latent std ≈ 1.0; planning runs within budget;
evaluation is deterministic), and everything above 0% on Split A is
earned by research.

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
