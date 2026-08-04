# Latent Sokoban Challenge

An open benchmark for pixel-based latent world models. Anyone can enter:
register a name, evaluate against the hidden set through the public API at
[sokoban.lulzx.space](https://sokoban.lulzx.space), and appear on the
leaderboard.

These rules define what counts as a valid entry. They are published in
advance and applied identically to every submission.

## Objective

Build an agent that receives:

* A rendered image of the current Sokoban board
* A rendered image of the desired goal board

The agent must plan and execute actions that solve the puzzle.

The agent should learn environment dynamics from pixel observations and action-labelled trajectories.

The primary system should use:

* A learned visual encoder
* An action-conditioned latent dynamics model
* Latent-space planning

## Competition Principles

The competition should reward:

* Puzzle-solving ability
* Generalization
* Efficient planning
* Compact models
* Original technical ideas
* Reproducible experiments

Every entrant uses the same evaluation environment, datasets, compute
limits, and scoring script. None of it is negotiated per submission.

## Getting to a submission

Submissions are rolling: there is no fixed round, and you can evaluate
against the live API whenever you are ready. The path below is a suggested
order of work, not a schedule you have to keep.

### Reproduce the baseline

Run the shared baseline before introducing your own ideas, and confirm
that:

* Training runs successfully
* Representations do not collapse
* The planner can execute latent rollouts
* Evaluation is deterministic

This catches a broken pipeline before it looks like a bad research result.

### Do the research

Areas open to improvement:

* Encoder architecture
* Dynamics model
* Anti-collapse loss
* Multi-step training
* Planner
* Goal-scoring function
* Uncertainty estimation
* Dataset sampling
* Training curriculum

### Submit

An entry consists of:

* Frozen source code
* Configuration files
* Model checkpoint
* Training logs
* Reproduction command
* Short technical report

The final evaluation is run on hidden levels.

## Shared Environment

Official environment configuration:

* Board size: 8 × 8
* Number of boxes: 3
* Number of goals: 3
* Observation size: 64 × 64 RGB
* Actions: up, down, left, right
* Dynamics: deterministic
* Maximum episode length: 80 actions
* Invalid actions: treated as no-op transitions

A 6 × 6 one-box warmup configuration (Split W, 40-action limit) exists for
verifying a baseline runs. It is never scored.

Rationale: with three boxes on an 8 × 8 board the reachable state space
(~250,000 states) cannot be enumerated within the per-action planning
budget, so planning must be guided by a learned heuristic rather than
exhaustive search.

The final benchmark may include larger or more difficult boards.

## Allowed Inputs

During inference, the agent may receive:

* Current board image
* Goal board image
* Action history
* Its own model predictions

The agent may not receive:

* Symbolic board state
* Player coordinates
* Box coordinates
* Goal coordinates
* Ground-truth transition rules
* Solver-generated hints
* Test-level solution paths

## Training Data

Every entrant uses the same base dataset.

Recommended composition:

* 50% random trajectories
* 30% solver-generated trajectories
* 20% perturbed solver trajectories

The shared dataset should include:

* Valid player movement
* Wall collisions
* Successful box pushes
* Blocked box pushes
* Goal completion
* Near-deadlock states
* Invalid actions
* Repeated observations

Additional training data may be generated only under the data-budget rules below.

## Data Budget

Declare which mode your entry used. Both are valid; the mode is reported
alongside the result so entries are compared like for like.

### Fixed Dataset Mode

Train only on the shared dataset.

This mode best isolates architectural and algorithmic improvements.

### Fixed Environment-Steps Mode

Generate additional data, with total environment interactions capped at:

* 2 million transition samples

Every generated transition counts toward the limit.

## Compute Budget

Recommended final limits:

* Maximum model parameters: 20 million
* Maximum training time: 12 GPU-hours
* Maximum number of training runs: unrestricted during development
* Final checkpoint must come from one declared training run
* Maximum inference time: 500 milliseconds per environment action
* Maximum learned-dynamics calls: 256 per environment action
* Maximum memory usage: 8 GB GPU memory

Dynamics-call accounting: one call = one predicted latent transition of
one candidate state. A batched forward pass over B candidates rolled H
steps costs B × H calls. Every planner must tick the shared CallMeter
(latent_sokoban.agent) on each dynamics forward pass; the evaluation
harness fails any episode whose per-action count exceeds the cap.
Encoder passes and goal-scoring passes are free. Correct metering is
verified by source review of the frozen submission; an unmetered or
under-metered dynamics call is a rules violation.

Hardware should be identical where possible.

If hardware differs, final models should be evaluated on the same machine.

## Reproducibility Rules

Each final submission must include:

* Random seed
* Exact configuration
* Dependency versions
* Dataset version
* Training command
* Evaluation command
* Git commit hash
* Model parameter count
* Total training time
* Total training samples

The final training run should use one of the shared seeds:

```text
13
42
137
```

The official score should be averaged across at least three evaluation seeds.

## Required Baseline

The shared baseline should contain:

* Small convolutional encoder
* 128-dimensional latent state
* Learned action embeddings
* Residual MLP dynamics predictor
* One-step latent prediction loss
* Variance and covariance regularization
* Beam-search planner
* Euclidean latent goal distance
* Model-predictive control with replanning after every action

Verify the baseline runs before introducing custom improvements.

## Allowed Modifications

Entries may change:

* CNN or Vision Transformer encoder
* Latent dimensionality
* Dynamics predictor
* Transformer or recurrent dynamics
* Multi-step rollout loss
* SIGReg-style regularization
* Contrastive learning
* Temporal-distance learning
* Object-centric representations
* Beam search
* Random shooting
* Cross-entropy method
* Monte Carlo tree search
* Learned goal scoring
* Deadlock prediction
* Model ensembles
* Uncertainty penalties
* Training curriculum
* Data augmentation

## Prohibited Methods

The final agent may not:

* Use a symbolic Sokoban solver during evaluation
* Read the internal board representation
* Memorize hidden test levels
* Use manually coded Sokoban transition rules
* Use manually coded deadlock tables
* Query an external model or API during evaluation
* Use privileged simulator state for planning
* Modify the evaluation environment
* Train on hidden test levels
* Reconstruct a discrete board representation at inference time. No
  tile classification, object detection to grid coordinates, or any
  other module whose output is an exact symbolic board state, whether
  hand-coded or learned
* Perform graph or tree search over enumerated exact discrete states
  (learned latent states are fine; a learned lookup table over decoded
  board states is not)
* Bypass or under-report the dynamics-call meter

A symbolic solver may be used for training-data generation and evaluation analysis only.

The intent of these rules: the winning system must plan in a learned
representation under a small, audited planning budget. "Decode the image
to a board and search it" is prohibited regardless of whether the decoder
or the transition function was learned from data.

## Benchmark Splits

### Split W: Warmup (unscored)

* 6 × 6 boards, one box
* Used only to verify baselines during Phase 2

### Split A: Standard

* 8 × 8 boards
* Three boxes, three goals
* Similar visual style to training
* New unseen layouts

Purpose: measure core task performance.

### Split B: Visual Generalization

Possible changes:

* Different floor colours
* Different wall textures
* Different sprites
* Lighting changes
* Small image noise
* Distractor patterns

Purpose: test whether the encoder learned state rather than surface appearance.

### Split C: Structural Generalization

Possible changes:

* 10 × 10 boards
* Longer solution paths
* Unseen room shapes
* More obstacles
* Different player starting distributions

Purpose: test generalization beyond the training distribution.

### Split D: Deadlock Challenge

Levels include:

* Boxes near corners
* Narrow passages
* Tempting but irreversible pushes
* Multiple paths with only one successful route

Purpose: test planning quality and irreversible-error avoidance.

### Optional Split E: Five-Box Challenge

* 10 × 10 boards
* Five boxes, five goals
* Longer horizons (160-action limit)

This split is an unscored bonus and does not contribute to the final
score.

## Hidden Test Set

The hidden set is generated once from a secret seed and held by the
evaluation server. Levels never leave it: agents receive rendered frames
only, and no entrant, including the maintainer, plays them directly.

Because generation is fully determined by `(script, constraints, seed)`,
the seed alone is a commitment to the exact levels. Publishing it when a
round closes lets anyone regenerate the set and verify what was scored,
without the levels having been visible while the round was open.

Rotating the seed replaces the whole set and invalidates every prior score,
so it is done only between rounds, and announced.

## Evaluation Metrics

### Puzzle Success Rate

Percentage of levels solved within the action limit.

This is the primary metric.

### Generalization Success Rate

Average success across visual and structural generalization splits.

### Solution Efficiency

Compare the model's number of actions with an optimal or reference solution.

Move Efficiency = Optimal Moves / Agent Moves

Solved levels only.

### Planning Speed

Measure average wall-clock planning time per executed action.

### Model Efficiency

Reward smaller models and fewer latent rollouts.

### Deadlock Rate

Percentage of episodes in which the agent enters an irreversible deadlock.

### Stability

Measure performance variation across random seeds.

## Final Score

Recommended 100-point scoring system:

Final Score = 45S + 20G + 10M + 10P + 10D + 5R

Where:

* S: standard success score
* G: generalization score
* M: move-efficiency score
* P: planning-speed score
* D: deadlock-avoidance score
* R: reproducibility score

Each component is normalized between 0 and 1.

### Standard Success: 45 points

Based on success rate across Split A.

### Generalization: 20 points

Average performance across Splits B and C.

### Move Efficiency: 10 points

Based on path length relative to reference solutions.

### Planning Speed: 10 points

Recommended normalization:

* 10 points: below 50 ms per action
* 8 points: below 100 ms
* 6 points: below 200 ms
* 4 points: below 500 ms
* 0 points: above the agreed inference limit

### Deadlock Avoidance: 10 points

Based on performance on Split D.

### Reproducibility: 5 points

Awarded when the model can be trained and evaluated using the submitted instructions.

## Tie-Breaking Rules

Ties are resolved in this order:

1. Higher total puzzle success
2. Higher structural-generalization success
3. Lower deadlock rate
4. Lower average planning time
5. Smaller model
6. Fewer training samples
7. Earlier submission

## Bonus Awards

These awards are editorial and do not affect the score.

### Best Research Idea

Awarded for the most original or insightful technical contribution.

### Best Visualization

Awarded for the clearest demonstration of the model's imagined rollouts.

### Most Efficient Model

Awarded for the best performance per parameter or per millisecond.

### Most Spectacular Failure

Awarded when a model executes a uniquely confident and catastrophic box push.

### Best Technical Report

Awarded for the clearest explanation of experiments, results, and failures.

## Publication

Entrants are encouraged but not required to publish what they learned.
Negative results are especially welcome: the two findings recorded in the
project README came from instrumenting the baseline, and both saved work
that others would otherwise have repeated.

Nothing here restricts what you may publish or when. Your code is yours.

## Submission Format

Each final submission should have:

```text
submission/
├── README.md
├── requirements.txt
├── config.yaml
├── checkpoint.pt
├── train.py
├── evaluate.py
├── report.md
└── results.json
```

The results file should contain:

```json
{
  "entrant": "name",
  "git_commit": "commit-hash",
  "random_seed": 42,
  "parameter_count": 0,
  "training_samples": 0,
  "training_time_hours": 0,
  "planner": "beam_search",
  "planning_horizon": 0,
  "average_planning_time_ms": 0
}
```

## Required Technical Report

Each entry includes a short report covering:

1. Model architecture
2. Training objective
3. Anti-collapse method
4. Dataset strategy
5. Planner
6. Goal-scoring method
7. Best improvements
8. Failed experiments
9. Known weaknesses
10. Final results

Recommended length: two to four pages.

## Leaderboard

Standings are live at
[sokoban.lulzx.space/leaderboard](https://sokoban.lulzx.space/leaderboard),
ranked by success rate over the hidden set, with move efficiency and then
deadlock rate as tie-breaks. Each entrant's best scorecard is shown, so a
bad run cannot displace a good one.

## Dispute Resolution

Raise anything ambiguous before submitting rather than after scoring.

When a rule is ambiguous:

1. Prefer the interpretation that advantages no particular entry.
2. Document the decision in the open, so it applies to everyone.
3. Apply it identically to every submission, including ones already scored.
4. Do not change scoring rules after seeing results.

Decisions are made by the benchmark maintainer and recorded in this file's
history, so the rule that applied to any past score can be reconstructed.

## Entry Agreement

By submitting a result, you confirm that your entry:

* Follows these evaluation rules
* Did not inspect or train on hidden test levels
* Accurately reports compute and data use
* Was frozen before evaluation
* Meters dynamics calls honestly

Misreporting is the one thing that cannot be checked automatically, which
is why it is the one thing stated as a condition of entry.

## Recommended Default Settings

To begin immediately, use:

* Fixed Dataset Mode
* 8 × 8 three-box Sokoban (6 × 6 one-box warmup for the baseline round)
* 20-million-parameter limit
* 12 GPU-hour training budget
* 500 ms inference limit
* 256 counted dynamics calls per action
* 80-action episode limit
* Three evaluation seeds
* Four official benchmark splits
* Seven-day research round

## Before you submit

Confirm the shared tooling runs end to end:

```bash
python scripts/generate_dataset.py --out data/train --episodes 2000 --seed 13
python scripts/generate_levels.py --all --n 100 --seed 1001 --out levels/
python scripts/evaluate.py --agent random --splits levels/split_a.json
```

The shared baseline should achieve:

* Non-collapsed latent representations
* Reasonable one-step prediction accuracy
* At least occasional success on simple levels
* Identical evaluation results when run from the same checkpoint

Once that holds, your setup matches the one every entry is scored on.
