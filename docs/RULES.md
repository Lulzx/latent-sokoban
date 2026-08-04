# Latent Sokoban Challenge

A two-person research competition to build the best pixel-based latent world model for Sokoban.

## Competitors

* Competitor A: ____________________
* Competitor B: ____________________

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

Both competitors must use the same evaluation environment, datasets, compute limits, and scoring script.

## Timeline

### Phase 1: Shared Infrastructure

Duration: 3 days

Build together:

* Sokoban environment
* Image renderer
* Dataset format
* Level generator
* Evaluation script
* Shared baseline
* Hidden test-set generation process

No competitive scoring occurs during this phase.

### Phase 2: Baseline Round

Duration: 4 days

Both competitors independently reproduce the shared baseline.

The goal is to verify that:

* Training runs successfully
* Representations do not collapse
* The planner can execute latent rollouts
* Evaluation is deterministic and fair

### Phase 3: Research Round

Duration: 7 to 14 days

Each competitor may independently improve:

* Encoder architecture
* Dynamics model
* Anti-collapse loss
* Multi-step training
* Planner
* Goal-scoring function
* Uncertainty estimation
* Dataset sampling
* Training curriculum

### Phase 4: Final Evaluation

Duration: 1 day

Each competitor submits:

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

A 6 × 6 one-box warmup configuration (Split W, 40-action limit) is used
during the baseline round only and is never scored.

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

Both competitors receive the same base dataset.

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

Each competitor may generate additional training data only under the agreed data-budget rules.

## Data Budget

Choose one competition mode before starting.

### Fixed Dataset Mode

Each competitor may train only on the shared dataset.

This mode best isolates architectural and algorithmic improvements.

### Fixed Environment-Steps Mode

Each competitor may generate additional data, but total environment interactions are capped.

Recommended limit:

* 2 million transition samples

Every generated transition counts toward the limit.

Selected mode:

* [ ] Fixed Dataset Mode
* [ ] Fixed Environment-Steps Mode

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

Both competitors must demonstrate that the baseline runs before introducing custom improvements.

## Allowed Modifications

Competitors may change:

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

This split should be considered a bonus round unless both competitors agree to make it official.

## Hidden Test Set

The final test set must remain unseen until evaluation.

Recommended procedure:

1. Write and freeze the level-generation script.
2. Agree on generation constraints.
3. Generate levels using a secret random seed.
4. Store the seed in a password-protected archive.
5. Do not inspect generated levels before final evaluation.
6. Reveal the password only after both submissions are frozen.

Alternatively, ask a trusted third person to generate and hold the test set.

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
7. Best-of-five live Sokoban match

## Live Final

For the live final:

1. Each competitor selects three unseen generated levels.
2. Neither competitor sees the levels beforehand.
3. Models alternate solving levels.
4. Each model gets one attempt per level.
5. The evaluation screen displays:

   * Current state
   * Goal state
   * Selected action
   * Imagined action sequence
   * Planning time
   * Current goal distance
6. No code changes are allowed during the final.

## Bonus Awards

These awards do not affect the main score unless agreed beforehand.

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

## Experiment Sharing Rules

During the research round, competitors may choose one of these modes.

### Fully Independent

No sharing of results, code, or technical ideas until final submission.

### Weekly Reveal

At the end of each week, both competitors share:

* Best score
* One successful idea
* One failed idea
* Current model size
* Current planning speed

### Open Research

Both competitors may discuss discoveries, but code remains separate.

Selected mode:

* [ ] Fully Independent
* [ ] Weekly Reveal
* [ ] Open Research

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
  "competitor": "name",
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

Each competitor submits a short report covering:

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

## Shared Leaderboard

| Competitor   | Standard Success | Generalization | Deadlock Success | Move Efficiency | Planning Time | Parameters | Final Score |
| ------------ | ---------------: | -------------: | ---------------: | --------------: | ------------: | ---------: | ----------: |
| Competitor A |                - |              - |                - |               - |             - |          - |           - |
| Competitor B |                - |              - |                - |               - |             - |          - |           - |

## Dispute Resolution

Any disagreement about fairness should be resolved before final evaluation.

When a rule is ambiguous:

1. Prefer the interpretation that gives neither competitor an advantage.
2. Document the decision.
3. Apply the decision equally to both submissions.
4. Do not change scoring rules after seeing final results.

## Competition Agreement

By signing below, both competitors agree to:

* Follow the same evaluation rules
* Avoid inspecting hidden test levels
* Accurately report compute and data use
* Freeze submissions before evaluation
* Accept the final scoring procedure
* Share results and lessons after the competition

Competitor A:

Name: ____________________

Signature: ____________________

Date: ____________________

Competitor B:

Name: ____________________

Signature: ____________________

Date: ____________________

## Recommended Default Settings

To begin immediately, use:

* Fixed Dataset Mode
* Weekly Reveal Mode
* 8 × 8 three-box Sokoban (6 × 6 one-box warmup for the baseline round)
* 20-million-parameter limit
* 12 GPU-hour training budget
* 500 ms inference limit
* 256 counted dynamics calls per action
* 80-action episode limit
* Three evaluation seeds
* Four official benchmark splits
* Seven-day research round

## First Shared Milestone

Before competing, both competitors must successfully run:

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

Once this milestone is complete, the competition officially begins.
