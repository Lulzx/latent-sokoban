# The spatial world model

The reference agent in `wm/`. It is the crate-count-generalising system the
benchmark's difficulty ramp is calibrated around: a spatial latent world
model that plans in latent space with a policy-guided beam search.

It is not part of the neutral benchmark infrastructure — it is one entry's
worth of research, kept in this repo as the shared baseline is, so the
numbers it produces are reproducible and the findings that shaped it are
on record.

## Architecture

The three measured facts that drove the design, from `lab/` on the earlier
global-vector model:

1. Open-loop latent rollouts lost the true state after **one** imagined step
   (retrieval 0.717, decaying to 0.49 by step 3). Planning past horizon 3
   was scoring noise.
2. Success at three crates was 0.000 across splits A, C and D, with
   deadlock rates of 0.73–0.93. A model trained at one crate transferred to
   nothing.
3. Giving the old planner *perfect* dynamics changed its score by exactly
   zero, so the losses were in the heads, not the transition model.

The structural response to 1 and 2 is to stop throwing away space. A
Sokoban transition changes at most three tiles — the tile the player
leaves, the tile it enters, and the tile a crate moves to — and that rule
is identical everywhere on the board and for every crate. A dense MLP over a
pooled 128-d vector can represent none of that; it has to memorise each
configuration separately, which is exactly what "works at one crate,
collapses at three" looks like.

| Component | Design | Why |
| --- | --- | --- |
| Encoder | `64×64×3 → 48×8×8`, downsample exactly 8 | At 8×8 one latent cell is one tile, because `render.py` draws `64 // board_size` pixels per tile. The sharp prior is the point. |
| Dynamics | residual 3×3 conv, receptive field 7 cells, action broadcast as channels | Local enough for the push rule, too narrow to memorise a board. Weight sharing is what makes crate count close to free. |
| Heads | dilated trunk (RF 15) over `(z, z_goal)`, mean+max pooling, value / policy / dead | Mean carries "how much is left", max carries "is one tile catastrophically wrong" — which is what a deadlock is. |
| Dead head | supervised on **both** the encoder and the rolled manifold | The previous model only ever saw dead labels through dynamics-produced latents, so its dead-state AUC was 0.918 on dynamics latents and 0.427 — below chance — on encoder ones. |

Scope, stated so it is not over-claimed: this buys crate-count
generalisation, **not** board-size generalisation. Observations are always
64×64, so a 10×10 board renders at 6 pixels per tile and the tile-to-cell
correspondence only holds at 8×8 — which is fine, because every one of the
hidden set's 100 levels is 8×8 and only crate count ramps. Splits W and C,
which do change board size, stay hard on purpose.

## The loss, and the divergence it fixes

The first formulation scored rollouts with raw MSE and contrasted **deltas**.
It diverged. The measured symptom, from `wm/train.py`'s logs:

- one-step prediction error *relative to the null model* ("nothing changes")
  ran 219× → 853× → 1,710× → 39,270× → 576,558×;
- the delta-contrastive accuracy sat at ~0.05, i.e. chance;
- the policy head sat flat at `ln 4 = 1.386`, the score of guessing.

The diagnosis is in the raw numbers: `null_1 ≈ 8e-6` while `pred_1 ≈ 7e-4`.
A single Sokoban move changes 3 of 64 tiles, so the encoder mapped
consecutive frames to nearly-identical latents. The dynamics therefore
answered "no change" and looked good under raw MSE, while the delta-NCE
normalised an already-degenerate delta and couldn't move it.

The fix is a **spherical-contrastive** objective, in the spirit of SCoW:

- the latent is globally L2-normalised (rescaled by `LATENT_SCALE = 8` so
  the conv heads see healthy magnitudes), putting every state on a
  fixed-radius sphere where a move is an *angular* delta comparable across
  states;
- the prediction objective is **cosine alignment** plus **infoNCE over
  positions** — "which candidate is my true next state?" — which cannot be
  answered by predicting no change.

Contrasting positions was tried before and discarded, because without the
sphere it pushed rollouts off the encoder manifold (relative prediction
error 4400×). On a sphere that degree of freedom does not exist, so the two
terms no longer fight. The VICReg regulariser became unnecessary (the sphere
plus the contrastive term prevent collapse) and was dropped.

After the fix, one 6000-step run reports `nceacc 1.000`, alignment 0.97,
pairwise-state spread 0.02 (no collapse), policy top-1 accuracy 0.98 and
dead loss 0.002.

## The planner

Beam-search MPC, replanning every action, ranked by

```
value(z_leaf)  −  β · Σ log π(a_t | z_t)  +  λ · sigmoid(dead(z_leaf))
```

Each term exists because a measurement said so: `lab/attribute.py` measured
that ranking by value alone is *worse* than taking the policy head's argmax
with no search at all, and `lab/probe.py` measured the policy head naming an
optimal action 0.894 of the time against the value head's 0.852. The policy
proposes, the value refines, the dead head spends the deadlock knowledge.

The search is **policy-guided** (`TOPK`): instead of a uniform 4-way
fan-out, each beam node expands only the top-`k` actions by policy
log-probability. Because the policy is right ~0.9 of the time, the pruned
branch rarely carries the optimum, while the saved dynamics calls buy
horizon instead of fan-out — `TOPK=2` takes the same 84-call plan from
horizon 3 to horizon ~8 inside the 256 cap. Only dynamics calls are metered;
encoder and head passes are free, on the same ground as the baseline's
goal-scoring.

## What it scores, and where the gap is

The converged 1-crate checkpoint (`wm/checkpoint_8x1.pt`), measured by
`lab/probe.py` and `lab/attribute.py`:

| Probe | Value |
| --- | --- |
| value monotonicity (Spearman vs true dist-to-go) | 0.945 |
| greedy action accuracy, perfect dynamics (value / policy) | 0.852 / 0.894 |
| dead-head AUC (encoder / dynamics latents) | 1.000 / 0.872 |
| open-loop retrieval, 1 → 6 imagined steps | 0.52 → 0.29 |

The crate-count bet paid off structurally: the 1-crate-trained model
produces the project's first non-zero three-crate result (10% via value
beam on Split A, where every prior model scored 0.000). The first live
evaluation against the hidden set posted **15 / 100 solved** (success rate
0.15, move efficiency 0.70, deadlock rate 0.11) — the opening tier plus one
solve into the two-crate tier.

Two gaps remain, and they are the same two the README recorded when
instrumenting the earlier baseline ("plan short, replan often", and "pure
greedy oscillates"):

- **Open-loop drift still caps the horizon.** One-step dynamics is sharp
  (alignment 0.97) but retrieval decays to 0.29 by step 3, so beam search
  adds little over the reactive policy at one crate. This is the single
  most obvious research direction.
- **The model has only ever seen one crate.** Its dead head knows exactly
  one pattern — a crate in a corner off-goal — while three- and four-crate
  deadlocks are structural (crates blocking crates). The fix is training on
  mixed 1–4 crate data, which the [labelling](labelling.md) tooling exists
  to make cheap.
