# RL environment and dense reward

`csrc/puffer/` ports the environment, renderer and solver to C and binds it
as a PufferLib environment, so a policy can be trained with PPO at hundreds
of thousands of steps per second. The port exists to test whether a
model-free policy — zero dynamics calls, no planner — can do what the
world model does.

## The sparse-reward dead end

The first version gave `+1` on solve and `0` otherwise, deliberately
matching how the leaderboard scores. That reward is a trap: a random policy
solves ~2% of episodes on 8×8, so PPO receives almost no reward signal and
has no gradient to follow. The symptom was empirical — warm-starting PPO
from the distilled policy made the policy *worse*, because there was no
signal to pull it anywhere better.

## Potential-based shaping

The fix is potential-based reward shaping, which is the one dense reward
that cannot change the optimal policy. For any state potential `Φ`, the
shaped reward

```
r = Φ(s') − Φ(s)   (+ a small per-step penalty, +1 on solve)
```

leaves the optimal policy unchanged (the result is exact for
`γΦ(s') − Φ(s)`; `γ ≈ 1` makes the correction negligible). So this is not
"training against a different objective" — it is giving the learner a dense
signal down the same objective.

The potential is the negative sum, over off-goal crates, of each crate's
free-grid distance to the nearest goal — a multi-source BFS over the wall
layout that ignores other crates, the standard admissible relaxation — with
a large penalty for a crate wedged in a corner off-goal. It is computed once
per reset; `sk_potential` in `csrc/puffer/sokoban_env.h` reads it in O(1)
per step.

`csrc/puffer/test_shaping.c` verifies the reward over a known optimal
solve: a six-move solution now yields a dense `+2.94` (one positive step
per crate-push toward goal, `+1` for the solve, `−0.01` per step) where the
sparse version paid a single `+1` at the very end. `SK_SHAPED=0` restores
the sparse reward for A/B comparison.

## Status

The environment and reward are in place and verified; the PPO training
itself is unverified work in progress. The open question is whether a
reactive policy can reach the longer-horizon, detour-requiring levels that
the world model's search is built for, or whether irreversibility makes
lookahead necessary. The two tracks are meant to be read together: the
world model plans, the RL policy proposes.
