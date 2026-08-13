# Fast distance-to-goal labelling

Training the world model's value and policy heads needs, for every state in
a rollout chain, its true optimal distance-to-goal and the set of optimal
actions. The symbolic solver computes that, and using it for training-data
generation is permitted — but it has to be fast enough to generate data at
3–4 crates, which is exactly where the naive version stalls.

## The problem

`wm/generate.py`'s original `label()` ran a BFS per state, plus one per
successor — up to five BFS per state, ~200 per level. Measured throughput:
0.5 levels/s at three crates and 0.16 levels/s at four.

The first fix was a straightforward C port: per-state forward BFS through
the solver in `csrc/solver.h`. It was correct but **slower than Python at
four crates** (42 s/level against 6 s/level). The reason is in the hash
table: it grows by powers of two, so a ~50k-node search gets a 1M-entry
table, and every probe is a cache miss into cold memory. CPython's dict,
sized to its contents, wins on large searches.

## Reverse BFS: one pass per level

The right answer is to invert the problem. Distance-to-goal is forward
distance to the goal; by the graph-reversal theorem that equals reverse
distance from the goal, over edges that are the exact reversals of the
forward moves. Sokoban's reverse move is well-defined:

- the reverse of a walk is a walk;
- the reverse of a push is a **pull** (the player steps back behind the
  crate and drags it back one tile).

So a single BFS from the goal states (any player position with every crate
on a goal) over these reversed edges fills the distance-to-goal for every
solvable state of the level at once. One subtlety, caught by
`csrc/verify_label.py` before it shipped: a direction can contribute **two**
reverse successors, not one — from a state with a crate ahead of the player,
both "walk back" and "pull the crate" are distinct reverse edges. The first
version collapsed them into one and silently overestimated distances.

## Dense indexing instead of a hash table

The first reverse-BFS implementation stored states in a hash table and was
still cache-hostile at 3–4 crates (random-access probes into a grown
table). The fast version stores distances in a **dense array indexed by the
combinadic rank of the crate positions over the level's free cells, times
the player cell**. For an 8×8 board with ~30 free cells the whole 4-crate
state space is `C(30,4)·30 ≈ 820k` entries (~1.6 MB as uint16), so the BFS
does direct indexed writes and each query is one array read.

Measured, versus the earlier per-state BFS in Python:

| Tier | Before | Dense reverse BFS |
| --- | --- | --- |
| 3 crates | 1.37 s/level | 0.22 s/level |
| 4 crates | 16–24 s/level | 1.45 s/level |

At every crate count the remaining cost is now Python *level generation*
(rejection sampling, ~1.1 levels/s at 4 crates), not labelling. There is no
longer a crate-count threshold: the dense table is cheap everywhere.

`csrc/verify_label.py` is the regression test — it regenerates levels and
asserts the C labels equal the Python solver's on every state, on every
crate count (currently 470/470 across one to four crates).

## Structural dead states

The dead head is only as good as the dead states it was trained on, and the
original generator only ever synthesised **corner** deadlocks (a crate in a
corner off-goal). At two or more crates the deadlocks that actually lose
episodes are structural — a crate flat against a wall with no goal in its
slide line, in a dead region, blocking a corridor.

`wm/generate.py --structural-dead N` adds N such states per level: a random
crate is relocated to a wall-adjacent (or random) non-goal cell, and the
result is confirmed dead — with the exact distance table (a dead state has
no entry), or a conservative wall/corner heuristic at four crates. This took
the dead-label fraction of chain steps from ~15% to ~29%.

## Interface

- `csrc/label_impl.c` exposes `fill_dist` / `lookup_dist` / `free_dist`
  over ctypes, built with
  `cc -O2 -shared -o csrc/liblabelsokoban.dylib csrc/label_impl.c`.
- `latent_sokoban/clabel.py` wraps it as a `DistTable`: build one per level,
  call `dist(state)` for O(1) lookups, `close()` to free it.
- `wm/generate.py --shard N` writes `shard_NNNN.npz`, so the four crate
  tiers can share one output directory and `wm/train.py` loads them all.
