# Fast distance-to-goal labelling

Training the world model's value and policy heads needs, for every state in
a rollout chain, its true optimal distance-to-goal and the set of optimal
actions. The symbolic solver computes that, and using it for training-data
generation is permitted — but it has to be fast enough to generate data at
3–4 crates, which is exactly where the naive version stalls.

## The problem

`wm/generate.py`'s original `label()` ran a BFS per state, plus one per
successor — up to five BFS per state, ~200 per level. Measured throughput:
0.5 levels/s at three crates and 0.16 levels/s at four. A few hundred
four-crate levels would take hours, and the four-crate shard is precisely
what the world model needs most.

The first fix was a straightforward C port: per-state forward BFS through
the solver in `csrc/solver.h`. It was correct but **slower than Python at
four crates** (42 s/level against 6 s/level). The reason is in the hash
table: it grows by powers of two, so a ~50k-node search gets a 1M-entry
table, and every probe is a cache miss into cold memory. CPython's dict,
sized to its contents, wins on large searches. The comment in `solver.h`
already warns about exactly this for small searches; the large-search
direction bites the same way.

## Reverse BFS: one pass per level

The right answer is to invert the problem. Distance-to-goal is forward
distance to the goal; by the graph-reversal theorem that equals reverse
distance from the goal, over edges that are the exact reversals of the
forward moves. Sokoban's reverse move is well-defined:

- the reverse of a walk is a walk;
- the reverse of a push is a **pull** (the player steps back behind the
  crate and drags it back one tile).

So `sk_fill_dist` in `csrc/solver.h` runs a single BFS from the goal states
(any player position with every crate on a goal) over these reversed edges,
filling the distance-to-goal for every non-dead state of the level at once.
Each query is then an O(1) hash lookup.

One subtlety, caught by `csrc/verify_label.py` before it shipped: a
direction can contribute **two** reverse successors, not one. From a state
with a crate sitting ahead of the player, both "walk back" and "pull the
crate" are distinct valid reverse edges leading to different states. The
first version collapsed them into one, which silently overestimated
distances. The checker compares C labels against the Python solver on
shared boards and now reports 470/470 matches across one to four crates.

## Where reverse BFS does not help

Reverse BFS costs O(non-dead states). That set is ~250k at three crates,
where the single pass is fast — but it explodes at four, where the
per-state forward BFS (bounded by the near-optimal states actually queried)
is faster. So `wm/generate.py` uses the C distance table for ≤3 crates and
falls back to the Python forward BFS for four. Measured effect: three crates
went 0.5 → 0.73 levels/s, with the remaining cost in Python rendering and
generation rather than labelling.

## Interface

- `csrc/label_impl.c` exposes `fill_dist` / `lookup_dist` / `free_dist`
  over ctypes, built with
  `cc -O2 -shared -o csrc/liblabelsokoban.dylib csrc/label_impl.c`.
- `latent_sokoban/clabel.py` wraps it as a `DistTable`: build one per level,
  call `dist(state)` for O(1) lookups, `close()` to free it.
- `wm/generate.py --shard N` writes `shard_NNNN.npz`, so the four crate
  tiers can share one output directory and `wm/train.py` loads them all.

`csrc/verify_label.py` is the regression test: it regenerates levels and
asserts the C labels equal the Python labels on every state, on every crate
count.
