# Level generation

The hidden set is 100 levels on an 8×8 board, ordered easiest first and
generated deterministically from a secret seed.

## How a single level is made

`latent_sokoban.levels.generate_level` uses rejection sampling:

1. Scatter walls at the requested density, then place crates, goals and the
   player on free squares.
2. Run [the BFS solver](reference/solver.md) on the result.
3. Keep the level only if it is solvable **and** its optimal solution length
   falls inside the requested band. Otherwise discard and resample.

This guarantees every level is solvable and gives direct control over
difficulty through the solution-length band. Crates start away from the
border ring, because a crate against the edge has no free square on one
push axis and could never move.

The generator is a pure function of the `numpy.random.Generator` handed to
it, which is what makes the hidden-set protocol work: freeze the file,
agree the constraints, generate from a secret seed, and publish the seed
afterwards so anyone can reproduce exactly what was scored.

## The difficulty ramp

Difficulty rises on two axes at once, defined in `HIDDEN_TIERS`:

| Levels | Crates | Wall density | Solution band (start → end) |
| --- | --- | --- | --- |
| 1–25 | 1 | 0.10 | 4–8 → 14–20 |
| 26–50 | 2 | 0.12 | 9–14 → 22–30 |
| 51–80 | 3 | 0.14 | 13–18 → 28–38 |
| 81–100 | 4 | 0.18 | 17–23 → 34–46 |

Within a tier the band slides linearly from its start to its end value, so
there is no cliff at a tier boundary: the tail of one tier overlaps the
head of the next.

The set actually in play on the server spans 6 to 42 optimal moves.

### Why crate count carries the ramp

Solution length alone cannot do it. Measured optimal-length distributions,
sampling 400 levels per configuration (170 for the 4-crate case):

| Crates | Density | p5 | p25 | p50 | p75 | p95 | max | generation rate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.10 | 4 | 7 | 9 | 13 | 18 | 33 | 2265/s |
| 2 | 0.12 | 9 | 13 | 16 | 20 | 28 | 52 | 81/s |
| 3 | 0.14 | 12 | 17 | 20 | 24 | 35 | 49 | 7.6/s |
| 4 | 0.18 | 16 | 21 | 26 | 31 | 42 | 54 | 1.1/s |

At a fixed 3 crates on 8×8 the ceiling is around 46 optimal moves, with a
median near 20. A ramp that only slid the band would barely separate level
1 from level 100.

Crate count is the real lever anyway, because it is *combinatorial*: each
extra crate multiplies the reachable state space and the number of ways to
deadlock irreversibly. Two crates are far more than twice one crate.

### Why the board stays 8×8

Growing the board would widen the range further, but a 10×10 level rendered
into the same 64×64 frame changes the pixels-per-tile scale from 8 to 6.4.
An agent would have to handle two tile scales inside one run, which tests
something other than what this benchmark is about. The board size is fixed
so a 64×64 observation always means the same thing.

### Generation cost

The 4-crate tier is roughly 2000× slower to generate than the 1-crate tier,
because the solver is being run on far more rejected candidates and each
BFS is over a much larger state space. The whole 100-level set takes about
70 seconds, nearly all of it in the last 20 levels.

That is why `scripts/generate_hidden.py` exists as an offline step. The
server can generate the set at boot if the file is missing, but the API is
unreachable while it does.

```bash
python scripts/generate_hidden.py --seed "$SOKOBAN_SEED" --out hidden_levels.json
```

## Other splits

`scripts/generate_levels.py` builds the public training and validation
splits, which are separate from the hidden set:

| Split | Board | Crates | Purpose |
| --- | --- | --- | --- |
| W | 6×6 | 1 | Warmup, used for the baseline round |
| A | 8×8 | 3 | Standard configuration |
| B | 8×8 | 3 | As A, with randomised colour themes |
| C | 10×10 | 3 | Larger board generalisation |
| D | 8×8 | 3 | Deadlock-heavy, higher wall density |
| E | 10×10 | 5 | Stress split |

Split B exists to test whether a representation survives a change of
palette. An agent that has memorised specific RGB values rather than
learning structure falls over on it.

## Calibration

The reference baseline scores 28% on split W and 0% on split A. Tier 1 of
the hidden set sits between the two: same single crate as W, but on the
larger 8×8 board. So the opening levels are not free, and the closing tiers
are, for now, out of reach of anything published. That is deliberate. A
benchmark where every entrant scores zero everywhere provides no gradient
to improve against.
