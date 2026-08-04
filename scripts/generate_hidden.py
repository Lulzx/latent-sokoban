"""Generate the hidden evaluation set: 100 levels on a difficulty ramp.

The set is ordered easiest-first. Difficulty rises on two axes at once:

  * **crate count** (1 -> 4), which is the real lever. Each extra crate
    multiplies the reachable state space and the number of ways to deadlock,
    so it raises difficulty far faster than solution length alone does.
  * **solution length**, via a band that slides upward across each tier.

Board size stays 8x8 throughout, so a 64x64 observation always means the
same 8 pixels per tile and an agent never has to handle two tile scales
mid-run.

Measured optimal-length distributions for the tier configs (n=400 samples
each, n=170 for the 4-crate tier) are what the bands below are drawn from:

    crates  density   p5   p25   p50   p75   p95   max   gen rate
      1       0.10     4     7     9    13    18    33   2265/s
      2       0.12     9    13    16    20    28    52     81/s
      3       0.14    12    17    20    24    35    49    7.6/s
      4       0.18    16    21    26    31    42    54    1.1/s

Bands run from roughly each tier's p25 to its p95, so the tail of one tier
overlaps the head of the next and the curve has no cliff at a boundary.

Generation is slow at the top of the ramp (4-crate levels with a long-band
constraint are heavily rejection-sampled), which is why this is an offline
script rather than something the server does on first boot -- a multi-minute
startup would trip systemd's start timeout.

Usage:
    python scripts/generate_hidden.py --seed $SOKOBAN_SEED --out hidden_levels.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latent_sokoban.levels import generate_level

# (n_levels, n_boxes, wall_density, band_start, band_end)
# band_start applies to the tier's first level, band_end to its last;
# every level in between interpolates linearly.
TIERS = [
    (25, 1, 0.10, (4, 8), (14, 20)),
    (25, 2, 0.12, (9, 14), (22, 30)),
    (30, 3, 0.14, (13, 18), (28, 38)),
    (20, 4, 0.18, (17, 23), (34, 46)),
]

# A step budget proportional to the level's own optimal solution, rather
# than one flat number that is generous for short levels and starving for
# long ones.
STEP_MULTIPLE = 3
MIN_STEPS = 30


def _band(i: int, n: int, start: tuple[int, int], end: tuple[int, int]) -> tuple[int, int]:
    """Interpolate the (min, max) solution-length band across a tier."""
    t = i / max(n - 1, 1)
    lo = round(start[0] + t * (end[0] - start[0]))
    hi = round(start[1] + t * (end[1] - start[1]))
    return lo, max(hi, lo + 2)


def generate(seed: int, verbose: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    levels: list[dict] = []
    t0 = time.time()

    for n, n_boxes, density, start, end in TIERS:
        for i in range(n):
            lo, hi = _band(i, n, start, end)
            # Widen the band on repeated failure rather than hanging: the
            # measured distributions are noisy at the tails, so an
            # occasional target lands where sampling is very unlikely.
            for attempt in range(4):
                try:
                    level, solution = generate_level(
                        rng, size=8, n_boxes=n_boxes, wall_density=density,
                        min_solution_len=lo, max_solution_len=hi,
                        max_tries=20000)
                    break
                except RuntimeError:
                    lo, hi = max(2, lo - 3), hi + 4
                    if attempt == 3:
                        raise
            optimal = len(solution)
            levels.append({
                "ascii": level.to_ascii(),
                "optimal_len": optimal,
                "n_crates": n_boxes,
                "max_steps": max(MIN_STEPS, STEP_MULTIPLE * optimal),
            })
            if verbose and len(levels) % 10 == 0:
                print(f"  {len(levels):3d}/100  crates={n_boxes} "
                      f"band=[{lo},{hi}] optimal={optimal} "
                      f"({time.time() - t0:.0f}s)", file=sys.stderr)

    return {
        "name": "hidden_public_v2",
        "n_levels": len(levels),
        "ramp": "1-25: 1 crate, 26-50: 2, 51-80: 3, 81-100: 4",
        "levels": levels,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    data = generate(args.seed, verbose=not args.quiet)
    Path(args.out).write_text(json.dumps(data))

    lens = [lv["optimal_len"] for lv in data["levels"]]
    print(f"wrote {args.out}: {len(lens)} levels, "
          f"optimal {min(lens)}-{max(lens)} moves, "
          f"total step budget {sum(lv['max_steps'] for lv in data['levels'])}")


if __name__ == "__main__":
    main()
