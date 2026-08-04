#!/usr/bin/env python3
"""Generate benchmark split files.

Each split is a JSON file: {"name", "seed", "max_steps", "levels": [...]},
where every level entry has an ascii board, its optimal solution length,
and (Split B only) a rendering theme.

Usage:
    python scripts/generate_levels.py --split A --n 100 --seed 1001 --out levels/split_a.json
    python scripts/generate_levels.py --all --n 100 --out levels/

Hidden test set protocol: freeze this script (git tag), agree on the
constraints, then one run per split with a SECRET seed. Store the seed in
a password-protected archive; reveal it only after submissions are
frozen. Do not open the generated files before final evaluation.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from latent_sokoban.levels import (MIN_STEPS, STEP_MULTIPLE, _tier_band,
                                   generate_deadlock_level, generate_level)
from latent_sokoban.render import random_theme

# These are the local development splits, held at 8x8 with 3 boxes: large
# enough that the reachable state space (~250k states) cannot be exhausted
# within the 256-calls-per-action planning budget, so search must be
# learned-heuristic guided. Split W is the 6x6 warmup for the baseline
# round only. The hidden set the leaderboard scores is generated separately
# by latent_sokoban.levels.generate_hidden_set, on a 1-to-4 crate ramp.
#
# Split S is a diagnostic, not a difficulty step. W and A differ in board
# size AND crate count at once, so a score that drops from one to the other
# does not say which of the two caused it -- and those have different fixes:
# an encoder that does not transfer across tile-grid size, or a planner that
# cannot handle multiple crates. S is 8x8 with ONE crate, which pins each
# comparison to a single variable:
#
#     W (6x6, 1 crate)  ->  S (8x8, 1 crate)   board size
#     S (8x8, 1 crate)  ->  A (8x8, 3 crates)  crate count, nothing else
#
# The S-to-A comparison is exactly controlled: same size, same 0.10 wall
# density, only the crate count moves. W-to-S also shifts density 0.12->0.10,
# because S doubles as the public proxy for hidden tier 1 (see HIDDEN_TIERS),
# and matching that mattered more than matching W's density.
SPLITS = {
    # split: (size, n_boxes, wall_density, min_len, max_len, themed, deadlock, max_steps)
    "W": dict(size=6, n_boxes=1, wall_density=0.12, min_len=4, max_len=25,
              themed=False, deadlock=False, max_steps=40),
    # band/per-level step budget copied from HIDDEN_TIERS[0] so S predicts the
    # hidden opening rather than merely resembling it.
    "S": dict(size=8, n_boxes=1, wall_density=0.10, min_len=4, max_len=20,
              band=((4, 8), (14, 20)), per_level_steps=True,
              themed=False, deadlock=False, max_steps=60),
    "A": dict(size=8, n_boxes=3, wall_density=0.10, min_len=10, max_len=50,
              themed=False, deadlock=False, max_steps=80),
    "B": dict(size=8, n_boxes=3, wall_density=0.10, min_len=10, max_len=50,
              themed=True, deadlock=False, max_steps=80),
    "C": dict(size=10, n_boxes=3, wall_density=0.10, min_len=15, max_len=70,
              themed=False, deadlock=False, max_steps=120),
    "D": dict(size=8, n_boxes=3, wall_density=0.14, min_len=8, max_len=50,
              themed=False, deadlock=True, max_steps=80),
    "E": dict(size=10, n_boxes=5, wall_density=0.08, min_len=15, max_len=90,
              themed=False, deadlock=False, max_steps=160),
}


def generate_split(name: str, n: int, seed: int) -> dict:
    cfg = SPLITS[name]
    rng = np.random.default_rng(seed)
    levels = []
    while len(levels) < n:
        gen = generate_deadlock_level if cfg["deadlock"] else generate_level
        # A flat [min_len, max_len] band accepts whatever the sampler happens
        # to produce, and short boards are far likelier: split S over a flat
        # 4-20 band came out with a mean optimal length of 9.4 and only two
        # levels past 15. Splits carrying `band` instead interpolate a narrow
        # band per level the way generate_hidden_set does, which forces the
        # long levels to exist rather than hoping for them.
        if cfg.get("band"):
            lo, hi = _tier_band(len(levels), n, *cfg["band"])
        else:
            lo, hi = cfg["min_len"], cfg["max_len"]
        for attempt in range(4):
            try:
                level, solution = gen(
                    rng,
                    size=cfg["size"],
                    n_boxes=cfg["n_boxes"],
                    wall_density=cfg["wall_density"],
                    min_solution_len=lo,
                    max_solution_len=hi,
                )
                break
            except RuntimeError:
                # Same escape hatch as the hidden set: widen rather than hang
                # when a target lands where sampling is very unlikely.
                if attempt == 3:
                    raise
                lo, hi = max(2, lo - 3), hi + 4
        # A flat budget is loose on short levels and starving on long ones; the
        # hidden set scales it per level, and S has to match or it scores its
        # easy levels more generously than the tier it stands in for.
        steps = (max(MIN_STEPS, STEP_MULTIPLE * len(solution))
                 if cfg.get("per_level_steps") else cfg["max_steps"])
        entry = {
            "ascii": level.to_ascii(),
            "optimal_len": len(solution),
            "max_steps": steps,
        }
        if cfg["themed"]:
            entry["theme"] = dataclasses.asdict(random_theme(rng))
        levels.append(entry)
    return {"name": f"split_{name}", "seed": seed, "max_steps": cfg["max_steps"],
            "levels": levels}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=sorted(SPLITS), help="single split to generate")
    parser.add_argument("--all", action="store_true",
                        help="generate warmup split W, splits A-D, and diagnostic S")
    parser.add_argument("--n", type=int, default=100, help="levels per split")
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--out", required=True, help="output file (or directory with --all)")
    args = parser.parse_args()

    if args.all:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        # S is appended rather than slotted in after W: the per-split seed is
        # `--seed + i`, so inserting it mid-string would renumber A-D and
        # regenerate the committed split files as different levels.
        for i, name in enumerate("WABCDS"):
            split = generate_split(name, args.n, args.seed + i)
            path = out_dir / f"split_{name.lower()}.json"
            path.write_text(json.dumps(split, indent=1))
            lens = [e["optimal_len"] for e in split["levels"]]
            print(f"split {name}: {len(lens)} levels, optimal len "
                  f"{min(lens)}-{max(lens)} (mean {sum(lens)/len(lens):.1f}) -> {path}")
    elif args.split:
        split = generate_split(args.split, args.n, args.seed)
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(split, indent=1))
        print(f"split {args.split}: {args.n} levels -> {path}")
    else:
        parser.error("pass --split X or --all")


if __name__ == "__main__":
    main()
