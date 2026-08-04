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

from latent_sokoban.levels import generate_deadlock_level, generate_level
from latent_sokoban.render import random_theme

# Official configuration is 8x8 with 3 boxes: large enough that the
# reachable state space (~250k states) cannot be exhausted within the
# 256-calls-per-action planning budget, so search must be learned-heuristic
# guided. Split W is the 6x6 warmup used during the baseline round only.
SPLITS = {
    # split: (size, n_boxes, wall_density, min_len, max_len, themed, deadlock, max_steps)
    "W": dict(size=6, n_boxes=1, wall_density=0.12, min_len=4, max_len=25,
              themed=False, deadlock=False, max_steps=40),
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
        level, solution = gen(
            rng,
            size=cfg["size"],
            n_boxes=cfg["n_boxes"],
            wall_density=cfg["wall_density"],
            min_solution_len=cfg["min_len"],
            max_solution_len=cfg["max_len"],
        )
        entry = {
            "ascii": level.to_ascii(),
            "optimal_len": len(solution),
            "max_steps": cfg["max_steps"],
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
                        help="generate warmup split W plus official splits A-D")
    parser.add_argument("--n", type=int, default=100, help="levels per split")
    parser.add_argument("--seed", type=int, default=1001)
    parser.add_argument("--out", required=True, help="output file (or directory with --all)")
    args = parser.parse_args()

    if args.all:
        out_dir = Path(args.out)
        out_dir.mkdir(parents=True, exist_ok=True)
        for i, name in enumerate("WABCD"):
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
