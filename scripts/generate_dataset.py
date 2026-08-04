#!/usr/bin/env python3
"""Generate the shared training dataset.

Composition follows the competition recommendation:
    50% random trajectories
    30% optimal solver trajectories
    20% perturbed solver trajectories

The dataset is sharded; every shard is an .npz plus a .json sidecar with
counts (see latent_sokoban/dataset.py for the exact format). Total
transitions are reported so Fixed Environment-Steps Mode budgets can be
audited.

Usage:
    python scripts/generate_dataset.py --out data/train --episodes 2000 --seed 13
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from latent_sokoban.dataset import generate_shard, save_shard


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, help="output directory")
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--shard-size", type=int, default=500, help="episodes per shard")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--size", type=int, default=8, help="board size")
    parser.add_argument("--boxes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--mix", type=float, nargs=3, default=(0.5, 0.3, 0.2),
                        metavar=("RANDOM", "SOLVER", "PERTURBED"))
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    total_transitions = 0
    total_frames = 0
    n_shards = (args.episodes + args.shard_size - 1) // args.shard_size
    for i in range(n_shards):
        n = min(args.shard_size, args.episodes - i * args.shard_size)
        shard = generate_shard(
            rng, n, size=args.size, n_boxes=args.boxes,
            max_steps=args.max_steps, mix=tuple(args.mix),
        )
        transitions = int((shard["actions"] != -1).sum())
        total_transitions += transitions
        total_frames += len(shard["frames"])
        path = out / f"shard_{i:04d}.npz"
        save_shard(shard, path, meta={
            "seed": args.seed, "shard": i, "board_size": args.size,
            "n_boxes": args.boxes, "mix": list(args.mix),
        })
        print(f"shard {i}: {n} episodes, {transitions} transitions -> {path}")

    print(f"\ntotal: {args.episodes} episodes, {total_frames} frames, "
          f"{total_transitions} transitions")
    print("(every transition counts toward the budget in "
          "Fixed Environment-Steps Mode)")


if __name__ == "__main__":
    main()
