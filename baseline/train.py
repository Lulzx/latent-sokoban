#!/usr/bin/env python3
"""Train the shared baseline world model on a generated dataset.

Usage:
    python baseline/train.py --data data/train --out baseline/checkpoint.pt \
        --steps 5000 --seed 13

Watch latent_std in the logs: it collapsing toward 0 means the variance
regularizer lost — the representation is dead and planning cannot work.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline.model import WorldModel, param_count
from latent_sokoban.dataset import load_shard


def load_transitions(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate all shards into (obs_idx, action, next_idx) over a frame bank."""
    frames_list, triples = [], []
    offset = 0
    for shard_path in sorted(data_dir.glob("*.npz")):
        shard = load_shard(shard_path)
        frames = shard["frames"]
        actions = shard["actions"]
        valid = np.where(actions != -1)[0]
        for i in valid:
            triples.append((offset + i, actions[i], offset + i + 1))
        frames_list.append(frames)
        offset += len(frames)
    if not triples:
        raise SystemExit(f"no transitions found in {data_dir}")
    frames = np.concatenate(frames_list)
    triples = np.array(triples, dtype=np.int64)
    return frames, triples[:, :2], triples[:, 2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", default="baseline/checkpoint.pt")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    device = args.device
    if device == "auto":
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    frames, obs_action, next_idx = load_transitions(Path(args.data))
    n = len(obs_action)
    print(f"{n} transitions, {len(frames)} frames, device={device}")

    model = WorldModel().to(device)
    print(f"parameters: {param_count(model):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    frames_t = torch.from_numpy(frames).to(device)  # uint8, (F, 64, 64, 3)

    def batch_images(idx):
        x = frames_t[idx].permute(0, 3, 1, 2).float() / 255.0
        return x

    t0 = time.time()
    for step in range(1, args.steps + 1):
        idx = rng.integers(0, n, size=args.batch)
        obs = batch_images(obs_action[idx, 0])
        nxt = batch_images(next_idx[idx])
        act = torch.from_numpy(obs_action[idx, 1]).long().to(device)
        out = model.loss(obs, act, nxt)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        if step % 500 == 0 or step == 1:
            print(f"step {step:6d}  loss {out['loss'].item():.4f}  "
                  f"pred {out['pred'].item():.4f}  var {out['var'].item():.4f}  "
                  f"cov {out['cov'].item():.4f}  latent_std {out['latent_std'].item():.3f}")

    elapsed = time.time() - t0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "seed": args.seed,
                "steps": args.steps, "transitions": n,
                "train_seconds": elapsed}, out_path)
    print(f"trained {args.steps} steps in {elapsed:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
