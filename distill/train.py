#!/usr/bin/env python3
"""Train the search-distilled model.

Usage:
    python distill/train.py --data data/distill8x1 --out distill/checkpoint.pt \
        --steps 5000 --seed 13

Watch value/policy in the logs: policy should fall well below ln(4)=1.386
(the score of guessing uniformly among four actions), and latent_std should
stay near 1 rather than collapsing.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distill.model import DistillModel, param_count


def load(data_dir: Path):
    shards = sorted(data_dir.glob("*.npz"))
    if not shards:
        raise SystemExit(f"no shards in {data_dir}")
    frames, goals, rows = [], [], []
    f_off = g_off = 0
    for p in shards:
        d = np.load(p)
        r = d["rows"].astype(np.int64).copy()
        r[:, 0] += f_off
        r[:, 3] += f_off
        r[:, 1] += g_off
        frames.append(d["frames"])
        goals.append(d["goals"])
        rows.append(r)
        f_off += len(d["frames"])
        g_off += len(d["goals"])
    return np.concatenate(frames), np.concatenate(goals), np.concatenate(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="distill/checkpoint.pt")
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    frames, goals, rows = load(Path(args.data))
    n = len(rows)
    print(f"{n} samples, {len(frames)} frames, {len(goals)} goals, device={device}")

    model = DistillModel().to(device)
    print(f"parameters: {param_count(model):,}")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           fused=(device == "cuda"))

    # Whole banks resident on device; see baseline/train.py for why the batch
    # indices are drawn up front rather than once per step.
    frames_t = torch.from_numpy(frames).to(device)
    goals_t = torch.from_numpy(goals).to(device)
    rows_t = torch.from_numpy(rows).to(device)
    all_idx = torch.from_numpy(np.stack(
        [rng.integers(0, n, size=args.batch) for _ in range(args.steps)])
    ).to(device)

    def imgs(bank, idx):
        return bank[idx].permute(0, 3, 1, 2).float() / 255.0

    t0 = time.time()
    for step in range(1, args.steps + 1):
        r = rows_t[all_idx[step - 1]]
        obs = imgs(frames_t, r[:, 0])
        goal = imgs(goals_t, r[:, 1])
        nxt = imgs(frames_t, r[:, 3])
        act = r[:, 2]
        d_here = r[:, 4].float()
        d_next = r[:, 5].float()
        mask = ((r[:, 6:7] >> torch.arange(4, device=device)) & 1).float()

        out = model.loss(obs, goal, act, nxt, d_here, d_next, mask)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        opt.step()

        if step % 500 == 0 or step == 1:
            print(f"step {step:6d}  loss {out['loss'].item():.4f}  "
                  f"pred {out['pred'].item():.4f}  value {out['value'].item():.4f}  "
                  f"policy {out['policy'].item():.4f}  "
                  f"latent_std {out['latent_std'].item():.3f}")

    elapsed = time.time() - t0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "seed": args.seed,
                "steps": args.steps, "samples": n,
                "train_seconds": elapsed}, out_path)
    print(f"trained {args.steps} steps in {elapsed:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
