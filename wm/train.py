#!/usr/bin/env python3
"""Train the spatial world model on chain data.

Usage:
    python wm/generate.py --out data/wm8x1 --levels 1500 --seed 13
    python wm/train.py --data data/wm8x1 --out wm/checkpoint.pt --steps 6000 --seed 13

What to watch, and what each number means if it goes wrong:

    align     mean cosine similarity between a predicted next latent and the
              true next latent, over the rollout. Should rise toward 1.0; a
              value stuck near 0 means the dynamics is not tracking the move.
    nceacc    fraction of rollout steps whose predicted latent picks out the
              TRUE next state against the whole batch (chance = 1/batch).
              This is the number that says the model can tell actions apart,
              which is what planning needs. It replaces the delta-NCE that
              sat at chance because the encoder collapsed one-move deltas.
    policy    below ln(4) = 1.386, which is the score of guessing.
    dead      binary cross-entropy on "is this state lost". Chance is
              ~0.54 at the 23% dead rate the generator produces.
    spread    mean pairwise cosine between distinct states in the batch.
              Near 1.0 is representation collapse; should stay well below.

None of these is the goal. The goal is lab/attribute.py and lab/probe.py,
which measure the thing that is actually being claimed.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from wm.model import DEAD, WorldModel, param_count  # noqa: E402


def load(data_dir: Path):
    shards = sorted(data_dir.glob("*.npz"))
    if not shards:
        raise SystemExit(f"no shards in {data_dir}")
    frames, goals, cf, ca, cd, cm, cg = [], [], [], [], [], [], []
    f_off = g_off = 0
    for p in shards:
        d = np.load(p)
        frames.append(d["frames"])
        goals.append(d["goals"])
        cf.append(d["chain_frames"].astype(np.int64) + f_off)
        ca.append(d["chain_actions"].astype(np.int64))
        cd.append(d["chain_d"].astype(np.float32))
        cm.append(d["chain_mask"].astype(np.int64))
        cg.append(d["chain_goal"].astype(np.int64) + g_off)
        f_off += len(d["frames"])
        g_off += len(d["goals"])
    return (np.concatenate(frames), np.concatenate(goals),
            np.concatenate(cf), np.concatenate(ca), np.concatenate(cd),
            np.concatenate(cm), np.concatenate(cg))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="wm/checkpoint.pt")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--pred-coef", type=float, default=1.0)
    ap.add_argument("--dead-coef", type=float, default=1.0)
    ap.add_argument("--nce-coef", type=float, default=1.0,
                    help="action-discrimination contrastive weight")
    ap.add_argument("--rolled-coef", type=float, default=0.25,
                    help="head-loss weight on rolled (predicted) latents")
    ap.add_argument("--save-every", type=int, default=1000,
                    help="save a checkpoint every N steps (0 = end only)")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = ("mps" if torch.backends.mps.is_available()
                  else "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    frames, goals, cf, ca, cd, cm, cg = load(Path(args.data))
    n = len(cf)
    K = cf.shape[1] - 1
    print(f"{n} chains of {K} actions, {len(frames)} frames, "
          f"{len(goals)} goals, device={device}")

    model = WorldModel().to(device)
    print(f"parameters: {param_count(model):,}  (cap 20,000,000)")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           fused=(device == "cuda"))

    frames_t = torch.from_numpy(frames).to(device)
    goals_t = torch.from_numpy(goals).to(device)
    cf_t = torch.from_numpy(cf).to(device)
    ca_t = torch.from_numpy(ca).to(device)
    cd_t = torch.from_numpy(cd).to(device)
    cm_t = torch.from_numpy(cm).to(device)
    cg_t = torch.from_numpy(cg).to(device)
    bits = torch.arange(4, device=device)

    def imgs(bank, idx):
        """Gather uint8 HWC frames and return float CHW, keeping idx's shape."""
        x = bank[idx]                       # (..., 64, 64, 3)
        return x.movedim(-1, idx.dim()).float() / 255.0

    t0 = time.time()
    for step in range(1, args.steps + 1):
        sel = torch.from_numpy(rng.integers(0, n, size=args.batch)).to(device)
        fidx = cf_t[sel]                              # (B, K+1)
        chain = imgs(frames_t, fidx)                  # (B, K+1, 3, 64, 64)
        goal = imgs(goals_t, cg_t[sel])               # (B, 3, 64, 64)
        acts = ca_t[sel]                              # (B, K)
        d = cd_t[sel]                                 # (B, K+1)
        is_dead = d >= DEAD
        mask = ((cm_t[sel][..., None] >> bits) & 1).float()   # (B, K+1, 4)

        # Frames are deduped per level by state key, so a repeated index means
        # the action was a no-op. The contrastive term has to drop those rows:
        # its planted negative is "the state did not change", which is the
        # correct answer exactly there.
        same = fidx[:, 1:] == fidx[:, :-1]

        out = model.rollout_loss(chain, goal, acts, d, is_dead, mask, same,
                                 pred_coef=args.pred_coef,
                                 dead_coef=args.dead_coef,
                                 nce_coef=args.nce_coef,
                                 rolled_coef=args.rolled_coef)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        opt.step()

        if step % 250 == 0 or step == 1:
            print(f"step {step:6d}  loss {out['loss'].item():7.4f}  "
                  f"align {out['align'].item():.3f}  "
                  f"nceacc {out['nce_acc'].item():.3f}  "
                  f"value {out['value'].item():.4f}  "
                  f"policy {out['policy'].item():.4f}  "
                  f"pacc {out['policy_acc'].item():.3f}  "
                  f"dead {out['dead'].item():.4f}  "
                  f"spread {out['spread'].item():.3f}  "
                  f"std {out['latent_std'].item():.3f}", flush=True)

        if args.save_every and step % args.save_every == 0:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "seed": args.seed,
                        "steps": step, "chains": n, "chain_len": K,
                        "transitions": n * K,
                        "train_seconds": time.time() - t0}, args.out)
            print(f"saved step {step} -> {args.out}", flush=True)

    elapsed = time.time() - t0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def save(step):
        torch.save({"model": model.state_dict(), "seed": args.seed,
                    "steps": step, "chains": n, "chain_len": K,
                    "transitions": n * K,
                    "train_seconds": time.time() - t0}, out_path)
        print(f"saved step {step} -> {out_path}", flush=True)

    save(args.steps)
    print(f"trained {args.steps} steps in {elapsed:.0f}s -> {out_path}")


if __name__ == "__main__":
    main()
