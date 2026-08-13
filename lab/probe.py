#!/usr/bin/env python3
"""Probe the distilled model's components on held-out levels.

ANALYSIS ONLY (symbolic solver used for labels, as RULES.md permits).

Generates fresh levels from a seed the model never trained on and asks four
separate questions, each answerable without running an episode:

  P1 value monotonicity   does predicted value track true distance-to-go?
  P2 dead separation      does the head call dead states dead -- and does it
                          matter whether the latent came from the ENCODER or
                          from the DYNAMICS model? (the loss only ever labels
                          dynamics latents as dead; see distill/model.py)
  P3 greedy accuracy      with perfect dynamics, does argmin-value pick a
                          truly optimal action? vs the policy head's argmax
  P4 dynamics drift       after k imagined steps, is the predicted latent
                          still nearest to the true state's latent?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from distill.model import DistillModel                        # noqa: E402
from latent_sokoban.env import Level, SokobanEnv, SokobanState  # noqa: E402
from latent_sokoban.levels import generate_level               # noqa: E402
from latent_sokoban.render import render, render_goal          # noqa: E402
from latent_sokoban.solver import bfs_solve                    # noqa: E402
from wm.model import WorldModel                                # noqa: E402

N_ACTIONS = 4
ARCHS = {"distill": DistillModel, "wm": WorldModel}


def dist_to_go(level, key):
    probe = Level(level.walls, level.goals, frozenset(key[1]), key[0])
    sol = bfs_solve(probe)
    return None if sol is None else len(sol)


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """P(a random pos scores higher than a random neg). Ties count half."""
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(len(order), float)
    ranks[order] = np.arange(1, len(order) + 1)
    r = ranks[: len(pos)].sum()
    return (r - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=str(REPO / "distill/checkpoint.pt"))
    ap.add_argument("--arch", choices=sorted(ARCHS), default="distill")
    ap.add_argument("--levels", type=int, default=150)
    ap.add_argument("--seed", type=int, default=9_999)
    ap.add_argument("--boxes", type=int, default=1)
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--density", type=float, default=0.10)
    args = ap.parse_args()

    device = ("mps" if torch.backends.mps.is_available()
              else "cuda" if torch.cuda.is_available() else "cpu")
    model = ARCHS[args.arch]().to(device).eval()
    model.load_state_dict(torch.load(args.ckpt, map_location="cpu",
                                     weights_only=True)["model"])

    @torch.no_grad()
    def enc(imgs):
        x = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().div_(255.)
        return model.encoder(x.to(device))

    @torch.no_grad()
    def heads(z, zg):
        """(value, logits, dead_logit); dead_logit is None for the distill arch."""
        g = zg.expand(z.shape[0], *([-1] * (zg.dim() - 1)))
        out = model.heads(z, g)
        return out if len(out) == 3 else (out[0], out[1], None)

    def rep(z, n):
        return z.expand(n, *([-1] * (z.dim() - 1)))

    @torch.no_grad()
    def dyn(z, a):
        return model.dynamics(z, torch.as_tensor(a, device=device))

    rng = np.random.default_rng(args.seed)

    v_enc, d_true = [], []           # P1
    dead_enc, alive_enc = [], []     # P2 (encoder latents, value as the score)
    dead_dyn, alive_dyn = [], []     # P2 (dynamics latents, value as the score)
    h_dead_enc, h_alive_enc = [], []  # P2b (dedicated dead head, if present)
    h_dead_dyn, h_alive_dyn = [], []
    val_hit = pol_hit = n_dec = 0    # P3
    drift_hit = np.zeros(6)          # P4
    drift_n = 0

    made = 0
    while made < args.levels:
        try:
            level, _ = generate_level(rng, size=args.size, n_boxes=args.boxes,
                                      wall_density=args.density,
                                      min_solution_len=4, max_solution_len=20,
                                      max_tries=20000)
        except RuntimeError:
            continue
        sol = bfs_solve(level)
        if not sol:
            continue
        made += 1
        theme = None
        zg = enc([render_goal(level)])

        # walk the optimal path plus short random perturbations off it
        env = SokobanEnv(level, max_steps=len(sol) + 8)
        env.reset()
        keys = [env.state.key()]
        for a in sol:
            env.step(a)
            keys.append(env.state.key())
        for _ in range(4):
            k = keys[rng.integers(0, len(keys))]
            for _ in range(rng.integers(1, 4)):
                k = SokobanEnv.apply(level, k, int(rng.integers(0, 4)))
            keys.append(k)

        for k in keys:
            d = dist_to_go(level, k)
            img = render(level, SokobanState(frozenset(k[1]), k[0]))
            z = enc([img])
            v, logits, dl = heads(z, zg)
            v = float(v[0])
            dl = None if dl is None else float(dl[0])

            if d is None:
                dead_enc.append(v)
                if dl is not None:
                    h_dead_enc.append(dl)
                continue
            alive_enc.append(v)
            if dl is not None:
                h_alive_enc.append(dl)
            v_enc.append(v)
            d_true.append(d)
            if d == 0:
                continue

            # P3: rank the four true successors, with the value head reading
            # encoder latents of the true next observations (perfect dynamics)
            succ_keys = [SokobanEnv.apply(level, k, a) for a in range(N_ACTIONS)]
            succ_d = [dist_to_go(level, s) for s in succ_keys]
            zs = enc([render(level, SokobanState(frozenset(s[1]), s[0]))
                      for s in succ_keys])
            vs, _, _ = heads(zs, zg)
            vs = vs.cpu().numpy()
            best = {a for a, sd in enumerate(succ_d) if sd is not None and sd == d - 1}
            if best:
                n_dec += 1
                val_hit += int(np.argmin(vs) in best)
                pol_hit += int(int(np.argmax(logits[0].cpu().numpy())) in best)

            # P2 (dynamics latents): same states, latent produced by dynamics
            zd = dyn(rep(z, N_ACTIONS), np.arange(N_ACTIONS))
            vd, _, dld = heads(zd, zg)
            vd = vd.cpu().numpy()
            dld = None if dld is None else dld.cpu().numpy()
            for a, sd in enumerate(succ_d):
                (dead_dyn if sd is None else alive_dyn).append(float(vd[a]))
                if dld is not None:
                    (h_dead_dyn if sd is None else h_alive_dyn).append(float(dld[a]))

        # P4: k-step open-loop drift, measured by retrieval not by MSE
        k0 = keys[0]
        z = enc([render(level, SokobanState(frozenset(k0[1]), k0[0]))])
        true_k = k0
        bank_keys, bank_imgs = [], []
        seen = set()
        probe = k0
        for _ in range(30):
            probe = SokobanEnv.apply(level, probe, int(rng.integers(0, 4)))
            if probe not in seen:
                seen.add(probe)
                bank_keys.append(probe)
                bank_imgs.append(render(level, SokobanState(frozenset(probe[1]),
                                                            probe[0])))
        acts = [int(rng.integers(0, 4)) for _ in range(6)]
        for step, a in enumerate(acts):
            z = dyn(z, [a])
            true_k = SokobanEnv.apply(level, true_k, a)
            if true_k not in seen:
                seen.add(true_k)
                bank_keys.append(true_k)
                bank_imgs.append(render(level, SokobanState(frozenset(true_k[1]),
                                                            true_k[0])))
            zb = enc(bank_imgs)
            # The wm arch's latent is a spatial grid (N, C, H, W); cdist needs
            # flat vectors. Flattening is the identity for the distill vector.
            nn = int(torch.cdist(z.flatten(1), zb.flatten(1)).argmin())
            drift_hit[step] += int(bank_keys[nn] == true_k)
        drift_n += 1

    v_enc = np.array(v_enc)
    d_true = np.array(d_true, float)
    print(f"\nheld-out levels: {made}  (seed {args.seed}, {args.boxes} crate, "
          f"{args.size}x{args.size})\n")

    print("P1 value monotonicity (encoder latents, alive states)")
    print(f"  spearman(pred, true d)   {_spearman(v_enc, d_true):+.3f}"
          f"   pearson {np.corrcoef(v_enc, d_true)[0,1]:+.3f}   n={len(v_enc)}")
    for lo, hi in [(0, 3), (4, 7), (8, 11), (12, 15), (16, 30)]:
        m = (d_true >= lo) & (d_true <= hi)
        if m.sum():
            print(f"    true d {lo:>2}-{hi:<2}  pred mean {v_enc[m].mean():6.3f} "
                  f"(log1p target {np.log1p(d_true[m]).mean():5.3f})  n={m.sum()}")

    print("\nP2 dead separation  (AUC: 1.0 = dead always scored worse)")
    print(f"  encoder latents    AUC {auc(np.array(dead_enc), np.array(alive_enc)):.3f}"
          f"   dead n={len(dead_enc)}  alive n={len(alive_enc)}")
    print(f"  dynamics latents   AUC {auc(np.array(dead_dyn), np.array(alive_dyn)):.3f}"
          f"   dead n={len(dead_dyn)}  alive n={len(alive_dyn)}")
    if dead_enc:
        print(f"  mean pred: dead {np.mean(dead_enc):.3f} vs alive "
              f"{np.mean(alive_enc):.3f}   (log1p(99) target = 4.605)")
    if h_dead_enc or h_dead_dyn:
        print("\nP2b dedicated dead head (AUC: 1.0 = dead always flagged)")
        print(f"  encoder latents    AUC "
              f"{auc(np.array(h_dead_enc), np.array(h_alive_enc)):.3f}"
              f"   dead n={len(h_dead_enc)}  alive n={len(h_alive_enc)}")
        print(f"  dynamics latents   AUC "
              f"{auc(np.array(h_dead_dyn), np.array(h_alive_dyn)):.3f}"
              f"   dead n={len(h_dead_dyn)}  alive n={len(h_alive_dyn)}")

    print("\nP3 one-step action choice, PERFECT dynamics (chance = "
          f"{_chance():.2f}, n={n_dec})")
    print(f"  value head argmin  {val_hit / max(n_dec,1):.3f}")
    print(f"  policy head argmax {pol_hit / max(n_dec,1):.3f}")

    print("\nP4 open-loop drift: predicted latent's nearest true state is correct")
    for i, h in enumerate(drift_hit):
        print(f"  after {i+1} imagined steps  {h/max(drift_n,1):.3f}")
    print()


def _chance() -> float:
    return 0.25


def _spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


if __name__ == "__main__":
    main()
