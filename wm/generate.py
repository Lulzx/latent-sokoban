#!/usr/bin/env python3
"""Generate chain-structured training data for the spatial world model.

The symbolic solver is used here and only here -- RULES.md permits it for
training-data generation. Nothing it produces reaches inference.

Why chains rather than the single transitions in distill/generate.py: the
measured failure of the previous model is open-loop drift (correct-state
retrieval 0.717 after one imagined step, 0.49 by three), and a one-step MSE
does not optimise for that. A K-step chain lets the loss ask the question
planning actually asks -- where do I end up after K imagined actions.

Each chain is K+1 frames with the K actions between them, and every frame
carries its own labels: true moves-to-go (DEAD where lost), the set of
optimal actions, and a dead flag.

Chain start states are drawn from three places, in deliberate proportion:
on the optimal path (where a good agent lives), perturbed off it (where a
real agent actually lives), and constructed dead states (which random play
almost never reaches -- under 1% of states, measured, which is far too few
to learn deadlock from).

Usage:
    python wm/generate.py --out data/wm8x1 --levels 1500 --seed 13
    python wm/generate.py --out data/wm8x3 --levels 1500 --boxes 3 --seed 13
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distill.generate import DEAD, dead_states, dist_to_go, successor  # noqa: E402
from latent_sokoban.env import ACTIONS, Level, SokobanEnv, SokobanState  # noqa: E402
from latent_sokoban.levels import generate_level                # noqa: E402
from latent_sokoban.render import render, render_goal           # noqa: E402
from latent_sokoban.solver import bfs_solve                     # noqa: E402

try:
    from latent_sokoban import clabel                          # noqa: E402
except OSError:
    clabel = None


def label(level: Level, s: SokobanState, cache: dict, dist):
    """(moves_to_go, optimal-action bitmask). Cached: distance lookups are O(1)
    against a per-level reverse-BFS distance table (latent_sokoban.clabel);
    the Python path runs a BFS per state. `dist` is a clabel.DistTable, or None
    to fall back to the Python solver."""
    key = (s.boxes, s.player)
    if key in cache:
        return cache[key]
    if dist is not None:
        d = dist.dist(s)
        d = DEAD if d < 0 else d
        mask = 0
        if d != DEAD and d > 0:
            for a in ACTIONS:
                if dist.dist(successor(level, s, a)) == d - 1:
                    mask |= 1 << a
    else:
        d = dist_to_go(level, s)
        mask = 0
        if d != DEAD and d > 0:
            for a in ACTIONS:
                if dist_to_go(level, successor(level, s, a)) == d - 1:
                    mask |= 1 << a
    cache[key] = (d, mask)
    return cache[key]


def start_states(level: Level, rng: np.random.Generator, n_off: int, n_dead: int):
    sol = bfs_solve(level)
    if not sol:
        return []
    env = SokobanEnv(level, max_steps=len(sol) + 8)
    env.reset()
    on_path = [env.state]
    for a in sol:
        env.step(a)
        on_path.append(env.state)

    off = []
    for _ in range(n_off):
        s = on_path[rng.integers(0, len(on_path))]
        for _ in range(rng.integers(1, 5)):
            s = successor(level, s, int(rng.integers(0, 4)))
        off.append(s)

    return on_path[:-1] + off + dead_states(level, rng, n_dead)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--boxes", type=int, default=1)
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=20)
    ap.add_argument("--chain", type=int, default=5, help="K, actions per chain")
    ap.add_argument("--off-path", type=int, default=6)
    ap.add_argument("--dead", type=int, default=4)
    ap.add_argument("--per-state", type=int, default=1,
                    help="chains sampled from each start state")
    ap.add_argument("--shard", type=int, default=0,
                    help="shard index for the output filename, so several "
                         "crate tiers can share one --out dir")
    args = ap.parse_args()

    K = args.chain
    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frames: list[np.ndarray] = []
    goals: list[np.ndarray] = []
    chain_frames: list[list[int]] = []
    chain_actions: list[list[int]] = []
    chain_d: list[list[int]] = []
    chain_mask: list[list[int]] = []
    chain_goal: list[int] = []

    t0 = time.time()
    made = 0
    while made < args.levels:
        try:
            level, _ = generate_level(
                rng, size=args.size, n_boxes=args.boxes,
                wall_density=args.density, min_solution_len=args.min_len,
                max_solution_len=args.max_len, max_tries=20000)
        except RuntimeError:
            continue
        starts = start_states(level, rng, args.off_path, args.dead)
        if not starts:
            continue

        gi = len(goals)
        goals.append(render_goal(level))
        lab_cache: dict = {}
        frame_cache: dict = {}       # dedupe frames within a level
        # Reverse-BFS labelling is O(non-dead states); that set is ~250k at
        # three crates but explodes at four, where the per-state Python BFS
        # (bounded by the near-optimal states we actually query) is faster.
        dist = (clabel.DistTable(level)
                if clabel is not None and args.boxes <= 3 else None)

        def frame_of(s: SokobanState) -> int:
            key = (s.boxes, s.player)
            if key not in frame_cache:
                frame_cache[key] = len(frames)
                frames.append(render(level, s))
            return frame_cache[key]

        for s0 in starts:
            for _ in range(args.per_state):
                s = s0
                fidx, acts, ds, masks = [frame_of(s)], [], [], []
                d, m = label(level, s, lab_cache, dist)
                ds.append(d)
                masks.append(m)
                for _ in range(K):
                    # Follow an optimal action where one exists, otherwise act
                    # randomly. Purely random chains almost never contain a
                    # push, so the model would spend its capacity on walking.
                    if m:
                        opts = [a for a in ACTIONS if m >> a & 1]
                        a = int(opts[rng.integers(0, len(opts))])
                        if rng.random() < 0.3:
                            a = int(rng.integers(0, 4))
                    else:
                        a = int(rng.integers(0, 4))
                    s = successor(level, s, a)
                    d, m = label(level, s, lab_cache, dist)
                    acts.append(a)
                    fidx.append(frame_of(s))
                    ds.append(d)
                    masks.append(m)
                chain_frames.append(fidx)
                chain_actions.append(acts)
                chain_d.append(ds)
                chain_mask.append(masks)
                chain_goal.append(gi)

        if dist is not None:
            dist.close()
        made += 1
        if made % 200 == 0:
            print(f"  {made}/{args.levels} levels, {len(chain_frames)} chains, "
                  f"{len(frames)} frames ({time.time()-t0:.0f}s)", file=sys.stderr)

    frames_a = np.stack(frames)
    goals_a = np.stack(goals)
    cf = np.array(chain_frames, dtype=np.int32)
    ca = np.array(chain_actions, dtype=np.int8)
    cd = np.array(chain_d, dtype=np.int16)
    cm = np.array(chain_mask, dtype=np.int8)
    cg = np.array(chain_goal, dtype=np.int32)

    np.savez_compressed(out / f"shard_{args.shard:04d}.npz",
                        frames=frames_a, goals=goals_a,
                        chain_frames=cf, chain_actions=ca, chain_d=cd,
                        chain_mask=cm, chain_goal=cg)

    transitions = int(cf.shape[0] * K)
    dead_frac = float((cd == DEAD).mean())
    meta = {"levels": made, "frames": len(frames_a), "goals": len(goals_a),
            "chains": int(cf.shape[0]), "chain_len": K,
            "transitions": transitions, "dead_fraction": round(dead_frac, 4),
            "seed": args.seed, "size": args.size, "n_boxes": args.boxes,
            "dead_value": DEAD}
    (out / f"shard_{args.shard:04d}.json").write_text(json.dumps(meta, indent=1))
    print(f"{made} levels, {len(frames_a)} frames, {cf.shape[0]} chains "
          f"-> {out} ({time.time()-t0:.0f}s)")
    print(f"dead labels: {dead_frac:.1%} of chain steps")
    print(f"{transitions} transitions count toward the Fixed Environment-Steps "
          f"budget of 2,000,000; the solver's internal search does not "
          f"(docs/RULES.md)")


if __name__ == "__main__":
    main()
