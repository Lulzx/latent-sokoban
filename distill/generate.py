#!/usr/bin/env python3
"""Generate A*-labelled training data from rendered observations.

The symbolic solver is used here and only here -- RULES.md permits it for
training-data generation. Nothing it produces reaches inference: the model
is trained on rendered 64x64 observations and never sees a board.

Each sample is (obs, goal, action, next_obs, moves_to_go). moves_to_go is
the TRUE optimal distance from that state, recomputed with BFS, so the value
head learns the real heuristic rather than latent Euclidean distance -- which
is wrong for Sokoban, where a crate often has to move away from its goal
before it can come back.

Deadlocked states are labelled DEAD (a value well past any real distance).
That is deliberate: the baseline's 18-40% deadlock rate is the single
largest bucket of lost episodes, and a value head is the natural place to
learn that a push is unrecoverable.

Usage:
    python distill/generate.py --out data/distill8x1 --levels 2000 --seed 13
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latent_sokoban.env import ACTIONS, Level, SokobanEnv, SokobanState
from latent_sokoban.levels import generate_level
from latent_sokoban.render import render, render_goal
from latent_sokoban.solver import bfs_solve, optimal_length

DEAD = 99


def dist_to_go(level: Level, state: SokobanState) -> int:
    """True optimal moves remaining, or DEAD if the state cannot be solved."""
    probe = Level(level.walls, level.goals, state.boxes, state.player)
    d = optimal_length(probe)
    return DEAD if d is None else d


def successor(level: Level, state: SokobanState, action: int) -> SokobanState:
    env = SokobanEnv(level, max_steps=2)
    env.state = state
    env.step(action)
    return env.state


def dead_states(level: Level, rng: np.random.Generator, want: int):
    """States that are already lost, sampled deliberately rather than found.

    Random perturbation off the optimal path almost never deadlocks: measured
    on held-out levels, under 1% of collected states were dead, which is far
    too few for a head to learn "this is unrecoverable" from. So construct
    them directly -- crate on a non-goal corner, player anywhere legal -- and
    confirm each one with the solver rather than trusting the construction.

    Constructing training labels with the solver is what RULES.md permits it
    for. The agent learns deadlock from pixels; nothing hand-coded reaches
    inference. (RULES.md lists "deadlock prediction" under allowed
    modifications and "manually coded deadlock tables" under prohibitions --
    the line is where the table lives, and this one exists only in the
    labeller.)
    """
    h, w = level.height, level.width

    def wall(r: int, c: int) -> bool:
        return not (0 <= r < h and 0 <= c < w) or level.walls[r][c]

    corners = [(r, c) for r in range(h) for c in range(w)
               if not wall(r, c) and (r, c) not in level.goals
               and (wall(r - 1, c) or wall(r + 1, c))
               and (wall(r, c - 1) or wall(r, c + 1))]
    free = [(r, c) for r in range(h) for c in range(w) if not wall(r, c)]
    if not corners:
        return []

    out, seen = [], set()
    for _ in range(want * 6):
        if len(out) >= want:
            break
        boxes = frozenset([corners[rng.integers(0, len(corners))]])
        player = free[rng.integers(0, len(free))]
        if player in boxes or (boxes, player) in seen:
            continue
        seen.add((boxes, player))
        s = SokobanState(boxes, player)
        if dist_to_go(level, s) == DEAD:
            out.append(s)
    return out


def collect(level: Level, rng: np.random.Generator, off_path: int):
    """States on the optimal path, plus perturbations around it.

    Purely on-path data would only ever show the model states an optimal
    agent visits. The agent will not be optimal, so it needs value estimates
    where it actually ends up -- including the states it should avoid.
    """
    sol = bfs_solve(level)
    if not sol:
        return []
    env = SokobanEnv(level, max_steps=len(sol) + 5)
    env.reset()
    states = [env.state]
    for a in sol:
        env.step(a)
        states.append(env.state)

    seen = {(s.boxes, s.player) for s in states}
    extra = []
    for _ in range(off_path):
        base = states[rng.integers(0, len(states))]
        s = base
        for _ in range(rng.integers(1, 4)):
            s = successor(level, s, int(rng.integers(0, 4)))
        key = (s.boxes, s.player)
        if key not in seen:
            seen.add(key)
            extra.append(s)
    return states[:-1] + extra


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--levels", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--size", type=int, default=8)
    ap.add_argument("--boxes", type=int, default=1)
    ap.add_argument("--density", type=float, default=0.10)
    ap.add_argument("--min-len", type=int, default=4)
    ap.add_argument("--max-len", type=int, default=20)
    ap.add_argument("--off-path", type=int, default=6,
                    help="perturbed states sampled per level")
    ap.add_argument("--dead", type=int, default=4,
                    help="deliberately-constructed dead states per level")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    frames, goals, rows = [], [], []
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
        states = collect(level, rng, args.off_path)
        if not states:
            continue
        states = states + dead_states(level, rng, args.dead)
        gi = len(goals)
        goals.append(render_goal(level))
        for s in states:
            d_here = dist_to_go(level, s)
            # Dead states are KEPT, with an empty optimal-action mask. They
            # used to be skipped here, which meant the value head only ever
            # saw a DEAD label through the v_next term -- on a latent produced
            # by the dynamics model, never on one produced by the encoder.
            # Measured consequence (lab/probe.py): dead-state AUC was 0.918 on
            # dynamics latents and 0.427 on encoder latents, i.e. below chance,
            # so with accurate dynamics the planner walked into deadlocks 65%
            # of the time. A state has to be renderable as an observation for
            # "this is lost" to be learnable from pixels.
            oi = len(frames)
            frames.append(render(level, s))
            best = []
            nexts, d_nexts = [], []
            for a in ACTIONS:
                s2 = successor(level, s, a)
                d2 = dist_to_go(level, s2)
                ni = len(frames)
                frames.append(render(level, s2))
                nexts.append(ni)
                d_nexts.append(d2)
                if d_here != DEAD and d2 == d_here - 1:
                    best.append(a)
            mask = sum(1 << a for a in best)
            for a in ACTIONS:
                rows.append((oi, gi, a, nexts[a], d_here, d_nexts[a], mask))
        made += 1
        if made % 200 == 0:
            print(f"  {made}/{args.levels} levels, {len(rows)} samples "
                  f"({time.time()-t0:.0f}s)", file=sys.stderr)

    frames = np.stack(frames)
    goals = np.stack(goals)
    rows = np.array(rows, dtype=np.int32)
    np.savez_compressed(out / "shard_0000.npz", frames=frames, goals=goals,
                        rows=rows)
    dead_here = int((rows[:, 4] == DEAD).sum())
    dead_next = int((rows[:, 5] == DEAD).sum())
    print(f"dead rows: {dead_here} by d_here ({dead_here/len(rows):.1%}), "
          f"{dead_next} by d_next ({dead_next/len(rows):.1%})")
    meta = {"levels": made, "frames": len(frames), "samples": len(rows),
            "seed": args.seed, "size": args.size, "n_boxes": args.boxes,
            "dead_rows_here": dead_here, "dead_rows_next": dead_next,
            "dead_value": DEAD,
            "columns": "obs_idx,goal_idx,action,next_idx,d_here,d_next,opt_mask"}
    (out / "shard_0000.json").write_text(json.dumps(meta, indent=1))
    print(f"{made} levels, {len(frames)} frames, {len(rows)} samples "
          f"-> {out} ({time.time()-t0:.0f}s)")
    print(f"{len(rows)} kept samples count toward the Fixed Environment-Steps "
          f"budget; the solver's internal search does not (docs/RULES.md)")


if __name__ == "__main__":
    main()
