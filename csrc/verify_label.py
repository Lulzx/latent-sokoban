#!/usr/bin/env python3
"""Verify csrc/liblabelsokoban.dylib (reverse-BFS distance table) agrees with
the Python solver on distance-to-goal and optimal-action masks."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distill.generate import DEAD, dist_to_go, successor
from latent_sokoban.clabel import DistTable
from latent_sokoban.env import ACTIONS, SokobanEnv
from latent_sokoban.levels import generate_level
from latent_sokoban.solver import bfs_solve


def py_label(level, s):
    d = dist_to_go(level, s)
    mask = 0
    if d != DEAD and d > 0:
        for a in ACTIONS:
            if dist_to_go(level, successor(level, s, a)) == d - 1:
                mask |= 1 << a
    return d, mask


def c_label(level, dist, s):
    d = dist.dist(s)
    d = DEAD if d < 0 else d
    mask = 0
    if d != DEAD and d > 0:
        for a in ACTIONS:
            if dist.dist(successor(level, s, a)) == d - 1:
                mask |= 1 << a
    return d, mask


def main():
    mismatches = 0
    checked = 0
    for boxes in (1, 2, 3, 4):
        rng = np.random.default_rng(1000 + boxes)
        for _ in range(6):
            try:
                level, _ = generate_level(rng, size=8, n_boxes=boxes,
                                          wall_density=0.10 + 0.02 * boxes,
                                          min_solution_len=4, max_solution_len=20,
                                          max_tries=20000)
            except RuntimeError:
                continue
            sol = bfs_solve(level)
            if not sol:
                continue
            env = SokobanEnv(level, max_steps=len(sol) + 8)
            env.reset()
            states = [env.state]
            for a in sol:
                env.step(a)
                states.append(env.state)
            for _ in range(4):
                s = states[rng.integers(0, len(states))]
                for _ in range(rng.integers(1, 4)):
                    s = successor(level, s, int(rng.integers(0, 4)))
                states.append(s)
            dist = DistTable(level)
            for s in states:
                pd, pm = py_label(level, s)
                cd, cm = c_label(level, dist, s)
                checked += 1
                if pd != cd or pm != cm:
                    mismatches += 1
                    if mismatches <= 5:
                        print(f"MISMATCH boxes={boxes} d py={pd} c={cd} "
                              f"mask py={pm:#x} c={cm:#x} at {s.player} {s.boxes}")
            dist.close()
    print(f"checked {checked} states, {mismatches} mismatches")
    return 1 if mismatches else 0


if __name__ == "__main__":
    sys.exit(main())
