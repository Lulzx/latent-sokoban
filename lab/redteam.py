#!/usr/bin/env python3
"""Red-team probe: recover the grid from the deterministic renderer.

ANALYSIS ONLY. This is the exploit docs/RULES.md forbids ("reconstruct a
discrete board representation at inference time... decode the image to a
board and search it"). It exists to prove the threat is real and to give
the server-side hardening work a test it must defeat, not as an entry.

Why it works: render.py draws the DEFAULT (noise-free) theme at exactly
64 // board_size pixels per tile -- 8 for the hidden set's 8x8 board. Every
tile type owns a unique interior colour, and the goal marker is a fixed
ring, so reading two pixels per tile (centre + ring) reconstructs the full
Level. From there the shipped BFS solver (latent_sokoban.solver) yields an
optimal path, and the agent replays it. Expected score: ~100% on any
default-theme split, at zero dynamics calls.

Usage (demonstrates the exploit on the local 8x8 splits):
    python scripts/evaluate.py --agent lab.redteam:RedTeamAgent \
        --splits levels/split_s.json levels/split_a.json \
                 levels/split_d.json --seeds 0

Hardening test: a server-side mitigation (per-episode noise, colour
jitter, anti-memorization) is working only when this agent's success rate
on the affected set collapses toward random.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latent_sokoban.agent import Agent          # noqa: E402
from latent_sokoban.env import Level            # noqa: E402
from latent_sokoban.solver import bfs_solve     # noqa: E402

# default_theme() colours, in the order render.py writes them. These are the
# only colours a noise-free observation can contain.
FLOOR = (222, 214, 186)
FLOOR_ALT = (212, 204, 176)
WALL = (94, 84, 74)
WALL_EDGE = (64, 56, 48)
GOAL = (196, 60, 60)
BOX = (176, 122, 54)
BOX_EDGE = (120, 80, 30)
BOX_ON_GOAL = (206, 160, 70)
PLAYER = (48, 108, 188)


def _nearest(px: np.ndarray, palette: dict[str, tuple]) -> str:
    """Classify one RGB pixel against the theme palette (closest colour)."""
    best, best_d = None, float("inf")
    for name, rgb in palette.items():
        d = int(((px.astype(int) - rgb) ** 2).sum())
        if d < best_d:
            best, best_d = name, d
    return best


def decode(img: np.ndarray, size: int = 8) -> Level:
    """Recover a Level from a rendered (size, size, 3) uint8 observation.

    Assumes the default theme and 64 // size pixels per tile (exact for the
    hidden set and every local split except the themed Split B). The goal
    image is not needed: goals are visible in the board render (as the goal
    ring or the box_on_goal sprite).
    """
    cell = 64 // size
    oy = (64 - size * cell) // 2
    ox = oy
    palette = {"wall": WALL, "box": BOX, "box_on_goal": BOX_ON_GOAL,
               "player": PLAYER, "floor": FLOOR, "floor_alt": FLOOR_ALT,
               "goal": GOAL, "edge": WALL_EDGE}
    ring = max(1, cell // 4)          # goal ring starts at render.py's `m`

    walls, goals, boxes = [], set(), set()
    player = None
    for r in range(size):
        row = []
        for c in range(size):
            y0, x0 = oy + r * cell, ox + c * cell
            centre = _nearest(img[y0 + cell // 2, x0 + cell // 2], palette)
            ringpx = _nearest(img[y0 + ring, x0 + ring], palette)
            is_goal = ringpx == "goal" or centre == "box_on_goal"
            if centre in ("wall", "edge"):
                row.append(True)
                continue
            row.append(False)
            if is_goal:
                goals.add((r, c))
            if centre in ("box", "box_on_goal"):
                boxes.add((r, c))
            if centre == "player":
                player = (r, c)
        walls.append(tuple(row))
    if player is None:
        raise ValueError("decode found no player; not a default-theme render?")
    return Level(tuple(walls), frozenset(goals), frozenset(boxes), player)


class RedTeamAgent(Agent):
    """Decodes the board once per episode, BFS-solves it, and replays.

    This is the exact "decode-then-search" the rules prohibit. It makes zero
    dynamics calls, so it never trips the call meter -- the only thing that
    would flag it is source review.
    """

    def __init__(self, size: int = 8):
        super().__init__()
        self.size = size
        self._plan: list[int] = []

    def reset(self) -> None:
        self._plan = []

    def act(self, obs, goal, action_history):
        if not self._plan:
            level = decode(obs, self.size)
            self._plan = bfs_solve(level) or []
        if not self._plan or len(action_history) >= len(self._plan):
            return 0
        return self._plan[len(action_history)]


if __name__ == "__main__":
    # Smoke test: generate one level, render it, decode it, and confirm the
    # recovered Level round-trips to the same ASCII.
    import numpy as np
    from latent_sokoban.levels import generate_level
    from latent_sokoban.render import render

    rng = np.random.default_rng(0)
    level, _ = generate_level(rng, size=8, n_boxes=3, wall_density=0.14,
                              min_solution_len=4, max_solution_len=20)
    img = render(level)
    back = decode(img, 8)
    print("round-trip matches:", back.to_ascii() == level.to_ascii())
    if back.to_ascii() != level.to_ascii():
        print("--- original ---\n", level.to_ascii())
        print("--- decoded  ---\n", back.to_ascii())
