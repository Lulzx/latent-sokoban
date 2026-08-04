"""Convert the classic Sokoban level file into JSON for the /play page.

Input is the standard ASCII notation:

    #  wall          $  crate         .  goal
    @  player        *  crate on goal +  player on goal
    (space)          floor, or nothing at all

The ambiguity worth handling is that space means two different things: a
walkable interior tile, and the blank margin outside the map. Flood-filling
from the player through non-wall cells separates them, so the renderer can
draw floor only where floor actually exists and leave the rest transparent
instead of painting a ragged rectangle of tiles.

The bundled input is the 50-level Thinking Rabbit "Original" collection,
transcribed from the default level pack in davidjoffe/sokoban.

Usage:
    python scripts/build_play_levels.py \
        --in levels/original.txt --out server/static/play-levels.json
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path


def parse_file(text: str) -> list[list[str]]:
    """Split the file into per-level lists of raw ASCII rows."""
    levels, current = [], []
    for raw in text.splitlines():
        if raw.strip().lower().startswith("level"):
            if current:
                levels.append(current)
            current = []
            continue
        if raw.strip() == "":
            # A blank line inside a level's block is padding, not a
            # separator -- only flush when we already have rows and the
            # next non-blank starts a new "Level" header.
            continue
        current.append(raw.rstrip("\n"))
    if current:
        levels.append(current)
    return levels


def build(rows: list[str]) -> dict:
    h = len(rows)
    w = max(len(r) for r in rows)
    grid = [r.ljust(w) for r in rows]

    walls, goals, crates = set(), set(), set()
    player = None
    for r in range(h):
        for c in range(w):
            ch = grid[r][c]
            if ch == "#":
                walls.add((r, c))
            elif ch in ".+*":
                goals.add((r, c))
            if ch in "$*":
                crates.add((r, c))
            if ch in "@+":
                player = (r, c)
    if player is None:
        raise ValueError("level has no player")

    # Interior = everything reachable from the player without crossing a
    # wall. Crates do not block this flood; they sit on floor by definition.
    floor = set()
    q = deque([player])
    seen = {player}
    while q:
        r, c = q.popleft()
        floor.add((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in seen or (nr, nc) in walls:
                continue
            seen.add((nr, nc))
            q.append((nr, nc))

    # Keep only walls that actually touch the interior, so the decorative
    # outer shell of the ASCII art does not become a thick tile border.
    keep_walls = {
        (r, c) for (r, c) in walls
        if any((r + dr, c + dc) in floor
               for dr in (-1, 0, 1) for dc in (-1, 0, 1))
    }

    # A wall is on the outer boundary if it faces the void: it sits on the
    # grid edge, or one of its neighbours is neither floor nor wall. Walls
    # fully enclosed by the playfield are interior obstacles instead, and
    # the two get different brick.
    def faces_outside(r: int, c: int) -> bool:
        if r in (0, h - 1) or c in (0, w - 1):
            return True
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                n = (r + dr, c + dc)
                if n not in floor and n not in walls:
                    return True
        return False

    outer = {p for p in keep_walls if faces_outside(*p)}

    return {
        "w": w, "h": h,
        "player": list(player),
        "walls": sorted([list(p) for p in keep_walls - outer]),
        "outer_walls": sorted([list(p) for p in outer]),
        "floor": sorted([list(p) for p in floor]),
        "goals": sorted([list(p) for p in goals]),
        "crates": sorted([list(p) for p in crates]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    raw_levels = parse_file(Path(args.src).read_text())
    out, skipped = [], []
    for i, rows in enumerate(raw_levels, 1):
        try:
            lv = build(rows)
        except ValueError as e:
            skipped.append((i, str(e)))
            continue
        if len(lv["crates"]) != len(lv["goals"]):
            skipped.append((i, f"{len(lv['crates'])} crates vs {len(lv['goals'])} goals"))
            continue
        lv["source_n"] = i
        lv["n"] = i
        out.append(lv)

    Path(args.out).write_text(json.dumps({
        "collection": "Original",
        "author": "Thinking Rabbit",
        "source": "https://github.com/davidjoffe/sokoban/blob/master/data/sokoban/levels/default.txt",
        "levels": out,
    }))
    sizes = [(lv["w"], lv["h"]) for lv in out]
    print(f"wrote {args.out}: {len(out)} levels, "
          f"widest {max(s[0] for s in sizes)}, tallest {max(s[1] for s in sizes)}, "
          f"crates {min(len(l['crates']) for l in out)}-{max(len(l['crates']) for l in out)}")
    for i, why in skipped:
        print(f"  skipped level {i}: {why}")


if __name__ == "__main__":
    main()
