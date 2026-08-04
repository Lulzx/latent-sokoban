"""Deterministic Sokoban environment operating on symbolic state.

The symbolic state exists only inside the environment, the solver and the
evaluation harness. Agents never see it: they receive rendered images only
(see the "Allowed Inputs" section of the competition rules).

Actions: 0 = up, 1 = down, 2 = left, 3 = right.
Invalid actions (walking into a wall, pushing a blocked box) are no-ops
that still consume a step.

ASCII level format (standard Sokoban notation):
    #  wall          .  goal
    (space) floor    $  box
    @  player        *  box on goal
                     +  player on goal
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ACTIONS: dict[int, tuple[int, int]] = {
    0: (-1, 0),  # up
    1: (1, 0),   # down
    2: (0, -1),  # left
    3: (0, 1),   # right
}
ACTION_NAMES = ["up", "down", "left", "right"]


@dataclass(frozen=True)
class Level:
    """Immutable level definition."""

    walls: tuple[tuple[bool, ...], ...]  # walls[r][c] is True where a wall is
    goals: frozenset[tuple[int, int]]
    boxes: frozenset[tuple[int, int]]
    player: tuple[int, int]

    @property
    def height(self) -> int:
        return len(self.walls)

    @property
    def width(self) -> int:
        return len(self.walls[0])

    def to_ascii(self) -> str:
        rows = []
        for r in range(self.height):
            row = []
            for c in range(self.width):
                pos = (r, c)
                if self.walls[r][c]:
                    ch = "#"
                elif pos in self.boxes:
                    ch = "*" if pos in self.goals else "$"
                elif pos == self.player:
                    ch = "+" if pos in self.goals else "@"
                elif pos in self.goals:
                    ch = "."
                else:
                    ch = " "
                row.append(ch)
            rows.append("".join(row))
        return "\n".join(rows)

    @staticmethod
    def from_ascii(text: str) -> "Level":
        lines = [line for line in text.split("\n") if line]
        width = max(len(line) for line in lines)
        walls, goals, boxes = [], set(), set()
        player = None
        for r, line in enumerate(lines):
            line = line.ljust(width)
            wall_row = []
            for c, ch in enumerate(line):
                wall_row.append(ch == "#")
                if ch in ".*+":
                    goals.add((r, c))
                if ch in "$*":
                    boxes.add((r, c))
                if ch in "@+":
                    player = (r, c)
            walls.append(tuple(wall_row))
        if player is None:
            raise ValueError("level has no player")
        return Level(tuple(walls), frozenset(goals), frozenset(boxes), player)


@dataclass
class SokobanState:
    """Mutable dynamic state on top of a fixed Level."""

    boxes: frozenset[tuple[int, int]]
    player: tuple[int, int]
    steps: int = 0

    def key(self) -> tuple:
        """Hashable identity used by the solver and cycle detection."""
        return (self.player, self.boxes)


@dataclass
class StepInfo:
    moved: bool = False       # player position changed
    pushed: bool = False      # a box was pushed
    invalid: bool = False     # action was a no-op
    solved: bool = False
    truncated: bool = False   # hit the step limit
    extras: dict = field(default_factory=dict)


class SokobanEnv:
    """Deterministic Sokoban with a fixed step limit."""

    def __init__(self, level: Level, max_steps: int = 40):
        self.level = level
        self.max_steps = max_steps
        self.state = SokobanState(level.boxes, level.player)

    def reset(self) -> SokobanState:
        self.state = SokobanState(self.level.boxes, self.level.player)
        return self.state

    @property
    def solved(self) -> bool:
        return self.state.boxes == self.level.goals

    def goal_state(self) -> SokobanState:
        """The target configuration: all boxes on goals. The player position
        in the goal image is undefined; by convention we omit the player
        when rendering goal observations (render(..., show_player=False))."""
        return SokobanState(frozenset(self.level.goals), self.level.player)

    def step(self, action: int) -> tuple[SokobanState, bool, StepInfo]:
        """Apply one action. Returns (state, done, info).

        done is True when the puzzle is solved or the step limit is reached.
        Invalid actions are no-op transitions that still consume a step.
        """
        if action not in ACTIONS:
            raise ValueError(f"invalid action id {action}")
        info = StepInfo()
        dr, dc = ACTIONS[action]
        r, c = self.state.player
        nr, nc = r + dr, c + dc
        boxes = self.state.boxes

        if self._is_wall(nr, nc):
            info.invalid = True
        elif (nr, nc) in boxes:
            br, bc = nr + dr, nc + dc
            if self._is_wall(br, bc) or (br, bc) in boxes:
                info.invalid = True
            else:
                boxes = (boxes - {(nr, nc)}) | {(br, bc)}
                self.state = SokobanState(boxes, (nr, nc), self.state.steps + 1)
                info.moved = True
                info.pushed = True
        else:
            self.state = SokobanState(boxes, (nr, nc), self.state.steps + 1)
            info.moved = True

        if info.invalid:
            self.state = SokobanState(boxes, (r, c), self.state.steps + 1)

        info.solved = self.solved
        info.truncated = self.state.steps >= self.max_steps and not info.solved
        done = info.solved or info.truncated
        return self.state, done, info

    def _is_wall(self, r: int, c: int) -> bool:
        if not (0 <= r < self.level.height and 0 <= c < self.level.width):
            return True
        return self.level.walls[r][c]

    # -- static dynamics (used by the solver, no env instance mutation) -----

    @staticmethod
    def apply(level: Level, key: tuple, action: int) -> tuple:
        """Pure transition on a state key ((player, boxes)) for search."""
        player, boxes = key
        dr, dc = ACTIONS[action]
        r, c = player
        nr, nc = r + dr, c + dc

        def wall(rr, cc):
            if not (0 <= rr < level.height and 0 <= cc < level.width):
                return True
            return level.walls[rr][cc]

        if wall(nr, nc):
            return key
        if (nr, nc) in boxes:
            br, bc = nr + dr, nc + dc
            if wall(br, bc) or (br, bc) in boxes:
                return key
            return ((nr, nc), (boxes - {(nr, nc)}) | {(br, bc)})
        return ((nr, nc), boxes)


def walls_array(level: Level) -> np.ndarray:
    """Wall layout as a boolean numpy array (renderer helper)."""
    return np.array(level.walls, dtype=bool)
