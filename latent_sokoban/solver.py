"""Breadth-first Sokoban solver and deadlock detection.

Competition note: per the rules, the symbolic solver may be used for
training-data generation and evaluation analysis ONLY. Agents may not call
it at inference time.
"""

from __future__ import annotations

from collections import deque

from latent_sokoban.env import ACTIONS, Level, SokobanEnv


def bfs_solve(level: Level, max_nodes: int = 200_000) -> list[int] | None:
    """Return an optimal (shortest) action sequence, or None if unsolvable
    or the node budget is exhausted."""
    start = (level.player, level.boxes)
    goal_boxes = level.goals
    if level.boxes == goal_boxes:
        return []
    frontier = deque([start])
    parents: dict[tuple, tuple[tuple, int] | None] = {start: None}
    nodes = 0
    while frontier:
        key = frontier.popleft()
        nodes += 1
        if nodes > max_nodes:
            return None
        for action in ACTIONS:
            nxt = SokobanEnv.apply(level, key, action)
            if nxt == key or nxt in parents:
                continue
            parents[nxt] = (key, action)
            if nxt[1] == goal_boxes:
                actions = []
                cur = nxt
                while parents[cur] is not None:
                    cur, a = parents[cur]
                    actions.append(a)
                return actions[::-1]
            # prune states with a corner-deadlocked box off-goal
            if not _has_corner_deadlock(level, nxt[1]):
                frontier.append(nxt)
    return None


def optimal_length(level: Level, max_nodes: int = 200_000) -> int | None:
    solution = bfs_solve(level, max_nodes)
    return None if solution is None else len(solution)


def _has_corner_deadlock(level: Level, boxes: frozenset) -> bool:
    for box in boxes:
        if box in level.goals:
            continue
        r, c = box

        def wall(rr, cc):
            if not (0 <= rr < level.height and 0 <= cc < level.width):
                return True
            return level.walls[rr][cc]

        up, down = wall(r - 1, c), wall(r + 1, c)
        left, right = wall(r, c - 1), wall(r, c + 1)
        if (up or down) and (left or right):
            return True
    return False


def is_deadlocked(level: Level, boxes: frozenset) -> bool:
    """Cheap sufficient deadlock test: a box off-goal stuck in a corner.

    Used by the evaluation harness to report deadlock rate; never
    available to agents. For an exact test use state_is_dead."""
    if boxes == level.goals:
        return False
    return _has_corner_deadlock(level, boxes)


def state_is_dead(level: Level, key: tuple, max_nodes: int = 50_000) -> bool:
    """Exact (bounded) deadlock test for a full state key (player, boxes)."""
    player, boxes = key
    if boxes == level.goals:
        return False
    probe = Level(level.walls, level.goals, frozenset(boxes), player)
    return bfs_solve(probe, max_nodes) is None
