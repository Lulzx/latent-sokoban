"""Agent interface for the evaluation harness.

An agent sees ONLY what the competition rules allow: the current board
image, the goal board image, and its own action history. The harness
never passes symbolic state.

Competitors implement Agent in their own repos and point evaluate.py at
it with --agent path.to.module:ClassName. The class is constructed with
no arguments (load your checkpoint in __init__ or reset()).
"""

from __future__ import annotations

import numpy as np


class Agent:
    """Base class. Subclass and override act()."""

    def reset(self) -> None:
        """Called once at the start of every episode."""

    def act(
        self,
        obs: np.ndarray,        # (64, 64, 3) uint8 current board image
        goal: np.ndarray,       # (64, 64, 3) uint8 goal board image
        action_history: list[int],
    ) -> int:
        """Return an action id in {0: up, 1: down, 2: left, 3: right}."""
        raise NotImplementedError


class RandomAgent(Agent):
    """Uniform-random policy. The floor of the leaderboard."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def act(self, obs, goal, action_history):
        return int(self.rng.integers(0, 4))
