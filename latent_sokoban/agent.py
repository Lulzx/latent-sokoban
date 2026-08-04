"""Agent interface for the evaluation harness.

An agent sees ONLY what the competition rules allow: the current board
image, the goal board image, and its own action history. The harness
never passes symbolic state.

Competitors implement Agent in their own repos and point evaluate.py at
it with --agent path.to.module:ClassName. The class is constructed with
no arguments (load your checkpoint in __init__ or reset()).

Dynamics-call budget
--------------------
Planning is budgeted in *counted learned-dynamics calls*, not wall-clock
rollouts. One unit = one predicted transition of one candidate state, so
a batched forward pass over B candidates rolled H steps costs B * H
units. Your planner MUST tick the meter the harness attaches to
self.call_meter every time your dynamics model runs:

    class MyAgent(Agent):
        def act(self, obs, goal, action_history):
            ...
            z_next = self.dynamics(z, a)          # z has batch size B
            self.call_meter.tick(z.shape[0])      # count it
            ...

The harness checks the meter after every act(): exceeding the per-action
cap fails the episode (strict mode, the default). Correct metering is
verified by source review at submission — an unmetered or under-metered
dynamics call is a rules violation. Encoder and goal-scoring passes are
free; only latent transition predictions count.
"""

from __future__ import annotations

import numpy as np


class CallMeter:
    """Counts learned-dynamics forward passes during planning."""

    def __init__(self) -> None:
        self.total = 0

    def tick(self, n: int = 1) -> None:
        self.total += int(n)


class Agent:
    """Base class. Subclass and override act()."""

    #: Attached by the harness before each episode. Tick it on every
    #: dynamics-model call (see module docstring).
    call_meter: CallMeter

    def __init__(self) -> None:
        self.call_meter = CallMeter()

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
    """Uniform-random policy. The floor of the leaderboard.

    Makes zero dynamics calls, so it never violates the call budget."""

    def __init__(self, seed: int = 0):
        super().__init__()
        self.rng = np.random.default_rng(seed)

    def act(self, obs, goal, action_history):
        return int(self.rng.integers(0, 4))
