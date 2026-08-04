"""Shared baseline agent: beam-search MPC in latent space.

Per the required-baseline spec: Euclidean latent distance to the encoded
goal image, beam search over imagined futures, replanning after every
action. Every dynamics forward pass is counted on the CallMeter.

The planning horizon is deliberately short (default 3): the baseline is
trained with a ONE-step loss, and measured open-loop drift reaches the
typical start-to-goal latent distance by ~6 imagined steps, so long beams
score noise. Short-horizon beams plus replanning after every action is
the right operating point for this model; extending usable horizon (e.g.
multi-step rollout losses) is exactly the research-round headroom.

Budget: horizon 3 with beam 16 costs 4 + 16 + 64 = 84 dynamics calls per
action, inside the 256-call cap.

Checkpoint path comes from $BASELINE_CKPT (default baseline/checkpoint.pt);
$BASELINE_BEAM and $BASELINE_HORIZON override the planner shape.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baseline.model import WorldModel
from latent_sokoban.agent import Agent

BEAM = int(os.environ.get("BASELINE_BEAM", 16))
HORIZON = int(os.environ.get("BASELINE_HORIZON", 3))
NOISE = float(os.environ.get("BASELINE_NOISE", 0.2))
N_ACTIONS = 4


class BaselineAgent(Agent):
    def __init__(self):
        super().__init__()
        ckpt_path = os.environ.get("BASELINE_CKPT", "baseline/checkpoint.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.device = ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = WorldModel().to(self.device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.goal_z: torch.Tensor | None = None
        self._episode = 0
        self._rng = np.random.default_rng(0)

    def reset(self) -> None:
        self.goal_z = None
        # fresh deterministic noise stream per episode: a purely greedy
        # argmin oscillates forever in local minima of the distance field,
        # so scores get small Gumbel noise (eta=NOISE) to break loops
        # while keeping descent bias. Seeded, so runs are reproducible.
        self._episode += 1
        self._rng = np.random.default_rng(7_000_000 + self._episode)

    @torch.no_grad()
    def act(self, obs: np.ndarray, goal: np.ndarray,
            action_history: list[int]) -> int:
        def encode(img):
            x = torch.from_numpy(img).permute(2, 0, 1).float()[None] / 255.0
            return self.model.encoder(x.to(self.device))

        if self.goal_z is None:
            self.goal_z = encode(goal)  # encoder passes are budget-free
        z = encode(obs)

        # beam entries: (latent, first_action, cost-so-far is implicit via score)
        beams = z.repeat(N_ACTIONS, 1)                       # (4, D)
        actions = torch.arange(N_ACTIONS, device=self.device)
        beams = self.model.dynamics(beams, actions)          # depth 1
        self.call_meter.tick(N_ACTIONS)
        first = torch.arange(N_ACTIONS, device=self.device)
        scores = torch.linalg.norm(beams - self.goal_z, dim=-1)

        for _ in range(HORIZON - 1):
            keep = torch.argsort(scores)[:BEAM]
            beams, first = beams[keep], first[keep]
            b = beams.shape[0]
            expanded = beams.repeat_interleave(N_ACTIONS, dim=0)
            acts = actions.repeat(b)
            expanded = self.model.dynamics(expanded, acts)
            self.call_meter.tick(expanded.shape[0])
            beams = expanded
            first = first.repeat_interleave(N_ACTIONS)
            scores = torch.linalg.norm(beams - self.goal_z, dim=-1)

        if NOISE > 0:
            noise = torch.from_numpy(
                self._rng.gumbel(0.0, NOISE, size=len(scores))
            ).float().to(scores.device)
            scores = scores + noise
        return int(first[int(torch.argmin(scores))].item())
