"""Beam-search MPC guided by the learned value head.

Identical in shape to baseline/agent.py -- beam over imagined latents,
replan every action, every dynamics call metered -- with the scoring
function swapped. The baseline ranks beams by ||z - z_goal||; this ranks
them by the value head's predicted moves-to-go.

Nothing here decodes a board. The agent sees the same 64x64 observation the
baseline does; the symbolic solver appears only in distill/generate.py, which
RULES.md permits for training-data generation.

Budget: horizon 3 with beam 16 costs 4 + 16 + 64 = 84 dynamics calls per
action, the same as the baseline and inside the 256 cap. Head evaluations
are not dynamics calls, but they are metered here anyway rather than
argued about.

Checkpoint path from $DISTILL_CKPT (default distill/checkpoint.pt);
$DISTILL_BEAM and $DISTILL_HORIZON override the planner shape.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from distill.model import DistillModel
from latent_sokoban.agent import Agent

BEAM = int(os.environ.get("DISTILL_BEAM", 16))
HORIZON = int(os.environ.get("DISTILL_HORIZON", 3))
NOISE = float(os.environ.get("DISTILL_NOISE", 0.1))
N_ACTIONS = 4


class DistillAgent(Agent):
    def __init__(self):
        super().__init__()
        ckpt_path = os.environ.get("DISTILL_CKPT", "distill/checkpoint.pt")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        self.device = ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
        self.model = DistillModel().to(self.device).eval()
        self.model.load_state_dict(ckpt["model"])
        self.goal_z: torch.Tensor | None = None
        self._episode = 0
        self._rng = np.random.default_rng(0)

    def reset(self) -> None:
        self.goal_z = None
        self._episode += 1
        self._rng = np.random.default_rng(11_000_000 + self._episode)

    @torch.no_grad()
    def act(self, obs: np.ndarray, goal: np.ndarray,
            action_history: list[int]) -> int:
        def encode(img):
            x = torch.from_numpy(img).permute(2, 0, 1).float()[None] / 255.0
            return self.model.encoder(x.to(self.device))

        if self.goal_z is None:
            self.goal_z = encode(goal)          # encoder passes are budget-free
        z = encode(obs)

        def score(latents):
            g = self.goal_z.expand(latents.shape[0], -1)
            v, _ = self.model.heads(latents, g)
            return v                            # predicted log1p(moves-to-go)

        actions = torch.arange(N_ACTIONS, device=self.device)
        beams = self.model.dynamics(z.repeat(N_ACTIONS, 1), actions)
        self.call_meter.tick(N_ACTIONS)
        first = torch.arange(N_ACTIONS, device=self.device)
        scores = score(beams)

        for _ in range(HORIZON - 1):
            keep = torch.argsort(scores)[:BEAM]
            beams, first = beams[keep], first[keep]
            expanded = beams.repeat_interleave(N_ACTIONS, dim=0)
            acts = actions.repeat(beams.shape[0])
            beams = self.model.dynamics(expanded, acts)
            self.call_meter.tick(beams.shape[0])
            first = first.repeat_interleave(N_ACTIONS)
            scores = score(beams)

        if NOISE > 0:
            # Same purpose as the baseline's Gumbel noise: a strict argmin
            # oscillates in local minima of the score field. Seeded per
            # episode, so runs stay reproducible.
            noise = torch.from_numpy(
                self._rng.gumbel(0.0, NOISE, size=len(scores))
            ).float().to(scores.device)
            scores = scores + noise
        return int(first[int(torch.argmin(scores))].item())
