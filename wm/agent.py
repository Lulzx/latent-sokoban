"""Beam-search MPC over spatial latents, ranked by value, policy and dead.

The score for a candidate plan is

    value(z_leaf)  -  BETA * sum_t log pi(a_t | z_t)  +  PENALTY * P(dead)

and each of the three terms is there because a measurement said so:

  value      the goal-distance heuristic. lab/attribute.py measured it as the
             WEAKEST of the three: with perfect dynamics and no policy prior
             the value head solves 10% of Split A while the policy argmax
             alone solves 40%, and removing the value term from the score
             changes nothing. It is kept because it is harmless and a better
             value head would help; the policy is what actually carries the
             search.
  policy     lab/attribute.py measured that ranking by value alone is WORSE
             than taking the policy head's argmax with no search at all. The
             policy is the stronger signal; it proposes, the value refines.
  dead       lab/probe.py measured the previous model's dead-state AUC at
             0.427 -- below chance -- on encoder latents, and its planner
             walked into deadlocks 65% of the time when given accurate
             dynamics. wm/model.py trains a dead head on both manifolds; this
             is what spends it.

Metering: only dynamics calls are ticked, which is what RULES.md bounds --
one predicted transition of one candidate state. Encoder and head passes are
free, on the same ground the baseline's goal-scoring passes are.

The search expands ALL four actions per node (TOPK=4), not a pruned top-k:
the policy head is only ~0.91 right, so pruning to top-2 occasionally drops
the optimal branch, and uniform expansion at HORIZON=5 costs 4 + 16 + 64 +
64 + 64 = 212 calls, still inside the 256 cap. Measured on Split A this is
the difference between 0.50 (TOPK=2, horizon 3) and 0.70 (uniform, horizon 5).

Checkpoint from $WM_CKPT (default wm/checkpoint_8x1.pt); $WM_BEAM,
$WM_HORIZON, $WM_BETA, $WM_DEAD_PENALTY, $WM_TOPK and $WM_NOISE override the
planner.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latent_sokoban.agent import Agent  # noqa: E402
from wm.model import WorldModel         # noqa: E402

BEAM = int(os.environ.get("WM_BEAM", 16))
HORIZON = int(os.environ.get("WM_HORIZON", 5))
BETA = float(os.environ.get("WM_BETA", 0.5))
DEAD_PENALTY = float(os.environ.get("WM_DEAD_PENALTY", 4.0))
NOISE = float(os.environ.get("WM_NOISE", 0.1))
TOPK = int(os.environ.get("WM_TOPK", 4))
USE_VALUE = int(os.environ.get("WM_USE_VALUE", 1))
N_ACTIONS = 4


class WMAgent(Agent):
    def __init__(self):
        super().__init__()
        ckpt_path = os.environ.get("WM_CKPT", "wm/checkpoint_8x1.pt")
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
        self._episode += 1
        self._rng = np.random.default_rng(11_000_000 + self._episode)

    def _encode(self, img: np.ndarray) -> torch.Tensor:
        x = torch.from_numpy(img).permute(2, 0, 1).float()[None] / 255.0
        return self.model.encoder(x.to(self.device))

    def _score(self, z: torch.Tensor, path: torch.Tensor) -> torch.Tensor:
        g = self.goal_z.expand(z.shape[0], -1, -1, -1)
        v, _, dead = self.model.heads(z, g)
        if USE_VALUE:
            return v - BETA * path + DEAD_PENALTY * torch.sigmoid(dead)
        return -BETA * path + DEAD_PENALTY * torch.sigmoid(dead)

    def _log_prior(self, z: torch.Tensor) -> torch.Tensor:
        g = self.goal_z.expand(z.shape[0], -1, -1, -1)
        _, logits, _ = self.model.heads(z, g)
        return torch.log_softmax(logits, dim=-1)

    @torch.no_grad()
    def act(self, obs: np.ndarray, goal: np.ndarray,
            action_history: list[int]) -> int:
        if self.goal_z is None:
            self.goal_z = self._encode(goal)     # encoder passes are budget-free
        z = self._encode(obs)

        actions = torch.arange(N_ACTIONS, device=self.device)
        path = self._log_prior(z)[0]
        beams = self.model.dynamics(z.expand(N_ACTIONS, -1, -1, -1), actions)
        self.call_meter.tick(N_ACTIONS)
        first = torch.arange(N_ACTIONS, device=self.device)
        scores = self._score(beams, path)

        for _ in range(HORIZON - 1):
            keep = torch.argsort(scores)[:BEAM]
            beams, first, path = beams[keep], first[keep], path[keep]
            node_logp = self._log_prior(beams)
            if TOPK < N_ACTIONS:
                # Expand only the top-TOPK actions by policy log-prob, instead
                # of a uniform 4-way fan-out. The policy head names an optimal
                # action ~0.90 of the time (lab/probe.py), so the pruned branch
                # rarely carries the optimum, while the saved dynamics calls buy
                # a far longer horizon: TOPK=2 takes the same 84-call plan from
                # horizon 3 to horizon ~8 within the 256 cap. TOPK=4 restores
                # the uniform expansion.
                topk = node_logp.topk(TOPK, dim=-1)
                expanded = beams.repeat_interleave(TOPK, dim=0)
                acts = topk.indices.flatten()
                beams = self.model.dynamics(expanded, acts)
                self.call_meter.tick(beams.shape[0])
                first = first.repeat_interleave(TOPK)
                path = path.repeat_interleave(TOPK) + topk.values.flatten()
            else:
                expanded = beams.repeat_interleave(N_ACTIONS, dim=0)
                acts = actions.repeat(beams.shape[0])
                beams = self.model.dynamics(expanded, acts)
                self.call_meter.tick(beams.shape[0])
                first = first.repeat_interleave(N_ACTIONS)
                path = path.repeat_interleave(N_ACTIONS) + node_logp.flatten()
            scores = self._score(beams, path)

        s = scores.cpu().numpy()
        if NOISE > 0:
            # Breaks the left-right oscillation a strict argmin falls into at
            # local minima of the score field. Seeded per episode, so runs
            # stay reproducible.
            s = s + self._rng.gumbel(0.0, NOISE, size=len(s))
        return int(first[int(np.argmin(s))].item())
