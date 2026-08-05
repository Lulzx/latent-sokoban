"""Search-distilled world model: latent dynamics plus a learned heuristic.

Same shape as baseline/model.py -- conv encoder, residual-MLP latent
dynamics, one-step prediction loss, anti-collapse regularizer -- with one
addition that changes what the planner can do:

A VALUE HEAD, trained on true BFS distance-to-go, replaces ||z - z_goal||
as the planning heuristic. That substitution is the whole point. Euclidean
latent distance assumes progress is monotone in similarity to the goal
image, and in Sokoban it is not: a crate frequently has to be pushed away
from its target before any solution exists. Measured on the baseline, the
latent metric is fine on its own terms (greedy accuracy 0.69 against 0.25
chance) and still scores 0% at 8x8, which is what a wrong-by-construction
heuristic looks like.

The value head also gets to see deadlocks, labelled at DEAD, so "this push
is unrecoverable" is learnable rather than something the planner discovers
by wasting the rest of the episode.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 128
N_ACTIONS = 4
DEAD = 99.0


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 4, stride=2, padding=1), nn.ReLU(),
        )
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x).flatten(1))


class Dynamics(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM, action_dim: int = 32,
                 hidden: int = 256):
        super().__init__()
        self.action_embed = nn.Embedding(N_ACTIONS, action_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return z + self.net(torch.cat([z, self.action_embed(a)], dim=-1))


class Heads(nn.Module):
    """Value (moves-to-go) and policy (which action is optimal) from (z, z_goal).

    Both are conditioned on the goal embedding, not just the state: the same
    board means different things under different goals.
    """

    def __init__(self, latent_dim: int = LATENT_DIM, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.value = nn.Linear(hidden, 1)
        self.policy = nn.Linear(hidden, N_ACTIONS)

    def forward(self, z: torch.Tensor, z_goal: torch.Tensor):
        h = self.trunk(torch.cat([z, z_goal], dim=-1))
        return self.value(h).squeeze(-1), self.policy(h)


class DistillModel(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.dynamics = Dynamics(latent_dim)
        self.heads = Heads(latent_dim)

    def loss(self, obs, goal, action, next_obs, d_here, d_next, opt_mask,
             var_coef: float = 1.0, cov_coef: float = 0.05,
             value_coef: float = 1.0, policy_coef: float = 1.0) -> dict:
        z = self.encoder(obs)
        z_goal = self.encoder(goal)
        z_next = self.encoder(next_obs)

        pred = self.dynamics(z, action)
        pred_loss = F.mse_loss(pred, z_next)

        var_z, cov_z = _var_cov_loss(z)
        var_n, cov_n = _var_cov_loss(z_next)
        var_loss, cov_loss = (var_z + var_n) / 2, (cov_z + cov_n) / 2

        # Value is regressed in log1p space. Raw targets span 0..99 because
        # deadlocks sit at DEAD, and an L2 fit on that range is dominated by
        # the dead states -- log1p keeps "dead" clearly worst without letting
        # it swamp the 4-20 band where every real decision is made.
        v_here, logits = self.heads(z, z_goal)
        v_next, _ = self.heads(self.dynamics(z, action), z_goal)
        value_loss = (F.mse_loss(v_here, torch.log1p(d_here))
                      + F.mse_loss(v_next, torch.log1p(d_next))) / 2

        # Several actions can be optimal; train against the set, not one label.
        valid = opt_mask.sum(-1) > 0
        if valid.any():
            logp = F.log_softmax(logits[valid], dim=-1)
            tgt = opt_mask[valid] / opt_mask[valid].sum(-1, keepdim=True)
            policy_loss = -(tgt * logp).sum(-1).mean()
        else:
            policy_loss = torch.zeros((), device=obs.device)

        total = (pred_loss + var_coef * var_loss + cov_coef * cov_loss
                 + value_coef * value_loss + policy_coef * policy_loss)
        return {"loss": total, "pred": pred_loss.detach(),
                "var": var_loss.detach(), "cov": cov_loss.detach(),
                "value": value_loss.detach(), "policy": policy_loss.detach(),
                "latent_std": z.std(dim=0).mean().detach()}


def _var_cov_loss(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    z = z - z.mean(dim=0)
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = F.relu(1.0 - std).mean()
    n, d = z.shape
    cov = (z.T @ z) / max(n - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    return var_loss, (off_diag ** 2).sum() / d


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
