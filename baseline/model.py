"""Shared baseline world model (see RULES.md "Required Baseline").

- Small convolutional encoder -> 128-d latent
- Learned action embeddings
- Residual MLP dynamics predictor
- One-step latent prediction loss
- Variance + covariance regularization (VICReg-style) against collapse
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_DIM = 128
N_ACTIONS = 4


class Encoder(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1),   # 64 -> 32
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 32 -> 16
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 16 -> 8
            nn.ReLU(),
            nn.Conv2d(128, 128, 4, stride=2, padding=1),  # 8 -> 4
            nn.ReLU(),
        )
        self.fc = nn.Linear(128 * 4 * 4, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, 64, 64) float in [0, 1]
        return self.fc(self.conv(x).flatten(1))


class Dynamics(nn.Module):
    """Residual MLP: z' = z + f([z, embed(a)])."""

    def __init__(self, latent_dim: int = LATENT_DIM, action_dim: int = 32,
                 hidden: int = 256):
        super().__init__()
        self.action_embed = nn.Embedding(N_ACTIONS, action_dim)
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return z + self.net(torch.cat([z, self.action_embed(a)], dim=-1))


class WorldModel(nn.Module):
    def __init__(self, latent_dim: int = LATENT_DIM):
        super().__init__()
        self.encoder = Encoder(latent_dim)
        self.dynamics = Dynamics(latent_dim)

    def loss(self, obs: torch.Tensor, action: torch.Tensor,
             next_obs: torch.Tensor, var_coef: float = 1.0,
             cov_coef: float = 0.05) -> dict:
        z = self.encoder(obs)
        z_next = self.encoder(next_obs)
        pred = self.dynamics(z, action)
        pred_loss = F.mse_loss(pred, z_next)
        reg = _var_cov_loss(z) + _var_cov_loss(z_next)
        var_loss, cov_loss = reg[0] / 2, reg[1] / 2
        total = pred_loss + var_coef * var_loss + cov_coef * cov_loss
        return {"loss": total, "pred": pred_loss.detach(),
                "var": var_loss.detach(), "cov": cov_loss.detach(),
                "latent_std": z.std(dim=0).mean().detach()}


def _var_cov_loss(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """VICReg variance hinge + covariance penalty."""
    z = z - z.mean(dim=0)
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    var_loss = F.relu(1.0 - std).mean()
    n, d = z.shape
    cov = (z.T @ z) / max(n - 1, 1)
    off_diag = cov - torch.diag(torch.diag(cov))
    cov_loss = (off_diag ** 2).sum() / d
    return var_loss, cov_loss


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
