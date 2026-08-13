"""A spatial latent world model for Sokoban from pixels.

Three measured facts drove this design, all from lab/ on the previous
(global-vector) model:

  1. Open-loop latent rollouts lose the true state after ONE imagined step
     (retrieval 0.717, decaying to 0.49 by step 3). Planning past horizon 3
     was scoring noise.
  2. Success at three crates is 0.000 on splits A, C and D, with deadlock
     rates of 0.73-0.93. A model trained at one crate transfers to nothing.
  3. Giving the old planner PERFECT dynamics changed its score by exactly
     zero, so the losses were in the heads, not the transition model.

The structural response to 1 and 2 is to stop throwing away space. A
Sokoban transition changes at most three tiles -- the tile the player
leaves, the tile it enters, and the tile a crate moves to -- and that rule
is identical everywhere on the board and identical for every crate. A dense
MLP over a pooled 128-d vector can represent none of that; it has to
memorise each configuration separately, which is exactly what "works at one
crate, collapses at three" looks like.

So:

  * The latent is an 8x8 grid of channels, not a vector. The encoder
    downsamples 64x64 by exactly 8, so at the 8x8 board size one latent cell
    corresponds to one board tile -- render.py draws 64 // board_size pixels
    per tile, which is exactly 8 there.
  * Dynamics is a residual CONVOLUTION over that grid, action broadcast as
    extra channels. Its receptive field is 7 cells: wide enough for the
    local push rule, far too narrow to memorise a board. Weight sharing
    across cells is what makes CRATE COUNT close to free rather than
    learned -- the rule for pushing a crate is one rule, not one per
    configuration.
  * Heads read (z, z_goal) concatenated channel-wise, so "which crate is
    not where it should be" is a LOCAL comparison, and mean+max pooling
    makes the head input independent of grid extent.

The latent lives on a sphere and the prediction loss is contrastive, which
is the fix for the measured training failure. The first formulation scored
rollouts with raw MSE and contrasted DELTAS; it diverged because a single
Sokoban move changes 3 of 64 tiles, so the encoder mapped consecutive
frames to nearly-identical latents (null_1 ~ 8e-6, the one-step MSE under
"predict no change"), the dynamics answered "no change" and looked good in
MSE while the delta-contrastive normalised an already-degenerate delta and
sat at chance. Global L2 normalisation puts every state on a fixed-radius
sphere, where a move is an angular delta comparable across states, and
infoNCE over POSITIONS -- not deltas -- asks "which candidate is my true
next state?", which cannot be answered by predicting no change. Cosine
alignment anchors the prediction on-manifold, so the two no longer fight
(the earlier contrast-on-positions attempt pushed rollouts off the encoder
manifold at 4400x relative error; on a sphere that degree of freedom is
gone).

Scope, stated so it is not over-claimed: this buys crate-count
generalisation, NOT board-size generalisation. Observations are always
64x64 whatever the board, so a 10x10 board renders at 6 pixels per tile and
a 6x6 at 10, and the tile-to-cell correspondence that makes the prior sharp
holds only at 8x8. That is the right trade here because every one of the
hidden set's 100 levels is 8x8 and the only thing that ramps is crate count,
1 to 4. Splits W and C, which do change board size, stay hard on purpose.
  * A separate DEAD head, trained on both the encoder manifold and the
    dynamics manifold. The previous model labelled dead states only through
    a term that read dynamics-produced latents, so its dead-state AUC was
    0.918 on dynamics latents and 0.427 -- below chance -- on encoder ones.
  * The dynamics loss is multi-step, because open-loop drift is the failure
    and one-step MSE is not a proxy for it.

Rules note: a tile-aligned spatial latent is still a continuous feature
map. Nothing here classifies tiles, emits grid coordinates, reconstructs a
symbolic board, or searches enumerated discrete states -- the prohibitions
in docs/RULES.md. What it uses is the same translation-invariance prior any
convnet on images uses.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LATENT_CH = 48
ACTION_CH = 16
N_ACTIONS = 4
DEAD = 99.0
LATENT_SCALE = 8.0


def _normalize(z: torch.Tensor) -> torch.Tensor:
    """Global L2-normalise each sample's latent, rescaled to LATENT_SCALE.

    On a fixed-radius sphere a single move is an angular delta comparable
    across states, so the dynamics cannot win the loss by predicting no
    change while the encoder shrinks consecutive states together. The scale
    keeps per-cell magnitudes reasonable for the conv heads (unit norm over
    3072 dims would feed them ~0.018 per cell).
    """
    return F.normalize(z.flatten(1), dim=1).view_as(z) * LATENT_SCALE


class Encoder(nn.Module):
    """64x64x3 -> (LATENT_CH, 8, 8), globally L2-normalised. Downsample is
    exactly 8, tile-aligned."""

    def __init__(self, ch: int = LATENT_CH):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, stride=2, padding=1), nn.ReLU(),    # 32x32
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),   # 16x16
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, ch, 4, stride=2, padding=1),              # 8x8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _normalize(self.net(x))


class Dynamics(nn.Module):
    """z' = normalize(z + f(z, a)), f convolutional and local."""

    def __init__(self, ch: int = LATENT_CH, action_ch: int = ACTION_CH,
                 hidden: int = 96):
        super().__init__()
        self.action_embed = nn.Embedding(N_ACTIONS, action_ch)
        self.net = nn.Sequential(
            nn.Conv2d(ch + action_ch, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.ReLU(),
            nn.Conv2d(hidden, ch, 3, padding=1),
        )

    def residual(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """The un-normalised update z + f(z, a). Local: f is three 3x3
        convolutions, so a cell only sees a 7-cell neighbourhood. Forward
        wraps this in a global L2 normalisation (the spherical latent); that
        rescales the whole grid uniformly when one cell changes, which is a
        scalar scale change carrying no information about WHICH cell moved --
        it does not widen the receptive field."""
        b, _, h, w = z.shape
        a_map = self.action_embed(a)[:, :, None, None].expand(b, -1, h, w)
        return z + self.net(torch.cat([z, a_map], dim=1))

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return _normalize(self.residual(z, a))


class Heads(nn.Module):
    """Value (moves-to-go), policy (optimal action) and dead, from (z, z_goal).

    Mean AND max pooling: mean carries "how much is left to do overall", max
    carries "is there a single tile that is catastrophically wrong", which is
    what a deadlock is.

    The trunk is DILATED (1, 2, 4), giving a receptive field of 15 cells, so
    every cell sees the whole 8x8 board. Dynamics deliberately does NOT get
    this treatment: a transition really is local, and keeping its receptive
    field at 7 is what stops it memorising whole-board configurations.
    """

    SPATIAL_GRID = 8
    SPATIAL_CH = 16

    def __init__(self, ch: int = LATENT_CH, hidden: int = 96):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Conv2d(ch * 2, hidden, 3, padding=1, dilation=1), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(hidden, hidden, 3, padding=4, dilation=4), nn.ReLU(),
        )
        self.reduce = nn.Conv2d(hidden, self.SPATIAL_CH, 1)
        self.spatial = nn.Linear(self.SPATIAL_CH * self.SPATIAL_GRID ** 2,
                                 hidden)
        self.fc = nn.Sequential(nn.Linear(hidden * 3, hidden), nn.ReLU())
        self.value = nn.Linear(hidden, 1)
        self.policy = nn.Linear(hidden, N_ACTIONS)
        self.dead = nn.Linear(hidden, 1)

    def forward(self, z: torch.Tensor, z_goal: torch.Tensor):
        h = self.trunk(torch.cat([z, z_goal], dim=1))
        pooled = torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1)
        r = F.adaptive_avg_pool2d(self.reduce(h), self.SPATIAL_GRID)
        spatial = self.spatial(r.flatten(1))
        h = self.fc(torch.cat([pooled, spatial], dim=1))
        return (self.value(h).squeeze(-1), self.policy(h),
                self.dead(h).squeeze(-1))


class WorldModel(nn.Module):
    def __init__(self, ch: int = LATENT_CH):
        super().__init__()
        self.encoder = Encoder(ch)
        self.dynamics = Dynamics(ch)
        self.heads = Heads(ch)

    # -- losses -----------------------------------------------------------

    def rollout_loss(self, frames, goal, actions, d, is_dead, opt_mask,
                     same=None,
                     var_coef: float = 0.0, cov_coef: float = 0.0,
                     value_coef: float = 1.0, policy_coef: float = 1.0,
                     dead_coef: float = 1.0, pred_coef: float = 1.0,
                     nce_coef: float = 1.0, rolled_coef: float = 0.25,
                     temperature: float = 0.1) -> dict:
        """One chain per batch row.

        frames    (B, K+1, 3, 64, 64)  observations along the chain
        goal      (B, 3, 64, 64)
        actions   (B, K)               action taken FROM frames[:, k]
        d         (B, K+1)             true moves-to-go, DEAD where lost
        is_dead   (B, K+1) bool
        opt_mask  (B, K+1, 4)          set of optimal actions, all-zero if none
        same      (B, K) bool          action k was a no-op (unused; retained
                                       for signature compatibility)

        The prediction objective is spherical contrastive: cosine alignment
        anchors each rollout step on-manifold, and infoNCE over POSITIONS
        makes the predicted next latent discriminable against the batch's
        other next latents. Head losses are applied to BOTH the encoder
        latent and the rolled latent at every step.
        """
        b, kp1 = frames.shape[0], frames.shape[1]
        flat = frames.reshape(b * kp1, *frames.shape[2:])
        z_all = self.encoder(flat)
        z_true = z_all.reshape(b, kp1, *z_all.shape[1:])
        z_goal = self.encoder(goal)

        # multi-step open-loop rollout from the first frame only
        z = z_true[:, 0]
        rolled = [z]
        align_losses, nce_losses, nce_hits = [], [], []
        labels = torch.arange(b, device=frames.device)
        scale2 = LATENT_SCALE ** 2
        for k in range(kp1 - 1):
            z = self.dynamics(z, actions[:, k])
            rolled.append(z)
            z_next = z_true[:, k + 1]
            cos = (z.flatten(1) * z_next.flatten(1)).sum(1) / scale2
            align_losses.append((1.0 - cos).mean())
            sim = (z.flatten(1) @ z_next.flatten(1).T) / scale2
            nce_losses.append(F.cross_entropy(sim / temperature, labels))
            nce_hits.append((sim.argmax(1) == labels).float().mean())
        align_loss = torch.stack(align_losses).mean()
        zero = torch.zeros((), device=frames.device)
        nce_loss = torch.stack(nce_losses).mean() if nce_losses else zero
        nce_acc = torch.stack(nce_hits).mean() if nce_hits else zero

        # head supervision on both manifolds
        v_losses, p_losses, d_losses = [], [], []
        v_report, p_report, d_report = [], [], []
        p_hit = p_n = 0
        for k in range(kp1):
            for zk, w in ((z_true[:, k], 1.0), (rolled[k], rolled_coef)):
                if w == 0.0:
                    continue
                v, logits, dead_logit = self.heads(zk, z_goal)
                alive = ~is_dead[:, k]
                if alive.any():
                    vl = F.mse_loss(v[alive], torch.log1p(d[:, k][alive]))
                    v_losses.append(w * vl)
                    if w == 1.0:
                        v_report.append(vl.detach())
                dl = F.binary_cross_entropy_with_logits(
                    dead_logit, is_dead[:, k].float())
                d_losses.append(w * dl)
                if w == 1.0:
                    d_report.append(dl.detach())
                m = opt_mask[:, k]
                valid = m.sum(-1) > 0
                if valid.any():
                    logp = F.log_softmax(logits[valid], dim=-1)
                    tgt = m[valid] / m[valid].sum(-1, keepdim=True)
                    pl = -(tgt * logp).sum(-1).mean()
                    p_losses.append(w * pl)
                    if w == 1.0:
                        p_report.append(pl.detach())
                    pick = logits[valid].argmax(-1)
                    if w == 1.0:
                        p_hit += m[valid].gather(1, pick[:, None]).sum().item()
                        p_n += int(valid.sum())

        value_loss = torch.stack(v_losses).mean() if v_losses else zero
        policy_loss = torch.stack(p_losses).mean() if p_losses else zero
        dead_loss = torch.stack(d_losses).mean() if d_losses else zero

        total = (pred_coef * align_loss + nce_coef * nce_loss
                 + value_coef * value_loss + policy_coef * policy_loss
                 + dead_coef * dead_loss)

        # Collapse detection: mean pairwise cosine among the batch's first
        # frames. Near 1 means every state maps to one point; the contrastive
        # and head losses should keep this well below 1.
        z0 = F.normalize(z_true[:, 0].flatten(1), dim=1)
        off_diag = (z0 @ z0.T) * (1.0 - torch.eye(b, device=frames.device))
        spread = off_diag.sum() / max(b * (b - 1), 1)

        return {"loss": total,
                "align": (1.0 - align_loss).detach(),
                "nce": nce_loss.detach(), "nce_acc": nce_acc,
                "policy_acc": torch.tensor(p_hit / max(p_n, 1)),
                "value": (torch.stack(v_report).mean() if v_report else zero),
                "policy": (torch.stack(p_report).mean() if p_report else zero),
                "dead": (torch.stack(d_report).mean() if d_report else zero),
                "spread": spread.detach(),
                "latent_std": z_all.permute(0, 2, 3, 1).reshape(
                    -1, z_all.shape[1]).std(dim=0).mean().detach()}


def param_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
