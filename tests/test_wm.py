"""Properties the spatial world model has to keep, stated as tests.

These are not accuracy checks -- accuracy lives in lab/attribute.py and
lab/probe.py, which need a trained checkpoint. These are the structural
claims wm/model.py makes in its docstring, each of which would otherwise
only be true by intention:

  * dynamics is LOCAL, so a board change cannot influence a distant cell
  * the model is fully convolutional, so grid extent is not baked in
  * the dead head is supervised on the ENCODER manifold, not only the
    dynamics manifold (the defect lab/probe.py found in the previous model)
  * the generator actually emits dead states, and labels them

torch is an optional dependency here as it is for baseline/, so the whole
module skips when it is absent.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from latent_sokoban.env import Level, SokobanState  # noqa: E402
from wm.model import DEAD, WorldModel, param_count  # noqa: E402


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return WorldModel().eval()


def test_parameter_count_within_cap(model):
    assert param_count(model) < 20_000_000


def test_encoder_is_tile_aligned_at_8x8(model):
    """64x64 in, 8x8 grid out: one latent cell per tile on an 8x8 board."""
    z = model.encoder(torch.zeros(2, 3, 64, 64))
    assert z.shape[0] == 2 and z.shape[2:] == (8, 8)


def test_dynamics_is_local(model):
    """Perturbing one latent cell must not move a cell outside the 7-cell
    receptive field. This is the property that makes crate count cheap: the
    push rule is shared across positions instead of memorised per board.

    Read the residual directly: Dynamics.forward wraps it in a global L2
    normalisation, which rescales every cell uniformly when any cell changes
    (a scalar scale change, not a learned cross-cell interaction), so the
    locality claim lives on the un-normalised residual."""
    ch = model.encoder.net[-1].out_channels
    z = torch.randn(1, ch, 8, 8)
    a = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        base = model.dynamics.residual(z, a)
        z2 = z.clone()
        z2[0, :, 0, 0] += 5.0            # perturb one corner
        moved = model.dynamics.residual(z2, a)
    delta = (moved - base).abs().sum(dim=1)[0]
    # three 3x3 convolutions reach 3 cells; the opposite corner is 7 away
    assert delta[0, 0] > 0, "the perturbed cell itself must change"
    assert delta[7, 7].item() == pytest.approx(0.0, abs=1e-6)


def test_fully_convolutional_over_grid_extent(model):
    """Heads pool over space, so a different grid extent still runs. Board
    size is NOT claimed to transfer (observations are always 64x64, so tiles
    per cell change with board size) -- this only pins that nothing hardcodes
    an 8x8 shape."""
    ch = model.encoder.net[-1].out_channels
    z = torch.randn(2, ch, 10, 10)
    zg = torch.randn(2, ch, 10, 10)
    with torch.no_grad():
        v, logits, dead = model.heads(z, zg)
    assert v.shape == (2,) and logits.shape == (2, 4) and dead.shape == (2,)


def test_heads_return_a_dead_logit(model):
    z = torch.randn(1, model.encoder.net[-1].out_channels, 8, 8)
    out = model.heads(z, z)
    assert len(out) == 3, "value, policy and dead"


def test_dead_head_is_supervised_on_the_encoder_manifold():
    """The previous model only ever labelled DEAD through a term reading a
    dynamics-produced latent, so its dead AUC was 0.918 on dynamics latents
    and 0.427 -- below chance -- on encoder ones, and the planner walked into
    deadlocks. Assert the encoder path carries gradient from the dead loss
    alone, which is what that bug would have failed.
    """
    torch.manual_seed(0)
    m = WorldModel()
    b, k = 2, 3
    frames = torch.rand(b, k + 1, 3, 64, 64)
    goal = torch.rand(b, 3, 64, 64)
    actions = torch.zeros(b, k, dtype=torch.long)
    d = torch.full((b, k + 1), DEAD)
    is_dead = torch.ones(b, k + 1, dtype=torch.bool)
    opt_mask = torch.zeros(b, k + 1, 4)

    out = m.rollout_loss(frames, goal, actions, d, is_dead, opt_mask,
                         pred_coef=0.0, var_coef=0.0, cov_coef=0.0,
                         value_coef=0.0, policy_coef=0.0, dead_coef=1.0)
    out["loss"].backward()
    grad = m.encoder.net[0].weight.grad
    assert grad is not None and grad.abs().sum().item() > 0, (
        "dead supervision must reach the encoder, not just the dynamics head")


def test_all_dead_batch_has_no_value_or_policy_loss():
    """Every state lost: the value regression has nothing to fit and the
    policy has no optimal action. Both must degrade to zero rather than to
    NaN, which is how an all-dead minibatch would otherwise poison a run."""
    torch.manual_seed(0)
    m = WorldModel()
    b, k = 2, 2
    out = m.rollout_loss(
        torch.rand(b, k + 1, 3, 64, 64), torch.rand(b, 3, 64, 64),
        torch.zeros(b, k, dtype=torch.long), torch.full((b, k + 1), DEAD),
        torch.ones(b, k + 1, dtype=torch.bool), torch.zeros(b, k + 1, 4))
    assert torch.isfinite(out["loss"])
    assert out["value"].item() == 0.0 and out["policy"].item() == 0.0


def test_policy_head_can_learn_a_positional_target():
    """The head must be able to express an answer that depends on WHERE a
    feature is, not just whether it is present. The target is "which quadrant
    holds the active cell", learnable only if position survives the readout.

    Honest note on what this test is worth: it did NOT catch a real bug. It
    was written to confirm a diagnosis of two runs whose policy loss sat at
    ln(4), and it refuted that diagnosis instead -- the previous pooled-only
    head also scores 1.000 here, and the real cause was a training plateau
    that breaks around step 1000. It is kept as a guard on a property the
    heads genuinely need, not as evidence about that incident.
    """
    torch.manual_seed(0)
    from wm.model import Heads

    ch, n = 8, 256
    heads = Heads(ch=ch, hidden=32)
    z = torch.zeros(n, ch, 8, 8)
    rows = torch.randint(0, 8, (n,))
    cols = torch.randint(0, 8, (n,))
    z[torch.arange(n), 0, rows, cols] = 1.0
    target = (rows // 4) * 2 + (cols // 4)          # 4 quadrants -> 4 actions
    zg = torch.zeros_like(z)

    opt = torch.optim.Adam(heads.parameters(), lr=3e-3)
    for _ in range(300):
        _, logits, _ = heads(z, zg)
        loss = torch.nn.functional.cross_entropy(logits, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    with torch.no_grad():
        _, logits, _ = heads(z, zg)
        acc = (logits.argmax(-1) == target).float().mean().item()
    assert acc > 0.8, (
        f"policy head cannot fit a positional target (acc {acc:.2f}); the "
        "readout has probably become permutation-invariant over cells")


def test_generator_emits_confirmed_dead_states():
    """dead_states must construct states the solver agrees are unrecoverable,
    not merely states that look cornered."""
    from distill.generate import DEAD as GEN_DEAD, dead_states, dist_to_go
    from latent_sokoban.levels import generate_level

    rng = np.random.default_rng(3)
    level, _ = generate_level(rng, size=8, n_boxes=1, wall_density=0.10,
                              min_solution_len=4, max_solution_len=20,
                              max_tries=20000)
    states = dead_states(level, rng, want=5)
    assert states, "no dead states constructed on a level with corners"
    for s in states:
        assert dist_to_go(level, s) == GEN_DEAD
