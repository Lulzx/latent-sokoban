"""Smoke tests for the shared infrastructure. Run: python -m pytest tests/"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from latent_sokoban import (
    Level,
    SokobanEnv,
    bfs_solve,
    generate_level,
    render,
)
from latent_sokoban.dataset import generate_shard, load_shard, save_shard
from latent_sokoban.render import default_theme, random_theme, render_goal
from latent_sokoban.solver import state_is_dead

SIMPLE = """\
######
#    #
# $. #
# @  #
#    #
######"""


def test_ascii_roundtrip():
    level = Level.from_ascii(SIMPLE)
    assert level.to_ascii() == SIMPLE
    assert level.player == (3, 2)
    assert level.boxes == frozenset({(2, 2)})
    assert level.goals == frozenset({(2, 3)})


def test_push_solves():
    level = Level.from_ascii(SIMPLE)
    env = SokobanEnv(level)
    # walk below the box's left side, then push right: up moves to (2,2)?
    # player at (3,2), box at (2,2): move left, up, up? Instead: go to (2,1)
    env.step(2)  # left -> (3,1)
    env.step(0)  # up   -> (2,1)
    _, done, info = env.step(3)  # right: pushes box (2,2)->(2,3) onto goal
    assert info.pushed and done and env.solved


def test_wall_is_noop():
    level = Level.from_ascii(SIMPLE)
    env = SokobanEnv(level)
    env.step(2)  # -> (3,1)
    state, _, info = env.step(2)  # into wall
    assert info.invalid and state.player == (3, 1) and state.steps == 2


def test_blocked_push_is_noop():
    level = Level.from_ascii("""\
######
#    #
#$@ .#
#    #
#    #
######""")
    env = SokobanEnv(level)
    state, _, info = env.step(2)  # push box into the left wall
    assert info.invalid and state.player == (2, 2)


def test_step_limit_truncates():
    level = Level.from_ascii(SIMPLE)
    env = SokobanEnv(level, max_steps=3)
    done = False
    for _ in range(3):
        _, done, info = env.step(1)
    assert done and info.truncated and not env.solved


def test_solver_optimal():
    level = Level.from_ascii(SIMPLE)
    solution = bfs_solve(level)
    assert solution is not None and len(solution) == 3
    env = SokobanEnv(level)
    for a in solution:
        env.step(a)
    assert env.solved


def test_solver_detects_unsolvable():
    dead = """\
######
#$   #
#  . #
# @  #
#    #
######"""
    assert bfs_solve(Level.from_ascii(dead)) is None


def test_state_is_dead():
    level = Level.from_ascii(SIMPLE)
    corner = ((3, 2), frozenset({(1, 1)}))  # box in top-left corner, off goal
    assert state_is_dead(level, corner)
    assert not state_is_dead(level, (level.player, level.boxes))


def test_generator_produces_solvable_levels():
    rng = np.random.default_rng(7)
    for _ in range(5):
        level, solution = generate_level(rng)
        assert bfs_solve(level) is not None
        assert len(solution) >= 2
        env = SokobanEnv(level)
        for a in solution:
            env.step(a)
        assert env.solved


def test_generator_deterministic():
    a, _ = generate_level(np.random.default_rng(42))
    b, _ = generate_level(np.random.default_rng(42))
    assert a.to_ascii() == b.to_ascii()


def test_render_shape_and_determinism():
    level = Level.from_ascii(SIMPLE)
    img1 = render(level)
    img2 = render(level)
    assert img1.shape == (64, 64, 3) and img1.dtype == np.uint8
    assert np.array_equal(img1, img2)
    goal = render_goal(level)
    assert goal.shape == (64, 64, 3)
    assert not np.array_equal(img1, goal)


def test_render_7x7_fits():
    rng = np.random.default_rng(3)
    level, _ = generate_level(rng, size=7, min_solution_len=2)
    assert render(level).shape == (64, 64, 3)


def test_themes_differ():
    level = Level.from_ascii(SIMPLE)
    rng = np.random.default_rng(5)
    themed = render(level, theme=random_theme(rng), rng=np.random.default_rng(0))
    assert not np.array_equal(themed, render(level, theme=default_theme()))


def test_dataset_roundtrip(tmp_path):
    rng = np.random.default_rng(11)
    shard = generate_shard(rng, n_episodes=6)
    path = tmp_path / "shard.npz"
    save_shard(shard, path, meta={"seed": 11})
    loaded = load_shard(path)
    assert loaded["frames"].dtype == np.uint8
    assert len(loaded["episode_starts"]) == 6
    assert loaded["goal_frames"].shape == (6, 64, 64, 3)
    # frame/action alignment: last action of each episode is -1
    ends = loaded["episode_starts"] + loaded["episode_lens"] - 1
    assert (loaded["actions"][ends] == -1).all()
    assert (loaded["actions"][loaded["actions"] != -1] < 4).all()
    meta = json.loads(path.with_suffix(".json").read_text())
    assert meta["episodes"] == 6 and meta["transitions"] > 0


def test_evaluation_deterministic(tmp_path):
    from latent_sokoban.evaluation import evaluate_split
    from latent_sokoban.agent import RandomAgent

    rng = np.random.default_rng(2)
    levels = []
    for _ in range(4):
        level, sol = generate_level(rng)
        levels.append({"ascii": level.to_ascii(), "optimal_len": len(sol),
                       "max_steps": 40})
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"name": "test", "max_steps": 40, "levels": levels}))

    s1, _ = evaluate_split(RandomAgent(seed=0), split, seed=0)
    s2, _ = evaluate_split(RandomAgent(seed=0), split, seed=0)
    # everything except wall-clock timing must be bit-identical
    import dataclasses
    d1, d2 = dataclasses.asdict(s1), dataclasses.asdict(s2)
    d1.pop("avg_plan_time_ms"), d2.pop("avg_plan_time_ms")
    assert d1 == d2
    assert s1.episodes == 4


def test_call_budget_enforced(tmp_path):
    from latent_sokoban.evaluation import evaluate_split
    from latent_sokoban.agent import Agent
    from latent_sokoban.solver import bfs_solve

    level = Level.from_ascii(SIMPLE)
    solution = bfs_solve(level)
    split = tmp_path / "split.json"
    split.write_text(json.dumps({"name": "test", "max_steps": 40, "levels": [
        {"ascii": level.to_ascii(), "optimal_len": len(solution), "max_steps": 40}]}))

    class OracleAgent(Agent):
        """Plays the optimal solution while ticking a fixed call count."""

        def __init__(self, calls_per_action):
            super().__init__()
            self.calls_per_action = calls_per_action

        def reset(self):
            self.plan = list(solution)

        def act(self, obs, goal, history):
            self.call_meter.tick(self.calls_per_action)
            return self.plan.pop(0)

    # within budget: solves
    s, eps = evaluate_split(OracleAgent(100), split, call_cap=256)
    assert s.success_rate == 1.0 and s.call_violations == 0
    assert s.avg_model_calls == 100.0 and s.max_model_calls == 100

    # over budget, strict: episode fails
    s, eps = evaluate_split(OracleAgent(1000), split, call_cap=256)
    assert s.success_rate == 0.0 and s.call_violations == 1

    # over budget, lenient: solves but violation is recorded
    s, eps = evaluate_split(OracleAgent(1000), split, call_cap=256,
                            strict_calls=False)
    assert s.success_rate == 1.0 and s.call_violations == 1

    # meter is fresh per episode (not cumulative across episodes)
    two = {"name": "t2", "max_steps": 40, "levels": [
        {"ascii": level.to_ascii(), "optimal_len": len(solution), "max_steps": 40},
        {"ascii": level.to_ascii(), "optimal_len": len(solution), "max_steps": 40}]}
    split.write_text(json.dumps(two))
    s, eps = evaluate_split(OracleAgent(200), split, call_cap=256)
    assert s.success_rate == 1.0 and s.max_model_calls == 200


def test_multibox_generation_and_solving():
    rng = np.random.default_rng(0)
    level, solution = generate_level(rng, size=8, n_boxes=3, wall_density=0.10,
                                     min_solution_len=10, max_solution_len=50)
    env = SokobanEnv(level, max_steps=80)
    for a in solution:
        env.step(a)
    assert env.solved
    assert render(level).shape == (64, 64, 3)


def test_score_computation():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from score import compute_score, planning_speed_score

    assert planning_speed_score(10) == 1.0
    assert planning_speed_score(75) == 0.8
    assert planning_speed_score(150) == 0.6
    assert planning_speed_score(300) == 0.4
    assert planning_speed_score(900) == 0.0

    def split(sr, me=0.5, ms=40.0, dl=0.1):
        return {"success_rate": sr, "move_efficiency": me,
                "avg_plan_time_ms": ms, "deadlock_rate": dl,
                "call_violations": 0}

    results = {"agent": "x", "splits": {
        "split_A": split(0.8), "split_B": split(0.6),
        "split_C": split(0.4), "split_D": split(0.5)}}
    score = compute_score(results, reproducible=True)
    # 45*.8 + 20*.5 + 10*.5 + 10*1.0 + 10*.5 + 5*1 = 71.0
    assert score["final_score"] == 71.0
    assert score["components"]["G"] == 0.5

    with pytest.raises(SystemExit):
        compute_score({"splits": {"split_A": split(1.0)}}, False)


def test_viz_helpers(tmp_path):
    from latent_sokoban.viz import contact_sheet, write_png

    frames = [np.full((64, 64, 3), v, dtype=np.uint8) for v in (0, 128, 255)]
    sheet = contact_sheet(frames, cols=2, gap=4, scale=2)
    assert sheet.shape == ((2 * 64 + 3 * 4) * 2, (2 * 64 + 3 * 4) * 2, 3)
    path = tmp_path / "sheet.png"
    write_png(path, sheet)
    assert path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_scripts_run(tmp_path):
    root = Path(__file__).resolve().parents[1]
    env_cmd = dict(PYTHONPATH=str(root))
    import os
    env = {**os.environ, **env_cmd}
    subprocess.run(
        [sys.executable, "scripts/generate_levels.py", "--split", "A",
         "--n", "3", "--seed", "1", "--out", str(tmp_path / "a.json")],
        cwd=root, env=env, check=True, capture_output=True)
    subprocess.run(
        [sys.executable, "scripts/generate_dataset.py", "--out",
         str(tmp_path / "data"), "--episodes", "4", "--shard-size", "4",
         "--seed", "1"],
        cwd=root, env=env, check=True, capture_output=True)
    out = subprocess.run(
        [sys.executable, "scripts/evaluate.py", "--agent", "random",
         "--splits", str(tmp_path / "a.json"), "--seeds", "0",
         "--out", str(tmp_path / "results.json")],
        cwd=root, env=env, check=True, capture_output=True, text=True)
    assert "success rate" in out.stdout
    report = json.loads((tmp_path / "results.json").read_text())
    assert "split_A" in report["splits"] or "split_a" in report["splits"] \
        or list(report["splits"])


def test_baseline_regularizer_averages_both_latents():
    """The anti-collapse term must cover the next observation too.

    `_var_cov_loss(z) + _var_cov_loss(z_next)` returns tuples, so + once
    concatenated them; indexing [0] and [1] kept z's terms at half weight
    and dropped z_next's entirely, quietly halving var_coef.
    """
    torch = pytest.importorskip("torch", reason="baseline extras not installed")
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from baseline.model import WorldModel, _var_cov_loss

    torch.manual_seed(0)
    model = WorldModel()
    obs = torch.rand(32, 3, 64, 64)
    nxt = torch.rand(32, 3, 64, 64)
    act = torch.randint(0, 4, (32,))

    with torch.no_grad():
        z, z_next = model.encoder(obs), model.encoder(nxt)
        var_z, cov_z = _var_cov_loss(z)
        var_n, cov_n = _var_cov_loss(z_next)
        out = model.loss(obs, act, nxt)

    assert torch.allclose(out["var"], (var_z + var_n) / 2)
    assert torch.allclose(out["cov"], (cov_z + cov_n) / 2)
    # the old behaviour, pinned so it cannot come back
    assert not torch.allclose(out["var"], var_z / 2)
