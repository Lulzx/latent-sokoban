"""Deterministic evaluation harness.

Runs an Agent on a benchmark split (a JSON file of levels produced by
scripts/generate_levels.py) and reports the official metrics:

  success_rate           solved within the action limit  (primary metric)
  move_efficiency        optimal_moves / agent_moves, solved levels only
  avg_plan_time_ms       wall-clock time per executed action
  deadlock_rate          episodes that entered a provably dead state
  avg_steps_solved       actions used on solved levels

Determinism: level order, themes and observation noise all derive from
the split file and the evaluation seed. Two runs of the same checkpoint
with the same seed must produce identical results (agents must seed their
own samplers from a fixed constant, or accept the reproducibility rules).
"""

from __future__ import annotations

import importlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from latent_sokoban.agent import Agent
from latent_sokoban.env import Level, SokobanEnv
from latent_sokoban.render import Theme, default_theme, render, render_goal
from latent_sokoban.solver import state_is_dead


@dataclass
class EpisodeResult:
    solved: bool
    steps: int
    optimal: int
    deadlocked: bool
    plan_times_ms: list[float] = field(default_factory=list)


@dataclass
class SplitResult:
    split: str
    episodes: int
    success_rate: float
    move_efficiency: float
    avg_plan_time_ms: float
    deadlock_rate: float
    avg_steps_solved: float


def load_split(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def theme_from_dict(d: dict | None) -> Theme:
    if not d:
        return default_theme()
    d = dict(d)
    for key in ("floor", "floor_alt", "wall", "wall_edge", "goal",
                "box", "box_edge", "box_on_goal", "player"):
        if key in d:
            d[key] = tuple(d[key])
    return Theme(**d)


def run_episode(
    agent: Agent,
    level: Level,
    theme: Theme,
    max_steps: int,
    noise_rng: np.random.Generator,
) -> EpisodeResult:
    from latent_sokoban.solver import bfs_solve

    optimal = len(bfs_solve(level) or [])
    env = SokobanEnv(level, max_steps=max_steps)
    env.reset()
    agent.reset()
    goal_img = render_goal(level, theme, rng=noise_rng)
    history: list[int] = []
    plan_times: list[float] = []
    deadlocked = False
    done = env.solved

    while not done:
        obs = render(level, env.state, theme, rng=noise_rng)
        t0 = time.perf_counter()
        action = agent.act(obs, goal_img, list(history))
        plan_times.append((time.perf_counter() - t0) * 1000.0)
        if action not in (0, 1, 2, 3):
            raise ValueError(f"agent returned invalid action {action!r}")
        history.append(int(action))
        _, done, info = env.step(int(action))
        if info.pushed and not deadlocked:
            deadlocked = state_is_dead(level, env.state.key())
    return EpisodeResult(env.solved, env.state.steps, optimal, deadlocked, plan_times)


def evaluate_split(
    agent: Agent,
    split_path: str | Path,
    seed: int = 0,
    max_episodes: int | None = None,
) -> tuple[SplitResult, list[EpisodeResult]]:
    split = load_split(split_path)
    noise_rng = np.random.default_rng(seed)
    results = []
    entries = split["levels"][:max_episodes] if max_episodes else split["levels"]
    for entry in entries:
        level = Level.from_ascii(entry["ascii"])
        theme = theme_from_dict(entry.get("theme"))
        max_steps = entry.get("max_steps", split.get("max_steps", 40))
        results.append(run_episode(agent, level, theme, max_steps, noise_rng))

    solved = [r for r in results if r.solved]
    all_times = [t for r in results for t in r.plan_times_ms]
    summary = SplitResult(
        split=split.get("name", Path(split_path).stem),
        episodes=len(results),
        success_rate=len(solved) / len(results) if results else 0.0,
        move_efficiency=(
            float(np.mean([r.optimal / max(r.steps, 1) for r in solved])) if solved else 0.0
        ),
        avg_plan_time_ms=float(np.mean(all_times)) if all_times else 0.0,
        deadlock_rate=float(np.mean([r.deadlocked for r in results])) if results else 0.0,
        avg_steps_solved=float(np.mean([r.steps for r in solved])) if solved else 0.0,
    )
    return summary, results


def load_agent(spec: str) -> Agent:
    """Instantiate an agent from 'module.path:ClassName' or a builtin name."""
    builtins = {"random": "latent_sokoban.agent:RandomAgent"}
    spec = builtins.get(spec, spec)
    module_name, _, class_name = spec.partition(":")
    if not class_name:
        raise ValueError(f"agent spec must be 'module:ClassName', got {spec!r}")
    cls = getattr(importlib.import_module(module_name), class_name)
    return cls()


def results_to_dict(summaries: list[SplitResult]) -> dict:
    return {s.split: asdict(s) for s in summaries}
