"""Trajectory generation and the shared dataset format.

Dataset format (one .npz shard + sidecar .json metadata):

    frames          uint8  (F, 64, 64, 3)   all frames, episodes concatenated
    actions         int8   (F,)             action taken FROM frame i; -1 on
                                            the final frame of each episode
    episode_starts  int64  (E,)             index of each episode's first frame
    episode_lens    int64  (E,)             number of frames (T+1) per episode
    goal_frames     uint8  (E, 64, 64, 3)   goal observation per episode
    pushed          bool   (F,)             transition from frame i pushed a box
    invalid         bool   (F,)             transition from frame i was a no-op
    solved          bool   (E,)             episode ended solved
    kind            int8   (E,)             0 random, 1 solver, 2 perturbed
    levels          str    (E,)             ascii level definitions

The transition (frames[i], actions[i], frames[i+1]) is a valid training
tuple whenever actions[i] != -1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from latent_sokoban.env import Level, SokobanEnv
from latent_sokoban.render import Theme, render, render_goal
from latent_sokoban.solver import bfs_solve

KIND_RANDOM, KIND_SOLVER, KIND_PERTURBED = 0, 1, 2


def rollout(
    env: SokobanEnv,
    actions: list[int],
    theme: Theme | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Execute actions from reset, rendering every frame."""
    env.reset()
    frames = [render(env.level, env.state, theme, rng=rng)]
    pushed, invalid, taken = [], [], []
    for a in actions:
        _, done, info = env.step(a)
        frames.append(render(env.level, env.state, theme, rng=rng))
        taken.append(a)
        pushed.append(info.pushed)
        invalid.append(info.invalid)
        if done:
            break
    return {
        "frames": np.stack(frames),
        "actions": np.array(taken, dtype=np.int8),
        "pushed": np.array(pushed, dtype=bool),
        "invalid": np.array(invalid, dtype=bool),
        "solved": env.solved,
    }


def random_actions(rng: np.random.Generator, length: int) -> list[int]:
    return [int(a) for a in rng.integers(0, 4, size=length)]


def perturbed_solution(
    rng: np.random.Generator, solution: list[int], n_perturb: int = 3
) -> list[int]:
    """Insert random detour actions into an optimal solution. The episode
    may or may not still solve — both outcomes are useful signal."""
    actions = list(solution)
    for _ in range(n_perturb):
        pos = int(rng.integers(0, len(actions) + 1))
        actions.insert(pos, int(rng.integers(0, 4)))
    return actions


def generate_shard(
    rng: np.random.Generator,
    n_episodes: int,
    size: int = 6,
    n_boxes: int = 1,
    max_steps: int = 40,
    mix: tuple[float, float, float] = (0.5, 0.3, 0.2),
    theme: Theme | None = None,
) -> dict:
    """Generate one dataset shard with the recommended composition:
    50% random / 30% solver / 20% perturbed-solver trajectories."""
    from latent_sokoban.levels import generate_level

    episodes = []
    kinds = rng.choice(3, size=n_episodes, p=list(mix))
    for kind in kinds:
        level, solution = generate_level(rng, size=size, n_boxes=n_boxes)
        env = SokobanEnv(level, max_steps=max_steps)
        if kind == KIND_RANDOM:
            actions = random_actions(rng, max_steps)
        elif kind == KIND_SOLVER:
            actions = solution
        else:
            actions = perturbed_solution(rng, solution)
        ep = rollout(env, actions, theme, rng=rng)
        ep["kind"] = int(kind)
        ep["level"] = level.to_ascii()
        ep["goal_frame"] = render_goal(level, theme, rng=rng)
        episodes.append(ep)
    return pack_episodes(episodes)


def pack_episodes(episodes: list[dict]) -> dict:
    frames, actions, pushed, invalid = [], [], [], []
    starts, lens = [], []
    cursor = 0
    for ep in episodes:
        f = ep["frames"]
        t = len(ep["actions"])
        starts.append(cursor)
        lens.append(len(f))
        cursor += len(f)
        frames.append(f)
        actions.append(np.concatenate([ep["actions"], [-1]]).astype(np.int8))
        pushed.append(np.concatenate([ep["pushed"], [False]]))
        invalid.append(np.concatenate([ep["invalid"], [False]]))
    return {
        "frames": np.concatenate(frames),
        "actions": np.concatenate(actions),
        "pushed": np.concatenate(pushed).astype(bool),
        "invalid": np.concatenate(invalid).astype(bool),
        "episode_starts": np.array(starts, dtype=np.int64),
        "episode_lens": np.array(lens, dtype=np.int64),
        "goal_frames": np.stack([ep["goal_frame"] for ep in episodes]),
        "solved": np.array([ep["solved"] for ep in episodes], dtype=bool),
        "kind": np.array([ep["kind"] for ep in episodes], dtype=np.int8),
        "levels": np.array([ep["level"] for ep in episodes]),
    }


def save_shard(shard: dict, path: str | Path, meta: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **shard)
    n_transitions = int((shard["actions"] != -1).sum())
    sidecar = {
        "episodes": int(len(shard["episode_starts"])),
        "frames": int(len(shard["frames"])),
        "transitions": n_transitions,
        "solved_fraction": float(shard["solved"].mean()),
        **(meta or {}),
    }
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2))


def load_shard(path: str | Path) -> dict:
    with np.load(path, allow_pickle=False) as data:
        return {k: data[k] for k in data.files}
