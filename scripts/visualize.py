#!/usr/bin/env python3
"""Render episode rollouts as PNG contact sheets.

For each episode: the goal observation first (framed in black), then every
frame of the trajectory in order. A meta.json sidecar records the actions,
per-action planning time and model calls, and the outcome — everything the
live-final display needs, and ready-made figures for technical reports.

    # what does the optimal solution look like?
    python scripts/visualize.py --split levels/eval_w.json --solver --episodes 3 --out viz/

    # what does my agent actually do?
    python scripts/visualize.py --split levels/eval_w.json \
        --agent baseline.agent:BaselineAgent --episodes 3 --out viz/
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latent_sokoban.agent import CallMeter
from latent_sokoban.env import ACTION_NAMES, Level, SokobanEnv
from latent_sokoban.evaluation import load_agent, theme_from_dict
from latent_sokoban.render import render, render_goal
from latent_sokoban.solver import bfs_solve
from latent_sokoban.viz import contact_sheet, write_png


def frame_goal(img: np.ndarray) -> np.ndarray:
    """Black border marks the goal panel."""
    out = img.copy()
    out[:2] = out[-2:] = 0
    out[:, :2] = out[:, -2:] = 0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", required=True)
    parser.add_argument("--agent", default=None, help="'random' or module:Class")
    parser.add_argument("--solver", action="store_true",
                        help="show the optimal solution instead of an agent")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cols", type=int, default=10)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if bool(args.agent) == bool(args.solver):
        parser.error("pass exactly one of --agent or --solver")

    split = json.loads(Path(args.split).read_text())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    noise_rng = np.random.default_rng(args.seed)
    agent = load_agent(args.agent) if args.agent else None

    for i, entry in enumerate(split["levels"][:args.episodes]):
        level = Level.from_ascii(entry["ascii"])
        theme = theme_from_dict(entry.get("theme"))
        max_steps = entry.get("max_steps", split.get("max_steps", 40))
        env = SokobanEnv(level, max_steps=max_steps)
        env.reset()
        goal_img = render_goal(level, theme, rng=noise_rng)
        frames = [frame_goal(goal_img), render(level, env.state, theme, rng=noise_rng)]
        actions, plan_ms, calls = [], [], []

        if args.solver:
            for a in bfs_solve(level) or []:
                env.step(a)
                actions.append(int(a))
                frames.append(render(level, env.state, theme, rng=noise_rng))
        else:
            agent.call_meter = meter = CallMeter()
            agent.reset()
            done = env.solved
            while not done:
                obs = render(level, env.state, theme, rng=noise_rng)
                before = meter.total
                t0 = time.perf_counter()
                a = agent.act(obs, goal_img, list(actions))
                plan_ms.append(round((time.perf_counter() - t0) * 1000, 2))
                calls.append(meter.total - before)
                actions.append(int(a))
                _, done, _ = env.step(int(a))
                frames.append(render(level, env.state, theme, rng=noise_rng))

        name = f"ep{i:03d}"
        write_png(out_dir / f"{name}.png", contact_sheet(frames, cols=args.cols))
        (out_dir / f"{name}.json").write_text(json.dumps({
            "level": entry["ascii"],
            "optimal_len": entry.get("optimal_len"),
            "actions": [ACTION_NAMES[a] for a in actions],
            "solved": env.solved,
            "steps": len(actions),
            "plan_times_ms": plan_ms,
            "model_calls": calls,
        }, indent=1))
        status = "solved" if env.solved else "failed"
        print(f"{name}: {status} in {len(actions)} steps -> {out_dir / name}.png")


if __name__ == "__main__":
    main()
