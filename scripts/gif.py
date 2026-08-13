#!/usr/bin/env python3
"""Render solved episodes as animated GIFs.

Board only, one frame per action, with the step number and the last move
drawn in the corner. Two modes:

    # how the agent actually solves
    python scripts/gif.py --split levels/eval_s.json \
        --agent wm.agent:WMAgent --out viz/gifs --max-gifs 4

    # the optimal solution (solver, for levels the agent cannot yet solve)
    python scripts/gif.py --split levels/split_a.json \
        --solver --out viz/gifs --max-gifs 4 --scale 8

Only solved episodes are kept. `--scale` is the nearest-neighbour upscale of
the 64x64 observation; 8 gives 512px crisp pixel art.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image, ImageDraw, ImageFont

from latent_sokoban.agent import CallMeter
from latent_sokoban.env import ACTION_NAMES, Level, SokobanEnv
from latent_sokoban.evaluation import load_agent, theme_from_dict
from latent_sokoban.render import render, render_goal
from latent_sokoban.solver import bfs_solve

ARROWS = {"up": "UP", "down": "DOWN", "left": "LEFT", "right": "RIGHT"}


def frame_img(obs: np.ndarray, step: int, action: str | None,
              scale: int) -> Image.Image:
    img = Image.fromarray(obs).resize(
        (obs.shape[1] * scale, obs.shape[0] * scale), Image.NEAREST)
    draw = ImageDraw.Draw(img)
    label = f"step {step:02d}" if action is None else f"step {step:02d}  {action}"
    fs = max(14, scale * 5)
    pad = fs // 3
    try:
        font = ImageFont.load_default(size=fs)
    except TypeError:
        font = ImageFont.load_default()
    tw = draw.textlength(label, font=font) if hasattr(draw, "textlength") else len(label) * fs // 2
    draw.rectangle([pad, pad, pad + int(tw) + pad, pad + fs + pad], fill=(0, 0, 0))
    draw.text((pad + pad // 2, pad), label, fill=(255, 255, 255), font=font)
    return img


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", required=True)
    ap.add_argument("--agent", default=None, help="'random' or module:Class")
    ap.add_argument("--solver", action="store_true",
                    help="show the optimal solution instead of an agent")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-gifs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale", type=int, default=8)
    ap.add_argument("--duration", type=int, default=350, help="ms per frame")
    args = ap.parse_args()
    if bool(args.agent) == bool(args.solver):
        ap.error("pass exactly one of --agent or --solver")

    split = json.loads(Path(args.split).read_text())
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    noise_rng = np.random.default_rng(args.seed)
    agent = load_agent(args.agent) if args.agent else None

    made = 0
    for i, entry in enumerate(split["levels"]):
        if made >= args.max_gifs:
            break
        level = Level.from_ascii(entry["ascii"])
        theme = theme_from_dict(entry.get("theme"))
        max_steps = entry.get("max_steps", split.get("max_steps", 40))
        env = SokobanEnv(level, max_steps=max_steps)
        env.reset()
        goal_img = render_goal(level, theme, rng=noise_rng)

        frames = [render(level, env.state, theme, rng=noise_rng)]
        actions: list[int] = []

        if args.solver:
            sol = bfs_solve(level) or []
            for a in sol:
                env.step(int(a))
                actions.append(int(a))
                frames.append(render(level, env.state, theme, rng=noise_rng))
        else:
            agent.call_meter = CallMeter()
            agent.reset()
            done = env.solved
            while not done:
                obs = render(level, env.state, theme, rng=noise_rng)
                a = int(agent.act(obs, goal_img, list(actions)))
                actions.append(a)
                _, done, _ = env.step(a)
                frames.append(render(level, env.state, theme, rng=noise_rng))

        if not env.solved:
            print(f"ep{i:03d}: failed, skipping")
            continue

        made += 1
        imgs = []
        for t, f in enumerate(frames):
            action = ARROWS[ACTION_NAMES[actions[t - 1]]] if t > 0 else None
            imgs.append(frame_img(f, t, action, args.scale))
        path = out_dir / f"solved_{made:02d}.gif"
        imgs[0].save(path, save_all=True, append_images=imgs[1:],
                     duration=args.duration, loop=0)
        print(f"ep{i:03d}: solved in {len(actions)} steps -> {path}")

    print(f"\n{made} gif(s) in {out_dir}")


if __name__ == "__main__":
    main()
