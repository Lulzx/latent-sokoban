"""Train a policy on the Sokoban Ocean env with PufferLib's PPO.

    python csrc/puffer/setup.py build_ext --inplace
    python csrc/puffer/train.py --timesteps 2_000_000

The env lives outside the pufferlib package, so `puffer train` cannot find it
through the registry. pufferl.train() accepts a prebuilt vecenv and policy,
which is the supported way in: config is assembled here from pufferlib's
default.ini plus config/sokoban.ini, rather than by writing our own ini into
site-packages.

Budget note: docs/RULES.md caps a submitted entry at 2 million environment
transitions. Development runs are unrestricted, so --timesteps defaults high;
pass 2_000_000 for a run you intend to submit.
"""

import argparse
import configparser
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

import pufferlib
import pufferlib.pufferl as pufferl
import pufferlib.vector
from sokoban import IMG, OBS_BYTES, Sokoban


class SokobanPolicy(nn.Module):
    """Conv encoder over the (observation, goal) pair.

    The observation is two stacked 64x64x3 renders, so it is reshaped to
    6x64x64 and run through the same four-layer stack the shared baseline
    uses (baseline/model.py). Keeping the architecture aligned means a
    result here is comparable with the distilled and beam-search agents
    rather than confounded by a different encoder.
    """

    def __init__(self, env, hidden_size=256):
        super().__init__()
        self.hidden_size = hidden_size
        self.is_continuous = False
        self.conv = nn.Sequential(
            nn.Conv2d(6, 32, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, 4, stride=2, padding=1), nn.ReLU(),
            nn.Flatten(),
        )
        self.proj = nn.Sequential(nn.Linear(128 * 4 * 4, hidden_size), nn.ReLU())
        n_actions = env.single_action_space.n
        self.actor = nn.Linear(hidden_size, n_actions)
        self.value = nn.Linear(hidden_size, 1)

    def encode_observations(self, observations, state=None):
        x = observations.reshape(-1, 2, IMG, IMG, 3).float() / 255.0
        # (B, 2, H, W, C) -> (B, 2*C, H, W): current frame then goal frame
        x = x.permute(0, 1, 4, 2, 3).reshape(-1, 6, IMG, IMG)
        return self.proj(self.conv(x)), None

    def decode_actions(self, hidden, lookup=None):
        return self.actor(hidden), self.value(hidden)

    def forward(self, observations, state=None):
        hidden, _ = self.encode_observations(observations)
        return self.decode_actions(hidden)

    def forward_eval(self, observations, state=None):
        return self.forward(observations, state)


def build_args(timesteps, device):
    """Merge pufferlib's default.ini with config/sokoban.ini."""
    puffer_dir = Path(os.path.dirname(pufferlib.__file__))
    parser = configparser.ConfigParser()
    parser.read([puffer_dir / "config" / "default.ini",
                 HERE.parents[1] / "config" / "sokoban.ini"])

    def typed(v):
        v = v.split(";")[0].strip()
        if v in ("True", "False"):
            return v == "True"
        if v == "auto":
            return v
        try:
            return int(v.replace("_", ""))
        except ValueError:
            pass
        try:
            return float(v)
        except ValueError:
            return v

    out = {s: {k: typed(v) for k, v in parser[s].items()} for s in parser.sections()}
    out["train"]["total_timesteps"] = timesteps
    out["train"]["device"] = device
    # load_config() derives this from rnn_name (pufferl.py:1179). SokobanPolicy
    # is feedforward, so no LSTM state is threaded through the rollout.
    out["train"]["use_rnn"] = False
    out.setdefault("vec", {})
    out["vec"].setdefault("backend", "Serial")
    # load_config() normally injects these from its argparse layer; building
    # the config directly means supplying them here.
    out.update({
        "neptune": False, "wandb": False, "tag": None,
        "load_model_path": None, "load_id": None, "max_runs": 200,
        "wandb_project": "pufferlib", "wandb_group": "debug",
        "neptune_name": "pufferai", "neptune_project": "ablations",
        "render_mode": "None", "save_frames": 0, "gif_path": "eval.gif",
        "fps": 15, "local_rank": 0,
    })
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--timesteps", type=int, default=5_000_000)
    ap.add_argument("--num-envs", type=int, default=256)
    ap.add_argument("--pool-size", type=int, default=1024)
    ap.add_argument("--n-crates", type=int, default=1)
    ap.add_argument("--device", default=(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"))
    ap.add_argument("--optimizer", default=None,
                    help="override; muon comes from heavyball and is CUDA-"
                         "oriented, so mps/cpu fall back to adam")
    args = ap.parse_args()

    cfg = build_args(args.timesteps, args.device)
    # batch_size is 'auto', derived as num_envs * bptt_horizon. A minibatch
    # larger than that is rejected outright, so scale it to the env count
    # rather than making the caller work the arithmetic out.
    batch = args.num_envs * cfg["train"]["bptt_horizon"]
    cfg["train"]["minibatch_size"] = min(cfg["train"]["minibatch_size"], batch)
    # default.ini asks for muon, which comes from heavyball and is written for
    # CUDA. Off CUDA, fall back to adam rather than fail deep inside the
    # optimizer on the first update.
    if args.optimizer:
        cfg["train"]["optimizer"] = args.optimizer
    elif args.device != "cuda":
        cfg["train"]["optimizer"] = "adam"
    env_kwargs = dict(board_size=8, n_crates=args.n_crates,
                      pool_size=args.pool_size, max_steps=30,
                      min_len=4, max_len=20, wall_density=10)

    def make_env(**kwargs):
        return Sokoban(num_envs=args.num_envs, **kwargs)

    vecenv = pufferlib.vector.make(make_env, env_kwargs=env_kwargs,
                                   backend=pufferlib.vector.Serial)
    policy = SokobanPolicy(vecenv.driver_env).to(args.device)
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"policy parameters: {n_params:,} (RULES.md cap 20,000,000)")

    pufferl.train("puffer_sokoban", args=cfg, vecenv=vecenv, policy=policy)


if __name__ == "__main__":
    main()
