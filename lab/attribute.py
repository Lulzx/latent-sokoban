#!/usr/bin/env python3
"""Attribution lattice: which component is actually losing the episodes?

ANALYSIS ONLY. This script uses the symbolic env and the BFS solver, which
RULES.md permits for "evaluation analysis". Nothing here is an entry; no
oracle component may appear in a submitted agent.

The distilled agent is a composition of four things:

    perception -> latent dynamics -> value heuristic -> beam planner

A single success rate cannot say which of them is the binding constraint.
So run the same planner with components swapped for oracles, one at a time:

    cell                       what it isolates
    ------------------------   ------------------------------------------
    policy                     reactive policy head, ZERO dynamics calls
    learned_dyn/learned_val    the real agent
    oracle_dyn/learned_val     value head + planner with perfect dynamics
    oracle_dyn/oracle_val      the planner/budget ceiling

    C(learned,learned) -> B(oracle,learned)   = cost of dynamics drift
    B(oracle,learned)  -> A(oracle,oracle)    = cost of the value head
    A                  -> 1.0                 = cost of the planner shape

Usage:
    python lab/attribute.py --split levels/eval_s.json --cells all
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from distill.model import DistillModel                       # noqa: E402
from latent_sokoban.env import Level, SokobanEnv, SokobanState  # noqa: E402
from latent_sokoban.evaluation import load_split, theme_from_dict  # noqa: E402
from latent_sokoban.render import render, render_goal        # noqa: E402
from latent_sokoban.solver import bfs_solve, state_is_dead   # noqa: E402
from wm.model import WorldModel                              # noqa: E402

N_ACTIONS = 4
DEAD = 999.0

# distill/ heads return (value, policy); wm/ heads return (value, policy, dead).
# The lattice normalises both to a 3-tuple so every cell is architecture-blind.
ARCHS = {"distill": DistillModel, "wm": WorldModel}


def device_of() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Lab:
    def __init__(self, ckpt: str, beam: int, horizon: int, noise: float,
                 beta: float = 0.0, arch: str = "distill",
                 dead_penalty: float = 0.0):
        self.device = device_of()
        self.arch = arch
        self.model = ARCHS[arch]().to(self.device).eval()
        blob = torch.load(ckpt, map_location="cpu", weights_only=True)
        self.model.load_state_dict(blob["model"])
        self.beam, self.horizon, self.noise = beam, horizon, noise
        self.beta, self.dead_penalty = beta, dead_penalty
        self.calls = 0

    def _heads(self, z: torch.Tensor, z_goal: torch.Tensor):
        """Normalised to (value, logits, dead_logit); dead is None for distill."""
        g = z_goal.expand(z.shape[0], *([-1] * (z_goal.dim() - 1)))
        out = self.model.heads(z, g)
        return out if len(out) == 3 else (out[0], out[1], None)

    def _repeat(self, z: torch.Tensor, n: int) -> torch.Tensor:
        return z.expand(n, *([-1] * (z.dim() - 1)))

    # -- shared helpers ---------------------------------------------------

    @torch.no_grad()
    def encode(self, imgs: np.ndarray) -> torch.Tensor:
        """imgs: (N,64,64,3) uint8. Encoder passes are budget-free."""
        x = torch.from_numpy(np.ascontiguousarray(imgs))
        x = x.permute(0, 3, 1, 2).float().div_(255.0).to(self.device)
        return self.model.encoder(x)

    @torch.no_grad()
    def value(self, z: torch.Tensor, z_goal: torch.Tensor) -> np.ndarray:
        """Plan score: goal distance, penalised by predicted deadlock."""
        v, _, dead = self._heads(z, z_goal)
        if dead is not None and self.dead_penalty:
            v = v + self.dead_penalty * torch.sigmoid(dead)
        return v.cpu().numpy()

    @torch.no_grad()
    def policy_logits(self, z: torch.Tensor, z_goal: torch.Tensor) -> np.ndarray:
        _, logits, _ = self._heads(z, z_goal)
        return logits.cpu().numpy()

    # -- the four cells ---------------------------------------------------

    @torch.no_grad()
    def act_policy(self, obs, z_goal, rng) -> int:
        """Reactive policy head. No dynamics calls at all."""
        z = self.encode(obs[None])
        logits = self.policy_logits(z, z_goal)[0]
        if self.noise > 0:
            logits = logits + rng.gumbel(0.0, self.noise, size=N_ACTIONS)
        return int(np.argmax(logits))

    @torch.no_grad()
    def act_learned(self, obs, z_goal, rng) -> int:
        """Exactly distill/agent.py: beam over imagined latents."""
        z = self.encode(obs[None])
        actions = torch.arange(N_ACTIONS, device=self.device)
        beams = self.model.dynamics(self._repeat(z, N_ACTIONS), actions)
        self.calls += N_ACTIONS
        first = torch.arange(N_ACTIONS, device=self.device)
        scores = torch.from_numpy(self.value(beams, z_goal)).to(self.device)

        for _ in range(self.horizon - 1):
            keep = torch.argsort(scores)[: self.beam]
            beams, first = beams[keep], first[keep]
            expanded = beams.repeat_interleave(N_ACTIONS, dim=0)
            acts = actions.repeat(beams.shape[0])
            beams = self.model.dynamics(expanded, acts)
            self.calls += beams.shape[0]
            first = first.repeat_interleave(N_ACTIONS)
            scores = torch.from_numpy(self.value(beams, z_goal)).to(self.device)

        s = scores.cpu().numpy()
        if self.noise > 0:
            s = s + rng.gumbel(0.0, self.noise, size=len(s))
        return int(first[int(np.argmin(s))].item())

    @torch.no_grad()
    def act_hybrid(self, obs, z_goal, rng) -> int:
        """Beam ranked by value PLUS the policy's log-prob along the path.

        The measured fact this exists to exploit: the policy head picks a
        truly optimal action 0.896 of the time against the value head's
        0.841, and the policy alone outscores the value-ranked beam. So the
        policy proposes and the value only refines. beta=0 is the existing
        value-only beam; beta large collapses to the reactive policy.

        Policy-head passes are not metered, on the same ground the value head
        is not: they score a latent rather than advance one. Only the
        dynamics calls below are ticked.
        """
        z = self.encode(obs[None])
        actions = torch.arange(N_ACTIONS, device=self.device)
        logp = torch.from_numpy(self.policy_logits(z, z_goal)[0]).to(self.device)
        logp = torch.log_softmax(logp, dim=-1)

        beams = self.model.dynamics(self._repeat(z, N_ACTIONS), actions)
        self.calls += N_ACTIONS
        first = torch.arange(N_ACTIONS, device=self.device)
        path = logp.clone()
        scores = torch.from_numpy(self.value(beams, z_goal)).to(self.device) \
            - self.beta * path

        for _ in range(self.horizon - 1):
            keep = torch.argsort(scores)[: self.beam]
            beams, first, path = beams[keep], first[keep], path[keep]
            node_logp = torch.log_softmax(
                torch.from_numpy(self.policy_logits(beams, z_goal)).to(self.device),
                dim=-1)
            expanded = beams.repeat_interleave(N_ACTIONS, dim=0)
            acts = actions.repeat(beams.shape[0])
            beams = self.model.dynamics(expanded, acts)
            self.calls += beams.shape[0]
            first = first.repeat_interleave(N_ACTIONS)
            path = path.repeat_interleave(N_ACTIONS) + node_logp.flatten()
            scores = torch.from_numpy(self.value(beams, z_goal)).to(self.device) \
                - self.beta * path

        s = scores.cpu().numpy()
        if self.noise > 0:
            s = s + rng.gumbel(0.0, self.noise, size=len(s))
        return int(first[int(np.argmin(s))].item())

    def act_oracle_dyn(self, level, key, theme, z_goal, rng, oracle_value):
        """Beam over TRUE successor states. Scoring is the swappable half.

        oracle_value=False: score encoded true renders with the value head,
        so the value head is judged on states it will really be asked about.
        oracle_value=True: score with true BFS distance-to-go = the ceiling.
        """
        frontier = [(key, a) for a in range(N_ACTIONS)]
        frontier = [(SokobanEnv.apply(level, key, a), a) for a in range(N_ACTIONS)]

        def score_keys(keys):
            if oracle_value:
                out = []
                for k in keys:
                    probe = Level(level.walls, level.goals,
                                  frozenset(k[1]), k[0])
                    sol = bfs_solve(probe)
                    out.append(DEAD if sol is None else float(len(sol)))
                return np.array(out)
            imgs = np.stack([
                render(level, SokobanState(frozenset(k[1]), k[0]), theme)
                for k in keys])
            return self.value(self.encode(imgs), z_goal)

        scores = score_keys([k for k, _ in frontier])
        for _ in range(self.horizon - 1):
            order = np.argsort(scores)[: self.beam]
            frontier = [frontier[i] for i in order]
            nxt = []
            for k, a0 in frontier:
                for a in range(N_ACTIONS):
                    nxt.append((SokobanEnv.apply(level, k, a), a0))
            frontier = nxt
            scores = score_keys([k for k, _ in frontier])

        if self.noise > 0:
            scores = scores + rng.gumbel(0.0, self.noise, size=len(scores))
        return frontier[int(np.argmin(scores))][1]


def run_cell(lab: Lab, split: dict, cell: str, seed: int, limit: int | None):
    rng_master = np.random.default_rng(seed)
    entries = split["levels"][:limit] if limit else split["levels"]
    solved = deadlocked = 0
    effs, calls_per_action = [], []
    t0 = time.time()

    for i, entry in enumerate(entries):
        level = Level.from_ascii(entry["ascii"])
        theme = theme_from_dict(entry.get("theme"))
        max_steps = entry.get("max_steps", split.get("max_steps", 40))
        optimal = len(bfs_solve(level) or [])
        env = SokobanEnv(level, max_steps=max_steps)
        env.reset()
        rng = np.random.default_rng(seed * 100_003 + i)
        z_goal = lab.encode(render_goal(level, theme)[None])
        lab.calls = 0
        n_actions = 0
        dead = False
        done = env.solved

        while not done:
            if cell == "policy":
                a = lab.act_policy(render(level, env.state, theme), z_goal, rng)
            elif cell == "learned":
                a = lab.act_learned(render(level, env.state, theme), z_goal, rng)
            elif cell == "hybrid":
                a = lab.act_hybrid(render(level, env.state, theme), z_goal, rng)
            elif cell == "oracle_dyn":
                a = lab.act_oracle_dyn(level, env.state.key(), theme, z_goal,
                                       rng, oracle_value=False)
            elif cell == "oracle_both":
                a = lab.act_oracle_dyn(level, env.state.key(), theme, z_goal,
                                       rng, oracle_value=True)
            else:
                raise SystemExit(f"unknown cell {cell}")
            n_actions += 1
            _, done, info = env.step(int(a))
            if info.pushed and not dead:
                dead = state_is_dead(level, env.state.key())

        if env.solved:
            solved += 1
            effs.append(optimal / max(env.state.steps, 1))
        deadlocked += dead
        calls_per_action.append(lab.calls / max(n_actions, 1))

    n = len(entries)
    return {
        "cell": cell, "episodes": n, "seed": seed,
        "success_rate": solved / n,
        "move_efficiency": float(np.mean(effs)) if effs else 0.0,
        "deadlock_rate": deadlocked / n,
        "calls_per_action": float(np.mean(calls_per_action)),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--split", default=str(REPO / "levels/eval_s.json"))
    ap.add_argument("--ckpt", default=str(REPO / "distill/checkpoint.pt"))
    ap.add_argument("--cells", nargs="+",
                    default=["policy", "learned", "oracle_dyn", "oracle_both"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--beam", type=int, default=16)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--noise", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=1.0,
                    help="policy-prior weight for the 'hybrid' cell")
    ap.add_argument("--arch", choices=sorted(ARCHS), default="distill")
    ap.add_argument("--dead-penalty", type=float, default=0.0,
                    help="weight on the dead head's P(dead) (wm arch only)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    split = load_split(args.split)
    lab = Lab(args.ckpt, args.beam, args.horizon, args.noise, args.beta,
              args.arch, args.dead_penalty)
    rows = []
    print(f"split={Path(args.split).name} arch={args.arch} beam={args.beam} "
          f"horizon={args.horizon} noise={args.noise} beta={args.beta} "
          f"dead_pen={args.dead_penalty} device={lab.device}")
    print(f"{'cell':<14}{'succ':>8}{'eff':>8}{'dead':>8}{'calls':>8}{'sec':>8}")
    for cell in args.cells:
        per_seed = [run_cell(lab, split, cell, s, args.limit) for s in args.seeds]
        succ = np.mean([r["success_rate"] for r in per_seed])
        eff = np.mean([r["move_efficiency"] for r in per_seed])
        dead = np.mean([r["deadlock_rate"] for r in per_seed])
        calls = np.mean([r["calls_per_action"] for r in per_seed])
        sec = sum(r["seconds"] for r in per_seed)
        rows.extend(per_seed)
        print(f"{cell:<14}{succ:>8.3f}{eff:>8.3f}{dead:>8.3f}{calls:>8.0f}{sec:>8.0f}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
