#!/usr/bin/env python3
"""Official evaluation script.

Runs an agent over one or more benchmark splits, averages across
evaluation seeds, and writes results.json. Both competitors are scored by
this exact script on the same split files.

Usage:
    # sanity check with the built-in random agent
    python scripts/evaluate.py --agent random --splits levels/split_a.json

    # a real submission
    python scripts/evaluate.py \
        --agent my_submission.agent:MyAgent \
        --splits levels/split_a.json levels/split_b.json \
                 levels/split_c.json levels/split_d.json \
        --seeds 0 1 2 --out results.json
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from latent_sokoban.evaluation import evaluate_split, load_agent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True,
                        help="'random' or 'module.path:ClassName'")
    parser.add_argument("--splits", nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--max-episodes", type=int, default=None,
                        help="cap episodes per split (debugging only)")
    parser.add_argument("--call-cap", type=int, default=256,
                        help="max counted dynamics calls per action (0 = uncapped)")
    parser.add_argument("--no-strict-calls", action="store_true",
                        help="report call-cap violations without failing episodes")
    parser.add_argument("--out", default=None, help="write results JSON here")
    args = parser.parse_args()

    call_cap = args.call_cap or None
    report: dict = {"agent": args.agent, "seeds": args.seeds,
                    "call_cap": call_cap, "strict_calls": not args.no_strict_calls,
                    "splits": {}}
    for split_path in args.splits:
        per_seed = []
        for seed in args.seeds:
            agent = load_agent(args.agent)
            summary, _ = evaluate_split(agent, split_path, seed=seed,
                                        max_episodes=args.max_episodes,
                                        call_cap=call_cap,
                                        strict_calls=not args.no_strict_calls)
            per_seed.append(summary)
        name = per_seed[0].split
        metrics = {}
        for field in ("success_rate", "move_efficiency", "avg_plan_time_ms",
                      "avg_model_calls", "deadlock_rate", "avg_steps_solved"):
            values = [getattr(s, field) for s in per_seed]
            metrics[field] = float(np.mean(values))
            metrics[field + "_std"] = float(np.std(values))
        metrics["max_model_calls"] = max(s.max_model_calls for s in per_seed)
        metrics["call_violations"] = sum(s.call_violations for s in per_seed)
        metrics["episodes"] = per_seed[0].episodes
        report["splits"][name] = metrics

        print(f"\n== {name} ({metrics['episodes']} levels, "
              f"{len(args.seeds)} seeds) ==")
        print(f"  success rate     {metrics['success_rate']:.3f} "
              f"± {metrics['success_rate_std']:.3f}")
        print(f"  move efficiency  {metrics['move_efficiency']:.3f}")
        print(f"  plan time        {metrics['avg_plan_time_ms']:.1f} ms/action")
        print(f"  model calls      {metrics['avg_model_calls']:.0f}/action "
              f"(max {metrics['max_model_calls']}, "
              f"violations {metrics['call_violations']})")
        print(f"  deadlock rate    {metrics['deadlock_rate']:.3f}")

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    main()
