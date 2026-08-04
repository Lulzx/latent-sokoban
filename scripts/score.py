#!/usr/bin/env python3
"""Official final-score calculator.

Consumes the results.json written by scripts/evaluate.py (run over splits
A-D) and computes the 100-point competition score:

    Final = 45*S + 20*G + 10*M + 10*P + 10*D + 5*R

    S  standard success        success_rate on Split A
    G  generalization          mean success_rate of Splits B and C
    M  move efficiency         move_efficiency on Split A (solved levels)
    P  planning speed          tiered from mean plan time across A-D:
                               <50ms: 1.0, <100ms: 0.8, <200ms: 0.6,
                               <500ms: 0.4, otherwise 0.0
    D  deadlock avoidance      success_rate on Split D
    R  reproducibility         0 or 1, asserted by the judge with
                               --reproducible after rerunning the
                               submission from its README

Both competitors are scored by this exact script on the same results
format. Usage:

    python scripts/score.py results.json --reproducible --out score.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REQUIRED = ["split_A", "split_B", "split_C", "split_D"]


def planning_speed_score(ms: float) -> float:
    for limit, points in ((50, 1.0), (100, 0.8), (200, 0.6), (500, 0.4)):
        if ms < limit:
            return points
    return 0.0


def compute_score(results: dict, reproducible: bool) -> dict:
    splits = results["splits"]
    missing = [s for s in REQUIRED if s not in splits]
    if missing:
        raise SystemExit(
            f"results are missing {missing}; run scripts/evaluate.py over "
            f"all of {REQUIRED} first")

    s = splits["split_A"]["success_rate"]
    g = (splits["split_B"]["success_rate"] + splits["split_C"]["success_rate"]) / 2
    m = splits["split_A"]["move_efficiency"]
    mean_ms = sum(splits[k]["avg_plan_time_ms"] for k in REQUIRED) / len(REQUIRED)
    p = planning_speed_score(mean_ms)
    d = splits["split_D"]["success_rate"]
    r = 1.0 if reproducible else 0.0

    violations = sum(splits[k].get("call_violations", 0) for k in REQUIRED)

    components = {"S": s, "G": g, "M": m, "P": p, "D": d, "R": r}
    final = 45 * s + 20 * g + 10 * m + 10 * p + 10 * d + 5 * r
    return {
        "agent": results.get("agent"),
        "components": components,
        "mean_plan_time_ms": mean_ms,
        "call_violations": violations,
        "final_score": round(final, 2),
        # tie-break keys, in rulebook order (higher-is-better sign applied)
        "tie_break": {
            "total_success": sum(splits[k]["success_rate"] for k in REQUIRED),
            "structural_generalization": splits["split_C"]["success_rate"],
            "deadlock_rate": -sum(splits[k]["deadlock_rate"] for k in REQUIRED),
            "plan_time": -mean_ms,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", help="results.json from scripts/evaluate.py")
    parser.add_argument("--reproducible", action="store_true",
                        help="judge confirms the submission retrained/reran "
                             "successfully from its own instructions")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text())
    score = compute_score(results, args.reproducible)

    print(f"agent: {score['agent']}")
    for name, label, weight in (("S", "standard success", 45),
                                ("G", "generalization", 20),
                                ("M", "move efficiency", 10),
                                ("P", "planning speed", 10),
                                ("D", "deadlock avoidance", 10),
                                ("R", "reproducibility", 5)):
        v = score["components"][name]
        print(f"  {name}  {label:<20} {v:.3f}  -> {weight * v:6.2f} / {weight}")
    if score["call_violations"]:
        print(f"  !! {score['call_violations']} call-budget violations "
              f"(episodes already failed by the harness)")
    print(f"\nFINAL SCORE: {score['final_score']} / 100")

    if args.out:
        Path(args.out).write_text(json.dumps(score, indent=2))
        print(f"written to {args.out}")


if __name__ == "__main__":
    main()
