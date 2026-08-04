"""Generate the hidden evaluation set: 100 levels on a difficulty ramp.

The ramp itself lives in latent_sokoban.levels.generate_hidden_set, so the
server and this script cannot drift apart on what the hidden set is.

Run this rather than letting the server generate on first boot when you care
about startup latency: generation takes roughly 70s, almost all of it in the
4-crate tier.

Usage:
    python scripts/generate_hidden.py --seed $SOKOBAN_SEED --out hidden_levels.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latent_sokoban.levels import generate_hidden_set


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    t0 = time.time()

    def tick(n: int, crates: int, optimal: int) -> None:
        if not args.quiet and n % 10 == 0:
            print(f"  {n:3d}/100  crates={crates} optimal={optimal} "
                  f"({time.time() - t0:.0f}s)", file=sys.stderr)

    data = generate_hidden_set(args.seed, progress=tick)
    Path(args.out).write_text(json.dumps(data))

    lens = [lv["optimal_len"] for lv in data["levels"]]
    print(f"wrote {args.out}: {len(lens)} levels, "
          f"optimal {min(lens)}-{max(lens)} moves, "
          f"total step budget {sum(lv['max_steps'] for lv in data['levels'])}")


if __name__ == "__main__":
    main()
