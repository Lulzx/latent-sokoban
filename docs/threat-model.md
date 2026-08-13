# Threat model and hardening

This page documents the ways the benchmark can be gamed and what the server
can actually do about it. It exists so the defences are honest about their
limits. `lab/redteam.py` is the working exploit for the first threat, and
doubles as the regression test a mitigation must defeat.

## The threats, in order of severity

### 1. Decode-then-search (proven)

The default renderer is deterministic and tile-aligned: at the hidden set's
8×8 board, `render.py` draws exactly 8 pixels per tile, and every tile type
owns a unique interior colour. Two pixels per tile (centre + ring)
reconstruct the full grid. `lab/redteam.py` does this in ~40 lines, feeds
the result to the shipped BFS, and replays:

| Split | Success | Move eff. | Dynamics calls |
| --- | --- | --- | --- |
| S | 1.000 | 1.000 | 0 |
| A | 1.000 | 1.000 | 0 |
| D | 1.000 | 1.000 | 0 |

Zero dynamics calls, so the call meter never trips; only source review can
catch it. It is not "almost" decoding — it is decoding, and it solves the
whole hidden set on the first scorecard.

### 2. Probe-and-memorize (the fixed set)

The hidden set is deterministic and fixed until rotation. One scorecard
reveals all 100 initial observations and goals, which a cheater can decode
locally (threat 1), solve, and replay on the next scorecard. The 24
scorecards/day cap slows this but does not stop it: one run already
suffices.

### 3. Metering dishonesty

An agent can under-report its dynamics calls (or report none, as the decode
agent legitimately does, having no model). HTTP cannot observe this; source
review is the only check.

## What actually helps

| Mitigation | Effect | Cost |
| --- | --- | --- |
| Rotate the hidden set frequently | Kills memorization; re-arms threat 1 against any precomputed maps | Already "between rounds"; do it at least every round |
| Source review for prize/headline claims | The only thing that stops threats 1 and 3 | Already stated; make it the real gate for any headline number |
| Mild per-episode noise / colour jitter | Breaks a colour-LUT decoder; a conv encoder with light augmentation still works | Breaks the exact 8px/tile prior `wm/` leans on; a median-filter decoder still recovers the grid, so it deters but does not stop |

What does not work: the call meter (a decoder makes 0 calls and stays in
budget), rate limits (they bound probe throughput, not a single-run solve),
and "no module whose output is an exact symbolic board state" as written —
that is a legal line, not a technical one, since nothing over HTTP can tell
a 48-channel 8×8 feature map from a one-hot tile map.

## The honest conclusion

Given a deterministic, tile-aligned, state-recoverable renderer,
decode-then-search is unstoppable over HTTP. The server can rotate the set,
add noise to make decoding noisier (not impossible), and treat the
leaderboard as honour-tier with source review as the real enforcement for
anything that matters. The feature that makes decode trivial — the exact
8px/tile alignment that keeps a 64×64 observation meaning the same tile
scale everywhere — is the same feature the spatial world model depends on
for its sharp prior, so there is no free hardening here.
