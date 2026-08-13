# Threat model & hardening notes

Companion to `lab/redteam.py`. This documents the ways the benchmark can be
gamed and what the server can actually do about it. It is analysis-only:
none of this is an entry, and several of these are exactly what
`docs/RULES.md` prohibits.

## The threats, in order of severity

### 1. Decode-then-search (proven)

The default renderer is deterministic and tile-aligned: at the hidden set's
8×8 board, `render.py` draws exactly 8 pixels per tile, and every tile type
owns a unique interior colour (wall `(94,84,74)`, box `(176,122,54)`,
box-on-goal `(206,160,70)`, player `(48,108,188)`, floor/floor-alt, and the
goal marker is a fixed ring `(196,60,60)`). Two pixels per tile (centre +
ring) reconstruct the full grid. `lab/redteam.py` does this in ~40 lines,
feeds the result to the shipped BFS, and replays:

```
split_S  1.000  eff 1.000  calls 0
split_A  1.000  eff 1.000  calls 0
split_D  1.000  eff 1.000  calls 0
```

Zero dynamics calls, so the call meter never trips. The only thing that
catches it is source review. It is not "almost" decoding: it is decoding,
and it solves the whole hidden set on the first scorecard.

### 2. Probe-and-memorize (the fixed set)

The hidden set is deterministic and fixed until rotation. One scorecard
reveals all 100 initial observations and goals. A cheater can decode them
locally (threat 1), BFS every level, store the `(obs, goal) -> solution`
map, and replay on the next scorecard. The 24-scorecards/day cap only slows
this; it does not stop it, because one run already suffices.

### 3. Owner-only replay is not a secret in practice

Replay action strings are `owner_view: true`, which is meant to stop one
person's perfect score being copied by everyone. But threat 1 makes that
defence moot: anyone can produce their own perfect action string in one run.

### 4. Metering dishonesty (dynamics-call under-reporting)

`docs/RULES.md` already flags this: an agent can tick the `CallMeter` less
than it actually calls its dynamics model (or not at all, as the decode
agent does legitimately-because-it-has-no-model). HTTP cannot observe this;
source review is the only check.

## What actually helps

| Mitigation | Effect | Cost |
| --- | --- | --- |
| Rotate the hidden set frequently | Kills memorization (threat 2), and re-arms threat 1 against any precomputed maps | Already "between rounds"; do it at least every round |
| Source review for prize/headline claims | The only thing that stops threats 1 and 4 | Already stated; make it the actual gate for any headline number |
| Mild per-episode noise / colour jitter (`noise_std > 0`) | Raises the decode bar: a colour-LUT decoder breaks; a conv encoder with light augmentation still works | Breaks the exact 8px/tile prior `wm/` leans on; a median-filter decoder still recovers the grid at `noise_std` up to ~6, so it deters but does not stop |

### What does not work

- **The call meter.** A decoder makes 0 calls and is still (trivially) within budget. Padding calls to look honest is equally trivial.
- **Rate limits.** They bound probe throughput, not a single-run solve.
- **"No module whose output is an exact symbolic board state" as written.** The rule is a legal line, not a technical one; nothing over HTTP can tell a 48-channel 8×8 feature map (`wm/`) from a one-hot tile map, and that ambiguity is where a reviewer will need a written precedent.

## The honest conclusion

Given a deterministic, tile-aligned, state-recoverable renderer, decode-
then-search is unstoppable over HTTP: it is ~40 lines and scores ~100%.
The server can only (a) rotate the set to stop memorization, (b) add noise
to make decoding noisier (not impossible), and (c) treat the leaderboard as
honor-tier with source review as the real enforcement for any claim that
matters. The feature that makes decode trivial — the exact 8px/tile
alignment that keeps a 64×64 observation meaning the same tile scale
everywhere — is the same feature the spatial world model depends on for its
sharp prior, so there is no free hardening here.

`lab/redteam.py` doubles as the regression test: a mitigation is working
only when its success rate on the affected split collapses toward random.
