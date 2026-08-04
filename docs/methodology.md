# Methodology

Why the benchmark is built the way it is, and what each design decision is
defending against.

## The claim being tested

> A system can learn the dynamics of an environment from raw pixels and
> plan inside its own learned representation well enough to solve
> irreversible puzzles.

Every rule exists to keep that claim the thing under test. Sokoban is a
good instrument for it because:

- **It is trivial symbolically.** BFS solves these levels instantly. So any
  score achieved by smuggling in symbolic search says nothing, and the
  rules have to close that door explicitly.
- **Mistakes are irreversible.** Push a crate into a corner and the level
  is dead, with no reward signal marking the moment. Model-free approaches
  that rely on recovering from exploration do badly.
- **The state is fully observable but not fully legible.** Everything
  needed is on screen. The difficulty is entirely in representation and
  planning, not in memory or partial observability.

## What the rules defend against

| Rule | The shortcut it closes |
| --- | --- |
| No symbolic solver | Running BFS on a recovered grid |
| No decode-then-search | Learning a pixels-to-grid decoder, then planning symbolically |
| ≤ 256 dynamics calls per action | Brute-forcing the tree instead of learning a heuristic |
| ≤ 20M parameters | Buying performance with scale |
| Hidden levels, server-side | Overfitting to a published test set |
| Unplayed episodes count as failures | Cherry-picking a favourable subset within one run |
| Scorecards counted when opened | Reconnaissance runs that are abandoned rather than closed |
| Replay moves are owner-only | Replaying someone else's solution on a fixed, deterministic set |

The decode-then-search rule is the subtle one. Learning a mapping from
pixels to a discrete board is legitimate representation learning, and it is
also a complete end-run around the benchmark, because once you have the
grid the problem is solved. The line drawn is about what the *planner*
consumes: planning must happen in a learned, continuous representation, not
over a reconstructed symbolic board.

Compliance with that is verified by source review at submission, not
automatically. It is a benchmark of good-faith research, and the rules say
so plainly.

## Why the hidden set stays hidden

Levels are generated once from a secret seed and never leave the server.
Agents receive rendered frames only.

This is not secrecy for its own sake. It is the only way to keep the test
set meaningful across a long-running public leaderboard: any published set
gets trained against, deliberately or otherwise, and stops measuring
generalisation.

The seed makes it auditable anyway. Publish the seed when a round closes,
and anyone can regenerate the exact 100 levels and check what was scored,
because [the generator](level-generation.md) is deterministic.

## Why success rate is the ranking metric

Move efficiency is the more interesting number and the wrong thing to rank
on. It is measured over solved episodes only, so an agent that solves one
easy level perfectly and fails the other 99 posts a better efficiency than
one that solves 40 competently. Ranking on it would reward failing
selectively.

Success rate has none of that structure. It is bounded, it cannot be gamed
by attempting less, and it maps directly onto the claim being tested.
Efficiency and deadlock rate break ties.

## Why the difficulty ramp exists

An earlier version was a flat set of 50 levels at a fixed configuration.
Every published agent scored zero, which is a useless measurement: it
cannot distinguish a bad agent from a nearly-good one, and it gives no
gradient to improve against.

The ramp fixes the bottom of the range without capping the top. The
opening tier is solvable by the reference baseline's class of model, and
the closing tier is out of reach of anything published, so the benchmark
has both signal now and headroom later. See
[level generation](level-generation.md) for the measured calibration.

## Known limitations

Stated plainly, because a benchmark that hides its weaknesses is worth
less:

- **Deadlock rate is a lower bound.** The server uses a cheap corner test,
  not an exact search, because the exact check would block the request
  path. It undercounts.
- **The compute rules are honour-based.** Parameter count and dynamics
  calls per action are checked by source review, not enforced by a sandbox.
- **One environment.** Results here are evidence about Sokoban from pixels,
  not about world models in general.
- **The step budget is a heuristic.** Three times optimal is generous
  enough not to punish imperfect play and tight enough to matter, but it
  was chosen by judgement, not derived.
