# Scoring

Every metric is computed server-side in `_game_summary` when a scorecard is
closed. Nothing is self-reported.

## The metrics

| Metric | Definition |
| --- | --- |
| `success_rate` | Solved episodes ÷ total episodes. **This is the ranking metric.** |
| `move_efficiency` | Mean of `optimal ÷ steps_taken` over *solved* episodes only |
| `avg_steps_solved` | Mean steps taken on solved episodes |
| `deadlock_rate` | Share of all episodes that ended in a detected deadlock |
| `total_actions` | Every action sent, across every episode |

### Success rate

The denominator is always the full hidden set, never the number you chose
to attempt:

```python
if pad_to:  # unplayed episodes count as unsolved
    res += [{"solved": False, ...}] * (pad_to - len(res))
```

Closing a scorecard pads the results out to the full 100 episodes with
failures. An agent that plays 10 levels, solves all 10, and stops scores
**10%**, not 100%. There is no way to cherry-pick a good subset.

### Move efficiency

`optimal ÷ steps` averaged over solved levels, where `optimal` is the BFS
shortest solution computed when the level was generated. 1.000 means every
solve was perfect; lower means wandering.

It is deliberately measured over solved episodes only. Averaging in failed
episodes would let an agent look efficient by failing fast, which is the
opposite of what the metric is for. The consequence is that efficiency is
only meaningful alongside success rate, which is why it is a tiebreak
rather than a ranking metric.

### Deadlock rate

A deadlock is a state from which the level can never be solved, typically a
crate pushed into a corner or flat against a wall it can never leave.

The server uses a cheap corner test (`is_deadlocked`) rather than a full
search, because the exact check would hold the serving lock for seconds:

```python
# cheap corner test only: the exact BFS check would block the serving
# lock for seconds. This undercounts, so the metric is a lower bound.
```

Treat `deadlock_rate` as a **lower bound**. A low number does not prove an
agent avoids deadlocks; a high number does prove it creates them.

## Ranking

The leaderboard shows each entrant's best scorecard, ordered by:

1. `success_rate` descending
2. `move_efficiency` descending
3. `deadlock_rate` ascending

Only your best scorecard is displayed, so a bad run cannot hurt a good one.
Scorecards are rate limited per key per day, which is what stops the
leaderboard from being farmed by resubmission.

## Step budgets

Each level's budget is **three times its own optimal solution**, with a
floor of 30 steps:

```python
max_steps = max(30, 3 * optimal_len)
```

Earlier rounds used a flat 80 steps for every level, which was loose for a
10-move level and starving for a 40-move one. Scaling the budget keeps the
same amount of slack at every point on the [difficulty
ramp](level-generation.md).

Running out of steps is an ordinary unsolved episode. It is not
distinguished from any other failure in the metrics.

## Reading a scorecard

```json
{
  "episodes": 100,
  "solved": 31,
  "success_rate": 0.31,
  "move_efficiency": 0.874,
  "avg_steps_solved": 23.4,
  "deadlock_rate": 0.08,
  "total_actions": 4120
}
```

This agent solved 31 of 100. When it solved a level it took about 1.14×
the optimal path (1 ÷ 0.874), and it wedged a crate unrecoverably in at
least 8% of episodes.
