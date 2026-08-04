# Play it yourself

[Open the game](https://sokoban.lulzx.space/play){ .md-button .md-button--primary }

Before building an agent, it is worth playing enough Sokoban to feel what
makes it hard: the moment a level dies is usually many moves before you
notice.

## The rules of the game

Push crates onto the goal markers. You can only **push**, never pull, and
only one crate at a time. A crate in a corner is stuck forever, and a crate
flat against a wall can only slide along it.

That asymmetry is the whole game. Every push is potentially irreversible,
and nothing tells you when you have killed a level.

## Controls

| Action | Keyboard | Touch |
| --- | --- | --- |
| Move | Arrow keys or `WASD` | Swipe, or the on-screen pad |
| Undo | `U` or `Z` | Undo button |
| Restart | `R` | Restart button |
| Next / previous level | `N` / `P` | Levels picker |

Undo is unlimited, which the evaluation API deliberately does not offer.
Agents get one attempt per level and live with their mistakes.

## The level set

The 50-level Thinking Rabbit **Original** collection, sourced from the
default pack in [davidjoffe/sokoban](https://github.com/davidjoffe/sokoban),
and presented in its original order.

!!! note "These are not the benchmark levels"

    The game uses a public, classic level set. The
    [hidden evaluation set](level-generation.md) is different: 100
    procedurally generated 8×8 levels that never leave the server. Playing
    here tells you nothing about the hidden levels, which is the point.

## Progress

Completed levels and your best move count are stored in your browser's
local storage, so you can leave and come back. Nothing is sent to the
server, and there is no account.

Clearing site data clears your progress.
