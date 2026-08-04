# Code reference

Generated from the docstrings in the source, so it cannot drift from what
the code actually does.

## The library

| Module | What it owns |
| --- | --- |
| [`env`](env.md) | `Level`, `SokobanEnv`: state, moves, push rules, the solved test |
| [`levels`](levels.md) | Level generation, and the hidden set's difficulty ramp |
| [`solver`](solver.md) | BFS optimal solutions and deadlock detection |
| [`render`](render.md) | Board state to 64×64×3 RGB |
| [`dataset`](dataset.md) | Trajectory generation for training |
| [`evaluation`](evaluation.md) | Local evaluation harness and metrics |
| [`agent`](agent.md) | The `Agent` interface your submission implements |

!!! warning "The solver is not for agents"

    `solver` exists so the *server* can know each level's optimal solution
    length and detect deadlocks. Calling anything like it from inside an
    agent is [against the rules](../RULES.md): it is exactly the symbolic
    search the benchmark is built to exclude.

## Where to start

Implementing a submission means implementing one interface,
[`latent_sokoban.agent.Agent`](agent.md). Everything else is either the
environment you are being tested against or tooling for producing training
data.

For the end-to-end picture of how these pieces fit together at evaluation
time, see [how it works](../how-it-works.md).

## Not documented here

`server/app.py` is the evaluation API; its interface is the HTTP surface
documented in [the agent protocol](../API.md), and there is a
machine-readable OpenAPI schema at
[`/api/openapi.json`](https://sokoban.lulzx.space/api/openapi.json).

`baseline/` is a reference implementation rather than a stable API. Read it
as an example, not as something to import against.
