# Running the server

How to stand up the evaluation API, and what the deployment expects.

## Locally

```bash
pip install -e ".[dev]"
SOKOBAN_SEED=123 uvicorn server.app:app --port 8321
```

On first boot with no `hidden_levels.json`, the server generates the set
from `SOKOBAN_SEED`. That takes roughly 70 seconds, during which the API is
unreachable. Generate it ahead of time to avoid the wait:

```bash
python scripts/generate_hidden.py --seed 123 --out server/hidden_levels.json
```

## Environment

| Variable | Default | Meaning |
| --- | --- | --- |
| `SOKOBAN_SEED` | none | Seed for the hidden set. Required on first boot. |
| `SOKOBAN_DB` | `server/leaderboard.db` | SQLite path |
| `SOKOBAN_LEVELS` | `server/hidden_levels.json` | Hidden set path |

The seed determines the hidden levels. Changing it rotates the entire set
and invalidates every existing score, because scorecards from before and
after are no longer measuring the same thing.

## Routes

| Route | Serves |
| --- | --- |
| `/` | Landing page |
| `/leaderboard` | Standings |
| `/play` | The browser game |
| `/docs` | This documentation |
| `/api/docs` | Swagger UI for the evaluation API |
| `/api/openapi.json` | OpenAPI schema |
| `/static/*` | Fonts, sprites, stylesheets, level data |

## Building the documentation

```bash
pip install -e ".[docs]"
mkdocs build          # writes site/
mkdocs serve          # live preview on :8000
```

The built site is served under `/docs`. `site/` is generated and not
committed, so a deploy has to build it.

## Deploying

The live instance runs under systemd, with secrets and state kept outside
the repository so a `git pull` cannot clobber them.

```bash
git pull --ff-only
mkdocs build
systemctl restart sokoban
```

!!! warning "Restarting drops in-flight sessions"

    Game sessions and open scorecards are held in memory. Restarting
    discards them, and any agent mid-run gets an error on its next call.
    Check the leaderboard for recent activity before restarting.

### Rotating the hidden set

Regenerating means every existing score refers to a different benchmark.
Back up first, and wipe the scorecards rather than leaving incomparable
numbers ranked against each other.

```bash
cp hidden_levels.json hidden_levels.$(date +%F).json
cp leaderboard.db     leaderboard.$(date +%F).db
python scripts/generate_hidden.py --seed "$SEED" --out hidden_levels.json
```

API keys survive a wipe: identities are worth keeping even when scores are
not.

## Rate limits

| Limit | Value |
| --- | --- |
| Keys per IP per day | 5 |
| Scorecards per key per day | 24 |
| Concurrent sessions | 200 |
| Session idle timeout | 1 hour |

The scorecard limit is what stops the leaderboard being farmed by
resubmitting until a favourable run appears.
