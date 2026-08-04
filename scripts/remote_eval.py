#!/usr/bin/env python3
"""Evaluate a local Agent against the public leaderboard API.

Your agent runs on YOUR machine; the server holds the hidden levels and
only ever sends rendered observations. Protocol spec: docs/API.md.

First time, create a key (bind your leaderboard name):

    python scripts/remote_eval.py --register "my-lab"

Then evaluate (key via $SOKOBAN_API_KEY or --key):

    python scripts/remote_eval.py --agent random
    python scripts/remote_eval.py --agent my_pkg.agent:MyAgent \
        --url https://sokoban.lulzx.space

Uses only the standard library (urllib) + numpy.
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latent_sokoban.evaluation import load_agent


class Client:
    def __init__(self, base: str, key: str | None):
        self.base = base.rstrip("/")
        self.key = key

    def call(self, path: str, payload: dict | None = None) -> dict:
        # urllib neither asks for gzip nor decodes it, and a frame is ~32 KB
        # of base64 that compresses roughly fiftyfold. Over a full run that
        # is the difference between tens of megabytes and a couple.
        headers = {"Content-Type": "application/json",
                   "Accept-Encoding": "gzip"}
        if self.key:
            headers["X-API-Key"] = self.key
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(payload).encode() if payload is not None else b"{}",
            headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    body = gzip.decompress(body)
                return json.loads(body)
        except urllib.error.HTTPError as e:
            detail = e.read().decode()[:300]
            sys.exit(f"HTTP {e.code} on {path}: {detail}")


def decode(o: dict, field: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(o[field]),
                         dtype=o["dtype"]).reshape(o["shape"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", metavar="NAME",
                        help="create an API key for this leaderboard name and exit")
    parser.add_argument("--agent", help="'random' or module:Class")
    parser.add_argument("--key", default=os.environ.get("SOKOBAN_API_KEY"))
    parser.add_argument("--url", default="https://sokoban.lulzx.space")
    args = parser.parse_args()

    client = Client(args.url, args.key)

    if args.register:
        out = client.call("/api/keys", {"name": args.register})
        print(f"api_key: {out['api_key']}")
        print("store it, e.g.:  export SOKOBAN_API_KEY=" + out["api_key"])
        return

    if not args.agent:
        parser.error("--agent is required (or use --register first)")
    if not args.key:
        parser.error("no API key: pass --key, set $SOKOBAN_API_KEY, "
                     "or run --register NAME first")

    agent = load_agent(args.agent)
    card = client.call("/api/scorecards")
    sid = card["scorecard_id"]
    frame = client.call(f"/api/scorecards/{sid}/games/standard/start")
    print(f"scorecard {sid}  session {frame['session_id']}  "
          f"{frame['episode']['of']} hidden episodes")

    episode = -1
    history: list[int] = []
    while frame["state"] != "GAME_OVER":
        if frame["episode"]["index"] != episode:
            episode = frame["episode"]["index"]
            agent.reset()
            history = []
        obs = decode(frame["observation"], "obs")
        goal = decode(frame["observation"], "goal")
        action = int(agent.act(obs, goal, list(history)))
        history.append(action)
        frame = client.call(f"/api/sessions/{frame['session_id']}/act",
                            {"action": action})
        last = frame.get("last") or {}
        if last.get("episode_done"):
            print("+" if last.get("solved") else ".", end="", flush=True)

    result = client.call(f"/api/scorecards/{sid}/close")
    print("\n\nscore:", json.dumps(result["games"]["standard"], indent=2))
    # Safe to share: outcomes are public, your action strings are not. They
    # come back only for requests carrying your key.
    print(f"replay: {args.url}/api/replays/{frame['session_id']}")


if __name__ == "__main__":
    main()
