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
import concurrent.futures
import gzip
import http.client
import json
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from latent_sokoban.evaluation import load_agent


class Client:
    """One connection held open for the whole run.

    urllib.request.urlopen has no connection pooling: it opens a fresh TCP
    connection and a fresh TLS handshake per request. Measured against the
    live server that is 650 ms per call, of which only ~209 ms is the actual
    request -- the other 441 ms is handshake, thrown away and redone for the
    next action. Over the ~6800 actions of a 100-level run that is the
    difference between roughly 74 minutes and roughly 24.
    """

    def __init__(self, base: str, key: str | None):
        self.base = base.rstrip("/")
        self.key = key
        parts = urllib.parse.urlsplit(self.base)
        self._https = parts.scheme != "http"
        self._host = parts.hostname
        self._port = parts.port
        self._prefix = parts.path.rstrip("/")
        self._conn: http.client.HTTPConnection | None = None

    def _connect(self) -> http.client.HTTPConnection:
        if self._conn is None:
            cls = (http.client.HTTPSConnection if self._https
                   else http.client.HTTPConnection)
            self._conn = cls(self._host, self._port, timeout=60)
        return self._conn

    def _drop(self) -> None:
        """Discard the connection so the next attempt builds a clean one.

        A kept-alive connection can be closed by the server or a middlebox at
        any time, and a half-used one cannot be recovered -- so every failure
        path drops it rather than trying to reuse it.
        """
        if self._conn is not None:
            try:
                self._conn.close()
            except OSError:
                pass
            self._conn = None

    def call(self, path: str, payload: dict | None = None,
             attempts: int = 6) -> dict:
        # A full run is ~5000 requests over an hour or more, so a transient
        # reset is likely rather than exceptional -- and losing one used to
        # kill the run with the scorecard still open, which leaves it
        # invisible to the leaderboard forever. Retry with backoff instead.
        #
        # Caveat on /act: if the server applied the action and the response
        # was lost on the way back, the retry applies it a second time. The
        # frame that comes back is still the true state and the agent replans
        # from it, so the cost is a wasted step rather than a corrupt episode.
        # Making this exact needs a client-supplied sequence number the API
        # does not currently take.
        #
        # urllib neither asks for gzip nor decodes it, and a frame is ~32 KB
        # of base64 that compresses roughly fiftyfold. Over a full run that
        # is the difference between tens of megabytes and a couple.
        headers = {"Content-Type": "application/json",
                   "Accept-Encoding": "gzip"}
        if self.key:
            headers["X-API-Key"] = self.key
        body_bytes = json.dumps(payload).encode() if payload is not None else b"{}"

        for attempt in range(attempts):
            try:
                conn = self._connect()
                conn.request("POST", self._prefix + path, body=body_bytes,
                             headers=headers)
                resp = conn.getresponse()
                body = resp.read()          # must drain before reusing the conn
                if resp.status >= 400:
                    # http.client does not raise on error statuses. 4xx is the
                    # client being wrong and will not fix itself; 429 and 5xx
                    # are worth waiting out.
                    if resp.status < 500 and resp.status != 429:
                        self._drop()
                        sys.exit(f"HTTP {resp.status} on {path}: "
                                 f"{body.decode(errors='replace')[:300]}")
                    reason = f"HTTP {resp.status}"
                    self._drop()
                else:
                    if resp.getheader("Content-Encoding") == "gzip":
                        body = gzip.decompress(body)
                    return json.loads(body)
            except (urllib.error.URLError, TimeoutError, ConnectionError,
                    http.client.HTTPException, OSError) as e:
                reason = type(e).__name__ + f": {e}"
                self._drop()
            if attempt == attempts - 1:
                sys.exit(f"\n{path} failed after {attempts} attempts: {reason}")
            delay = 2 ** attempt
            print(f"\n[retry {attempt+1}/{attempts-1} in {delay}s] {path}: {reason}",
                  file=sys.stderr, flush=True)
            time.sleep(delay)
        raise AssertionError("unreachable")


def decode(o: dict, field: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(o[field]),
                         dtype=o["dtype"]).reshape(o["shape"])


def run_parallel(client: "Client", args, sid: str, frame: dict) -> dict:
    """Drive several in-flight episodes at once.

    A run is bounded by round-trip latency, not by the agent: ~6800 sequential
    actions at ~190 ms is over an hour, while the agent itself accounts for
    ~4 ms of each. Acting on every open episode concurrently collapses that to
    roughly (steps in the longest episode) x RTT per tier.

    One Client per worker thread: a kept-alive HTTP connection carries one
    request at a time, so sharing a single one across threads would interleave
    responses rather than parallelise them.
    """
    gid = frame["session_id"]
    total = frame["episodes"][0]["of"] if frame.get("episodes") else 0
    print(f"scorecard {sid}  session {gid}  {total} hidden episodes, "
          f"up to {args.parallel} in flight")

    local = threading.local()

    def conn() -> Client:
        if not hasattr(local, "client"):
            local.client = Client(args.url, args.key)
        return local.client

    agents: dict[int, object] = {}
    history: dict[int, list[int]] = {}
    obs: dict[int, dict] = {}

    def adopt(ep: dict) -> None:
        i = ep["index"]
        if i not in agents:
            agents[i] = load_agent(args.agent)
            history[i] = []
        obs[i] = ep["observation"]

    for ep in frame["episodes"]:
        adopt(ep)

    final = frame
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
        while True:
            live = sorted(obs)
            if not live:
                break
            # Actions are computed here rather than in the workers: the agent
            # is torch and the whole point is that it is not the bottleneck.
            chosen = {}
            for i in live:
                o = obs[i]
                chosen[i] = int(agents[i].act(decode(o, "obs"),
                                              decode(o, "goal"),
                                              list(history[i])))
                history[i].append(chosen[i])

            def send(i: int):
                return i, conn().call(f"/api/sessions/{gid}/act",
                                      {"action": chosen[i], "episode": i})

            over = False
            for i, fr in pool.map(send, live):
                final = fr
                last = fr.get("last") or {}
                if last.get("episode_done"):
                    print("+" if last.get("solved") else ".", end="", flush=True)
                    obs.pop(i, None)
                    agents.pop(i, None)
                    history.pop(i, None)
                else:
                    # only trust this response for the episode it acted on;
                    # the other entries may already be a round stale
                    for ep in fr.get("episodes", []):
                        if ep["index"] == i:
                            obs[i] = ep["observation"]
                if fr["state"] == "GAME_OVER":
                    over = True
            if over:
                break
            for ep in final.get("episodes", []):      # newly opened episodes
                adopt(ep)
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--register", metavar="NAME",
                        help="create an API key for this leaderboard name and exit")
    parser.add_argument("--agent", help="'random' or module:Class")
    parser.add_argument("--key", default=os.environ.get("SOKOBAN_API_KEY"))
    parser.add_argument("--url", default="https://sokoban.lulzx.space")
    parser.add_argument("--parallel", type=int, default=1,
                        help="episodes of the current tier to run at once "
                             "(1 = original serial protocol)")
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

    card = client.call("/api/scorecards")
    sid = card["scorecard_id"]
    frame = client.call(
        f"/api/scorecards/{sid}/games/standard/start?parallel={args.parallel}")

    if args.parallel > 1:
        frame = run_parallel(client, args, sid, frame)
    else:
        agent = load_agent(args.agent)
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

    # Try harder here than anywhere else: an unclosed scorecard scores
    # nothing no matter how the run went.
    result = client.call(f"/api/scorecards/{sid}/close", attempts=10)
    print("\n\nscore:", json.dumps(result["games"]["standard"], indent=2))
    # Safe to share: outcomes are public, your action strings are not. They
    # come back only for requests carrying your key.
    print(f"replay: {args.url}/api/replays/{frame['session_id']}")


if __name__ == "__main__":
    main()
