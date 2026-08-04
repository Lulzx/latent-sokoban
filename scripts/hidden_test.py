#!/usr/bin/env python3
"""Hidden-test-set commitment protocol.

Because level generation is fully determined by (frozen script,
constraints, seed), committing to a secret seed IS committing to the
exact test levels, so nothing needs to be generated, stored, or resisted
peeking at until reveal day.

    # 1. Commit day (done by whoever holds the seed):
    python scripts/hidden_test.py commit --seed 987654321
    #    prompts for a password; writes hidden_seed.enc + hidden_commitment.json
    #    Check BOTH files into the repo. Forget the seed. Keep the password
    #    somewhere nobody will open early.

    # 2. Reveal day (after submissions are frozen):
    python scripts/hidden_test.py reveal
    #    prompts for the password; verifies the commitment hash, prints the
    #    seed, and generates the final splits into levels/final/

The encrypted file is OpenSSL AES-256-CBC with PBKDF2; the commitment
JSON stores a SHA-256 of the plaintext seed string so a tampered or
wrong-password reveal is detected. The generator version is pinned by
recording the git commit at commit time, so evaluate at that commit.
"""

from __future__ import annotations

import argparse
import getpass
import hashlib
import json
import os
import secrets
import subprocess
import sys
from pathlib import Path

ENC_FILE = Path("hidden_seed.enc")
COMMIT_FILE = Path("hidden_commitment.json")
OPENSSL = ["openssl", "enc", "-aes-256-cbc", "-pbkdf2", "-iter", "200000"]


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:
        return "unknown"


def commit(args) -> None:
    if ENC_FILE.exists():
        sys.exit(f"{ENC_FILE} already exists; refusing to overwrite a commitment")
    password = getpass.getpass("password to protect the seed: ")
    if password != getpass.getpass("repeat password: "):
        sys.exit("passwords do not match")
    # a bare integer seed would be brute-forceable from its public hash;
    # binding a random nonce into the committed plaintext prevents that
    seed_text = f"{args.seed}:{secrets.token_hex(16)}"
    subprocess.run(OPENSSL + ["-salt", "-out", str(ENC_FILE),
                              "-pass", "env:HIDDEN_PW"],
                   input=seed_text.encode(), check=True,
                   env={**os.environ, "HIDDEN_PW": password})
    COMMIT_FILE.write_text(json.dumps({
        "seed_sha256": hashlib.sha256(seed_text.encode()).hexdigest(),
        "generator_git_commit": git_commit(),
        "splits": args.splits,
        "levels_per_split": args.n,
    }, indent=2))
    print(f"committed. check in {ENC_FILE} and {COMMIT_FILE}, then forget the seed.")
    print("do NOT run the generator with this seed until reveal day.")


def reveal(args) -> None:
    meta = json.loads(COMMIT_FILE.read_text())
    password = getpass.getpass("password: ")
    out = subprocess.run(OPENSSL + ["-d", "-in", str(ENC_FILE),
                                    "-pass", "env:HIDDEN_PW"],
                         capture_output=True,
                         env={**os.environ, "HIDDEN_PW": password})
    if out.returncode != 0:
        sys.exit("decryption failed (wrong password?)")
    seed_text = out.stdout.decode().strip()
    digest = hashlib.sha256(seed_text.encode()).hexdigest()
    if digest != meta["seed_sha256"]:
        sys.exit("COMMITMENT MISMATCH: decrypted seed does not hash to the "
                 "committed value: the archive or commitment was altered")
    seed = int(seed_text.split(":")[0])
    print(f"seed: {seed} (commitment verified)")
    print(f"generator was frozen at commit {meta['generator_git_commit']}")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, name in enumerate(meta["splits"]):
        subprocess.run([sys.executable, "scripts/generate_levels.py",
                        "--split", name, "--n", str(meta["levels_per_split"]),
                        "--seed", str(seed + i),
                        "--out", str(out_dir / f"split_{name.lower()}.json")],
                       check=True)
    print(f"final hidden splits generated in {out_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("commit", help="encrypt and commit to a secret seed")
    c.add_argument("--seed", type=int, required=True)
    c.add_argument("--splits", nargs="+", default=["A", "B", "C", "D"])
    c.add_argument("--n", type=int, default=100, help="levels per split")
    r = sub.add_parser("reveal", help="decrypt, verify, and generate final splits")
    r.add_argument("--out", default="levels/final")
    args = parser.parse_args()
    commit(args) if args.cmd == "commit" else reveal(args)


if __name__ == "__main__":
    main()
