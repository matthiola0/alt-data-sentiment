"""Poll the already-pushed Kaggle kernel until it finishes, then download
its output. Use when the initial run_on_kaggle.py polling loop times
out but the kernel is still RUNNING on Kaggle.

Usage:
    python scripts/kaggle/poll_and_download.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[2]
KAGGLE_DIR = REPO_ROOT / "scripts" / "kaggle"
PROCESSED = REPO_ROOT / "data" / "processed"
USERNAME = "mindest"
SLUG = "alt-data-sentiment-score-leukipp"


def _sh(cmd: list[str]) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return subprocess.run(cmd, check=False, text=True, capture_output=True,
                          env=env, encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--timeout-min", type=int, default=360)
    p.add_argument("--poll-seconds", type=int, default=60)
    args = p.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    if not os.environ.get("KAGGLE_API_TOKEN"):
        sys.exit("KAGGLE_API_TOKEN missing")

    ref = f"{USERNAME}/{SLUG}"
    t0 = time.time()
    while True:
        r = _sh(["kaggle", "kernels", "status", ref])
        text = r.stdout.strip()
        print(text, flush=True)
        lowered = text.lower()
        if "complete" in lowered:
            break
        if any(s in lowered for s in ("error", "cancel", "fail")):
            sys.exit(f"kernel ended badly: {text}")
        if time.time() - t0 > args.timeout_min * 60:
            sys.exit(f"still running after {args.timeout_min} min")
        time.sleep(args.poll_seconds)

    print(f"[complete] after {(time.time()-t0)/60:.1f} min; downloading ...")
    local_out = PROCESSED / "scored_leukipp"
    local_out.mkdir(parents=True, exist_ok=True)
    tmp = KAGGLE_DIR / "output_tmp" / SLUG
    tmp.mkdir(parents=True, exist_ok=True)
    r = _sh(["kaggle", "kernels", "output", ref, "-p", str(tmp)])
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr); sys.exit("output download failed")

    moved = 0
    for p_path in tmp.rglob("*.parquet"):
        dest = local_out / p_path.name
        shutil.copy2(p_path, dest)
        moved += 1
    print(f"[done] moved {moved} parquet -> {local_out}")


if __name__ == "__main__":
    main()
