"""Push the generated Kaggle notebook, wait for it to finish, and pull the
resulting parquet parts back to local disk. End-to-end automation so we
never touch the Kaggle web UI after the first phone-verification.

Usage:
    python scripts/kaggle/run_on_kaggle.py
"""

from __future__ import annotations

import argparse
import json
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
USERNAME = "mindest"  # Kaggle user; the account behind KAGGLE_API_TOKEN in .env

# IMPORTANT:
#   title must slugify to the same string as the slug portion of id,
#   otherwise Kaggle silently replaces the id and the polling URL breaks.
#   Kaggle slugify = lowercase + spaces-to-hyphens. So keep title pure ASCII.
CFG = {
    "slug": "alt-data-sentiment-score-leukipp",
    "title": "Alt Data Sentiment Score Leukipp",
    "notebook": "score_leukipp_on_kaggle.ipynb",
    "input_dataset": "leukipp/reddit-finance-data",
    "output_folder_in_working": "scored_leukipp",
    "local_out": PROCESSED / "scored_leukipp",
}


def _auth() -> None:
    load_dotenv(REPO_ROOT / ".env")
    token = os.environ.get("KAGGLE_API_TOKEN")
    if not token:
        sys.exit("KAGGLE_API_TOKEN missing from .env")
    os.environ["KAGGLE_API_TOKEN"] = token


def _sh(cmd: list[str]) -> subprocess.CompletedProcess:
    print(f"$ {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    # Force UTF-8 so kaggle-cli doesn't crash on unicode responses under
    # Windows cp950 locale.
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return subprocess.run(
        cmd, check=False, text=True, capture_output=True, env=env,
        encoding="utf-8",
    )


def _prepare_push_dir(cfg: dict) -> Path:
    push_dir = KAGGLE_DIR / "push" / cfg["slug"]
    push_dir.mkdir(parents=True, exist_ok=True)
    # Copy the generated notebook into the push dir
    shutil.copy2(KAGGLE_DIR / cfg["notebook"], push_dir / cfg["notebook"])
    metadata = {
        "id": f"{USERNAME}/{cfg['slug']}",
        "title": cfg["title"],
        "code_file": cfg["notebook"],
        "language": "python",
        "kernel_type": "notebook",
        "is_private": "true",
        "enable_gpu": "true",
        "enable_internet": "true",
        # Force T4 — P100 (sm_60) is incompatible with the preinstalled
        # PyTorch on Kaggle (which drops sm_60). T4 (sm_75) is compatible.
        "machine_shape": "GPU T4 x2",
        "dataset_sources": [cfg["input_dataset"]],
        "competition_sources": [],
        "kernel_sources": [],
    }
    (push_dir / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2))
    return push_dir


def _push(push_dir: Path) -> None:
    # --accelerator overrides enable_gpu / machine_shape in metadata. Try
    # the human-visible label ("GPU T4 x2") that Kaggle's web UI uses.
    r = _sh([
        "kaggle", "kernels", "push",
        "-p", str(push_dir),
        "--accelerator", "GPU T4 x2",
    ])
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        sys.exit(f"kernels push failed (exit {r.returncode})")


def _wait(ref: str, poll_s: int = 30, timeout_min: int = 90) -> None:
    t0 = time.time()
    while True:
        r = _sh(["kaggle", "kernels", "status", ref])
        print(r.stdout.strip())
        status_text = r.stdout.lower()
        # Common states: "queued", "running", "complete", "error", "cancelled"
        if "complete" in status_text:
            return
        if "error" in status_text or "cancel" in status_text or "fail" in status_text:
            sys.exit(f"kernel ended badly: {r.stdout}")
        if time.time() - t0 > timeout_min * 60:
            sys.exit(f"timeout after {timeout_min} min")
        time.sleep(poll_s)


def _download(ref: str, cfg: dict) -> None:
    cfg["local_out"].mkdir(parents=True, exist_ok=True)
    # Pull ALL output files; Kaggle flattens /kaggle/working/... paths.
    tmp = KAGGLE_DIR / "output_tmp" / cfg["slug"]
    tmp.mkdir(parents=True, exist_ok=True)
    r = _sh(["kaggle", "kernels", "output", ref, "-p", str(tmp)])
    print(r.stdout.strip())
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        sys.exit("kernels output failed")

    # Kaggle returns outputs as flat files; parts are named scored_X/part_YYYYY.parquet
    # OR flattened as part_YYYYY.parquet. Move all *.parquet into local_out.
    moved = 0
    for p in tmp.rglob("*.parquet"):
        dest = cfg["local_out"] / p.name
        shutil.copy2(p, dest)
        moved += 1
    print(f"[download] moved {moved} parquet files -> {cfg['local_out']}")


def main() -> None:
    argparse.ArgumentParser().parse_args()
    _auth()

    ref = f"{USERNAME}/{CFG['slug']}"
    push_dir = _prepare_push_dir(CFG)
    _push(push_dir)
    print(f"[push] kernel ref = {ref}")
    print("[wait] polling status every 30s (max 90 min) ...")
    _wait(ref)
    print("[done] kernel complete; downloading output ...")
    _download(ref, CFG)


if __name__ == "__main__":
    main()
