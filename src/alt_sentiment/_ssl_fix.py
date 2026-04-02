"""Workaround for a conda-on-Windows quirk.

On Windows, conda's activation scripts set `SSL_CERT_FILE` to
`<env>/ssl/cacert.pem`. That file is only produced for envs built with
conda's openssl / ca-certificates; pip-only envs leave the path dangling,
which makes every httpx / urllib3 call fail with `FileNotFoundError: No
such file or directory`.

Importing this module early (`from alt_sentiment import _ssl_fix`)
rewrites `SSL_CERT_FILE` to the first existing bundle it can find, or
unsets it so Python falls back to the system default.
"""

from __future__ import annotations

import os
from pathlib import Path


def _candidate_paths() -> list[Path]:
    out: list[Path] = []
    try:
        import certifi

        out.append(Path(certifi.where()))
    except ImportError:
        pass
    # Common conda-on-Windows capital-L Library location
    prefix = os.environ.get("CONDA_PREFIX")
    if prefix:
        out.append(Path(prefix) / "Library" / "ssl" / "cacert.pem")
    return out


def fix() -> None:
    current = os.environ.get("SSL_CERT_FILE", "")
    if current and Path(current).exists():
        return
    for c in _candidate_paths():
        if c.exists():
            os.environ["SSL_CERT_FILE"] = str(c)
            return
    # Give up — unset so ssl falls back to platform defaults
    os.environ.pop("SSL_CERT_FILE", None)


# Run on import.
fix()
