"""Record where a result came from: code version, data checksum, environment."""

import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from mqc import __version__
from mqc.data import md5sum

_PACKAGES = ["numpy", "scikit-learn", "joblib", "torch", "pyyaml"]


def git_info() -> dict[str, Any]:
    """Current commit and whether there are uncommitted changes.

    A result produced from uncommitted code ("dirty") cannot be reproduced
    exactly from the repository, so we record it rather than hide it.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--untracked-files=no"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def _package_versions() -> dict[str, str | None]:
    versions = {}
    for name in _PACKAGES:
        try:
            versions[name] = version(name)
        except PackageNotFoundError:
            versions[name] = None
    return versions


def collect(dataset_path: str | Path) -> dict[str, Any]:
    return {
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "mqc_version": __version__,
        "git": git_info(),
        "dataset_md5": md5sum(dataset_path),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": _package_versions(),
    }
