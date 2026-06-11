"""Chromium binary management: extract from bundled zip and locate executable."""

import os
import stat
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]  # PPT-AGENTS-SHJ/
CHROMIUM_ZIP = REPO_ROOT / "bin" / "chromium-linux64.zip"
CHROMIUM_DIR = REPO_ROOT / "bin" / "chromium"


def get_chromium_executable() -> str:
    """
    Extract chromium-linux64.zip on first call, then return the path to the
    chrome/chromium binary. Raises FileNotFoundError if zip is missing.
    """
    if not CHROMIUM_ZIP.exists():
        raise FileNotFoundError(
            f"Chromium zip not found at {CHROMIUM_ZIP}. "
            "Upload chromium-linux64.zip to the bin/ directory."
        )

    if not CHROMIUM_DIR.exists():
        _extract(CHROMIUM_ZIP, CHROMIUM_DIR)

    return _find_binary(CHROMIUM_DIR)


def _extract(zip_path: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)

    # Make all files without extension executable (chrome, chromedriver, etc.)
    for f in dest.rglob("*"):
        if f.is_file() and not f.suffix:
            f.chmod(f.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


def _find_binary(base: Path) -> str:
    for name in ("chrome", "chromium", "chromium-browser"):
        for candidate in base.rglob(name):
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)

    raise FileNotFoundError(
        f"Could not find a chrome/chromium executable inside {base}. "
        "Check that chromium-linux64.zip contains a valid Chromium build."
    )
