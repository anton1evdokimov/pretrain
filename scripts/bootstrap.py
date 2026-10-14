#!/usr/bin/env python3
"""Clone DOVER sources at the pinned commit and download official weights."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOVER_DIR = ROOT / "third_party" / "DOVER"
DOVER_REPO = "https://github.com/VQAssessment/DOVER.git"
# Коммит, на котором проверено совпадение с официальным кодом (scripts/dover_parity.py).
DOVER_COMMIT = "f1ddc96215bc7fbcf8f315c65d47905f339c3419"


def _git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(DOVER_DIR), *args], text=True).strip()


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    DOVER_DIR.mkdir(parents=True, exist_ok=True)
    if not (DOVER_DIR / "dover" / "models").exists():
        _git("init", "-q")
        if "origin" not in _git("remote").split():
            _git("remote", "add", "origin", DOVER_REPO)
        _git("fetch", "-q", "--depth", "1", "origin", DOVER_COMMIT)
        _git("checkout", "-q", "FETCH_HEAD")
    head = _git("rev-parse", "HEAD")
    if head != DOVER_COMMIT:
        print(f"ВНИМАНИЕ: third_party/DOVER на {head[:7]}, проверен {DOVER_COMMIT[:7]}")
    init_py = DOVER_DIR / "dover" / "__init__.py"
    if init_py.exists() and "from .datasets import" in init_py.read_text(encoding="utf-8"):
        init_py.write_text("from .models import *\n", encoding="utf-8")
    from video_processing.weights import ensure_dover_weights

    path = ensure_dover_weights(ROOT / "weights" / "DOVER.pth")
    print(f"DOVER sources: {DOVER_DIR} @ {head[:7]}")
    print(f"DOVER weights: {path}")


if __name__ == "__main__":
    main()
