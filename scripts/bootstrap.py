#!/usr/bin/env python3
"""Clone DOVER sources and download official weights."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOVER_DIR = ROOT / "third_party" / "DOVER"
DOVER_REPO = "https://github.com/VQAssessment/DOVER.git"


def main() -> None:
    sys.path.insert(0, str(ROOT / "src"))
    DOVER_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not (DOVER_DIR / "dover" / "models").exists():
        subprocess.check_call(["git", "clone", "--depth", "1", DOVER_REPO, str(DOVER_DIR)])
    init_py = DOVER_DIR / "dover" / "__init__.py"
    if init_py.exists() and "from .datasets import" in init_py.read_text(encoding="utf-8"):
        init_py.write_text("from .models import *\n", encoding="utf-8")
    from video_processing.weights import ensure_dover_weights

    path = ensure_dover_weights(ROOT / "weights" / "DOVER.pth")
    print(f"DOVER sources: {DOVER_DIR}")
    print(f"DOVER weights: {path}")


if __name__ == "__main__":
    main()
