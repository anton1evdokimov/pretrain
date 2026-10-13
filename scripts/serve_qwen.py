#!/usr/bin/env python3
"""Start the Qwen3-VL vLLM server with the pipeline's A100 settings.

The pipeline starts this server itself after DOVER. Run the script only
when you want the OpenAI endpoint left up, and leave qwen.api_base empty
in that case so a second copy is not spawned.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_processing.config import load_config  # noqa: E402
from video_processing.qwen_vllm import vllm_serve_command  # noqa: E402


def main() -> None:
    cfg = load_config(ROOT / "configs" / "default.yaml")
    cmd = vllm_serve_command(cfg.qwen)
    print(" ".join(cmd), flush=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    os.execve(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
