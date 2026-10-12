from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

import numpy as np

from video_processing.config import load_config
from video_processing.device import configure_cuda
from video_processing.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Video Processing: DOVER + Qwen3-VL")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", default=None, help="каталог с видео")
    parser.add_argument("--output", default=None, help="каталог результатов")
    parser.add_argument("--replicate", type=int, default=None)
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

    cfg = load_config(args.config)
    if args.input:
        cfg.input_dir = Path(args.input).expanduser().resolve()
    if args.output:
        cfg.output_dir = Path(args.output).expanduser().resolve()
    if args.replicate is not None:
        cfg.replicate = args.replicate

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    try:
        import torch

        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
            configure_cuda()
    except Exception:
        pass

    payload = run_pipeline(cfg)
    t = payload["throughput"]
    print("\n=== throughput (stable) ===")
    print(f"gpu:        {t.get('gpu') or t.get('device')}")
    print(f"attn:       {t.get('attn_implementation')}")
    print(f"videos/sec: {t['videos_per_sec']:.4f}")
    print(f"sec/video:  {t['pipeline_sec_per_video']:.3f}")
    print(f"DOVER:      {t['dover_sec_per_video']:.3f} s")
    print(f"Qwen3-VL:   {t['qwen_sec_per_video']:.3f} s")
    print(f"results:    {cfg.output_dir / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
