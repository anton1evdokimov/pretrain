from __future__ import annotations

import os

# Before the CUDA caching allocator starts. DOVER's batch-16 activation is one
# ~11 GiB block; without this, PyTorch holds that much in fragments and the
# new block fails even though the sum of free cache would cover it.
_alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" not in _alloc:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        f"{_alloc},expandable_segments:True".lstrip(",")
    )

import argparse
import random
from pathlib import Path

import numpy as np

from video_processing.config import load_config
from video_processing.device import configure_cuda
from video_processing.pipeline import run_pipeline


# [1] Вход. Читает конфиг, ставит сид и зовёт [5]. Флаг --ablate сюда не входит.
def main() -> None:
    parser = argparse.ArgumentParser(description="Video Processing: DOVER + Qwen3-VL")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", default=None, help="каталог с видео")
    parser.add_argument("--output", default=None, help="каталог результатов")
    parser.add_argument("--replicate", type=int, default=None)
    parser.add_argument(
        "--ablate",
        action="store_true",
        help="прогон точек скорость/качество и батчей 8/10/(16)",
    )
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
    if cfg.cpu_threads > 0:
        import cv2

        cv2.setNumThreads(cfg.cpu_threads)
    try:
        import torch

        if cfg.cpu_threads > 0:
            torch.set_num_threads(cfg.cpu_threads)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
            configure_cuda()
    except Exception:
        pass

    if args.ablate:
        from video_processing.ablate import run_ablation

        report = run_ablation(cfg)
        chosen = report.get("chosen") or {}
        print("\n=== ablation done ===")
        print(f"quality:    {chosen.get('quality')}")
        print(f"qwen batch: {chosen.get('qwen_batch_size')}")
        print(f"dover batch:{chosen.get('dover_batch_size')}")
        print(f"table:      {cfg.output_dir / 'ABLATION.md'}")
        return

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
