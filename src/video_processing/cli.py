from __future__ import annotations

import os

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "0")

# Задаём настройку CUDA allocator до инициализации CUDA.
# expandable_segments может уменьшить фрагментацию при крупных аллокациях
# переменного размера, например при обработке батчей DOVER.
_alloc = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
if "expandable_segments" not in _alloc:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = (
        f"{_alloc},expandable_segments:True".lstrip(",")
    )

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from video_processing.config import load_config
from video_processing.device import configure_cuda
from video_processing.pipeline import run_pipeline


# [1] Вход. Читает конфиг [2], ставит сиды, [4] и зовёт [5].
def main() -> None:
    parser = argparse.ArgumentParser(description="Video Processing: DOVER + Qwen3-VL")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--input", default=None, help="каталог с видео")
    parser.add_argument("--output", default=None, help="каталог результатов")
    parser.add_argument("--replicate", type=int, default=None)
    args = parser.parse_args()

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
        torch.set_num_threads(cfg.cpu_threads)
        
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    
    configure_cuda()

    payload = run_pipeline(cfg)
    t = payload["throughput"]
    print("\n=== throughput (stable) ===")
    engine = t.get("engine")
    if engine == "vllm":
        engine = "vLLM"
    print(f"gpu:        {t.get('gpu') or t.get('device')}")
    print(f"engine:     {engine}")
    print(f"videos/sec: {t['videos_per_sec']:.4f}")
    print(f"sec/video:  {t['pipeline_sec_per_video']:.3f}")
    print(f"DOVER:      {t['dover_sec_per_video']:.3f} s")
    print(f"Qwen3-VL:   {t['qwen_sec_per_video']:.3f} s")
    print(f"results:    {cfg.output_dir / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
