"""DOVER prepare throughput: decode + views + normalize, without the forward pass.

Same pool shape as the pipeline (one ffmpeg thread per worker for cpu).
The first pass over the videos is warm-up (CUDA context, NVDEC init, page cache).

    python scripts/dover_decode_bench.py --decoder cpu nvdec --workers 1 8 --repeat 3
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch

from video_processing.config import load_config
from video_processing.dover_infer import DoverScorer


def _run(scorer: DoverScorer, paths: list[Path], workers: int) -> float:
    threads = 1 if workers > 1 else 0
    t0 = time.perf_counter()
    if workers == 1:
        for path in paths:
            scorer.prepare(path, threads)
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda p: scorer.prepare(p, threads), paths))
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter() - t0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoder", nargs="+", default=["cpu", "nvdec"])
    parser.add_argument("--workers", nargs="+", type=int, default=[1, 8])
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()

    cfg = load_config(ROOT / "configs" / "default.yaml")
    scorer = DoverScorer(cfg.dover)
    scorer.load()
    videos = sorted(cfg.input_dir.glob("*.mp4"), key=lambda p: int(p.stem) if p.stem.isdigit() else p.stem)
    paths = videos * args.repeat
    for decoder in args.decoder:
        scorer.cfg.decoder = decoder
        for workers in args.workers:
            _run(scorer, videos, workers)
            sec = _run(scorer, paths, workers)
            print(
                f"decoder={decoder:5s} workers={workers:2d} videos={len(paths)} "
                f"sec={sec:7.2f} videos/s={len(paths) / sec:6.2f}",
                flush=True,
            )


if __name__ == "__main__":
    main()
