from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

from video_processing.config import AppConfig
from video_processing.decode import probe_video
from video_processing.device import gpu_name, release_memory, sync
from video_processing.dover_infer import DoverScorer
from video_processing.qwen_infer import QwenCaptioner


VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".avi", ".webm"}


@dataclass
class VideoResult:
    video_id: str
    source: str
    duration_sec: float
    width: int
    height: int
    technical: float
    aesthetic: float
    overall: float
    caption: str
    dover_sec: float
    qwen_sec: float


def list_videos(input_dir: Path) -> list[Path]:
    videos = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    ]
    if not videos:
        raise FileNotFoundError(f"в {input_dir} нет видео")
    return videos


def expand_for_throughput(videos: list[Path], replicate: int) -> list[tuple[int, Path]]:
    replicate = max(1, replicate)
    return [(copy, path) for copy in range(replicate) for path in videos]


def _chunks(items: list, size: int):
    size = max(1, size)
    for start in range(0, len(items), size):
        yield items[start : start + size]


def run_pipeline(cfg: AppConfig) -> dict:
    videos = list_videos(cfg.input_dir)
    jobs = expand_for_throughput(videos, cfg.replicate)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    dover = DoverScorer(cfg.dover)
    qwen = QwenCaptioner(cfg.qwen, device=cfg.dover.device)

    print(
        f"device={dover.device} gpu={gpu_name()} videos={len(videos)} "
        f"replicate={cfg.replicate} dover_bs={cfg.dover.batch_size} "
        f"qwen_bs={cfg.qwen.batch_size}"
    )
    dover.load()
    dover_times: list[float] = []
    dover_scores: dict[str, dict] = {}
    dover_t0 = time.perf_counter()
    n_done = 0
    for batch in tqdm(list(_chunks(jobs, cfg.dover.batch_size)), desc="DOVER"):
        paths = [path for _, path in batch]
        t0 = time.perf_counter()
        scores = dover.score_batch(paths)
        sync(dover.device)
        elapsed = time.perf_counter() - t0
        per_item = elapsed / max(len(batch), 1)
        for (copy, path), item in zip(batch, scores):
            if n_done >= cfg.warmup_videos:
                dover_times.append(per_item)
            if copy == 0 and path.name not in dover_scores:
                dover_scores[path.name] = {**item, "elapsed": per_item}
            n_done += 1
    dover_stage_sec = time.perf_counter() - dover_t0
    dover.unload()

    print(f"Qwen model={cfg.qwen.model_id}")
    qwen.load()
    unique_results: dict[str, VideoResult] = {}
    qwen_times: list[float] = []
    qwen_t0 = time.perf_counter()
    n_done = 0
    prefetch = ThreadPoolExecutor(max_workers=2)
    job_batches = list(_chunks(jobs, cfg.qwen.batch_size))
    next_frames = prefetch.submit(_prepare_batch, qwen, job_batches[0]) if job_batches else None
    try:
        for i, batch in enumerate(tqdm(job_batches, desc="Qwen3-VL")):
            prepared = next_frames.result()
            if i + 1 < len(job_batches):
                next_frames = prefetch.submit(_prepare_batch, qwen, job_batches[i + 1])
            t0 = time.perf_counter()
            texts = qwen.caption_batch(prepared)
            sync(qwen.device)
            elapsed = time.perf_counter() - t0
            per_item = elapsed / max(len(batch), 1)
            for (copy, path), text in zip(batch, texts):
                if n_done >= cfg.warmup_videos:
                    qwen_times.append(per_item)
                if copy == 0 and path.name not in unique_results:
                    meta = probe_video(path)
                    scores = dover_scores[path.name]
                    unique_results[path.name] = VideoResult(
                        video_id=path.stem,
                        source=str(path.relative_to(cfg.input_dir)),
                        duration_sec=round(meta["duration"], 3),
                        width=meta["width"],
                        height=meta["height"],
                        technical=round(scores["technical"], 4),
                        aesthetic=round(scores["aesthetic"], 4),
                        overall=round(scores["overall"], 4),
                        caption=text,
                        dover_sec=round(scores["elapsed"], 3),
                        qwen_sec=round(per_item, 3),
                    )
                n_done += 1
    finally:
        prefetch.shutdown(wait=False)
        qwen.unload()
    qwen_stage_sec = time.perf_counter() - qwen_t0

    results = [unique_results[path.name] for path in videos]
    throughput = _throughput_stats(
        n_unique=len(videos),
        n_jobs=len(jobs),
        warmup=cfg.warmup_videos,
        dover_times=dover_times,
        qwen_times=qwen_times,
        dover_stage_sec=dover_stage_sec,
        qwen_stage_sec=qwen_stage_sec,
        device=dover.device,
        attn=qwen.attn_implementation,
    )
    payload = {
        "results": [asdict(item) for item in results],
        "throughput": throughput,
        "config": {
            "replicate": cfg.replicate,
            "warmup_videos": cfg.warmup_videos,
            "qwen_model": cfg.qwen.model_id,
            "qwen_num_frames": cfg.qwen.num_frames,
            "qwen_max_side": cfg.qwen.max_side,
            "qwen_max_tokens": cfg.qwen.max_tokens,
            "qwen_batch_size": cfg.qwen.batch_size,
            "dover_batch_size": cfg.dover.batch_size,
            "device": dover.device,
            "gpu": gpu_name(),
            "attn_implementation": qwen.attn_implementation,
        },
    }
    _write_outputs(cfg.output_dir, payload)
    release_memory()
    return payload


def _prepare_batch(qwen: QwenCaptioner, batch: list[tuple[int, Path]]):
    return [(path, qwen.prepare_frames(path)) for _, path in batch]


def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def _throughput_stats(
    n_unique: int,
    n_jobs: int,
    warmup: int,
    dover_times: list[float],
    qwen_times: list[float],
    dover_stage_sec: float,
    qwen_stage_sec: float,
    device: str,
    attn: str | None,
) -> dict:
    dover_mean = _mean(dover_times)
    qwen_mean = _mean(qwen_times)
    per_video = dover_mean + qwen_mean
    videos_per_sec = (1.0 / per_video) if per_video > 0 else 0.0
    return {
        "device": device,
        "gpu": gpu_name(),
        "attn_implementation": attn,
        "unique_videos": n_unique,
        "processed_jobs": n_jobs,
        "warmup_videos": warmup,
        "stable_samples": min(len(dover_times), len(qwen_times)),
        "dover_sec_per_video": round(dover_mean, 4),
        "qwen_sec_per_video": round(qwen_mean, 4),
        "pipeline_sec_per_video": round(per_video, 4),
        "videos_per_sec": round(videos_per_sec, 6),
        "videos_per_hour": round(videos_per_sec * 3600, 2),
        "dover_stage_sec": round(dover_stage_sec, 3),
        "qwen_stage_sec": round(qwen_stage_sec, 3),
        "note": "стабильные числа: среднее после warmup, по репликам датасета, cuda.synchronize()",
    }


def _write_outputs(output_dir: Path, payload: dict) -> None:
    (output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "video_id,source,duration_sec,width,height,technical,aesthetic,overall,caption",
    ]
    for item in payload["results"]:
        caption = item["caption"].replace('"', "'")
        lines.append(
            f'{item["video_id"]},{item["source"]},{item["duration_sec"]},{item["width"]},'
            f'{item["height"]},{item["technical"]},{item["aesthetic"]},{item["overall"]},"{caption}"'
        )
    (output_dir / "results.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output_dir / "throughput.json").write_text(
        json.dumps(payload["throughput"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_markdown(output_dir / "RESULTS.md", payload)


def _write_markdown(path: Path, payload: dict) -> None:
    t = payload["throughput"]
    gpu = t.get("gpu") or t.get("device")
    rows = [
        "# Результаты Video Processing",
        "",
        f"Устройство: **{gpu}**. Attention: `{t.get('attn_implementation')}`.",
        "",
        "## Пропускная способность (после warmup)",
        "",
        f"- Обработано джобов: **{t['processed_jobs']}** (уникальных видео: {t['unique_videos']}, replicate учтён)",
        f"- Стабильных замеров: **{t['stable_samples']}**",
        f"- DOVER: **{t['dover_sec_per_video']:.3f} с/видео**",
        f"- Qwen3-VL: **{t['qwen_sec_per_video']:.3f} с/видео**",
        f"- Пайплайн: **{t['pipeline_sec_per_video']:.3f} с/видео** → **{t['videos_per_sec']:.4f} видео/с** ({t['videos_per_hour']:.1f} видео/ч)",
        "",
        "## Оценки и описания",
        "",
        "| Видео | Technical | Aesthetic | Overall | Описание |",
        "|---|---:|---:|---:|---|",
    ]
    for item in payload["results"]:
        caption = item["caption"].replace("|", "\\|")
        rows.append(
            f"| {item['video_id']} | {item['technical']:.4f} | {item['aesthetic']:.4f} | "
            f"{item['overall']:.4f} | {caption} |"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
