from __future__ import annotations

import json
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path

from tqdm import tqdm

from video_processing.config import AppConfig
from video_processing.decode import probe_video
from video_processing.device import gpu_name, release_memory, sync
from video_processing.dover_infer import DoverScorer
from video_processing.qwen_infer import QwenCaptioner, make_captioner


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


# vLLM process reserves this slice; DOVER allocates in the parent from the rest.
# 2B bf16 does not need 70%. At batch 16 the DOVER forward asks for ~11 GiB

OVERLAP_VLLM_FRACTION = 0.50


def _advance_overlap(state: dict, batch_len: int, now: float, warmup: int) -> None:
    """Start the stable clock when the first batch that covers warmup finishes."""
    state["finished"] = int(state.get("finished", 0)) + batch_len
    if state.get("t0") is None and state["finished"] >= warmup:
        state["t0"] = now
        return
    if state.get("t0") is not None:
        state["stable_jobs"] = int(state.get("stable_jobs", 0)) + batch_len
        state["t1"] = now


def run_pipeline(
    cfg: AppConfig,
    *,
    write_outputs: bool = True,
    skip_qwen: bool = False,
    cached_dover_scores: dict | None = None,
    cached_dover_times: list[float] | None = None,
    cached_dover_stage_sec: float | None = None,
) -> dict:
    videos = list_videos(cfg.input_dir)
    jobs = expand_for_throughput(videos, cfg.replicate)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    dover = DoverScorer(cfg.dover)
    qwen = make_captioner(cfg.qwen, device=cfg.dover.device)

    print(
        f"device={dover.device} gpu={gpu_name()} videos={len(videos)} "
        f"replicate={cfg.replicate} dover_bs={cfg.dover.batch_size} "
        f"qwen_backend={cfg.qwen.backend} qwen_bs={cfg.qwen.batch_size} "
        f"parallel={cfg.parallel and not skip_qwen and cached_dover_scores is None} "
        f"skip_qwen={skip_qwen} reuse_dover={cached_dover_scores is not None}"
    )

    overlapped = False
    overlap_wall_sec = 0.0
    overlap_stable_jobs = 0
    if cfg.parallel and not skip_qwen and cached_dover_scores is None:
        overlapped = True
        bundle = _run_overlapped(cfg, dover, qwen, jobs)
        dover_scores = bundle["dover_scores"]
        dover_times = bundle["dover_times"]
        dover_stage_sec = bundle["dover_stage_sec"]
        unique_results = bundle["unique_results"]
        qwen_times = bundle["qwen_times"]
        qwen_stage_sec = bundle["qwen_stage_sec"]
        overlap_wall_sec = bundle["overlap_wall_sec"]
        overlap_stable_jobs = bundle["overlap_stable_jobs"]
    elif cached_dover_scores is not None:
        dover_scores = cached_dover_scores
        dover_times = list(cached_dover_times or [])
        dover_stage_sec = float(cached_dover_stage_sec or 0.0)
    else:
        dover.load()
        dover_times = []
        dover_scores = {}
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

    if not overlapped:
        unique_results = {}
        qwen_times = []
        qwen_stage_sec = 0.0
        if skip_qwen:
            for path in videos:
                scores = dover_scores[path.name]
                meta = probe_video(path)
                unique_results[path.name] = VideoResult(
                    video_id=path.stem,
                    source=str(path.relative_to(cfg.input_dir)),
                    duration_sec=round(meta["duration"], 3),
                    width=meta["width"],
                    height=meta["height"],
                    technical=round(scores["technical"], 4),
                    aesthetic=round(scores["aesthetic"], 4),
                    overall=round(scores["overall"], 4),
                    caption="",
                    dover_sec=round(scores["elapsed"], 3),
                    qwen_sec=0.0,
                )
        else:
            print(f"Qwen model={cfg.qwen.model_id} backend={cfg.qwen.backend}")
            n_done = 0
            prefetch = ThreadPoolExecutor(max_workers=2)
            job_batches = list(_chunks(jobs, cfg.qwen.batch_size))
            next_frames = prefetch.submit(_prepare_batch, qwen, job_batches[0]) if job_batches else None
            try:
                qwen.load()
                qwen_t0 = time.perf_counter()
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
        overlapped=overlapped,
        overlap_wall_sec=overlap_wall_sec,
        overlap_stable_jobs=overlap_stable_jobs,
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
            "qwen_backend": cfg.qwen.backend,
            "qwen_batch_size": cfg.qwen.batch_size,
            "qwen_max_num_seqs": cfg.qwen.max_num_seqs,
            "qwen_max_model_len": cfg.qwen.max_model_len,
            "parallel": overlapped,
            "dover_batch_size": cfg.dover.batch_size,
            "device": dover.device,
            "gpu": gpu_name(),
            "attn_implementation": qwen.attn_implementation,
        },
        "dover_scores": dover_scores,
        "dover_times": dover_times,
        "dover_stage_sec": dover_stage_sec,
    }
    if write_outputs:
        _write_outputs(cfg.output_dir, payload)
    release_memory()
    return payload


def _run_overlapped(cfg: AppConfig, dover: DoverScorer, qwen, jobs) -> dict:
    """Both models get the full job list from t=0.

    vLLM is a child process and already owns its VRAM slice; DOVER stays in
    the parent. Captions do not wait for scores. A job counts as done for
    wall-clock when both stages have finished that index.
    """
    stop = threading.Event()
    errors: list[BaseException] = []
    lock = threading.Lock()
    gpu_lock = threading.Lock() if cfg.qwen.backend != "vllm" else nullcontext()
    dover_scores: dict = {}
    dover_times: list[float] = []
    qwen_times: list[float] = []
    captions: dict[str, tuple[str, float]] = {}
    overlap = {"finished": 0, "stable_jobs": 0, "t0": None, "t1": None}
    dover_done: dict[int, float] = {}
    qwen_done: dict[int, float] = {}

    def _mark(kind: str, start: int, count: int, now: float) -> None:
        with lock:
            store = dover_done if kind == "dover" else qwen_done
            newly = 0
            done_at = now
            for index in range(start, start + count):
                store[index] = now
                if index in dover_done and index in qwen_done:
                    newly += 1
                    done_at = max(dover_done[index], qwen_done[index])
            if newly:
                _advance_overlap(overlap, newly, done_at, cfg.warmup_videos)

    def produce() -> None:
        n_done = 0
        stage_t0 = time.perf_counter()
        try:
            for batch in tqdm(
                list(_chunks(jobs, cfg.dover.batch_size)), desc="DOVER", position=0
            ):
                if stop.is_set():
                    break
                paths = [path for _, path in batch]
                t0 = time.perf_counter()
                with gpu_lock:
                    scores = dover.score_batch(paths)
                    sync(dover.device)
                elapsed = time.perf_counter() - t0
                per_item = elapsed / max(len(batch), 1)
                start = n_done
                for (copy, path), item in zip(batch, scores):
                    if n_done >= cfg.warmup_videos:
                        dover_times.append(per_item)
                    if copy == 0 and path.name not in dover_scores:
                        dover_scores[path.name] = {**item, "elapsed": per_item}
                    n_done += 1
                _mark("dover", start, len(batch), time.perf_counter())
        except BaseException as exc:
            errors.append(exc)
            stop.set()
        finally:
            overlap["dover_stage_sec"] = time.perf_counter() - stage_t0

    def consume() -> None:
        prefetch = ThreadPoolExecutor(max_workers=2)
        job_batches = list(_chunks(jobs, cfg.qwen.batch_size))
        next_frames = prefetch.submit(_prepare_batch, qwen, job_batches[0]) if job_batches else None
        n_done = 0
        stage_t0 = time.perf_counter()
        try:
            for i, batch in enumerate(tqdm(job_batches, desc="Qwen3-VL", position=1)):
                if stop.is_set():
                    break
                prepared = next_frames.result()
                if i + 1 < len(job_batches):
                    next_frames = prefetch.submit(_prepare_batch, qwen, job_batches[i + 1])
                t0 = time.perf_counter()
                with gpu_lock:
                    texts = qwen.caption_batch(prepared)
                    if cfg.qwen.backend != "vllm":
                        sync(qwen.device)
                elapsed = time.perf_counter() - t0
                per_item = elapsed / max(len(batch), 1)
                start = n_done
                for (copy, path), text in zip(batch, texts):
                    if n_done >= cfg.warmup_videos:
                        qwen_times.append(per_item)
                    if copy == 0 and path.name not in captions:
                        captions[path.name] = (text, per_item)
                    n_done += 1
                _mark("qwen", start, len(batch), time.perf_counter())
        except BaseException as exc:
            errors.append(exc)
            stop.set()
        finally:
            prefetch.shutdown(wait=False)
            overlap["qwen_stage_sec"] = time.perf_counter() - stage_t0

    qwen_thread = threading.Thread(target=consume, name="qwen-overlap")
    dover_thread = threading.Thread(target=produce, name="dover-overlap")
    try:
        _load_qwen(qwen, cfg, overlapped=True)
        dover.load()
        qwen_thread.start()
        dover_thread.start()
        qwen_thread.join()
        dover_thread.join()
    finally:
        stop.set()
        if qwen_thread.is_alive():
            qwen_thread.join()
        if dover_thread.is_alive():
            dover_thread.join()
        dover.unload()
        qwen.unload()
    if errors:
        raise errors[0]

    unique_results: dict[str, VideoResult] = {}
    for copy, path in jobs:
        if copy != 0 or path.name in unique_results:
            continue
        scores = dover_scores[path.name]
        text, qwen_sec = captions.get(path.name, ("", 0.0))
        meta = probe_video(path)
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
            qwen_sec=round(qwen_sec, 3),
        )
    wall = 0.0
    stable = int(overlap.get("stable_jobs") or 0)
    if overlap.get("t0") is not None and overlap.get("t1") is not None:
        wall = float(overlap["t1"]) - float(overlap["t0"])
    return {
        "dover_scores": dover_scores,
        "dover_times": dover_times,
        "dover_stage_sec": float(overlap.get("dover_stage_sec") or 0.0),
        "unique_results": unique_results,
        "qwen_times": qwen_times,
        "qwen_stage_sec": float(overlap.get("qwen_stage_sec") or 0.0),
        "overlap_wall_sec": wall,
        "overlap_stable_jobs": stable,
    }


def _load_qwen(qwen, cfg: AppConfig, *, overlapped: bool) -> None:
    if overlapped and cfg.qwen.backend == "vllm":
        util = min(float(cfg.qwen.gpu_memory_utilization), OVERLAP_VLLM_FRACTION)
        print(f"overlap: vLLM gpu_memory_utilization={util:.2f}, DOVER в оставшейся памяти")
        qwen.load(gpu_memory_utilization=util)
        return
    qwen.load()


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
    overlapped: bool = False,
    overlap_wall_sec: float = 0.0,
    overlap_stable_jobs: int = 0,
) -> dict:
    dover_mean = _mean(dover_times)
    qwen_mean = _mean(qwen_times)
    if overlapped and overlap_stable_jobs > 0 and overlap_wall_sec > 0:
        per_video = overlap_wall_sec / overlap_stable_jobs
        videos_per_sec = overlap_stable_jobs / overlap_wall_sec
        note = (
            "стадии идут одновременно: видео/с — wall-clock после прогрева, "
            "а не сумма DOVER и Qwen"
        )
    else:
        per_video = dover_mean + qwen_mean
        videos_per_sec = (1.0 / per_video) if per_video > 0 else 0.0
        note = "стабильные числа: среднее после warmup, по репликам датасета, cuda.synchronize()"
    return {
        "device": device,
        "gpu": gpu_name(),
        "attn_implementation": attn,
        "overlapped": overlapped,
        "unique_videos": n_unique,
        "processed_jobs": n_jobs,
        "warmup_videos": warmup,
        "stable_samples": overlap_stable_jobs if overlapped else min(len(dover_times), len(qwen_times)),
        "dover_sec_per_video": round(dover_mean, 4),
        "qwen_sec_per_video": round(qwen_mean, 4),
        "pipeline_sec_per_video": round(per_video, 4),
        "videos_per_sec": round(videos_per_sec, 6),
        "videos_per_hour": round(videos_per_sec * 3600, 2),
        "dover_stage_sec": round(dover_stage_sec, 3),
        "qwen_stage_sec": round(qwen_stage_sec, 3),
        "note": note,
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
        f"- Режим: **{'DOVER ∥ Qwen' if t.get('overlapped') else 'последовательно'}**",
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
