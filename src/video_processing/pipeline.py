from __future__ import annotations

import csv
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

# Порядок основного прогона. Номер стоит над функцией.
# 1 main → 2 load_config → 4 configure_cuda → 5 run_pipeline
# 6 list_videos → 7 expand → 8 DoverScorer → 10 make_captioner → 13 _run_overlapped
# 14 _load_qwen → 15..18 vLLM → 19..24 загрузка DOVER
# дальше одновременно:
#   25d produce → 27s iter_scores → 28p prepare (пул) → 34 ffmpeg → 37f на GPU → 40 fuse
#   25q consume → 44 кадры → 49 caption_batch → 52 POST
# 53–54 unload → 55 скорость → 57 запись → 59 release_memory


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
    caption_finish_reason: str
    dover_sec: float
    qwen_sec: float


# [6] Файлы video/ с известным расширением, по имени.
def list_videos(input_dir: Path) -> list[Path]:
    videos = [
        path
        for path in sorted(input_dir.iterdir())
        if path.is_file() and path.suffix.lower() in VIDEO_EXTS
    ]
    if not videos:
        raise FileNotFoundError(f"в {input_dir} нет видео")
    return videos


# [7] 10 роликов × replicate. В таблицу потом пойдёт только копия 0.
def expand_for_throughput(videos: list[Path], replicate: int) -> list[tuple[int, Path]]:
    replicate = max(1, replicate)
    return [(copy, path) for copy in range(replicate) for path in videos]


# [26] Режет список джобов на батчи DOVER и, отдельно, Qwen.
def _chunks(items: list, size: int):
    size = max(1, size)
    for start in range(0, len(items), size):
        yield items[start : start + size]


# vLLM process reserves this slice; DOVER allocates in the parent from the rest.
# 2B bf16 does not need 70%. At batch 16 the DOVER forward asks for ~11 GiB

OVERLAP_VLLM_FRACTION = 0.50


# [42] Часы видео/с по событиям «ролик закончили оба потока».
# Старт — событие, на котором набралось warmup роликов. Стабильное окно кончается,
# когда любой поток исчерпал список: дальше второй идёт без соседа по GPU/CPU,
# и такой хвост в бесконечном потоке роликов не повторяется.
def _steady_window(events: list[tuple[float, int]], warmup: int, end: float | None) -> dict:
    finished = 0
    t0 = t1 = None
    jobs = 0
    for at, count in sorted(events):
        if t0 is None:
            finished += count
            if finished >= warmup:
                t0 = at
            continue
        if end is not None and at > end:
            break
        jobs += count
        t1 = at
    wall = (t1 - t0) if t0 is not None and t1 is not None else 0.0
    return {"jobs": jobs, "wall": wall}


# [5] Основной прогон. При parallel зовёт [13], иначе стадии по очереди.
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
    overlap_info: dict = {}
    load_sec: dict[str, float] = {}
    qwen_truncated = 0
    if cfg.parallel and not skip_qwen and cached_dover_scores is None:
        overlapped = True
        bundle = _run_overlapped(cfg, dover, qwen, jobs)
        dover_scores = bundle["dover_scores"]
        dover_times = bundle["dover_times"]
        dover_stage_sec = bundle["dover_stage_sec"]
        unique_results = bundle["unique_results"]
        qwen_times = bundle["qwen_times"]
        qwen_stage_sec = bundle["qwen_stage_sec"]
        qwen_truncated = bundle["qwen_truncated"]
        overlap_info = bundle["overlap"]
        load_sec = bundle["load_sec"]
    elif cached_dover_scores is not None:
        dover_scores = cached_dover_scores
        dover_times = list(cached_dover_times or [])
        dover_stage_sec = float(cached_dover_stage_sec or 0.0)
    else:
        t_load = time.perf_counter()
        dover.load()
        load_sec["dover"] = time.perf_counter() - t_load
        dover_times = []
        dover_scores = {}
        dover_t0 = last = time.perf_counter()
        n_done = 0
        bar = tqdm(total=len(jobs), desc="DOVER")
        for indices, scores in dover.iter_scores(list(enumerate(path for _, path in jobs))):
            now = time.perf_counter()
            per_item = (now - last) / max(len(indices), 1)
            last = now
            for index, item in zip(indices, scores):
                copy, path = jobs[index]
                if n_done >= cfg.warmup_videos:
                    dover_times.append(per_item)
                if copy == 0 and path.name not in dover_scores:
                    dover_scores[path.name] = {**item, "elapsed": per_item}
                n_done += 1
            bar.update(len(indices))
        bar.close()
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
                    caption_finish_reason="",
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
                t_load = time.perf_counter()
                qwen.load()
                load_sec["qwen"] = time.perf_counter() - t_load
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
                    for (copy, path), (text, finish) in zip(batch, texts):
                        if n_done >= cfg.warmup_videos:
                            qwen_times.append(per_item)
                        qwen_truncated += finish == "length"
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
                                caption_finish_reason=finish,
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
        overlap=overlap_info,
        load_sec=load_sec,
        qwen_truncated=0 if skip_qwen else qwen_truncated,
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


# [13] Поднимает обе модели и с t=0 пускает [25d] и [25q] на одном списке роликов.
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
    captions: dict[str, tuple[str, str, float]] = {}
    overlap: dict = {}
    dover_done: dict[int, float] = {}
    qwen_done: dict[int, float] = {}
    events: list[tuple[float, int]] = []
    last_mark: dict[str, float] = {}
    truncated = [0]

    # [41] Ролик готов для замера, когда отметку поставили и DOVER, и Qwen.
    def _mark(kind: str, indices, now: float) -> None:
        with lock:
            store = dover_done if kind == "dover" else qwen_done
            last_mark[kind] = now
            newly = 0
            for index in indices:
                store[index] = now
                if index in dover_done and index in qwen_done:
                    newly += 1
            if newly:
                events.append((now, newly))

    # [25d] Поток DOVER: пачки [27s] до конца списка.
    def produce() -> None:
        n_done = 0
        stage_t0 = last = time.perf_counter()
        bar = tqdm(total=len(jobs), desc="DOVER", position=0)
        try:
            items = [(index, path) for index, (_, path) in enumerate(jobs)]
            for indices, scores in dover.iter_scores(items, stop=stop, forward_lock=gpu_lock):
                now = time.perf_counter()
                per_item = (now - last) / max(len(indices), 1)
                last = now
                for index, item in zip(indices, scores):
                    copy, path = jobs[index]
                    if n_done >= cfg.warmup_videos:
                        dover_times.append(per_item)
                    if copy == 0 and path.name not in dover_scores:
                        dover_scores[path.name] = {**item, "elapsed": per_item}
                    n_done += 1
                bar.update(len(indices))
                _mark("dover", indices, now)
        except BaseException as exc:
            errors.append(exc)
            stop.set()
        finally:
            bar.close()
            overlap["dover_stage_sec"] = time.perf_counter() - stage_t0

    # [25q] Поток Qwen: кадры [43], затем подписи [49]. Идёт одновременно с [25d].
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
                for (copy, path), (text, finish) in zip(batch, texts):
                    if n_done >= cfg.warmup_videos:
                        qwen_times.append(per_item)
                    truncated[0] += finish == "length"
                    if copy == 0 and path.name not in captions:
                        captions[path.name] = (text, finish, per_item)
                    n_done += 1
                _mark("qwen", range(start, start + len(batch)), time.perf_counter())
        except BaseException as exc:
            errors.append(exc)
            stop.set()
        finally:
            prefetch.shutdown(wait=False)
            overlap["qwen_stage_sec"] = time.perf_counter() - stage_t0

    qwen_thread = threading.Thread(target=consume, name="qwen-overlap")
    dover_thread = threading.Thread(target=produce, name="dover-overlap")
    load_sec: dict[str, float] = {}
    try:
        t_load = time.perf_counter()
        _load_qwen(qwen, cfg, overlapped=True)
        load_sec["qwen"] = time.perf_counter() - t_load
        t_load = time.perf_counter()
        dover.load()
        load_sec["dover"] = time.perf_counter() - t_load
        run_t0 = time.perf_counter()
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
        text, finish, qwen_sec = captions.get(path.name, ("", "", 0.0))
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
            caption_finish_reason=finish,
            dover_sec=round(scores["elapsed"], 3),
            qwen_sec=round(qwen_sec, 3),
        )
    both_active_until = min(last_mark.values()) if len(last_mark) == 2 else None
    steady = _steady_window(events, cfg.warmup_videos, both_active_until)
    with_tail = _steady_window(events, cfg.warmup_videos, None)
    return {
        "dover_scores": dover_scores,
        "dover_times": dover_times,
        "dover_stage_sec": float(overlap.get("dover_stage_sec") or 0.0),
        "unique_results": unique_results,
        "qwen_times": qwen_times,
        "qwen_stage_sec": float(overlap.get("qwen_stage_sec") or 0.0),
        "qwen_truncated": truncated[0],
        "load_sec": load_sec,
        "overlap": {
            "steady": steady,
            "with_tail": with_tail,
            "makespan_sec": (max(t for t, _ in events) - run_t0) if events else 0.0,
            "dover_end_sec": last_mark.get("dover", run_t0) - run_t0,
            "qwen_end_sec": last_mark.get("qwen", run_t0) - run_t0,
        },
    }


# [14] Грузит Qwen. В параллельном режиме режет долю VRAM до OVERLAP_VLLM_FRACTION.
def _load_qwen(qwen, cfg: AppConfig, *, overlapped: bool) -> None:
    if overlapped and cfg.qwen.backend == "vllm":
        util = min(float(cfg.qwen.gpu_memory_utilization), OVERLAP_VLLM_FRACTION)
        print(f"overlap: vLLM gpu_memory_utilization={util:.2f}, DOVER в оставшейся памяти")
        qwen.load(gpu_memory_utilization=util)
        return
    qwen.load()


# [43] Кадры следующего батча Qwen, пока текущий ещё считается. Зовёт [44].
def _prepare_batch(qwen: QwenCaptioner, batch: list[tuple[int, Path]]):
    return [(path, qwen.prepare_frames(path)) for _, path in batch]


# [56] Среднее с/видео по замерам после warmup.
def _mean(values: list[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


# [55] Видео/с по wall-clock перекрытия, не по сумме DOVER и Qwen.
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
    overlap: dict | None = None,
    load_sec: dict | None = None,
    qwen_truncated: int = 0,
) -> dict:
    overlap = overlap or {}
    dover_mean = _mean(dover_times)
    qwen_mean = _mean(qwen_times)
    steady = overlap.get("steady") or {"jobs": 0, "wall": 0.0}
    with_tail = overlap.get("with_tail") or {"jobs": 0, "wall": 0.0}
    extra: dict = {}
    if overlapped and steady["jobs"] > 0 and steady["wall"] > 0:
        stable_samples = steady["jobs"]
        per_video = steady["wall"] / steady["jobs"]
        videos_per_sec = steady["jobs"] / steady["wall"]
        note = (
            "стадии идут одновременно: видео/с — wall-clock после прогрева и только "
            "пока заняты оба потока; хвост, где один поток уже закончил, не входит"
        )
    elif overlapped and with_tail["jobs"] > 0 and with_tail["wall"] > 0:
        stable_samples = with_tail["jobs"]
        per_video = with_tail["wall"] / with_tail["jobs"]
        videos_per_sec = with_tail["jobs"] / with_tail["wall"]
        note = (
            "ВНИМАНИЕ: окно, где заняты оба потока, короче прогрева; число включает "
            "хвост одной стадии и завышено. Поднимите replicate"
        )
    else:
        stable_samples = min(len(dover_times), len(qwen_times))
        per_video = dover_mean + qwen_mean
        videos_per_sec = (1.0 / per_video) if per_video > 0 else 0.0
        note = "стабильные числа: среднее после warmup, по репликам датасета, cuda.synchronize()"
    if overlapped:
        tail_vps = with_tail["jobs"] / with_tail["wall"] if with_tail["wall"] > 0 else 0.0
        extra = {
            "steady_window_sec": round(steady["wall"], 3),
            "videos_per_sec_with_tail": round(tail_vps, 6),
            "makespan_sec": round(float(overlap.get("makespan_sec") or 0.0), 3),
            "dover_end_sec": round(float(overlap.get("dover_end_sec") or 0.0), 3),
            "qwen_end_sec": round(float(overlap.get("qwen_end_sec") or 0.0), 3),
        }
    return {
        "device": device,
        "gpu": gpu_name(),
        "attn_implementation": attn,
        "overlapped": overlapped,
        "unique_videos": n_unique,
        "processed_jobs": n_jobs,
        "warmup_videos": warmup,
        "stable_samples": stable_samples,
        "dover_sec_per_video": round(dover_mean, 4),
        "qwen_sec_per_video": round(qwen_mean, 4),
        "pipeline_sec_per_video": round(per_video, 4),
        "videos_per_sec": round(videos_per_sec, 6),
        "videos_per_hour": round(videos_per_sec * 3600, 2),
        "dover_stage_sec": round(dover_stage_sec, 3),
        "qwen_stage_sec": round(qwen_stage_sec, 3),
        **extra,
        "load_sec": {k: round(v, 2) for k, v in (load_sec or {}).items()},
        "captions_truncated_jobs": qwen_truncated,
        "note": note,
    }


# [57] results.json, results.csv, throughput.json и [58].
def _write_outputs(output_dir: Path, payload: dict) -> None:
    (output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    columns = [
        "video_id", "source", "duration_sec", "width", "height",
        "technical", "aesthetic", "overall", "caption", "caption_finish_reason",
    ]
    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        for item in payload["results"]:
            writer.writerow([item[key] for key in columns])
    (output_dir / "throughput.json").write_text(
        json.dumps(payload["throughput"], ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_markdown(output_dir / "RESULTS.md", payload)


# [58] RESULTS.md: скорость и таблица десяти исходных роликов.
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
        f"- Описаний, обрезанных по max_tokens: **{t.get('captions_truncated_jobs', 0)}** из {t['processed_jobs']}",
    ]
    if "steady_window_sec" in t:
        rows.append(
            f"- Окно, где заняты оба потока: **{t['steady_window_sec']:.1f} с** "
            f"(DOVER до {t['dover_end_sec']:.1f} с, Qwen до {t['qwen_end_sec']:.1f} с). "
            f"С хвостом одной стадии было бы {t['videos_per_sec_with_tail']:.4f} видео/с"
        )
    rows += [
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
