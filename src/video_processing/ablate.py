from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from video_processing.config import AppConfig, ROOT
from video_processing.device import release_memory
from video_processing.pipeline import run_pipeline

BOOST = 1.05
QUALITY_POINTS = [
    {"name": "f4_s256", "num_frames": 4, "max_side": 256},
    {"name": "f8_s384", "num_frames": 8, "max_side": 384},
    {"name": "f16_s384", "num_frames": 16, "max_side": 384},
]


# Не основной прогон. Только python -m video_processing --ablate.
def run_ablation(cfg: AppConfig) -> dict:
    out_dir = cfg.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    report: dict = {"quality": [], "qwen_batch": [], "dover_batch": [], "chosen": {}}

    dover_rows, dover_cache = _sweep_dover_batch(cfg, out_dir)
    report["dover_batch"] = dover_rows
    best_dover_bs = _best_batch(dover_rows, "dover_batch_size")
    if best_dover_bs is not None:
        cfg = replace(cfg, dover=replace(cfg.dover, batch_size=best_dover_bs))
        report["chosen"]["dover_batch_size"] = best_dover_bs

    quality_rows = _sweep_quality(cfg, out_dir, dover_cache)
    report["quality"] = quality_rows
    chosen_quality = _choose_quality(quality_rows)
    report["chosen"]["quality"] = chosen_quality
    if chosen_quality:
        cfg = replace(
            cfg,
            qwen=replace(
                cfg.qwen,
                num_frames=int(chosen_quality["num_frames"]),
                max_side=int(chosen_quality["max_side"]),
            ),
        )

    qwen_rows = _sweep_qwen_batch(cfg, out_dir, dover_cache)
    report["qwen_batch"] = qwen_rows
    best_qwen_bs = _best_batch(qwen_rows, "qwen_batch_size")
    if best_qwen_bs is not None:
        cfg = replace(cfg, qwen=replace(cfg.qwen, batch_size=best_qwen_bs))
        report["chosen"]["qwen_batch_size"] = best_qwen_bs

    _write_ablation(out_dir, report)
    _apply_chosen(cfg)
    return report


def _sweep_dover_batch(cfg: AppConfig, out_dir: Path) -> tuple[list[dict], dict]:
    rows = []
    cache = {"scores": None, "times": None, "stage_sec": None, "sec_per_video": None}
    sizes = [8, 10]
    for size in sizes:
        row = _run_one(
            cfg,
            out_dir / f"ablate_dover_bs{size}",
            dover_batch=size,
            skip_qwen=True,
        )
        rows.append(row)
        if row.get("ok") and cache["scores"] is None:
            cache = {
                "scores": row.pop("_scores"),
                "times": row.pop("_times"),
                "stage_sec": row.get("dover_stage_sec"),
                "sec_per_video": row.get("dover_sec_per_video"),
            }
        else:
            row.pop("_scores", None)
            row.pop("_times", None)
    if _boosted(rows, 8, 10, "dover_batch_size"):
        extra = _run_one(
            cfg,
            out_dir / "ablate_dover_bs16",
            dover_batch=16,
            skip_qwen=True,
        )
        extra.pop("_scores", None)
        extra.pop("_times", None)
        rows.append(extra)
    return rows, cache


def _sweep_quality(cfg: AppConfig, out_dir: Path, dover_cache: dict) -> list[dict]:
    rows = []
    for point in QUALITY_POINTS:
        row = _run_one(
            cfg,
            out_dir / f"ablate_{point['name']}",
            num_frames=point["num_frames"],
            max_side=point["max_side"],
            qwen_batch=8,
            cached=dover_cache,
        )
        row["name"] = point["name"]
        rows.append(row)
    return rows


def _sweep_qwen_batch(cfg: AppConfig, out_dir: Path, dover_cache: dict) -> list[dict]:
    rows = []
    for size in [8, 10]:
        row = _run_one(
            cfg,
            out_dir / f"ablate_qwen_bs{size}",
            qwen_batch=size,
            cached=dover_cache,
        )
        row["name"] = f"qwen_bs{size}"
        rows.append(row)
    if _boosted(rows, 8, 10, "qwen_batch_size"):
        extra = _run_one(
            cfg,
            out_dir / "ablate_qwen_bs16",
            qwen_batch=16,
            cached=dover_cache,
        )
        extra["name"] = "qwen_bs16"
        rows.append(extra)
    return rows


def _run_one(
    cfg: AppConfig,
    out_dir: Path,
    *,
    dover_batch: int | None = None,
    qwen_batch: int | None = None,
    num_frames: int | None = None,
    max_side: int | None = None,
    skip_qwen: bool = False,
    cached: dict | None = None,
) -> dict:
    run_cfg = deepcopy(cfg)
    run_cfg.output_dir = out_dir
    if dover_batch is not None:
        run_cfg.dover.batch_size = dover_batch
    if qwen_batch is not None:
        run_cfg.qwen.batch_size = qwen_batch
    if num_frames is not None:
        run_cfg.qwen.num_frames = num_frames
    if max_side is not None:
        run_cfg.qwen.max_side = max_side
    run_cfg.warmup_videos = max(
        run_cfg.warmup_videos,
        run_cfg.dover.batch_size if skip_qwen else run_cfg.qwen.batch_size,
    )
    row = {
        "dover_batch_size": run_cfg.dover.batch_size,
        "qwen_batch_size": run_cfg.qwen.batch_size,
        "num_frames": run_cfg.qwen.num_frames,
        "max_side": run_cfg.qwen.max_side,
        "ok": False,
    }
    try:
        payload = run_pipeline(
            run_cfg,
            write_outputs=True,
            skip_qwen=skip_qwen,
            cached_dover_scores=None if cached is None else cached.get("scores"),
            cached_dover_times=None if cached is None else cached.get("times"),
            cached_dover_stage_sec=None if cached is None else cached.get("stage_sec"),
        )
        t = payload["throughput"]
        captions = [item["caption"] for item in payload["results"] if item.get("caption")]
        row.update(
            {
                "ok": True,
                "oom": False,
                "videos_per_sec": t["videos_per_sec"],
                "pipeline_sec_per_video": t["pipeline_sec_per_video"],
                "dover_sec_per_video": t["dover_sec_per_video"],
                "qwen_sec_per_video": t["qwen_sec_per_video"],
                "mean_caption_words": _mean_words(captions),
                "sample_caption": captions[0][:240] if captions else "",
                "_scores": payload.get("dover_scores"),
                "_times": payload.get("dover_times"),
                "dover_stage_sec": payload.get("dover_stage_sec"),
            }
        )
    except Exception as exc:
        oom = "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"
        row.update({"ok": False, "oom": oom, "error": f"{type(exc).__name__}: {exc}"})
        release_memory()
    return row


def _mean_words(captions: list[str]) -> float:
    if not captions:
        return 0.0
    return round(sum(len(text.split()) for text in captions) / len(captions), 1)


def _boosted(rows: list[dict], small: int, large: int, batch_key: str) -> bool:
    by_bs = {row[batch_key]: row for row in rows if row.get("ok")}
    a, b = by_bs.get(small), by_bs.get(large)
    if not a or not b:
        return False
    return float(b.get("videos_per_sec") or 0) >= float(a.get("videos_per_sec") or 0) * BOOST


def _best_batch(rows: list[dict], batch_key: str) -> int | None:
    ok = [row for row in rows if row.get("ok") and row.get("videos_per_sec")]
    if not ok:
        return None
    winner = max(ok, key=lambda row: float(row["videos_per_sec"]))
    return int(winner[batch_key])


def _choose_quality(rows: list[dict]) -> dict | None:
    ok = [r for r in rows if r.get("ok")]
    if not ok:
        return None
    mid = next((r for r in ok if r.get("name") == "f8_s384"), ok[len(ok) // 2])
    return {
        "name": mid.get("name"),
        "num_frames": mid["num_frames"],
        "max_side": mid["max_side"],
        "reason": "баланс: 8 кадров / 384, если точка посчиталась; иначе первая успешная",
    }


def _write_ablation(out_dir: Path, report: dict) -> None:
    public = {
        key: [{k: v for k, v in row.items() if not str(k).startswith("_")} for row in rows]
        if isinstance(rows, list)
        else rows
        for key, rows in report.items()
    }
    (out_dir / "ablation.json").write_text(
        json.dumps(public, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    lines = [
        "# Ablation: скорость vs качество",
        "",
        "DOVER-оценки от visual-бюджета Qwen не зависят. Для описаний прокси — средняя длина в словах плюс пример caption.",
        "Батч 16 запускается только если 10 даёт ≥5% к `videos/sec` относительно 8.",
        "",
        "## Кадры и разрешение Qwen",
        "",
        "| Точка | Кадры | Сторона | Qwen с/видео | Пайплайн видео/с | Слов в описании | Ок |",
        "|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report["quality"]:
        lines.append(_quality_line(row))
    lines += [
        "",
        "## Батч Qwen",
        "",
        "| Батч | Qwen с/видео | Пайплайн видео/с | OOM |",
        "|---:|---:|---:|:---:|",
    ]
    for row in report["qwen_batch"]:
        lines.append(
            f"| {row.get('qwen_batch_size')} | {_fmt(row.get('qwen_sec_per_video'))} | "
            f"{_fmt(row.get('videos_per_sec'))} | {'да' if row.get('oom') else 'нет'} |"
        )
    lines += [
        "",
        "## Батч DOVER",
        "",
        "| Батч | DOVER с/видео | Пайплайн видео/с | OOM |",
        "|---:|---:|---:|:---:|",
    ]
    for row in report["dover_batch"]:
        lines.append(
            f"| {row.get('dover_batch_size')} | {_fmt(row.get('dover_sec_per_video'))} | "
            f"{_fmt(row.get('videos_per_sec'))} | {'да' if row.get('oom') else 'нет'} |"
        )
    chosen = report.get("chosen") or {}
    lines += [
        "",
        "## Выбранная точка",
        "",
        f"- Qwen visual: `{chosen.get('quality')}`",
        f"- qwen.batch_size: **{chosen.get('qwen_batch_size')}**",
        f"- dover.batch_size: **{chosen.get('dover_batch_size')}**",
        "",
        "Эти значения записаны в `configs/default.yaml`.",
        "",
        "### Примеры описаний",
        "",
    ]
    for row in report["quality"]:
        if row.get("sample_caption"):
            lines.append(f"- **{row.get('name')}**: {row['sample_caption']}")
    (out_dir / "ABLATION.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _quality_line(row: dict) -> str:
    return (
        f"| {row.get('name')} | {row.get('num_frames')} | {row.get('max_side')} | "
        f"{_fmt(row.get('qwen_sec_per_video'))} | {_fmt(row.get('videos_per_sec'))} | "
        f"{_fmt(row.get('mean_caption_words'))} | {'да' if row.get('ok') else 'нет'} |"
    )


def _fmt(value) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.4f}" if value < 10 else f"{value:.2f}"
    return str(value)


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _apply_chosen(cfg: AppConfig) -> None:
    path = ROOT / "configs" / "default.yaml"
    text = path.read_text(encoding="utf-8")
    text = _replace_yaml_int(text, "batch_size", cfg.dover.batch_size, section="dover")
    text = _replace_yaml_int(text, "batch_size", cfg.qwen.batch_size, section="qwen")
    text = _replace_yaml_int(text, "num_frames", cfg.qwen.num_frames, section="qwen")
    text = _replace_yaml_int(text, "max_side", cfg.qwen.max_side, section="qwen")
    path.write_text(text, encoding="utf-8")
    chosen = {
        "dover": {"batch_size": cfg.dover.batch_size},
        "qwen": {
            "batch_size": cfg.qwen.batch_size,
            "num_frames": cfg.qwen.num_frames,
            "max_side": cfg.qwen.max_side,
        },
    }
    (cfg.output_dir / "chosen.yaml").write_text(
        json.dumps(chosen, indent=2), encoding="utf-8"
    )


def _replace_yaml_int(text: str, key: str, value: int, section: str) -> str:
    lines = text.splitlines()
    in_section = False
    out = []
    for line in lines:
        stripped = line.split("#", 1)[0]
        if stripped.strip() and not stripped.startswith(" ") and stripped.strip().endswith(":"):
            in_section = stripped.strip()[:-1] == section
        if in_section and stripped.strip().startswith(f"{key}:"):
            indent = line[: len(line) - len(line.lstrip())]
            out.append(f"{indent}{key}: {value}")
            continue
        out.append(line)
    return "\n".join(out) + "\n"
