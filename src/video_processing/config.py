from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


@dataclass
class DoverConfig:
    weights: Path
    config: Path
    device: str = "auto"
    batch_size: int = 8
    decode_workers: int = 0


@dataclass
class QwenConfig:
    model_id: str
    prompt: str
    num_frames: int = 8
    max_side: int = 384
    max_tokens: int = 256
    temperature: float = 0.0
    repetition_penalty: float = 1.0
    batch_size: int = 8
    attn_implementation: str = "auto"
    backend: str = "vllm"
    max_model_len: int = 4096
    max_num_seqs: int = 8
    gpu_memory_utilization: float = 0.90
    dtype: str = "bfloat16"
    port: int = 8000
    api_base: str = ""


@dataclass
class AppConfig:
    input_dir: Path
    output_dir: Path
    replicate: int
    warmup_videos: int
    seed: int
    parallel: bool
    dover: DoverConfig
    qwen: QwenConfig


# [3] Путь из yaml. Относительный считается от корня репозитория.
def _as_path(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = ROOT / path
    return path


# [2] configs/default.yaml → AppConfig. Пути собирает через [3].
def load_config(path: str | Path | None) -> AppConfig:
    cfg_path = _as_path(path or ROOT / "configs" / "default.yaml")
    raw: dict[str, Any] = yaml.safe_load(cfg_path.read_text()) or {}
    dover_raw = raw.get("dover") or {}
    qwen_raw = raw.get("qwen") or {}
    return AppConfig(
        input_dir=_as_path(raw.get("input_dir", "video")),
        output_dir=_as_path(raw.get("output_dir", "results")),
        replicate=int(raw.get("replicate", 1)),
        warmup_videos=int(raw.get("warmup_videos", 1)),
        seed=int(raw.get("seed", 42)),
        parallel=bool(raw.get("parallel", True)),
        dover=DoverConfig(
            weights=_as_path(dover_raw.get("weights", "weights/DOVER.pth")),
            config=_as_path(dover_raw.get("config", "third_party/DOVER/dover.yml")),
            device=str(dover_raw.get("device", "auto")),
            batch_size=int(dover_raw.get("batch_size", 8)),
            decode_workers=int(dover_raw.get("decode_workers", 0)),
        ),
        qwen=QwenConfig(
            model_id=str(qwen_raw.get("model_id", "Qwen/Qwen3-VL-2B-Instruct")),
            prompt=str(
                qwen_raw.get(
                    "prompt",
                    "Сгенерируй детальное описание всего, что происходит на видео",
                )
            ),
            num_frames=int(qwen_raw.get("num_frames", 8)),
            max_side=int(qwen_raw.get("max_side", 384)),
            max_tokens=int(qwen_raw.get("max_tokens", 256)),
            temperature=float(qwen_raw.get("temperature", 0.0)),
            repetition_penalty=float(qwen_raw.get("repetition_penalty", 1.0)),
            batch_size=int(qwen_raw.get("batch_size", 8)),
            attn_implementation=str(qwen_raw.get("attn_implementation", "auto")),
            backend=str(qwen_raw.get("backend", "vllm")),
            max_model_len=int(qwen_raw.get("max_model_len", 4096)),
            max_num_seqs=int(qwen_raw.get("max_num_seqs", 8)),
            gpu_memory_utilization=float(qwen_raw.get("gpu_memory_utilization", 0.90)),
            dtype=str(qwen_raw.get("dtype", "bfloat16")),
            port=int(qwen_raw.get("port", 8000)),
            api_base=str(qwen_raw.get("api_base") or ""),
        ),
    )
