"""Qwen captions through a local vLLM server.

The pipeline starts the server before DOVER so both stay resident: vLLM
keeps a slice of VRAM and DOVER uses the rest. Continuous batching is the
part that helps:
inside one chunk, a short caption frees its slot instead of holding the
whole static batch.
"""

from __future__ import annotations

import base64
import io
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from video_processing.config import ROOT, QwenConfig
from video_processing.decode import frames_as_pil, probe_video, sample_qwen_frames, uniform_indices
from video_processing.device import pick_torch_device


# [17] Команда vllm serve для [16]: dtype и контекст из конфига (bf16, 4096),
# max_pixels = max_side², у input: video — ещё × num_frames.
def vllm_serve_command(cfg: QwenConfig, gpu_memory_utilization: float | None = None) -> list[str]:
    seqs = max(int(cfg.max_num_seqs), int(cfg.batch_size), 1)
    util = cfg.gpu_memory_utilization if gpu_memory_utilization is None else gpu_memory_utilization
    # Startup profiles the vision encoder at processor max_pixels. Qwen's
    # default is millions of pixels per frame and OOMs the engine before /health.
    pixels = max(int(cfg.max_side) * int(cfg.max_side), 4096)
    video = cfg.input == "video"
    if video:
        # У видео Qwen3-VL max_pixels — бюджет на все кадры (T·H·W), не на один.
        # С бюджетом одной картинки 8 кадров ужались бы до ~180×100.
        pixels *= max(int(cfg.num_frames), 1)
    mm_kwargs = json.dumps({"min_pixels": 4096, "max_pixels": pixels})
    return [
        sys.executable,
        "-m",
        "vllm.entrypoints.cli.main",
        "serve",
        cfg.model_id,
        "--dtype",
        cfg.dtype,
        "--max-model-len",
        str(cfg.max_model_len),
        "--max-num-seqs",
        str(seqs),
        "--gpu-memory-utilization",
        f"{util:g}",
        "--limit-mm-per-prompt.image",
        "0" if video else str(cfg.num_frames),
        "--limit-mm-per-prompt.video",
        "1" if video else "0",
        "--mm-processor-kwargs",
        mm_kwargs,
        "--mm-processor-cache-gb",
        "0",
        # Копии replicate побайтно совпадают: с prefix cache их prefill и vision
        # encoder берутся из кэша, и видео/с Qwen завышается. У уникальных роликов
        # общий префикс — только шаблон чата, так что в бою кэш ничего не даёт.
        "--no-enable-prefix-caching",
        "--trust-remote-code",
        "--host",
        "127.0.0.1",
        "--port",
        str(cfg.port),
    ]


class VllmCaptioner:
    concurrent = True

    # [11] Клиент ещё без процесса. Сервер поднимает [15].
    def __init__(self, cfg: QwenConfig, device: str = "auto"):
        self.cfg = cfg
        self.device = pick_torch_device(device)
        self.attn_implementation = "vllm"
        self._proc: subprocess.Popen | None = None
        self._log = None
        self._log_path = ROOT / "logs" / "vllm-serve.log"
        self._owns_server = not bool(cfg.api_base)
        self._mem_util: float | None = None

    @property
    # [18a] http://127.0.0.1:порт, либо api_base из конфига.
    def api_base(self) -> str:
        if self.cfg.api_base:
            return self.cfg.api_base.rstrip("/")
        return f"http://127.0.0.1:{self.cfg.port}"

    # [15] Спавнит сервер [16] и ждёт /health [18].
    def load(self, gpu_memory_utilization: float | None = None) -> None:
        self._mem_util = gpu_memory_utilization
        if self.device != "cuda":
            raise RuntimeError(
                "vLLM для Qwen3-VL нужен CUDA (A100). "
                "На этой машине поставьте qwen.backend: transformers"
            )
        try:
            import vllm  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "нет vLLM. На A100: pip install -e '.[vllm]'"
            ) from exc
        if self._owns_server:
            self._spawn()
        self._wait_until_ready()

    # [54] SIGTERM группе процесса vLLM.
    def unload(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None and proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait(timeout=10)
        if self._log is not None:
            self._log.close()
            self._log = None

    # [44] 8 кадров ролика: [29], [45], декодер [44d], [48].
    def prepare_frames(self, path: Path) -> list[Image.Image]:
        meta = probe_video(path)
        indices = uniform_indices(meta["nb_frames"], self.cfg.num_frames)
        threads = 1 if self.cfg.decode_workers > 1 else 0
        captured = sample_qwen_frames(
            path, indices, self.cfg.max_side, threads, self.cfg.decoder
        )
        return frames_as_pil(captured, indices)

    # Один ролик через [49]. Основной прогон идёт через [49c], а [25q] — через [49].
    def caption(self, path: Path, frames: list[Image.Image] | None = None) -> str:
        images = frames if frames is not None else self.prepare_frames(path)
        return self.caption_batch([(path, images)])[0][0]

    # [49c] Один ролик без пачки: непрерывная подача [25c] держит vLLM полным.
    def caption_one(self, path: Path, frames: list[Image.Image]) -> tuple[str, str, int]:
        return self._caption_one((path, frames))

    # [49] До 32 запросов [50] параллельно. Слоты пакует уже vLLM.
    # Каждый элемент — (текст, finish_reason, токенов промпта); "length" — обрезано по max_tokens.
    def caption_batch(
        self, items: list[tuple[Path, list[Image.Image]]]
    ) -> list[tuple[str, str, int]]:
        if not items:
            return []
        workers = min(len(items), 32)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(self._caption_one, items))

    # [50] JPEG кадров [51] и POST [52]. Системное сообщение, temperature, штрафы и seed — отсюда.
    # input: video — те же кадры одним роликом video/jpeg: Qwen3-VL склеивает их
    # попарно и ставит метки времени. fps vLLM для такой последовательности — 1.
    def _caption_one(self, item: tuple[Path, list[Image.Image]]) -> tuple[str, str, int]:
        _, images = item
        content: list[dict] = []
        if self.cfg.input == "video":
            frames = ",".join(_jpeg_base64(image) for image in images)
            content.append(
                {"type": "video_url", "video_url": {"url": f"data:video/jpeg;base64,{frames}"}}
            )
        else:
            for image in images:
                content.append(
                    {"type": "image_url", "image_url": {"url": _jpeg_data_url(image)}}
                )
        content.append({"type": "text", "text": self.cfg.prompt})
        messages = [{"role": "user", "content": content}]
        if self.cfg.system_prompt:
            messages.insert(0, {"role": "system", "content": self.cfg.system_prompt})
        payload = {
            "model": self.cfg.model_id,
            "messages": messages,
            "max_tokens": self.cfg.max_tokens,
            "temperature": self.cfg.temperature,
            "repetition_penalty": self.cfg.repetition_penalty,
            "presence_penalty": self.cfg.presence_penalty,
        }
        if self.cfg.seed is not None:
            payload["seed"] = self.cfg.seed
        return _post_chat(self.api_base, payload)

    # [16] Процесс vLLM с командой [17]. Семплер flashinfer выключен, воркеры через spawn.
    def _spawn(self) -> None:
        cmd = vllm_serve_command(self.cfg, self._mem_util)
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(self._log_path, "w", encoding="utf-8")
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "1"
        env["TOKENIZERS_PARALLELISM"] = "false"
        # Parent already imported torch. Forked engine workers then die in
        # CUDA init and the API process only reports an empty core-proc set.
        env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        # FlashInfer JIT-builds its top-k/top-p sampling kernel at warmup and
        # ninja takes the engine down. The PyTorch sampler covers the same params.
        env["VLLM_USE_FLASHINFER_SAMPLER"] = "0"
        print("vLLM:", " ".join(cmd), flush=True)
        self._proc = subprocess.Popen(
            cmd,
            stdout=self._log,
            stderr=subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

    # [18] Ждёт /health. Если процесс умер, в ошибку попадает [18b].
    def _wait_until_ready(self, timeout_sec: float = 1200) -> None:
        deadline = time.perf_counter() + timeout_sec
        health = self.api_base + "/health"
        last_error: Exception | None = None
        while time.perf_counter() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                raise RuntimeError(
                    f"vLLM завершился с кодом {self._proc.returncode}\n{_failure_excerpt(self._log_path)}"
                )
            try:
                with urllib.request.urlopen(health, timeout=2) as resp:
                    if resp.status == 200:
                        return
            except Exception as exc:
                last_error = exc
            time.sleep(1)
        raise RuntimeError(
            f"vLLM не поднялся за {timeout_sec:.0f} с: {last_error}\n{_failure_excerpt(self._log_path)}"
        )


# [51] Кадр уже уменьшен, здесь только JPEG в base64 / data-url.
def _jpeg_base64(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _jpeg_data_url(image: Image.Image) -> str:
    return f"data:image/jpeg;base64,{_jpeg_base64(image)}"


# [52] POST /v1/chat/completions: текст подписи, finish_reason и токенов промпта.
def _post_chat(api_base: str, payload: dict) -> tuple[str, str, int]:
    req = urllib.request.Request(
        api_base.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=3600) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"vLLM HTTP {exc.code}: {detail}") from exc
    choice = (body.get("choices") or [{}])[0]
    text = (choice.get("message") or {}).get("content") or ""
    prompt_tokens = int((body.get("usage") or {}).get("prompt_tokens") or 0)
    return (
        " ".join(str(text).split()).strip(),
        str(choice.get("finish_reason") or ""),
        prompt_tokens,
    )


# [18b] Кусок лога vLLM с настоящей ошибкой EngineCore, не обёртка в хвосте.
def _failure_excerpt(path: Path) -> str:
    """The process exit is a short wrapper. The cause is the earlier EngineCore error."""
    if not path.exists():
        return ""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    markers = (
        "EngineCore failed",
        "OutOfMemory",
        "Free memory",
        "CUDA error",
        "CalledProcessError",
        "AttributeError",
        "unrecognized arguments",
    )
    for index, line in enumerate(lines):
        if any(marker in line for marker in markers):
            return "\n".join(lines[max(0, index - 2) : index + 50])
    return "\n".join(lines[-80:])
