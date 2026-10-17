from __future__ import annotations

import os
import sys
import threading
import types
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import yaml

from video_processing.config import ROOT, DoverConfig
from video_processing.device import pick_torch_device, release_memory, sync
from video_processing.dover_views import (
    decompose_video,
    decompose_video_nvdec,
    make_samplers,
    normalize_and_split,
)
from video_processing.weights import ensure_dover_weights


# Сырой скор ветки → z относительно калибровки DOVER (official evaluate_a_set_of_videos.py).
def dover_z(aesthetic_raw: float, technical_raw: float) -> tuple[float, float]:
    return (aesthetic_raw + 0.08285) / 0.03774, (technical_raw - 0.1107) / 0.07355


# z-score и веса как в evaluate_a_set_of_videos: aesthetic, technical, overall.
def fuse_results(results: list[float]) -> dict[str, float]:
    # results[0] — aesthetic, results[1] — technical
    aesthetic_z, technical_z = dover_z(results[0], results[1])
    fused = technical_z * 0.6104 + aesthetic_z * 0.3896
    return {
        "aesthetic": float(1 / (1 + np.exp(-aesthetic_z))),
        "technical": float(1 / (1 + np.exp(-technical_z))),
        "overall": float(1 / (1 + np.exp(-fused))),
    }


class DoverScorer:
    # Конфиг и устройство. Веса загружает load.
    def __init__(self, cfg: DoverConfig):
        self.cfg = cfg
        self.device = pick_torch_device(cfg.device)
        self._model = None
        self._sample_types = None
        self._samplers = None
        self._streams = threading.local()

    # Чекпоинт, семплеры и сама сеть. Выключение checkpointing — здесь же.
    def load(self) -> None:
        self._ensure_third_party()
        weights = ensure_dover_weights(self.cfg.weights)
        opt = yaml.safe_load(self.cfg.config.read_text())
        self._sample_types = opt["data"]["val-l1080p"]["args"]["sample_types"]
        self._samplers = make_samplers(self._sample_types)
        self._disable_convnext_imagenet_init()
        from dover.models import DOVER

        model = DOVER(**opt["model"]["args"])
        state = torch.load(weights, map_location="cpu", weights_only=False)
        model.load_state_dict(state, strict=True)
        model.eval()
        for module in model.modules():
            if hasattr(module, "use_checkpoint"):
                module.use_checkpoint = False
        model.to(self.device)
        self._model = model

    # Снимает DOVER с GPU после обоих потоков.
    def unload(self) -> None:
        self._model = None
        release_memory()

    # Поток пачек для основного прогона. Декод идёт в пуле без барьера на
    # границе батча: длинный ролик не держит остальные воркеры, пачка собирается
    # из тех, что готовы первыми. Отдаёт (индексы items, скоры) не по порядку.
    def iter_scores(
        self,
        items: list[tuple[int, Path]],
        stop: threading.Event | None = None,
        forward_lock=None,
    ):
        if self._model is None:
            self.load()
        if not items:
            return
        batch_size = max(1, int(self.cfg.batch_size))
        workers = self._decode_workers(len(items))
        ffmpeg_threads = 1 if workers > 1 else 0
        cap = max(2 * batch_size, batch_size + workers)
        lock = forward_lock if forward_lock is not None else nullcontext()
        queue = iter(items)
        pending: dict = {}
        ready: list[tuple[int, dict]] = []
        pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="dover-decode")

        def _fill() -> None:
            while len(pending) + len(ready) < cap:
                item = next(queue, None)
                if item is None:
                    return
                index, path = item
                pending[pool.submit(self.prepare, path, ffmpeg_threads)] = index

        try:
            _fill()
            while pending or ready:
                if stop is not None and stop.is_set():
                    return
                if len(ready) < batch_size and pending:
                    done, _ = wait(list(pending), return_when=FIRST_COMPLETED)
                    for fut in done:
                        ready.append((pending.pop(fut), fut.result()))
                    _fill()
                    if len(ready) < batch_size and pending:
                        continue
                batch, ready = ready[:batch_size], ready[batch_size:]
                with lock:
                    scores = self.score_prepared([views for _, views in batch])
                yield [index for index, _ in batch], scores
                _fill()
        finally:
            pool.shutdown(wait=True, cancel_futures=True)

    # Подготовка одного ролика: NVDEC или CPU, затем нормализация.
    # Вызывается из пула декодирования.
    def prepare(self, path: Path, ffmpeg_threads: int = 1) -> dict[str, torch.Tensor]:
        if self.cfg.decoder == "nvdec" and str(self.device).startswith("cuda"):
            try:
                return self._prepare_nvdec(path)
            except Exception as exc:
                print(f"[dover] NVDEC: {path.name} → cpu ({exc})", file=sys.stderr)
        views = decompose_video(
            path,
            self._sample_types,
            self._samplers,
            ffmpeg_threads=ffmpeg_threads,
        )
        return normalize_and_split(views, self._sample_types, "cpu")

    # Подготовка на своём CUDA stream потока пула: не ждёт forward
    # на default stream.
    def _prepare_nvdec(self, path: Path) -> dict[str, torch.Tensor]:
        stream = getattr(self._streams, "stream", None)
        if stream is None:
            stream = self._streams.stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            views = decompose_video_nvdec(path, self._sample_types, self._samplers)
            prepared = normalize_and_split(views, self._sample_types, self.device)
        stream.synchronize()
        return prepared

    # Один forward DOVER на уже подготовленные ролики.
    def score_prepared(self, prepared: list[dict[str, torch.Tensor]]) -> list[dict[str, float]]:
        n_videos = len(prepared)
        aesthetic = torch.cat([item["aesthetic"] for item in prepared], dim=0)
        technical = torch.cat([item["technical"] for item in prepared], dim=0)
        clips = {
            "aesthetic": aesthetic.to(self.device),
            "technical": technical.to(self.device),
        }
        aes_clips = self._sample_types["aesthetic"].get("num_clips", 1)
        tech_clips = self._sample_types["technical"].get("num_clips", 1)
        if aesthetic.is_cuda:
            # Виды выделены на stream воркера: до освобождения cat должен
            # дочитать их, иначе аллокатор отдаст память следующему ролику.
            torch.cuda.current_stream().synchronize()
        del prepared, aesthetic, technical
        if self.device == "cuda":
            torch.cuda.empty_cache()

        with torch.inference_mode():
            raw = self._model(clips, reduce_scores=False)
            sync(self.device)
            aesthetic_scores = _mean_per_video(raw[0], n_videos, aes_clips)
            technical_scores = _mean_per_video(raw[1], n_videos, tech_clips)

        out = []
        for aesthetic_raw, technical_raw in zip(aesthetic_scores, technical_scores):
            fused = fuse_results([aesthetic_raw, technical_raw])
            fused["aesthetic_raw"] = aesthetic_raw
            fused["technical_raw"] = technical_raw
            out.append(fused)
        return out

    # По одному ffmpeg с одним потоком на воркер. decode_workers: 0 — по числу ядер.
    def _decode_workers(self, n_items: int) -> int:
        workers = self.cfg.decode_workers
        if workers <= 0:
            workers = os.cpu_count() or 8
        return max(1, min(workers, n_items))

    @staticmethod
    # Кладёт third_party/DOVER в sys.path без импорта decord.
    def _ensure_third_party() -> None:
        dover_root = ROOT / "third_party" / "DOVER"
        pkg_dir = dover_root / "dover"
        if not (pkg_dir / "models").exists():
            raise RuntimeError(
                "нет third_party/DOVER. Клонируйте https://github.com/VQAssessment/DOVER"
            )
        root = str(dover_root)
        if root not in sys.path:
            sys.path.insert(0, root)
        # Официальный dover/__init__.py импортирует datasets → decord. Нам нужна только модель.
        pkg = types.ModuleType("dover")
        pkg.__path__ = [str(pkg_dir)]
        pkg.__file__ = str(pkg_dir / "__init__.py")
        pkg.__package__ = "dover"
        sys.modules["dover"] = pkg

    @staticmethod
    # ConvNeXt не качает ImageNet: DOVER.pth всё равно перезаписывает веса.
    def _disable_convnext_imagenet_init() -> None:
        import dover.models.conv_backbone as conv_backbone
        import dover.models.evaluator as evaluator

        original = conv_backbone.convnext_3d_tiny

        def _tiny(pretrained=False, **kwargs):
            return original(pretrained=False, **kwargs)

        conv_backbone.convnext_3d_tiny = _tiny
        evaluator.convnext_3d_tiny = _tiny


# Сырой скор ветки: среднее по клипам одного ролика.
def _mean_per_video(tensor: torch.Tensor, n_videos: int, clips: int) -> list[float]:
    values = tensor.detach().float().reshape(n_videos, clips, -1).mean(dim=(1, 2))
    return [float(x) for x in values.cpu()]
