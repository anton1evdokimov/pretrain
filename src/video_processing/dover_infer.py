from __future__ import annotations

import os
import sys
import types
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import yaml

from video_processing.config import ROOT, DoverConfig
from video_processing.device import pick_torch_device, release_memory, sync
from video_processing.dover_views import decompose_video, make_samplers, normalize_and_split
from video_processing.weights import ensure_dover_weights


def fuse_results(results: list[float]) -> dict[str, float]:
    # results[0] aesthetic, results[1] technical (official evaluate_a_set_of_videos.py)
    technical_z = (results[1] - 0.1107) / 0.07355
    aesthetic_z = (results[0] + 0.08285) / 0.03774
    fused = technical_z * 0.6104 + aesthetic_z * 0.3896
    return {
        "aesthetic": float(1 / (1 + np.exp(-aesthetic_z))),
        "technical": float(1 / (1 + np.exp(-technical_z))),
        "overall": float(1 / (1 + np.exp(-fused))),
    }


class DoverScorer:
    def __init__(self, cfg: DoverConfig):
        self.cfg = cfg
        self.device = pick_torch_device(cfg.device)
        self._model = None
        self._sample_types = None
        self._samplers = None

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
        model.load_state_dict(state, strict=False)
        model.eval()
        for module in model.modules():
            if hasattr(module, "use_checkpoint"):
                module.use_checkpoint = False
        model.to(self.device)
        self._model = model

    def unload(self) -> None:
        self._model = None
        release_memory()

    def score(self, path: Path) -> dict[str, float]:
        return self.score_batch([path])[0]

    def score_batch(self, paths: list[Path]) -> list[dict[str, float]]:
        if self._model is None:
            self.load()
        workers = self.cfg.decode_workers
        if workers <= 0:
            workers = min(len(paths), os.cpu_count() or 8)
        workers = max(1, min(workers, len(paths)))
        ffmpeg_threads = 1 if workers > 1 else 0

        def _views(path: Path):
            return decompose_video(
                path,
                self._sample_types,
                self._samplers,
                ffmpeg_threads=ffmpeg_threads,
            )

        if workers == 1:
            views_list = [_views(path) for path in paths]
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                views_list = list(pool.map(_views, paths))
        prepared = [
            normalize_and_split(views, self._sample_types, self.device)
            for views in views_list
        ]

        aesthetic = torch.cat([item["aesthetic"] for item in prepared], dim=0)
        technical = torch.cat([item["technical"] for item in prepared], dim=0)
        aes_clips = self._sample_types["aesthetic"].get("num_clips", 1)
        tech_clips = self._sample_types["technical"].get("num_clips", 1)
        clips = {"aesthetic": aesthetic, "technical": technical}
        del views_list, prepared
        if self.device == "cuda":
            torch.cuda.empty_cache()

        with torch.inference_mode():
            raw = self._model(clips, reduce_scores=False)
            sync(self.device)
            aesthetic_scores = _mean_per_video(raw[0], len(paths), aes_clips)
            technical_scores = _mean_per_video(raw[1], len(paths), tech_clips)

        out = []
        for aesthetic_raw, technical_raw in zip(aesthetic_scores, technical_scores):
            fused = fuse_results([aesthetic_raw, technical_raw])
            fused["aesthetic_raw"] = aesthetic_raw
            fused["technical_raw"] = technical_raw
            out.append(fused)
        return out

    @staticmethod
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
        # Official dover/__init__.py imports datasets → decord. We only need the model.
        pkg = types.ModuleType("dover")
        pkg.__path__ = [str(pkg_dir)]
        pkg.__file__ = str(pkg_dir / "__init__.py")
        pkg.__package__ = "dover"
        sys.modules["dover"] = pkg

    @staticmethod
    def _disable_convnext_imagenet_init() -> None:
        import dover.models.conv_backbone as conv_backbone
        import dover.models.evaluator as evaluator

        original = conv_backbone.convnext_3d_tiny

        def _tiny(pretrained=False, **kwargs):
            return original(pretrained=False, **kwargs)

        conv_backbone.convnext_3d_tiny = _tiny
        evaluator.convnext_3d_tiny = _tiny


def _mean_per_video(tensor: torch.Tensor, n_videos: int, clips: int) -> list[float]:
    values = tensor.detach().float().reshape(n_videos, clips, -1).mean(dim=(1, 2))
    return [float(x) for x in values.cpu()]
