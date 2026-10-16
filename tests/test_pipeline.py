"""Smoke-тест без GPU и моделей: python -m unittest discover tests

DOVER и Qwen заменены заглушками, всё остальное — настоящий run_pipeline:
два потока, окно замера, запись results.* и RESULTS.md. Нужен только ffprobe.
"""

from __future__ import annotations

import json
import math
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_processing import pipeline
from video_processing.config import load_config
from video_processing.dover_infer import dover_z, fuse_results


class FakeDover:
    device = "cpu"

    def __init__(self, cfg):
        self.cfg = cfg

    def load(self):
        pass

    def unload(self):
        pass

    def iter_scores(self, items, stop=None, forward_lock=None):
        size = max(1, int(self.cfg.batch_size))
        for start in range(0, len(items), size):
            batch = items[start : start + size]
            time.sleep(0.005)
            yield [index for index, _ in batch], [fuse_results([0.1, -0.05]) | {
                "aesthetic_raw": 0.1, "technical_raw": -0.05} for _ in batch]


class FakeQwen:
    concurrent = True
    engine = "vllm"
    attn_implementation = None
    device = "cpu"

    def __init__(self, cfg, device="auto"):
        self.cfg = cfg

    def load(self, gpu_memory_utilization=None):
        pass

    def unload(self):
        pass

    def prepare_frames(self, path):
        return [path.name]

    def caption_one(self, path, frames):
        time.sleep(0.002)
        return f"Описание {path.stem}: один два три", "stop", 850

    def caption_batch(self, items):
        return [self.caption_one(path, frames) for path, frames in items]


@unittest.skipIf(shutil.which("ffprobe") is None, "нужен ffprobe")
class PipelineSmoke(unittest.TestCase):
    def _run(self, continuous: bool) -> tuple[dict, Path]:
        cfg = load_config(ROOT / "configs" / "default.yaml")
        cfg.replicate = 3
        cfg.warmup_videos = 2
        cfg.qwen.continuous = continuous
        out = Path(tempfile.mkdtemp(prefix="vp_smoke_"))
        self.addCleanup(shutil.rmtree, out, True)
        cfg.output_dir = out
        with mock.patch.object(pipeline, "DoverScorer", FakeDover), \
             mock.patch.object(pipeline, "make_captioner", FakeQwen):
            payload = pipeline.run_pipeline(cfg)
        return payload, out

    def _check(self, payload: dict, out: Path) -> None:
        n_videos = len(pipeline.list_videos(ROOT / "video"))
        t = payload["throughput"]
        self.assertEqual(t["processed_jobs"], 3 * n_videos)
        self.assertEqual(len(payload["results"]), n_videos)
        self.assertEqual(len(payload["captions_all"]), 3 * n_videos)
        self.assertTrue(all(r["caption"].startswith("Описание") for r in payload["results"]))
        self.assertGreater(t["videos_per_sec"], 0)
        self.assertEqual(t["captions_truncated_jobs"], 0)
        for name in ("results.json", "results.csv", "throughput.json", "captions_all.jsonl", "RESULTS.md"):
            self.assertTrue((out / name).exists(), name)
        md = (out / "RESULTS.md").read_text(encoding="utf-8")
        self.assertIn("| 1 |", md)
        self.assertLess(md.index("| 2 |"), md.index("| 10 |"))
        self.assertEqual(json.loads((out / "throughput.json").read_text())["processed_jobs"], 3 * n_videos)

    def test_continuous(self):
        payload, out = self._run(continuous=True)
        self.assertTrue(payload["throughput"]["qwen_continuous"])
        self._check(payload, out)

    def test_batched(self):
        payload, out = self._run(continuous=False)
        self.assertFalse(payload["throughput"]["qwen_continuous"])
        self._check(payload, out)


class Units(unittest.TestCase):
    def test_steady_window_stops_at_first_finished_stage(self):
        events = [(1.0, 2), (2.0, 2), (3.0, 2), (4.0, 2)]
        # Прогрев закрывает событие на 1.0 с; дальше считаются события после него.
        self.assertEqual(pipeline._steady_window(events, 2, 3.0), {"jobs": 4, "wall": 2.0})
        self.assertEqual(pipeline._steady_window(events, 2, None), {"jobs": 6, "wall": 3.0})

    def test_loop_metric(self):
        self.assertEqual(pipeline._uniq8("a b c d e f g h i j"), 1.0)
        self.assertLess(pipeline._uniq8(" ".join(["x y z w"] * 10)), pipeline.LOOP_UNIQ8)

    def test_dover_z_matches_fusion(self):
        fused = fuse_results([0.1, -0.05])
        aes_z, tech_z = dover_z(0.1, -0.05)
        self.assertAlmostEqual(fused["aesthetic"], 1 / (1 + math.exp(-aes_z)))
        self.assertAlmostEqual(fused["technical"], 1 / (1 + math.exp(-tech_z)))


if __name__ == "__main__":
    unittest.main()
