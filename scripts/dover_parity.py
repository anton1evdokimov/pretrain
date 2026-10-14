"""DOVER parity: project view decomposition vs official DOVER code.

Both paths get the same RNG state, so frame indices and fragment offsets are
identical. Frames for the reference are decoded independently with OpenCV.
The reference Resize runs with antialias=False, as torchvision did for uint8
tensors under DOVER's pinned torch~=1.13; the fusion constants and
dover_predictions/*.pkl come from that setup.

    python scripts/dover_parity.py --videos 1 6 9 --device cpu
    python scripts/dover_parity.py --demo --device cpu   # vs README of DOVER
    python scripts/dover_parity.py --videos 1 3 6 9 --device cuda --decoder nvdec
"""

from __future__ import annotations

import argparse
import pickle
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import cv2
import numpy as np
import torch
import torchvision

from video_processing.config import load_config
from video_processing.decode import probe_video
from video_processing.dover_infer import DoverScorer, fuse_results
from video_processing.dover_views import MEAN, STD

DEMO = ROOT / "third_party" / "DOVER" / "demo" / "17734.mp4"
DEMO_DBS = ["livevqc", "kv1k", "ltest", "l1080p", "ytugc"]
# README of DOVER, "Try on Demos", 17734.mp4, normalized score per dataset.
DEMO_README = {
    "aesthetic": [0.51, 1.21, 0.85, 0.25, 0.86],
    "technical": [0.10, 0.77, 0.59, 0.25, 0.65],
}


def _official_module(antialias: bool):
    for name in ("decord", "skvideo", "skvideo.io"):
        mod = types.ModuleType(name)
        mod.VideoReader = mod.cpu = mod.gpu = None
        mod.bridge = types.SimpleNamespace(set_bridge=lambda *_: None)
        mod.io = mod
        sys.modules.setdefault(name, mod)
    import dover.datasets.dover_datasets as official

    def _resize_function(size_h, size_w, target_ratio=1, random_crop=False):
        return torchvision.transforms.Resize((size_h, size_w), antialias=antialias)

    official.get_resize_function = _resize_function
    return official


def _decode_cv2(path: Path, needed: set[int]) -> dict[int, torch.Tensor]:
    cap = cv2.VideoCapture(str(path))
    frames: dict[int, torch.Tensor] = {}
    idx = 0
    while len(frames) < len(needed):
        ok, bgr = cap.read()
        if not ok:
            break
        if idx in needed:
            frames[idx] = torch.from_numpy(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        idx += 1
    cap.release()
    missing = needed - frames.keys()
    if missing:
        raise RuntimeError(f"{path}: OpenCV did not return frames {sorted(missing)[:5]}")
    return frames


def _official_views(official, path, sample_types, samplers):
    n_frames = max(int(probe_video(path)["nb_frames"]), 1)
    frame_inds = {stype: samplers[stype](n_frames) for stype in samplers}
    needed = {int(i) for inds in frame_inds.values() for i in inds}
    frames = _decode_cv2(path, needed)
    out = {}
    for key in ("aesthetic", "technical"):
        sopt = sample_types[key]
        video = torch.stack([frames[int(i)] for i in frame_inds[key]], 0).permute(3, 0, 1, 2)
        v = official.get_single_view(video, key, **sopt)
        num_clips = sopt.get("num_clips", 1)
        out[key] = (
            ((v.permute(1, 2, 3, 0) - MEAN) / STD)
            .permute(3, 0, 1, 2)
            .reshape(v.shape[0], num_clips, -1, *v.shape[2:])
            .transpose(0, 1)
        )
    return out


def _raw(model, clips, device) -> tuple[float, float]:
    clips = {k: v.to(device) for k, v in clips.items()}
    with torch.inference_mode():
        raw = model(clips, reduce_scores=False)
    aes, tech = (float(r.float().mean().cpu()) for r in raw)
    return aes, tech


def _seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _compare(scorer, official, paths, device, seed) -> None:
    model, st, samplers = scorer._model, scorer._sample_types, scorer._samplers
    for path in paths:
        _seed(seed)
        ours = scorer.prepare(path)
        _seed(seed)
        ref = _official_views(official, path, st, samplers)
        print(f"\n{path.name}")
        for key in ours:
            diff = (ours[key].cpu() - ref[key]).abs()
            print(f"  input diff {key:9s} max={float(diff.max()):.4f} mean={float(diff.mean()):.5f}")
        for label, clips in (("project", ours), ("official", ref)):
            s = fuse_results(list(_raw(model, clips, device)))
            print(
                f"  {label:9s} tech={s['technical']:.4f} aes={s['aesthetic']:.4f} "
                f"overall={s['overall']:.4f}"
            )


def _demo(scorer, device, seeds) -> None:
    sets = {
        db: pickle.load(open(ROOT / f"third_party/DOVER/dover_predictions/val-{db}.pkl", "rb"))
        for db in DEMO_DBS
    }

    def _z(x, arr):
        a = np.concatenate([[x], arr])
        return round(float(((a - a.mean()) / a.std())[0]), 2)

    print(f"README  aes_z={DEMO_README['aesthetic']} tech_z={DEMO_README['technical']}")
    for seed in seeds:
        _seed(seed)
        aes, tech = _raw(scorer._model, scorer.prepare(DEMO), device)
        az = [_z(aes, sets[db]["resize"]) for db in DEMO_DBS]
        tz = [_z(tech, sets[db]["fragments"]) for db in DEMO_DBS]
        print(f"seed={seed} aes_z={az} tech_z={tz}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos", nargs="*", default=["1", "6", "9"])
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--demo", action="store_true", help="17734.mp4 против README DOVER")
    parser.add_argument("--decoder", default="cpu", choices=["cpu", "nvdec"])
    args = parser.parse_args()

    cfg = load_config(ROOT / "configs" / "default.yaml")
    cfg.dover.device = args.device
    cfg.dover.decoder = args.decoder
    scorer = DoverScorer(cfg.dover)
    scorer.load()
    if args.demo:
        _demo(scorer, args.device, seeds=(0, 1, 2))
        return
    official = _official_module(antialias=False)
    paths = [cfg.input_dir / f"{name}.mp4" for name in args.videos]
    _compare(scorer, official, paths, args.device, args.seed)


if __name__ == "__main__":
    main()
