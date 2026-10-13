"""DOVER view decomposition without decord.

Keeps official fragment/resize sampling, but never stores a full-resolution
clip in memory: each decoded frame is immediately copied into 224x224 views.
"""

from __future__ import annotations

from pathlib import Path
from threading import Lock

import numpy as np
import torch
import torchvision

from video_processing.decode import iter_selected_rgb_frames, probe_video

_SAMPLER_LOCK = Lock()

MEAN = torch.FloatTensor([123.675, 116.28, 103.53])
STD = torch.FloatTensor([58.395, 57.12, 57.375])


class UnifiedFrameSampler:
    def __init__(self, fsize_t, fragments_t, frame_interval=1, num_clips=1, drop_rate=0.0):
        self.fragments_t = fragments_t
        self.fsize_t = fsize_t
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.drop_rate = drop_rate

    def get_frame_indices(self, num_frames: int) -> np.ndarray:
        tgrids = np.array(
            [num_frames // self.fragments_t * i for i in range(self.fragments_t)],
            dtype=np.int32,
        )
        tlength = num_frames // self.fragments_t
        if tlength > self.fsize_t * self.frame_interval:
            rnd_t = np.random.randint(
                0, tlength - self.fsize_t * self.frame_interval, size=len(tgrids)
            )
        else:
            rnd_t = np.zeros(len(tgrids), dtype=np.int32)
        ranges_t = (
            np.arange(self.fsize_t)[None, :] * self.frame_interval
            + rnd_t[:, None]
            + tgrids[:, None]
        )
        return np.concatenate(ranges_t)

    def __call__(self, total_frames: int) -> np.ndarray:
        frame_inds = [self.get_frame_indices(total_frames) for _ in range(self.num_clips)]
        frame_inds = np.concatenate(frame_inds)
        return np.mod(frame_inds, max(total_frames, 1)).astype(np.int32)


def make_samplers(sample_types: dict) -> dict[str, UnifiedFrameSampler]:
    samplers = {}
    for stype, sopt in sample_types.items():
        if "t_frag" not in sopt:
            samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"], sopt["num_clips"], sopt["frame_interval"]
            )
        else:
            samplers[stype] = UnifiedFrameSampler(
                sopt["clip_len"] // sopt["t_frag"],
                sopt["t_frag"],
                sopt["frame_interval"],
                sopt["num_clips"],
            )
    return samplers


def _fragment_coords(h: int, w: int, t: int, sopt: dict):
    fragments_h = sopt["fragments_h"]
    fragments_w = sopt["fragments_w"]
    fsize_h = sopt["fsize_h"]
    fsize_w = sopt["fsize_w"]
    aligned = sopt.get("aligned", 32)
    size_h = fragments_h * fsize_h
    size_w = fragments_w * fsize_w
    hgrids = [min(h // fragments_h * i, h - fsize_h) for i in range(fragments_h)]
    wgrids = [min(w // fragments_w * i, w - fsize_w) for i in range(fragments_w)]
    hlength, wlength = h // fragments_h, w // fragments_w
    blocks = max(t // aligned, 1)
    if hlength > fsize_h:
        rnd_h = torch.randint(hlength - fsize_h, (fragments_h, fragments_w, blocks))
    else:
        rnd_h = torch.zeros((fragments_h, fragments_w, blocks), dtype=torch.int64)
    if wlength > fsize_w:
        rnd_w = torch.randint(wlength - fsize_w, (fragments_h, fragments_w, blocks))
    else:
        rnd_w = torch.zeros((fragments_h, fragments_w, blocks), dtype=torch.int64)
    return {
        "hgrids": hgrids,
        "wgrids": wgrids,
        "rnd_h": rnd_h,
        "rnd_w": rnd_w,
        "fsize_h": fsize_h,
        "fsize_w": fsize_w,
        "aligned": aligned,
        "size_h": size_h,
        "size_w": size_w,
    }


def _fill_fragments(target: torch.Tensor, rgb: np.ndarray, t: int, coords: dict) -> None:
    aligned = coords["aligned"]
    block = min(t // aligned, coords["rnd_h"].shape[-1] - 1)
    frame = torch.from_numpy(rgb).permute(2, 0, 1)
    for i, hs in enumerate(coords["hgrids"]):
        for j, ws in enumerate(coords["wgrids"]):
            h_s, h_e = i * coords["fsize_h"], (i + 1) * coords["fsize_h"]
            w_s, w_e = j * coords["fsize_w"], (j + 1) * coords["fsize_w"]
            h_so = hs + int(coords["rnd_h"][i, j, block])
            w_so = ws + int(coords["rnd_w"][i, j, block])
            h_eo = h_so + coords["fsize_h"]
            w_eo = w_so + coords["fsize_w"]
            target[:, t, h_s:h_e, w_s:w_e] = frame[:, h_so:h_eo, w_so:w_eo]


def _resize_to_chw(rgb: np.ndarray, size_h: int, size_w: int) -> torch.Tensor:
    img = torch.from_numpy(rgb).permute(2, 0, 1).float()
    resized = torchvision.transforms.functional.resize(
        img, [size_h, size_w], antialias=True
    )
    return resized


def decompose_video(
    path: Path,
    sample_types: dict,
    samplers: dict[str, UnifiedFrameSampler],
    ffmpeg_threads: int = 1,
) -> dict[str, torch.Tensor]:
    meta = probe_video(path)
    n_frames = max(int(meta["nb_frames"]), 1)
    width, height = int(meta["width"]), int(meta["height"])
    with _SAMPLER_LOCK:
        frame_inds = {stype: sampler(n_frames) for stype, sampler in samplers.items()}
    needed = sorted({int(i) for inds in frame_inds.values() for i in inds.tolist()})

    aesthetic_inds = frame_inds.get("aesthetic")
    technical_inds = frame_inds.get("technical")
    aesthetic = None
    technical = None
    aes_opt = sample_types.get("aesthetic") or {}
    tech_opt = sample_types.get("technical") or {}
    if aesthetic_inds is not None:
        aesthetic = torch.zeros(
            3, len(aesthetic_inds), aes_opt["size_h"], aes_opt["size_w"], dtype=torch.float32
        )
        aes_slots: dict[int, list[int]] = {}
        for t, src in enumerate(aesthetic_inds.tolist()):
            aes_slots.setdefault(int(src), []).append(t)
    else:
        aes_slots = {}
    tech_coords = None
    tech_slots: dict[int, list[int]] = {}
    if technical_inds is not None:
        technical = torch.zeros(
            3,
            len(technical_inds),
            tech_opt["fragments_h"] * tech_opt["fsize_h"],
            tech_opt["fragments_w"] * tech_opt["fsize_w"],
            dtype=torch.float32,
        )
        tech_coords = _fragment_coords(height, width, len(technical_inds), tech_opt)
        for t, src in enumerate(technical_inds.tolist()):
            tech_slots.setdefault(int(src), []).append(t)

    for idx, rgb in iter_selected_rgb_frames(
        path, needed, width, height, ffmpeg_threads=ffmpeg_threads
    ):
        for t in aes_slots.get(idx, []):
            aesthetic[:, t] = _resize_to_chw(rgb, aes_opt["size_h"], aes_opt["size_w"])
        for t in tech_slots.get(idx, []):
            _fill_fragments(technical, rgb, t, tech_coords)

    views = {}
    # порядок как в official evaluate_a_set_of_videos: aesthetic, technical
    if aesthetic is not None:
        views["aesthetic"] = aesthetic
    if technical is not None:
        views["technical"] = technical
    return views


def normalize_and_split(views: dict[str, torch.Tensor], sample_types: dict, device: str) -> dict[str, torch.Tensor]:
    prepared = {}
    mean = MEAN.to(device)
    std = STD.to(device)
    for key, video in views.items():
        num_clips = sample_types[key].get("num_clips", 1)
        normed = ((video.permute(1, 2, 3, 0) - mean.cpu()) / std.cpu()).permute(3, 0, 1, 2)
        prepared[key] = (
            normed.reshape(normed.shape[0], num_clips, -1, *normed.shape[2:])
            .transpose(0, 1)
            .to(device, non_blocking=device == "cuda")
        )
    return prepared
