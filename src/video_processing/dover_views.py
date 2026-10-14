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
    # [23] Параметры одной ветки из dover.yml: кадров в отрезке, число отрезков, шаг, повторы.
    def __init__(self, fsize_t, fragments_t, frame_interval=1, num_clips=1, drop_rate=0.0):
        self.fragments_t = fragments_t
        self.fsize_t = fsize_t
        self.frame_interval = frame_interval
        self.num_clips = num_clips
        self.drop_rate = drop_rate

    # [31] fragments_t отрезков по fsize_t кадров, у каждого свой случайный сдвиг.
    # С dover.yml: technical — 3 отрезка по 32 кадра (3 клипа), aesthetic — 32 по 1 кадру.
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

    # [30] Индексы ветки: [31] num_clips раз (в dover.yml — один). np.random, поэтому под lock.
    def __call__(self, total_frames: int) -> np.ndarray:
        frame_inds = [self.get_frame_indices(total_frames) for _ in range(self.num_clips)]
        frame_inds = np.concatenate(frame_inds)
        return np.mod(frame_inds, max(total_frames, 1)).astype(np.int32)


# [22] По семплеру на aesthetic и technical. Один раз при [19].
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


# [32] Случайные окна 7×7 для technical. Сетка общая, сдвиг окон свой на каждый блок
# из aligned=32 кадров, то есть на каждый из 3 клипов.
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


# [36t] Кладёт фрагменты кадра сразу в тензор 224², полный кадр не хранит.
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


# [36a] Кадр aesthetic → 224².
# Fusion-константы и dover_predictions/*.pkl получены на torch~=1.13, где Resize
# по uint8-тензору шёл без antialias. С antialias=True aesthetic уезжает на ~1.5σ
# (scripts/dover_parity.py --demo).
def _resize_to_chw(rgb: np.ndarray, size_h: int, size_w: int) -> torch.Tensor:
    img = torch.from_numpy(rgb).permute(2, 0, 1)
    return torchvision.transforms.functional.resize(
        img, [size_h, size_w], antialias=False
    )


# [28] Один ролик: индексы, ffmpeg [34], сборка клипов на CPU.
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

    return _ordered_views(aesthetic, technical)


def _ordered_views(aesthetic, technical) -> dict[str, torch.Tensor]:
    views = {}
    # порядок как в official evaluate_a_set_of_videos: aesthetic, technical
    if aesthetic is not None:
        views["aesthetic"] = aesthetic
    if technical is not None:
        views["technical"] = technical
    return views


# [32g] Левые верхние углы окон [32] для каждого слота technical: [T, fh, fw].
def _fragment_offsets(coords: dict, n_slots: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    blocks = torch.clamp(
        torch.arange(n_slots) // coords["aligned"], max=coords["rnd_h"].shape[-1] - 1
    )
    hgrids = torch.tensor(coords["hgrids"], dtype=torch.int64)
    wgrids = torch.tensor(coords["wgrids"], dtype=torch.int64)
    h0 = hgrids[None, :, None] + coords["rnd_h"][:, :, blocks].permute(2, 0, 1)
    w0 = wgrids[None, None, :] + coords["rnd_w"][:, :, blocks].permute(2, 0, 1)
    return h0.to(device), w0.to(device)


# [36g] Все окна [36t] пачки кадров одним gather: frames [n,3,H,W] → target[:, slots].
def _fill_fragments_gpu(
    target: torch.Tensor,
    frames: torch.Tensor,
    slots: torch.Tensor,
    positions: torch.Tensor,
    h0: torch.Tensor,
    w0: torch.Tensor,
    coords: dict,
) -> None:
    fsize_h, fsize_w = coords["fsize_h"], coords["fsize_w"]
    rows = h0[slots][..., None] + torch.arange(fsize_h, device=frames.device)
    cols = w0[slots][..., None] + torch.arange(fsize_w, device=frames.device)
    # [S, fh, fw, fsize_h, fsize_w, 3]
    patches = frames[
        positions[:, None, None, None, None],
        :,
        rows[:, :, :, :, None],
        cols[:, :, :, None, :],
    ]
    n, fh, fw = patches.shape[:3]
    target[:, slots] = (
        patches.permute(5, 0, 1, 3, 2, 4).reshape(3, n, fh * fsize_h, fw * fsize_w).float()
    )


# [28g] То же, что [28], но кадры декодирует NVDEC (torchcodec), а resize и окна
# считаются на GPU. Порядок вызовов RNG как в [28], поэтому при равном seed
# индексы и окна совпадают с CPU-путём.
def decompose_video_nvdec(
    path: Path,
    sample_types: dict,
    samplers: dict[str, UnifiedFrameSampler],
    device: str = "cuda",
    chunk: int = 16,
) -> dict[str, torch.Tensor]:
    from torchcodec.decoders import VideoDecoder

    meta = probe_video(path)
    n_frames = max(int(meta["nb_frames"]), 1)
    width, height = int(meta["width"]), int(meta["height"])
    with _SAMPLER_LOCK:
        frame_inds = {stype: sampler(n_frames) for stype, sampler in samplers.items()}
    needed = sorted({int(i) for inds in frame_inds.values() for i in inds.tolist()})

    aesthetic_inds = frame_inds.get("aesthetic")
    technical_inds = frame_inds.get("technical")
    aes_opt = sample_types.get("aesthetic") or {}
    tech_opt = sample_types.get("technical") or {}
    aesthetic = technical = None
    aes_slots: dict[int, list[int]] = {}
    tech_slots: dict[int, list[int]] = {}
    if aesthetic_inds is not None:
        aesthetic = torch.empty(
            3, len(aesthetic_inds), aes_opt["size_h"], aes_opt["size_w"],
            dtype=torch.float32, device=device,
        )
        for t, src in enumerate(aesthetic_inds.tolist()):
            aes_slots.setdefault(int(src), []).append(t)
    if technical_inds is not None:
        tech_coords = _fragment_coords(height, width, len(technical_inds), tech_opt)
        technical = torch.empty(
            3,
            len(technical_inds),
            tech_opt["fragments_h"] * tech_opt["fsize_h"],
            tech_opt["fragments_w"] * tech_opt["fsize_w"],
            dtype=torch.float32,
            device=device,
        )
        h0, w0 = _fragment_offsets(tech_coords, len(technical_inds), device)
        for t, src in enumerate(technical_inds.tolist()):
            tech_slots.setdefault(int(src), []).append(t)

    decoder = VideoDecoder(str(path), device=device, seek_mode="exact")
    last = int(decoder.metadata.num_frames or n_frames) - 1
    for start in range(0, len(needed), chunk):
        part = needed[start : start + chunk]
        frames = decoder.get_frames_at(indices=[min(idx, last) for idx in part]).data
        if tuple(frames.shape[-2:]) != (height, width):
            raise RuntimeError(
                f"NVDEC отдал {tuple(frames.shape[-2:])}, ffprobe — {(height, width)}: {path}"
            )
        aes_pairs = [(t, k) for k, idx in enumerate(part) for t in aes_slots.get(idx, [])]
        if aes_pairs:
            slots, positions = zip(*aes_pairs)
            resized = torchvision.transforms.functional.resize(
                frames[list(positions)], [aes_opt["size_h"], aes_opt["size_w"]], antialias=False
            )
            aesthetic[:, list(slots)] = resized.permute(1, 0, 2, 3).float()
        tech_pairs = [(t, k) for k, idx in enumerate(part) for t in tech_slots.get(idx, [])]
        if tech_pairs:
            slots, positions = (
                torch.tensor(v, dtype=torch.int64, device=device) for v in zip(*tech_pairs)
            )
            _fill_fragments_gpu(technical, frames, slots, positions, h0, w0, tech_coords)
        del frames
    return _ordered_views(aesthetic, technical)


# [37] Нормализация ImageNet и раскладка по клипам. Зовётся из пула [28p]. После [28g]
# тензоры уже на GPU; CPU-путь остаётся на cpu, на GPU их переносит [37f].
def normalize_and_split(views: dict[str, torch.Tensor], sample_types: dict, device: str) -> dict[str, torch.Tensor]:
    prepared = {}
    for key, video in views.items():
        mean = MEAN.to(video.device)
        std = STD.to(video.device)
        num_clips = sample_types[key].get("num_clips", 1)
        normed = ((video.permute(1, 2, 3, 0) - mean) / std).permute(3, 0, 1, 2)
        prepared[key] = (
            normed.reshape(normed.shape[0], num_clips, -1, *normed.shape[2:])
            .transpose(0, 1)
            .to(device, non_blocking=device == "cuda")
        )
    return prepared
