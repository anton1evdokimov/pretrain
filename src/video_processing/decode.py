from __future__ import annotations

import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def probe_video(path: Path) -> dict:
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,nb_frames,avg_frame_rate,duration,codec_name",
        "-show_entries",
        "format=duration,size",
        "-of",
        "json",
        str(path),
    ]
    payload = json.loads(subprocess.check_output(cmd, text=True))
    stream = (payload.get("streams") or [{}])[0]
    fmt = payload.get("format") or {}
    fps_raw = stream.get("avg_frame_rate") or "0/1"
    if "/" in fps_raw:
        num, den = fps_raw.split("/", 1)
        fps = float(num) / float(den) if float(den) else 0.0
    else:
        fps = float(fps_raw or 0)
    duration = float(stream.get("duration") or fmt.get("duration") or 0)
    nb_frames = int(stream.get("nb_frames") or 0)
    if nb_frames <= 0 and duration > 0 and fps > 0:
        nb_frames = max(int(round(duration * fps)), 1)
    return {
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "fps": fps,
        "duration": duration,
        "nb_frames": nb_frames,
        "codec": stream.get("codec_name"),
        "size": int(fmt.get("size") or 0),
    }


def uniform_indices(n_frames: int, count: int) -> np.ndarray:
    if n_frames <= 0:
        return np.zeros((0,), dtype=np.int32)
    count = max(1, min(int(count), n_frames))
    if count == 1:
        return np.array([n_frames // 2], dtype=np.int32)
    return np.linspace(0, n_frames - 1, count).round().astype(np.int32)


def _resize_rgb(rgb: np.ndarray, max_side: int) -> np.ndarray:
    h, w = rgb.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale >= 1.0:
        return rgb
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def sample_rgb_frames(path: Path, indices: np.ndarray, max_side: int | None = None) -> dict[int, np.ndarray]:
    """Decode only the requested frames, keeping at most one full-res frame in RAM."""
    needed = {int(i) for i in indices.tolist()}
    captured: dict[int, np.ndarray] = {}
    if not needed:
        return captured
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"не удалось открыть видео: {path}")
    idx = 0
    last = max(needed)
    try:
        while idx <= last:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx in needed:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                if max_side is not None:
                    rgb = _resize_rgb(rgb, max_side)
                captured[idx] = rgb
                if len(captured) == len(needed):
                    break
            idx += 1
    finally:
        cap.release()
    missing = needed - captured.keys()
    if missing:
        # short files / bad timestamps: reuse the last decoded frame
        fallback = captured[max(captured)] if captured else None
        if fallback is None:
            raise RuntimeError(f"кадры не декодированы: {path}")
        for miss in missing:
            captured[miss] = fallback
    return captured


def frames_as_pil(frames: dict[int, np.ndarray], indices: np.ndarray) -> list[Image.Image]:
    images = []
    for idx in indices.tolist():
        images.append(Image.fromarray(frames[int(idx)]).convert("RGB"))
    return images
