from __future__ import annotations

import json
import subprocess
import threading
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
    if not needed:
        return {}
    ordered = sorted(needed)
    captured: dict[int, np.ndarray] = {}
    try:
        meta = probe_video(path)
        width, height = int(meta["width"]), int(meta["height"])
        if width > 0 and height > 0:
            for idx, rgb in _iter_ffmpeg_rgb(path, ordered, width, height, ffmpeg_threads=0):
                if max_side is not None:
                    rgb = _resize_rgb(rgb, max_side)
                captured[idx] = rgb
            if captured.keys() >= needed:
                return captured
    except Exception:
        captured = {}
    return _sample_rgb_frames_opencv(path, needed, max_side)


def _sample_rgb_frames_opencv(
    path: Path, needed: set[int], max_side: int | None
) -> dict[int, np.ndarray]:
    captured: dict[int, np.ndarray] = {}
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
        fallback = captured[max(captured)] if captured else None
        if fallback is None:
            raise RuntimeError(f"кадры не декодированы: {path}")
        for miss in missing:
            captured[miss] = fallback
    return captured


def iter_selected_rgb_frames(
    path: Path,
    indices: list[int],
    width: int,
    height: int,
    ffmpeg_threads: int = 1,
):
    """Yield (frame_index, rgb) for sorted unique indices via ffmpeg select.

    ffmpeg still walks the bitstream, but only requested frames are decoded
    to RGB. Falls back to a sequential OpenCV scan if the pipe is empty.
    """
    ordered = sorted({int(i) for i in indices})
    if not ordered:
        return
    try:
        stream = _iter_ffmpeg_rgb(path, ordered, width, height, ffmpeg_threads)
        first = next(stream)
    except Exception:
        captured = _sample_rgb_frames_opencv(path, set(ordered), max_side=None)
        for idx in ordered:
            yield idx, captured[idx]
        return
    yield first
    yield from stream


def _iter_ffmpeg_rgb(
    path: Path,
    ordered: list[int],
    width: int,
    height: int,
    ffmpeg_threads: int,
):
    frame_bytes = width * height * 3
    expr = "+".join(f"eq(n,{idx})" for idx in ordered)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-threads",
        str(max(int(ffmpeg_threads), 0)),
        "-i",
        str(path),
        "-map",
        "0:v:0",
        "-vf",
        f"select='{expr}',scale={width}:{height}:flags=fast_bilinear",
        "-vsync",
        "0",
        "-an",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert proc.stdout is not None
    stderr_chunks: list[bytes] = []

    def _drain_stderr() -> None:
        if proc.stderr is None:
            return
        stderr_chunks.append(proc.stderr.read())

    drain = threading.Thread(target=_drain_stderr, daemon=True)
    drain.start()

    def _stderr_text() -> str:
        drain.join(timeout=2)
        return b"".join(stderr_chunks).decode("utf-8", "replace").strip()

    try:
        for idx in ordered:
            buf = _read_exact(proc.stdout, frame_bytes)
            if buf is None:
                raise RuntimeError(
                    f"ffmpeg недодал кадр {idx} из {path}: {_stderr_text()[:400]}"
                )
            rgb = np.frombuffer(buf, dtype=np.uint8).reshape((height, width, 3)).copy()
            yield idx, rgb
        proc.stdout.close()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait()
        drain.join(timeout=1)


def _read_exact(stream, size: int) -> bytes | None:
    chunks = bytearray()
    while len(chunks) < size:
        piece = stream.read(size - len(chunks))
        if not piece:
            return None
        chunks.extend(piece)
    return bytes(chunks)


def frames_as_pil(frames: dict[int, np.ndarray], indices: np.ndarray) -> list[Image.Image]:
    images = []
    for idx in indices.tolist():
        images.append(Image.fromarray(frames[int(idx)]).convert("RGB"))
    return images
