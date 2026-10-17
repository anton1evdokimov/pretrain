from __future__ import annotations

from pathlib import Path

DOVER_URLS = [
    "https://huggingface.co/teowu/DOVER/resolve/main/DOVER.pth",
    "https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth",
    "https://github.com/VQAssessment/DOVER/releases/download/v0.1.0/DOVER.pth",
]


# DOVER.pth на диске. Если файла нет, скачивает.
def ensure_dover_weights(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    last_error: Exception | None = None
    for url in DOVER_URLS:
        try:
            _download(url, path)
            if path.exists() and path.stat().st_size > 1_000_000:
                return path
        except Exception as exc:
            last_error = exc
    raise RuntimeError(f"не удалось скачать DOVER.pth: {last_error}") from last_error


# Скачивает DOVER.pth. В обычном прогоне файл уже лежит в weights/.
def _download(url: str, path: Path) -> None:
    if "huggingface.co" in url:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import HfHubHTTPError
        try:
            downloaded = hf_hub_download(
                repo_id="teowu/DOVER",
                filename="DOVER.pth",
                local_dir=str(path.parent),
            )
            src = Path(downloaded)
            if src.resolve() != path.resolve():
                path.write_bytes(src.read_bytes())
            return
        except HfHubHTTPError as e:
            print(f"HF download failed, trying direct URL: {e}")
    import urllib.request

    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
