from __future__ import annotations

from pathlib import Path

DOVER_URLS = [
    "https://huggingface.co/teowu/DOVER/resolve/main/DOVER.pth",
    "https://github.com/QualityAssessment/DOVER/releases/download/v0.1.0/DOVER.pth",
    "https://github.com/VQAssessment/DOVER/releases/download/v0.1.0/DOVER.pth",
]


# [21] DOVER.pth на диске. Если файла нет, качает через [21a].
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
    raise RuntimeError(f"не удалось скачать DOVER.pth: {last_error}")


# [21a] Запасной путь [21]: скачать чекпоинт, в обычном прогоне не вызывается.
def _download(url: str, path: Path) -> None:
    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import HfHubHTTPError

    if "huggingface.co" in url:
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
        except (HfHubHTTPError, Exception):
            pass
    import urllib.request

    tmp = path.with_suffix(".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
