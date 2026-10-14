from __future__ import annotations

import gc
import os


# [9] cuda, если карта есть. Иначе mps или cpu. Зовут [8] и [11].
def pick_torch_device(requested: str = "auto") -> str:
    import torch

    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
        return "mps"
    return "cpu"


# [4] TF32 и cudnn.benchmark для фиксированных шейпов DOVER. Из [1], до прогона.
def configure_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


# [38] cuda.synchronize после forward DOVER, чтобы замер не убегал вперёд.
def sync(device: str) -> None:
    import torch

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


# [12] Имя GPU в лог и в throughput.json.
def gpu_name() -> str | None:
    import torch

    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return None


# [59] gc и empty_cache после записи результатов.
def release_memory() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if hasattr(torch, "mps") and torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:
        pass
