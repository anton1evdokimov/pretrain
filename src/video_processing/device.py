from __future__ import annotations

import gc

import torch


# cuda, если карта есть. Иначе cpu.
def pick_torch_device(requested: str = "auto") -> str:
    if requested and requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# TF32 и cudnn.benchmark для фиксированных шейпов DOVER. Вызывается до прогона.
def configure_cuda() -> None:    
    if not torch.cuda.is_available():
        return
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")


# cuda.synchronize после forward DOVER, чтобы замер не убегал вперёд.
def sync(device: str) -> None:
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize()


# Имя GPU в лог и в throughput.json.
def gpu_name() -> str | None:
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    return None


# Сборка мусора и освобождения неиспользуемого кэш CUDA.
def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
