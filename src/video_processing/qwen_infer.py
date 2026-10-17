from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from video_processing.config import QwenConfig
from video_processing.decode import (
    frames_as_pil,
    probe_video,
    sample_qwen_frames,
    uniform_indices,
)
from video_processing.device import pick_torch_device, release_memory, sync


# Фабрика captioner: vLLM или Transformers.
def make_captioner(cfg: QwenConfig, device: str = "auto"):
    if cfg.backend == "vllm":
        from video_processing.qwen_vllm import VllmCaptioner

        return VllmCaptioner(cfg, device=device)
    if cfg.backend not in ("transformers", "hf", ""):
        raise ValueError(f"Неизвестный qwen.backend: {cfg.backend!r}")
    if cfg.input != "images":
        raise ValueError(f"qwen.input={cfg.input!r} поддерживается только backend='vllm'")
    return QwenCaptioner(cfg, device=device)


class QwenCaptioner:
    # Запасной путь. Основной прогон создаёт VllmCaptioner, не этот класс.
    def __init__(self, cfg: QwenConfig, device: str = "auto"):
        self.cfg = cfg
        self.device = pick_torch_device(device)
        self._model = None
        self._processor = None
        self.dtype = None
        self.engine = "transformers"
        self.attn_implementation = None

    # Грузит Qwen3-VL в этот процесс.
    def load(self) -> None:
        from transformers import AutoProcessor

        model_cls = _model_class()
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        attn_candidates = _attn_candidates(self.cfg.attn_implementation, self.device)
        last_error: Exception | None = None
        model = None
        for attn in attn_candidates:
            kwargs = {
                "dtype": dtype,
                "trust_remote_code": True,
            }
            if attn is not None:
                kwargs["attn_implementation"] = attn
            if self.device == "cuda":
                kwargs["device_map"] = {"": 0}
            try:
                model = model_cls.from_pretrained(self.cfg.model_id, **kwargs)
                self.attn_implementation = attn or "default"
                break
            except TypeError:
                kwargs.pop("dtype", None)
                kwargs["torch_dtype"] = dtype
                try:
                    model = model_cls.from_pretrained(self.cfg.model_id, **kwargs)
                    self.attn_implementation = attn or "default"
                    break
                except Exception as exc:
                    last_error = exc
            except Exception as exc:
                last_error = exc
        if model is None:
            raise RuntimeError(f"не удалось загрузить {self.cfg.model_id}: {last_error}")
        if self.device != "cuda":
            model = model.to(self.device)
        model.eval()
        self._model = model
        self._processor = AutoProcessor.from_pretrained(
            self.cfg.model_id, trust_remote_code=True
        )
        # Батчевый generate decoder-only модели: при правом паддинге короткий промпт
        # продолжается после pad-токенов. Ролики с другим аспектом дают другую длину.
        self._processor.tokenizer.padding_side = "left"
        self.dtype = dtype

    # Отпускает модель.
    def unload(self) -> None:
        self._model = None
        self._processor = None
        release_memory()

    # Те же шаги подготовки кадров, что у vLLM.
    def prepare_frames(self, path: Path) -> list[Image.Image]:
        meta = probe_video(path)
        indices = uniform_indices(meta["nb_frames"], self.cfg.num_frames)
        threads = 1 if self.cfg.decode_workers > 1 else 0
        captured = sample_qwen_frames(
            path, indices, self.cfg.max_side, threads, self.cfg.decoder
        )
        return frames_as_pil(captured, indices)

    # Один ролик через caption_batch.
    def caption(self, path: Path, frames: list[Image.Image] | None = None) -> str:
        images = frames if frames is not None else self.prepare_frames(path)
        return self.caption_batch([(path, images)])[0][0]

    # Один generate на пачку, do_sample если temperature > 0.
    # Как у vLLM: (текст, finish_reason, токенов промпта), "length" — без EOS до max_tokens.
    def caption_batch(
        self, items: list[tuple[Path, list[Image.Image]]]
    ) -> list[tuple[str, str, int]]:
        if self._model is None or self._processor is None:
            self.load()
        messages_batch = [
            _messages(images, self.cfg.prompt, self.cfg.system_prompt) for _, images in items
        ]
        inputs = _prepare_inputs(self._processor, messages_batch)
        inputs = _to_device(inputs, self._model.device)
        generate_kwargs = {
            "max_new_tokens": self.cfg.max_tokens,
            "do_sample": self.cfg.temperature > 0,
            "use_cache": True,
            "repetition_penalty": self.cfg.repetition_penalty,
        }
        if self.cfg.temperature > 0:
            generate_kwargs["temperature"] = self.cfg.temperature
        with torch.inference_mode():
            generated = self._model.generate(**inputs, **generate_kwargs)
            sync(self.device)
        input_len = inputs["input_ids"].shape[-1]
        trimmed = generated[:, input_len:]
        texts = self._processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        eos = self._model.generation_config.eos_token_id
        eos_ids = torch.tensor(eos if isinstance(eos, (list, tuple)) else [eos], device=trimmed.device)
        stopped = torch.isin(trimmed, eos_ids).any(dim=1).tolist()
        mask = inputs.get("attention_mask")
        prompt_tokens = (
            mask.sum(dim=1).tolist() if mask is not None else [input_len] * len(texts)
        )
        return [
            (" ".join(text.split()).strip(), "stop" if done else "length", int(n))
            for text, done, n in zip(texts, stopped, prompt_tokens)
        ]


# Класс Qwen3-VL из установленного пакета.
def _model_class():
    import transformers

    if hasattr(transformers, "Qwen3VLForConditionalGeneration"):
        return transformers.Qwen3VLForConditionalGeneration
    if hasattr(transformers, "AutoModelForImageTextToText"):
        return transformers.AutoModelForImageTextToText
    if hasattr(transformers, "AutoModelForVision2Seq"):
        return transformers.AutoModelForVision2Seq
    raise RuntimeError("нужен transformers>=4.57 с Qwen3-VL")


# sdpa, при возможности flash_attention_2.
def _attn_candidates(requested: str, device: str) -> list[str | None]:
    if requested and requested != "auto":
        return [requested, "sdpa", None]
    if device == "cuda":
        return ["flash_attention_2", "sdpa", None]
    return ["sdpa", None]


# Кадры и промпт в формате чата.
def _messages(images: list[Image.Image], prompt: str, system: str = "") -> list[dict]:
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    if system:
        messages.insert(0, {"role": "system", "content": [{"type": "text", "text": system}]})
    return messages


# Токены и пиксели пачки.
def _prepare_inputs(processor, messages_batch: list):
    try:
        return processor.apply_chat_template(
            messages_batch,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
        )
    except TypeError:
        return processor.apply_chat_template(
            messages_batch[0] if len(messages_batch) == 1 else messages_batch,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )


# Тензоры на GPU.
def _to_device(inputs, device):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
