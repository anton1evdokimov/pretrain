from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from video_processing.config import QwenConfig
from video_processing.decode import frames_as_pil, probe_video, sample_qwen_frames, uniform_indices
from video_processing.device import pick_torch_device, release_memory, sync


USER_PROMPT = "Сгенерируй детальное описание всего, что происходит на видео"


# [10] backend=vllm → [11]. transformers — отдельная ветка, в текущем прогоне не идёт.
def make_captioner(cfg: QwenConfig, device: str = "auto"):
    if cfg.backend == "vllm":
        from video_processing.qwen_vllm import VllmCaptioner

        return VllmCaptioner(cfg, device=device)
    if cfg.backend not in ("transformers", "hf", ""):
        raise ValueError(f"неизвестный qwen.backend: {cfg.backend}")
    if cfg.input != "images":
        raise ValueError(f"qwen.input: {cfg.input} есть только у backend: vllm")
    return QwenCaptioner(cfg, device=device)


class QwenCaptioner:
    # Ветка transformers. Основной прогон создаёт VllmCaptioner, не этот класс.
    def __init__(self, cfg: QwenConfig, device: str = "auto"):
        self.cfg = cfg
        self.device = pick_torch_device(device)
        self._model = None
        self._processor = None
        self.dtype = None
        self.attn_implementation = None

    # Ветка transformers: грузит Qwen3-VL в этот процесс.
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

    # Ветка transformers: отпускает модель.
    def unload(self) -> None:
        self._model = None
        self._processor = None
        release_memory()

    # Ветка transformers: те же [45]–[48], что и у vLLM.
    def prepare_frames(self, path: Path) -> list[Image.Image]:
        meta = probe_video(path)
        indices = uniform_indices(meta["nb_frames"], self.cfg.num_frames)
        threads = 1 if self.cfg.decode_workers > 1 else 0
        captured = sample_qwen_frames(
            path, indices, self.cfg.max_side, threads, self.cfg.decoder
        )
        return frames_as_pil(captured, indices)

    # Ветка transformers: один ролик через caption_batch.
    def caption(self, path: Path, frames: list[Image.Image] | None = None) -> str:
        images = frames if frames is not None else self.prepare_frames(path)
        return self.caption_batch([(path, images)])[0][0]

    # Ветка transformers: один generate на пачку, do_sample если temperature > 0.
    # Как у vLLM: (текст, finish_reason, токенов промпта), "length" — без EOS до max_tokens.
    def caption_batch(
        self, items: list[tuple[Path, list[Image.Image]]]
    ) -> list[tuple[str, str, int]]:
        if self._model is None or self._processor is None:
            self.load()
        prompt = self.cfg.prompt or USER_PROMPT
        messages_batch = [
            _messages(images, prompt, self.cfg.system_prompt) for _, images in items
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


# Ветка transformers: класс Qwen3-VL из установленного пакета.
def _model_class():
    import transformers

    if hasattr(transformers, "Qwen3VLForConditionalGeneration"):
        return transformers.Qwen3VLForConditionalGeneration
    if hasattr(transformers, "AutoModelForImageTextToText"):
        return transformers.AutoModelForImageTextToText
    if hasattr(transformers, "AutoModelForVision2Seq"):
        return transformers.AutoModelForVision2Seq
    raise RuntimeError("нужен transformers>=4.57 с Qwen3-VL")


# Ветка transformers: sdpa, при возможности flash_attention_2.
def _attn_candidates(requested: str, device: str) -> list[str | None]:
    if requested and requested != "auto":
        return [requested, "sdpa", None]
    if device == "cuda":
        return ["flash_attention_2", "sdpa", None]
    return ["sdpa", None]


# Ветка transformers: кадры и промпт в формате чата.
def _messages(images: list[Image.Image], prompt: str, system: str = "") -> list[dict]:
    content = [{"type": "image", "image": image} for image in images]
    content.append({"type": "text", "text": prompt})
    messages = [{"role": "user", "content": content}]
    if system:
        messages.insert(0, {"role": "system", "content": [{"type": "text", "text": system}]})
    return messages


# Ветка transformers: токены и пиксели пачки.
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
        try:
            return processor.apply_chat_template(
                messages_batch[0] if len(messages_batch) == 1 else messages_batch,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        except Exception:
            return _prepare_inputs_legacy(processor, messages_batch)
    except Exception:
        return _prepare_inputs_legacy(processor, messages_batch)


# Ветка transformers: старый процессор, если прямой вызов не принял картинки.
def _prepare_inputs_legacy(processor, messages_batch):
    try:
        from qwen_vl_utils import process_vision_info
    except ImportError as exc:
        raise RuntimeError(
            "процессор не принял изображения напрямую, установите qwen-vl-utils"
        ) from exc
    texts = [
        processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        for messages in messages_batch
    ]
    image_inputs, video_inputs = process_vision_info(messages_batch)
    return processor(
        text=texts,
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )


# Ветка transformers: тензоры на GPU.
def _to_device(inputs, device):
    if hasattr(inputs, "to"):
        return inputs.to(device)
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in inputs.items()
    }
