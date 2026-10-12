# Video Processing

Пайплайн для A100: техническое и эстетическое качество **DOVER** и описание **Qwen3-VL-2B-Instruct**.

Промпт к VLM фиксирован:

> Сгенерируй детальное описание всего, что происходит на видео

Целевое железо — **NVIDIA A100**. 10 коротких роликов мало, чтобы выйти на плато, поэтому по умолчанию датасет размножается (`replicate: 8`). В таблицу качества попадают только оригинальные 10 файлов.

## Запуск на A100

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .
pip install flash-attn --no-build-isolation   # опционально, но это один из главных ускорителей
python scripts/bootstrap.py                   # DOVER-код и DOVER.pth
python -m video_processing --input video --output results

# точки скорость/качество + батч 8 vs 10 (16 — только если 10 даёт ≥5%)
python -m video_processing --input video --output results --ablate
```

`attn_implementation: auto` сначала пробует FlashAttention-2, затем SDPA.

Результаты:

- `results/RESULTS.md` — оценки и описания
- `results/results.json` / `results.csv`
- `results/throughput.json` — стабильная пропускная способность после warmup (`cuda.synchronize()`)

Оценки DOVER — в `[0, 1]` после официального score-level fusion.

## Что сделано из оптимизаций

Узкое место — Qwen3-VL, не DOVER. Реализованы только приёмы с наибольшим вкладом в end-to-end throughput на одной A100.

### A. Высокое влияние (сделано)

| Приём | Зачем |
|---|---|
| Бюджет visual tokens: 8 кадров, длинная сторона 384 | Время Qwen растёт почти линейно с числом патчей. Полный 1440p/25fps ролик даёт лишние порядки compute и KV-cache. |
| FlashAttention-2 + bf16 | На A100 это основной выигрыш у attention в prefill/decode. Без FA2 остаётся SDPA. |
| Greedy decode, `max_tokens=256`, `use_cache=True` | Sampling и длинный хвост почти не добавляют фактов, но линейно растягивают decode. |
| Батч Qwen (`batch_size=4`) | 2B в bf16 на A100 недогружает GPU на одном ролике. Один `generate` на 4 видео с одинаковым числом кадров. |
| Батч DOVER (`batch_size=8`) + TF32 | Клипы фиксированного размера 224². Веса в fp32, чтобы не сдвигать официальный fusion. |
| Модель грузится один раз, стадии DOVER → Qwen | Амортизация загрузки. Qwen не делит SM с DOVER. |
| Prefetch кадров на CPU, пока GPU считает предыдущий батч | Декод 1440p не должен простоивать A100. |
| Репликация датасета + warmup | 10 роликов — это ещё компиляция CUDA-графов/ядер. Стабильные видео/с считаются после плато. |

### B. Среднее влияние (частично / не трогали специально)

- Выключить activation checkpointing у Swin в DOVER (иначе eval считает граф дважды).
- Не качать ImageNet-init ConvNeXt — `DOVER.pth` и так перезаписывает веса.
- Инкрементальные fragments: full-res клип не держится в RAM.
- `cudnn.benchmark` на фиксированных шейпах DOVER.
- Общий декод на обе модели почти не окупается: DOVER нужен полный кадр, Qwen — маленький resize.

### C. Низкое влияние (не делалось)

- Формат JSON/CSV, микрооптимизации Python, `num_workers` на коротком наборе.
- `torch.compile` для VLM — нестабильно и мало даёт на decode.
- Квантизация Qwen 4/8-bit: на A100 2B и так крошечный, качество описания важнее.

### Имеет смысл на кластере, но не на одной A100

- vLLM / continuous batching и speculative decoding.
- NVDEC вместо CPU-decode на очень длинных роликах.
- Pipeline parallel: DOVER на одной карте, Qwen на другой.
- TensorRT для DOVER, если его доля вдруг станет заметной.

## Ablation на A100

```bash
python -m video_processing --input video --output results --ablate
```

Три точки visual-бюджета Qwen: `4×256`, `8×384`, `16×384`. Батч Qwen и DOVER: 8 vs 10; 16 только если 10 быстрее 8 хотя бы на 5%. Победитель батча и точка `8×384` (баланс) записываются в `configs/default.yaml`. Сводка — `results/ABLATION.md`.

После аблейшна обычный прогон уже идёт с выбранными батчами:

```bash
python -m video_processing --input video --output results
```

## Конфиг

`configs/default.yaml`:

- `qwen.num_frames` / `qwen.max_side` — главный рычаг качества vs скорость
- `qwen.batch_size` — поднять до 8, если VRAM позволяет
- `replicate` — увеличить, если warmup ещё виден на графике времени

На 40/80 ГБ A100 обе модели влезают с запасом: Qwen3-VL-2B-Instruct в bf16 ≈ 4 ГБ, DOVER значительно меньше.
