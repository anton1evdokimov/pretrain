# Video Processing

Пайплайн для NVIDIA A100. На каждом ролике две модели:

1. **DOVER** — техническое и эстетическое качество, плюс общая оценка. Шкала `[0, 1]`, fusion как в официальном `evaluate_a_set_of_videos.py` на конфиге `val-l1080p`.
2. **Qwen3-VL-2B-Instruct** — описание. Промпт фиксирован: «Сгенерируй детальное описание всего, что происходит на видео».

В `video/` десять роликов. Для выхода на стабильную скорость набор размножается (`replicate: 8`). В таблицу качества попадают только оригинальные десять файлов.

## Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e '.[vllm]'
pip install flash-attn --no-build-isolation   # опционально: FlashAttention-2 для пути transformers
python scripts/bootstrap.py                   # код DOVER и weights/DOVER.pth
python -m video_processing --input video --output results
```

Пайплайн сам поднимает vLLM и сам его гасит. Отдельный `python scripts/serve_qwen.py` нужен только чтобы посмотреть endpoint вручную. Вместе с пайплайном его запускать не нужно: оба займут порт 8000 и одну и ту же память.

Ablation, если нужно выбрать батч и бюджет кадров по замерам:

```bash
python -m video_processing --input video --output results --ablate
```

## Что делает один прогон

1. Собирает ролики и размножает список в `replicate` копий.
2. Поднимает vLLM с Qwen3-VL-2B в bf16. При `parallel: true` сервер берёт 70% VRAM.
3. Загружает DOVER в оставшуюся память.
4. Считает оба фильтра одновременно с t=0: Qwen и DOVER получают один и тот же список роликов. Подпись не ждёт оценку. На 80 ГБ vLLM берёт 70% VRAM (~56 ГБ), DOVER живёт в остальных ~24 ГБ.
5. После `warmup_videos` считает видео/с по wall-clock перекрытия, а не по сумме стадий.
6. Пишет оценки и описания только для исходных десяти файлов и останавливает vLLM.

`parallel: false` возвращает старый порядок: сначала весь DOVER, потом Qwen. В этом режиме vLLM берёт `gpu_memory_utilization` из конфига (0.90).

## Результаты

- `results/RESULTS.md` — таблица technical / aesthetic / overall и описания
- `results/results.json`, `results/results.csv` — то же самое
- `results/throughput.json` — стабильная пропускная способность
- `results/ABLATION.md` — только после `--ablate`

## Оптимизации

Узкое место — Qwen, не DOVER. В код вошло то, что двигает видео/с всего пайплайна.

### Высокое влияние

| Приём | Зачем |
|---|---|
| 8 кадров, длинная сторона 384 | Время Qwen растёт почти линейно с числом патчей. Полный 1440p/25 fps даёт лишние порядки compute и KV. |
| bf16, greedy, `max_tokens=256` | Sampling и длинный хвост почти не добавляют фактов и линейно растягивают decode. |
| vLLM, `max_num_seqs=8`, контекст 4096 | Короткая подпись освобождает слот. Статический `generate` держит весь батч до самой длинной. 8 кадров и 256 новых токенов укладываются примерно в 2k токенов, поэтому контекст обрезан до 4096. |
| Батч Qwen 8 и батч DOVER | Один ролик не загружает A100. Чанк Qwen заполняет `max_num_seqs`. У DOVER клипы фиксированы, 224². |
| DOVER и Qwen одновременно | На 80 ГБ обе модели живут вместе. Секунда пайплайна стремится ко времени более медленной стадии. |
| Модели грузятся один раз | vLLM — отдельный процесс, DOVER — в родительском. |
| Prefetch кадров | Следующий чанк декодируется на CPU, пока GPU занят текущим. |
| ffmpeg `select=eq(n,…)` и параллельный decode в `score_batch` | DOVER не гоняет полный RGB-скан OpenCV и не декодирует батч по одному ролику. `decode_workers: 0` — по числу роликов в батче. Тензоры на GPU собираются уже в главном потоке. |
| `replicate` и warmup | Десять роликов — это ещё прогрев CUDA. Стабильные видео/с считаются после него. |

FlashAttention-2 включается на пути `qwen.backend: transformers` (`attn_implementation: auto`). На пути vLLM attention выбирает сам сервер.

### Среднее влияние

Сделано точечно, без отдельного разгона:

- У Swin в DOVER выключен activation checkpointing, иначе eval считает граф дважды.
- ConvNeXt не качает ImageNet-init: `DOVER.pth` и так перезаписывает веса.
- Фрагменты DOVER собираются сразу в 224², полный кадр в RAM не держится.
- Кадры DOVER и Qwen читаются ffmpeg по списку индексов; OpenCV остаётся запасным путём, если pipe короткий.
- Для фиксированных шейпов DOVER включены TF32 и `cudnn.benchmark`. Веса DOVER остаются в fp32, чтобы не сдвигать официальный fusion.
- Общий декод на обе модели не делался: DOVER нужен полный кадр, Qwen — уже уменьшенный.

### Низкое влияние

Не делалось:

- формат JSON/CSV и прочий Python вокруг записи результатов;
- `torch.compile` для VLM;
- квантизация Qwen в 4/8 bit: 2B на A100 и так маленький, а текст описания от этого страдает;
- ONNX и TensorRT для DOVER: forward этой сети — единицы гигабайт и доля секунды, основное время DOVER сидит в CPU-decode.

### Одна A100 и кластер

На одной A100 не ускоряют конец пайплайна:

- speculative decoding;
- NVDEC вместо CPU-decode, пока ролики короткие;
- вторая карта: DOVER на одной, Qwen на другой.

8B и FP8 сюда не подходят. У A100 нет FP8-ядер, а Qwen3-VL-8B в несколько раз дороже 2B на prefill тех же кадров. `--max-model-len 32768` и `--kv-cache-dtype fp8_e5m2` для этого промпта почти ничего не дают: KV и так маленький.

Флаги, с которыми пайплайн поднимает сервер в параллельном режиме:

```bash
vllm serve Qwen/Qwen3-VL-2B-Instruct \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --max-num-seqs 8 \
  --gpu-memory-utilization 0.70 \
  --limit-mm-per-prompt.image 8 \
  --limit-mm-per-prompt.video 0 \
  --mm-processor-cache-gb 0 \
  --async-scheduling \
  --trust-remote-code \
  --host 127.0.0.1 \
  --port 8000
```

30% VRAM, которые остаются DOVER, ему хватает: чекпоинт 229 МБ, батч клипов 224² занимает доли гигабайта, пик активаций — единицы гигабайт. На 80 ГБ это 24 ГБ, на 40 ГБ — 12 ГБ.

## Ablation

```bash
python -m video_processing --input video --output results --ablate
```

Три точки Qwen: 4 кадра / сторона 256, 8 / 384, 16 / 384. Батчи Qwen и DOVER: 8 и 10. Батч 16 запускается только если 10 быстрее 8 хотя бы на 5% по видео/с. Выбранные батч и точка `8×384` записываются в `configs/default.yaml`. Сводка — `results/ABLATION.md`.

Ablation гоняет стадии по отдельности и переиспользует уже посчитанный DOVER. Совместный режим `parallel: true` измеряет обычный прогон, не таблица ablation.

## Конфиг

`configs/default.yaml`:

| Ключ | Смысл |
|---|---|
| `parallel` | `true` — DOVER и Qwen одновременно, vLLM на 70%. `false` — сначала весь DOVER, потом Qwen. |
| `replicate` | Сколько копий набора прогнать ради стабильной скорости. |
| `warmup_videos` | Сколько первых роликов не входит в видео/с. |
| `dover.batch_size` | Роликов в одном forward DOVER. |
| `qwen.num_frames`, `qwen.max_side` | Главный рычаг качества описания против скорости. |
| `qwen.batch_size`, `qwen.max_num_seqs` | Сколько роликов Qwen держит одновременно. Имеет смысл поднимать парой. |
| `qwen.max_model_len` | Потолок контекста vLLM. Для 8 кадров хватает 4096. |
| `qwen.backend` | `vllm` или `transformers`. |
| `qwen.gpu_memory_utilization` | Доля VRAM в последовательном режиме. В параллельном пайплайн подставляет 0.70. |
