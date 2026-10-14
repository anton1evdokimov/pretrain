# Video Processing

Пайплайн для одной NVIDIA A100. На каждом ролике две модели:

1. **DOVER** — техническое и эстетическое качество, плюс общая оценка. Шкала `[0, 1]`, fusion как в официальном `evaluate_a_set_of_videos.py` на конфиге `val-l1080p`.
2. **Qwen3-VL-2B-Instruct** — описание. Промпт задания уходит модели дословно: «Сгенерируй детальное описание всего, что происходит на видео». Перед ним стоит системное сообщение, которое просит писать кратко и без выдумок (см. [Конфиг](#конфиг)).

**Результат на 10 видео** — [`results_final/RESULTS.md`](results_final/RESULTS.md): technical, aesthetic, overall и описание каждого ролика, плюс скорость. То же в `results.csv` и `results.json`.

Скорость на A100 80 ГБ — **3.41 видео/с в установившемся режиме** (около 12 300 видео в час), замер на 640 заданиях (`configs/default.yaml`, вывод в `results_final/`). Загрузка моделей — ещё около 50 с, один раз. Baseline — 0.69 видео/с. Какие подходы рассматривали, как их сгруппировали по влиянию и что реализовали — в [REPORT.md](REPORT.md).

## Запуск

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
# torch, torchvision и torchcodec одной CUDA-сборки (на сервере — cu130)
pip install torch torchvision torchcodec --index-url https://download.pytorch.org/whl/cu130
pip install -e '.[vllm]'
python scripts/bootstrap.py                   # код DOVER на проверенном коммите и weights/DOVER.pth
python -m video_processing                    # configs/default.yaml → results/, около 4 минут вместе с загрузкой
python -m unittest discover tests             # smoke-тест без GPU: заглушки вместо моделей
```

- Другой конфиг: `--config configs/....yaml`. Каталоги и число копий: `--input`, `--output`, `--replicate`.
- Пайплайн сам поднимает vLLM и сам его гасит. `scripts/serve_qwen.py` нужен, только чтобы посмотреть endpoint вручную. Вместе с пайплайном его не запускать: оба займут порт 8000 и одну и ту же память.
- Без `torchcodec` или без NVDEC каждый ролик автоматически уходит на ffmpeg на CPU. В stderr при этом пишется `[dover] NVDEC: … → cpu` или `[qwen] NVDEC: … → cpu`. Результат тот же, только медленнее.

## Что делает один прогон

1. Собирает ролики из `video/` и размножает список до `replicate` копий, чтобы выйти на установившуюся скорость.
2. Поднимает vLLM с Qwen3-VL-2B в bf16 на 50% VRAM и загружает DOVER в оставшуюся память. Загрузка считается отдельно от скорости (`load_sec`).
3. Запускает обе модели одновременно на одном списке роликов:
   - **DOVER.** Пул потоков готовит ролики без барьера на границе батча. Кадры декодирует NVDEC, виды aesthetic и technical собираются на GPU. Forward идёт батчами по `dover.batch_size`.
   - **Qwen.** Каждый ролик — отдельная задача: 8 кадров через NVDEC, уменьшение до 384 px по длинной стороне, запрос в vLLM. Кадры одновременно готовятся не больше чем для `qwen.decode_workers` роликов, в vLLM одновременно до `max_num_seqs` запросов. Готовая подпись сразу освобождает место следующей.
4. Считает видео/с по wall-clock (см. [Результаты](#результаты)).
5. Пишет оценки и описания и останавливает vLLM.

`parallel: false` — сначала весь DOVER, потом Qwen, vLLM берёт `gpu_memory_utilization` из конфига. `qwen.continuous: false` — подача пачками по `qwen.batch_size`, как в baseline.

## Результаты

Итоговый прогон лежит в `results_final/`, прогоны серии замеров — в `experiments/`. В каждом каталоге:

| Файл | Что внутри |
|---|---|
| `RESULTS.md` | Скорость, сводка подписей по всем копиям, таблица technical / aesthetic / overall с z-оценками и описаний копии 0 |
| `results.json`, `results.csv` | Оценки и описания копии 0. В JSON ещё `caption_stats` и `captions_all` |
| `captions_all.jsonl` | Подписи всех заданий: `finish_reason`, число слов, доля уникальных 8-грамм, токенов промпта |
| `throughput.json` | Скорость |

Поля `throughput.json`:

| Поле | Смысл |
|---|---|
| `videos_per_sec` | Установившийся режим: после `warmup_videos` и только пока заняты обе стадии. Если `note` начинается с «ВНИМАНИЕ», окно короче прогрева, и вместо него взято значение с хвостом. Поднимите `replicate`. |
| `videos_per_sec_with_tail` | То же, но с хвостом, когда одна стадия уже закончила |
| `makespan_sec`, `dover_end_sec`, `qwen_end_sec` | Общее время прогона и момент окончания каждой стадии, без загрузки моделей. Кто кончил последним — то и узкое место. |
| `load_sec` | Загрузка vLLM и DOVER |
| `captions_truncated_jobs`, `captions_looped_jobs` | Подписи, обрезанные по `max_tokens` и зациклившиеся (доля уникальных 8-грамм меньше 90%) |
| `qwen_prep_sec_mean`, `qwen_request_sec_mean` | В непрерывном режиме: подготовка кадров и запрос на один ролик |

## Конфиг

`configs/default.yaml` — итог серии замеров. У ключей с замером в комментарии указан прогон, на котором значение выбрано.

| Ключ | Значение | Смысл |
|---|---|---|
| `replicate`, `warmup_videos` | 16, 4 | Копий набора и сколько первых роликов не входит в видео/с |
| `parallel` | `true` | DOVER и Qwen одновременно, vLLM на 50% VRAM |
| `dover.decoder` | `nvdec` | `cpu` — ffmpeg и нарезка в Python |
| `dover.batch_size`, `dover.decode_workers` | 8, 8 | Роликов в forward и одновременно готовящихся роликов |
| `qwen.system_prompt` | краткость и запрет выдумок | Пусто — без системного сообщения |
| `qwen.num_frames`, `qwen.max_side` | 8, 384 | Визуальный бюджет: главный рычаг «качество против скорости» |
| `qwen.input` | `images` | `video` — те же кадры одним роликом. Хуже по фактам (`experiments/results_v7`, `results_v9`) |
| `qwen.decoder` | `nvdec` | `cpu` — ffmpeg |
| `qwen.continuous`, `qwen.max_num_seqs` | `true`, 64 | Непрерывная подача и предел одновременных запросов в vLLM |
| `qwen.decode_workers` | 8 | Одновременно готовящихся роликов |
| `qwen.max_tokens` | 1536 | Страховка от обрезания, длину задаёт системное сообщение |
| `qwen.temperature`, `repetition_penalty`, `presence_penalty`, `seed` | 0.3, 1.1, 0.0, 42 | Сэмплинг. Seed не даёт побайтной воспроизводимости: результат vLLM зависит от состава батча |
| `qwen.backend` | `vllm` | `transformers` — запасной путь без режима видео |
| `qwen.max_model_len` | 4096 | 8 кадров — около 850 токенов промпта плюс ответ |
| `cpu_threads` | 0 | Потоки torch и OpenCV. 2 замедлили DOVER (`experiments/results_v4`) |

С этими флагами пайплайн поднимает vLLM в параллельном режиме:

```bash
vllm serve Qwen/Qwen3-VL-2B-Instruct \
  --dtype bfloat16 --max-model-len 4096 --max-num-seqs 64 \
  --gpu-memory-utilization 0.50 \
  --limit-mm-per-prompt.image 8 --limit-mm-per-prompt.video 0 \
  --mm-processor-kwargs '{"min_pixels": 4096, "max_pixels": 147456}' \
  --mm-processor-cache-gb 0 --no-enable-prefix-caching \
  --trust-remote-code --host 127.0.0.1 --port 8000
```

- `max_pixels = max_side²`. Без него vLLM на старте профилирует кадр дефолтного размера Qwen и падает ещё до `/health`. В режиме видео бюджет умножается на `num_frames`, потому что для видео это бюджет на все кадры.
- Prefix caching выключен: копии одного ролика побайтно совпадают, и кэш завысил бы скорость.

## Скрипты

| Скрипт | Зачем |
|---|---|
| `scripts/run_series.sh name config [name config …]` | Прогоны по очереди, у каждого свои `logs/vmstat_*.txt`, `logs/*_stderr.txt` и лог vLLM. В конце — сравнение с `BASE` (по умолчанию `experiments/results_v11`) |
| `scripts/compare_runs.py dir_a dir_b …` | Скорость и качество подписей нескольких прогонов рядом |
| `scripts/dover_parity.py [--decoder nvdec] [--demo]` | Входы и оценки DOVER против официального кода, `--demo` — против z-оценок из README DOVER. Вывод на A100 — `experiments/results_parity_*.txt` |
| `scripts/dover_decode_bench.py` | Только подготовка DOVER: CPU против NVDEC, 1 и 8 потоков |
| `tests/test_pipeline.py` | Smoke-тест: весь `run_pipeline` с заглушками DOVER и Qwen, обе схемы подачи в Qwen, запись результатов |

`experiments/configs/v8_*.yaml`–`v12_*.yaml` — точные настройки прогонов из истории в [REPORT.md](REPORT.md), каждый отличается от предыдущего одним изменением. `ab_*.yaml` — ранний A/B штрафов за повторы. `v12_detailed.yaml` — отвергнутый вариант системного сообщения без «кратко».
