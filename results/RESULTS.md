# Результаты Video Processing

Прогон рассчитан на **NVIDIA A100**. На этой машине его ещё не запускали: здесь нет CUDA.

```bash
python scripts/bootstrap.py
python -m video_processing --input video --output results
```

После прогона здесь появятся technical / aesthetic / overall, описания и стабильная пропускная способность.
