# Ablation: скорость vs качество

Таблица заполняется на A100:

```bash
python -m video_processing --input video --output results --ablate
```

Точки Qwen: 4 кадра / 256, 8 / 384, 16 / 384. Батч 8 vs 10, затем 16 если 10 даёт ≥5% к throughput.
