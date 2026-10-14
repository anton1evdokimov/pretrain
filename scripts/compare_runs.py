"""Side-by-side throughput and caption quality of several results_* dirs.

    python scripts/compare_runs.py results_v5 results_v6 results_v7
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from video_processing.pipeline import LOOP_UNIQ8, _uniq8

THROUGHPUT = [
    "videos_per_sec_with_tail",
    "videos_per_sec",
    "makespan_sec",
    "dover_end_sec",
    "qwen_end_sec",
    "captions_truncated_jobs",
    "captions_looped_jobs",
]


def _load(path: Path) -> dict | None:
    file = path / "results.json"
    if not file.exists():
        print(f"{path}: нет results.json")
        return None
    return json.loads(file.read_text(encoding="utf-8"))


def _stats(payload: dict) -> dict:
    if payload.get("caption_stats"):
        return payload["caption_stats"]
    # Старые прогоны хранят одну подпись на ролик (копию 0).
    out = {}
    for row in payload["results"]:
        uniq = _uniq8(row["caption"])
        out[Path(row["source"]).name] = {
            "jobs": 1,
            "truncated": int(row["caption_finish_reason"] == "length"),
            "looped": int(uniq < LOOP_UNIQ8),
            "words_mean": len(row["caption"].split()),
            "uniq8_mean": uniq,
            "prompt_tokens": 0,
        }
    return out


def main() -> None:
    dirs = [Path(arg) for arg in sys.argv[1:]]
    runs = {d.name: p for d in dirs if (p := _load(d)) is not None}
    names = list(runs)
    width = max(len(n) for n in names) + 2 if names else 10

    print("\n" + "metric".ljust(26) + "".join(n.rjust(width) for n in names))
    for key, default in (("dover_decoder", "cpu"), ("qwen_input", "images")):
        print(key.ljust(26) + "".join(str(runs[n]["config"].get(key, default)).rjust(width) for n in names))
    for key in THROUGHPUT:
        print(key.ljust(26) + "".join(str(runs[n]["throughput"].get(key, "—")).rjust(width) for n in names))

    stats = {n: _stats(runs[n]) for n in names}
    videos = sorted({v for s in stats.values() for v in s}, key=lambda v: int(Path(v).stem) if Path(v).stem.isdigit() else v)
    print(f"\nзациклено / копий (уник. 8-грамм < {LOOP_UNIQ8:.0%}), слов в среднем, токенов промпта")
    print("video".ljust(10) + "".join(n.rjust(26) for n in names))
    for video in videos:
        cells = []
        for n in names:
            s = stats[n].get(video)
            cells.append(
                "—".rjust(26)
                if s is None
                else f"{s['looped']}/{s['jobs']}  {s['words_mean']:.0f}w  {s['prompt_tokens']:.0f}t".rjust(26)
            )
        print(video.ljust(10) + "".join(cells))


if __name__ == "__main__":
    main()
