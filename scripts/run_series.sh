#!/usr/bin/env bash
# Прогоны по очереди, у каждого свой vmstat, stderr и лог vLLM в logs/.
# Не одновременно: все делят один GPU, и скорости были бы несравнимы.
#
#   bash scripts/run_series.sh v12 experiments/configs/v12_detailed.yaml
#   В конце — compare_runs по базе (BASE, по умолчанию experiments/results_v11)
#   и output_dir всех запущенных конфигов.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs

run() {
  local name=$1 config=$2
  echo "=== $name: $config ==="
  vmstat 5 > "logs/vmstat_$name.txt" &
  local vm=$!
  python -m video_processing --config "$config" 2> >(tee "logs/${name}_stderr.txt" >&2)
  local code=$?
  kill "$vm" 2>/dev/null
  cp logs/vllm-serve.log "logs/vllm-serve_$name.log" 2>/dev/null
  echo "$name: exit=$code, NVDEC → cpu: $(grep -c 'NVDEC:' "logs/${name}_stderr.txt")"
}

output_dir() {
  python -c "import sys; from video_processing.config import load_config; print(load_config(sys.argv[1]).output_dir)" "$1"
}

dirs=("${BASE:-experiments/results_v11}")
while [ $# -ge 2 ]; do
  run "$1" "$2"
  dirs+=("$(output_dir "$2")")
  shift 2
done
python scripts/compare_runs.py "${dirs[@]}"
