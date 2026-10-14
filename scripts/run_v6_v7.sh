#!/usr/bin/env bash
# v6 (NVDEC) и v7 (NVDEC + видео в Qwen) подряд, каждый со своим vmstat и stderr.
# Прогоны не одновременные: оба делят один GPU, и их скорость была бы несравнима.
#
#   bash scripts/run_v6_v7.sh
#   python scripts/compare_runs.py results_v5 results_v6 results_v7
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
  echo "$name: exit=$code, роликов DOVER ушло в cpu: $(grep -c 'NVDEC:' "logs/${name}_stderr.txt")"
  return "$code"
}

run v6 configs/v6_nvdec.yaml; v6=$?
run v7 configs/v7_nvdec_video.yaml; v7=$?
echo "v6 exit=$v6, v7 exit=$v7"
python scripts/compare_runs.py results_v5 results_v6 results_v7
