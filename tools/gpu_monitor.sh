#!/usr/bin/env bash
# GPU monitor launcher: records nvidia-smi stats with timestamps to logs/
#
# Usage:
#   tools/gpu_monitor.sh                  # monitor until Ctrl-C
#   tools/gpu_monitor.sh -i 1             # set interval (seconds)
#   tools/gpu_monitor.sh -- CMD ARGS...   # run CMD while monitoring; stop when CMD exits
#
# Output: logs/gpu_mon_YYYYmmdd_HHMMSS.csv (CSV with timestamped metrics)

set -euo pipefail

interval=1
if [[ "${1:-}" == "-i" && -n "${2:-}" ]]; then
  interval="$2"; shift 2
fi

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
LOG_DIR="$ROOT_DIR/logs"
mkdir -p "$LOG_DIR"
ts=$(date +%Y%m%d_%H%M%S)
out="$LOG_DIR/gpu_mon_${ts}.csv"

query="timestamp,clocks.sm,clocks.mem,temperature.gpu,power.draw,utilization.gpu,utilization.memory,memory.total,memory.used,memory.free"

echo "[gpu-monitor] Writing to: $out"

if [[ "${1:-}" == "--" ]]; then
  shift
  echo "[gpu-monitor] Starting nvidia-smi (interval=${interval}s)"
  nvidia-smi --query-gpu=${query} --format=csv -l ${interval} -f "$out" &
  mon_pid=$!
  echo "[gpu-monitor] Monitor PID: $mon_pid"
  echo "[gpu-monitor] Running command: $*"
  set +e
  "$@"
  rc=$?
  set -e
  echo "[gpu-monitor] Stopping monitor (PID $mon_pid)"
  kill "$mon_pid" 2>/dev/null || true
  wait "$mon_pid" 2>/dev/null || true
  echo "[gpu-monitor] Done. Exit code: $rc. Log: $out"
  exit $rc
else
  echo "[gpu-monitor] Monitoring only. Ctrl-C to stop."
  nvidia-smi --query-gpu=${query} --format=csv -l ${interval} -f "$out"
fi

