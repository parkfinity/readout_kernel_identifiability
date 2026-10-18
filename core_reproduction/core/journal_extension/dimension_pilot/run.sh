#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
mode="${1:-smoke}"
shift || true
case "$mode" in
  smoke) python3 -m unittest -v test_engine ;;
  pilot) python3 -m unittest -v test_engine && python3 run_pilot.py "$@" ;;
  replay) python3 run_pilot.py --audit results/pilot_20261001T161006_106481Z --audit-output server_replay.json ;;
  *) echo 'Usage: bash run.sh smoke|pilot|replay' >&2; exit 2 ;;
esac
