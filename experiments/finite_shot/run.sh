#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python3 -m unittest -v test_shots
python3 -c 'from core import load_plan; load_plan(); print("Frozen plan hashes: OK")'
case "${1:-smoke}" in
  smoke) ;;
  pilot) python3 experiment.py --mode pilot; python3 analyze.py ;;
  all|full) python3 experiment.py --mode full; python3 analyze.py ;;
  *) echo "Usage: bash run.sh smoke|pilot|all"; exit 2 ;;
esac
