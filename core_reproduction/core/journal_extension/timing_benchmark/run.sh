#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
PYTHON="${PYTHON:-python3}"
case "${1:-}" in
  smoke|pilot|full|validation)
    "$PYTHON" -m unittest -v test_certificates test_incremental
    "$PYTHON" runner.py "$1"
    ;;
  audit)
    "$PYTHON" audit.py "${2:?Supply a results run directory}"
    ;;
  *) echo 'Usage: bash run.sh smoke|pilot|full|validation; bash run.sh audit results/<run>'; exit 2;;
esac
