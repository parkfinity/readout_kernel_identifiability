#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
case "${1:-pilot}" in
  smoke) python3 -m unittest -v test_design.py ;;
  pilot|all) python3 -m unittest -v test_design.py; python3 experiment.py; python3 exact_controls.py; python3 verify_certificates.py; python3 followup.py; python3 correlation_control.py ;;
  exact) python3 exact_controls.py ;;
  *) echo 'Usage: bash run.sh smoke|pilot|all|exact'; exit 2 ;;
esac
