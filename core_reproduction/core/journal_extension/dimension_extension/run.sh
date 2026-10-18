#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
case "${1:-smoke}" in
  smoke) python3 -m unittest discover -v ;;
  full) python3 experiment.py ;;
  replay) python3 experiment.py --replay results/extension_20261001T232302_991451Z --audit-output server_replay.json ;;
  source-check) python3 source_check.py results/extension_20261001T232302_991451Z --output server_source_check.json ;;
  closure) python3 structural_closure.py results/extension_20261001T232302_991451Z --output server_closure.json ;;
  nonlinear) python3 nonlinear_hubregtsen.py --output server_nonlinear_hubregtsen.json; python3 check_nonlinear.py --output server_nonlinear_source_check.json --proof server_nonlinear_hubregtsen.json ;;
  *) echo 'Use smoke | full | replay | source-check | closure | nonlinear'; exit 2 ;;
esac
