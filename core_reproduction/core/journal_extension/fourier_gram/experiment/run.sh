#!/usr/bin/env bash
# Fourier-Gram rank certification. Reads ../../certified_dimension_extension_v1 (never writes it).
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONHASHSEED=0
case "${1:-full}" in
  test) python3 -m unittest -v test_fourier_gram ;;
  full) python3 -m unittest test_fourier_gram && python3 run_experiment.py --workers "${WORKERS:-8}" ;;
  brackets) python3 run_experiment.py --workers 3 --only \
      kyriienko_nonlinear_encoding_circuit_n2_l1_local_xyz \
      kyriienko_nonlinear_encoding_circuit_n4_l1_local_xyz \
      kyriienko_nonlinear_encoding_circuit_n4_l1_pauli_weight_le_2 ;;
  *) echo 'Use: test | full | brackets'; exit 2 ;;
esac
