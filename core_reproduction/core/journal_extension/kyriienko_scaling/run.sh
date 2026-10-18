#!/usr/bin/env bash
# Kyriienko-type kernel rank scaling (n = 2, 4 validation gate; n = 6, 8 certification).
# Reads (never writes) ../cross_family_resource_v1, ../certified_dimension_extension_v1,
# ../manuscript_theory_reference.  Uses the Qiskit 1.2.4 venv of the frozen census if present.
set -euo pipefail
cd "$(dirname "$0")"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1
PY="${PY:-../.venv_compiler_v2/bin/python}"
[ -x "$PY" ] || PY=python3
"$PY" -c 'import qiskit,sys; v=qiskit.__version__; print("python", sys.version.split()[0], "qiskit", v); sys.exit(0 if v=="1.2.4" else 3)' \
  || { echo "Qiskit 1.2.4 (frozen census compiler) not found; set PY=<python with qiskit==1.2.4>"; exit 3; }
case "${1:-full}" in
  test) "$PY" -m unittest -v test_kyriienko_scaling ;;
  gate) "$PY" -m unittest test_kyriienko_scaling && "$PY" kyriienko_scaling.py --sizes 2 4 ;;
  full) "$PY" -m unittest test_kyriienko_scaling && "$PY" kyriienko_scaling.py --sizes 2 4 6 8 ;;
  verify) "$PY" kyriienko_scaling.py --verify "${2:?results folder}" ;;
  *) echo 'Use: test | gate | full | verify <results/TIMESTAMP>'; exit 2 ;;
esac
