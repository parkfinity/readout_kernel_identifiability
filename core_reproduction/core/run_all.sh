#!/usr/bin/env bash
# reproduction_bundle_v1 -- re-run both certificate computations and compare with the stored references.
#   (1) Fourier-Gram closure of the 30 frozen settings (fourier_gram_identifiability_v1, ref 20261002T044925Z)
#   (2) Kyriienko scaling certificates n = 2, 4, 6, 8 (kyriienko_scaling_v1, ref 20261003T121202Z)
#
# Usage:   [PY=/path/to/python] [WORKERS=8] [OUT_ROOT=<dir>] bash run_all.sh
# Output:  $OUT_ROOT/<UTC>/ (default runs/<UTC>/) and $OUT_ROOT/server_return_<UTC>.tar.gz (send this back).
# Exit:    0 = PASS or PASS_WITH_WARNINGS, 1 = FAIL (see runs/<UTC>/comparison_verdict.json).
# The scientific algorithms are preserved in journal_extension/ (a mirror of the needed subset).
set -uo pipefail

BUNDLE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY="${PY:-python3}"
OUT_ROOT="${OUT_ROOT:-$BUNDLE/runs}"
if [ -z "${WORKERS:-}" ]; then
  NCPU="$( (nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 8) | head -n 1)"
  WORKERS=$(( NCPU < 8 ? NCPU : 8 ))
fi

# ---- deterministic, single-threaded numerics -------------------------------------------------
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
       VECLIB_MAXIMUM_THREADS=1 RAYON_NUM_THREADS=1 PYTHONHASHSEED=0 PYTHONDONTWRITEBYTECODE=1 \
       PYTHONUNBUFFERED=1 PYTHONINTMAXSTRDIGITS=0
# PYTHONINTMAXSTRDIGITS=0: newer CPython patch releases cap int<->str at 4300 digits; the exact
# minor determinants have up to ~16k digits.  Ignored by interpreters without the cap (e.g. 3.8.10).

command -v "$PY" >/dev/null 2>&1 || { echo "Python interpreter '$PY' not found; set PY=..."; exit 2; }
TS="$(date -u +%Y%m%dT%H%M%SZ)"
RUN="$OUT_ROOT/$TS"
mkdir -p "$RUN/logs" || exit 2
exec 3>&1 4>&2
exec > >(tee -a "$RUN/logs/run_all.log") 2>&1

HELPER="$BUNDLE/tools/repro_helper.py"
FG="$BUNDLE/journal_extension/fourier_gram_identifiability_v1/experiment"
KS="$BUNDLE/journal_extension/kyriienko_scaling_v1"
STATUS="$RUN/stage_status.json"
T0=$SECONDS

stage() {  # stage <name> <cwd> <command...>
  local name="$1" cwd="$2"; shift 2
  "$PY" "$HELPER" run --name "$name" --cwd "$cwd" --log "$RUN/logs/$name.log" --status "$STATUS" -- "$@"
}

echo "== reproduction_bundle_v1  run $TS  (bundle $BUNDLE, workers $WORKERS)"
echo "== [0/5] environment and bundle integrity"
"$PY" "$HELPER" env --out "$RUN/environment_check.json"
"$PY" "$HELPER" integrity --bundle "$BUNDLE" --out "$RUN/bundle_integrity.json"

echo "== [1/5] unit tests (11 Fourier-Gram, 14 Kyriienko)"
stage fourier_unit_tests "$FG" "$PY" -m unittest -v test_fourier_gram
# pyexec installs the variadic math.gcd shim on Python < 3.9 (kyriienko_scaling.py needs it).
stage kyriienko_unit_tests "$KS" "$PY" "$HELPER" pyexec -- -m unittest -v test_kyriienko_scaling

echo "== [2/5] Fourier-Gram closure, 30 settings ($WORKERS worker processes)"
mkdir -p "$FG/results"
BEFORE="$(ls -1 "$FG/results")"
stage fourier_gram_run "$FG" "$PY" run_experiment.py --workers "$WORKERS"
NEW=""
for d in $(ls -1 "$FG/results"); do
  printf '%s\n' "$BEFORE" | grep -qx "$d" || NEW="$d"
done
if [ -n "$NEW" ]; then
  mv "$FG/results/$NEW" "$RUN/fourier_gram" && echo "Fourier-Gram results -> $RUN/fourier_gram (internal timestamp $NEW)"
else
  echo "WARNING: no new Fourier-Gram results folder was created"
fi

echo "== [3/5] Kyriienko scaling certificates, n = 2, 4, 6, 8"
stage kyriienko_scaling_run "$KS" "$PY" "$HELPER" pyexec -- kyriienko_scaling.py --sizes 2 4 6 8 \
      --output "$RUN/kyriienko_scaling"

echo "== [4/5] comparison with the stored reference results"
"$PY" "$BUNDLE/compare_results.py" --run-dir "$RUN"
VERDICT_CODE=$?

echo "== [5/5] packing"
echo "total wall time: $(( SECONDS - T0 )) s" | tee "$RUN/total_wall_seconds.txt"
exec 1>&3 2>&4 3>&- 4>&-   # close the log tee before packing it
sleep 1
tar -czf "$OUT_ROOT/server_return_$TS.tar.gz" -C "$OUT_ROOT" "$TS" \
  && echo "SEND BACK: $OUT_ROOT/server_return_$TS.tar.gz" \
  || echo "WARNING: packing failed; send the folder $RUN instead"
exit $VERDICT_CODE
