#!/bin/sh
# Regenerate reference tables under tests/data/ from the bundled cdflib.c
# sources. Run from the repository root: `tests/regenerate/regenerate.sh`.
#
# Re-run when the parameter grid or the cdflib.c reference changes; the
# generated CSVs are committed.

set -eu

ROOT=$(cd "$(dirname "$0")/../.." && pwd)
cd "$ROOT"

BUILD_DIR="tests/regenerate/build"
mkdir -p "$BUILD_DIR" tests/data

REFS_DIR="tests/regenerate/refs"
# One generator per logical group. Each writes one or more CSVs under
# tests/data/. The Rust test files in tests/ read those CSVs.
GENERATORS="
    erf_normal_kernels
    normal_distribution
    gamma_kernels
    beta_kernels
    beta_distributions
    discrete_distributions
    noncentral_distributions
"
for name in $GENERATORS; do
    SRC="tests/regenerate/gen_${name}.c"
    if [ ! -f "$SRC" ]; then
        continue
    fi
    BIN="$BUILD_DIR/gen_${name}"
    echo "compiling $SRC -> $BIN"
    cc -O2 -Wall -Wno-unused-result \
        -I tests/regenerate \
        -o "$BIN" "$SRC" "$REFS_DIR/cdflib.c" -lm
    echo "running $BIN"
    "$BIN"
done

echo "done"
