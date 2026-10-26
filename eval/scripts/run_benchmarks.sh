#!/usr/bin/env bash
set -eu
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${SCRIPT_DIR%/scripts}${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python3}"

if [ "${ZWZ_PLATFORM:-generic}" = ppu ]; then
    test -r /usr/local/PPU_SDK/envsetup.sh
    set +u
    source /usr/local/PPU_SDK/envsetup.sh CUDA
    set -u
fi

if [ "$#" -lt 2 ]; then
    echo 'Usage: bash scripts/run_benchmarks.sh CONFIG BENCH,BENCH|all [infer|score|all|audit|plan] [--continue-on-error]' >&2
    exit 2
fi
CONFIG=$1
BENCHMARKS=$2
shift 2
PHASE=all
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    PHASE=$1
    shift
fi
"$PYTHON" -m zwz_eval run-suite --config "$CONFIG" --bench "$BENCHMARKS" --phase "$PHASE" "$@"
