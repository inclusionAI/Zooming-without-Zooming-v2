#!/usr/bin/env bash
set -eu
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export PYTHONPATH="${SCRIPT_DIR%/scripts}${PYTHONPATH:+:$PYTHONPATH}"

if [ "${ZWZ_PLATFORM:-generic}" = ppu ]; then
    test -r /usr/local/PPU_SDK/envsetup.sh
    # The vendor SDK reads positional parameters and is not nounset-safe.
    set +u
    source /usr/local/PPU_SDK/envsetup.sh CUDA
    set -u
fi

if [ "$#" -lt 1 ]; then
    echo 'Usage: bash scripts/evaluate.sh PROJECT_DIR [BENCH,BENCH|all] [infer|score|run] [--services local|external]' >&2
    echo 'An explicit ready.json CONFIG is also supported instead of PROJECT_DIR.' >&2
    exit 2
fi

PYTHON="${PYTHON:-python3}"
CONFIG=$1
shift
if [ -d "$CONFIG" ]; then CONFIG="$CONFIG/ready.json"; fi
BENCHMARKS=all
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    BENCHMARKS=$1
    shift
fi
PHASE=all
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    PHASE=$1
    shift
fi
if [ "$PHASE" = run ]; then PHASE=all; fi
"$PYTHON" -m zwz_eval evaluate --config "$CONFIG" --bench "$BENCHMARKS" --phase "$PHASE" "$@"
