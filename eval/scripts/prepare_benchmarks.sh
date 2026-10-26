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

if [ "$#" -lt 1 ]; then
    echo 'Usage: bash scripts/prepare_benchmarks.sh PROJECT_DIR [BENCH,BENCH|all]' >&2
    echo 'Advanced: use MANIFEST instead of PROJECT_DIR, with optional --config TEMPLATE --output-config LOCAL.' >&2
    exit 2
fi
MANIFEST=$1
shift
PROJECT=
if [ -d "$MANIFEST" ]; then
    PROJECT=$MANIFEST
    MANIFEST="$PROJECT/datasets.json"
    test -r "$MANIFEST" && test -r "$PROJECT/model.json" || {
        echo 'Project directory requires datasets.json and model.json; run init first.' >&2
        exit 2
    }
fi
BENCHMARKS=all
if [ "$#" -gt 0 ] && [[ "$1" != --* ]]; then
    BENCHMARKS=$1
    shift
fi
if [ -n "$PROJECT" ]; then
    for ARG in "$@"; do
        case "$ARG" in
            --config|--config=*|--output-config|--output-config=*)
                echo 'Use a manifest file, not a project directory, to override config paths.' >&2
                exit 2 ;;
        esac
    done
    "$PYTHON" -m zwz_eval prepare-suite --manifest "$MANIFEST" --bench "$BENCHMARKS" \
        --config "$PROJECT/model.json" --output-config "$PROJECT/ready.json" "$@"
else
    "$PYTHON" -m zwz_eval prepare-suite --manifest "$MANIFEST" --bench "$BENCHMARKS" "$@"
fi
