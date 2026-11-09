#!/bin/bash
# grid_bench.py on one GPU, in run.sh's environment, restarted (it resumes) if it crashes.
#
#   tmux new -d -s grid "./grid_bench.sh SHAPES.csv OUT.csv 3"
set -u
V2=/home/jbowley/repos/aiter/streamk_study/tile_vs_tail/v2
SHAPES=$(realpath "$1")
OUT=$(realpath -m "$2")
GPU=$3
mkdir -p "$(dirname "$OUT")"
source $V2/../env.sh
export PYTHONPATH=$V2/flydsl_0341:$V2:$PYTHONPATH
# TVT_CACHE: a separate cache for the grid, so its kernels can be deleted when the study ends.
export TRITON_CACHE_DIR=${TVT_CACHE:-$V2/triton_cache}
export HIP_VISIBLE_DEVICES=$GPU
LOG=${OUT%.csv}.log
for attempt in 1 2 3 4 5; do
    python3 $V2/grid_bench.py "$SHAPES" "$OUT" "${@:4}" >> "$LOG" 2>&1 && break
    echo "attempt $attempt exited with $?, restarting $(date -Is)" >> "$LOG"
done
echo "finished $(date -Is)" >> "$LOG"
