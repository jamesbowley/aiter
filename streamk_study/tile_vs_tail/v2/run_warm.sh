#!/bin/bash
# Compile every candidate of a run (RUN/candidates.csv) into the shared Triton cache with
# warm.py, on one GPU, in run.sh's environment.
#
#   ./run_warm.sh runs/<name> 3
set -eu
V2=/home/jbowley/repos/aiter/streamk_study/tile_vs_tail/v2
RUN=$(realpath "$1")
GPU=$2
source $V2/../env.sh
export PYTHONPATH=$V2/flydsl_0341:$PYTHONPATH
# TVT_CACHE: a separate cache for the grid, so its kernels can be deleted when the study ends.
export TRITON_CACHE_DIR=${TVT_CACHE:-$V2/triton_cache}
python3 $V2/warm.py $RUN --jobs 64 --gpus $GPU > $RUN/warm.log 2>&1
