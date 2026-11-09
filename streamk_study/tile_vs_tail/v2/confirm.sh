#!/bin/bash
# Replay the picks of RUN/confirm/*.csv (analyze_grid.py confirm-setup) through aiter's tuner,
# --run_config with run.sh's replay flags, on one GPU, then score the rules (confirm.md).
#
#   tmux new -d -s confirm "./confirm.sh runs/grid 3"
set -eu
V2=/home/jbowley/repos/aiter/streamk_study/tile_vs_tail/v2
RUN=$(realpath "$1")
GPU=$2
source $V2/../env.sh
export PYTHONPATH=$V2/flydsl_0341:$PYTHONPATH
export HIP_VISIBLE_DEVICES=$GPU
cd /home/jbowley/repos/aiter
for f in $RUN/confirm/*.csv; do
    log=${f%.csv}.log
    [ -s $log ] && grep -q "| *OK" $log && continue
    case $(basename $f) in
        aiter_*) export TRITON_CACHE_DIR=$V2/triton_cache ;;
        *) export TRITON_CACHE_DIR=$V2/triton_cache_grid ;;
    esac
    python csrc/gemm_a16w16/gemm_a16w16_tune.py --run_config $f --warmup 5 --iters 101 > $log 2>&1
done
cd $V2
python3 analyze_grid.py confirm $RUN > $RUN/confirm.out 2>&1
