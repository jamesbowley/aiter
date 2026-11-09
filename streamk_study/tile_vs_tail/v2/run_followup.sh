#!/bin/bash
# After the grid, on one GPU: the padding supplement (run_grid.sh on runs/grid_pad), then the
# MFMA counters and thread traces of the grid's winners (mfma_profile.sh).
#
#   tmux new -d -s followup "./run_followup.sh 3"
set -eu
cd /home/jbowley/repos/aiter/streamk_study/tile_vs_tail/v2
GPU=$1
export TVT_CACHE=$PWD/triton_cache_grid
[ -f runs/grid_pad/DONE ] || ./run_grid.sh runs/grid_pad $GPU > runs/grid_pad/run.out 2>&1
[ -f runs/grid/mfma_counters.csv ] || ./mfma_profile.sh runs/grid $GPU counters > runs/grid/mfma_counters.log 2>&1
[ -f runs/grid/mfma_att.csv ] || ./mfma_profile.sh runs/grid $GPU att > runs/grid/mfma_att.log 2>&1
echo done > runs/grid/FOLLOWUP_DONE
