#!/bin/bash
# The leftover sweep (ours and hipBLASLt's pick) and then hipBLASLt's exhaustive tuning of the
# same shapes, one after the other on one GPU.
#
#   tmux new -d -s leftover "./run_leftover.sh 3"
set -eu
cd "$(dirname "$0")"
GPU=$1
mkdir -p runs/sweep_leftover runs/hbl_tune
HIP_VISIBLE_DEVICES=$GPU python3 sweep_leftover.py runs/sweep_leftover > runs/sweep_leftover/sweep.log 2>&1
./hbl_tune.sh runs/hbl_tune "$GPU" > runs/hbl_tune/tune.log 2>&1
python3 hbl_tune_parse.py runs/hbl_tune > runs/hbl_tune/summary.txt 2>&1
echo done > runs/hbl_tune/DONE
