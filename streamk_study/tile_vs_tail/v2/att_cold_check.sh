#!/bin/bash
# Why the cold 256x256 trace on 4096x4096x8192 reads lower than elsewhere: trace the second call
# (right after compiling and allocating, the GPU idle before it) against a call deep in a cold
# rotation (the 30th of 40), with warm as a control. One GPU; raw output and cache deleted.
#
#   tmux new -d -s coldchk "./att_cold_check.sh runs/att_cold_check 3"
set -eu
V2=/home/jbowley/repos/aiter/streamk_study/tile_vs_tail/v2
RUN=$(realpath -m "$1")
GPU=$2
mkdir -p $RUN
cd $V2
source $V2/../env.sh
export PYTHONPATH=$V2/flydsl_0341:$V2:$PYTHONPATH
export TRITON_CACHE_DIR=$V2/triton_cache_coldchk
export HIP_VISIBLE_DEVICES=$GPU
SDK=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel
export PATH=$SDK/bin:$PATH
export LD_LIBRARY_PATH=$SDK/lib:${LD_LIBRARY_PATH:-}
PJ=/home/jbowley/repos/gfx950-gluon-tutorials/scripts/process_json.py
OUT=$RUN/cold_check.csv
TMP=$RUN/prof_tmp
echo "cond,iteration,reps,sets,repeat,pj_per_wave,ours_per_wave" > $OUT
for repeat in 1 2; do
    for cond in "second_cold 2 10 cold" "deep_cold 30 40 cold" "second_warm 2 10 1" "deep_warm 30 40 1"; do
        set -- $cond
        name=$1; it=$2; reps=$3; sets=$4
        rm -rf $TMP && mkdir -p $TMP
        MFMA_SETS=$sets MFMA_REPS=$reps rocprofv3 -d $TMP/att -o att \
            --att --att-target-cu 0 --att-shader-engine-mask 0x1 --att-buffer-size 0x40000000 \
            --kernel-include-regex _gemm_a16w16 --kernel-iteration-range "[$it]" \
            -- python3 $V2/mfma_run.py 4096 4096 8192 gluon_compute_bound_256x256 > $TMP/run.out 2>&1 || true
        ui=$(ls -d $TMP/att/ui_output_agent_* 2>/dev/null | head -1)
        pj=$(python3 $PJ "$ui" 2>/dev/null | python3 -c "import json,sys; print(float(json.load(sys.stdin)['mfma efficiency'].rstrip('%'))/100)" || echo "")
        ours=$(python3 -c "
from mfma_parse import parse_att_dir
r=[x for x in parse_att_dir('$TMP/att') if x['loop']=='main']; print(r[0]['mfma_issue_eff'] if r else '')")
        echo "$name,$it,$reps,$sets,$repeat,$pj,$ours" >> $OUT
    done
done
rm -rf $TMP $TRITON_CACHE_DIR
echo done > $RUN/DONE
