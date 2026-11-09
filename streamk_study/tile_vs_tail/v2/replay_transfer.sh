#!/bin/bash
# --run_config replay of replay_transfer.py's CSVs, A and B passes, one file list per GPU.
#   GPUS=4,5,6,7 bash replay_transfer.sh runs/<name> [DIR]
# DIR is the replay directory in the run (default replay_transfer; replay_v9 for
# v9_transfer.py --replay-make).
set -u
V2=/home/jbowley/repos/aiter/streamk_study/tile_vs_tail/v2
RUN=$(cd "$1" && pwd)
DIR=${2:-replay_transfer}
GPUS=${GPUS:-4,5,6,7}
source $V2/../env.sh
export PYTHONPATH=$V2/flydsl_0341:$PYTHONPATH
export TRITON_CACHE_DIR=$V2/triton_cache
cd /home/jbowley/repos/aiter
T=csrc/gemm_a16w16/gemm_a16w16_tune.py
O=$RUN/$DIR
echo "start $DIR gpus=$GPUS $(date -Is)" >> $RUN/status.log
i=0
for g in $(echo $GPUS | tr , ' '); do
    (
        export HIP_VISIBLE_DEVICES=$g
        for p in a b; do
            while read -r f; do
                [ -n "$f" ] && python $T --run_config $O/$f --warmup 5 --iters 101 \
                    > $O/${f%.csv}_$p.log 2>&1
            done < $O/gpu$i.txt
        done
    ) &
    i=$((i + 1))
done
wait
echo "exit=$? end $DIR $(date -Is)" >> $RUN/status.log
