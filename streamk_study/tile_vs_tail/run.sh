#!/bin/bash
# One iteration of the experiment on a candidates directory made by candidates.py:
#   warm (compile everything in parallel), tune (time every candidate, aiter's tuner with
#   AITER_TUNE_GLUON_CANDIDATES), picks (each method's best), replay (--run_config of
#   each method's pick and of the shipped kernel, one chunk of shapes per GPU), analyze.
#     GPUS=4,5,6,7 JOBS=64 bash run.sh runs/<name> [warm|tune|picks|replay|analyze ...]
# With no step names every step runs. Status lines go to runs/<name>/status.log.
set -u
TVT=/home/jbowley/repos/aiter/streamk_study/tile_vs_tail
RUN=$(cd "$1" && pwd); shift
STEPS=${*:-warm tune picks replay analyze}
GPUS=${GPUS:-4,5,6,7}
JOBS=${JOBS:-64}
NG=$(echo $GPUS | tr , '\n' | wc -l)
source $TVT/env.sh
cd /home/jbowley/repos/aiter
T=csrc/gemm_a16w16/gemm_a16w16_tune.py
log() { echo "$* $(date -Is)" >> $RUN/status.log; }

for step in $STEPS; do
    log "start $step gpus=$GPUS commit=$(git rev-parse --short HEAD)"
    case $step in
    warm)
        python $TVT/warm.py $RUN --jobs $JOBS --gpus $GPUS > $RUN/warm.log 2>&1 ;;
    tune)
        rm -f $RUN/tuned.csv $RUN/profile.csv
        HIP_VISIBLE_DEVICES=$GPUS AITER_TUNE_GLUON_CANDIDATES=$RUN/candidates.csv \
            python $T -i $RUN/untuned.csv -o $RUN/tuned.csv -o2 $RUN/profile.csv \
            --libtype gluon --mp $NG --shape_grouped --timeout 7200 > $RUN/tune.log 2>&1 ;;
    picks)
        python $TVT/picks.py $RUN > $RUN/picks.log 2>&1 ;;
    replay)
        python $TVT/replay_split.py $RUN $NG
        i=0
        for g in $(echo $GPUS | tr , ' '); do
            (
                export HIP_VISIBLE_DEVICES=$g
                R=$RUN/replay/$i
                python $T -i $R/shapes.csv -o $TVT/empty_tuned.csv --run_config --warmup 5 --iters 101 \
                    > $R/shipped.log 2>&1
                for grp in pow2_split pow2_dp uneven subtile prev; do
                    [ -s $R/$grp.csv ] && python $T --run_config $R/$grp.csv --warmup 5 --iters 101 \
                        > $R/$grp.log 2>&1
                done
            ) &
            i=$((i + 1))
        done
        wait ;;
    analyze)
        python $TVT/analyze.py $RUN > $RUN/analyze.log 2>&1 ;;
    esac
    log "exit=$? end $step"
done
