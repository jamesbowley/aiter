#!/bin/bash
# The tile-against-tail comparison on v2's kernels (gemm_a16w16.py, gemm_a16w16_uneven.py,
# gemm_a16w16_subtile.py, gemm_a16w16_streamk.py), each step timed with aiter's tuner:
#   shapes (shapes.py), candidates (candidates.py), warm (compile everything in parallel),
#   tune (time every candidate: AITER_TUNE_GLUON_CANDIDATES), picks (each method's best),
#   replay (--run_config of each method's pick and of the shipped kernel, one chunk of
#   shapes per GPU), analyze.
#     GPUS=4,5,6,7 JOBS=64 bash run.sh runs/<name> [shapes|candidates|warm|tune|picks|replay|analyze ...]
# With no step names every step runs. Status lines go to runs/<name>/status.log.
set -u
V2=/home/jbowley/repos/aiter/streamk_study/tile_vs_tail/v2
mkdir -p "$1"
RUN=$(cd "$1" && pwd); shift
STEPS=${*:-shapes candidates warm tune picks replay analyze}
GPUS=${GPUS:-4,5,6,7}
JOBS=${JOBS:-64}
NG=$(echo $GPUS | tr , '\n' | wc -l)
source $V2/../env.sh
# env.sh's flydsl 0.3.4.1 directory is gone; the same package, linked from a venv.
export PYTHONPATH=$V2/flydsl_0341:$PYTHONPATH
export TRITON_CACHE_DIR=$V2/triton_cache
cd /home/jbowley/repos/aiter
T=csrc/gemm_a16w16/gemm_a16w16_tune.py
log() { echo "$* $(date -Is)" >> $RUN/status.log; }

for step in $STEPS; do
    log "start $step gpus=$GPUS commit=$(git rev-parse --short HEAD) dirty=$(git status --porcelain -uno | wc -l)"
    case $step in
    shapes)
        python $V2/shapes.py $RUN > $RUN/shapes.log 2>&1 ;;
    candidates)
        HIP_VISIBLE_DEVICES=${GPUS%%,*} python $V2/candidates.py $RUN > $RUN/candidates.log 2>&1 ;;
    warm)
        python $V2/warm.py $RUN --jobs $JOBS --gpus $GPUS > $RUN/warm.log 2>&1 ;;
    tune)
        rm -f $RUN/tuned.csv $RUN/profile.csv
        HIP_VISIBLE_DEVICES=$GPUS AITER_TUNE_GLUON_CANDIDATES=$RUN/candidates.csv \
            python $T -i $RUN/untuned.csv -o $RUN/tuned.csv -o2 $RUN/profile.csv \
            --libtype gluon --mp $NG --shape_grouped --timeout 14400 > $RUN/tune.log 2>&1 ;;
    picks)
        python $V2/picks.py $RUN > $RUN/picks.log 2>&1 ;;
    replay)
        python $V2/replay_split.py $RUN $NG
        i=0
        for g in $(echo $GPUS | tr , ' '); do
            (
                export HIP_VISIBLE_DEVICES=$g
                R=$RUN/replay/$i
                python $T -i $R/shapes.csv -o $V2/../empty_tuned.csv --run_config --warmup 5 --iters 101 \
                    > $R/shipped.log 2>&1
                for grp in base uneven subtile subtile_rule streamk streamk_dp; do
                    [ -s $R/$grp.csv ] && python $T --run_config $R/$grp.csv --warmup 5 --iters 101 \
                        > $R/$grp.log 2>&1
                done
            ) &
            i=$((i + 1))
        done
        wait ;;
    analyze)
        python $V2/analyze.py $RUN > $RUN/analyze.log 2>&1 ;;
    esac
    log "exit=$? end $step"
done
