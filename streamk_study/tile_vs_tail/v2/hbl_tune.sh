#!/bin/bash
# Exhaustive hipBLASLt tuning (every supported solution, --algo_method all) of the leftover sweep
# (M = 1024, K = 7168, N = 16384..32768) under three Stream-K grid policies:
#   default   origami's k_split_aware grid (TENSILE_STREAMK_DYNAMIC_GRID unset, = 6)
#   fixed256  TENSILE_STREAMK_FIXED_GRID=256: every Stream-K kernel splits its tiles over all 256 CUs
#   dp        TENSILE_STREAMK_DYNAMIC_GRID=4: data-parallel, one workgroup per tile, no split
#
#   ./hbl_tune.sh runs/hbl_tune "0 1 2 4 5 6 7"
set -u
OUT=$(realpath -m "$1")
GPUS=($2)
BENCH=/home/jbowley/therock/10.0.0
mkdir -p "$OUT"

job() {
    local mode=$1 n=$2 gpu=$3
    local env=()
    case $mode in
        fixed256) env=(TENSILE_STREAMK_FIXED_GRID=256) ;;
        dp) env=(TENSILE_STREAMK_DYNAMIC_GRID=4) ;;
    esac
    env "${env[@]}" HIP_VISIBLE_DEVICES=$gpu LD_LIBRARY_PATH=$BENCH/lib \
        $BENCH/bin/hipblaslt-bench -m $n -n 1024 -k 7168 --transA T --transB N \
        --a_type bf16_r --b_type bf16_r --c_type bf16_r --d_type bf16_r --compute_type f32_r \
        -i 20 -j 5 --rotating 512 --algo_method all --print_kernel_info \
        > "$OUT/${mode}_${n}.txt" 2>&1
}
JOBS=()
for n in $(seq 16384 1024 32768); do
    for mode in default fixed256 dp; do
        JOBS+=("$mode $n")
    done
done

# One sequential queue per GPU, so no two runs share a GPU.
for g in "${!GPUS[@]}"; do
    (
        for ((j = g; j < ${#JOBS[@]}; j += ${#GPUS[@]})); do
            job ${JOBS[$j]} ${GPUS[$g]}
        done
    ) &
done
wait
