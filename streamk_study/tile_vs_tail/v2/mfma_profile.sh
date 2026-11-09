#!/bin/bash
# MFMA efficiency of each method's winning kernel, on one GPU:
#   counters   rocprofv3 --pmc on the shapes in RUN/mfma_shapes.csv (M,N,K,kernels...)
#   att        rocprofv3 --att on the kernels of RUN/att_shape.csv, one dispatch each
# Raw profiler output goes to RUN/prof_tmp and is deleted once parsed (mfma_parse.py).
#
#   ./mfma_profile.sh runs/grid 3 counters|att
set -eu
V2=/home/jbowley/repos/aiter/streamk_study/tile_vs_tail/v2
RUN=$(realpath "$1")
GPU=$2
MODE=$3
source $V2/../env.sh
export PYTHONPATH=$V2/flydsl_0341:$V2:$PYTHONPATH
export TRITON_CACHE_DIR=${TVT_CACHE:-$V2/triton_cache_grid}
export HIP_VISIBLE_DEVICES=$GPU
SDK=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel
export PATH=$SDK/bin:$PATH
export LD_LIBRARY_PATH=$SDK/lib:${LD_LIBRARY_PATH:-}
TMP=$RUN/prof_tmp
mkdir -p $TMP
if [ "$MODE" = counters ]; then
    tail -n +2 $RUN/mfma_shapes.csv | while IFS=, read -r M N K KS; do
        for NAME in ${KS//;/ }; do
            tag=${M}x${N}x${K}__${NAME}
            [ -d $TMP/$tag ] && continue
            rocprofv3 -d $TMP/$tag -o pmc --output-format csv \
                --pmc "SQ_VALU_MFMA_BUSY_CYCLES GRBM_GUI_ACTIVE SQ_BUSY_CU_CYCLES SQ_WAVE_CYCLES" \
                --pmc "SQ_WAIT_INST_LDS SQ_LDS_BANK_CONFLICT SQ_INSTS_MFMA SQ_INSTS_LDS" \
                -- python3 $V2/mfma_run.py $M $N $K $NAME > $TMP/$tag.out 2>&1
        done
    done
    python3 $V2/mfma_parse.py counters $RUN
else
    tail -n +2 $RUN/att_shape.csv | while IFS=, read -r M N K NAME; do
        tag=att_$(echo $NAME | tr -c 'a-zA-Z0-9_\n' '_')
        rocprofv3 -d $TMP/$tag -o att --att --att-target-cu 0 --att-shader-engine-mask 0x1 --att-buffer-size 0x40000000 \
            --kernel-include-regex _gemm_a16w16 --kernel-iteration-range "[2]" \
            -- python3 $V2/mfma_run.py $M $N $K $NAME > $TMP/$tag.out 2>&1
    done
    python3 $V2/mfma_parse.py att $RUN
fi
rm -rf $TMP
