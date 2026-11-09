#!/bin/bash
# Main-loop MFMA issue efficiency of the base's power-of-two tiles the way the v9 tutorial's
# v9_beyond_hotloop_tiles measures it: M = 16 x BM, N = 16 x BN (256 workgroups, one per CU),
# K = 8192, cold inputs (rotating copies over at least 2 GB; the traced launch's inputs were
# last touched SETS launches earlier). One GPU; traces parsed and deleted at once.
#
#   tmux new -d -s attcold "./att_cold.sh runs/att_cold 3"
set -eu
V2=/home/jbowley/repos/aiter/streamk_study/tile_vs_tail/v2
RUN=$(realpath -m "$1")
GPU=$2
mkdir -p $RUN
cd $V2
source $V2/../env.sh
export PYTHONPATH=$V2/flydsl_0341:$V2:$PYTHONPATH
export TRITON_CACHE_DIR=$RUN/triton_cache
export HIP_VISIBLE_DEVICES=$GPU
SDK=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel
export PATH=$SDK/bin:$PATH
export LD_LIBRARY_PATH=$SDK/lib:${LD_LIBRARY_PATH:-}
K=8192
OUT=$RUN/att_cold.csv
for TILE in 256x256 256x128 128x256 128x128 256x64 64x256 128x64 64x128 64x64; do
    BM=${TILE%x*}; BN=${TILE#*x}
    M=$((16 * BM)); N=$((16 * BN))
    # copies of A and B (bf16) to cover 2 GB, at least 12
    SETS=$(python3 -c "import math; print(max(12, math.ceil(2 * 2**30 / (2 * ($M + $N) * $K))))")
    for rep in 1 2 3; do
        tag=${TILE}_r${rep}
        [ -f $OUT ] && grep -q "^$tag," $OUT && continue
        TMP=$RUN/prof_tmp && rm -rf $TMP && mkdir -p $TMP
        # launch 1 compiles; launches 2.. rotate over the copies; trace the last copy's first use
        MFMA_SETS=$SETS MFMA_REPS=$((SETS + 2)) rocprofv3 -d $TMP/att -o att --att --att-target-cu 0 \
            --att-shader-engine-mask 0x1 --att-buffer-size 0x40000000 --kernel-include-regex _gemm_a16w16 \
            --kernel-iteration-range "[$((SETS + 1))]" \
            -- python3 $V2/mfma_run.py $M $N $K gluon_compute_bound_$TILE > $TMP/run.out 2>&1 || true
        python3 - "$TMP/att" "$OUT" "$tag" "$TILE" "$M" "$N" "$K" "$SETS" <<'EOF'
import sys
from pathlib import Path
import pandas as pd
from mfma_parse import parse_att_dir
d, out, tag, tile, M, N, K, sets = sys.argv[1:]
rows = [dict(tag=tag, tile=tile, M=int(M), N=int(N), K=int(K), sets=int(sets), **r) for r in parse_att_dir(d) if r["loop"] == "main"]
rows = rows or [dict(tag=tag, tile=tile, M=int(M), N=int(N), K=int(K), sets=int(sets), loop="none")]
pd.DataFrame(rows).to_csv(out, mode="a", header=not Path(out).exists(), index=False)
EOF
    done
done
rm -rf $RUN/prof_tmp $RUN/triton_cache
echo done > $RUN/DONE
