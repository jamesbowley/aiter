#!/bin/bash
# Main-loop MFMA issue efficiency of every tile, on one GPU: a thread trace (rocprofv3 --att,
# one CU, the kernel's second call) of each base tile, each finer tile and each Stream-K
# data-parallel plan, on each shape of RUN/shapes.csv. Each trace is parsed and deleted at
# once (RUN/att_tiles.csv); the run's Triton cache is deleted at the end.
#
#   tmux new -d -s att "./att_tiles.sh runs/att_tiles 3"
set -eu
V2=/home/jbowley/repos/aiter/streamk_study/tile_vs_tail/v2
RUN=$(realpath "$1")
GPU=$2
cd $V2
export TVT_CACHE=$V2/triton_cache_att
source $V2/../env.sh
export PYTHONPATH=$V2/flydsl_0341:$V2:$PYTHONPATH
export TRITON_CACHE_DIR=$TVT_CACHE
export HIP_VISIBLE_DEVICES=$GPU
if [ ! -s $RUN/candidates.csv ]; then
    python3 candidates.py $RUN
    python3 - $RUN <<'EOF'
import sys
import pandas as pd
c = pd.read_csv(f"{sys.argv[1]}/candidates.csv")
keep = c.method.isin(["base", "uneven"]) | c.kernelName.str.endswith("_dp")
c[keep].to_csv(f"{sys.argv[1]}/candidates.csv", index=False)
print(keep.sum(), "kernels to trace")
EOF
fi
[ -f $RUN/warm.done ] || { ./run_warm.sh $RUN $GPU && touch $RUN/warm.done; }
SDK=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel
export PATH=$SDK/bin:$PATH
export LD_LIBRARY_PATH=$SDK/lib:${LD_LIBRARY_PATH:-}
TMP=$RUN/prof_tmp
OUT=$RUN/att_tiles.csv
tail -n +2 $RUN/candidates.csv | while IFS=, read -r M N K NAME METHOD RULE; do
    tag=${M}x${N}x${K}__${NAME}
    [ -f $OUT ] && grep -q "^$tag," $OUT && continue
    rm -rf $TMP && mkdir -p $TMP
    rocprofv3 -d $TMP/att -o att --att --att-target-cu 0 --att-shader-engine-mask 0x1 \
        --att-buffer-size 0x40000000 --kernel-include-regex _gemm_a16w16 --kernel-iteration-range "[2]" \
        -- python3 $V2/mfma_run.py $M $N $K $NAME > $TMP/run.out 2>&1 || true
    python3 - "$TMP/att" "$OUT" "$tag" "$M" "$N" "$K" "$NAME" "$METHOD" <<'EOF'
import sys
from pathlib import Path
import pandas as pd
from mfma_parse import parse_att_dir
d, out, tag, M, N, K, name, method = sys.argv[1:]
rows = [dict(tag=tag, M=int(M), N=int(N), K=int(K), kernelName=name, method=method, **r) for r in parse_att_dir(d)]
if not rows:
    rows = [dict(tag=tag, M=int(M), N=int(N), K=int(K), kernelName=name, method=method, loop="none")]
pd.DataFrame(rows).to_csv(out, mode="a", header=not Path(out).exists(), index=False)
EOF
done
rm -rf $TMP
rm -rf $TVT_CACHE
echo done > $RUN/DONE
