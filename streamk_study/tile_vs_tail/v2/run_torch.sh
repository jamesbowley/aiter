#!/bin/bash
# torch (hipBLASLt's own choice through F.linear) for a run, timed like the other methods:
# aiter's tuner with --libtype torch, then a --run_config replay chunked over GPUs, each GPU
# replaying the base picks again just before torch so the two are timed in the same session.
#   GPUS=5,6,7 bash run_torch.sh runs/<name>
set -u
V2=/home/jbowley/repos/aiter/streamk_study/tile_vs_tail/v2
RUN=$(cd "$1" && pwd)
GPUS=${GPUS:-5,6,7}
NG=$(echo $GPUS | tr , '\n' | wc -l)
source $V2/../env.sh
export PYTHONPATH=$V2/flydsl_0341:$PYTHONPATH
export TRITON_CACHE_DIR=$V2/triton_cache
cd /home/jbowley/repos/aiter
T=csrc/gemm_a16w16/gemm_a16w16_tune.py
O=$RUN/torch
mkdir -p $O
log() { echo "$* $(date -Is)" >> $O/status.log; }

log "start tune"
HIP_VISIBLE_DEVICES=$GPUS python $T -i $RUN/untuned.csv -o $O/tuned.csv -o2 $O/profile.csv \
    --libtype torch --mp $NG --shape_grouped --timeout 7200 > $O/tune.log 2>&1
log "exit=$? end tune"

python - "$RUN" "$NG" <<'EOF'
import sys
from pathlib import Path
import pandas as pd
run, n = Path(sys.argv[1]), int(sys.argv[2])
KEY = ["M", "N", "K", "bias"]
torch_rows = pd.read_csv(run / "torch" / "tuned.csv")
base_rows = pd.read_csv(run / "picks" / "base.csv")
shapes = pd.read_csv(run / "shapes.csv")
cost = shapes.tuned_us.fillna(1.0)
load, chunk = [0.0] * n, {}
for i in cost.sort_values(ascending=False).index:
    g = min(range(n), key=load.__getitem__)
    chunk[i] = g
    load[g] += cost[i]
shapes["chunk"] = pd.Series(chunk)
for g in range(n):
    d = run / "torch" / "replay" / str(g)
    d.mkdir(parents=True, exist_ok=True)
    keys = shapes[shapes.chunk == g][KEY]
    for name, rows in (("base", base_rows), ("torch", torch_rows)):
        rows.merge(keys, on=KEY).to_csv(d / f"{name}.csv", index=False)
EOF
log "start replay"
i=0
for g in $(echo $GPUS | tr , ' '); do
    (
        export HIP_VISIBLE_DEVICES=$g
        R=$O/replay/$i
        for grp in base torch; do
            python $T --run_config $R/$grp.csv --warmup 5 --iters 101 > $R/$grp.log 2>&1
        done
    ) &
    i=$((i + 1))
done
wait
log "end replay"
