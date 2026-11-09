#!/bin/bash
# Calibrate grid_bench.py against aiter's tuner on 60 shapes of the main run, in one session
# on one GPU: the harness times every candidate, then aiter's --run_config replays each
# method's pick from the main run (the same flags as run.sh's replay step).
#
#   tmux new -d -s calib "./calib.sh 3"
set -eu
V2=/home/jbowley/repos/aiter/streamk_study/tile_vs_tail/v2
GPU=$1
OUT=$V2/runs/calib
mkdir -p $OUT
cd $V2
python3 - <<'EOF'
import pandas as pd
from pathlib import Path
v2 = Path("/home/jbowley/repos/aiter/streamk_study/tile_vs_tail/v2")
out = v2 / "runs/calib"
KEY = ["M", "N", "K", "bias"]
res = pd.read_csv(v2 / "runs/full/results.csv")
main = res[res.set == "main"]
# 3 shapes per (last-wave fill quarter, K group) cell, spread over each cell's sorted shapes
pick = []
for _, g in main.sort_values(["M", "N", "K"]).groupby(["fill_bin", "k_bin"]):
    idx = [round(i * (len(g) - 1) / 2) for i in range(3)] if len(g) >= 3 else range(len(g))
    pick.append(g.iloc[sorted(set(idx))])
pick = pd.concat(pick)
if len(pick) < 60:
    rest = main.drop(pick.index).sample(60 - len(pick), random_state=0)
    pick = pd.concat([pick, rest])
pick[KEY].to_csv(out / "shapes.csv", index=False)
for grp in ("base", "uneven", "subtile", "streamk", "streamk_dp"):
    rows = pd.concat(pd.read_csv(f) for f in sorted((v2 / "runs/full/replay").glob(f"*/{grp}.csv")))
    rows.merge(pick[KEY], on=KEY).drop_duplicates(KEY).to_csv(out / f"replay_{grp}.csv", index=False)
print(len(pick), "shapes")
EOF
./grid_bench.sh $OUT/shapes.csv $OUT/bench.csv $GPU
source $V2/../env.sh
export PYTHONPATH=$V2/flydsl_0341:$PYTHONPATH
export TRITON_CACHE_DIR=$V2/triton_cache
export HIP_VISIBLE_DEVICES=$GPU
cd /home/jbowley/repos/aiter
for grp in base uneven subtile streamk streamk_dp; do
    [ -s $OUT/replay_$grp.csv ] && python csrc/gemm_a16w16/gemm_a16w16_tune.py --run_config $OUT/replay_$grp.csv \
        --warmup 5 --iters 101 > $OUT/replay_$grp.log 2>&1
done
echo done > $OUT/DONE
