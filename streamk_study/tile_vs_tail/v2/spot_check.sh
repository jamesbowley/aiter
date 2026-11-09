#!/bin/bash
# Re-replay a sample of the main run's picks through aiter's tuner (--run_config, the same
# flags as run.sh's replay step) on one GPU, then compare with the main run's replay times:
# 30 finer-tile picks (gemm_a16w16_uneven.py changed a minute after the main replay) and 15
# base picks (is the machine still timing kernels as it did then).
#
#   tmux new -d -s spot "./spot_check.sh 3"
set -eu
V2=/home/jbowley/repos/aiter/streamk_study/tile_vs_tail/v2
GPU=$1
OUT=$V2/runs/spotcheck
mkdir -p $OUT
source $V2/../env.sh
export PYTHONPATH=$V2/flydsl_0341:$PYTHONPATH
export TRITON_CACHE_DIR=$V2/triton_cache
export HIP_VISIBLE_DEVICES=$GPU
python3 - <<'EOF'
import pandas as pd
from pathlib import Path
v2 = Path("/home/jbowley/repos/aiter/streamk_study/tile_vs_tail/v2")
res = pd.read_csv(v2 / "runs/full/results.csv")
main = res[res.set == "main"][["M", "N", "K", "bias"]]
for grp, n in (("uneven", 30), ("base", 15)):
    rows = pd.concat(pd.read_csv(f) for f in sorted((v2 / "runs/full/replay").glob(f"*/{grp}.csv")))
    rows = rows.merge(main, on=["M", "N", "K", "bias"]).sort_values(["K", "M", "N"])
    step = max(1, len(rows) // n)
    rows.iloc[::step].head(n).to_csv(v2 / f"runs/spotcheck/{grp}.csv", index=False)
EOF
cd /home/jbowley/repos/aiter
for grp in uneven base; do
    python csrc/gemm_a16w16/gemm_a16w16_tune.py --run_config $OUT/$grp.csv --warmup 5 --iters 101 \
        > $OUT/$grp.log 2>&1
done
cd $V2
python3 - > $OUT/compare.txt <<'EOF'
import numpy as np, pandas as pd
from pathlib import Path
from issue_figs import parse_log, KEY
run = Path("runs/full")
for grp in ("uneven", "base"):
    old = pd.concat(parse_log(p, "old") for p in sorted((run / "replay").glob(f"*/{grp}.log"))).drop_duplicates(KEY)
    new = parse_log(Path(f"runs/spotcheck/{grp}.log"), "new")
    j = new.merge(old, on=KEY).dropna()
    r = j.new / j.old
    print(f"{grp}: {len(j)} shapes, new/old median {r.median():.4f}, geomean {np.exp(np.log(r).mean()):.4f}, "
          f"5-95% {r.quantile(0.05):.3f}-{r.quantile(0.95):.3f}, worst {r.min():.3f}/{r.max():.3f}")
    j.assign(ratio=r.round(4)).to_csv(f"runs/spotcheck/{grp}_compare.csv", index=False)
EOF
echo done > $OUT/DONE
