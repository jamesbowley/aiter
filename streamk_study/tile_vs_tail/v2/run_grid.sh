#!/bin/bash
# The grid of the dispatch-rule study on one GPU: compile every candidate (warm.py) into a
# separate Triton cache, time every candidate (pass A), then re-time each method's three
# fastest candidates (pass B, a different shape order) to measure the noise. Shapes run in
# a seeded random order, so drift over the hours of the run cannot look like a K or W effect.
#
#   tmux new -d -s grid "./run_grid.sh runs/grid 3"
set -eu
V2=/home/jbowley/repos/aiter/streamk_study/tile_vs_tail/v2
RUN=$(realpath "$1")
GPU=$2
export TVT_CACHE=$V2/triton_cache_grid
cd $V2
[ -s $RUN/candidates.csv ] || python3 candidates.py $RUN
[ -f $RUN/warm.done ] || { ./run_warm.sh $RUN $GPU && touch $RUN/warm.done; }
python3 - "$RUN" <<'EOF'
import sys
import pandas as pd
run = sys.argv[1]
s = pd.read_csv(f"{run}/shapes.csv")
s.sample(frac=1, random_state=1).to_csv(f"{run}/order_a.csv", index=False)
s.sample(frac=1, random_state=2).to_csv(f"{run}/order_b.csv", index=False)
EOF
./grid_bench.sh $RUN/order_a.csv $RUN/bench_a.csv $GPU
python3 - "$RUN" <<'EOF'
import sys
sys.path.insert(0, ".")
from analyze_grid import load_bench, KEY
run = sys.argv[1]
d = load_bench(f"{run}/bench_a.csv")
top = d.sort_values("us").groupby(KEY + ["method"]).head(3)
top[KEY + ["kernelName"]].to_csv(f"{run}/pass_b.csv", index=False)
print(len(top), "candidates for pass B")
EOF
./grid_bench.sh $RUN/order_b.csv $RUN/bench_b.csv $GPU --only $RUN/pass_b.csv
echo done > $RUN/DONE
