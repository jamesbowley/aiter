"""Split a run's shapes into one replay chunk per GPU, balanced by expected time.

    python replay_split.py runs/<name> <num_gpus>

Writes runs/<name>/replay/<i>/shapes.csv (the shipped replay's -i) and, for every group of
picks/, replay/<i>/<group>.csv.
"""

import sys
from pathlib import Path

import pandas as pd

KEY = ["M", "N", "K", "bias"]


def main(run, n):
    run, n = Path(run), int(n)
    shapes = pd.read_csv(run / "shapes.csv")
    # Each shape is replayed about six times (shipped and five groups); balance on its
    # shipped tuned time.
    cost = shapes["tuned_us"].fillna(1.0)
    order = cost.sort_values(ascending=False).index
    load = [0.0] * n
    chunk = pd.Series(0, index=shapes.index)
    for i in order:
        g = min(range(n), key=load.__getitem__)
        chunk[i] = g
        load[g] += cost[i]
    groups = {p.stem: pd.read_csv(p) for p in sorted((run / "picks").glob("*.csv"))}
    for g in range(n):
        d = run / "replay" / str(g)
        d.mkdir(parents=True, exist_ok=True)
        part = shapes[chunk == g]
        part[KEY].assign(
            dtype="torch.bfloat16", outdtype="torch.bfloat16", scaleAB=False, bpreshuffle=False
        ).to_csv(d / "shapes.csv", index=False)
        keys = part[KEY]
        for name, rows in groups.items():
            rows.merge(keys, on=KEY).to_csv(d / f"{name}.csv", index=False)
    print(f"{len(shapes)} shapes over {n} chunks, expected load {[round(x) for x in load]} us")


if __name__ == "__main__":
    main(*sys.argv[1:])
