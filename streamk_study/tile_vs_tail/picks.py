"""Each method's best candidate per shape from a run's tuning profile, as tuned-format CSVs
for the --run_config replay (picked in the tuning run, timed in the replay, so a method
with more candidates gains nothing from the minimum over noise).

Groups, by kernelName:
    pow2_split  power-of-two Stream-K that splits (one_tile, spread, two_tile)
    pow2_dp     power-of-two data-parallel (compute-bound kernel or Stream-K dp)
    uneven      non-power-of-two data-parallel (compute-bound, uneven kernel)
    subtile     a sub-tile tail (_tail<TM>x<TN>)
    prev        our previous pick (the stock retune's tuned row), whatever it is

    python picks.py runs/<name>     # writes runs/<name>/picks/<group>.csv and picks.csv
"""

import re
import sys
from pathlib import Path

import pandas as pd

KEY = ["M", "N", "K", "bias"]
GROUPS = ("pow2_split", "pow2_dp", "uneven", "subtile", "prev")
POW2 = {64, 128, 256}


def group_of(name):
    m = re.match(r"^gluon_compute_bound_(\d+)x(\d+)$", name)
    if m:
        return "pow2_dp" if {int(m[1]), int(m[2])} <= POW2 else "uneven"
    if "_tail" in name:
        return "subtile"
    if name.endswith("_dp"):
        return "pow2_dp"
    return "pow2_split"


def main(run):
    run = Path(run)
    prof = pd.read_csv(run / "profile.csv")
    shapes = pd.read_csv(run / "shapes.csv")
    prof = prof[(prof.us > 0) & (prof.err_ratio <= 0.05)].copy()
    prof["group"] = prof.kernelName.map(group_of)
    prev = shapes[KEY + ["prev_kernel"]].rename(columns={"prev_kernel": "kernelName"})
    prev_rows = prof.merge(prev, on=KEY + ["kernelName"]).assign(group="prev")
    allrows = pd.concat([prof, prev_rows], ignore_index=True)
    best = allrows.sort_values("us").groupby(KEY + ["group"], as_index=False).first()
    (run / "picks").mkdir(exist_ok=True)
    cols = [c for c in prof.columns if c != "group"]
    for g in GROUPS:
        best[best.group == g][cols].to_csv(run / "picks" / f"{g}.csv", index=False)
    best[KEY + ["group", "kernelName", "us"]].rename(columns={"us": "tune_us"}).to_csv(
        run / "picks.csv", index=False
    )
    failed = pd.read_csv(run / "profile.csv")
    failed = failed[(failed.us <= 0) | (failed.err_ratio > 0.05)]
    failed.to_csv(run / "tune_failed.csv", index=False)
    print(best.groupby("group").size().to_string())
    print(f"{len(failed)} failed candidates (tune_failed.csv)")


if __name__ == "__main__":
    main(sys.argv[1])
