"""Each method's best candidate per shape from a run's tuning profile, as tuned-format
CSVs for the --run_config replay: picked in the tuning run, timed in the replay, so a
method with more candidates gains nothing from the minimum over noise.

Groups (candidates.csv's method, with Stream-K split in two):
    base          gemm_a16w16.py, power-of-two tiles
    uneven        gemm_a16w16_uneven.py, tiles with a non-power-of-two side
    subtile       gemm_a16w16_subtile.py, any main tile and sub-tile
    subtile_rule  the sub-tile gfx1250's rule picks (best main tile)
    streamk       Stream-K plans that split (one_tile, two_tile, spread)
    streamk_dp    Stream-K's data-parallel plan (dp): the Stream-K kernel without Stream-K

    python picks.py runs/<name>     # writes runs/<name>/picks/<group>.csv and picks.csv
"""

import sys
from pathlib import Path

import pandas as pd

KEY = ["M", "N", "K", "bias"]
GROUPS = ("base", "uneven", "subtile", "subtile_rule", "streamk", "streamk_dp")


def main(run):
    run = Path(run)
    prof = pd.read_csv(run / "profile.csv")
    cands = pd.read_csv(run / "candidates.csv")
    ok = prof[(prof.us > 0) & (prof.err_ratio <= 0.05)].copy()
    ok = ok.merge(cands, on=["M", "N", "K", "kernelName"], how="left")
    ok["group"] = ok.method
    ok.loc[(ok.method == "streamk") & ok.kernelName.str.endswith("_dp"), "group"] = "streamk_dp"
    rule = ok[ok.gfx1250_rule.fillna(False).astype(bool)].assign(group="subtile_rule")
    rows = pd.concat([ok, rule], ignore_index=True)
    best = rows.sort_values("us").groupby(KEY + ["group"], as_index=False).first()
    (run / "picks").mkdir(exist_ok=True)
    cols = list(prof.columns)
    for g in GROUPS:
        best[best.group == g][cols].to_csv(run / "picks" / f"{g}.csv", index=False)
    best[KEY + ["group", "kernelName", "us"]].rename(columns={"us": "tune_us"}).to_csv(
        run / "picks.csv", index=False
    )
    failed = prof[(prof.us <= 0) | (prof.err_ratio > 0.05)]
    failed.to_csv(run / "tune_failed.csv", index=False)
    print(best.groupby("group").size().to_string())
    print(f"{len(failed)} failed candidates (tune_failed.csv)")


if __name__ == "__main__":
    main(sys.argv[1])
