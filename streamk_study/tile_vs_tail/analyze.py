"""Report of one run: uneven data-parallel tiles against sub-tile tails, against the
shipped kernels and against power-of-two Stream-K.

Times are the --run_config replays (replay/<i>/<group>.log, shipped.log): every method's
pick is chosen in the tuning run and timed here. Speedups are t_other / t_ours, so above 1
means ours is faster; a shape is faster or slower beyond +-3% (--compare's rule), parity
otherwise.

    python analyze.py runs/<name> [--peak-tbps 5.484] [--mem-bound 0.8] [--mixed 0.5]

Writes <run>/results.csv (one row per shape) and <run>/RESULTS.md.
"""

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import candidates as cand  # noqa: E402

KEY = ["M", "N", "K", "bias"]
GROUPS = ("pow2_split", "pow2_dp", "uneven", "subtile", "prev")
ROW = re.compile(r"^\((\d+), (\d+), (\d+), \S+, bias=(True|False)\)\s*\|\s*(\S+)\s*\|\s*(.+?)\s*$")
THRESHOLD = 0.03
P = cand.P


def parse_log(path, label):
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            m = ROW.match(line)
            if m:
                M, N, K, bias, us, status = m.groups()
                try:
                    us = float(us)
                except ValueError:
                    us = math.nan
                if status.strip() != "OK" or us <= 0:
                    us = math.nan
                rows.append((int(M), int(N), int(K), bias == "True", us))
    return pd.DataFrame(rows, columns=KEY + [label])


def load_replay(run):
    out = None
    for label in ("shipped",) + GROUPS:
        parts = [parse_log(d / f"{label}.log", label) for d in sorted((run / "replay").iterdir())]
        df = pd.concat(parts, ignore_index=True).drop_duplicates(KEY)
        out = df if out is None else out.merge(df, on=KEY, how="outer")
    return out


def fmt_stats(speedup):
    s = speedup.dropna()
    if len(s) == 0:
        return dict(shapes=0, faster="", parity="", slower="", geomean="", median="")
    n = len(s)
    f = int((s >= 1 + THRESHOLD).sum())
    sl = int((s <= 1 - THRESHOLD).sum())
    return dict(
        shapes=n,
        faster=f"{f} ({100 * f / n:.0f}%)",
        parity=f"{n - f - sl} ({100 * (n - f - sl) / n:.0f}%)",
        slower=f"{sl} ({100 * sl / n:.0f}%)",
        geomean=f"{math.exp(np.log(s).mean()):.3f}",
        median=f"{100 * (s.median() - 1):+.1f}%",
    )


def table(df, speedup_col, by=None, order=None):
    rows = []
    if by is None:
        rows.append(dict(group="all", **fmt_stats(df[speedup_col])))
    else:
        keys = order if order is not None else sorted(df[by].dropna().unique())
        for k in keys:
            rows.append(dict(group=k, **fmt_stats(df.loc[df[by] == k, speedup_col])))
        rows.append(dict(group="all", **fmt_stats(df[speedup_col])))
    return pd.DataFrame(rows)


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and math.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(lines)


def k_bin(K):
    for hi in (1024, 2048, 4096, 8192):
        if K <= hi:
            return f"K<={hi}"
    return "K>8192"


K_ORDER = ["K<=1024", "K<=2048", "K<=4096", "K<=8192", "K>8192"]
SP_ORDER = ["S/P<=1/4", "S/P<=1/2", "S/P<=3/4", "S/P<1"]
REGIMES = ["below_one_wave", "one_wave_tail", "two_plus_waves_tail", "no_tail"]


def sp_bin(S):
    f = S / P
    if S == 0:
        return "none"
    return "S/P<=1/4" if f <= 0.25 else "S/P<=1/2" if f <= 0.5 else "S/P<=3/4" if f <= 0.75 else "S/P<1"


def ai_bin(ai):
    for hi in (64, 128, 256, 512):
        if ai < hi:
            return f"AI<{hi}"
    return "AI>=512"


def tile_of(name):
    p = cand.parse_name(str(name)) if isinstance(name, str) else None
    return (p[1], p[2]) if p else None


def sub_wave_estimate(shapes, run):
    """Shapes below one wave at their previous tile where an uneven tile gives a near-exact
    half or quarter wave (0.9 < T / (P / n) <= 1), with the time of that tile split n ways
    estimated from the uneven tile's data-parallel time (measured in this run's tuning
    profile, else modelled) and the owner-n split efficiency measured at the previous tile
    in the stock profile: t = t_dp(uneven) / (n eta_n), eta_n = t_dp(prev) / (n t_split(prev, n))."""
    sk, cb = cand._aiter_modules()
    if (run / "calibration.json").exists():
        import json

        cand.CAL.update(json.loads((run / "calibration.json").read_text()))
    stock = pd.read_csv(cand.STOCK / "profile.csv")
    stock_by = {k: g for k, g in stock.groupby(KEY)}
    prof = pd.read_csv(run / "profile.csv")
    prof_by = {k: g for k, g in prof.groupby(KEY)}
    rows = []
    for r in shapes[shapes.regime == "below_one_wave"].itertuples():
        key = (r.M, r.N, r.K, r.bias)
        if key not in stock_by:
            continue
        model = cand.ShapeModel(r.M, r.N, r.K, stock_by[key])
        prev = cand.parse_name(r.prev_kernel)
        if prev is None:
            continue
        bm0, bn0, bk0 = prev[1], prev[2], prev[3]
        t_dp_prev = model.dp.get((bm0, bn0, bk0))
        measured = {}
        if key in prof_by:
            for name, us in zip(prof_by[key].kernelName, prof_by[key].us):
                t = tile_of(name)
                if t and us > 0 and name.startswith("gluon_compute_bound_"):
                    measured[t] = us
        best = None
        for n in (2, 4):
            split = model.split.get((bm0, bn0, bk0, "owner", n))
            if not t_dp_prev or not split:
                continue
            eta = t_dp_prev / (n * split)
            for bm in cand.SIDES_UNEVEN:
                for bn in cand.SIDES_UNEVEN:
                    if bm * bn > cand.MAX_ACC or not cand.valid_uneven(cb, bm, bn):
                        continue
                    T = cand.tiles(r.M, r.N, bm, bn)
                    fill = T / (P / n)
                    if not (0.9 < fill <= 1):
                        continue
                    t_dp = measured.get((bm, bn), model.t_uneven(bm, bn))
                    est = t_dp / (n * eta)
                    if best is None or est < best[0]:
                        best = (est, f"{bm}x{bn}", n, fill, eta, (bm, bn) in measured)
        if best:
            rows.append(dict(M=r.M, N=r.N, K=r.K, bias=r.bias, est_us=best[0], est_tile=best[1],
                             est_n=best[2], est_fill=best[3], est_eta=best[4], est_tile_measured=best[5]))
    return pd.DataFrame(rows, columns=KEY + ["est_us", "est_tile", "est_n", "est_fill", "est_eta", "est_tile_measured"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--peak-tbps", type=float, default=None)
    ap.add_argument("--mem-bound", type=float, default=0.8)
    ap.add_argument("--mixed", type=float, default=0.5)
    args = ap.parse_args()
    run = Path(args.run)
    peak = args.peak_tbps
    if peak is None:
        txt = (run.parent / "bw_peak.txt").read_text()
        peak = float(re.search(r"peak_TBps ([\d.]+)", txt)[1])

    shapes = pd.read_csv(run / "shapes.csv")
    rep = load_replay(run)
    df = shapes.merge(rep, on=KEY, how="left")
    picks = pd.read_csv(run / "picks.csv")
    for g in GROUPS:
        names = picks[picks.group == g][KEY + ["kernelName", "tune_us"]].rename(
            columns={"kernelName": f"{g}_pick", "tune_us": f"{g}_tune_us"}
        )
        df = df.merge(names, on=KEY, how="left")

    # Composite methods (each from replayed picks).
    df["pow2_best"] = df[["pow2_split", "pow2_dp", "prev"]].min(axis=1)
    df["dp_any"] = df[["uneven", "pow2_dp"]].min(axis=1)
    df["v32_any"] = df[["subtile", "pow2_best"]].min(axis=1)
    df["ours_new"] = df[["pow2_best", "uneven", "subtile"]].min(axis=1)

    # Labels.
    df["k_bin6"] = df.K.map(k_bin)
    df["sp_bin"] = df.S.map(sp_bin)
    bytes_min = 2 * (df.M * df.K + df.N * df.K + df.M * df.N)
    best_t = df[["shipped", "ours_new"]].min(axis=1)
    df["bw_frac"] = bytes_min / (best_t * 1e-6) / (peak * 1e12)
    df["bound"] = np.where(df.bw_frac >= args.mem_bound, "memory", np.where(df.bw_frac >= args.mixed, "mixed", "compute"))
    df["ai"] = 2 * df.M * df.N * df.K / bytes_min
    df["ai_bin"] = df.ai.map(ai_bin)
    df["short"] = df.shipped_us < 20
    df["shipped_class"] = np.where(df.shipped_pow2, "pow2", np.where(df.shipped_exact, "uneven_exact", "uneven_approx"))
    uneven_tile = df.uneven_pick.map(tile_of)
    df["uneven_waves"] = [
        (cand.tiles(m, n, *t) / P) if t else math.nan for m, n, t in zip(df.M, df.N, uneven_tile)
    ]
    sub = df.subtile_pick.map(lambda n: cand.parse_name(n) if isinstance(n, str) else None)
    df["subtile_main"] = sub.map(lambda p: f"{p[1]}x{p[2]}x{p[3]}" if p else None)
    prev = df.prev_kernel.map(lambda n: cand.parse_name(n) if isinstance(n, str) else None)
    df["prev_main"] = prev.map(lambda p: f"{p[1]}x{p[2]}x{p[3]}" if p else None)

    # Speedups (t_other / t_ours).
    df["uneven_over_subtile"] = df.subtile / df.uneven
    df["dpany_over_v32any"] = df.v32_any / df.dp_any
    df["new_over_shipped"] = df.shipped / df.ours_new
    df["prev_over_shipped"] = df.shipped / df.pow2_best
    df["dpany_over_split"] = df.pow2_split / df.dp_any
    df["dpany_over_pow2best"] = df.pow2_best / df.dp_any
    df["new_over_pow2best"] = df.pow2_best / df.ours_new

    est = sub_wave_estimate(shapes, run)
    df = df.merge(est, on=KEY, how="left")
    df["est_over_new"] = df.ours_new / df.est_us
    df.to_csv(run / "results.csv", index=False)

    n_missing = df[list(GROUPS) + ["shipped"]].isna().sum()
    out = [f"# Uneven tiles against sub-tile tails: {run.name}", ""]
    out.append(
        f"{len(df)} shapes. Times from aiter's --run_config replay of each method's pick "
        f"(picked in the tuning run). Speedup = t_other / t_ours; faster / slower beyond "
        f"+-{100 * THRESHOLD:.0f}%. HBM peak {peak:.2f} TB/s (bw_peak.txt); memory-bound "
        f"when the minimum traffic 2(MK + NK + MN) at the best time reaches "
        f"{args.mem_bound:.0%} of it, mixed from {args.mixed:.0%}."
    )
    out.append("")
    out.append("Missing replays per column: " + ", ".join(f"{k} {v}" for k, v in n_missing.items() if v))
    out.append("")
    out.append("Methods: `uneven` = best non-power-of-two data-parallel tile; `subtile` = best "
               "(main tile, sub-tile tail); `pow2_split` = best power-of-two Stream-K split; "
               "`pow2_dp` = best power-of-two data-parallel; `pow2_best` = the best of those and our "
               "previous pick (the PR's kernels); `dp_any` = min(uneven, pow2_dp); `v32_any` = "
               "min(subtile, pow2_best); `ours_new` = the best of everything.")
    out.append("")

    out.append("## Shapes by regime (at our previous tile) and K")
    out.append("")
    out.append(md(df.groupby(["regime", "k_bin6"]).size().unstack(fill_value=0).reindex(REGIMES).fillna(0).astype(int).reset_index()))
    out.append("")

    out.append("## Q1. Uneven data-parallel against sub-tile tails")
    out.append("")
    both = df.dropna(subset=["uneven", "subtile"])
    out.append(f"Pure methods, on the {len(both)} shapes where both have a candidate (speedup = t_subtile / t_uneven, above 1 the uneven tile wins):")
    out.append("")
    out.append(md(table(both, "uneven_over_subtile", "regime", [r for r in REGIMES if r in set(both.regime)])))
    out.append("")
    out.append(md(table(both, "uneven_over_subtile", "k_bin6", [k for k in K_ORDER if k in set(both.k_bin6)])))
    out.append("")
    after = both[both.regime.isin(["one_wave_tail", "two_plus_waves_tail"])]
    if len(after):
        out.append("After at least one full wave, by tail fraction S / P at our previous tile:")
        out.append("")
        out.append(md(table(after, "uneven_over_subtile", "sp_bin", [s for s in SP_ORDER if s in set(after.sp_bin)])))
        out.append("")
    out.append("With fallbacks on every shape (speedup = t_v32_any / t_dp_any): each method may also use "
               "its power-of-two kernels.")
    out.append("")
    out.append(md(table(df, "dpany_over_v32any", "regime", [r for r in REGIMES if r in set(df.regime)])))
    out.append("")
    js = df.dropna(subset=["subtile_main"])
    changed = (js.subtile_main != js.prev_main).sum()
    out.append(f"Joint search of the sub-tile tail's main tile: the best sub-tile pick's main tile differs "
               f"from our previous pick's tile on {changed} of {len(js)} shapes.")
    out.append("")

    out.append("## Q2. Against the shipped kernels")
    out.append("")
    np2 = df[~df.shipped_pow2]
    filters = [
        ("all non-power-of-two shipped", np2),
        ("without memory-bound", np2[np2.bound != "memory"]),
        ("without short (< 20 us)", np2[~np2.short]),
        ("without below one wave", np2[np2.regime != "below_one_wave"]),
        ("without all three", np2[(np2.bound != "memory") & ~np2.short & (np2.regime != "below_one_wave")]),
        ("shipped tile we run exactly", np2[np2.shipped_exact]),
        ("shipped tile we only approximate", np2[~np2.shipped_exact]),
    ]
    rows = []
    for name, sub_df in filters:
        a = fmt_stats(sub_df.new_over_shipped)
        b = fmt_stats(sub_df.prev_over_shipped)
        rows.append(dict(subset=name, shapes=a["shapes"], faster=a["faster"], parity=a["parity"],
                         slower=a["slower"], geomean=a["geomean"], median=a["median"],
                         geomean_before=b["geomean"], slower_before=b["slower"]))
    out.append("`ours_new` against the shipped kernel on shapes whose shipped tile is not a power of two "
               "(`_before` columns: `pow2_best`, the PR's kernels, re-timed in the same session):")
    out.append("")
    out.append(md(pd.DataFrame(rows)))
    out.append("")
    rows = []
    for name, sub_df in [("all", df), ("power-of-two shipped", df[df.shipped_pow2]), ("non-power-of-two shipped", np2)]:
        a = fmt_stats(sub_df.new_over_shipped)
        b = fmt_stats(sub_df.prev_over_shipped)
        rows.append(dict(subset=name, shapes=a["shapes"], faster=a["faster"], slower=a["slower"],
                         geomean=a["geomean"], geomean_before=b["geomean"]))
    out.append(md(pd.DataFrame(rows)))
    out.append("")
    out.append("By how memory-bound the shape is (non-power-of-two shipped):")
    out.append("")
    out.append(md(table(np2, "new_over_shipped", "bound", ["compute", "mixed", "memory"])))
    out.append("")
    out.append(md(table(np2, "new_over_shipped", "ai_bin", [b for b in ["AI<64", "AI<128", "AI<256", "AI<512", "AI>=512"] if b in set(np2.ai_bin)])))
    out.append("")

    out.append("## Q3. Is power-of-two Stream-K needed when a tile can fill the waves?")
    out.append("")
    out.append("Speedup = t_pow2_split / t_dp_any: above 1 a data-parallel tile (any tile, uneven included) "
               "beats the best power-of-two Stream-K split.")
    out.append("")
    out.append(md(table(df.dropna(subset=["pow2_split"]), "dpany_over_split", "regime", [r for r in REGIMES if r in set(df.regime)])))
    out.append("")
    after = df[df.regime.isin(["one_wave_tail", "two_plus_waves_tail"])].dropna(subset=["pow2_split", "dp_any"])
    if len(after):
        cells = []
        for sp in SP_ORDER:
            for kb in K_ORDER:
                c = after[(after.sp_bin == sp) & (after.k_bin6 == kb)]
                if len(c) == 0:
                    continue
                s = c.dpany_over_split
                gm = math.exp(np.log(s).mean())
                cells.append(dict(sp=sp, k=kb, shapes=len(c), geomean=f"{gm:.3f}",
                                  split_wins=int((s <= 1 - THRESHOLD).sum()),
                                  dp_wins=int((s >= 1 + THRESHOLD).sum()),
                                  verdict="Stream-K needed" if gm < 1 - THRESHOLD / 3 and (s <= 1 - THRESHOLD).sum() > (s >= 1 + THRESHOLD).sum() else ""))
        out.append("After at least one full wave, by tail fraction (at our previous tile) and K:")
        out.append("")
        out.append(md(pd.DataFrame(cells)))
        out.append("")
    out.append("`ours_new` over `pow2_best` (what the two methods add to the PR's kernels):")
    out.append("")
    out.append(md(table(df, "new_over_pow2best", "regime", [r for r in REGIMES if r in set(df.regime)])))
    out.append("")
    win = df.loc[df[["pow2_best", "uneven", "subtile"]].notna().any(axis=1), ["pow2_best", "uneven", "subtile"]].idxmin(axis=1)
    out.append("Winner of `ours_new` per regime:")
    out.append("")
    out.append(md(pd.crosstab(df.loc[win.index, "regime"], win).reset_index()))
    out.append("")

    out.append("## Below one wave: fill a fraction of the wave, then split (estimate)")
    out.append("")
    e = df.dropna(subset=["est_us"])
    if len(e):
        s = e.est_over_new
        out.append(f"{len(e)} shapes below one wave at our previous tile have an uneven tile giving 90-100% of "
                   f"a half or quarter wave. Estimated time of that tile split 2 or 4 ways (owner ending, "
                   f"efficiency measured at the previous tile) against the measured `ours_new`: geomean "
                   f"{math.exp(np.log(s).mean()):.3f}, estimated faster by >3% on {(s >= 1.03).sum()}, "
                   f"slower on {(s <= 0.97).sum()}. Needs an uneven Stream-K kernel to measure.")
        out.append("")
        out.append(md(e.sort_values("est_over_new", ascending=False).head(15)[["M", "N", "K", "est_tile", "est_n", "est_fill", "est_eta", "est_us", "ours_new", "est_over_new"]].round(3)))
    else:
        out.append("No shapes qualify.")
    out.append("")
    (run / "RESULTS.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    main()
