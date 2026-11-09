"""Report of one v2 run: uneven tiles, sub-tile tails and Stream-K against gemm_a16w16.py.

Times are the --run_config replays (replay/<i>/<group>.log, shipped.log): each method's
pick is chosen in the tuning run and timed here. A speedup is t_base / t_method (above 1
the method is faster than gemm_a16w16.py at its best power-of-two tile); faster or slower
means beyond +-3%, parity otherwise.

Methods, alone and with the base as a fallback (min with base; "+base"):
    uneven      gemm_a16w16_uneven.py, a tile with a non-power-of-two side
    subtile     gemm_a16w16_subtile.py, a power-of-two tile with a sub-tile tail
    streamk     gemm_a16w16_streamk.py at its best plan (splitting or dp)

    python analyze.py runs/<name> [--peak-tbps 5.484]

Writes <run>/results.csv (one row per shape) and <run>/RESULTS.md.
"""

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

KEY = ["M", "N", "K", "bias"]
GROUPS = ("base", "uneven", "subtile", "subtile_rule", "streamk", "streamk_dp")
ROW = re.compile(
    r"^\((\d+), (\d+), (\d+), \S+, bias=(True|False)\)\s*\|\s*(\S+)\s*\|\s*(.+?)\s*$"
)
THRESHOLD = 0.03
P = 256
K_ORDER = ["K<=1024", "K<=2048", "K<=4096", "K<=8192", "K>8192"]
FILL_ORDER = ["S/P<=1/4", "S/P<=1/2", "S/P<=3/4", "S/P<1"]
SETS = ["main", "full_waves", "below_one_wave"]
TILE = re.compile(r"_(\d+)x(\d+)(?:x\d+)?(?:_|$)")


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


def stats(s):
    s = s.replace([np.inf, -np.inf], np.nan).dropna()
    if len(s) == 0:
        return dict(shapes=0, faster="", parity="", slower="", geomean="", median="", worst="", best="")
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
        worst=f"{s.min():.3f}",
        best=f"{s.max():.3f}",
    )


def table(df, col, by=None, order=None):
    rows = []
    if by is not None:
        for k in order or sorted(df[by].dropna().unique()):
            rows.append(dict(group=k, **stats(df.loc[df[by] == k, col])))
    rows.append(dict(group="all", **stats(df[col])))
    return pd.DataFrame(rows)


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        cells = ["" if (isinstance(v, float) and math.isnan(v)) else str(v) for v in r]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def tile_of(name):
    if not isinstance(name, str):
        return None
    m = TILE.search(name)
    return (int(m[1]), int(m[2])) if m else None


def eff(M, N, tile):
    """Useful fraction of the tile slots the GPU runs: M N over waves x P x tile area
    (padding of ragged edges and the idle part of the last wave both count)."""
    if tile is None:
        return math.nan
    bm, bn = tile
    t = math.ceil(M / bm) * math.ceil(N / bn)
    return M * N / (math.ceil(t / P) * P * bm * bn)


def pad_eff(M, N, tile):
    if tile is None:
        return math.nan
    bm, bn = tile
    return M * N / (math.ceil(M / bm) * bm * math.ceil(N / bn) * bn)


def wave_fill(M, N, tile):
    if tile is None:
        return math.nan
    bm, bn = tile
    t = math.ceil(M / bm) * math.ceil(N / bn)
    return t / (math.ceil(t / P) * P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--peak-tbps", type=float, default=5.484)
    args = ap.parse_args()
    run = Path(args.run)
    shapes = pd.read_csv(run / "shapes.csv")
    times = load_replay(run)
    picks = pd.read_csv(run / "picks.csv")
    names = picks.pivot_table(index=KEY, columns="group", values="kernelName", aggfunc="first")
    names.columns = [f"{c}_kernel" for c in names.columns]
    d = shapes.merge(times, on=KEY, how="left").merge(names.reset_index(), on=KEY, how="left")

    d["streamk_any"] = d[["streamk", "streamk_dp"]].min(axis=1)
    for m in ("uneven", "subtile", "streamk_any"):
        d[f"{m}+base"] = d[[m, "base"]].min(axis=1)
    d["best"] = d[["base", "uneven", "subtile", "streamk_any"]].min(axis=1)
    for c in ("uneven", "subtile", "subtile_rule", "streamk_any", "streamk_dp", "best"):
        d[f"sp_{c}"] = d.base / d[c]
    for c in ("uneven", "subtile", "streamk_any"):
        d[f"sp_{c}+base"] = d.base / d[f"{c}+base"]
    d["sp_best_vs_shipped"] = d.shipped / d.best
    d["sp_base_vs_shipped"] = d.shipped / d.base
    d["uneven_vs_subtile"] = d["subtile+base"] / d["uneven+base"]
    d["uneven_vs_streamk"] = d["streamk_any+base"] / d["uneven+base"]
    d["subtile_vs_streamk"] = d["streamk_any+base"] / d["subtile+base"]
    d["mem_frac"] = (
        2 * (d.M * d.K + d.N * d.K + d.M * d.N) / (d.best * 1e-6) / (args.peak_tbps * 1e12)
    )
    for c in ("base", "uneven"):
        tiles = d[f"{c}_kernel"].map(tile_of)
        d[f"eff_{c}"] = [eff(M, N, t) for M, N, t in zip(d.M, d.N, tiles)]
        d[f"pad_{c}"] = [pad_eff(M, N, t) for M, N, t in zip(d.M, d.N, tiles)]
        d[f"fill_{c}"] = [wave_fill(M, N, t) for M, N, t in zip(d.M, d.N, tiles)]

    def winner(r):
        cand = {k: r[k] for k in ("base", "uneven", "subtile", "streamk_any") if not math.isnan(r[k])}
        return min(cand, key=cand.get) if cand else None

    d["winner"] = d.apply(winner, axis=1)
    d.to_csv(run / "results.csv", index=False)

    main_set = d[d.set == "main"]
    out = [f"# Uneven tiles, sub-tile tails and Stream-K against gemm_a16w16.py: {run.name}", ""]
    out.append(
        f"{len(d)} shapes ({', '.join(f'{(d.set == s).sum()} {s}' for s in SETS)}). Times "
        "from aiter's --run_config replay of each method's pick (picked in the tuning run). "
        "Speedup = t_base / t_method, where base is gemm_a16w16.py at its best power-of-two "
        "tile; faster / slower beyond +-3%. The claim is relative to that kernel only."
    )
    missing = {g: int(d.loc[d.set == "main", g].isna().sum()) for g in ("shipped",) + GROUPS}
    out += ["", f"Missing replays on the main set: {missing}", ""]

    out += ["## 1. Each method against the base", ""]
    out += ["Main set (more than one wave of 256x256 tiles, partial last wave), alone:", ""]
    rows = []
    for c, label in (
        ("sp_uneven", "uneven"),
        ("sp_subtile", "subtile"),
        ("sp_subtile_rule", "subtile, gfx1250's rule"),
        ("sp_streamk_any", "streamk"),
        ("sp_uneven+base", "uneven + base"),
        ("sp_subtile+base", "subtile + base"),
        ("sp_streamk_any+base", "streamk + base"),
        ("sp_best", "best of all"),
    ):
        rows.append(dict(method=label, **stats(main_set[c])))
    out += [md(pd.DataFrame(rows)), ""]
    for c, label in (("sp_uneven+base", "uneven + base"), ("sp_subtile+base", "subtile + base"), ("sp_streamk_any+base", "streamk + base")):
        out += [f"{label}, by K:", "", md(table(main_set, c, "k_bin", K_ORDER)), ""]
        out += [f"{label}, by last-wave fill at 256x256:", "", md(table(main_set, c, "fill_bin", FILL_ORDER)), ""]

    out += ["## 2. Head to head (each with the base as fallback)", ""]
    for c, label in (
        ("uneven_vs_subtile", "uneven over subtile (t_subtile / t_uneven)"),
        ("uneven_vs_streamk", "uneven over streamk (t_streamk / t_uneven)"),
        ("subtile_vs_streamk", "subtile over streamk (t_streamk / t_subtile)"),
    ):
        out += [f"{label}, by K:", "", md(table(main_set, c, "k_bin", K_ORDER)), ""]
        out += [f"{label}, by fill:", "", md(table(main_set, c, "fill_bin", FILL_ORDER)), ""]
    w = main_set.pivot_table(index="k_bin", columns="winner", values="M", aggfunc="count").reindex(K_ORDER)
    out += ["Fastest method per shape (no threshold), main set, by K:", "", md(w.fillna(0).astype(int).reset_index()), ""]
    w = main_set.pivot_table(index="fill_bin", columns="winner", values="M", aggfunc="count").reindex(FILL_ORDER)
    out += ["By fill:", "", md(w.fillna(0).astype(int).reset_index()), ""]

    out += ["## 3. Controls", ""]
    for s in ("full_waves", "below_one_wave"):
        part = d[d.set == s]
        rows = [dict(method=l, **stats(part[c])) for c, l in (
            ("sp_uneven+base", "uneven + base"), ("sp_subtile+base", "subtile + base"), ("sp_streamk_any+base", "streamk + base"))]
        out += [f"{s}:", "", md(pd.DataFrame(rows)), ""]

    out += ["## 4. Implementation checks", ""]
    out += [
        "Stream-K's dp plan (no Stream-K) against the base: the Stream-K kernel is separate, "
        "persistent code, so its gains are comparable only if its dp plan matches the base.",
        "",
        md(table(d, "sp_streamk_dp", "set", SETS)),
        "",
    ]

    out += ["## 5. Wave fill against padding (uneven)", ""]
    out += [
        "eff = M N / (waves x 256 x tile area): the useful share of the tile slots, from the "
        "ragged-edge padding (pad) and the idle part of the last wave (fill). At the base's and "
        "the uneven method's picks, main set:",
        "",
    ]
    rows = []
    for c in ("base", "uneven"):
        rows.append(dict(pick=c, **{k: f"{main_set[f'{k}_{c}'].median():.3f}" for k in ("eff", "pad", "fill")}))
    out += [md(pd.DataFrame(rows)), ""]
    gain_from = np.where(
        main_set.fill_uneven - main_set.fill_base > main_set.pad_uneven - main_set.pad_base, "fill", "pad"
    )
    g = main_set.assign(gain_from=gain_from)
    g = g[g["sp_uneven+base"] >= 1 + THRESHOLD]
    out += [
        f"Of the {len(g)} main shapes where uneven + base is faster by >3%, the larger "
        f"improvement over the base's pick is in wave fill on {(g.gain_from == 'fill').sum()} and in "
        f"padding on {(g.gain_from == 'pad').sum()}.",
        "",
    ]

    out += ["## 6. Against the shipped aiter kernel (reference)", ""]
    rows = [dict(method=l, **stats(d.loc[d.set == "main", c])) for c, l in (
        ("sp_base_vs_shipped", "base"), ("sp_best_vs_shipped", "best of all four"))]
    out += ["Speedup = t_shipped / t_ours, main set:", "", md(pd.DataFrame(rows)), ""]
    mb = d[(d.set == "main") & (d.mem_frac >= 0.5)]
    out += [f"{len(mb)} main shapes reach half the HBM peak at the best time (memory-bound or mixed).", ""]

    (run / "RESULTS.md").write_text("\n".join(out) + "\n")
    print("\n".join(out))


if __name__ == "__main__":
    sys.exit(main())
