"""Figures and tables for the tile-quantization issue, from a v2 run.

    python issue_figs.py runs/<name>     # writes runs/<name>/../../issue/ (figures, tables.md)

Inputs: results.csv (replay times per method), profile.csv (every candidate, tuning run),
shapes.csv; and when present torch/ (torch replay, with the base re-timed in the same
session) and torch_kernels.csv (hipBLASLt's kernel and grid per shape).

Fig 1  the mechanism at a fixed 256x256 tile: best Stream-K split and best sub-tile tail over
       whole 256x256 tiles (gemm_a16w16.py), against the last wave's fill S/P, by K.
Fig 2  end to end: each method's best (with whole tiles as fallback; Stream-K = its splitting
       plans, so the Stream-K kernel's other tiles and BLOCK_K = 128 do not count for it) over gemm_a16w16.py at
       its best power-of-two tile, against S/P.
Fig 3  the method with the best geomean in each S/P x K cell.
"""

import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

KEY = ["M", "N", "K", "bias"]
P = 256
THRESHOLD = 0.03
ROW = re.compile(
    r"^\((\d+), (\d+), (\d+), \S+, bias=(True|False)\)\s*\|\s*(\S+)\s*\|\s*(.+?)\s*$"
)
FILL_EDGES = np.arange(0, 9) / 8
FILL_LABELS = [f"{i}/8-{i + 1}/8" for i in range(8)]
K_GROUPS = [
    ("K<=2048", 0, 2048),
    ("2048<K<=4096", 2048, 4096),
    ("4096<K<=8192", 4096, 8192),
    ("K>8192", 8192, 10**9),
]
# Decode-like shapes (M <= 128) never use 256x256 tiles, so their 256x256 leftover is not the
# base's real one: the main set leaves them out.
MIN_M = 129
K_ORDER = ["K<=1024", "K<=2048", "K<=4096", "K<=8192", "K>8192"]
FILL_ORDER = ["S/P<=1/4", "S/P<=1/2", "S/P<=3/4", "S/P<1"]


def gmean(s):
    s = pd.Series(s).replace([np.inf, -np.inf], np.nan).dropna()
    return math.exp(np.log(s).mean()) if len(s) else np.nan


def cell(s):
    s = pd.Series(s).replace([np.inf, -np.inf], np.nan).dropna()
    if not len(s):
        return ""
    f = 100 * (s >= 1 + THRESHOLD).mean()
    sl = 100 * (s <= 1 - THRESHOLD).mean()
    return f"{gmean(s):.3f} / {f:.0f}% / {sl:.0f}% (n={len(s)})"


def md(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        lines.append("| " + " | ".join("" if (isinstance(v, float) and math.isnan(v)) else str(v) for v in r) + " |")
    return "\n".join(lines)


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


def fill_bin8(fill):
    return pd.cut(fill, FILL_EDGES, labels=FILL_LABELS, include_lowest=False, right=True)


def fixed_tile(run, main):
    """Per main shape: whole 256x256 tiles, the best Stream-K split at 256x256x64 and the best
    sub-tile tail on a 256x256 main tile, all from the tuning profile."""
    prof = pd.read_csv(run / "profile.csv")
    prof = prof[(prof.us > 0) & (prof.err_ratio <= 0.05)].merge(main[KEY], on=KEY)
    name = prof.kernelName
    whole = prof[name == "gluon_compute_bound_256x256"].groupby(KEY).us.min().rename("whole")
    split = (
        prof[name.str.startswith("gluon_streamk_256x256x64_") & ~name.str.endswith("_dp")]
        .groupby(KEY)
        .us.min()
        .rename("split")
    )
    sub = prof[name.str.startswith("gluon_subtile_256x256_tail")].groupby(KEY).us.min().rename("sub")
    d = pd.concat([whole, split, sub], axis=1).reset_index().merge(main, on=KEY)
    d["split_gain"] = d.whole / d.split
    d["sub_gain"] = d.whole / d["sub"]
    return d


def fig1(d, out):
    d = d.assign(fb=fill_bin8(d.S / P))
    x = (np.arange(8) + 0.5) / 8
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    for label, lo, hi in K_GROUPS:
        q = d[(d.K > lo) & (d.K <= hi)]
        y = [gmean(q.loc[q.fb == b, "split_gain"]) for b in FILL_LABELS]
        n = int(q.split_gain.notna().sum())
        ax.plot(x, y, marker="o", label=f"Stream-K split, {label} ({n})")
    y = [gmean(d.loc[d.fb == b, "sub_gain"]) for b in FILL_LABELS]
    ax.plot(x, y, marker="s", color="black", linewidth=2.2, label=f"sub-tile tail, all K ({int(d.sub_gain.notna().sum())})")
    ax.axhline(1, color="grey", linewidth=0.8)
    ax.axvline(0.5, color="grey", linestyle=":", linewidth=0.8)
    ax.text(0.505, 0.99, "S > P/2: no 2-way split fits one wave", fontsize=8, va="top",
            transform=ax.get_xaxis_transform())
    ax.set_xlabel("last-wave fill S/P at 256x256 (bins of 1/8)")
    ax.set_ylabel("speedup over whole 256x256 tiles (geomean)")
    ax.set_title(f"Tail methods at a fixed 256x256 tile, {len(d)} shapes with more than one wave (M > 128)")
    ax.set_xlim(0, 1)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig1_fixed_tile.png", dpi=150)
    plt.close(fig)


def load_torch(run, d):
    tdir = run / "torch" / "replay"
    if not tdir.exists():
        return None
    parts_t, parts_b = [], []
    for c in sorted(tdir.iterdir()):
        parts_t.append(parse_log(c / "torch.log", "torch"))
        parts_b.append(parse_log(c / "base.log", "base_now"))
    t = pd.concat(parts_t).drop_duplicates(KEY).merge(pd.concat(parts_b).drop_duplicates(KEY), on=KEY)
    return d.merge(t, on=KEY, how="left")


def fig2(m, out):
    m = m.assign(fb=fill_bin8(m.S / P))
    x = (np.arange(8) + 0.5) / 8
    series = [
        ("finer tiles", "sp_uneven+base", "-", "o"),
        ("sub-tile tail", "sp_subtile+base", "-", "s"),
        ("Stream-K", "sp_split+base", "-", "^"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    for ax, (title, sel) in zip(axes, (("K <= 4096", m.K <= 4096), ("K > 4096", m.K > 4096))):
        q = m[sel]
        for label, col, ls, mk in series:
            ax.plot(x, [gmean(q.loc[q.fb == b, col]) for b in FILL_LABELS], ls, marker=mk, label=label)
        ax.axhline(1, color="grey", linewidth=0.8)
        ax.axvline(0.5, color="grey", linestyle=":", linewidth=0.8)
        ax.set_title(f"{title} ({len(q)} shapes)")
        ax.set_xlabel("last-wave fill S/P at 256x256 (bins of 1/8)")
        ax.set_xlim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("speedup over base (geomean)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Each method's best (with whole-tile fallback) against the base at its best power-of-two tile")
    fig.tight_layout()
    fig.savefig(out / "fig2_end_to_end.png", dpi=150)
    plt.close(fig)


def fig3(m, out):
    """Every method's geomean speedup in each fill x K cell. The cell takes the winner's colour,
    at full strength when it leads the runner-up by 3% or more and pale when the lead is small."""
    methods = [("finer tiles", "sp_uneven+base", "#4c72b0"), ("sub-tile tail", "sp_subtile+base", "#dd8452"),
               ("Stream-K", "sp_split+base", "#55a868")]
    fig, ax = plt.subplots(figsize=(11, 5.6))
    for i, f in enumerate(FILL_ORDER):
        for j, k in enumerate(K_ORDER):
            q = m[(m.fill_bin == f) & (m.k_bin == k)]
            if not len(q):
                continue
            g = [gmean(q[col]) for _, col, _ in methods]
            order = np.argsort(g)[::-1]
            lead = g[order[0]] / g[order[1]] - 1
            strength = 0.15 + 0.85 * min(lead / 0.03, 1)
            colour = 1 - (1 - np.array(matplotlib.colors.to_rgb(methods[order[0]][2]))) * strength
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, color=colour))
            ink = "white" if strength > 0.75 else "black"
            for r, (name, _, _) in enumerate(methods):
                ax.text(j, i - 0.3 + 0.19 * r, f"{name} {g[r]:.3f}", ha="center", va="center", fontsize=7.5,
                        color=ink, fontweight="bold" if r == order[0] else "normal")
            ax.text(j, i + 0.3, f"lead {100 * lead:.1f}%, n={len(q)}" + (" (few)" if len(q) < 10 else ""),
                    ha="center", va="center", fontsize=7, color=ink)
    ax.set_xlim(-0.5, len(K_ORDER) - 0.5)
    ax.set_ylim(len(FILL_ORDER) - 0.5, -0.5)
    ax.set_xticks(range(len(K_ORDER)), ["<=1024", "1024 to 2048", "2048 to 4096", "4096 to 8192", ">8192"])
    ax.set_yticks(range(len(FILL_ORDER)), ["0 to 1/4", "1/4 to 1/2", "1/2 to 3/4", "3/4 to 1"])
    ax.set_xlabel("K")
    ax.set_ylabel("last-wave fill S/P at 256x256")
    ax.set_title("Speedup over the base (geomean, each method with whole-tile fallback)\n"
                 "colour: best method, pale when its lead over the next is under 1%, full at 3% or more")
    fig.tight_layout()
    fig.savefig(out / "fig3_best_method.png", dpi=150)
    plt.close(fig)


def fig4(transfer_run, out):
    """The M-transfer experiment (v9_transfer.py, runs/transfer12) in the issue's terms: per
    prefill anchor, how many tiles are best over M0/2 < M < M0 and what the tile tuned at M0
    loses against the best tile per M, for the base menu and the menu with finer tiles."""
    import v9_transfer as vt

    _, side = vt.summary(pd.read_csv(transfer_run / "v9_transfer.csv"))
    labels = [f"N={r.N}\nK={r.K}\nM0={r.M0}" for r in side.itertuples()]
    x = np.arange(len(side))
    w = 0.38
    fig, (ax_d, ax_l) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax_d.bar(x - w / 2, side.v9_distinct, w, color="C0", label="base (power-of-two tiles)")
    ax_d.bar(x + w / 2, side.uneven_distinct, w, color="C4", label="with finer tiles")
    ax_d.set_ylabel("tiles that are best\nsomewhere in the range")
    ax_d.legend(fontsize=8)
    ax_d.grid(alpha=0.3, axis="y")
    ax_l.bar(x - w / 2, side.v9_fixed_loss, w, color="C0", label="base: tuned at M0 / best per M")
    ax_l.bar(x + w / 2, side.uneven_fixed_loss, w, color="C4", label="with finer tiles: tuned at M0 / best per M")
    ax_l.axhline(1, color="grey", lw=1)
    ax_l.set_ylim(0.98, max(side.v9_fixed_loss.max(), side.uneven_fixed_loss.max()) + 0.03)
    ax_l.set_ylabel("slowdown of the tile\ntuned at M0 (geomean)")
    ax_l.set_xticks(list(x), labels, fontsize=7)
    ax_l.legend(fontsize=8)
    ax_l.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Prefill, M0/2 < M < M0 (M > {vt.M_MIN}): with finer tiles the best tile changes with M")
    fig.tight_layout()
    fig.savefig(out / "fig4_tile_reuse_over_m.png", dpi=150)
    plt.close(fig)


def main(run):
    run = Path(run)
    out = run.parent.parent / "issue"
    out.mkdir(exist_ok=True)
    d = pd.read_csv(run / "results.csv")
    t = load_torch(run, d)
    if t is not None:
        d = t
    d["split+base"] = d[["streamk", "base"]].min(axis=1)
    d["sp_split+base"] = d.base / d["split+base"]
    main_set = d[(d.set == "main") & (d.M >= MIN_M)].copy()
    ft = fixed_tile(run, main_set[KEY + ["S", "fill", "k_bin", "fill_bin"]].copy())
    fig1(ft, out)
    fig2(main_set, out)
    fig3(main_set, out)
    transfer_run = run.parent / "transfer12"
    if (transfer_run / "v9_transfer.csv").exists():
        fig4(transfer_run, out)

    t_md = ["# Tables for the tile-quantization issue", ""]
    t_md += [
        f"Run `{run}`: {len(main_set)} shapes with more than one wave of 256x256 tiles and a partial "
        "last wave (S/P below is at 256x256). Cells: geomean / faster by >3% / slower by >3%.",
        "",
    ]

    t_md += ["## Fixed 256x256 tile (tuning profile): tail method over whole 256x256 tiles", ""]
    prof = pd.read_csv(run / "profile.csv")
    prof = prof[(prof.us > 0) & (prof.err_ratio <= 0.05)].merge(main_set[KEY], on=KEY)
    name = prof.kernelName
    vs = pd.concat(
        [
            prof[name.str.startswith("gluon_compute_bound_")].groupby(KEY).us.min().rename("best_base"),
            prof[name.str.startswith("gluon_streamk_") & ~name.str.endswith("_dp")].groupby(KEY).us.min().rename("split_any"),
            prof[name.str.startswith("gluon_subtile_")].groupby(KEY).us.min().rename("sub_any"),
        ],
        axis=1,
    ).reset_index().merge(ft, on=KEY)
    for half, cond in (("at most half a wave left", vs.S <= P // 2), ("more than half a wave left", vs.S > P // 2)):
        rows = []
        for label, sel in [(k, vs.k_bin == k) for k in K_ORDER] + [("all", vs.K > 0)]:
            q = vs[sel & cond]
            rows.append({"K": label, "n": len(q),
                         "Stream-K split vs whole 256x256": cell(q.split_gain),
                         "sub-tile vs whole 256x256": cell(q.sub_gain),
                         "Stream-K split, tuned tile, vs base": cell(q.best_base / q.split_any),
                         "sub-tile, tuned tile, vs base": cell(q.best_base / q.sub_any)})
        t_md += [f"By K, {half} (tuned columns: no fallback to whole tiles):", "", md(pd.DataFrame(rows)), ""]

    rows = []
    for label in FILL_ORDER + ["all"]:
        sel = ft.fill_bin == label if label != "all" else ft.K > 0
        rows.append(dict(fill=label, n=int(sel.sum()), stream_k_split=cell(ft.loc[sel, "split_gain"]),
                         sub_tile=cell(ft.loc[sel, "sub_gain"])))
    t_md += ["By last-wave fill:", "", md(pd.DataFrame(rows)), ""]

    comps = [
        ("uneven vs base", main_set.base / main_set["uneven+base"]),
        ("sub-tile vs base", main_set.base / main_set["subtile+base"]),
        ("Stream-K vs base", main_set.base / main_set["split+base"]),
        ("uneven vs sub-tile", main_set["subtile+base"] / main_set["uneven+base"]),
        ("uneven vs Stream-K", main_set["split+base"] / main_set["uneven+base"]),
        ("sub-tile vs Stream-K", main_set["split+base"] / main_set["subtile+base"]),
        ("Stream-K kernel, any plan or tile, vs base", main_set.base / main_set["streamk_any+base"]),
    ]
    for by, order in (("fill_bin", FILL_ORDER), ("k_bin", K_ORDER)):
        rows = []
        for name, s in comps:
            r = {"comparison": name}
            for o in order:
                r[f"{o} ({int((main_set[by] == o).sum())})"] = cell(s[main_set[by] == o])
            r["all"] = cell(s)
            rows.append(r)
        t_md += [f"## End to end, by {'last-wave fill' if by == 'fill_bin' else 'K'} (each method with whole tiles as fallback)", "", md(pd.DataFrame(rows)), ""]

    main_set["Wb"] = pd.cut(main_set.W, [0, 1, 2, 4, 8, 10**6], labels=["1", "2", "3-4", "5-8", ">8"])
    rows = []
    for w in ["1", "2", "3-4", "5-8", ">8"]:
        q = main_set[main_set.Wb == w]
        rows.append({"full waves at 256x256": w, "n": len(q),
                     "uneven vs base": cell(q.base / q["uneven+base"]),
                     "sub-tile vs base": cell(q.base / q["subtile+base"]),
                     "Stream-K vs base": cell(q.base / q["split+base"]),
                     "uneven vs Stream-K": cell(q["split+base"] / q["uneven+base"])})
    t_md += ["## By number of full waves", "", md(pd.DataFrame(rows)), ""]

    ai = lambda n: (lambda a, b: a * b / (a + b))(*map(int, re.search(r"_(\d+)x(\d+)", n).groups())) if isinstance(n, str) else np.nan  # noqa: E731
    t_md += [
        "## Intensity of the picks",
        "",
        f"Median BM BN / (BM + BN) of the pick: base {main_set.base_kernel.map(ai).median():.1f}, "
        f"uneven {main_set.uneven_kernel.map(ai).median():.1f} (256x256: 128).",
        "",
    ]

    prof = pd.read_csv(run / "profile.csv")
    ex = prof[(prof.M == 1024) & (prof.N == 16896) & (prof.K == 7168) & (prof.us > 0)]
    rows = []
    for tile in ("256x256", "256x224", "256x192", "256x160", "128x288"):
        bm, bn = map(int, tile.split("x"))
        kind = "compute_bound" if (bm & (bm - 1) == 0 and bn & (bn - 1) == 0) else "uneven"
        us = ex.loc[ex.kernelName == f"gluon_{kind}_{tile}", "us"]
        tiles = math.ceil(1024 / bm) * math.ceil(16896 / bn)
        waves = math.ceil(tiles / P)
        rows.append(dict(tile=tile, tiles=tiles, last_wave=tiles - (waves - 1) * P,
                         useful_share=f"{1024 * 16896 / (waves * P * bm * bn):.2f}",
                         intensity=round(bm * bn / (bm + bn)), us=f"{us.min():.1f}" if len(us) else ""))
    r = d[(d.M == 1024) & (d.N == 16896) & (d.K == 7168)].iloc[0]
    t_md += [
        "## Worked example: 1024x16896x7168 (264 tiles at 256x256, 8 in the last wave)",
        "",
        md(pd.DataFrame(rows)),
        "",
        f"Replay: base {r.base:.1f} us, uneven {r.uneven:.1f}, sub-tile {r.subtile:.1f}, Stream-K {r.streamk_any:.1f}.",
        "",
    ]

    tk = run / "torch_kernels.csv"
    if tk.exists():
        k = pd.read_csv(tk).merge(d[KEY + ["set", "T", "S", "fill_bin"]], on=KEY)
        k["hbl_S"] = k.hbl_tiles % P
        k["hbl_fill"] = np.where(k.hbl_S == 0, 1.0, k.hbl_S / P)
        k["area"] = k.hbl_bm_n * k.hbl_bn_m
        km = k[(k.set == "main") & (k.M >= MIN_M)]
        small = km[km.fill_bin == "S/P<=1/4"]
        t_md += [
            "## hipBLASLt's choice (torch, this session)",
            "",
            f"Main shapes: {len(km)}. Launch grid against tile count: one workgroup per tile on "
            f"{int((km.hbl_wgs == km.hbl_tiles).sum())}, a whole number of tiles per workgroup on "
            f"{int(((km.hbl_wgs != km.hbl_tiles) & (km.hbl_tiles % km.hbl_wgs == 0)).sum())}, a fractional "
            f"number (where a Stream-K split can run) on {int((km.hbl_tiles % km.hbl_wgs != 0).sum())}. "
            f"Stream-K mode in the kernel names: {km.hbl_sk.value_counts().to_dict()}. Tiles larger than "
            f"256x256 of accumulators on {int((km.area > 65536).sum())}; the largest is "
            f"{km.loc[km.area.idxmax(), 'hbl_bm_n']}x{km.loc[km.area.idxmax(), 'hbl_bn_m']} (N x M).",
            "",
            f"With the last wave at most 1/4 full at 256x256 ({len(small)} shapes), hipBLASLt's tile leaves "
            f"the last wave exactly full on {int((small.hbl_S == 0).sum())}, at least 3/4 full on "
            f"{int((small.hbl_fill >= 0.75).sum())}, and at most 1/4 full on {int(((small.hbl_fill <= 0.25) & (small.hbl_S > 0)).sum())}.",
            "",
        ]
        tiles = km.assign(t=km.hbl_bm_n.astype("Int64").astype(str) + "x" + km.hbl_bn_m.astype("Int64").astype(str)).t.value_counts().head(10)
        t_md += ["Most used tiles (N x M):", "", md(tiles.rename_axis("tile").reset_index(name="shapes")), ""]
    if t is not None and tk.exists():
        # Our replay times, scaled by the base's drift between our replay and the torch session.
        q = main_set.merge(k[KEY + ["area"]], on=KEY)
        drift = q.base_now / q.base
        best3 = q[["uneven+base", "subtile+base", "split+base"]].min(axis=1)
        ours = [("base (power-of-two tiles)", q.base), ("finer tiles", q["uneven+base"]),
                ("sub-tile tail", q["subtile+base"]), ("Stream-K", q["split+base"]), ("best of the three", best3)]
        big = q.area > 65536
        rows = []
        for label, col in ours:
            r = q.torch / (col * drift)
            rows.append({"ours": label, "all": cell(r),
                         f"hipBLASLt tile <= 256x256 ({int((~big).sum())})": cell(r[~big]),
                         f"hipBLASLt tile > 256x256 ({int(big.sum())})": cell(r[big])})
        t_md += [
            "## Against hipBLASLt (raw, t_hipBLASLt / t_ours)",
            "",
            md(pd.DataFrame(rows)),
            "",
            f"Base re-timed in the torch session against its earlier replay: median {drift.median():.3f}, "
            f"5-95% {drift.quantile(0.05):.3f}-{drift.quantile(0.95):.3f}.",
            "",
        ]
    (out / "tables.md").write_text("\n".join(t_md) + "\n")
    print("\n".join(t_md))


if __name__ == "__main__":
    main(sys.argv[1])
