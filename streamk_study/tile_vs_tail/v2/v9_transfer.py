"""Prefill (M > M_MIN): v9 (gemm_a16w16.py, power-of-two tiles) with its tile fixed at
M0 against v9 with a tile chosen per M, and the same for the menu with uneven tiles.
From the transfer12 profile (every base and uneven tile timed at every M).

    python v9_transfer.py runs/<name>                  # v9_transfer.csv, summary, figures
    python v9_transfer.py runs/<name> --replay-make 4  # replay CSVs (every fourth M)
    python v9_transfer.py runs/<name> --replay-compare # replayed ratios -> summary

Per anchor (N, K, M0 > M_MIN) and M (> M_MIN) of its range:
  v9_fixed      v9's best tile at M0, run at M
  v9_best       v9's best tile at M
  uneven_fixed  the best base or uneven tile at M0, run at M
  best_uneven   the best base or uneven tile at M
  torch         hipBLASLt through F.linear (kernel trace), reference only
Comparisons, below M0:
  1  v9_fixed / v9_best          fixing v9's tile, with the tiles we have
  2  v9_best / best_uneven       what v9's menu leaves on the table
  3  uneven_fixed / best_uneven  fixing the tile with the finer menu
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from analyze_transfer import TRANSFER, efficiency, load, tile_of, torch_times  # noqa: E402
from make_rows import m_grid  # noqa: E402
from replay_transfer import COLUMNS, parse  # noqa: E402

M_MIN = 512
SMALL_N = 2560
OVERVIEW = [(2560, 8192, 8192), (1536, 1536, 8192), (5120, 5120, 4096), (5120, 5120, 16384)]
METHODS = ("v9_fixed", "v9_best", "uneven_fixed", "best_uneven")
STYLE = dict(v9_fixed=("C0", "-", "v9, tile fixed at M0"),
             v9_best=("C1", "-", "v9, best tile per M"),
             uneven_fixed=("C4", "-", "with uneven, tile fixed at M0"),
             best_uneven=("C2", "--", "with uneven, best tile per M"))


def short(name):
    return name.replace("gluon_compute_bound_", "").replace("gluon_", "")


def best(at, methods):
    names = [n for n, m in at["_method"].items() if m in methods]
    return min(names, key=at.get)


def anchor_rows(N, K, M0, prof, torch):
    at0 = prof[(M0, N, K)]
    v9_m0, un_m0 = best(at0, ("base",)), best(at0, ("base", "uneven"))
    rows = []
    for M in m_grid(M0):
        if M <= M_MIN or (M, N, K) not in prof:
            continue
        at = prof[(M, N, K)]
        names = dict(v9_fixed=v9_m0, v9_best=best(at, ("base",)), uneven_fixed=un_m0,
                     best_uneven=best(at, ("base", "uneven")))
        row = dict(N=N, K=K, M0=M0, M=M, torch=torch.get((N, K, M0, M), float("nan")))
        for meth, n in names.items():
            row[meth] = at[n]
            row[f"{meth}_name"] = n
            tiles, waves, pad, slot = efficiency(M, N, tile_of(n))
            row.update({f"{meth}_tiles": tiles, f"{meth}_waves": waves,
                        f"{meth}_pad": pad, f"{meth}_slot": slot, f"{meth}_eff": pad * slot})
        rows.append(row)
    return rows


def build(run):
    prof = load(run)
    torch = torch_times()
    anchors = pd.concat([pd.read_csv(TRANSFER / f)
                         for f in ("anchors_hbl.csv", "anchors_hbl_large_m.csv")])
    rows = []
    for a in anchors.itertuples():
        if a.M0 > M_MIN:
            rows += anchor_rows(a.N, a.K, a.M0, prof, torch)
    d = pd.DataFrame(rows)
    d.to_csv(run / "v9_transfer.csv", index=False)
    return d


def geomean(s):
    s = pd.Series(s).dropna()
    return math.exp(s.map(math.log).mean()) if len(s) else float("nan")


def stats(r):
    r = pd.Series(r).dropna()
    return dict(points=len(r), geomean=round(geomean(r), 3), worst=round(float(r.max()), 3),
                above_5pct=f"{100 * (r > 1.05).mean():.0f}%")


def menu_stats(g, fixed, per_m):
    r = g[fixed] / g[per_m]
    return dict(**{f"{k}": v for k, v in stats(r).items()},
                changes=f"{int((g[f'{per_m}_name'] != g[f'{fixed}_name']).sum())}/{len(g)}",
                distinct=g[f"{per_m}_name"].nunique())


def summary(d):
    below = d[d.M < d.M0]
    out = ["# Prefill (M > 512): a tile fixed at M0 against a tile chosen per M (v9)", "",
           f"Our Gluon v9 kernel (gemm_a16w16.py, power-of-two tiles) and the uneven tiles "
           f"(gemm_a16w16_uneven.py), every tile timed at every M with the tuner "
           f"(transfer12). {below.groupby(['N', 'K', 'M0']).ngroups} anchors with M0 > "
           f"{M_MIN}, {len(below)} points with {M_MIN} < M < M0. Comparisons: (1) v9_fixed / "
           "v9_best, fixing v9's tile with the tiles we have; (2) v9_best / best_uneven, what "
           "v9's menu leaves on the table; (3) uneven_fixed / best_uneven, fixing the tile "
           "with the finer menu.", ""]

    pooled = []
    for label, g in (("all anchors", below), (f"small N (N <= {SMALL_N})", below[below.N <= SMALL_N]),
                     (f"large N (N > {SMALL_N})", below[below.N > SMALL_N])):
        for comp, (num, den) in (("1: v9_fixed / v9_best", ("v9_fixed", "v9_best")),
                                 ("2: v9_best / best_uneven", ("v9_best", "best_uneven")),
                                 ("3: uneven_fixed / best_uneven", ("uneven_fixed", "best_uneven")),
                                 ("v9_fixed / best_uneven", ("v9_fixed", "best_uneven"))):
            pooled.append(dict(group=label, comparison=comp, **stats(g[num] / g[den])))
    out += ["## 1. Pooled", "", pd.DataFrame(pooled).to_markdown(index=False), ""]

    side = []
    for (N, K, M0), g in below.groupby(["N", "K", "M0"]):
        v9, un = menu_stats(g, "v9_fixed", "v9_best"), menu_stats(g, "uneven_fixed", "best_uneven")
        side.append(dict(N=N, K=K, M0=M0, M_range=f"{g.M.min()}-{g.M.max()}",
                         v9_tile_M0=short(g.v9_fixed_name.iloc[0]), v9_fixed_loss=v9["geomean"],
                         v9_worst=v9["worst"], v9_above_5pct=v9["above_5pct"],
                         v9_changes=v9["changes"], v9_distinct=v9["distinct"],
                         uneven_tile_M0=short(g.uneven_fixed_name.iloc[0]),
                         uneven_fixed_loss=un["geomean"], uneven_worst=un["worst"],
                         uneven_above_5pct=un["above_5pct"], uneven_changes=un["changes"],
                         uneven_distinct=un["distinct"],
                         v9_best_vs_uneven=round(geomean(g.v9_best / g.best_uneven), 3),
                         v9_fixed_vs_uneven=round(geomean(g.v9_fixed / g.best_uneven), 3)))
    side = pd.DataFrame(side)
    out += ["## 2. Per anchor: v9 menu against the menu with uneven tiles", "",
            "fixed_loss = geomean of (tile fixed at M0 / best tile of the same menu per M); "
            "changes = M values where the menu's best tile differs from its M0 pick; "
            "distinct = number of different best tiles over the range.", "",
            side.to_markdown(index=False), ""]

    tiles = []
    for (N, K, M0), g in below.groupby(["N", "K", "M0"]):
        tiles.append(dict(N=N, K=K, M0=M0,
                          v9_best_tiles=", ".join(dict.fromkeys(map(short, g.sort_values("M").v9_best_name))),
                          best_uneven_tiles=", ".join(dict.fromkeys(map(short, g.sort_values("M").best_uneven_name)))))
    out += ["## 3. Best tiles over the range (in M order)", "",
            pd.DataFrame(tiles).to_markdown(index=False), ""]
    text = "\n".join(out)
    return text, side


def figure(g, path):
    g = g.set_index("M")
    N, K, M0 = g.N.iloc[0], g.K.iloc[0], g.M0.iloc[0]
    fig, (ax_t, ax_r, ax_w, ax_e) = plt.subplots(
        4, 1, figsize=(9, 12), sharex=True, gridspec_kw=dict(height_ratios=[3, 2, 1.6, 2]))
    for meth, (c, ls, label) in STYLE.items():
        tile = short(g[f"{meth}_name"].iloc[-1]) if meth.endswith("fixed") else ""
        ax_t.plot(g.index, g[meth], color=c, ls=ls, lw=1.5, label=f"{label} {tile}".strip())
    for meth, dy in (("v9_best", -14), ("best_uneven", 8)):
        prev = None
        for M, r in g.iterrows():
            if r[f"{meth}_name"] != prev:
                ax_t.annotate(short(r[f"{meth}_name"]), (M, r[meth]), textcoords="offset points",
                              xytext=(0, dy), fontsize=6, ha="center", color=STYLE[meth][0])
                prev = r[f"{meth}_name"]
    ax_t.plot(g.index, g.torch, color="grey", lw=1, alpha=0.5, label="torch (kernel trace, reference)")
    ax_t.set_ylabel("time (us)")
    ax_t.set_title(f"N={N} K={K}, anchor M0={M0} (M > {M_MIN})")
    ax_t.legend(fontsize=7)
    ax_t.grid(alpha=0.3)
    for meth in ("v9_fixed", "v9_best", "uneven_fixed"):
        c, ls, label = STYLE[meth]
        ax_r.plot(g.index, g[meth] / g.best_uneven, color=c, ls=ls, lw=1.5, label=label)
    ax_r.axhline(1, color="grey", ls=":", lw=1)
    ax_r.set_ylabel("/ best tile per M (with uneven)")
    ax_r.legend(fontsize=7)
    ax_r.grid(alpha=0.3)
    for meth in ("v9_fixed", "v9_best"):
        c, _, label = STYLE[meth]
        ax_w.step(g.index, g[f"{meth}_waves"], where="mid", color=c, label=label)
        ax_e.plot(g.index, g[f"{meth}_pad"], color=c, ls="--", lw=1)
        ax_e.plot(g.index, g[f"{meth}_slot"], color=c, ls=":", lw=1.2)
        ax_e.plot(g.index, g[f"{meth}_eff"], color=c, lw=1.8, label=label)
    top = int(max(g.v9_fixed_waves.max(), g.v9_best_waves.max())) + 2
    for w in range(1, top):
        ax_w.axhline(w, color="grey", lw=0.8, ls=":")
    ax_w.set_ylabel("waves (tiles / 256)")
    ax_w.legend(fontsize=7)
    ax_e.set_ylabel("padding --, wave slots :, overall -")
    ax_e.set_xlabel("M")
    ax_e.legend(fontsize=7)
    ax_e.grid(alpha=0.3)
    for ax in (ax_t, ax_r, ax_w, ax_e):
        ax.axvline(M0, color="grey", ls="--", lw=1)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def overview(d, path):
    fig, axes = plt.subplots(1, len(OVERVIEW), figsize=(5 * len(OVERVIEW), 4), sharey=True)
    for ax, (N, K, M0) in zip(axes, OVERVIEW):
        g = d[(d.N == N) & (d.K == K) & (d.M0 == M0)].set_index("M")
        for meth in ("v9_fixed", "v9_best", "uneven_fixed"):
            c, ls, label = STYLE[meth]
            ax.plot(g.index, g[meth] / g.best_uneven, color=c, ls=ls, lw=1.6, label=label)
        ax.axhline(1, color="grey", ls=":", lw=1)
        ax.axvline(M0, color="grey", ls="--", lw=1)
        kind = "effect" if N <= SMALL_N else "control"
        ax.set_title(f"N={N} K={K}, M0={M0} ({kind})", fontsize=10)
        ax.set_xlabel("M")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("/ best tile per M (with uneven)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def tile_changes(side, path):
    labels = [f"N={r.N}\nK={r.K}\nM0={r.M0}" for r in side.itertuples()]
    x = range(len(side))
    fig, (ax_d, ax_l) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    w = 0.38
    ax_d.bar([i - w / 2 for i in x], side.v9_distinct, w, color="C0", label="v9 menu")
    ax_d.bar([i + w / 2 for i in x], side.uneven_distinct, w, color="C4", label="with uneven")
    ax_d.set_ylabel("distinct best tiles\nover the range")
    ax_d.legend(fontsize=8)
    ax_d.grid(alpha=0.3, axis="y")
    ax_l.bar([i - w / 2 for i in x], side.v9_fixed_loss, w, color="C0", label="v9: fixed / best per M")
    ax_l.bar([i + w / 2 for i in x], side.uneven_fixed_loss, w, color="C4",
             label="with uneven: fixed / best per M")
    ax_l.axhline(1, color="grey", lw=1)
    ax_l.set_ylim(0.98, max(side.v9_fixed_loss.max(), side.uneven_fixed_loss.max()) + 0.03)
    ax_l.set_ylabel("loss of the tile fixed\nat M0 (geomean)")
    ax_l.set_xticks(list(x), labels, fontsize=7)
    ax_l.legend(fontsize=8)
    ax_l.grid(alpha=0.3, axis="y")
    fig.suptitle(f"Prefill (M > {M_MIN}): a finer tile menu makes the per-M choice matter more")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def replay_make(run, ngpus):
    d = pd.read_csv(run / "v9_transfer.csv")
    sel = []
    for _, g in d.groupby(["N", "K", "M0"]):
        g = g.sort_values("M")
        sel.append(pd.concat([g.iloc[::4], g[g.M == g.M0]]).drop_duplicates("M"))
    d = pd.concat(sel)
    out = run / "replay_v9"
    out.mkdir(exist_ok=True)
    files = []
    for meth in METHODS:
        rows = [dict(gfx="gfx950", cu_num=256, M=r.M, N=r.N, K=r.K, bias=False,
                     dtype="torch.bfloat16", outdtype="torch.bfloat16", scaleAB=False,
                     bpreshuffle=False, libtype="gluon", solidx=0, splitK=0,
                     us=getattr(r, meth), kernelName=getattr(r, f"{meth}_name"),
                     err_ratio=0.0, tflops=0.0, bw=0.0) for r in d.itertuples()]
        f = pd.DataFrame(rows, columns=COLUMNS)
        dup = f.groupby(["M", "N", "K"]).kernelName.nunique()
        assert (dup == 1).all(), f"{meth}: two kernels for one shape {dup[dup > 1]}"
        f.drop_duplicates(["M", "N", "K"]).to_csv(out / f"{meth}.csv", index=False)
        files.append(f"{meth}.csv")
    for i in range(ngpus):
        (out / f"gpu{i}.txt").write_text("\n".join(files[i::ngpus]) + "\n")
    print(f"{len(files)} files, {len(d)} (anchor, M) points")


def replay_compare(run):
    out = run / "replay_v9"
    d = pd.read_csv(run / "v9_transfer.csv")
    times, noise, tune = {}, [], []
    for meth in METHODS:
        a, b = parse(out / f"{meth}_a.log"), parse(out / f"{meth}_b.log")
        f = pd.read_csv(out / f"{meth}.csv")
        for r in f.itertuples():
            k = (r.M, r.N, r.K)
            if k in a and k in b:
                times[(meth,) + k] = (a[k] + b[k]) / 2
                noise.append(abs(a[k] - b[k]) / min(a[k], b[k]))
                tune.append(times[(meth,) + k] / r.us)
    rows = []
    for r in d[d.M < d.M0].itertuples():
        k = (r.M, r.N, r.K)
        t = {m: times.get((m,) + k) for m in METHODS}
        if all(t.values()):
            rows.append(dict(N=r.N, M=r.M, **t))
    r = pd.DataFrame(rows)
    lines = ["", f"## 4. Replay (--run_config, A/B mean) at every fourth M", "",
             f"{len(noise)} replayed (method, shape) points; median A/B noise "
             f"{100 * pd.Series(noise).median():.1f}%; replay / tuner time geomean "
             f"{geomean(tune):.3f}.", "",
             "| comparison (below M0, replay times) | points | geomean | worst |", "|---|---|---|---|"]
    for num, den in (("v9_fixed", "v9_best"), ("v9_best", "best_uneven"),
                     ("uneven_fixed", "best_uneven"), ("v9_fixed", "best_uneven")):
        s = stats(r[num] / r[den])
        lines.append(f"| {num} / {den} | {s['points']} | {s['geomean']} | {s['worst']} |")
    text = "\n".join(lines) + "\n"
    with open(run / "v9_transfer_summary.md", "a") as f:
        f.write(text)
    print(text)


def main():
    run = Path(sys.argv[1])
    if len(sys.argv) > 2 and sys.argv[2] == "--replay-make":
        return replay_make(run, int(sys.argv[3]))
    if len(sys.argv) > 2 and sys.argv[2] == "--replay-compare":
        return replay_compare(run)
    d = build(run)
    text, side = summary(d)
    (run / "v9_transfer_summary.md").write_text(text + "\n")
    print(text)
    for (N, K, M0), g in d.groupby(["N", "K", "M0"]):
        figure(g, run / f"fig_v9_N{N}_K{K}_M{M0}.png")
    overview(d, run / "fig_v9_overview.png")
    tile_changes(side, run / "fig_v9_tile_changes.png")


if __name__ == "__main__":
    main()
