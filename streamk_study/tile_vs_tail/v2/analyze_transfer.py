"""Does our runtime Stream-K plan on a tile picked at M0 rescue the fixed-tile loss?
The 12 hipBLASLt anchors (anchor_shapes.py), every candidate timed at every M.

    python analyze_transfer.py runs/<name>   # writes transfer_summary.md, transfer.csv
                                             # and fig_transfer_N*_K*_M*.png in the run

Per anchor (N, K, M0) and M of its range, from profile.csv (tuner times):
  tile_fixed     the best base or uneven tile at M0, run with that tile at M
                 (a fixed tile, no Stream-K)
  sk_runtime     the tile of the best Stream-K config at M0, with complete_config's
                 default plan at M (what a tile-only config runs: the hypothesis)
  sk_tile_best   that tile with its best plan at M (the ceiling for any plan rule)
  sk_fixed       the best Stream-K config at M0, run as configured at M when its plan
                 still fits (found by its launch constants), else sk_runtime
  best_tile      the best base or uneven tile at M (per-M tile choice, no Stream-K)
  best_sk        the best Stream-K config at M
  best_all       the best of every candidate at M (sub-tiles included)
  torch          hipBLASLt through F.linear at M, from the kernel traces (another
                 measurement: drawn for reference, never divided)
Our tiles run BLOCK_M along M and BLOCK_N along N.
"""

import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic import gemm_a16w16_streamk as sk  # noqa: E402

TRANSFER = Path("/home/jbowley/repos/aiter/streamk_study/aiter_tune/runs/transfer")
sys.path.insert(0, str(TRANSFER))
from make_rows import m_grid  # noqa: E402

P, XCDS = 256, 8
TILE_RE = re.compile(r"_(\d+)x(\d+)")
METHODS = ("tile_fixed", "sk_runtime", "sk_tile_best", "sk_fixed", "best_tile", "best_sk",
           "best_all")


def tile_of(name):
    if name.startswith("gluon_streamk_"):
        c = sk.parse_kernel_name(name)
        return c["BLOCK_M"], c["BLOCK_N"]
    bm, bn = TILE_RE.search(name).groups()
    return int(bm), int(bn)


def efficiency(M, N, tile):
    bm, bn = tile
    tiles = math.ceil(M / bm) * math.ceil(N / bn)
    pad = M * N / (tiles * bm * bn)
    slot = tiles / (math.ceil(tiles / P) * P)
    return tiles, tiles / P, pad, slot


def launch_key(M, N, K, config):
    c, launch = sk._launch_constants(M, N, K, config, P, XCDS)
    return (c.tile, c.group_size_m, c.fixup, tuple(launch.items()))


def load(run):
    prof = pd.read_csv(run / "profile.csv")
    cands = pd.read_csv(run / "candidates.csv")
    ok = prof[(prof.us > 0) & (prof.err_ratio <= 0.05)]
    ok = ok.merge(cands, on=["M", "N", "K", "kernelName"], how="left")
    return {k: dict(zip(g.kernelName, g.us)) | {"_method": dict(zip(g.kernelName, g.method))}
            for k, g in ok.groupby(["M", "N", "K"])}


def torch_times():
    m = pd.concat([pd.read_csv(TRANSFER / f) for f in ("modes.csv", "modes_large_m.csv")])
    t = m[m.variant == "torch"]
    return {(r.N, r.K, r.M0, r.M): r.us for r in t.itertuples()}


def best(at, methods):
    names = [n for n, meth in at["_method"].items() if meth in methods]
    return min(names, key=at.get) if names else None


def anchor_rows(N, K, M0, prof, torch):
    at0 = prof.get((M0, N, K))
    if at0 is None:
        print(f"skip N={N} K={K} M0={M0}: M0 not in the profile")
        return []
    tile_name = best(at0, ("base", "uneven"))
    sk_name = best(at0, ("streamk",))
    sk_cfg = sk.parse_kernel_name(sk_name)
    sk_tile = {k: sk_cfg[k] for k in sk.TILE_KEYS}
    prefix = sk_name.split("_g")[0] + f"_g{sk_cfg['GROUP_SIZE_M']}_"
    rows = []
    for M in m_grid(M0):
        at = prof.get((M, N, K))
        if at is None:
            continue
        runtime = sk.kernel_name(sk.complete_config(sk_tile, M, N, K, P))
        if sk.unsupported_reason(M, N, K, config=sk_cfg) is None:
            key = launch_key(M, N, K, sk_cfg)
            same = [n for n, meth in at["_method"].items() if meth == "streamk"
                    and sk.unsupported_reason(M, N, K, config=sk.parse_kernel_name(n)) is None
                    and launch_key(M, N, K, sk.parse_kernel_name(n)) == key]
            fixed, fallback = (same[0] if same else None), False
        else:
            fixed, fallback = runtime, True
        tile_best = [n for n in at if n.startswith(prefix)]
        names = dict(
            tile_fixed=tile_name, sk_runtime=runtime,
            sk_tile_best=min(tile_best, key=at.get) if tile_best else None,
            sk_fixed=fixed, best_tile=best(at, ("base", "uneven")),
            best_sk=best(at, ("streamk",)), best_all=best(at, ("base", "uneven", "subtile", "streamk")),
        )
        row = dict(N=N, K=K, M0=M0, M=M, sk_fixed_fallback=fallback,
                   sk_runtime_plan=sk.complete_config(sk_tile, M, N, K, P)["STREAMK_POLICY"],
                   torch=torch.get((N, K, M0, M), float("nan")))
        for meth, n in names.items():
            row[meth] = at.get(n, float("nan")) if n else float("nan")
            row[f"{meth}_name"] = n
        for meth in ("tile_fixed", "sk_runtime", "best_tile"):
            tiles, waves, pad, slot = efficiency(M, N, tile_of(names[meth]))
            row.update({f"{meth}_tiles": tiles, f"{meth}_waves": waves,
                        f"{meth}_pad": pad, f"{meth}_slot": slot, f"{meth}_eff": pad * slot})
        rows.append(row)
    return rows


def geomean(s):
    s = s.dropna()
    return math.exp(s.map(math.log).mean()) if len(s) else float("nan")


def stats(r):
    r = r.dropna()
    return dict(points=len(r), geomean=round(geomean(r), 3),
                worst=round(float(r.max()), 3) if len(r) else float("nan"),
                slower_5pct=f"{100 * (r > 1.05).mean():.0f}%" if len(r) else "")


def figure(g, path):
    g = g.set_index("M")
    N, K, M0 = g.N.iloc[0], g.K.iloc[0], g.M0.iloc[0]
    tf, st = g.tile_fixed_name.iloc[-1], g.sk_runtime_name.iloc[-1]
    fig, (ax_t, ax_r, ax_w, ax_e) = plt.subplots(
        4, 1, figsize=(9, 12), sharex=True, gridspec_kw=dict(height_ratios=[3, 2, 1.6, 2]))
    styles = dict(tile_fixed=("C0", "-", f"tile fixed at M0 ({'x'.join(map(str, tile_of(tf)))})"),
                  sk_runtime=("C2", "-", f"M0 Stream-K tile ({'x'.join(map(str, tile_of(st)))}) + runtime plan"),
                  sk_fixed=("C4", "--", "M0 Stream-K config fixed"),
                  best_tile=("C1", "-", "best tile per M (no Stream-K)"),
                  best_sk=("C3", ":", "best Stream-K config per M"))
    for meth, (c, ls, label) in styles.items():
        ax_t.plot(g.index, g[meth], color=c, ls=ls, lw=1.4, label=label)
    split = g.sk_runtime_plan != "dp"
    ax_t.scatter(g.index[split], g.sk_runtime[split], marker="^", facecolors="none",
                 edgecolors="C2", s=30, zorder=3, label="runtime plan splits")
    ax_t.plot(g.index, g.torch, color="grey", lw=1, alpha=0.6, label="torch (kernel trace, reference)")
    ax_t.set_ylabel("time (us)")
    ax_t.set_title(f"N={N} K={K}, anchor M0={M0}: our kernels")
    ax_t.legend(fontsize=7)
    ax_t.grid(alpha=0.3)
    for meth in ("tile_fixed", "sk_runtime", "sk_fixed", "best_sk"):
        c, ls, label = styles[meth]
        ax_r.plot(g.index, g[meth] / g.best_tile, color=c, ls=ls, lw=1.4, label=label)
    ax_r.axhline(1, color="grey", ls=":", lw=1)
    ax_r.set_ylabel("/ best tile per M")
    ax_r.legend(fontsize=7)
    ax_r.grid(alpha=0.3)
    for meth in ("tile_fixed", "sk_runtime", "best_tile"):
        c, _, label = styles[meth]
        ax_w.step(g.index, g[f"{meth}_waves"], where="mid", color=c, label=label.split(" (")[0])
        ax_e.plot(g.index, g[f"{meth}_pad"], color=c, ls="--", lw=1)
        ax_e.plot(g.index, g[f"{meth}_slot"], color=c, ls=":", lw=1.2)
        ax_e.plot(g.index, g[f"{meth}_eff"], color=c, lw=1.8, label=label.split(" (")[0])
    for w in range(1, int(g[[c for c in g.columns if c.endswith("_waves")]].max().max()) + 2):
        ax_w.axhline(w, color="grey", lw=0.8, ls=":")
    ax_w.set_ylabel(f"waves (tiles / {P})")
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


def main(run):
    run = Path(run)
    prof = load(run)
    torch = torch_times()
    anchors = pd.concat([pd.read_csv(TRANSFER / f)
                         for f in ("anchors_hbl.csv", "anchors_hbl_large_m.csv")])
    rows = []
    for a in anchors.itertuples():
        rows += anchor_rows(a.N, a.K, a.M0, prof, torch)
    d = pd.DataFrame(rows)
    d.to_csv(run / "transfer.csv", index=False)
    below, at_m0 = d[d.M < d.M0], d[d.M == d.M0]

    out = ["# Does our runtime Stream-K rescue a tile fixed at M0?", "",
           "Our Gluon kernels on the 12 hipBLASLt anchors (every candidate timed at every M "
           "with the tuner). Ratios are to the best tile per M (base or uneven, no Stream-K), "
           "below M0 unless stated. See analyze_transfer.py for the definitions.", ""]
    rows = []
    for meth, label in (("tile_fixed", "tile fixed at M0"),
                        ("sk_runtime", "M0 Stream-K tile + runtime plan"),
                        ("sk_tile_best", "M0 Stream-K tile + best plan at M"),
                        ("sk_fixed", "M0 Stream-K config fixed"),
                        ("best_sk", "best Stream-K config per M"),
                        ("best_all", "best of everything per M")):
        rows.append(dict(method=label, **stats(below[meth] / below.best_tile)))
    out += ["## 1. Against the best tile per M", "", pd.DataFrame(rows).to_markdown(index=False), ""]
    out += ["## 2. Does the runtime plan rescue the fixed tile?", "",
            "sk_runtime / tile_fixed below M0 (below 1: the Stream-K tile with its runtime "
            "plan beats the tile fixed at M0).", "",
            pd.DataFrame([dict(comparison="sk_runtime / tile_fixed", **stats(below.sk_runtime / below.tile_fixed)),
                          dict(comparison="sk_tile_best / tile_fixed", **stats(below.sk_tile_best / below.tile_fixed)),
                          dict(comparison="sk_runtime / sk_fixed", **stats(below.sk_runtime / below.sk_fixed))]
                         ).to_markdown(index=False), ""]
    out += ["## 3. At M0 (tuned point)", "",
            pd.DataFrame([dict(comparison="best Stream-K / best tile", **stats(at_m0.best_sk / at_m0.best_tile)),
                          dict(comparison="best of everything / best tile", **stats(at_m0.best_all / at_m0.best_tile))]
                         ).to_markdown(index=False), ""]
    per = []
    for (N, K, M0), g in below.groupby(["N", "K", "M0"]):
        per.append(dict(N=N, K=K, M0=M0,
                        tile_fixed=g.tile_fixed_name.iloc[0].replace("gluon_", ""),
                        sk_tile=g.sk_runtime_name.iloc[0].split("_g")[0].replace("gluon_streamk_", ""),
                        fixed_vs_best=round(geomean(g.tile_fixed / g.best_tile), 3),
                        sk_runtime_vs_best=round(geomean(g.sk_runtime / g.best_tile), 3),
                        sk_runtime_vs_fixed=round(geomean(g.sk_runtime / g.tile_fixed), 3),
                        worst_fixed=round((g.tile_fixed / g.best_tile).max(), 3),
                        worst_sk_runtime=round((g.sk_runtime / g.best_tile).max(), 3),
                        runtime_splits=int((g.sk_runtime_plan != "dp").sum()),
                        sk_fixed_fallbacks=int(g.sk_fixed_fallback.sum())))
    out += ["## 4. Per anchor (geomean below M0)", "", pd.DataFrame(per).to_markdown(index=False), ""]

    dp = []
    for M, N, K in d[["M", "N", "K"]].drop_duplicates().itertuples(index=False):
        at = prof.get((M, N, K), {})
        best_dp = {}
        for n, us in at.items():
            if n.startswith("gluon_streamk_") and n.endswith("_dp"):
                t = tile_of(n)
                best_dp[t] = min(best_dp.get(t, us), us)
        for (bm, bn), us in best_dp.items():
            base = at.get(f"gluon_compute_bound_{bm}x{bn}")
            if base:
                dp.append(us / base)
    out += ["## 5. Checks", "",
            f"- At M0, tile_fixed equals best_tile on {int((at_m0.tile_fixed == at_m0.best_tile).sum())} "
            f"of {len(at_m0)} anchors, sk_fixed equals best_sk on "
            f"{int((at_m0.sk_fixed == at_m0.best_sk).sum())} of {len(at_m0)}.",
            "- Stream-K's dp plan (best BLOCK_K) / base at the same tile, every shape: "
            + ", ".join(f"{k} {v}" for k, v in stats(pd.Series(dp, dtype=float)).items()) + ".",
            f"- sk_fixed fell back to the runtime plan at {int(below.sk_fixed_fallback.sum())} "
            f"of {len(below)} points below M0.", ""]
    text = "\n".join(out)
    (run / "transfer_summary.md").write_text(text + "\n")
    print(text)
    for (N, K, M0), g in d.groupby(["N", "K", "M0"]):
        figure(g, run / f"fig_transfer_N{N}_K{K}_M{M0}.png")


if __name__ == "__main__":
    main(sys.argv[1])
