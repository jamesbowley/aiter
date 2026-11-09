"""Report on att_tiles.sh: main-loop MFMA issue efficiency per tile (median and range over the
traced shapes), and a test of whether it can choose the tile: a model that predicts each
candidate's time as waves x tile area / efficiency, scored against the measured times of the
grid (grid_bench.py) and of the main run's tuning profile.

    python att_tiles_report.py runs/att_tiles      # writes att_tiles.md there
"""

import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import analyze_grid as A
import issue_figs as F

P = 256


def tile_of(name):
    m = re.search(r"gluon_(compute_bound|uneven)_(\d+)x(\d+)$", name)
    if m:
        return int(m[2]), int(m[3]), 64
    m = re.search(r"gluon_streamk_(\d+)x(\d+)x(\d+)_", name)
    return (int(m[1]), int(m[2]), int(m[3])) if m else (None, None, None)


# VGPRs (arch + accumulator) from a kernel trace of the eight power-of-two base tiles
# (runs/att_doc); other tiles are estimated from the tile area, fitted to those.
VGPR = {(256, 256): 456, (128, 256): 352, (256, 128): 352, (128, 128): 168, (256, 64): 200, (64, 256): 192,
        (128, 64): 112, (64, 128): 112}


def wg_per_cu(r):
    """Workgroups resident per CU: the persistent Stream-K kernel runs one; the others as many as
    LDS (two stages of A and B, ~3% padding), VGPRs and the grid allow."""
    if r.method == "streamk":
        return 1
    lds = 2 * (r.BM + r.BN) * 64 * 2 * 1.03
    vgpr = VGPR.get((r.BM, r.BN), 0.0055 * r.BM * r.BN + 100)
    wgs = math.ceil(r.M / r.BM) * math.ceil(r.N / r.BN)
    return int(max(1, min(163840 // lds, 512 // (8 * math.ceil(vgpr / 8)), 8, math.ceil(wgs / P))))


def main(run):
    run = Path(run)
    d = pd.read_csv(run / "att_tiles.csv")
    d = d[d.loop == "main"].copy()
    d[["BM", "BN", "BK"]] = pd.DataFrame(d.kernelName.map(tile_of).tolist(), index=d.index)
    d["intensity"] = d.BM * d.BN / (d.BM + d.BN)
    # A trace records waves; with several workgroups per CU (one wave per SIMD each) the SIMD's
    # matrix pipe is shared, so per-SIMD efficiency = per-wave x workgroups per CU.
    d["wg_per_cu"] = d.apply(wg_per_cu, axis=1)
    d["eff_wave"] = d.mfma_issue_eff
    d["mfma_issue_eff"] = d.mfma_issue_eff * d.wg_per_cu
    kind = {"base": "base", "uneven": "finer", "streamk": "Stream-K, no split"}
    d["kind"] = d.method.map(kind)
    g = d.groupby(["kind", "BM", "BN", "BK"]).agg(
        intensity=("intensity", "first"), eff=("mfma_issue_eff", "median"), lo=("mfma_issue_eff", "min"),
        hi=("mfma_issue_eff", "max"), eff_wave=("eff_wave", "median"), wg=("wg_per_cu", "max"), lds=("lat_lds", "median"), vmem=("lat_vmem", "median"),
        waitcnt=("lat_waitcnt", "median"), n=("mfma_issue_eff", "size")).reset_index()
    g = g.sort_values(["kind", "eff"], ascending=[True, False])
    lines = [f"# Main-loop MFMA issue efficiency per tile, per SIMD ({d.tag.nunique()} traces, shapes: "
             + ", ".join(f"{a}x{b}x{c}" for a, b, c in d[["M", "N", "K"]].drop_duplicates().itertuples(index=False)) + ")", ""]
    for k, q in g.groupby("kind", sort=False):
        rows = [{"tile": f"{r.BM}x{r.BN}" + (f" (BK {r.BK})" if k.startswith("Stream") else ""), "intensity": f"{r.intensity:.0f}",
                 "workgroups per CU": r.wg, "efficiency (median)": f"{100 * r.eff:.1f}%", "range": f"{100 * r.lo:.1f}-{100 * r.hi:.1f}%",
                 "LDS / VMEM / waitcnt share of loop time": f"{100 * r.lds:.0f} / {100 * r.vmem:.0f} / {100 * r.waitcnt:.0f}%"}
                for r in q.itertuples()]
        lines += [f"## {k}", "", A.md(pd.DataFrame(rows)), ""]
    g.to_csv(run / "att_tiles_summary.csv", index=False)

    # Stream-K's main loop against the base's at the same tile
    b = g[g.kind == "base"].set_index(["BM", "BN"]).eff
    s = g[(g.kind == "Stream-K, no split") & (g.BK == 64)].set_index(["BM", "BN"]).eff
    j = pd.concat([b.rename("base"), s.rename("sk")], axis=1).dropna()
    lines += ["Stream-K (BK 64) against the base at the same tile: " + ", ".join(
        f"{bm}x{bn} {100 * r.sk:.1f}% vs {100 * r.base:.1f}%" for (bm, bn), r in j.iterrows()), ""]

    # Efficiency against intensity
    q = g[g.kind != "Stream-K, no split"]
    c = np.corrcoef(q.intensity, q.eff)[0, 1]
    lines += [f"Correlation of efficiency with intensity (base and finer tiles): {c:.2f}.", ""]

    # Can efficiency choose the tile? Model: time ~ waves x BM x BN / efficiency(tile).
    eff = {(r.BM, r.BN): r.eff for r in g[g.kind != "Stream-K, no split"].itertuples()}
    eff_wave = {(r.BM, r.BN): r.eff_wave for r in g[g.kind != "Stream-K, no split"].itertuples()}

    def score(times, label):
        t = times[times.kernelName.str.match(r"gluon_(compute_bound|uneven)_\d+x\d+$")].copy().reset_index(drop=True)
        t[["BM", "BN", "BK"]] = pd.DataFrame(t.kernelName.map(tile_of).tolist(), index=t.index)
        t = t[[(a, b2) in eff for a, b2 in zip(t.BM, t.BN)]]
        tiles = np.ceil(t.M / t.BM) * np.ceil(t.N / t.BN)
        # With w workgroups per CU, each round runs w tiles at once on the shared SIMDs.
        t["method"] = "base"
        w = t.apply(wg_per_cu, axis=1)
        rounds = np.ceil(tiles / (P * w))
        t["pred_eff"] = rounds * w * t.BM * t.BN / [eff[(a, b2)] for a, b2 in zip(t.BM, t.BN)]
        t["pred_wave"] = np.ceil(tiles / P) * t.BM * t.BN / [eff_wave[(a, b2)] for a, b2 in zip(t.BM, t.BN)]
        t["pred_flat"] = np.ceil(tiles / P) * t.BM * t.BN
        out = []
        for name, col in (("rounds x workgroups per CU x area / per-SIMD efficiency", "pred_eff"), ("waves x area / per-wave efficiency", "pred_wave"), ("waves x area (efficiency ignored)", "pred_flat")):
            pick = t.loc[t.groupby(A.KEY)[col].idxmin()].set_index(A.KEY).us
            best = t.groupby(A.KEY).us.min()
            r = (pick / best).dropna()
            out.append({"set": label, "model": name, "shapes": len(r), "loss vs best tile (geomean)": f"{100 * (A.gm(r) - 1):.1f}%",
                        "picks within 1% of best": f"{100 * (r <= 1.01).mean():.0f}%", "worst": f"{r.max():.2f}"})
        return out

    rows = []
    gb = pd.concat([A.load_bench(Path("runs/grid/bench_a.csv")), A.load_bench(Path("runs/grid_pad/bench_a.csv"))])
    rows += score(gb, "grid (912 shapes)")
    prof = pd.read_csv("runs/full/profile.csv")
    prof = prof[(prof.us > 0) & (prof.err_ratio <= 0.05)]
    res = pd.read_csv("runs/full/results.csv")
    main_set = res[(res.set == "main") & (res.M >= F.MIN_M)][A.KEY]
    prof["bias"] = prof.bias.astype(str).str.lower().eq("true")
    rows += score(prof.merge(main_set, on=A.KEY), f"AITER shapes ({len(main_set)}), tuning profile")
    lines += ["## Choosing the tile from efficiency (base and finer tiles only)", "", A.md(pd.DataFrame(rows)), ""]
    text = "\n".join(lines) + "\n"
    (run / "att_tiles.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main(sys.argv[1])
