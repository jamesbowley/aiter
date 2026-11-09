"""The issue reply's numbers that are not in tables.md (issue_figs.py) or grid.md / confirm.md
(analyze_grid.py), for the main set (M > 128), written to runs/full/reply_numbers.md.

    python reply_numbers.py
"""

import math
import zlib
from pathlib import Path

import msgpack
import numpy as np
import pandas as pd

import analyze_grid as A
import issue_figs as F

V2 = Path(__file__).resolve().parent
RUN = V2 / "runs/full"
P = 256
gm = A.gm


def md(df):
    return A.md(df)


def main():
    d = pd.read_csv(RUN / "results.csv")
    d = F.load_torch(RUN, d)
    m = d[(d.set == "main") & (d.M >= F.MIN_M)].copy()
    m["split+base"] = m[["streamk", "base"]].min(axis=1)
    m = A.add_cols(m)
    out = [f"# Reply numbers ({len(m)} shapes, M > 128)", ""]

    # The experiment
    out += [f"K <= 1024: {(m.K <= 1024).sum()} shapes; K > 8192: {(m.K > 8192).sum()}; "
            f"base pick 256x256: {(m.base_kernel == 'gluon_compute_bound_256x256').sum()}", ""]

    # Persistent no-split plan against whole tiles at 256x256 (tuning profile)
    prof = pd.read_csv(RUN / "profile.csv")
    prof = prof[(prof.us > 0) & (prof.err_ratio <= 0.05)].merge(m[F.KEY], on=F.KEY)
    w = prof[prof.kernelName == "gluon_compute_bound_256x256"].groupby(F.KEY).us.min()
    dp = prof[prof.kernelName.str.startswith("gluon_streamk_256x256x64_") & prof.kernelName.str.endswith("_dp")].groupby(F.KEY).us.min()
    j = pd.concat([w.rename("w"), dp.rename("dp")], axis=1).dropna().reset_index()
    rows = []
    for lab, lo, hi in (("K <= 1024", 0, 1024), ("1024 < K <= 2048", 1024, 2048), ("K > 2048", 2048, 10**9),
                        ("2048 < K <= 4096", 2048, 4096), ("4096 < K <= 8192", 4096, 8192), ("K > 8192", 8192, 10**9)):
        q = j[(j.K > lo) & (j.K <= hi)]
        rows.append({"K": lab, "persistent 256x256 vs whole 256x256": f"{gm(q.w / q.dp):.3f}", "n": len(q)})
    out += ["## Persistent no-split plan vs whole tiles, both 256x256 (tuning profile)", "", md(pd.DataFrame(rows)), ""]

    # Section 2: rules by region
    B, Fn, S, K = m.base, m["uneven+base"], m["subtile+base"], m["split+base"]
    best = np.minimum(np.minimum(Fn, S), K)
    small = m.S <= 128
    rules = {
        "finer tiles everywhere": Fn, "sub-tile tail everywhere": S, "Stream-K everywhere": K,
        "finer, Stream-K K>4096 S<=P/2": pd.Series(np.where(small & (m.K > 4096), K, Fn), index=m.index),
        "+ sub-tile K<=2048 S<=P/2": pd.Series(np.where(~small, Fn, np.where(m.K <= 2048, S, np.where(m.K <= 4096, Fn, K))), index=m.index),
        "best per shape": best}
    regions = {"all": m.K > 0, "S<=P/2, K<=2048": small & (m.K <= 2048), "of which S<=P/4": (m.S <= 64) & (m.K <= 2048),
               "S<=P/2, 2048<K<=4096": small & (m.K > 2048) & (m.K <= 4096), "S<=P/2, K>4096": small & (m.K > 4096), "S>P/2": ~small}
    rows = []
    for name, t in rules.items():
        r = {"rule": name}
        for reg, sel in regions.items():
            r[f"{reg} ({int(sel.sum())})"] = f"{gm(B[sel] / t[sel]):.3f}"
        rows.append(r)
    out += ["## Section 2 rules by region", "", md(pd.DataFrame(rows)), ""]
    s2 = small & (m.K <= 2048)
    out += [f"sub-tile vs finer, S<=P/2 K<=2048: {gm(Fn[s2] / S[s2]):.4f}; S<=P/4 K<=2048: "
            f"{gm(Fn[(m.S <= 64) & (m.K <= 2048)] / S[(m.S <= 64) & (m.K <= 2048)]):.4f}; "
            f"2048<K<=4096 S<=P/2 finer vs sub: {gm(S[regions['S<=P/2, 2048<K<=4096']] / Fn[regions['S<=P/2, 2048<K<=4096']]):.4f}", ""]

    # Section 2: fig3 cells (leads)
    rows = []
    for f in F.FILL_ORDER:
        for kb in F.K_ORDER:
            q = m[(m.fill_bin == f) & (m.k_bin == kb)]
            if not len(q):
                continue
            g = {n: gm(q.base / q[c]) for n, c in A.COL.items()}
            o = sorted(g, key=g.get, reverse=True)
            rows.append({"fill": f, "K": kb, "n": len(q), **{n: f"{v:.3f}" for n, v in g.items()},
                         "winner": o[0], "lead": f"{100 * (g[o[0]] / g[o[1]] - 1):.1f}%"})
    out += ["## Figure 3 cells", "", md(pd.DataFrame(rows)), ""]

    # Section 2: persistent loop vs split
    persist = m[["base", "streamk_dp"]].min(axis=1)
    rows = []
    for kb in F.K_ORDER:
        q = small & (m.k_bin == kb)
        rows.append({"K": kb, "persistent vs base": f"{gm(B[q] / persist[q]):.3f}",
                     "split vs persistent": f"{gm(persist[q] / K[q]):.3f}",
                     "persistent >3% faster than base": f"{100 * (B[q] / persist[q] > 1.03).mean():.0f}%", "n": int(q.sum())})
    out += ["## Persistent loop vs split, S<=P/2", "", md(pd.DataFrame(rows)), ""]

    # Section 4: hipBLASLt by region (drift-adjusted, as tables.md)
    k = pd.read_csv(RUN / "torch_kernels.csv")
    q = m.merge(k[F.KEY + ["hbl_bm_n", "hbl_bn_m", "hbl_tiles", "hbl_wgs"]], on=F.KEY)
    drift = q.base_now / q.base
    rows = []
    for lab, sel in (("S<=P/2, K<=4096", (q.S <= 128) & (q.K <= 4096)), ("S<=P/2, K>4096", (q.S <= 128) & (q.K > 4096)),
                     ("S>P/2, K<=4096", (q.S > 128) & (q.K <= 4096)), ("S>P/2, K>4096", (q.S > 128) & (q.K > 4096))):
        r = {"region": lab, "n": int(sel.sum())}
        for nm, col in (("base", "base"), ("finer", "uneven+base"), ("sub-tile", "subtile+base"), ("Stream-K", "split+base")):
            r[nm] = f"{gm(q.torch[sel] / (q[col][sel] * drift[sel])):.3f}"
        rows.append(r)
    out += ["## hipBLASLt by region (t_hipBLASLt / t_ours)", "", md(pd.DataFrame(rows)), ""]
    reg = q[(q.S <= 128) & (q.K > 4096)]
    area = reg.hbl_bm_n * reg.hbl_bn_m
    useful = reg.M * reg.N / (np.ceil(reg.hbl_tiles / P) * P * area)
    out += [f"S<=P/2, K>4096: hipBLASLt tile > 256x256 on {(area > 65536).sum()} of {len(reg)}; "
            f"median useful share {useful.median():.2f} (256x256: {(reg.M * reg.N / (np.ceil(reg['T'] / P) * P * 65536)).median():.2f}); "
            f"median intensity {np.median(reg.hbl_bm_n * reg.hbl_bn_m / (reg.hbl_bm_n + reg.hbl_bn_m)):.0f}", ""]
    frac = q[q.hbl_tiles % q.hbl_wgs != 0]
    out += ["Fractional-grid shapes: " + "; ".join(f"{r.M}x{r.N}x{r.K} (S={r.S}, {r.hbl_tiles}/{r.hbl_wgs})" for r in frac.itertuples()), ""]

    # Section 4: shapes in hipBLASLt's tuned tables
    base = "/opt/venv/lib/python3.12/site-packages/_rocm_sdk_libraries/lib/hipblaslt/library/gfx950/TensileLibrary_BB_BB_HA_Bias_SAV_UA_Type_BB_HPA_Contraction_l_Alik_Bljk_Cijk_Dijk_"
    keys = set()
    for tag in ("ID75a3-75a2", "CU256_ID75a0", "ID75a8"):
        dd = msgpack.unpackb(zlib.decompress(open(base + tag + "_gfx950.dat.zlib", "rb").read()), raw=False, strict_map_key=False)
        for row in dd["library"]["rows"]:
            for e in row["library"].get("table", []):
                keys.add(tuple(e["key"]))
    hit = sum(((r.N, r.M, 1, r.K) in keys) or ((r.M, r.N, 1, r.K) in keys) for r in m.itertuples())
    out += [f"Main shapes in hipBLASLt's tuned tables (either key order): at most {hit} of {len(m)}", ""]

    # Section 6: padding table on AITER shapes
    h = m[m.S <= 128].copy()
    h["sub"] = h["uneven+base"] / h["subtile+base"]
    h["sk"] = h["uneven+base"] / h["split+base"]
    h["pb"] = pd.cut(h["pad"], [-0.01, 0.0, 0.02, 0.10, 1], labels=["none", "<=2%", "2-10%", ">10%"])
    t = h.groupby(["pb", h.K > 4096], observed=True).agg(sub=("sub", gm), sk=("sk", gm), n=("sub", "size")).round(3)
    out += ["## AITER, S<=P/2: finer over tail method by padding share", "", t.to_string(), ""]
    tailbest = h[["subtile+base", "split+base"]].min(axis=1)
    out += [f"pad > 10%: {(h['pad'] > 0.1).sum()} shapes, finer vs best tail {gm(tailbest[h['pad'] > 0.1] / h['uneven+base'][h['pad'] > 0.1]):.3f}", ""]
    text = "\n".join(out) + "\n"
    (RUN / "reply_numbers.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
