"""Every candidate of every method for each shape of a run (exhaustive, no pruning).

    base     gemm_a16w16.py at each power-of-two tile (gluon_compute_bound_<BM>x<BN>)
    uneven   gemm_a16w16_uneven.py at each tile with a side that is not a power of
             two (gluon_uneven_<BM>x<BN>); its power-of-two tiles build the base's code
    subtile  gemm_a16w16_subtile.py: each power-of-two tile with a partial last wave,
             with each legal sub-tile (gluon_subtile_<BM>x<BN>_tail<TM>x<TN>)
    streamk  gemm_a16w16_streamk.py's own tuning candidates (gluon_streamk_...)

The gfx1250 rule's sub-tile (PR #7364 _pick_split: double the M split, then the N
split, while the sub-tiles fit one wave and keep a side of at least 64) is marked
gfx1250_rule = True on its subtile row.

    python candidates.py runs/<name>     # reads shapes.csv, writes candidates.csv
"""

import sys
from pathlib import Path

import pandas as pd

from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic import gemm_a16w16 as base
from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic import gemm_a16w16_streamk as sk
from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic import gemm_a16w16_subtile as st
from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic import gemm_a16w16_uneven as un
from aiter.tuned_gemm import gluon_kernel_name

P = 256
POW2 = (64, 128, 256)


def gfx1250_tail(tile, s, floor=64):
    bm, bn = tile
    sm = sn = 1
    while s * sm * sn * 2 <= P and bm % (sm * 2) == 0 and bm // (sm * 2) >= floor:
        sm *= 2
    while s * sm * sn * 2 <= P and bn % (sn * 2) == 0 and bn // (sn * 2) >= floor:
        sn *= 2
    return (bm // sm, bn // sn) if sm * sn > 1 else None


def candidates(M, N, K):
    rows = []
    for bm in POW2:
        for bn in POW2:
            if base.unsupported_reason(M, N, K, tile=(bm, bn)) is None:
                rows.append((gluon_kernel_name("compute_bound", (bm, bn)), "base", False))
    for bm in un.TILE_SIDES:
        for bn in un.TILE_SIDES:
            if bm in POW2 and bn in POW2:
                continue
            if un.unsupported_reason(M, N, K, tile=(bm, bn)) is None:
                rows.append((gluon_kernel_name("uneven", (bm, bn)), "uneven", False))
    for bm in POW2:
        for bn in POW2:
            if st.unsupported_reason(M, N, K, tile=(bm, bn)) is not None:
                continue
            s, _ = st.tail_plan(M, N, (bm, bn), (64, 64), P)
            if s == 0:
                continue
            rule = gfx1250_tail((bm, bn), s)
            for tm in POW2:
                for tn in POW2:
                    if (tm, tn) == (bm, bn) or bm % tm or bn % tn:
                        continue
                    if st.unsupported_tail_reason(M, N, (bm, bn), (tm, tn), P):
                        continue
                    name = gluon_kernel_name("subtile", (bm, bn), (tm, tn))
                    rows.append((name, "subtile", rule == (tm, tn)))
    if sk.unsupported_reason(M, N, K) is None:
        for c in sk.tuning_candidates(M, N, K):
            rows.append((sk.kernel_name(c), "streamk", False))
    return rows


def main(run):
    run = Path(run)
    shapes = pd.read_csv(run / "shapes.csv")
    out = []
    for M, N, K in shapes[["M", "N", "K"]].drop_duplicates().itertuples(index=False):
        for name, method, rule in candidates(int(M), int(N), int(K)):
            out.append((M, N, K, name, method, rule))
    c = pd.DataFrame(out, columns=["M", "N", "K", "kernelName", "method", "gfx1250_rule"])
    c = c.drop_duplicates(["M", "N", "K", "kernelName"])
    c.to_csv(run / "candidates.csv", index=False)
    print(c.groupby("method").size().to_string())
    print(f"{len(c)} candidates over {len(shapes)} shapes")


if __name__ == "__main__":
    main(sys.argv[1])
