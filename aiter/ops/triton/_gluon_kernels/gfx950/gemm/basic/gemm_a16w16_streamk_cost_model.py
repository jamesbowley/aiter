# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""Cost model for the gfx950 a16w16 Gluon GEMMs (compute-bound kernel and Stream-K): a
predicted time per complete candidate (tile, plan, split, exchange, partials), for ranking a
shape's candidates. The lowest prediction is the pick; the lowest few are a tuning shortlist.
See README_cost_model.md for the form, the constants and how to refit them.

    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic import gemm_a16w16_streamk_cost_model as cm
    cm.shortlist(M, N, K, k=5)          # kernel names, lowest predicted time first
    cm.predict_kernel_us(M, N, K, name) # predicted time of one kernel name (us)

The constants are effective, fitted with a ranking loss on MI355X timings: absolute predictions
are only good to about +-25%, but the ordering of a shape's candidates is what they are fitted for.
"""

from dataclasses import dataclass

P, GROUP_M, CLOCK_GHZ = 256, 4, 2.4

# Fitted on MI355X (gfx950, 256 CUs, 1400 W); refit for another part (README_cost_model.md).
C = dict(
    c=89.4349,  # per-side overhead of a K-step, in a fraction of MFMA cycles x side length
    s0=499.269,  # fixed cycles per K-step (two per K-pair)
    s_x=27.4907,  # cycles per K-step per extra (non-power-of-two) block or unequal half
    e_pf=1.06142,  # pipeline fill and drain per tile, in K-pairs of that tile
    B_hbm=4.85531,  # TB/s, effective
    p=3.33494,  # smoothing exponent of max(compute, memory)
    s_ow=1.39035,  # us: owner fixup's sync latency
    s_rs=0.271086,  # us: reduce-scatter (wt exchange) sync latency
    s_tt=12.1766,  # us: two_tile's sync latency
    b_cu=29.4911,  # GB/s: one program's partial-read rate
)


@dataclass
class Cand:
    kind: str  # "cb" (compute-bound kernel) or "sk" (Stream-K)
    bm: int
    bn: int
    bk: int = 64
    policy: str = "dp"  # dp, dp_sub, one_tile, spread, two_tile
    fixup: str = ""  # owner or rs (one_tile)
    exchange: str = ""  # wt or l2 (one_tile rs)
    split: int = 1  # one_tile's split along K
    p16: bool = False  # 16-bit partials
    readers: int = 1  # rs readers per tile
    tbm: int = 0  # dp_sub sub-tile
    tbn: int = 0
    extra: int = 0  # extra blocks + unequal halves of the tile
    extra_t: int = 0  # the same for dp_sub's sub-tile


def cdiv(a, b):
    return -(-a // b)


def hbm_bytes(n, M, N, K, bm, bn):
    """Unique HBM bytes of n consecutive tiles over full K under the GROUP_SIZE_M swizzle."""
    gm, gn = cdiv(M, bm), cdiv(N, bn)
    g = min(GROUP_M, gm)
    if n <= g * gn:
        rows, cols = min(g, n), min(gn, cdiv(n, g))
    else:
        rows, cols = min(gm, cdiv(n, gn)), gn
    return 2.0 * K * (rows * bm + cols * bn) + 2.0 * bm * bn * n


def cycles_per_pair(bm, bn, bk, extra):
    mfma = bm * bn * bk / 1024.0
    return mfma * (1 + C["c"] / bm + C["c"] / bn) + 2 * (C["s0"] + C["s_x"] * extra)


def k_loop_us(pairs, cyc, hbm):
    tc = pairs * cyc / (CLOCK_GHZ * 1e3)
    tm = hbm / (C["B_hbm"] * 1e6)
    return (tc ** C["p"] + tm ** C["p"]) ** (1 / C["p"])


def per_tile_us(cyc):
    return C["e_pf"] * cyc / (CLOCK_GHZ * 1e3)


def predict_us(M, N, K, c: Cand):
    """Predicted time (us) of candidate c at (M, N, K)."""
    T = cdiv(M, c.bm) * cdiv(N, c.bn)
    W, S = T // P, T % P
    if c.kind == "cb":
        pairs = K / 128.0
    else:
        kp = cdiv(K, 2 * c.bk) * 2 * c.bk
        pairs = kp / (2.0 * c.bk)
    cyc = cycles_per_pair(c.bm, c.bn, c.bk, c.extra)
    full_waves = 0 if T < P else (W - 1 if c.policy == "two_tile" else W)
    t = full_waves * (
        k_loop_us(pairs, cyc, hbm_bytes(P, M, N, K, c.bm, c.bn)) + per_tile_us(cyc)
    )
    tail = T if T < P else S
    if c.kind == "cb" or c.policy == "dp":
        if tail:
            t += k_loop_us(
                pairs, cyc, hbm_bytes(tail, M, N, K, c.bm, c.bn)
            ) + per_tile_us(cyc)
        return t
    if c.policy == "dp_sub":
        q = (c.bm // c.tbm) * (c.bn // c.tbn)
        cyt = cycles_per_pair(c.tbm, c.tbn, c.bk, c.extra_t)
        return (
            t
            + k_loop_us(pairs, cyt, hbm_bytes(S * q, M, N, K, c.tbm, c.tbn))
            + per_tile_us(cyt)
        )
    tile_bytes = (2 if c.p16 else 4) * c.bm * c.bn
    if c.policy == "one_tile":
        t += k_loop_us(
            cdiv(kp, 2 * c.bk * c.split), cyc, hbm_bytes(tail, M, N, K, c.bm, c.bn)
        ) + per_tile_us(cyc)
        peers = c.split - 1
        if c.fixup == "owner":
            sync = C["s_ow"]
            one, all_ = tile_bytes * peers, 2 * tile_bytes * peers * tail
        else:
            sync = C["s_rs"] if c.exchange == "wt" else 0.0
            one = tile_bytes * (1 + c.split / max(c.readers, 1))
            all_ = 2 * tile_bytes * c.split * tail
    else:  # spread, two_tile: every program takes an even share of the tail's K-pairs
        units = S if c.policy == "spread" else P + S
        t += k_loop_us(
            cdiv(int(units * (kp // (2 * c.bk))), P),
            cyc,
            hbm_bytes(units, M, N, K, c.bm, c.bn),
        )
        t += 2 * per_tile_us(cyc)
        sync = 0.0 if c.policy == "spread" else C["s_tt"]
        one, all_ = tile_bytes * P / max(units, 1), 2 * tile_bytes * P
    return t + sync + max(one / (C["b_cu"] * 1e3), all_ / (C["B_hbm"] * 1e6))


def _side_blocks(side, first):
    """(extra blocks beyond the two halves' power-of-two blocks, 1 if the halves differ)."""
    if not side:
        return 0, 0
    top = first if first > 0 else side // 2
    halves = (top, side - top)
    return sum(0 if h & (h - 1) == 0 else 1 for h in halves), int(
        halves[0] != halves[1]
    )


def candidate(M, N, K, name):
    """The Cand of a Gluon kernel name at (M, N, K); raises ValueError if it cannot run there."""
    if name.startswith("gluon_compute_bound_"):
        bm, bn = (int(x) for x in name.rsplit("_", 1)[1].split("x"))
        return Cand(kind="cb", bm=bm, bn=bn)
    from . import gemm_a16w16_streamk as sk

    config = sk.parse_kernel_name(name)
    reason = sk.unsupported_reason(M, N, K, config=config)
    if reason is not None:
        raise ValueError(reason)
    num_programs, num_xcds = sk.streamk_topology()
    c, L = sk._launch_constants(M, N, K, config, num_programs, num_xcds)
    tbm, tbn = L["TAIL_BLOCK_M"], L["TAIL_BLOCK_N"]
    xm, um = _side_blocks(c.block_m, L["TOP_M"])
    xn, un = _side_blocks(c.block_n, L["LEFT_N"])
    xtm, utm = _side_blocks(tbm, L["TAIL_TOP_M"]) if tbm else (0, 0)
    xtn, utn = _side_blocks(tbn, L["TAIL_LEFT_N"]) if tbn else (0, 0)
    return Cand(
        kind="sk",
        bm=c.block_m,
        bn=c.block_n,
        bk=c.block_k,
        policy=c.policy,
        fixup=c.fixup,
        exchange=c.exchange,
        split=L["RESOLVED_TAIL_SPLIT_K"],
        p16=bool(L["PARTIALS_16BIT"]),
        readers=L["RS_R"],
        tbm=tbm,
        tbn=tbn,
        extra=xm + xn + um + un,
        extra_t=xtm + xtn + utm + utn,
    )


def predict_kernel_us(M, N, K, name):
    return predict_us(M, N, K, candidate(M, N, K, name))


def menu(M, N, K):
    """Every Gluon kernel the model ranks for (M, N, K): the compute-bound tiles, and Stream-K at
    the tuner's tiles and every non-power-of-two tile, with fp32 and 16-bit partials."""
    from . import gemm_a16w16 as base
    from . import gemm_a16w16_streamk as sk

    names = [
        f"gluon_compute_bound_{bm}x{bn}"
        for bm in (64, 128, 256)
        for bn in (64, 128, 256)
        if base.unsupported_reason(M, N, K, tile=(bm, bn)) is None
    ]
    if sk.unsupported_reason(M, N, K) is None:
        uneven = [t for t in sk.menu_tiles() if sk.TileSpec(*t).uneven]
        for tiles in (None, uneven):
            names += [
                sk.kernel_name(c)
                for c in sk.tuning_candidates(M, N, K, partials_16bit=True, tiles=tiles)
            ]
    return list(dict.fromkeys(names))


def shortlist(M, N, K, k=5):
    """The k kernel names with the lowest predicted time at (M, N, K), lowest first."""
    scored = []
    for name in menu(M, N, K):
        try:
            scored.append((predict_kernel_us(M, N, K, name), name))
        except ValueError:
            continue
    return [n for _, n in sorted(scored)[:k]]


if __name__ == "__main__":
    import sys

    M, N, K = (int(x) for x in sys.argv[1:4])
    for name in shortlist(M, N, K, int(sys.argv[4]) if len(sys.argv) > 4 else 5):
        print(f"{predict_kernel_us(M, N, K, name):10.1f} us  {name}")
