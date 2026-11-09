"""Candidate kernelNames per shape for the uneven-tile against sub-tile-tail experiment.

Reads the stock retune (../aiter_tune/runs/stock_all: profile.csv has every power-of-two
data-parallel and Stream-K candidate's time on every shape, cmp_all/compare_all.csv the
shipped kernel, its tile and our previous pick) and writes, for each supported shape:

- base_*     our previous pick (tuned.csv), the best power-of-two Stream-K split and the
             best power-of-two data-parallel launch, re-timed in the same session;
- uneven     the top UNEVEN_TOP non-power-of-two data-parallel tiles by predicted time,
             plus the shipped tile (both orientations), or the nearest tile we can run;
- subtile    the top SUBTILE_TOP (main tile, sub-tile tail) pairs over every
             power-of-two main tile with a full wave and a tail, plus the best pair at our
             previous pick's tile.

Predicted times come from this shape's measured data-parallel times: a tile's time per
wave t_wave = t_dp / ceil(T / P), its per-CU rate 2 BM BN K_pad / t_wave, interpolated in
log(BM BN / (BM + BN)) for tiles that were not measured. A sub-tile split's fixup is the
main tile's measured split at the same n minus its modelled MAC time, scaled by the
sub-tile's area.

    python candidates.py [--pilot 150] [--out runs/<name>] [--calibrate runs/pilot]

--calibrate fits correction factors from a finished run (its candidates' predictions
against its tuning profile) and applies them to the ranking: per uneven tile (bm, bn),
measured / predicted (the global median for tiles that run did not time); per sub-tile
tail ending (dp, owner n, rs n), the measured tail time (total minus the full waves)
over the predicted one. They are written to <out>/calibration.json.

Writes <out>/candidates.csv (M, N, K, kernelName, method, pred_us), <out>/untuned.csv
(the tuner's input) and <out>/shapes.csv (one row per shape with the labels the
analysis uses).
"""

import argparse
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STOCK = HERE.parent / "aiter_tune" / "runs" / "stock_all"
P = 256
NUM_XCDS = 8
UNEVEN_TOP = 6
SUBTILE_TOP = 7
SIDES_UNEVEN = range(64, 321, 32)
MAX_ACC = 256 * 256
POW2 = (64, 128, 256)
SUB_SPLITS = (("dp", None, None), ("one_tile", "owner", 2), ("one_tile", "rs", 4), ("one_tile", "rs", 8))
DEFAULT_FIXUP_US = 3.0

SK_NAME = re.compile(
    r"^gluon_streamk_(?P<bm>\d+)x(?P<bn>\d+)x(?P<bk>\d+)_g(?P<g>\d+)_"
    r"(?:(?P<whole>dp|two_tile|spread)|one_tile_(?P<fixup>owner|rs)_n(?P<n>\d+)_(?P<ex>wt|l2))"
    r"(?:_tail(?P<tm>\d+)x(?P<tn>\d+))?$"
)
CB_NAME = re.compile(r"^gluon_compute_bound_(?P<bm>\d+)x(?P<bn>\d+)$")


def _aiter_modules():
    """aiter's Stream-K and compute-bound modules with a fixed 256-CU, 8-XCD topology,
    so candidates can be generated without a GPU."""
    sys.path.insert(0, str(HERE.parents[1]))
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic import gemm_a16w16 as cb
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic import gemm_a16w16_streamk as sk

    sk.streamk_topology = lambda: (P, NUM_XCDS)
    return sk, cb


def parse_name(name):
    """(kind, bm, bn, bk, plan, tail) of a gluon kernelName."""
    m = CB_NAME.match(name)
    if m:
        return "cb", int(m["bm"]), int(m["bn"]), 64, ("dp", None, None), None
    m = SK_NAME.match(name)
    if not m:
        return None
    plan = (m["whole"], None, None) if m["whole"] else ("one_tile", m["fixup"], int(m["n"]))
    tail = (int(m["tm"]), int(m["tn"])) if m["tm"] else None
    return "sk", int(m["bm"]), int(m["bn"]), int(m["bk"]), plan + (m["ex"],), tail


def tiles(M, N, bm, bn):
    return math.ceil(M / bm) * math.ceil(N / bn)


def k_pad(K, bk):
    return math.ceil(K / (2 * bk)) * 2 * bk


def sk_name(bm, bn, bk, policy, fixup=None, n=None, exchange="wt", tail=None):
    name = f"gluon_streamk_{bm}x{bn}x{bk}_g4_"
    name += policy if policy != "one_tile" else f"one_tile_{fixup}_n{n}_{exchange}"
    if tail and tail != (bm, bn):
        name += f"_tail{tail[0]}x{tail[1]}"
    return name


def sk_config(bm, bn, bk, policy, fixup=None, n=None, tail=None):
    c = {
        "BLOCK_M": bm,
        "BLOCK_N": bn,
        "BLOCK_K": bk,
        "GROUP_SIZE_M": 4,
        "STREAMK_POLICY": policy,
        "STREAMK_FIXUP": fixup or "owner",
        "SPLIT_N": n,
        "STREAMK_EXCHANGE": "wt",
    }
    if tail and tail != (bm, bn):
        c.update(TAIL_BLOCK_M=tail[0], TAIL_BLOCK_N=tail[1])
    return c


CAL = {"uneven": {}, "uneven_default": 1.0, "tail": {}, "tail_default": 1.0}


def tail_kind(policy, fixup, n):
    return "dp" if policy == "dp" else f"{fixup}{n}"


def calibrate(run, by_shape):
    """Correction factors from a finished run (see the module docstring); by_shape is the
    stock profile grouped by (M, N, K, bias), to rebuild each sub-tile pair's full waves."""
    c = pd.read_csv(run / "candidates.csv")
    p = pd.read_csv(run / "profile.csv")
    p = p[p.us > 0].groupby(["M", "N", "K", "kernelName"], as_index=False).us.min()
    d = c.merge(p, on=["M", "N", "K", "kernelName"])
    cal = {"uneven": {}, "tail": {}}
    u = d[d.method == "uneven"].copy()
    u["tile"] = u.kernelName.str.extract(r"_(\d+x\d+)$")[0]
    u["ratio"] = u.us / u.pred_us
    for tile, g in u.groupby("tile"):
        if len(g) >= 2:
            cal["uneven"][tile] = float(g.ratio.median())
    cal["uneven_default"] = float(u.ratio.median())
    models = {}
    rows = []
    for r in d[d.method.str.startswith("subtile")].itertuples():
        parsed = parse_name(r.kernelName)
        if parsed is None:
            continue
        _, bm, bn, bk, plan, _ = parsed
        key = next((k for k in ((r.M, r.N, r.K, False), (r.M, r.N, r.K, True)) if k in by_shape), None)
        if key is None:
            continue
        if key not in models:
            models[key] = ShapeModel(r.M, r.N, r.K, by_shape[key])
        full = (tiles(r.M, r.N, bm, bn) // P) * models[key].t_wave(bm, bn, bk)
        pred_tail, tail = r.pred_us - full, r.us - full
        if pred_tail > 0 and tail > 0:
            rows.append((tail_kind(plan[0], plan[1], plan[2]), tail / pred_tail))
    t = pd.DataFrame(rows, columns=["kind", "ratio"])
    for kind, g in t.groupby("kind"):
        cal["tail"][kind] = float(g.ratio.median())
    cal["tail_default"] = float(t.ratio.median()) if len(t) else 1.0
    return cal


def valid_uneven(cb, bm, bn):
    return cb.unsupported_reason(4096, 4096, 4096, tile=(bm, bn)) is None


class ShapeModel:
    """Per-shape wave times and rates from the stock profile."""

    def __init__(self, M, N, K, rows):
        self.M, self.N, self.K = M, N, K
        self.dp = {}  # (bm, bn, bk) -> best measured data-parallel us
        self.split = {}  # (bm, bn, bk, fixup, n) -> best measured one_tile us (wt or l2)
        self.best_split = None  # (us, name) of the best non-dp Stream-K launch
        self.best_dp = None
        for name, us in zip(rows.kernelName, rows.us):
            if not us or us <= 0:
                continue
            parsed = parse_name(name)
            if parsed is None:
                continue
            kind, bm, bn, bk, plan, _ = parsed
            if plan[0] == "dp":
                key = (bm, bn, bk)
                self.dp[key] = min(self.dp.get(key, math.inf), us)
                if self.best_dp is None or us < self.best_dp[0]:
                    self.best_dp = (us, name)
            else:
                if self.best_split is None or us < self.best_split[0]:
                    self.best_split = (us, name)
                if plan[0] == "one_tile":
                    key = (bm, bn, bk, plan[1], plan[2])
                    self.split[key] = min(self.split.get(key, math.inf), us)
        points = {}
        for (bm, bn, bk), us in self.dp.items():
            if bk != 64:
                continue
            h = bm * bn / (bm + bn)
            points.setdefault(h, []).append(self.rate(bm, bn, bk, us))
        self.h = np.array(sorted(points))
        self.r = np.array([np.mean(points[h]) for h in sorted(points)])

    def waves(self, bm, bn):
        return math.ceil(tiles(self.M, self.N, bm, bn) / P)

    def rate(self, bm, bn, bk, us):
        """Per-CU MACs (as FLOPs) per us of one wave of this tile."""
        return 2 * bm * bn * k_pad(self.K, bk) / (us / self.waves(bm, bn))

    def t_wave(self, bm, bn, bk=64):
        """Measured, else interpolated, time of one wave of (bm, bn) tiles."""
        if (bm, bn, bk) in self.dp:
            return self.dp[(bm, bn, bk)] / self.waves(bm, bn)
        if (bm, bn, 64) in self.dp:
            return self.dp[(bm, bn, 64)] / self.waves(bm, bn)
        if len(self.h) == 0:
            return math.nan
        h = bm * bn / (bm + bn)
        r = float(np.interp(math.log(h), np.log(self.h), self.r))
        return 2 * bm * bn * k_pad(self.K, bk) / r

    def t_uneven(self, bm, bn, calibrated=True):
        t = self.waves(bm, bn) * self.t_wave(bm, bn)
        if calibrated and not (bm in POW2 and bn in POW2):
            t *= CAL["uneven"].get(f"{bm}x{bn}", CAL["uneven_default"])
        return t

    def fixup(self, bm, bn, bk, fixup, n, area_ratio):
        """The measured fixup of the main tile's split at n (time beyond its modelled
        MAC time), scaled by the sub-tile's share of the tile's area."""
        key = (bm, bn, bk, fixup, n)
        T = tiles(self.M, self.N, bm, bn)
        W = T // P
        tw = self.t_wave(bm, bn, bk)
        if key in self.split and not math.isnan(tw):
            f = self.split[key] - W * tw - tw / n
            return max(0.0, f) * area_ratio
        return DEFAULT_FIXUP_US * area_ratio


def uneven_candidates(model, cb, shipped):
    """(name, pred_us) of the top uneven data-parallel tiles and the shipped tile."""
    scored = []
    for bm in SIDES_UNEVEN:
        for bn in SIDES_UNEVEN:
            if bm in POW2 and bn in POW2:
                continue
            if bm * bn > MAX_ACC or not valid_uneven(cb, bm, bn):
                continue
            if cb.unsupported_reason(model.M, model.N, model.K, tile=(bm, bn)) is not None:
                continue
            scored.append((model.t_uneven(bm, bn), bm, bn))
    scored.sort()
    out = [(f"gluon_compute_bound_{bm}x{bn}", t, "uneven") for t, bm, bn in scored[:UNEVEN_TOP]]
    have = {(bm, bn) for _, bm, bn in scored}
    if shipped is not None:
        for bm, bn in {shipped, shipped[::-1]}:
            if (bm, bn) in have or (bm in POW2 and bn in POW2):
                tile = (bm, bn)
            elif have:
                # Nearest tile we can run: closest sides, the smaller area on a tie.
                tile = min(have, key=lambda t: (abs(t[0] - bm) + abs(t[1] - bn), t[0] * t[1]))
            else:
                continue
            if cb.unsupported_reason(model.M, model.N, model.K, tile=tile) is None:
                out.append(
                    (f"gluon_compute_bound_{tile[0]}x{tile[1]}", model.t_uneven(*tile), "shipped_tile")
                )
    return out


def subtile_candidates(model, sk, previous_tile):
    """(name, pred_us) of the top (main tile, sub-tile tail) pairs, plus the best pair at
    the previous pick's tile."""
    M, N, K = model.M, model.N, model.K
    scored = []
    for bm, bn, bk in sk.TUNING_TILES:
        T = tiles(M, N, bm, bn)
        W, S = divmod(T, P)
        if W < 1 or S == 0:
            continue
        tw_main = model.t_wave(bm, bn, bk)
        if math.isnan(tw_main):
            continue
        pairs = k_pad(K, bk) // (2 * bk)
        for tm in POW2:
            for tn in POW2:
                if bm % tm or bn % tn or (tm, tn) == (bm, bn):
                    continue
                q = (bm // tm) * (bn // tn)
                tw_sub = model.t_wave(tm, tn, bk)
                for policy, fixup, n in SUB_SPLITS:
                    units = S * q * (n or 1)
                    if units > P or (n and n > pairs):
                        continue
                    cfg = sk_config(bm, bn, bk, policy, fixup, n, (tm, tn))
                    if sk.unsupported_reason(M, N, K, config=cfg) is not None:
                        continue
                    t_tail = tw_sub / (n or 1)
                    if n:
                        t_tail += model.fixup(bm, bn, bk, fixup, n, 1.0 / q)
                    t_tail *= CAL["tail"].get(tail_kind(policy, fixup, n), CAL["tail_default"])
                    scored.append((W * tw_main + t_tail, bm, bn, bk, sk_name(bm, bn, bk, policy, fixup, n, "wt", (tm, tn))))
    scored.sort()
    out = [(name, t, "subtile") for t, *_, name in scored[:SUBTILE_TOP]]
    if previous_tile is not None:
        at_prev = [s for s in scored if s[1:4] == previous_tile]
        if at_prev and at_prev[0][4] not in {o[0] for o in out}:
            out.append((at_prev[0][4], at_prev[0][0], "subtile_prev_tile"))
    return out


def regime(M, N, tile):
    if tile is None:
        return "unknown", 0, 0
    T = tiles(M, N, tile[0], tile[1])
    W, S = divmod(T, P)
    if W == 0:
        name = "below_one_wave"
    elif S == 0:
        name = "no_tail"
    elif W == 1:
        name = "one_wave_tail"
    else:
        name = "two_plus_waves_tail"
    return name, W, S


def k_bin(K):
    return "K<=1024" if K <= 1024 else "K<=4096" if K <= 4096 else "K>4096"


def pilot_sample(shapes, n, seed=0):
    """n shapes stratified by regime, K bin and shipped tile power of two."""
    groups = shapes.groupby(["regime", "k_bin", "shipped_pow2"], group_keys=False)
    share = n / len(shapes)
    rng = np.random.default_rng(seed)
    picks = []
    for _, g in groups:
        k = max(1, round(len(g) * share))
        picks.append(g.iloc[rng.choice(len(g), size=min(k, len(g)), replace=False)])
    return pd.concat(picks).sort_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0, help="stratified sample size (0: all)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--calibrate", default=None, help="a finished run to fit corrections on")
    args = ap.parse_args()
    out = Path(args.out or HERE / "runs" / ("pilot" if args.pilot else "full"))
    out.mkdir(parents=True, exist_ok=True)
    sk, cb = _aiter_modules()

    cmp = pd.read_csv(STOCK / "cmp_all" / "compare_all.csv")
    cmp = cmp[cmp.like_for_like].reset_index(drop=True)
    prof = pd.read_csv(STOCK / "profile.csv")
    prof = prof[prof.libtype == "gluon"]
    by_shape = {k: g for k, g in prof.groupby(["M", "N", "K", "bias"])}
    if args.calibrate:
        import json

        CAL.update(calibrate(Path(args.calibrate), by_shape))
        (out / "calibration.json").write_text(json.dumps(CAL, indent=1, sort_keys=True))
        print(f"calibration: uneven default {CAL['uneven_default']:.3f} ({len(CAL['uneven'])} tiles), "
              f"tail {', '.join(f'{k} {v:.3f}' for k, v in sorted(CAL['tail'].items()))}")

    shape_rows, cand_rows = [], []
    for r in cmp.itertuples():
        key = (int(r.M), int(r.N), int(r.K), bool(r.bias))
        rows = by_shape.get(key)
        if rows is None:
            continue
        model = ShapeModel(*key[:3], rows)
        prev = parse_name(r.ours_kernel)
        prev_tile = (prev[1], prev[2], prev[3]) if prev else None
        shipped = (int(r.BM), int(r.BN)) if not pd.isna(r.BM) else None
        reg, W, S = regime(r.M, r.N, prev_tile[:2] if prev_tile else None)
        shipped_exact = shipped is not None and (
            (shipped[0] in POW2 and shipped[1] in POW2)
            or any(valid_uneven(cb, *t) for t in {shipped, shipped[::-1]})
        )
        shape_rows.append(
            dict(
                M=r.M, N=r.N, K=r.K, bias=r.bias,
                shipped_kernel=r.kernelName, shipped_libtype=r.libtype,
                shipped_BM=shipped[0] if shipped else None,
                shipped_BN=shipped[1] if shipped else None,
                shipped_pow2=bool(r.pow2_tile), shipped_exact=shipped_exact,
                shipped_us=r.pre_us, prev_kernel=r.ours_kernel, prev_us=r.ours_us,
                regime=reg, W=W, S=S, k_bin=k_bin(r.K),
            )
        )
        cands = [(r.ours_kernel, r.ours_tuned_us, "base_prev")]
        if model.best_split:
            cands.append((model.best_split[1], model.best_split[0], "base_split"))
        if model.best_dp:
            cands.append((model.best_dp[1], model.best_dp[0], "base_dp"))
        cands += uneven_candidates(model, cb, shipped)
        cands += subtile_candidates(model, sk, prev_tile)
        seen = set()
        for name, pred, method in cands:
            if name in seen:
                continue
            seen.add(name)
            cand_rows.append(dict(M=r.M, N=r.N, K=r.K, kernelName=name, method=method, pred_us=pred))

    shapes = pd.DataFrame(shape_rows)
    cands = pd.DataFrame(cand_rows)
    if args.pilot:
        shapes = pilot_sample(shapes, args.pilot)
        keys = set(zip(shapes.M, shapes.N, shapes.K))
        cands = cands[[k in keys for k in zip(cands.M, cands.N, cands.K)]]
    cands = cands.drop_duplicates(["M", "N", "K", "kernelName"])
    shapes.to_csv(out / "shapes.csv", index=False)
    cands.to_csv(out / "candidates.csv", index=False)
    untuned = shapes[["M", "N", "K", "bias"]].assign(
        dtype="torch.bfloat16", outdtype="torch.bfloat16", scaleAB=False, bpreshuffle=False
    )
    # Largest FLOP count first, as the stock retune, so long shapes start early.
    untuned = untuned.assign(f=untuned.M * untuned.N * untuned.K).sort_values("f", ascending=False).drop(columns="f")
    untuned.to_csv(out / "untuned.csv", index=False)
    per = cands.groupby(["M", "N", "K"]).size()
    print(f"{len(shapes)} shapes, {len(cands)} candidates ({per.mean():.1f} per shape, max {per.max()})")
    print(cands.method.value_counts().to_string())
    print(shapes.groupby(["regime", "k_bin"]).size().unstack(fill_value=0).to_string())


if __name__ == "__main__":
    main()
