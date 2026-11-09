"""Analysis of the dispatch-rule study.

    python analyze_grid.py calib runs/calib      # grid_bench.py against aiter's tuner, same session

Per shape, each method's time is its best candidate with whole tiles (the base) as fallback,
as in the main run: finer tiles = min(uneven, base), sub-tile tail = min(subtile, base),
Stream-K = min(Stream-K splitting plans, base), persistent loop = min(Stream-K
data-parallel plan, base). Candidates failing the correctness check (err_ratio > 0.05) are
dropped.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from issue_figs import KEY, MIN_M, parse_log  # noqa: E402

METHODS = {"finer tiles": "uneven", "sub-tile tail": "subtile", "Stream-K": "split"}


def gm(x):
    x = pd.Series(x).dropna()
    return float(np.exp(np.log(x).mean())) if len(x) else np.nan


def load_bench(path):
    d = pd.read_csv(path)
    d["bias"] = d.bias.astype(str).str.lower().eq("true")
    d = d[(d.error.isna() | (d.error == "")) & (d.us > 0) & (d.err_ratio <= 0.05)].copy()
    dp = d.kernelName.str.startswith("gluon_streamk_") & d.kernelName.str.endswith("_dp")
    d.loc[d.method == "streamk", "method"] = np.where(dp[d.method == "streamk"], "dp", "split")
    return d


def method_times(d):
    """Per shape: the best candidate of each method, and each method with base fallback."""
    t = d.groupby(KEY + ["method"]).us.min().unstack("method")
    for m in ("uneven", "subtile", "split", "dp"):
        if m not in t:
            t[m] = np.nan
        t[m + "+base"] = t[[m, "base"]].min(axis=1)
    return t.reset_index()


def calib(run):
    run = Path(run)
    b = load_bench(run / "bench.csv")
    groups = {"base": "base", "uneven": "uneven", "subtile": "subtile", "streamk": "split", "streamk_dp": "dp"}
    lines = ["# Calibration: grid_bench.py against aiter's tuner (--run_config), same session", ""]
    same, rep = [], {}
    for grp, m in groups.items():
        f = run / f"replay_{grp}.csv"
        if not f.exists():
            continue
        picks = pd.read_csv(f)[KEY + ["kernelName"]]
        picks["bias"] = picks.bias.astype(str).str.lower().eq("true")
        r = parse_log(run / f"replay_{grp}.log", "aiter")
        r = r.merge(picks, on=KEY)
        rep[m] = r.set_index(KEY).aiter
        j = r.merge(b[KEY + ["kernelName", "us"]], on=KEY + ["kernelName"]).dropna()
        j["ratio"] = j.us / j.aiter
        j["method"] = m
        same.append(j)
    s = pd.concat(same)
    lines += ["## The same kernel on the same shape (harness / aiter)", "",
              "| Method | Pairs | Median | Geomean | 5% | 95% |", "|---|---|---|---|---|---|"]
    for m, g in s.groupby("method"):
        lines.append(f"| {m} | {len(g)} | {g.ratio.median():.4f} | {gm(g.ratio):.4f} | "
                     f"{g.ratio.quantile(0.05):.3f} | {g.ratio.quantile(0.95):.3f} |")
    lines.append(f"| all | {len(s)} | {s.ratio.median():.4f} | {gm(s.ratio):.4f} | "
                 f"{s.ratio.quantile(0.05):.3f} | {s.ratio.quantile(0.95):.3f} |")
    s.to_csv(run / "calib_pairs.csv", index=False)

    # Winning method: the harness's best of the three (each with base fallback) against aiter's
    # replays of the main run's picks, both in this session.
    t = method_times(b).set_index(KEY)
    a = pd.DataFrame(rep)
    for m in ("uneven", "subtile", "split"):
        a[m + "+base"] = a[[m, "base"]].min(axis=1)
    cols = [c + "+base" for c in ("uneven", "subtile", "split")]
    j = t[cols].join(a[cols], lsuffix="_h", rsuffix="_a", how="inner").dropna()
    hw = j[[c + "_h" for c in cols]].idxmin(axis=1).str.replace("_h", "")
    aw = j[[c + "_a" for c in cols]].idxmin(axis=1).str.replace("_a", "")
    best_a = j[[c + "_a" for c in cols]].min(axis=1)
    h_in_a = np.array([j.loc[i, hw[i] + "_a"] for i in j.index])
    agree = (hw == aw) | (h_in_a <= best_a * 1.01)
    lines += ["", "## Winning method", "",
              f"{len(j)} shapes. Same winner, or the harness's winner within 1% of the best in aiter's "
              f"replays: {agree.sum()} ({100 * agree.mean():.0f}%). Identical winner: {(hw == aw).sum()}.", ""]
    ratio = {c: gm(j[c + "_h"] / j[c + "_a"]) for c in cols}
    lines.append("Each method's best (harness) over aiter's replay of the main run's pick, geomean: "
                 + ", ".join(f"{c} {v:.4f}" for c, v in ratio.items()) + ".")
    text = "\n".join(lines) + "\n"
    (run / "calibration.md").write_text(text)
    print(text)


# --- the grid -----------------------------------------------------------------------------

P = 256
ORDER = ["finer tiles", "sub-tile tail", "Stream-K"]  # preference when tied within TIE
COL = {"finer tiles": "uneven+base", "sub-tile tail": "subtile+base", "Stream-K": "split+base"}
TIE = 0.01
K_LEVELS = [1024, 2048, 4096, 8192, 16384]


def grid_times(run):
    """Per grid shape: each method's time (mean of the two passes for candidates timed twice),
    and the noise between the passes."""
    shapes = pd.read_csv(run / "shapes.csv")
    shapes["bias"] = False
    a = load_bench(run / "bench_a.csv")
    noise = None
    if (run / "bench_b.csv").exists():
        b = load_bench(run / "bench_b.csv")
        ta, tb = method_times(a).set_index(KEY), method_times(b).set_index(KEY)
        cols = ["base", "uneven", "subtile", "split", "dp"]
        noise = (tb[cols] / ta[cols].reindex(tb.index)).stack()
        m = a.merge(b[KEY + ["kernelName", "us"]], on=KEY + ["kernelName"], how="left", suffixes=("", "_b"))
        m["us"] = m[["us", "us_b"]].mean(axis=1)
        a = m.drop(columns="us_b")
    t = add_cols(method_times(a).merge(shapes, on=KEY))
    return t, noise


def padding(M, N):
    """Share of the 256x256 tile grid that is padding."""
    return 1 - M * N / (np.ceil(M / 256) * 256 * np.ceil(N / 256) * 256)


def k_level(K):
    """The grid's K levels as the issue's K groups: K <= 1024, 1024 < K <= 2048, ..., K > 8192."""
    lv = np.array(K_LEVELS)
    return lv[np.minimum(np.searchsorted(lv, np.asarray(K)), len(lv) - 1)]


def add_cols(t):
    t["f"] = t.S / P
    t["pad"] = padding(t.M, t.N)
    t["K_level"] = k_level(t.K)
    t["best"] = t[list(COL.values())].min(axis=1)
    return t


def apply_rule(t, rule):
    """rule = {"gate": padding share above which finer tiles are used (None: no gate),
    "K": {K level: (tail method, largest S/P that uses it)}}; finer tiles otherwise."""
    out = t[COL["finer tiles"]].copy()
    ok = (t.pad <= rule["gate"] + 1e-12) if rule.get("gate") is not None else pd.Series(True, index=t.index)
    for K, (m, s) in rule["K"].items():
        sel = (t.K_level == K) & (t.f > 0) & (t.f <= s + 1e-9) & ok
        out[sel] = t.loc[sel, COL[m]]
    return out


SK_HALF = {"gate": None, "K": {K: ("Stream-K", 0.5 if K > 4096 else 0.0) for K in K_LEVELS}}
AITER_RULE = {"gate": None, "K": {K: ("sub-tile tail", 0.5) if K <= 2048 else ("Stream-K", 0.5 if K > 4096 else 0.0)
                                  for K in K_LEVELS}}
# The recommended rule: the fitted rule's shape with Stream-K capped at half a wave.
SIMPLE_RULE = {"gate": 0.10, "K": {K: ("sub-tile tail" if K <= 4096 else "Stream-K", 0.5) for K in K_LEVELS}}
GATES = [None, 0.02, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3]


def winner(row_times):
    best = min(row_times.values())
    for name in ORDER:
        if row_times[name] <= best * (1 + TIE):
            return name


def fit_k(q):
    """For one K: the tail method and S/P threshold that minimise the geomean loss against the
    best of the three methods for each shape. The threshold 0 means finer tiles throughout."""
    best_loss, best = np.inf, ("Stream-K", 0.0)
    for method in ("sub-tile tail", "Stream-K"):
        for s in [i / 16 for i in range(16)]:
            r = q[COL["finer tiles"]].where(~((q.f > 0) & (q.f <= s + 1e-9)), q[COL[method]])
            loss = gm(r / q.best)
            if loss < best_loss - 1e-12:
                best_loss, best = loss, (method, s)
    return best, best_loss


def summarize(t, times, label):
    r = times / t.best
    return {"rule": label, "vs base": f"{gm(t.base / times):.3f}", "slower than best per shape": f"{gm(r):.4f}",
            "shapes >3% slower than best": f"{100 * (r > 1.03).mean():.0f}%"}


def fit_rule(t, gate):
    th = {}
    for K in K_LEVELS:
        q = t[t.K_level == K]
        if gate is not None:
            q = q[q.pad <= gate + 1e-12]
        th[K] = fit_k(q)[0] if len(q) else ("Stream-K", 0.0)
    rule = {"gate": gate, "K": th}
    return rule, gm(apply_rule(t, rule) / t.best)


def grid(run, *extra):
    run = Path(run)
    t, noise = grid_times(run)
    for e in extra:
        t = pd.concat([t, grid_times(Path(e))[0]], ignore_index=True)
    g = t[t.set == "grid"]
    lines = [f"# Grid: {len(t)} shapes ({len(g)} factorial, {(t.set == 'grid_ragged').sum()} ragged N, "
             f"{(t.set == 'padded').sum()} ragged M)", ""]
    if noise is not None:
        lines += ["## Noise between the two passes (pass B / pass A, each method's best per shape)", "",
                  f"median {noise.median():.4f}, 5-95% {noise.quantile(0.05):.3f}-{noise.quantile(0.95):.3f}, "
                  f"median |difference| {100 * (noise - 1).abs().median():.2f}% over {len(noise)} pairs.", ""]

    # The padding gate: for each candidate, fit the per-K thresholds on the shapes it lets
    # through, then score the whole rule on every shape.
    rows, fits = [], {}
    for gate in GATES:
        rule_g, loss = fit_rule(t, gate)
        fits[gate] = (rule_g, loss)
        rows.append({"finer tiles when padding share >": "no gate" if gate is None else f"{gate:.3f}",
                     "geomean loss vs best (grid)": f"{loss:.4f}",
                     "shapes >3% slower": f"{100 * (apply_rule(t, rule_g) / t.best > 1.03).mean():.1f}%"})
    gate = min(fits, key=lambda g: fits[g][1])
    rule = fits[gate][0]
    lines += ["## Padding gate", "", md(pd.DataFrame(rows)), "",
              f"Chosen: {'no gate' if gate is None else f'finer tiles when more than {100 * gate:.1f}% of the 256x256 tile grid is padding'}.", ""]

    # The per-K thresholds, with bootstrap over the shapes the gate lets through.
    rng = np.random.default_rng(0)
    rows = []
    tg = t if gate is None else t[t.pad <= gate + 1e-12]
    for K in K_LEVELS:
        q = tg[tg.K_level == K]
        (m, s), loss = fit_k(q)
        boots = []
        for _ in range(200):
            qq = q.iloc[rng.integers(0, len(q), len(q))]
            boots.append(fit_k(qq)[0])
        bs = pd.DataFrame(boots, columns=["m", "s"])
        same_m = (bs.m == m).mean()
        rows.append({"K": K, "tail method": m, "use it up to S/P": f"{s:.4f}" if s else "never",
                     "bootstrap: same method": f"{100 * same_m:.0f}%",
                     "bootstrap S/P 5-95%": f"{bs.s.quantile(0.05):.3f}-{bs.s.quantile(0.95):.3f}",
                     "geomean loss vs best": f"{loss:.4f}"})
    lines += ["## Fitted rule (per K, over all W and M under the gate; finer tiles where no tail method applies)", "",
              md(pd.DataFrame(rows)), ""]

    # Is the threshold the same for each W and each M? Fit per (K, W) and per (K, M).
    for by in ("W", "M"):
        rows = []
        for K in K_LEVELS:
            r = {"K": K}
            for v, q in tg[tg.K_level == K].groupby(by):
                (m, s), _ = fit_k(q)
                r[f"{by}={v}"] = f"{'SK' if m == 'Stream-K' else 'sub'} <= {s:.3f}" if s else "finer"
            rows.append(r)
        lines += [f"Fitted per K and {by}:", "", md(pd.DataFrame(rows)), ""]

    # Candidate rules on the grid.
    cands = [("finer tiles everywhere", t[COL["finer tiles"]]),
             ("sub-tile tail everywhere", t[COL["sub-tile tail"]]),
             ("Stream-K everywhere", t[COL["Stream-K"]]),
             ("finer tiles; Stream-K for K > 4096, S/P <= 1/2", apply_rule(t, SK_HALF)),
             ("AITER-study rule (+ sub-tile for K <= 2048, S/P <= 1/2)", apply_rule(t, AITER_RULE)),
             ("fitted rule, no padding gate", apply_rule(t, fits[None][0])),
             ("fitted rule", apply_rule(t, rule)),
             ("recommended rule (padding <= 10%, S/P <= 1/2: sub-tile K <= 4096, Stream-K K > 4096)", apply_rule(t, SIMPLE_RULE)),
             ("best per shape", t.best)]
    lines += ["## Rules on the grid", "", md(pd.DataFrame([summarize(t, v, k) for k, v in cands])), ""]

    # Persistent loop against the split (Stream-K kernel), per K, at most half a wave left.
    rows = []
    for K in K_LEVELS:
        q = t[(t.K == K) & (t.f > 0) & (t.f <= 0.5)]
        persist = q[["dp", "base"]].min(axis=1)
        rows.append({"K": K, "persistent loop vs base": f"{gm(q.base / persist):.3f}",
                     "split vs persistent loop": f"{gm(persist / q['split+base']):.3f}"})
    lines += ["## Stream-K's split against its persistent loop (at most half a wave left)", "", md(pd.DataFrame(rows)), ""]

    t["winner"] = [winner({k: r[c] for k, c in COL.items()}) for _, r in t.iterrows()]
    t.to_csv(run / "grid_times.csv", index=False)
    import json
    (run / "rule.json").write_text(json.dumps({"gate": gate, "K": {str(k): list(v) for k, v in rule["K"].items()}}))
    validate(rule, fits[None][0], lines)
    figure(t, SIMPLE_RULE, run)
    figure_padding(t, run)
    text = "\n".join(lines) + "\n"
    (run / "grid.md").write_text(text)
    print(text)


def validate(rule, rule_nogate, lines):
    """The rules on the AITER shapes (main run's replays; M > 128)."""
    d = pd.read_csv(Path(__file__).resolve().parent / "runs/full/results.csv")
    d = d[(d.set == "main") & (d.M >= MIN_M)].copy()
    d["split+base"] = d[["streamk", "base"]].min(axis=1)
    d = add_cols(d)
    rules = [("finer tiles everywhere", d[COL["finer tiles"]]), ("finer tiles; Stream-K for K > 4096, S/P <= 1/2", apply_rule(d, SK_HALF)),
             ("AITER-study rule", apply_rule(d, AITER_RULE)), ("fitted rule, no padding gate", apply_rule(d, rule_nogate)),
             ("fitted rule", apply_rule(d, rule)), ("recommended rule", apply_rule(d, SIMPLE_RULE)), ("best per shape", d.best)]
    regions = {"all": d.K > 0, "S/P <= 1/2, K <= 2048": (d.f <= 0.5) & (d.K <= 2048),
               "S/P <= 1/2, 2048 < K <= 4096": (d.f <= 0.5) & (d.K > 2048) & (d.K <= 4096),
               "S/P <= 1/2, K > 4096": (d.f <= 0.5) & (d.K > 4096), "S/P > 1/2": d.f > 0.5}
    rows = []
    for name, tt in rules:
        r = {"rule": name}
        for reg, sel in regions.items():
            r[f"{reg} ({int(sel.sum())})"] = f"{gm(d.base[sel] / tt[sel]):.3f}"
        r["time-weighted, all"] = f"{d.base.sum() / tt.sum():.3f}"
        r["shapes >3% slower than best"] = f"{100 * (tt / d.best > 1.03).mean():.0f}%"
        rows.append(r)
    lines += [f"## Validation on the {len(d)} AITER shapes with M > 128 (main run's replays; speedup over the base)", "",
              md(pd.DataFrame(rows)), ""]


def figure(t, rule, run):
    """Best method per factorial grid shape. Solid: one method leads the next by 1% or more.
    Hatched: the methods are within 1% of each other, but the best gains 1% or more over whole
    tiles. Grey: no method gains 1% over whole tiles. Each cell: the best gain over whole tiles (%)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    colours = {"finer tiles": "#3a68b4", "sub-tile tail": "#e07b2a", "Stream-K": "#3a9a4f"}
    grey = "#e6e6e6"
    g = t[(t.set == "grid") & (t.f > 0)]
    Ws, Ms = sorted(g.W.unique()), sorted(g.M.unique())
    fig, axes = plt.subplots(len(Ws), len(Ms), figsize=(15, 14), sharex=True, sharey=True)
    for i, W in enumerate(Ws):
        for j, M in enumerate(Ms):
            ax = axes[i][j]
            for _, r in g[(g.W == W) & (g.M == M)].iterrows():
                times = {k: r[c] for k, c in COL.items()}
                best = min(times.values())
                w = min(times, key=times.get)
                lead = sorted(times.values())[1] / best - 1
                gain = r.base / best - 1
                x, y = K_LEVELS.index(r.K), r.f
                if gain < TIE:
                    ax.add_patch(Rectangle((x - 0.5, y - 1 / 32), 1, 1 / 16, facecolor=grey, edgecolor="white", lw=0.5))
                    ink = "#777777"
                elif lead < TIE:
                    ax.add_patch(Rectangle((x - 0.5, y - 1 / 32), 1, 1 / 16, facecolor="white", edgecolor="white", lw=0.5))
                    ax.add_patch(Rectangle((x - 0.5, y - 1 / 32), 1, 1 / 16, facecolor="none", hatch="////",
                                           edgecolor=colours[w], lw=0))
                    ink = "black"
                else:
                    ax.add_patch(Rectangle((x - 0.5, y - 1 / 32), 1, 1 / 16, facecolor=colours[w], edgecolor="white", lw=0.5))
                    ink = "white"
                ax.text(x, y, f"{100 * gain:.0f}", ha="center", va="center", fontsize=7, color=ink)
            for k, K in enumerate(K_LEVELS):
                m, s = rule["K"][K]
                if s:
                    ax.plot([k - 0.5, k + 0.5], [s + 1 / 32, s + 1 / 32], color="black", lw=2.5)
            ax.set_xlim(-0.5, 4.5)
            ax.set_ylim(1 / 32, 1 - 1 / 32)
            ax.set_title(f"{W} full wave{'s' if W > 1 else ''}, M = {M}", fontsize=11)
            ax.set_xticks(range(5), ["1K", "2K", "4K", "8K", "16K"], fontsize=10)
            ax.set_yticks([i / 8 for i in range(1, 8)], [f"{i}/8" for i in range(1, 8)], fontsize=9)
            if j == 0:
                ax.set_ylabel("last-wave fill S/P", fontsize=10)
            if i == len(Ws) - 1:
                ax.set_xlabel("K", fontsize=10)
    handles = [Patch(facecolor=c, label=f"{n}: 1% or more ahead") for n, c in colours.items()]
    handles += [Patch(facecolor="white", edgecolor="#555555", hatch="////", label="methods within 1% (hatch colour: the fastest)"),
                Patch(facecolor=grey, label="no method gains 1% over whole tiles"),
                plt.Line2D([0], [0], color="black", lw=2.5, label="rule (no padding): below, sub-tile tail for K <= 4K, Stream-K for K >= 8K; above, finer tiles")]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=10, frameon=False)
    fig.suptitle("Best method for each grid shape; the number is the best method's gain over whole 256x256 tiles (%)",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(run / "fig_grid_best.png", dpi=150)
    plt.close(fig)


def figure_padding(t, run):
    """Finer tiles against the better tail method, with at most half a wave left, as a function
    of padding: left, the share of the 256x256 tile grid that is padding (the rule's gate);
    right, the padding the fastest finer tile removes (256x256 padding minus its own)."""
    import re

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401

    def tile(n):
        m = re.search(r"_(\d+)x(\d+)$", str(n))
        return (int(m[1]), int(m[2])) if m else (np.nan, np.nan)

    def removed(q, names):
        bm, bn = (np.array(v, dtype=float) for v in zip(*names.map(tile)))
        own = 1 - q.M * q.N / (np.ceil(q.M / bm) * bm * np.ceil(q.N / bn) * bn)
        return q.pad - own

    v2 = Path(__file__).resolve().parent
    d = pd.read_csv(v2 / "runs/full/results.csv")
    d = d[(d.set == "main") & (d.M >= MIN_M)].copy()
    d["split+base"] = d[["streamk", "base"]].min(axis=1)
    d = add_cols(d)
    d = d[(d.f > 0) & (d.f <= 0.5)].copy()
    d["removed"] = removed(d, d.uneven_kernel)
    g = t[(t.f > 0) & (t.f <= 0.5)].copy()
    bk = pd.concat([best_kernels(run / "bench_a.csv"), best_kernels(Path(str(run) + "_pad") / "bench_a.csv")])
    g = g.merge(bk[KEY + ["uneven"]].rename(columns={"uneven": "uneven_kernel"}), on=KEY, how="left")
    g["removed"] = removed(g, g.uneven_kernel)
    for q in (d, g):
        q["ratio"] = q[["subtile+base", "split+base"]].min(axis=1) / q["uneven+base"]

    # Shapes with exactly no padding (or none removed) all sit at x = 0: spread them over the
    # axis's linear region around 0 so that they can be seen.
    rng = np.random.default_rng(0)
    for q in (d, g):
        for x in ("pad", "removed"):
            z = q[x].abs() < 1e-9
            q.loc[z, x] = rng.uniform(-0.004, 0.004, z.sum())
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=True)
    for ax, x, title in ((axes[0], "pad", "share of the 256x256 tile grid that is padding"),
                         (axes[1], "removed", "padding removed by the fastest finer tile (256x256 padding minus its own)")):
        plain = g[g.set != "padded"]
        ax.scatter(100 * plain[x], plain.ratio, s=10, color="#888888", alpha=0.6,
                   label=f"grid, no padding (M and N multiples of 256) or N - 72 ({len(plain)})", zorder=3)
        for M, mk in zip((300, 450, 700, 1100), ("^", "v", "<", ">")):
            q = g[(g.set == "padded") & (g.M == M)]
            ax.scatter(100 * q[x], q.ratio, s=34, marker=mk, color="#7b3fa0", alpha=0.8,
                       label=f"grid, M = {M} ({100 * q.pad.iloc[0]:.0f}% padding)", zorder=3)
        for lo, col, lab in ((False, "#e07b2a", "AITER shapes, K <= 4096"), (True, "#3a9a4f", "AITER shapes, K > 4096")):
            q = d[(d.K > 4096) == lo]
            ax.scatter(100 * q[x], q.ratio, s=22, color=col, edgecolor="black", lw=0.3, label=f"{lab} ({len(q)})", zorder=2)
        allx = pd.concat([d[[x, "ratio"]], g[[x, "ratio"]]])
        bins = [-100, -0.5, 0.5, 2, 5, 10, 20, 40, 101] if x == "pad" else [-100, -0.5, 0.5, 2, 5, 10, 20, 40, 101]
        cut = pd.cut(100 * allx[x], bins)
        med = allx.groupby(cut, observed=True).agg(x=(x, "median"), r=("ratio", gm), n=("ratio", "size"))
        med = med[med.n >= 5]
        ax.plot(100 * med.x, med.r, color="black", lw=2, marker="o", label="geomean of each padding band", zorder=4)
        ax.axhline(1, color="black", lw=0.8)
        if x == "pad":
            ax.axvline(10, color="#7b3fa0", ls="--", lw=1.5)
            ax.text(9.3, 1.28, "rule's gate: 10%", color="#7b3fa0", fontsize=9, ha="right")
            q = g[(g.set == "padded") & (g.M == 450)]
            ax.annotate("M = 450: 12% padding; the best finer\ntile (160x288) removes only half", xy=(100 * q.pad.iloc[0], gm(q.ratio)),
                        xytext=(25, 0.86), fontsize=9, arrowprops=dict(arrowstyle="->", color="#7b3fa0"), color="#7b3fa0")
        ax.set_xscale("symlog", linthresh=1)
        if x == "pad":
            ax.set_xlim(-0.8, 110)
            ticks = [0, 1, 2, 5, 10, 20, 50, 100]
        else:
            ax.set_xlim(-12, 110)
            ticks = [-10, -5, -2, -1, 0, 1, 2, 5, 10, 20, 50, 100]
            ax.text(0.02, 0.10, "finer tile adds padding", transform=ax.transAxes, fontsize=8, color="#777777")
        ax.set_xticks(ticks, [str(v) for v in ticks])
        ax.set_yscale("log")
        yt = [0.85, 0.9, 0.95, 1, 1.05, 1.1, 1.2, 1.3]
        ax.set_yticks(yt, [f"{v:.2f}" for v in yt])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_xlabel(f"{title} (%)")
        ax.grid(alpha=0.3)
        ax.text(0.01, 0.98, "finer tiles faster", transform=ax.transAxes, va="top", fontsize=9, color="#3a68b4")
        ax.text(0.01, 0.02, "tail method faster", transform=ax.transAxes, va="bottom", fontsize=9, color="#555555")
    axes[0].set_ylabel("best tail method's time / finer tiles' time")
    axes[1].legend(fontsize=8, loc="lower right")
    fig.suptitle("At most half a wave left: finer tiles win where they can remove padding", fontsize=12)
    fig.tight_layout()
    fig.savefig(run / "fig_padding.png", dpi=150)
    plt.close(fig)


GROUPS = {"base": "gluon_compute_bound_", "uneven": "gluon_uneven_", "subtile": "gluon_subtile_", "split": "gluon_streamk_"}


def best_kernels(path):
    """Per shape and method: the fastest candidate's kernelName (Stream-K: splitting plans)."""
    d = load_bench(path)
    return d.sort_values("us").groupby(KEY + ["method"]).head(1).pivot(index=KEY, columns="method",
                                                                         values="kernelName").reset_index()


def confirm_setup(run):
    """Rows for aiter's --run_config, one file per method with one row per shape: the main
    run's pick of each method for the AITER shapes, this run's best for ~100 grid shapes."""
    run = Path(run)
    v2 = Path(__file__).resolve().parent
    tmpl = pd.read_csv(v2 / "runs/full/replay/0/base.csv").iloc[0]
    out = run / "confirm"
    out.mkdir(exist_ok=True)

    def write(pairs, name):
        r = pd.DataFrame([tmpl] * len(pairs)).reset_index(drop=True)
        for c in KEY + ["kernelName"]:
            r[c] = pairs[c].values
        r["us"] = 0
        r.to_csv(out / name, index=False)

    res = pd.read_csv(v2 / "runs/full/results.csv")
    main = res[res.set == "main"][KEY]
    for grp, src in (("base", "base"), ("uneven", "uneven"), ("subtile", "subtile"), ("split", "streamk")):
        p = pd.concat(pd.read_csv(f) for f in sorted((v2 / "runs/full/replay").glob(f"*/{src}.csv")))
        write(p[KEY + ["kernelName"]].merge(main, on=KEY).drop_duplicates(KEY), f"aiter_{grp}.csv")

    t = pd.read_csv(run / "grid_times.csv")
    sel = t[(t.set == "grid") & (t.S % 32 == 16)]
    sel = pd.concat(g.iloc[::3] for _, g in sel.groupby(["K", "W", "M"]))
    bk = best_kernels(run / "bench_a.csv").merge(sel[KEY], on=KEY)
    for grp in GROUPS:
        q = bk[KEY + [grp]].dropna().rename(columns={grp: "kernelName"})
        write(q, f"grid_{grp}.csv")
    print(f"{len(main)} AITER shapes, {len(sel)} grid shapes")


def confirm(run):
    """Score the rules with aiter's replays (confirm/*.log)."""
    run = Path(run)
    v2 = Path(__file__).resolve().parent
    import json
    r = json.loads((run / "rule.json").read_text())
    rule = {"gate": r["gate"], "K": {int(k): tuple(v) for k, v in r["K"].items()}}
    lines = ["# Confirmation through aiter's harness (--run_config replays, one session)", ""]
    for which in ("aiter", "grid"):
        t = None
        for grp in GROUPS:
            lg = parse_log(run / f"confirm/{which}_{grp}.log", grp).drop_duplicates(KEY)
            t = lg if t is None else t.merge(lg, on=KEY, how="outer")
        for m in ("uneven", "subtile", "split"):
            t[m + "+base"] = t[[m, "base"]].min(axis=1)
        if which == "aiter":
            shapes = pd.read_csv(v2 / "runs/full/results.csv")
            shapes = shapes[(shapes.set == "main") & (shapes.M >= MIN_M)][KEY + ["S"]]
        else:
            shapes = pd.read_csv(run / "shapes.csv")[KEY + ["S"]]
            shapes["bias"] = False
        t = add_cols(t.merge(shapes, on=KEY).dropna(subset=["base"]))

        rules = [("finer tiles everywhere", t[COL["finer tiles"]]),
                 ("finer tiles; Stream-K for K > 4096, S/P <= 1/2", apply_rule(t, SK_HALF)),
                 ("AITER-study rule", apply_rule(t, AITER_RULE)), ("fitted rule", apply_rule(t, rule)),
                 ("recommended rule", apply_rule(t, SIMPLE_RULE)), ("best per shape", t.best)]
        rows = [{"rule": n, "vs base": f"{gm(t.base / v):.3f}", "slower than best": f"{gm(v / t.best):.4f}",
                 "shapes >3% slower than best": f"{100 * (v / t.best > 1.03).mean():.0f}%",
                 "time-weighted vs base": f"{t.base.sum() / v.sum():.3f}"} for n, v in rules]
        title = f"{len(t)} AITER shapes" if which == "aiter" else f"{len(t)} grid shapes"
        lines += [f"## {title}", "", md(pd.DataFrame(rows)), ""]
    text = "\n".join(lines) + "\n"
    (run / "confirm.md").write_text(text)
    print(text)


def md(df):
    cols = list(df.columns)
    out = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        out.append("| " + " | ".join(map(str, r)) + " |")
    return "\n".join(out)


if __name__ == "__main__":
    {"calib": calib, "grid": grid, "confirm-setup": confirm_setup, "confirm": confirm}[sys.argv[1]](*sys.argv[2:])
