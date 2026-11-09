"""Replay CSVs for the transfer analysis: at every fourth M of each anchor's range (and
M0), the kernel each method runs (transfer.csv from analyze_transfer.py), one
tuned-format CSV per (method, anchor group) for --run_config, split over GPUs.

    python replay_transfer.py runs/<name> NGPUS     # writes runs/<name>/replay_transfer/
    python replay_transfer.py runs/<name> --compare # replay vs tuner ratios, appended to
                                                    # transfer_summary.md

aiter's run mode looks configs up by (M, N, K), so a CSV holds one kernel per shape:
anchors whose M ranges share an M go into different groups.
"""

import math
import re
import sys
from pathlib import Path

import pandas as pd

METHODS = ("tile_fixed", "sk_runtime", "sk_fixed", "best_tile", "best_sk")
KEY = ["M", "N", "K"]
ROW = re.compile(r"^\((\d+), (\d+), (\d+), \S+, bias=(True|False)\)\s*\|\s*(\S+)\s*\|\s*(.+?)\s*$")
COLUMNS = ["gfx", "cu_num", "M", "N", "K", "bias", "dtype", "outdtype", "scaleAB",
           "bpreshuffle", "libtype", "solidx", "splitK", "us", "kernelName", "err_ratio",
           "tflops", "bw"]


def groups(d):
    """Anchor groups whose (M, N, K) sets are disjoint, greedily."""
    out = []
    for anchor, g in d.groupby(["N", "K", "M0"]):
        keys = set(map(tuple, g[KEY].values))
        for grp in out:
            if not keys & grp["keys"]:
                grp["anchors"].append(anchor)
                grp["keys"] |= keys
                break
        else:
            out.append(dict(anchors=[anchor], keys=keys))
    return [grp["anchors"] for grp in out]


def make(run, ngpus):
    d = pd.read_csv(run / "transfer.csv")
    sel = []
    for _, g in d.groupby(["N", "K", "M0"]):
        g = g.sort_values("M")
        sel.append(pd.concat([g.iloc[::4], g[g.M == g.M0]]).drop_duplicates("M"))
    d = pd.concat(sel)
    out = run / "replay_transfer"
    out.mkdir(exist_ok=True)
    files = []
    for gi, anchors in enumerate(groups(d)):
        g = d[d.set_index(["N", "K", "M0"]).index.isin(anchors)]
        for meth in METHODS:
            rows = [dict(gfx="gfx950", cu_num=256, M=r.M, N=r.N, K=r.K, bias=False,
                         dtype="torch.bfloat16", outdtype="torch.bfloat16", scaleAB=False,
                         bpreshuffle=False, libtype="gluon", solidx=0, splitK=0,
                         us=getattr(r, meth), kernelName=getattr(r, f"{meth}_name"),
                         err_ratio=0.0, tflops=0.0, bw=0.0)
                    for r in g.itertuples() if isinstance(getattr(r, f"{meth}_name"), str)]
            f = out / f"{meth}_g{gi}.csv"
            pd.DataFrame(rows, columns=COLUMNS).to_csv(f, index=False)
            files.append(f.name)
    for i in range(ngpus):
        (out / f"gpu{i}.txt").write_text("\n".join(files[i::ngpus]) + "\n")
    print(f"{len(files)} files, {len(d)} (anchor, M) points, {len(groups(d))} anchor groups")


def parse(path):
    t = {}
    if path.exists():
        for line in path.read_text().splitlines():
            m = ROW.match(line)
            if m and m.group(6) == "OK" and float(m.group(5)) > 0:
                t[tuple(map(int, m.groups()[:3]))] = float(m.group(5))
    return t


def compare(run):
    out = run / "replay_transfer"
    d = pd.read_csv(run / "transfer.csv")
    rows = []
    for f in sorted(out.glob("*_g*.csv")):
        meth = f.stem.rsplit("_g", 1)[0]
        a, b = parse(out / f"{f.stem}_a.log"), parse(out / f"{f.stem}_b.log")
        for r in pd.read_csv(f).itertuples():
            k = (r.M, r.N, r.K)
            if k in a and k in b:
                rows.append(dict(method=meth, M=r.M, N=r.N, K=r.K, tune_us=r.us,
                                 replay_us=(a[k] + b[k]) / 2,
                                 noise=abs(a[k] - b[k]) / min(a[k], b[k])))
    r = pd.DataFrame(rows)
    r = r.merge(d[["N", "K", "M0", "M"]], on=["M", "N", "K"])
    r.to_csv(run / "replay_transfer.csv", index=False)
    piv = r.pivot_table(index=["N", "K", "M0", "M"], columns="method", values="replay_us")
    below = piv[piv.index.get_level_values("M") < piv.index.get_level_values("M0")]

    def gm(s):
        s = s.dropna()
        return round(math.exp(s.map(math.log).mean()), 3) if len(s) else float("nan")

    lines = ["", "## 6. Replay (--run_config, A/B mean) at every fourth M", "",
             f"{len(r)} replayed (method, M) points; median A/B noise "
             f"{100 * r.noise.median():.1f}%; replay / tuner time geomean "
             f"{gm(r.replay_us / r.tune_us)}.", "",
             "| comparison (below M0, replay times) | points | geomean |", "|---|---|---|"]
    for num, den in (("tile_fixed", "best_tile"), ("sk_runtime", "best_tile"),
                     ("sk_fixed", "best_tile"), ("best_sk", "best_tile"),
                     ("sk_runtime", "tile_fixed")):
        s = (below[num] / below[den]).dropna()
        lines.append(f"| {num} / {den} | {len(s)} | {gm(s)} |")
    text = "\n".join(lines) + "\n"
    with open(run / "transfer_summary.md", "a") as f:
        f.write(text)
    print(text)


if __name__ == "__main__":
    run = Path(sys.argv[1])
    if sys.argv[2] == "--compare":
        compare(run)
    else:
        make(run, int(sys.argv[2]))
