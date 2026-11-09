"""The shapes of the tile-against-tail comparison, from aiter's tuned bf16 GEMM configs.

Main set: every unique (M, N, K, bias) tuned for 256 CUs with a bf16 output that the
gfx950 gluon kernels support (N and K multiples of 8, K >= 256) and whose 256x256
tile count T is more than one wave (T > 256) with a partial last wave (T mod 256 > 0).
Two controls, where the methods should tie: exactly full waves of 256x256 (T a
multiple of 256) and below one wave (T < 256), CONTROL shapes each, spread over K.

    python shapes.py runs/<name>     # writes shapes.csv and untuned.csv
"""

import glob
import math
import sys
from pathlib import Path

import pandas as pd

AITER = Path("/home/jbowley/repos/aiter")
P = 256
CONTROL = 30
KEY = ["M", "N", "K", "bias"]


def tuned_rows():
    files = [AITER / "aiter/configs/bf16_tuned_gemm.csv"] + sorted(
        Path(p) for p in glob.glob(str(AITER / "aiter/configs/model_configs/*bf16_tuned_gemm*.csv"))
    )
    frames = []
    for f in files:
        d = pd.read_csv(f)
        if not {"M", "N", "K", "cu_num", "outdtype"} <= set(d.columns):
            continue
        d["source"] = f.name
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    if "bias" not in d.columns:
        d["bias"] = False
    d["bias"] = d.bias.fillna(False).astype(str).str.lower().eq("true")
    d = d[(d.cu_num == P) & (d.outdtype.astype(str) == "torch.bfloat16")]
    if "dtype" in d.columns:
        d = d[d.dtype.astype(str) == "torch.bfloat16"]
    d = d[(d.N % 8 == 0) & (d.K % 8 == 0) & (d.K >= 256)]
    d = d.sort_values("us").groupby(KEY, as_index=False).first()
    return d[KEY + ["libtype", "kernelName", "us", "source"]].rename(
        columns={"libtype": "shipped_libtype", "kernelName": "shipped_kernel", "us": "tuned_us"}
    )


def label(d):
    d = d.copy()
    d["T"] = [math.ceil(m / 256) * math.ceil(n / 256) for m, n in zip(d.M, d.N)]
    d["W"] = d["T"] // P
    d["S"] = d["T"] % P
    d["fill"] = d.S / P
    d["set"] = "main"
    d.loc[d["T"] < P, "set"] = "below_one_wave"
    d.loc[(d["T"] >= P) & (d.S == 0), "set"] = "full_waves"
    d["k_bin"] = pd.cut(
        d.K, [0, 1024, 2048, 4096, 8192, 10**9], labels=["K<=1024", "K<=2048", "K<=4096", "K<=8192", "K>8192"]
    ).astype(str)
    d["fill_bin"] = pd.cut(
        d.fill, [-1, 0, 0.25, 0.5, 0.75, 1], labels=["none", "S/P<=1/4", "S/P<=1/2", "S/P<=3/4", "S/P<1"]
    ).astype(str)
    return d


def spread(d, n):
    """n shapes spread over K (and then M N): every len(d) / n-th in that order."""
    d = d.sort_values(["K", "M", "N"]).reset_index(drop=True)
    if len(d) <= n:
        return d
    idx = sorted({round(i * (len(d) - 1) / (n - 1)) for i in range(n)})
    return d.iloc[idx]


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    d = label(tuned_rows())
    main_set = d[d.set == "main"]
    controls = [spread(d[d.set == s], CONTROL) for s in ("full_waves", "below_one_wave")]
    shapes = pd.concat([main_set] + controls, ignore_index=True).sort_values(KEY)
    shapes.to_csv(out / "shapes.csv", index=False)
    shapes[KEY].assign(
        dtype="torch.bfloat16", outdtype="torch.bfloat16", scaleAB=False, bpreshuffle=False
    ).to_csv(out / "untuned.csv", index=False)
    print(shapes.groupby("set").size().to_string())
    print(main_set.groupby("k_bin").size().to_string())
    print(main_set.groupby("fill_bin").size().to_string())
    print(f"bias: {int(shapes.bias.sum())} of {len(shapes)}")


if __name__ == "__main__":
    main(sys.argv[1])
