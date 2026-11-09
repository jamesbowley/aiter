"""The controlled grid of the dispatch-rule study: every combination of

    K     1024, 2048, 4096, 8192, 16384
    W     1, 2, 4 full waves of 256x256 tiles
    S     0, 16, ..., 240 tiles left for the last wave (S/P = 0, 1/16, ..., 15/16; 0 is the control)
    M     512, 1024, 4096 (2, 4 and 16 rows of 256x256 tiles)

with N = 256 (W P + S) / (M / 256), a whole number of 256x256 tiles (720 shapes, set "grid"),
and a ragged copy, N - 72, of M = 1024, W in {1, 2}, K in {2048, 8192} (64 shapes,
set "grid_ragged"; the same tile count with a partial last tile column).

    python grid_shapes.py runs/<name>     # writes shapes.csv
"""

import math
import sys
from pathlib import Path

import pandas as pd

P = 256
KS = (1024, 2048, 4096, 8192, 16384)
WS = (1, 2, 4)
SS = tuple(range(0, P, 16))
MS = (512, 1024, 4096)


def row(M, N, K, W, S, kind):
    T = math.ceil(M / 256) * math.ceil(N / 256)
    assert T == W * P + S, (M, N, K, W, S, T)
    return dict(M=M, N=N, K=K, bias=False, T=T, W=W, S=S, fill=S / P, set=kind)


def padded(out):
    """The padding supplement: M not a multiple of 256 (300, 450, 700, 1100: 41%, 12%, 9% and
    14% of the 256x256 tile rows are padding), K in {2048, 8192}, W in {1, 2}, S/P = 1/16 to
    1/2; N is the multiple of 256 nearest to the tile count, so S is recomputed (set "padded")."""
    rows = []
    for K in (2048, 8192):
        for W in (1, 2):
            for S in range(16, P // 2 + 1, 16):
                for M in (300, 450, 700, 1100):
                    rows_m = math.ceil(M / 256)
                    cols = max(1, round((W * P + S) / rows_m))
                    N = 256 * cols
                    T = rows_m * cols
                    rows.append(dict(M=M, N=N, K=K, bias=False, T=T, W=T // P, S=T % P, fill=(T % P) / P, set="padded"))
    d = pd.DataFrame(rows).drop_duplicates(["M", "N", "K"])
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "shapes.csv", index=False)
    print(f"{len(d)} padded shapes")


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for K in KS:
        for W in WS:
            for S in SS:
                for M in MS:
                    rows_m = M // 256
                    N = 256 * (W * P + S) // rows_m
                    rows.append(row(M, N, K, W, S, "grid"))
                    if M == 1024 and W in (1, 2) and K in (2048, 8192):
                        rows.append(row(M, N - 72, K, W, S, "grid_ragged"))
    d = pd.DataFrame(rows)
    d.to_csv(out / "shapes.csv", index=False)
    print(d.groupby("set").size().to_string(), f"\n{len(d)} shapes, N from {d.N.min()} to {d.N.max()}")


if __name__ == "__main__":
    padded(sys.argv[2]) if sys.argv[1] == "--padded" else main(sys.argv[1])
