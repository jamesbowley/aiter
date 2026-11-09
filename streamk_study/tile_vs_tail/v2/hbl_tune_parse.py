"""Parse hbl_tune.sh output: per N and grid mode, hipBLASLt's fastest solution overall, its
fastest at a 256x256 tile, and its fastest with a tile larger than 256x256.

    python hbl_tune_parse.py runs/hbl_tune     # writes hbl_tune.csv and prints a summary
"""

import math
import re
import sys
from pathlib import Path

import pandas as pd

P, M = 256, 1024
MT = re.compile(r"MT(\d+)x(\d+)x(\d+)")
SK = re.compile(r"_SK(\d+)_")


def parse(path):
    rows, us = [], None
    for line in path.read_text(errors="replace").splitlines():
        s = line.strip()
        if s.startswith("T,N,"):
            us = float(s.split(",")[-1])
        elif s.startswith("--kernel name:") and us is not None:
            name = s.split(":", 1)[1].strip()
            mt, sk = MT.search(name), SK.search(name)
            if mt:
                a, b, k = map(int, mt.groups())
                rows.append(dict(tile_n=a, tile_m=b, depth_k=k, sk=int(sk.group(1)) if sk else 0, us=us, kernel=name))
            us = None
    return pd.DataFrame(rows)


def main(run):
    run = Path(run)
    parts = []
    for f in sorted(run.glob("*_*.txt")):
        mode, n = f.stem.rsplit("_", 1)
        d = parse(f)
        if len(d):
            parts.append(d.assign(mode=mode, N=int(n)))
    d = pd.concat(parts)
    d = d[d.us > 0]
    d["area"] = d.tile_n * d.tile_m / 65536
    d["tiles"] = [math.ceil(n / a) * math.ceil(M / b) for n, a, b in zip(d.N, d.tile_n, d.tile_m)]
    d.to_csv(run / "hbl_tune.csv.gz", index=False)

    out = []
    for (n, mode), g in d.groupby(["N", "mode"]):
        best = g.loc[g.us.idxmin()]
        t256 = g[(g.tile_n == 256) & (g.tile_m == 256)]
        big = g[g.area > 1]
        bb = big.loc[big.us.idxmin()] if len(big) else None
        out.append(dict(
            N=n, S=(math.ceil(n / 256) * 4) % P, mode=mode,
            best_us=best.us, best_tile=f"{best.tile_n}x{best.tile_m}", best_tiles=best.tiles,
            t256_us=t256.us.min() if len(t256) else float("nan"),
            big_us=bb.us if bb is not None else float("nan"),
            big_tile=f"{bb.tile_n}x{bb.tile_m}" if bb is not None else "",
            big_tiles=bb.tiles if bb is not None else 0,
        ))
    s = pd.DataFrame(out)
    print(s.to_string(index=False))
    s.to_csv(run / "hbl_tune_summary.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1])
