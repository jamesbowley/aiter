"""M = 1024, K = 7168, N swept so the leftover S after one full wave of 256x256 tiles grows
from 0 to a full wave: our whole 256x256 tiles, our best finer tile, our best Stream-K plan
and hipBLASLt (with the tile it picks), all timed in one session with do_bench.

    python sweep_leftover.py runs/sweep_leftover      # writes sweep.csv there
"""

import math
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from triton.testing import do_bench

import torch_kernels
from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.tuned_gemm import parse_gluon_kernel_name

M, K = 1024, 7168
NS = list(range(16384, 32768 + 1, 1024))


def menus():
    p = pd.read_csv("runs/full/profile.csv")
    names = p.kernelName.dropna().unique()
    uneven = sorted(n for n in names if n.startswith("gluon_uneven_"))
    sk = sorted(n for n in names if n.startswith("gluon_streamk_256x256x64_") and not n.endswith("_dp"))
    pow2 = sorted(n for n in names if n.startswith("gluon_compute_bound_") and n != "gluon_compute_bound_256x256")
    return uneven, sk, pow2


def timer(x, w, y, name):
    if name == "torch":
        fn = lambda: F.linear(x, w)  # noqa: E731
    else:
        kind, cfg = parse_gluon_kernel_name(name)
        fn = lambda: gemm_a16w16(x, w, None, torch.bfloat16, y, backend="gluon", kernel_type=kind, config=cfg)  # noqa: E731
    try:
        fn()
        torch.cuda.synchronize()
    except Exception:
        return float("nan")
    return min(do_bench(fn, warmup=25, rep=100) for _ in range(2)) * 1000


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    uneven, sk, pow2 = menus()
    rows = []
    for N in NS:
        x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        y = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        hbl_name = torch_kernels.launch(M, N, K, False)[0]
        a, b, _ = map(int, torch_kernels.MT.search(hbl_name).groups())
        for kind, names in (("base", ["gluon_compute_bound_256x256"]), ("torch", ["torch"]), ("pow2", pow2), ("uneven", uneven), ("streamk", sk)):
            for name in names:
                rows.append(dict(N=N, kind=kind, name=name, us=timer(x, w, y, name), hbl_tile_n=a, hbl_tile_m=b))
        best = pd.DataFrame([r for r in rows if r["N"] == N]).groupby("kind").us.min()
        print(N, math.ceil(N / 256) * 4, best.round(1).to_dict(), f"hipBLASLt MT{a}x{b}", flush=True)
        pd.DataFrame(rows).to_csv(out / "sweep.csv", index=False)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "runs/sweep_leftover")
