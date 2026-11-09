"""The hipBLASLt kernel torch launches for each shape of a run, and its launch grid.

aiter's torch backend (tuned_gemm.torch_gemm) is F.linear for a bf16 output, so this calls
F.linear once to warm, then once under the torch profiler, and reads the kernel name and grid
from the exported trace. The tile is the kernel name's MT<a>x<b>x<k>; torch runs the
transposed problem (C^T = W X^T), so <a> tiles N and <b> tiles M.

    python torch_kernels.py runs/<name>     # writes runs/<name>/torch_kernels.csv
"""

import json
import math
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

MT = re.compile(r"MT(\d+)x(\d+)x(\d+)")
SK = re.compile(r"_SK(\d+)_")


def launch(M, N, K, bias):
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, device="cuda", dtype=torch.bfloat16) if bias else None
    F.linear(x, w, b)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        F.linear(x, w, b)
        torch.cuda.synchronize()
    with tempfile.NamedTemporaryFile(suffix=".json") as f:
        prof.export_chrome_trace(f.name)
        events = json.load(open(f.name))["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    gemm = [e for e in kernels if "Cijk_" in e["name"]] or kernels
    e = gemm[0]
    grid = e.get("args", {}).get("grid", [None])
    block = e.get("args", {}).get("block", [None])
    return e["name"], grid, block, len(kernels)


def main(run):
    run = Path(run)
    shapes = pd.read_csv(run / "shapes.csv")
    out = []
    for r in shapes.itertuples():
        name, grid, block, nk = launch(int(r.M), int(r.N), int(r.K), bool(r.bias))
        m = MT.search(name)
        sk = SK.search(name)
        a, b, bk = map(int, m.groups()) if m else (None, None, None)
        tiles = math.ceil(r.N / a) * math.ceil(r.M / b) if m else None
        grid_x = grid[0] if isinstance(grid, list) else grid
        wgs = None
        if isinstance(grid, list) and isinstance(block, list) and block[0]:
            wgs = (grid[0] // block[0]) * grid[1] * grid[2]
        out.append(
            dict(
                M=r.M, N=r.N, K=r.K, bias=r.bias, hbl_kernel=name, hbl_bm_n=a, hbl_bn_m=b,
                hbl_bk=bk, hbl_sk=sk.group(1) if sk else None, hbl_tiles=tiles,
                hbl_grid=json.dumps(grid), hbl_block=json.dumps(block), hbl_wgs=wgs,
                hbl_grid_x=grid_x, nkernels=nk,
            )
        )
    d = pd.DataFrame(out)
    d.to_csv(run / "torch_kernels.csv", index=False)
    print(f"{len(d)} shapes; one workgroup per tile on {int((d.hbl_wgs == d.hbl_tiles).sum())}")


if __name__ == "__main__":
    main(sys.argv[1])
