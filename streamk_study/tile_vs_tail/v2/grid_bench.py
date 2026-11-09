"""Time every candidate of every method (candidates.py's menus) for a list of shapes, with
aiter's own timing (test_common.run_iters_rotate under torch.profiler, then get_trace_perf:
device time of every kernel in an iteration, IQR outliers dropped), but with the per-shape
work done once instead of once per candidate:

    inputs     torch.randn with seed 0, as gemm_a16w16_tune.generate_data
    rotation   perftest's rule: min(iters, cache_size / input bytes) copies of (x, w, bias),
               cache_size = min(L2 size * 64 * 128, 0.9 * free memory); iteration i uses copy
               i mod n, so no iteration finds its inputs in L2 or MALL from an earlier one
    reference  fp32 F.linear (+ bias), as get_gemm_ref
    check      share of elements failing isclose(rtol = atol = 5e-2), as checkAllclose
    call       gemm_a16w16(x, w, bias=b, dtype=bf16, backend="gluon", kernel_type, config),
               as tuned_gemm.gluon_gemm

Resumes: shapes already in the output CSV are skipped, so a crash only loses one shape.

    python grid_bench.py SHAPES.csv OUT.csv [--warmup 10] [--iters 101]
"""

import argparse
import copy
import math
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
import torch.profiler as tpf

from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.test_common import get_trace_perf, run_iters, run_iters_rotate
from aiter.tuned_gemm import parse_gluon_kernel_name

sys.path.insert(0, str(Path(__file__).resolve().parent))
from candidates import candidates  # noqa: E402

KEY = ["M", "N", "K", "bias"]
COLS = KEY + ["kernelName", "method", "us", "err_ratio", "n_rotate", "error"]


def rotate_count(args, iters):
    nbytes = sum(a.nbytes for a in args if isinstance(a, torch.Tensor)) + 1
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    free = torch.cuda.mem_get_info()[0]
    cache = max(min(getattr(props, "L2_cache_size", 4096 * 1024) * 64 * 128, free * 0.9 - nbytes), 0)
    return max(1, min(iters, int((cache + nbytes - 1) // nbytes)))


def bench_shape(M, N, K, bias, warmup, iters, only=None):
    torch.manual_seed(0)
    x = torch.randn((M, K), device="cuda").to(torch.bfloat16)
    w = torch.randn((N, K), device="cuda").to(torch.bfloat16)
    b = torch.randn(N, device="cuda").to(torch.bfloat16) if bias else None
    ref = F.linear(x.float(), w.float())
    if b is not None:
        ref = ref + b.float()
    ref = ref.to(torch.bfloat16)
    n = rotate_count((x, w, b), iters)
    sets = [copy.deepcopy((x, w, b)) for _ in range(n - 1)] + [(x, w, b)]
    rotate = [(s, {}) for s in sets]
    rows = []
    for name, method, _ in candidates(M, N, K):
        if only is not None and name not in only:
            continue
        kind, cfg = parse_gluon_kernel_name(name)

        def fn(xx, ww, bb):
            return gemm_a16w16(xx, ww, bias=bb, dtype=torch.bfloat16, backend="gluon", kernel_type=kind, config=cfg)

        row = dict(M=M, N=N, K=K, bias=bias, kernelName=name, method=method, us=math.nan,
                   err_ratio=math.nan, n_rotate=n, error="")
        try:
            out = fn(x, w, b)
            torch.cuda.synchronize()
            close = torch.isclose(out, ref, rtol=5e-2, atol=5e-2)
            row["err_ratio"] = 1.0 - close.float().mean().item()
            del out, close
            run_iters(warmup, fn, x, w, b)
            torch.cuda.synchronize()
            # Device activity only: get_trace_perf reads only device events, and host events
            # cost 40 ms per candidate to parse.
            with tpf.profile(activities=[tpf.ProfilerActivity.CUDA]) as prof:
                run_iters_rotate(iters, fn, rotate)
                torch.cuda.synchronize()
            row["us"] = get_trace_perf(prof, iters, warm_iter=1)
        except Exception as e:  # unsupported config or launch failure: recorded, not fatal
            row["error"] = f"{type(e).__name__}: {e}"[:200]
        rows.append(row)
    del sets, rotate, x, w, b, ref
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("shapes")
    ap.add_argument("out")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=101)
    ap.add_argument("--only", help="CSV of M,N,K,bias,kernelName: time only these candidates")
    a = ap.parse_args()
    shapes = pd.read_csv(a.shapes)
    shapes["bias"] = shapes.bias.astype(str).str.lower().eq("true")
    only = None
    if a.only:
        o = pd.read_csv(a.only)
        o["bias"] = o.bias.astype(str).str.lower().eq("true")
        only = {k: set(g.kernelName) for k, g in o.groupby(KEY)}
    out = Path(a.out)
    done = set()
    if out.exists():
        d = pd.read_csv(out)
        d["bias"] = d.bias.astype(str).str.lower().eq("true")
        done = set(map(tuple, d[KEY].drop_duplicates().itertuples(index=False)))
    todo = [r for r in shapes[KEY].drop_duplicates().itertuples(index=False) if tuple(r) not in done]
    print(f"{len(done)} shapes done, {len(todo)} to go, gpu {os.environ.get('HIP_VISIBLE_DEVICES')}", flush=True)
    for i, (M, N, K, bias) in enumerate(todo):
        t0 = time.time()
        sel = None if only is None else only.get((M, N, K, bias), set())
        rows = bench_shape(int(M), int(N), int(K), bool(bias), a.warmup, a.iters, sel)
        pd.DataFrame(rows, columns=COLS).to_csv(out, mode="a", header=not out.exists(), index=False)
        ok = sum(r["error"] == "" for r in rows)
        print(f"[{i + 1}/{len(todo)}] {M}x{N}x{K} bias={bias}: {ok}/{len(rows)} candidates, "
              f"{time.time() - t0:.1f} s", flush=True)


if __name__ == "__main__":
    main()
