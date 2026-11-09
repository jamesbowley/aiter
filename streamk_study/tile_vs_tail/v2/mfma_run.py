"""Run given kernels on one shape for rocprofv3 (counters or thread trace): each kernel
REPS times in turn on rotating inputs, after one untimed call that compiles or loads it.
Prints the order, so the profiler's dispatches can be matched to kernels.

    python mfma_run.py M N K KERNEL [KERNEL ...]
"""

import math
import os
import sys

import torch

from aiter.ops.triton.gemm.basic.gemm_a16w16 import gemm_a16w16
from aiter.tuned_gemm import parse_gluon_kernel_name

REPS = int(os.environ.get("MFMA_REPS", "10"))


def main():
    M, N, K = map(int, sys.argv[1:4])
    names = sys.argv[4:]
    torch.manual_seed(0)
    x = torch.randn((M, K), device="cuda").to(torch.bfloat16)
    w = torch.randn((N, K), device="cuda").to(torch.bfloat16)
    # MFMA_SETS: copies of the inputs to rotate through; "cold" = enough to cover 1 GB, so no
    # call finds its inputs in L2 or MALL; 1 = warm, the same inputs every call. Default 4.
    want = os.environ.get("MFMA_SETS", "4")
    nbytes = x.nbytes + w.nbytes
    n = min(64, max(2, math.ceil(2**30 / nbytes))) if want == "cold" else int(want)
    sets = [(x, w)] if n == 1 else [(x.clone(), w.clone()) for _ in range(n)]
    fns = []
    for name in names:
        kind, cfg = parse_gluon_kernel_name(name)
        fns.append(lambda a, b, kind=kind, cfg=cfg: gemm_a16w16(a, b, bias=None, dtype=torch.bfloat16,
                                                              backend="gluon", kernel_type=kind, config=cfg))
    for fn in fns:
        fn(x, w)
    torch.cuda.synchronize()
    for name, fn in zip(names, fns):
        print(f"RUN {name} {REPS}", flush=True)
        for i in range(REPS):
            fn(*sets[i % len(sets)])
        torch.cuda.synchronize()


if __name__ == "__main__":
    main()
