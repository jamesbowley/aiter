"""Checks of the temporary v32 sub-tile tail port onto aiter's Stream-K kernel.

1. Correctness: sub-tile tails (dp, owner and reduce-scatter) and whole-tile plans
   against torch on even and ragged shapes.
2. Whole-tile builds are instruction-identical to the pre-port module
   (ref/gemm_a16w16_streamk_pre_port.py with ref/gemm_a16w16_pre_port.py's launcher),
   so the baseline Stream-K numbers carry over.

    source ../aiter_tune/env_stock.sh; GPU=3 python check_port.py [--skip-isa]
"""

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
SK_MOD = "aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16_streamk"


def plan(bm, bn, bk, policy, fixup="owner", n=None, exchange="wt", tail=None):
    c = {
        "BLOCK_M": bm,
        "BLOCK_N": bn,
        "BLOCK_K": bk,
        "GROUP_SIZE_M": 4,
        "STREAMK_POLICY": policy,
        "STREAMK_FIXUP": fixup,
        "SPLIT_N": n,
        "STREAMK_EXCHANGE": exchange,
    }
    if tail:
        c.update(TAIL_BLOCK_M=tail[0], TAIL_BLOCK_N=tail[1])
    return c


# (M, N, K, config)
CASES = [
    # 288 tiles of 256x256: S = 32
    (4096, 4608, 2048, plan(256, 256, 64, "dp", tail=(128, 128))),
    (4096, 4608, 2048, plan(256, 256, 64, "dp", tail=(128, 64))),
    (4096, 4608, 2048, plan(256, 256, 64, "one_tile", "owner", 2, tail=(128, 128))),
    (4096, 4608, 2048, plan(256, 256, 64, "one_tile", "rs", 2, tail=(128, 128))),
    # 272 tiles of 256x256: S = 16
    (4096, 4352, 2048, plan(256, 256, 64, "dp", tail=(64, 64))),
    (4096, 4352, 2048, plan(256, 256, 64, "one_tile", "rs", 4, tail=(128, 128))),
    (4096, 4352, 2048, plan(256, 256, 64, "one_tile", "owner", 4, tail=(128, 128))),
    (4096, 4352, 2048, plan(256, 256, 64, "one_tile", "rs", 2, tail=(128, 64))),
    # ragged M, N and K: 288 tiles, S = 32
    (4000, 4600, 1008, plan(256, 256, 64, "dp", tail=(128, 128))),
    (4000, 4600, 1008, plan(256, 256, 64, "one_tile", "owner", 2, tail=(128, 128))),
    (4000, 4600, 1008, plan(256, 256, 64, "one_tile", "rs", 2, tail=(128, 128))),
    # 128x256 main tile: S = 128, then S = 32
    (4096, 9216, 4096, plan(128, 256, 64, "dp", tail=(128, 128))),
    (4096, 8448, 4096, plan(128, 256, 64, "one_tile", "owner", 2, tail=(64, 128))),
    # 128x128x128 main tile: S = 48
    (6144, 6272, 3072, plan(128, 128, 128, "dp", tail=(64, 64))),
    # whole-tile plans still run
    (4096, 4608, 2048, plan(256, 256, 64, "dp")),
    (4096, 4608, 2048, plan(256, 256, 64, "one_tile", "owner", 2)),
    (4096, 4608, 2048, plan(256, 256, 64, "one_tile", "rs", 8, exchange="l2")),
]

# Whole-tile configs whose ISA must not change.
ISA_CASES = [
    (4096, 4608, 2048, plan(256, 256, 64, "dp")),
    (4096, 4608, 2048, plan(256, 256, 64, "one_tile", "owner", 2)),
    (4096, 4608, 2048, plan(256, 256, 64, "one_tile", "rs", 4)),
    (4096, 4608, 2048, plan(256, 256, 64, "one_tile", "rs", 8, exchange="l2")),
    (4096, 6656, 2048, plan(256, 256, 64, "spread")),
    (4000, 4600, 1008, plan(256, 256, 64, "one_tile", "owner", 2)),
    (6144, 6272, 3072, plan(128, 128, 128, "one_tile", "rs", 4)),
]


def tensors(M, N, K, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16, generator=g)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16, generator=g)
    return x, w


def check_correctness():
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import _gemm_a16w16_gfx950_streamk
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic import gemm_a16w16_streamk as sk

    bad = 0
    for M, N, K, cfg in CASES:
        name = sk.kernel_name(cfg)
        assert sk.parse_kernel_name(name) == cfg, (name, sk.parse_kernel_name(name), cfg)
        x, w = tensors(M, N, K)
        ref = (x.float() @ w.float().T).to(torch.bfloat16)
        y = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        y.fill_(float("nan"))
        _gemm_a16w16_gfx950_streamk(x, w, y, None, dict(cfg))
        torch.cuda.synchronize()
        err = (y.float() - ref.float()).abs().max().item()
        ok = torch.allclose(y.float(), ref.float(), atol=1.0, rtol=2e-2) and not y.isnan().any()
        bad += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {M}x{N}x{K} {name} max_abs_err={err:.3g}")
    return bad


class _Capture:
    """Stands in for _KERNEL_MAP['streamk'] and records the compiled kernel."""

    def __init__(self, kernel, drop=()):
        self.kernel, self.drop, self.compiled = kernel, drop, None

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            for key in self.drop:
                kwargs.pop(key, None)
            self.compiled = self.kernel[grid](*args, **kwargs)
            return self.compiled

        return launch


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def normalized_isa(compiled):
    asm = compiled.asm["amdgcn"]
    lines = []
    for line in asm.splitlines():
        if not line.startswith("\t") or line.startswith("\t."):
            continue
        line = re.sub(r"_gemm_a16w16_streamk_kernel\w*", "KERNEL", line)
        lines.append(line.split(";")[0].rstrip())
    return lines


def isa_of(launcher, sk, M, N, K, cfg, drop=()):
    cap = _Capture(sk._KERNEL_MAP["streamk"], drop)
    saved = sk._KERNEL_MAP["streamk"]
    sk._KERNEL_MAP["streamk"] = cap
    try:
        x, w = tensors(M, N, K)
        y = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
        launcher(x, w, y, None, dict(cfg))
        torch.cuda.synchronize()
    finally:
        sk._KERNEL_MAP["streamk"] = saved
    return normalized_isa(cap.compiled)


def check_isa():
    from aiter.ops.triton.gemm.basic import gemm_a16w16 as new_wrapper
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic import gemm_a16w16_streamk as new_sk

    old_sk = load(SK_MOD + "_pre_port", HERE / "ref/gemm_a16w16_streamk_pre_port.py")
    old_wrapper = load("gemm_a16w16_pre_port", HERE / "ref/gemm_a16w16_pre_port.py")
    bad = 0
    for M, N, K, cfg in ISA_CASES:
        new = isa_of(new_wrapper._gemm_a16w16_gfx950_streamk, new_sk, M, N, K, cfg)
        # The old launcher imports the Stream-K module by name: point it at the old one.
        package = sys.modules[SK_MOD.rsplit(".", 1)[0]]
        sys.modules[SK_MOD] = old_sk
        package.gemm_a16w16_streamk = old_sk
        try:
            old = isa_of(old_wrapper._gemm_a16w16_gfx950_streamk, old_sk, M, N, K, cfg)
        finally:
            sys.modules[SK_MOD] = new_sk
            package.gemm_a16w16_streamk = new_sk
        same = new == old
        bad += not same
        print(
            f"{'SAME' if same else 'DIFF'} {M}x{N}x{K} {new_sk.kernel_name(cfg)} "
            f"({len(new)} vs {len(old)} instructions)"
        )
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-isa", action="store_true")
    ap.add_argument("--skip-correctness", action="store_true")
    args = ap.parse_args()
    bad = 0
    if not args.skip_correctness:
        bad += check_correctness()
    if not args.skip_isa:
        bad += check_isa()
    print("ALL OK" if bad == 0 else f"{bad} FAILED")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
