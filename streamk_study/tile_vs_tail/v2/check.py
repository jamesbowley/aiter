"""Quick correctness checks of the uneven and sub-tile variants against torch.

    python check.py uneven [TILE ...]      e.g. 224x224 256x256
    python check.py subtile [CFG ...]      e.g. 256x256:128x128
"""

import sys

import torch

SHAPES = [
    (1024, 1024, 1024),
    (1000, 1000, 1000),  # ragged M, N, K
    (4096, 4096, 512),
    (4096, 5120, 2880),
    (777, 1336, 1032),
    (8192, 2048, 4096),
    # more than one wave of 256x256 tiles, with a partial last wave
    (4352, 4096, 1024),  # T = 272, S = 16
    (4100, 5000, 1032),  # ragged M, N, K; T = 340, S = 84
    (2048, 9000, 520),  # ragged N and K; T = 288, S = 32
]


def ref(x, w, bias):
    y = x.float() @ w.float().T
    if bias is not None:
        y = y + bias.float()
    return y


def run(fn, label, shapes=SHAPES):
    fails = 0
    for M, N, K in shapes:
        for dtype in (torch.bfloat16, torch.float16):
            for with_bias in (False, True):
                x = torch.randn(M, K, device="cuda", dtype=dtype) / K**0.25
                w = torch.randn(N, K, device="cuda", dtype=dtype) / K**0.25
                bias = torch.randn(N, device="cuda", dtype=dtype) if with_bias else None
                y = torch.full((M, N), float("nan"), device="cuda", dtype=dtype)
                try:
                    fn(x, w, y, bias)
                except ValueError as e:
                    print(f"{label} {M}x{N}x{K}: skipped ({e})")
                    break
                r = ref(x, w, bias)
                err = ((y.float() - r).abs() / (r.abs() + 1)).max().item()
                ok = err < 2e-2 and not torch.isnan(y).any()
                fails += not ok
                if not ok:
                    print(f"FAIL {label} {M}x{N}x{K} {dtype} bias={with_bias} err={err}")
    print(f"{label}: {'ok' if not fails else f'{fails} failures'}")
    return fails


def main():
    kind, specs = sys.argv[1], sys.argv[2:]
    fails = 0
    if kind == "uneven":
        from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic import gemm_a16w16_uneven as u

        for spec in specs:
            tile = tuple(int(v) for v in spec.split("x"))
            fails += run(lambda x, w, y, b: u.gemm_a16w16_uneven(x, w, y, b, tile=tile), spec)
    else:
        from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic import gemm_a16w16_subtile as s

        for spec in specs:
            t, tail = spec.split(":")
            tile = tuple(int(v) for v in t.split("x"))
            tail = tuple(int(v) for v in tail.split("x"))
            fails += run(
                lambda x, w, y, b: s.gemm_a16w16_subtile(x, w, y, b, tile=tile, tail=tail), spec
            )
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
