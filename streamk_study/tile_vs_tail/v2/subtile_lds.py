"""LDS bytes, registers and spills of sub-tile builds against the whole-tile build.

The two tile paths are branches of one kernel, so their LDS must overlap: a build's
shared bytes must equal the whole tile's.
"""

import torch

import gate
from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic import gemm_a16w16 as base
from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic import gemm_a16w16_subtile as st

got_b = gate.capture(base, "_gemm_a16w16_compute_bound_kernel")
got_s = gate.capture(st, "_gemm_a16w16_subtile_kernel")
fails = 0
for tile in [(256, 256), (256, 128), (128, 256), (128, 128)]:
    bm, bn = tile
    # 17 x 16 tiles: S = 16 at 256 CUs, so every legal sub-tile fits one wave.
    M, N, K = 17 * bm, 16 * bn, 1024
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    y = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    base.gemm_a16w16_compute_bound(x, w, y, tile=tile)
    sb = gate.stats(got_b[-1])
    for tm in (64, 128, 256):
        for tn in (64, 128, 256):
            if bm % tm or bn % tn or (tm, tn) == tile:
                continue
            st.gemm_a16w16_subtile(x, w, y, tile=tile, tail=(tm, tn))
            ss = gate.stats(got_s[-1])
            ok = ss["shared"] == sb["shared"] and not ss["spill"]
            fails += not ok
            print(f"{bm}x{bn} tail {tm}x{tn}: {'ok' if ok else 'FAIL'}  base {sb}  subtile {ss}")
print(f"{fails} failures")
