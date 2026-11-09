"""Machine-code gate: a variant with its feature off must build gemm_a16w16.py's code.

    python gate.py uneven     power-of-two tiles through gemm_a16w16_uneven.py
    python gate.py subtile    shapes with no tail through gemm_a16w16_subtile.py

Each tile is built in 4 variants (plain, bias, ragged M and N, ragged K) through the
base and through the variant, and their .amdgcn compared after dropping the kernel's
name, .Ltmp labels and debug lines. Also prints LDS bytes, VGPR / AGPR counts and
spills of each build.
"""

import re
import sys

import torch

from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic import gemm_a16w16 as base

TILES = [(bm, bn) for bm in (64, 128, 256) for bn in (64, 128, 256)]
# (M, N, K, bias): plain, bias, ragged M and N, ragged K
VARIANTS = [
    ("plain", 0, 0, 0, False),
    ("bias", 0, 0, 0, True),
    ("ragged_mn", -24, -40, 0, False),
    ("ragged_k", 0, 0, 72, False),
]


def capture(mod, name):
    """Wrap mod.<name> so that each launch's CompiledKernel is recorded."""
    orig = getattr(mod, name)
    got = []

    class Proxy:
        def __getitem__(self, grid):
            launch = orig[grid]

            def run(*a, **k):
                r = launch(*a, **k)
                got.append(r)
                return r

            return run

        def __getattr__(self, item):
            return getattr(orig, item)

    setattr(mod, name, Proxy())
    return got


def normalize(asm, kernel_name):
    out = []
    for line in asm.splitlines():
        s = line.strip()
        if not s or s.startswith(";") or s.startswith(".loc") or s.startswith(".file"):
            continue
        if re.match(r"^\.Ltmp\d+:$", s) or ".Ltmp" in s and s.startswith(".") and "debug" in s:
            continue
        if s.startswith(".section") and "debug" in s:
            break  # everything after the first debug section is DWARF
        s = s.split(";")[0].strip()  # trailing comments are padded to the name's length
        s = s.replace(kernel_name, "KERNEL")
        s = re.sub(r"\.Ltmp\d+", ".Ltmp", s)
        s = re.sub(r"\.Lfunc_end\d+", ".Lfunc_end", s)
        out.append(s)
    return out


def stats(k):
    m = k.metadata
    asm = k.asm["amdgcn"]
    vg = re.search(r"\.vgpr_count:\s+(\d+)", asm)
    ag = re.search(r"\.agpr_count:\s+(\d+)", asm)
    sp = re.search(r"\.vgpr_spill_count:\s+(\d+)", asm)
    return dict(
        shared=m.shared,
        vgpr=int(vg.group(1)) if vg else None,
        agpr=int(ag.group(1)) if ag else None,
        spill=int(sp.group(1)) if sp else None,
    )


def problem(tile, v, waves=1):
    _, dm, dn, dk, with_bias = v
    bm, bn = tile
    # Enough tiles for one full wave of 256 CUs (no tail at any tile).
    gm = 16
    gn = 16 * waves
    M, N, K = bm * gm + dm, bn * gn + dn, 1024 + dk
    x = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    y = torch.empty(M, N, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, device="cuda", dtype=torch.bfloat16) if with_bias else None
    return x, w, y, b


def main():
    kind = sys.argv[1]
    got_base = capture(base, "_gemm_a16w16_compute_bound_kernel")
    if kind == "uneven":
        from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic import gemm_a16w16_uneven as mod

        got_var = capture(mod, "_gemm_a16w16_uneven_kernel")
        name = "_gemm_a16w16_uneven_kernel"

        def run_var(x, w, y, b, tile):
            mod.gemm_a16w16_uneven(x, w, y, b, tile=tile)

    else:
        from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic import gemm_a16w16_subtile as mod

        got_var = capture(mod, "_gemm_a16w16_subtile_kernel")
        name = "_gemm_a16w16_subtile_kernel"

        def run_var(x, w, y, b, tile):
            mod.gemm_a16w16_subtile(x, w, y, b, tile=tile, tail=None)

    same = diff = 0
    for tile in TILES:
        for v in VARIANTS:
            x, w, y, b = problem(tile, v)
            base.gemm_a16w16_compute_bound(x, w, y, b, tile=tile)
            run_var(x, w, y, b, tile)
            kb, kv = got_base[-1], got_var[-1]
            a = normalize(kb.asm["amdgcn"], "_gemm_a16w16_compute_bound_kernel")
            c = normalize(kv.asm["amdgcn"], name)
            ok = a == c
            same += ok
            diff += not ok
            sb, sv = stats(kb), stats(kv)
            print(
                f"{tile[0]}x{tile[1]} {v[0]:10s} {'identical' if ok else 'DIFFERENT'}"
                f"  base {sb}  {kind} {sv}"
            )
            if not ok:
                n = sum(1 for p, q in zip(a, c) if p != q) + abs(len(a) - len(c))
                print(f"    {n} lines differ of {len(a)} / {len(c)}")
    print(f"{same} identical, {diff} different")
    sys.exit(1 if diff else 0)


if __name__ == "__main__":
    main()
