# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Gluon BF16/FP16 GEMM for gfx950, compute-bound variant with uneven quadrant halves.

gemm_a16w16.py's kernel with the tile sides any multiple of 32 from 64 to 320, up to
256x256 of accumulators, from the tutorial's v9_uneven2 (kernels/gemm/intra_wave/a16w16/
v9_uneven2): each half of a side is a power-of-two block plus an optional extra block of
32 or 64 (224 = 128 + 96 = 128 + (64 + 32)). Power-of-two tiles compile to
gemm_a16w16.py's code. Everything below is gemm_a16w16.py's description.

Ported from ROCm/gfx950-gluon-tutorials (ef48b5f),
kernels/gemm/intra_wave/a16w16/v9_beyond_hotloop: one wave per SIMD (4 warps), a
256x256x64 tile split into four 128x128 quadrants, a 2-deep async-copy LDS pipeline
with local prefetch, the K loop unrolled by 2, MFMA accumulators pinned to AGPRs with
``cd_regclass="a"``, and an XCD-aware PID remap. Its instruction schedule comes from
Triton's MFMA scheduler, enabled per launch with ``schedule_hint="mfma-schedule"``;
``AITER_MFMA_SCHED=0`` turns it off for A/B runs, which costs about 15%. The kernel
needs an upstream Triton newer than triton-lang/triton#12209 (``3b0c7f081``), which
brought the scheduler; ``mfma(..., cd_regclass=)`` (triton-lang/triton#11792) was
already there. The tutorial explains every step:
https://github.com/ROCm/gfx950-gluon-tutorials/tree/main/kernels/gemm/intra_wave/a16w16

The tile is any (BLOCK_M, BLOCK_N) with sides 64, 128 or 256, with BLOCK_K 64; every
layout is derived from it. Which tile a shape runs with comes from the gluon GEMM
config (``configs/gfx950/gluon/gemm/gemm_a16w16``): a tuned config names it, the
default config's cost model picks the tile whose workgroup waves finish first
(``pick_gemm_tile``; its per-tile throughputs live in the JSON). Ragged shapes
are handled per dimension, each specialized at compile time: a ragged M or N clamps
the loads of the last tile to the matrix edge and masks its stores; a ragged K
(EVEN_K false) consumes the K tail first, unpipelined, so the hot loop stays
unmasked. The kernel needs a TN problem (x row-major, w row-major (N, K)) with N
and K multiples of 8 (16-byte loads and stores) and K at least 256.
``unsupported_reason`` says why a problem is outside that.
"""

import functools
import math
import os

import torch
import triton
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd import cdna4 as ttgl_cdna4
from triton.runtime.jit import constexpr_function

from aiter.ops.triton.utils.gemm_config_utils import get_gemm_config, pick_gemm_tile

BLOCK_K = 64
TILE_SIDES = tuple(range(64, 321, 32))  # every one splits into two valid halves
MAX_ACC = 256 * 256  # accumulator elements of a tile: 256 fp32 AGPRs per lane
NUM_WARPS = 4
NUM_XCDS = 8
GROUP_SIZE_M = 4


@constexpr_function
def compute_gload_layout(shared_layout, load_contig, num_warps, threads_per_warp=64):
    """The coalesced global-load linear layout that matches a PaddedSharedLayout.

    Partitions the shared layout's offset bases into register, lane and warp bases so
    that each warp writes contiguous LDS offsets (what CoalesceAsyncCopy derives for a
    padded encoding). At 256x256x64 this reproduces the tutorial's hand-written
    layouts exactly; for the other tiles it follows BLOCK_M, BLOCK_N and BLOCK_K.
    """
    bases = [list(b) for b in shared_layout.offset_bases]
    rank = len(shared_layout.shape)
    n_reg = int(math.log2(load_contig))
    n_lane = int(math.log2(threads_per_warp))
    n_warp = int(math.log2(num_warps))
    reg = bases[:n_reg]
    lane = bases[n_reg : n_reg + n_lane]
    warp = bases[n_reg + n_lane : n_reg + n_lane + n_warp]
    while len(warp) < n_warp:  # zero-pad (broadcast) if we ran out of bases
        warp.append([0] * rank)
    reg += bases[n_reg + n_lane + n_warp :]
    return gl.DistributedLinearLayout(
        reg_bases=reg,
        lane_bases=lane,
        warp_bases=warp,
        block_bases=[],
        shape=list(shared_layout.shape),
    )


@gluon.jit
def _get_pids(
    M,
    N,
    BM: gl.constexpr,
    BN: gl.constexpr,
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
):
    """XCD-aware PID remapping + GROUP_SIZE_M swizzle. Active at any grid_mn."""
    pid = gl.program_id(axis=0)
    num_pid_m = gl.cdiv(M, BM)
    num_pid_n = gl.cdiv(N, BN)

    if NUM_XCDS != 1:
        ## pid remapping on xcds
        # Number of pids per XCD in the new arrangement
        pids_per_xcd = (GRID_MN + NUM_XCDS - 1) // NUM_XCDS
        # When GRID_MN cannot divide NUM_XCDS, some xcds will have
        # pids_per_xcd pids, the other will have pids_per_xcd - 1 pids.
        # We calculate the number of xcds that have pids_per_xcd pids as
        # tall_xcds
        tall_xcds = GRID_MN % NUM_XCDS
        tall_xcds = NUM_XCDS if tall_xcds == 0 else tall_xcds
        # Compute current XCD and local pid within the XCD
        xcd = pid % NUM_XCDS
        local_pid = pid // NUM_XCDS
        # Calculate new pid based on the new grouping
        if xcd < tall_xcds:
            pid = xcd * pids_per_xcd + local_pid
        else:
            pid = (
                tall_xcds * pids_per_xcd
                + (xcd - tall_xcds) * (pids_per_xcd - 1)
                + local_pid
            )

    if GROUP_SIZE_M == 1:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n
    else:
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

    return pid_m, pid_n


@gluon.jit
def _gemm_a16w16_uneven_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    bias_ptr,
    M,
    N,
    K: gl.constexpr,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: gl.constexpr,
    BLOCK_N: gl.constexpr,
    BLOCK_K: gl.constexpr,
    GRID_MN: gl.constexpr,
    NUM_XCDS: gl.constexpr,
    GROUP_SIZE_M: gl.constexpr,
    ADD_BIAS: gl.constexpr,
    EVEN_M: gl.constexpr,
    EVEN_N: gl.constexpr,
    EVEN_K: gl.constexpr,
    TOP_M: gl.constexpr = 0,
    LEFT_N: gl.constexpr = 0,
):
    """
    Uneven halves (v9_uneven2): the rows split into a top half of TOP_M rows and a
    bottom half of the rest, the columns into a left half of LEFT_N and a right half of
    the rest (0: BLOCK / 2). A half is a power-of-two main block plus an optional extra
    block of 32 or 64 rows, on the same MFMA layout; the extra is read with its main
    block and copied in the same commit group, so the regions and wait counts below are
    v9's. Halves are tuples of blocks (main[, extra]), and a quadrant's accumulators a
    tuple in quad_mma's order.

    Beyond the hot loop: L2 locality + interleaved epilogue on top of v8_sliceMN.

    Builds on v8's slice-both-M-and-N design (4 quadrant accumulators, unrolled by 2,
    pre-computed _next offsets) and adds:

    1. XCD-aware PID remapping + GROUP_SIZE_M workgroup swizzling for L2 cache locality.
    2. Interleaved epilogue using extract_slice: each 128x128 accumulator is split into
       two 64x128 sub-tiles along M. Stores for sub-tile i are pipelined with the MFMA
       computing sub-tile i+1, spreading write traffic over time.
    """

    pid_m, pid_n = _get_pids(M, N, BLOCK_M, BLOCK_N, GRID_MN, NUM_XCDS, GROUP_SIZE_M)

    # Half sizes: TOP = HALF_M (main) + TX (extra) rows, the bottom BOT_M + BX; the
    # columns likewise. A power-of-two tile has no extras and equal halves.
    TOP: gl.constexpr = TOP_M if TOP_M > 0 else BLOCK_M // 2
    LEFT: gl.constexpr = LEFT_N if LEFT_N > 0 else BLOCK_N // 2
    HALF_M: gl.constexpr = pow2_floor(TOP)
    TX: gl.constexpr = TOP - HALF_M
    BOT_M: gl.constexpr = pow2_floor(BLOCK_M - TOP)
    BX: gl.constexpr = BLOCK_M - TOP - BOT_M
    HALF_N: gl.constexpr = pow2_floor(LEFT)
    LX: gl.constexpr = LEFT - HALF_N
    RIGHT_N: gl.constexpr = pow2_floor(BLOCK_N - LEFT)
    RX: gl.constexpr = BLOCK_N - LEFT - RIGHT_N
    UNEVEN: gl.constexpr = (BLOCK_M != 2 * HALF_M) or (BLOCK_N != 2 * HALF_N)

    # Every layout follows the tile: the shared layouts come from the dot-operand
    # layouts and the global-load layouts from the shared layouts.
    mfmaLayout: gl.constexpr = gl.amd.AMDMFMALayout(
        version=4, instr_shape=[16, 16, 32], transposed=True, warps_per_cta=[2, 2]
    )
    dotOpLayoutA: gl.constexpr = gl.DotOperandLayout(
        operand_index=0, parent=mfmaLayout, k_width=8
    )
    dotOpLayoutB: gl.constexpr = gl.DotOperandLayout(
        operand_index=1, parent=mfmaLayout, k_width=8
    )
    sharedLayoutA: gl.constexpr = ttgl_cdna4.compute_efficient_padded_shared_layout(
        dotOpLayoutA, [HALF_M, BLOCK_K], a_ptr.dtype.element_ty
    )
    sharedLayoutB: gl.constexpr = ttgl_cdna4.compute_efficient_padded_shared_layout(
        dotOpLayoutB, [BLOCK_K, HALF_N], b_ptr.dtype.element_ty
    )
    gLoadLayoutA: gl.constexpr = compute_gload_layout(sharedLayoutA, 8, gl.num_warps())
    gLoadLayoutB: gl.constexpr = compute_gload_layout(sharedLayoutB, 8, gl.num_warps())
    sharedLayoutAb: gl.constexpr = ttgl_cdna4.compute_efficient_padded_shared_layout(
        dotOpLayoutA, [BOT_M, BLOCK_K], a_ptr.dtype.element_ty
    )
    sharedLayoutBr: gl.constexpr = ttgl_cdna4.compute_efficient_padded_shared_layout(
        dotOpLayoutB, [BLOCK_K, RIGHT_N], b_ptr.dtype.element_ty
    )
    gLoadLayoutAb: gl.constexpr = compute_gload_layout(
        sharedLayoutAb, 8, gl.num_warps()
    )
    gLoadLayoutBr: gl.constexpr = compute_gload_layout(
        sharedLayoutBr, 8, gl.num_warps()
    )

    nBuffers: gl.constexpr = 2
    smemA_top = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [nBuffers, HALF_M, BLOCK_K], sharedLayoutA
    )
    smemA_bot = gl.allocate_shared_memory(
        a_ptr.dtype.element_ty, [nBuffers, BOT_M, BLOCK_K], sharedLayoutAb
    )
    smemB_left = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, HALF_N], sharedLayoutB
    )
    smemB_right = gl.allocate_shared_memory(
        b_ptr.dtype.element_ty, [nBuffers, BLOCK_K, RIGHT_N], sharedLayoutBr
    )

    offs_am = gl.arange(0, HALF_M, gl.SliceLayout(1, gLoadLayoutA))
    offs_ak = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutA))
    offs_akb = gl.arange(0, BLOCK_K, gl.SliceLayout(0, gLoadLayoutAb))

    offs_bn = gl.arange(0, HALF_N, gl.SliceLayout(0, gLoadLayoutB))
    offs_bk = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutB))
    offs_bkr = gl.arange(0, BLOCK_K, gl.SliceLayout(1, gLoadLayoutBr))

    # Tile bases in 64-bit: a tile's first element can sit beyond 2^31 elements
    # (16384x201088 outputs, for one). Offsets within a tile stay 32-bit.
    a_base = a_ptr + (pid_m * BLOCK_M).to(gl.int64) * stride_am
    b_base = b_ptr + (pid_n * BLOCK_N).to(gl.int64) * stride_bn

    # Rows and columns of the two halves, relative to the tile. On a ragged last
    # tile they are clamped to the matrix edge: the clamped rows and columns load
    # valid data (duplicates), compute rows nobody keeps, and the masked store
    # drops them. Clamping keeps the buffer offsets inside one tile's range.
    rows_top = offs_am
    rows_bot = gl.arange(0, BOT_M, gl.SliceLayout(1, gLoadLayoutAb)) + TOP
    if not EVEN_M:
        rows_top = gl.minimum(rows_top, M - 1 - pid_m * BLOCK_M)
        rows_bot = gl.minimum(rows_bot, M - 1 - pid_m * BLOCK_M)
    cols_left = offs_bn
    cols_right = gl.arange(0, RIGHT_N, gl.SliceLayout(0, gLoadLayoutBr)) + LEFT
    if not EVEN_N:
        cols_left = gl.minimum(cols_left, N - 1 - pid_n * BLOCK_N)
        cols_right = gl.minimum(cols_right, N - 1 - pid_n * BLOCK_N)

    # Row and column terms of the offsets. The row strides are multiples of 8
    # elements (the wrapper checks), which keeps every 8-element group of a row
    # 16-byte aligned; the hint says so, because Triton's argument specialization
    # only records strides that are multiples of 16.
    a_rows_top = gl.multiple_of(rows_top * stride_am, 8)
    a_rows_bot = gl.multiple_of(rows_bot * stride_am, 8)
    b_cols_left = gl.multiple_of(cols_left * stride_bn, 8)
    b_cols_right = gl.multiple_of(cols_right * stride_bn, 8)

    # Two sets of offsets: base (even K-steps) and _next (odd K-steps).
    a_top_offsets = a_rows_top[:, None] + offs_ak[None, :] * stride_ak
    a_bot_offsets = a_rows_bot[:, None] + offs_akb[None, :] * stride_ak
    b_left_offsets = offs_bk[:, None] * stride_bk + b_cols_left[None, :]
    b_right_offsets = offs_bkr[:, None] * stride_bk + b_cols_right[None, :]

    # The extra blocks: LDS and load offsets, with the K index of each element (the
    # ragged-K masks). They follow the main blocks' clamping.
    lim_m = M - 1 - pid_m * BLOCK_M
    lim_n = N - 1 - pid_n * BLOCK_N
    a_dtype: gl.constexpr = a_ptr.dtype.element_ty
    b_dtype: gl.constexpr = b_ptr.dtype.element_ty
    if TX > 0:
        sharedLayoutAtx: gl.constexpr = _padded(dotOpLayoutA, [TX, BLOCK_K], a_dtype)
        smemA_topx = gl.allocate_shared_memory(
            a_dtype, [nBuffers, TX, BLOCK_K], sharedLayoutAtx
        )
        a_topx_offsets, k_of_atx = _extra_offsets(
            sharedLayoutAtx, HALF_M, stride_am, stride_ak, lim_m, EVEN_M, False
        )
    if BX > 0:
        sharedLayoutAbx: gl.constexpr = _padded(dotOpLayoutA, [BX, BLOCK_K], a_dtype)
        smemA_botx = gl.allocate_shared_memory(
            a_dtype, [nBuffers, BX, BLOCK_K], sharedLayoutAbx
        )
        a_botx_offsets, k_of_abx = _extra_offsets(
            sharedLayoutAbx, TOP + BOT_M, stride_am, stride_ak, lim_m, EVEN_M, False
        )
    if LX > 0:
        sharedLayoutBlx: gl.constexpr = _padded(dotOpLayoutB, [BLOCK_K, LX], b_dtype)
        smemB_leftx = gl.allocate_shared_memory(
            b_dtype, [nBuffers, BLOCK_K, LX], sharedLayoutBlx
        )
        b_leftx_offsets, k_of_blx = _extra_offsets(
            sharedLayoutBlx, HALF_N, stride_bn, stride_bk, lim_n, EVEN_N, True
        )
    if RX > 0:
        sharedLayoutBrx: gl.constexpr = _padded(dotOpLayoutB, [BLOCK_K, RX], b_dtype)
        smemB_rightx = gl.allocate_shared_memory(
            b_dtype, [nBuffers, BLOCK_K, RX], sharedLayoutBrx
        )
        b_rightx_offsets, k_of_brx = _extra_offsets(
            sharedLayoutBrx, LEFT + RIGHT_N, stride_bn, stride_bk, lim_n, EVEN_N, True
        )

    # Each half as a tuple of blocks, main then extra (absent extras left out).
    k_of_a_top = (gl.zeros_like(a_top_offsets) + offs_ak[None, :],) + (
        (k_of_atx,) if TX > 0 else ()
    )
    k_of_a_bot = (gl.zeros_like(a_bot_offsets) + offs_akb[None, :],) + (
        (k_of_abx,) if BX > 0 else ()
    )
    k_of_b_left = (gl.zeros_like(b_left_offsets) + offs_bk[:, None],) + (
        (k_of_blx,) if LX > 0 else ()
    )
    k_of_b_right = (gl.zeros_like(b_right_offsets) + offs_bkr[:, None],) + (
        (k_of_brx,) if RX > 0 else ()
    )
    smemA_top = (smemA_top,) + ((smemA_topx,) if TX > 0 else ())
    smemA_bot = (smemA_bot,) + ((smemA_botx,) if BX > 0 else ())
    smemB_left = (smemB_left,) + ((smemB_leftx,) if LX > 0 else ())
    smemB_right = (smemB_right,) + ((smemB_rightx,) if RX > 0 else ())
    a_top_offsets = (a_top_offsets,) + ((a_topx_offsets,) if TX > 0 else ())
    a_bot_offsets = (a_bot_offsets,) + ((a_botx_offsets,) if BX > 0 else ())
    b_left_offsets = (b_left_offsets,) + ((b_leftx_offsets,) if LX > 0 else ())
    b_right_offsets = (b_right_offsets,) + ((b_rightx_offsets,) if RX > 0 else ())

    a_top_offsets_next = _shift(a_top_offsets, BLOCK_K * stride_ak)
    a_bot_offsets_next = _shift(a_bot_offsets, BLOCK_K * stride_ak)
    b_left_offsets_next = _shift(b_left_offsets, BLOCK_K * stride_bk)
    b_right_offsets_next = _shift(b_right_offsets, BLOCK_K * stride_bk)

    # Rows (columns) of each half's blocks.
    ROWS_T: gl.constexpr = (HALF_M,) + ((TX,) if TX > 0 else ())
    ROWS_B: gl.constexpr = (BOT_M,) + ((BX,) if BX > 0 else ())
    COLS_L: gl.constexpr = (HALF_N,) + ((LX,) if LX > 0 else ())
    COLS_R: gl.constexpr = (RIGHT_N,) + ((RX,) if RX > 0 else ())
    acc_tl = _quad_zeros(ROWS_T, COLS_L, mfmaLayout)
    acc_bl = _quad_zeros(ROWS_B, COLS_L, mfmaLayout)
    acc_tr = _quad_zeros(ROWS_T, COLS_R, mfmaLayout)
    acc_br = _quad_zeros(ROWS_B, COLS_R, mfmaLayout)

    # A ragged K: the tail (K % (2 * BLOCK_K) elements) is consumed first through
    # buffer 0, unpipelined, with the loads masked at 8-element granularity. The
    # pipelined loop below then covers K_MAIN exactly as for an even K; only the
    # accumulation order differs.
    if EVEN_K:
        K_MAIN: gl.constexpr = K
    else:
        K_MAIN: gl.constexpr = K - K % (2 * BLOCK_K)
        K_TAIL: gl.constexpr = K - K_MAIN
        a_tail = a_base + K_MAIN * stride_ak
        b_tail = b_base + K_MAIN * stride_bk
        for t in gl.static_range(gl.cdiv(K_TAIL, BLOCK_K)):
            copy_half(
                smemB_left,
                0,
                b_tail,
                _shift(b_left_offsets, t * BLOCK_K * stride_bk),
                _k_masks(k_of_b_left, t * BLOCK_K, K_TAIL),
            )
            gl.amd.cdna4.async_copy.commit_group()
            copy_half(
                smemA_top,
                0,
                a_tail,
                _shift(a_top_offsets, t * BLOCK_K * stride_ak),
                _k_masks(k_of_a_top, t * BLOCK_K, K_TAIL),
            )
            gl.amd.cdna4.async_copy.commit_group()
            copy_half(
                smemA_bot,
                0,
                a_tail,
                _shift(a_bot_offsets, t * BLOCK_K * stride_ak),
                _k_masks(k_of_a_bot, t * BLOCK_K, K_TAIL),
            )
            gl.amd.cdna4.async_copy.commit_group()
            copy_half(
                smemB_right,
                0,
                b_tail,
                _shift(b_right_offsets, t * BLOCK_K * stride_bk),
                _k_masks(k_of_b_right, t * BLOCK_K, K_TAIL),
            )
            gl.amd.cdna4.async_copy.commit_group()
            gl.amd.cdna4.async_copy.wait_group(0)
            b_left = read_half(smemB_left, 0, dotOpLayoutB)
            a_top = read_half(smemA_top, 0, dotOpLayoutA)
            a_bot = read_half(smemA_bot, 0, dotOpLayoutA)
            b_right = read_half(smemB_right, 0, dotOpLayoutB)
            acc_tl = quad_mma(a_top, b_left, acc_tl)
            acc_bl = quad_mma(a_bot, b_left, acc_bl)
            acc_tr = quad_mma(a_top, b_right, acc_tr)
            acc_br = quad_mma(a_bot, b_right, acc_br)

    iterMax = gl.cdiv(K_MAIN, BLOCK_K)

    ## Prologue — same as v8
    g_idx = 0
    copy_half(smemB_left, g_idx, b_base, b_left_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    copy_half(smemA_top, g_idx, a_base, a_top_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    copy_half(smemA_bot, g_idx, a_base, a_bot_offsets)
    gl.amd.cdna4.async_copy.commit_group()
    copy_half(smemB_right, g_idx, b_base, b_right_offsets)
    gl.amd.cdna4.async_copy.commit_group()

    g_idx = 1
    copy_half(smemB_left, g_idx, b_base, b_left_offsets_next)
    gl.amd.cdna4.async_copy.commit_group()
    copy_half(smemA_top, g_idx, a_base, a_top_offsets_next)
    gl.amd.cdna4.async_copy.commit_group()
    copy_half(smemA_bot, g_idx, a_base, a_bot_offsets_next)
    gl.amd.cdna4.async_copy.commit_group()
    copy_half(smemB_right, g_idx, b_base, b_right_offsets_next)
    gl.amd.cdna4.async_copy.commit_group()

    a_base += BLOCK_K * stride_ak * 2
    b_base += BLOCK_K * stride_bk * 2

    gl.amd.cdna4.async_copy.wait_group(6)
    b_left = read_half(smemB_left, 0, dotOpLayoutB)
    a_top = read_half(smemA_top, 0, dotOpLayoutA)

    gl.assume(iterMax > 3)

    ## Main loop — same as v8
    for k in range(0, iterMax - 2, 2):

        ## =============================================================
        ## Sub-iteration 0: consume buffer 0, prefetch into buffer 0
        ## =============================================================

        ########################################
        ## Region 0: C_tl = DOT(a_top, b_left)
        ########################################
        acc_tl = quad_mma(a_top, b_left, acc_tl)

        gl.amd.cdna4.async_copy.wait_group(5)
        a_bot = read_half(smemA_bot, 0, dotOpLayoutA)

        copy_half(smemB_left, 0, b_base, b_left_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 1: C_bl = DOT(a_bot, b_left)
        ########################################
        acc_bl = quad_mma(a_bot, b_left, acc_bl)

        gl.amd.cdna4.async_copy.wait_group(5)
        b_right = read_half(smemB_right, 0, dotOpLayoutB)

        copy_half(smemA_top, 0, a_base, a_top_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 2: C_tr = DOT(a_top, b_right)
        ########################################
        acc_tr = quad_mma(a_top, b_right, acc_tr)

        gl.amd.cdna4.async_copy.wait_group(5)
        b_left = read_half(smemB_left, 1, dotOpLayoutB)

        copy_half(smemA_bot, 0, a_base, a_bot_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 3: C_br = DOT(a_bot, b_right)
        ########################################
        acc_br = quad_mma(a_bot, b_right, acc_br)

        gl.amd.cdna4.async_copy.wait_group(5)
        a_top = read_half(smemA_top, 1, dotOpLayoutA)

        copy_half(smemB_right, 0, b_base, b_right_offsets)
        gl.amd.cdna4.async_copy.commit_group()

        ## =============================================================
        ## Loop unroll: Sub-iteration 1: consume buffer 1, prefetch
        ## into buffer 1. AC uses _next offsets (odd K-step).
        ## =============================================================

        ########################################
        ## Region 0: C_tl = DOT(a_top, b_left)
        ########################################
        acc_tl = quad_mma(a_top, b_left, acc_tl)

        gl.amd.cdna4.async_copy.wait_group(5)
        a_bot = read_half(smemA_bot, 1, dotOpLayoutA)

        copy_half(smemB_left, 1, b_base, b_left_offsets_next)
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 1: C_bl = DOT(a_bot, b_left)
        ########################################
        acc_bl = quad_mma(a_bot, b_left, acc_bl)

        gl.amd.cdna4.async_copy.wait_group(5)
        b_right = read_half(smemB_right, 1, dotOpLayoutB)

        copy_half(smemA_top, 1, a_base, a_top_offsets_next)
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 2: C_tr = DOT(a_top, b_right)
        ########################################
        acc_tr = quad_mma(a_top, b_right, acc_tr)

        gl.amd.cdna4.async_copy.wait_group(5)
        b_left = read_half(smemB_left, 0, dotOpLayoutB)

        copy_half(smemA_bot, 1, a_base, a_bot_offsets_next)
        gl.amd.cdna4.async_copy.commit_group()

        ########################################
        ## Region 3: C_br = DOT(a_bot, b_right)
        ########################################
        acc_br = quad_mma(a_bot, b_right, acc_br)

        gl.amd.cdna4.async_copy.wait_group(5)
        a_top = read_half(smemA_top, 0, dotOpLayoutA)

        copy_half(smemB_right, 1, b_base, b_right_offsets_next)
        gl.amd.cdna4.async_copy.commit_group()

        a_base += BLOCK_K * stride_ak * 2
        b_base += BLOCK_K * stride_bk * 2

    ## Epilogue: 4-quadrant stores with natural-pipeline ordering (matches v8).
    ## v9's contribution lives in the prologue / pid remapping (§3); the epilogue
    ## is unchanged from v8 because the sub-tile variant produced only ~200 cycles
    ## of additional savings — within noise relative to the full kernel.

    mask_tl, mask_tr, mask_bl, mask_br = None, None, None, None
    bias_l, bias_r = None, None
    if not UNEVEN:
        # Store layout sized to the half tile: 8 contiguous elements per thread keep the
        # 16-byte store, and one warp spans exactly BLOCK_N // 2 columns ([4, 16] lanes at
        # BLOCK_N = 256, the tutorial's constant).
        STORE_CONTIG: gl.constexpr = 8
        STORE_LANES_N: gl.constexpr = (BLOCK_N // 2) // STORE_CONTIG
        STORE_LANES_M: gl.constexpr = 64 // STORE_LANES_N
        gStoreLayoutC: gl.constexpr = gl.BlockedLayout(
            [1, STORE_CONTIG], [STORE_LANES_M, STORE_LANES_N], [4, 1], [1, 0]
        )

        offs_cm = gl.arange(0, BLOCK_M // 2, gl.SliceLayout(1, gStoreLayoutC))
        offs_cn = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, gStoreLayoutC))
        c_base = (
            c_ptr
            + (pid_m * BLOCK_M).to(gl.int64) * stride_cm
            + (pid_n * BLOCK_N).to(gl.int64) * stride_cn
        )
        # same alignment hint as the loads
        c_rows = gl.multiple_of(offs_cm * stride_cm, 8)
        c_tl_offsets = c_rows[:, None] + stride_cn * offs_cn[None, :]
        c_tr_offsets = c_tl_offsets + BLOCK_N * stride_cn // 2
        c_bl_offsets = c_tl_offsets + BLOCK_M * stride_cm // 2
        c_br_offsets = c_bl_offsets + BLOCK_N * stride_cn // 2
        if not EVEN_N:
            # Columns of C left from this tile's origin: a multiple of 8 (N and
            # BLOCK_N are), which the hint states so that each thread's 8-column
            # group keeps one predicate and one 16-byte store. The hint cannot go on
            # N itself: Triton drops hints on kernel arguments.
            n_left = gl.multiple_of(N - pid_n * BLOCK_N, 8)
        if not (EVEN_M and EVEN_N):
            # The ragged last tile stores only the rows and columns inside C.
            m_left = M - pid_m * BLOCK_M
            in_top = offs_cm[:, None] < m_left
            in_bot = offs_cm[:, None] + BLOCK_M // 2 < m_left
            if EVEN_N:
                in_left = (
                    offs_cn[None, :] < BLOCK_N
                )  # no column limit; gives the mask its shape
                in_right = in_left
            else:
                in_left = offs_cn[None, :] < n_left
                in_right = offs_cn[None, :] + BLOCK_N // 2 < n_left
            mask_tl = in_top & in_left
            mask_tr = in_top & in_right
            mask_bl = in_bot & in_left
            mask_br = in_bot & in_right

        if ADD_BIAS:
            # bias[N] is added in fp32 before the downcast; loaded here so its latency
            # overlaps the last MFMAs.
            offs_bias = gl.arange(0, BLOCK_N // 2, gl.SliceLayout(0, mfmaLayout))
            if EVEN_N:
                bias_l = gl.load(bias_ptr + pid_n * BLOCK_N + offs_bias).to(gl.float32)
                bias_r = gl.load(
                    bias_ptr + pid_n * BLOCK_N + BLOCK_N // 2 + offs_bias
                ).to(gl.float32)
            else:
                bias_l = gl.load(
                    bias_ptr + pid_n * BLOCK_N + offs_bias,
                    mask=offs_bias < n_left,
                    other=0,
                ).to(gl.float32)
                bias_r = gl.load(
                    bias_ptr + pid_n * BLOCK_N + BLOCK_N // 2 + offs_bias,
                    mask=offs_bias + BLOCK_N // 2 < n_left,
                    other=0,
                ).to(gl.float32)
        c_tl_offsets, mask_tl, bias_l = (c_tl_offsets,), (mask_tl,), (bias_l,)
        c_bl_offsets, mask_bl = (c_bl_offsets,), (mask_bl,)
        c_tr_offsets, mask_tr, bias_r = (c_tr_offsets,), (mask_tr,), (bias_r,)
        c_br_offsets, mask_br = (c_br_offsets,), (mask_br,)
    else:
        # Uneven halves: offsets and masks per block, bias per column block.
        c_base = (
            c_ptr
            + (pid_m * BLOCK_M).to(gl.int64) * stride_cm
            + (pid_n * BLOCK_N).to(gl.int64) * stride_cn
        )
        m_left = M - pid_m * BLOCK_M
        n_left = N - pid_n * BLOCK_N
        R0_T: gl.constexpr = (0,) + ((HALF_M,) if TX > 0 else ())
        R0_B: gl.constexpr = (TOP,) + ((TOP + BOT_M,) if BX > 0 else ())
        C0_L: gl.constexpr = (0,) + ((HALF_N,) if LX > 0 else ())
        C0_R: gl.constexpr = (LEFT,) + ((LEFT + RIGHT_N,) if RX > 0 else ())
        c_tl_offsets, mask_tl = _c_blocks(
            R0_T, ROWS_T, C0_L, COLS_L, stride_cm, stride_cn, m_left, n_left
        )
        c_bl_offsets, mask_bl = _c_blocks(
            R0_B, ROWS_B, C0_L, COLS_L, stride_cm, stride_cn, m_left, n_left
        )
        c_tr_offsets, mask_tr = _c_blocks(
            R0_T, ROWS_T, C0_R, COLS_R, stride_cm, stride_cn, m_left, n_left
        )
        c_br_offsets, mask_br = _c_blocks(
            R0_B, ROWS_B, C0_R, COLS_R, stride_cm, stride_cn, m_left, n_left
        )
        if ADD_BIAS:
            bias_l = _bias_cols(
                bias_ptr + pid_n * BLOCK_N, C0_L, COLS_L, n_left, EVEN_N, mfmaLayout
            )
            bias_r = _bias_cols(
                bias_ptr + pid_n * BLOCK_N, C0_R, COLS_R, n_left, EVEN_N, mfmaLayout
            )
    MASKED: gl.constexpr = not (EVEN_M and EVEN_N)

    ## Iter iterMax - 2: same 4-region pattern as main loop, no AC
    acc_tl = quad_mma(a_top, b_left, acc_tl)
    gl.amd.cdna4.async_copy.wait_group(5)
    l_idx = (iterMax - 2) % 2
    a_bot = read_half(smemA_bot, l_idx, dotOpLayoutA)

    acc_bl = quad_mma(a_bot, b_left, acc_bl)
    gl.amd.cdna4.async_copy.wait_group(4)
    b_right = read_half(smemB_right, l_idx, dotOpLayoutB)

    acc_tr = quad_mma(a_top, b_right, acc_tr)
    gl.amd.cdna4.async_copy.wait_group(3)
    g_idx = 1 - l_idx
    b_left = read_half(smemB_left, g_idx, dotOpLayoutB)

    acc_br = quad_mma(a_bot, b_right, acc_br)
    gl.amd.cdna4.async_copy.wait_group(2)
    a_top = read_half(smemA_top, g_idx, dotOpLayoutA)

    ## Iter iterMax - 1
    ## Natural-pipeline epilogue: each store follows its MFMA with one
    ## MFMA cycle of gap, yielding uniform MFMA-store interleaving.
    acc_tl = quad_mma(a_top, b_left, acc_tl)
    gl.amd.cdna4.async_copy.wait_group(1)
    a_bot = read_half(smemA_bot, g_idx, dotOpLayoutA)

    acc_bl = quad_mma(a_bot, b_left, acc_bl)
    gl.amd.cdna4.async_copy.wait_group(0)
    b_right = read_half(smemB_right, g_idx, dotOpLayoutB)

    _store_quad(acc_tl, bias_l, c_base, c_tl_offsets, mask_tl, ADD_BIAS, MASKED)

    acc_tr = quad_mma(a_top, b_right, acc_tr)

    _store_quad(acc_bl, bias_l, c_base, c_bl_offsets, mask_bl, ADD_BIAS, MASKED)

    acc_br = quad_mma(a_bot, b_right, acc_br)

    _store_quad(acc_tr, bias_r, c_base, c_tr_offsets, mask_tr, ADD_BIAS, MASKED)

    _store_quad(acc_br, bias_r, c_base, c_br_offsets, mask_br, ADD_BIAS, MASKED)


@constexpr_function
def _padded(dot_operand_layout, shape, dtype):
    return ttgl_cdna4.compute_efficient_padded_shared_layout(
        dot_operand_layout, shape, dtype
    )


@constexpr_function
def pow2_floor(n):
    return 1 << (int(n).bit_length() - 1)


@gluon.jit
def quad_mma(a, b, acc):
    """A quadrant's MFMAs. a, b: its A and B halves' blocks (main[, extra]); block
    (i, j) accumulates in acc[i + len(a) * j]."""
    out = ()
    for j in gl.static_range(len(b)):
        for i in gl.static_range(len(a)):
            out = out + (
                gl.amd.cdna3.mfma(a[i], b[j], acc[i + len(a) * j], cd_regclass="a"),
            )
    return out


@gluon.jit
def read_half(smem, idx, layout: gl.constexpr):
    """A half's blocks from LDS buffer idx, main then extra."""
    out = ()
    for j in gl.static_range(len(smem)):
        out = out + (smem[j].index(idx).load(layout),)
    return out


@gluon.jit
def copy_half(smem, idx, base, offs, masks=None):
    """A half's async copies into LDS buffer idx; masks: one per block, or None."""
    for j in gl.static_range(len(smem)):
        if masks is None:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                smem[j].index(idx), base, offs[j]
            )
        else:
            gl.amd.cdna4.async_copy.buffer_load_to_shared(
                smem[j].index(idx), base, offs[j], mask=masks[j]
            )


@gluon.jit
def _shift(offs, d):
    out = ()
    for j in gl.static_range(len(offs)):
        out = out + (offs[j] + d,)
    return out


@gluon.jit
def _k_masks(k_of, k0, K_TAIL: gl.constexpr):
    """Each block's ragged-K load mask at K-step offset k0."""
    out = ()
    for j in gl.static_range(len(k_of)):
        out = out + (k_of[j] + k0 < K_TAIL,)
    return out


@gluon.jit
def _quad_zeros(ROWS: gl.constexpr, COLS: gl.constexpr, layout: gl.constexpr):
    """A quadrant's fp32 accumulator blocks, in quad_mma's order."""
    out = ()
    for j in gl.static_range(len(COLS)):
        for i in gl.static_range(len(ROWS)):
            out = out + (gl.zeros((ROWS[i], COLS[j]), gl.float32, layout),)
    return out


@gluon.jit
def _extra_offsets(
    SHARED: gl.constexpr,
    ROW0: gl.constexpr,
    stride_r,
    stride_k,
    lim,
    EVEN: gl.constexpr,
    K_FIRST: gl.constexpr,
):
    """Load offsets of an extra block, rows ROW0 on of A's [rows, BLOCK_K] or, K_FIRST,
    columns of B's [BLOCK_K, cols], built as the main blocks' are; and each element's
    K index."""
    layout: gl.constexpr = compute_gload_layout(SHARED, 8, gl.num_warps())
    if K_FIRST:
        r = gl.arange(0, SHARED.shape[1], gl.SliceLayout(0, layout)) + ROW0
        k = gl.arange(0, SHARED.shape[0], gl.SliceLayout(1, layout))
        if not EVEN:
            r = gl.minimum(r, lim)
        offs = k[:, None] * stride_k + gl.multiple_of(r * stride_r, 8)[None, :]
        k_of = gl.zeros_like(offs) + k[:, None]
    else:
        r = gl.arange(0, SHARED.shape[0], gl.SliceLayout(1, layout)) + ROW0
        k = gl.arange(0, SHARED.shape[1], gl.SliceLayout(0, layout))
        if not EVEN:
            r = gl.minimum(r, lim)
        offs = gl.multiple_of(r * stride_r, 8)[:, None] + k[None, :] * stride_k
        k_of = gl.zeros_like(offs) + k[None, :]
    return offs, k_of


@constexpr_function
def _block_store_layout(rows, cols):
    """C-store layout of a [rows, cols] block: up to 8 values per lane along N, the
    4 warps along M, no value stored twice."""
    rows, cols = int(rows), int(cols)
    spt = max(1, min(8, rows * cols // 256))
    tpw_n = min(64, cols // spt)
    return gl.BlockedLayout([1, spt], [64 // tpw_n, tpw_n], [4, 1], [1, 0])


@gluon.jit
def _c_blocks(
    ROW0S: gl.constexpr,
    ROWS: gl.constexpr,
    COL0S: gl.constexpr,
    COLS: gl.constexpr,
    stride_cm,
    stride_cn,
    m_left,
    n_left,
):
    """C offsets and in-C masks of a quadrant's blocks, in quad_mma's order."""
    offs = ()
    masks = ()
    for j in gl.static_range(len(COLS)):
        for i in gl.static_range(len(ROWS)):
            o, m = _c_block(
                ROW0S[i],
                ROWS[i],
                COL0S[j],
                COLS[j],
                stride_cm,
                stride_cn,
                m_left,
                n_left,
            )
            offs = offs + (o,)
            masks = masks + (m,)
    return offs, masks


@gluon.jit
def _c_block(
    ROW0: gl.constexpr,
    ROWS: gl.constexpr,
    COL0: gl.constexpr,
    COLS: gl.constexpr,
    stride_cm,
    stride_cn,
    m_left,
    n_left,
):
    layout: gl.constexpr = _block_store_layout(ROWS, COLS)
    r = gl.arange(0, ROWS, gl.SliceLayout(1, layout)) + ROW0
    c = gl.arange(0, COLS, gl.SliceLayout(0, layout)) + COL0
    offs = gl.multiple_of(r * stride_cm, 8)[:, None] + stride_cn * c[None, :]
    return offs, (r[:, None] < m_left) & (c[None, :] < n_left)


@gluon.jit
def _bias_cols(
    bias_base,
    COL0S: gl.constexpr,
    COLS: gl.constexpr,
    n_left,
    EVEN_N: gl.constexpr,
    layout: gl.constexpr,
):
    """bias[N] of a column half's blocks, in fp32."""
    out = ()
    for j in gl.static_range(len(COLS)):
        offs = gl.arange(0, COLS[j], gl.SliceLayout(0, layout)) + COL0S[j]
        if EVEN_N:
            b = gl.load(bias_base + offs)
        else:
            b = gl.load(bias_base + offs, mask=offs < n_left, other=0)
        out = out + (b.to(gl.float32),)
    return out


@gluon.jit
def _store_quad(
    acc, bias, c_base, offsets, masks, ADD_BIAS: gl.constexpr, MASKED: gl.constexpr
):
    """C stores of a quadrant's blocks; bias holds one vector per column block."""
    for b in gl.static_range(len(acc)):
        a = acc[b]
        if ADD_BIAS:
            a = a + bias[b // (len(acc) // len(bias))][None, :]
        c = a.to(c_base.dtype.element_ty)
        c = gl.convert_layout(c, layout=offsets[b].type.layout)
        if MASKED:
            gl.amd.cdna3.buffer_store(
                ptr=c_base, offsets=offsets[b], stored_value=c, mask=masks[b]
            )
        else:
            gl.amd.cdna3.buffer_store(ptr=c_base, offsets=offsets[b], stored_value=c)


MFMA_SCHEDULE_HINT = "mfma-schedule"


def schedule_hint():
    """The schedule_hint the kernel launches with: the MFMA scheduler, unless
    AITER_MFMA_SCHED=0 (an unscheduled A/B run)."""
    return "" if os.environ.get("AITER_MFMA_SCHED", "1") == "0" else MFMA_SCHEDULE_HINT


@functools.cache
def _cu_count(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


def default_tile(M, N, K, device=None):
    """The tile the gluon GEMM config picks for a shape: a tuned config names it, the
    default config's cost model picks the tile whose workgroup waves finish
    first (see ``pick_gemm_tile``)."""
    config, _ = get_gemm_config("GEMM-A16W16", M, N, K, backend="gluon")
    device = torch.cuda.current_device() if device is None else device
    return pick_gemm_tile(config, M, N, _cu_count(device))


def unsupported_reason(M, N, K, x=None, w=None, bias=None, activation=None, tile=None):
    """Why this kernel cannot run the problem, or None if it can. ``tile`` is the
    (BLOCK_M, BLOCK_N) to check; None checks the config's pick for the shape."""
    if activation:
        return "no activation support yet"
    if tile is None:
        tile = default_tile(M, N, K)
    bm, bn = tile
    if bm not in TILE_SIDES or bn not in TILE_SIDES:
        return f"tile {bm}x{bn} has a side outside {TILE_SIDES}"
    if bm * bn > MAX_ACC:
        return f"tile {bm}x{bn} has more than {MAX_ACC} accumulators"
    if M < 1 or N % 8:
        return f"N must be a multiple of 8 (16-byte stores; got M={M}, N={N})"
    if K % 8 or K < 4 * BLOCK_K:
        return f"K must be a multiple of 8 and at least {4 * BLOCK_K} (got K={K})"
    if x is not None and (x.stride(1) != 1 or x.stride(0) % 8):
        return "x must be row-major (M, K) with a row stride that is a multiple of 8"
    if w is not None and (w.stride(1) != 1 or w.stride(0) % 8):
        return "w must be row-major (N, K) with a row stride that is a multiple of 8"
    if x is not None and x.dtype not in (torch.float16, torch.bfloat16):
        return f"dtype {x.dtype} is not fp16/bf16"
    return None


def _valid_half(rows):
    """A half of `rows` rows: a power of two, plus nothing or a power of two of at least 32."""
    extra = rows - (1 << (rows.bit_length() - 1))
    return extra == 0 or (extra >= 32 and extra & (extra - 1) == 0)


def uneven_first(side):
    """The first half (top or left) of a tile side: 0 (BLOCK / 2) for a power of two,
    else the side's 32-wide pieces split as evenly as two valid halves allow, the
    larger half first (224 = 128 + 96, 448 = 256 + 192)."""
    if side & (side - 1) == 0:
        return 0
    pieces = side // 32
    for big in range((pieces + 1) // 2, pieces):
        if _valid_half(32 * big) and _valid_half(32 * (pieces - big)):
            return 32 * big
    raise ValueError(
        f"side {side}: no split into two halves of a power of two + 32 or 64"
    )


def gemm_a16w16_uneven(x, w, y, bias=None, tile=None):
    """y = x @ w.T (+ bias) with x (M, K) and w (N, K), both row-major; y (M, N).
    ``tile`` is (BLOCK_M, BLOCK_N); None takes the config's pick for the shape."""
    M, K = x.shape
    N, _ = w.shape
    if tile is None:
        tile = default_tile(M, N, K, x.device)
    reason = unsupported_reason(M, N, K, x, w, tile=tile)
    if reason is None and (y.stride(1) != 1 or y.stride(0) % 8):
        reason = "y must be row-major (M, N) with a row stride that is a multiple of 8"
    if reason is not None:
        raise ValueError(f"gfx950 gluon uneven a16w16: {reason}")
    bm, bn = tile
    b = w.T  # (K, N) view, K contiguous: the layout the kernel loads
    grid_mn = triton.cdiv(M, bm) * triton.cdiv(N, bn)
    _gemm_a16w16_uneven_kernel[(grid_mn, 1)](
        x,
        b,
        y,
        bias,
        M,
        N,
        K,
        x.stride(0),
        x.stride(1),
        b.stride(0),
        b.stride(1),
        y.stride(0),
        y.stride(1),
        BLOCK_M=bm,
        BLOCK_N=bn,
        BLOCK_K=BLOCK_K,
        GRID_MN=grid_mn,
        NUM_XCDS=NUM_XCDS,
        GROUP_SIZE_M=GROUP_SIZE_M,
        ADD_BIAS=bias is not None,
        EVEN_M=M % bm == 0,
        EVEN_N=N % bn == 0,
        EVEN_K=K % (2 * BLOCK_K) == 0,
        TOP_M=uneven_first(bm),
        LEFT_N=uneven_first(bn),
        num_warps=NUM_WARPS,
        schedule_hint=schedule_hint(),
    )
    return y


_KERNEL_MAP = {"uneven": gemm_a16w16_uneven}
