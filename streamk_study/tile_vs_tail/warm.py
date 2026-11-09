"""Compile every candidate of a run into TRITON_CACHE_DIR before timing, in parallel.

Each worker calls the same launchers the tuner reaches (the Stream-K wrapper and the
compute-bound kernel, which runs the uneven kernel for non-power-of-two tiles) with
uninitialized tensors of the shape's sizes, and with JITFunction.run forced to warmup:
the kernels compile but never launch. Shapes with bias compile both variants.

    source env.sh; python warm.py runs/<name> [--jobs 64] [--gpus 0,1,2,3]
"""

import argparse
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd


def _init(gpus, counter):
    with counter.get_lock():
        i = counter.value
        counter.value += 1
    os.environ["HIP_VISIBLE_DEVICES"] = gpus[i % len(gpus)]


def _fake(shape, device):
    """An uninitialized tensor: the kernel never reads it, only its specialization
    (sizes, strides, alignment) matters."""
    import torch

    return torch.empty(shape, dtype=torch.bfloat16, device=device)


def _compile_one(M, N, K, bias, name):
    import torch
    from triton.runtime import jit

    if not getattr(jit.JITFunction, "_tvt_patched", False):
        run = jit.JITFunction.run

        def warm_run(self, *args, grid, warmup, **kwargs):
            return run(self, *args, grid=grid, warmup=True, **kwargs)

        jit.JITFunction.run = warm_run
        jit.JITFunction._tvt_patched = True

    from aiter.tuned_gemm import parse_gluon_kernel_name
    from aiter.ops.triton.gemm.basic.gemm_a16w16 import _gemm_a16w16_gfx950_streamk
    from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16 import (
        gemm_a16w16_compute_bound,
    )

    dev = torch.device("cuda", 0)
    x = _fake((M, K), dev)
    w = _fake((N, K), dev)
    y = _fake((M, N), dev)
    b = _fake((N,), dev) if bias else None
    kind, config = parse_gluon_kernel_name(name)
    if kind == "streamk":
        _gemm_a16w16_gfx950_streamk(x, w, y, b, config)
    else:
        gemm_a16w16_compute_bound(x, w, y, b, tile=(config["BLOCK_M"], config["BLOCK_N"]))
    del x, w, y, b
    torch.cuda.empty_cache()


def compile_one(task):
    t0 = time.time()
    try:
        _compile_one(*task)
        return task, None, time.time() - t0
    except Exception as error:  # report and continue: the tuner will mark it failed
        return task, f"{type(error).__name__}: {error}\n{traceback.format_exc(limit=3)}", time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--jobs", type=int, default=64)
    ap.add_argument("--gpus", default=os.environ.get("HIP_VISIBLE_DEVICES", "0"))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    run = Path(args.run_dir)
    cands = pd.read_csv(run / "candidates.csv")
    shapes = pd.read_csv(run / "shapes.csv")
    biases = shapes.groupby(["M", "N", "K"]).bias.apply(lambda b: sorted(set(map(bool, b))))
    tasks = []
    for r in cands.itertuples():
        for bias in biases.get((r.M, r.N, r.K), [False]):
            tasks.append((int(r.M), int(r.N), int(r.K), bias, r.kernelName))
    if args.limit:
        tasks = tasks[: args.limit]
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    counter = ctx.Value("i", 0)
    gpus = args.gpus.split(",")
    t0 = time.time()
    failed = []
    with ProcessPoolExecutor(args.jobs, mp_context=ctx, initializer=_init, initargs=(gpus, counter)) as pool:
        futures = [pool.submit(compile_one, t) for t in tasks]
        for i, f in enumerate(as_completed(futures), 1):
            task, err, dt = f.result()
            if err:
                failed.append((task, err))
            if i % 200 == 0 or i == len(tasks):
                print(f"{i}/{len(tasks)} compiled, {len(failed)} failed, {time.time() - t0:.0f} s", flush=True)
    with open(run / "warm_failed.txt", "w") as f:
        for task, err in failed:
            f.write(f"{task}\n{err}\n")
    print(f"done: {len(tasks)} in {time.time() - t0:.0f} s, {len(failed)} failed (warm_failed.txt)")


if __name__ == "__main__":
    sys.exit(main())
