"""Measured HBM peak of one GPU: the best of a device-to-device copy and a read-only
reduction over buffers much larger than the caches (MALL included), median of many runs.

    source env.sh; GPU=3 python bw_peak.py > runs/bw_peak.txt
"""

import torch


def time_us(fn, iters=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        times.append(s.elapsed_time(e) * 1e3)
    times.sort()
    return times[len(times) // 2]


def main():
    results = {}
    for gib in (1, 2, 4):
        n = gib * 2**30 // 2
        a = torch.empty(n, dtype=torch.bfloat16, device="cuda").uniform_()
        b = torch.empty_like(a)
        us = time_us(lambda: b.copy_(a))
        results[f"copy_{gib}GiB"] = 2 * a.numel() * 2 / us / 1e6  # TB/s, read + write
        us = time_us(lambda: a.sum(dtype=torch.float32))
        results[f"read_{gib}GiB"] = a.numel() * 2 / us / 1e6
        del a, b
    for k, v in results.items():
        print(f"{k} {v:.3f} TB/s")
    print(f"peak_TBps {max(results.values()):.3f}")


if __name__ == "__main__":
    main()
