"""Parse mfma_profile.sh's rocprofv3 output (RUN/prof_tmp) into small CSVs.

counters -> RUN/mfma_counters.csv, one row per (shape, kernel), the mean over the timed
dispatches (the first, untimed call is dropped):
    mfma_util_busy  SQ_VALU_MFMA_BUSY_CYCLES / (SQ_BUSY_CU_CYCLES x 4 SIMDs): MFMA issue rate
                    while a CU is busy (16 busy cycles per bf16 16x16x32 MFMA is the peak rate)
    mfma_util_gpu   SQ_VALU_MFMA_BUSY_CYCLES / (GRBM_GUI_ACTIVE x 128 SIMDs per XCD): over the
                    whole GPU, so idle CUs in a partial wave count against it
    useful_share    useful MACs (M N K) / MACs issued (SQ_INSTS_MFMA x 16 x 16 x 32): padding
    lds_wait_per_mfma  SQ_WAIT_INST_LDS / SQ_INSTS_MFMA

att -> RUN/mfma_att.csv, one row per kernel: the main loop is the address range spanning the
MFMAs with the highest hit count; within it, MFMA issue efficiency = 16 cycles x MFMA hits /
total instruction latency, and the share of the loop's latency that is stall.

    python mfma_parse.py counters|att RUN
"""

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

MAC_PER_MFMA = 16 * 16 * 32


def counters(run):
    rows = []
    for d in sorted((run / "prof_tmp").glob("*__*")):
        shape, kernel = d.name.split("__", 1)
        M, N, K = map(int, shape.split("x"))
        parts = []
        for f in glob.glob(str(d / "pass_*" / "*counter_collection.csv")):
            c = pd.read_csv(f)
            c = c[c.Kernel_Name.str.startswith("_gemm")]
            parts.append(c.pivot_table(index="Dispatch_Id", columns="Counter_Name", values="Counter_Value", aggfunc="sum"))
        if not parts:
            continue
        # The passes replay the same dispatches in the same order: align them by rank.
        parts = [p.sort_index().reset_index(drop=True) for p in parts]
        w = pd.concat(parts, axis=1).iloc[1:]
        r = dict(M=M, N=N, K=K, kernelName=kernel, dispatches=len(w))
        r["mfma_util_busy"] = (w.SQ_VALU_MFMA_BUSY_CYCLES / (w.SQ_BUSY_CU_CYCLES * 4)).mean()
        r["mfma_util_gpu"] = (w.SQ_VALU_MFMA_BUSY_CYCLES / (w.GRBM_GUI_ACTIVE * 128)).mean()
        r["useful_share"] = (M * N * K / (w.SQ_INSTS_MFMA * MAC_PER_MFMA)).mean()
        r["lds_wait_per_mfma"] = (w.SQ_WAIT_INST_LDS / w.SQ_INSTS_MFMA).mean()
        r["lds_bank_conflict_per_lds"] = (w.SQ_LDS_BANK_CONFLICT / w.SQ_INSTS_LDS).mean()
        rows.append(r)
    out = pd.DataFrame(rows)
    out.to_csv(run / "mfma_counters.csv", index=False)
    print(out.round(4).to_string(index=False))


def parse_att_dir(d):
    """Rows (main loop, second loop) for one rocprofv3 --att output directory."""
    stats = sorted(Path(d).glob("stats_*.csv"))
    if not stats:
        return []
    s = pd.read_csv(stats[0]).reset_index(drop=True)
    is_mfma = s.Instruction.str.contains("v_mfma", na=False)
    # Loops: runs of executed MFMAs, split where more than 64 instructions separate two
    # MFMAs (the sub-tile kernel's tail loop runs over the same K, so has the same hits).
    idx = s.index[is_mfma & (s.Hitcount > 0)].to_numpy()
    if not len(idx):
        return []
    clusters = np.split(idx, np.where(np.diff(idx) > 64)[0] + 1)
    rows = []
    for rank, cl in enumerate(sorted(clusters, key=len, reverse=True)[:2]):
        loop = s.loc[cl.min():cl.max()]
        lm = loop[loop.Instruction.str.contains("v_mfma", na=False)]
        lat = loop.Latency.sum()
        op = loop.Instruction.str.split().str[0]
        cls = np.select([op.str.startswith("v_mfma"), op.str.startswith("s_waitcnt"), op.str.startswith("ds_"),
                         op.str.startswith("buffer_") | op.str.startswith("global_"), op.str.startswith("s_barrier")],
                        ["mfma", "waitcnt", "lds", "vmem", "barrier"], "other")
        by = loop.assign(cls=cls).groupby("cls").Latency.sum() / lat
        rows.append(dict(loop="main" if rank == 0 else "second", loop_mfma=len(lm),
                         loop_hits=int(lm.Hitcount.max()), mfma_issue_eff=16 * lm.Hitcount.sum() / lat,
                         loop_share_of_kernel=lat / s.Latency.sum(), **{f"lat_{k}": v for k, v in by.items()}))
    return rows


def att(run):
    rows = []
    for d in sorted((run / "prof_tmp").glob("att_*")):
        if d.is_dir():
            rows += [dict(kernel=d.name[4:], **r) for r in parse_att_dir(d)]
    out = pd.DataFrame(rows)
    out.to_csv(run / "mfma_att.csv", index=False)
    print(out.round(4).to_string(index=False))


if __name__ == "__main__":
    {"counters": counters, "att": att}[sys.argv[1]](Path(sys.argv[2]))
