#!/bin/bash
# Why does aiter's base kernel trace lower in-loop MFMA efficiency than
# gfx950-gluon-tutorials docs/tcp_bound_tile_model.md (branch tcp-bound-tile-model)? For the
# document's eight tiles: its shape (M = 16 BM, N = 16 BN, K = 8192: 256 workgroups, one per CU)
# with cold and with warm inputs, our earlier 8192x8192x8192 (cold), and its shape with Triton's
# MFMA scheduler off (AITER_MFMA_SCHED=0). Each trace is reduced by our parser and by the
# tutorial's scripts/process_json.py, with the kernel's VGPR/LDS use from a kernel trace;
# raw output and the run's Triton cache are deleted. One GPU.
#
#   tmux new -d -s attdoc "./att_doc.sh runs/att_doc 3"
set -eu
V2=/home/jbowley/repos/aiter/streamk_study/tile_vs_tail/v2
RUN=$(realpath -m "$1")
GPU=$2
mkdir -p $RUN
cd $V2
source $V2/../env.sh
export PYTHONPATH=$V2/flydsl_0341:$V2:$PYTHONPATH
export TRITON_CACHE_DIR=$V2/triton_cache_attdoc
export HIP_VISIBLE_DEVICES=$GPU
SDK=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel
export PATH=$SDK/bin:$PATH
export LD_LIBRARY_PATH=$SDK/lib:${LD_LIBRARY_PATH:-}
PJ=/home/jbowley/repos/gfx950-gluon-tutorials/scripts/process_json.py
OUT=$RUN/att_doc.csv
TMP=$RUN/prof_tmp
for tile in 256x256 256x128 128x256 128x128 256x64 64x256 128x64 64x128; do
    BM=${tile%x*}; BN=${tile#*x}
    for cond in doc_cold doc_warm big_cold doc_cold_nosched; do
        case $cond in
            big_cold) M=8192; N=8192 ;;
            *) M=$((16 * BM)); N=$((16 * BN)) ;;
        esac
        K=8192
        sets=cold; [ $cond = doc_warm ] && sets=1
        sched=1; [ $cond = doc_cold_nosched ] && sched=0
        tag=${tile}__${cond}
        [ -f $OUT ] && grep -q "^$tag," $OUT && continue
        rm -rf $TMP && mkdir -p $TMP
        # Trace the 30th of 40 calls: steady state (the second call, right after allocating the
        # input copies, reads cold inputs 10-15 points slower).
        MFMA_REPS=40 MFMA_SETS=$sets AITER_MFMA_SCHED=$sched rocprofv3 -d $TMP/att -o att --kernel-trace --output-format csv \
            --att --att-target-cu 0 --att-shader-engine-mask 0x1 --att-buffer-size 0x40000000 \
            --kernel-include-regex _gemm_a16w16 --kernel-iteration-range "[30]" \
            -- python3 $V2/mfma_run.py $M $N $K gluon_compute_bound_$tile > $TMP/run.out 2>&1 || true
        ui=$(ls -d $TMP/att/ui_output_agent_* 2>/dev/null | head -1)
        python3 $PJ "$ui" > $TMP/pj.json 2> $TMP/pj.err || true
        python3 - "$TMP" "$OUT" "$tag" "$tile" "$cond" "$M" "$N" "$K" <<'EOF'
import glob, json, math, sys
from pathlib import Path
import pandas as pd
from mfma_parse import parse_att_dir
tmp, out, tag, tile, cond, M, N, K = sys.argv[1:]
r = dict(tag=tag, tile=tile, cond=cond, M=int(M), N=int(N), K=int(K))
rows = parse_att_dir(Path(tmp) / "att")
main = [x for x in rows if x["loop"] == "main"]
r["ours_per_wave"] = main[0]["mfma_issue_eff"] if main else None
try:
    pj = json.loads(Path(tmp, "pj.json").read_text())
    r["pj_per_wave"] = float(pj["mfma efficiency"].rstrip("%")) / 100
    r["pj_loop_ratio"] = pj.get("loop_ratio")
    r["pj_iterations"] = pj.get("num_iterations")
except Exception as e:
    r["pj_per_wave"] = None
    r["pj_error"] = str(e)[:80]
kt = glob.glob(f"{tmp}/att/**/*kernel_trace.csv", recursive=True)
if kt:
    k = pd.read_csv(kt[0])
    k = k[k.Kernel_Name.str.startswith("_gemm")].iloc[0]
    vg = int(k.VGPR_Count) + int(k.Accum_VGPR_Count)
    bm, bn = map(int, tile.split("x"))
    # The kernel trace reports no LDS for these kernels: two stages of A and B, plus ~3% padding.
    lds = int(2 * (bm + bn) * 64 * 2 * 1.03)
    wgs = int(k.Grid_Size_X * k.Grid_Size_Y * k.Grid_Size_Z) // int(k.Workgroup_Size_X * k.Workgroup_Size_Y * k.Workgroup_Size_Z)
    by_vgpr = 512 // max(vgpr_alloc := 8 * math.ceil(vg / 8), 1)
    by_lds = 163840 // max(lds, 1)
    r.update(vgpr=vg, lds=lds, workgroups=wgs, max_wg_per_cu=min(by_vgpr, by_lds, 8),
             wg_per_cu=min(by_vgpr, by_lds, 8, max(1, math.ceil(wgs / 256))))
pd.DataFrame([r]).reindex(columns=["tag", "tile", "cond", "M", "N", "K", "ours_per_wave", "pj_per_wave", "pj_loop_ratio", "pj_iterations", "pj_error", "vgpr", "lds", "workgroups", "max_wg_per_cu", "wg_per_cu"]).to_csv(out, mode="a", header=not Path(out).exists(), index=False)
print(r)
EOF
    done
done
rm -rf $TMP $TRITON_CACHE_DIR
echo done > $RUN/DONE
