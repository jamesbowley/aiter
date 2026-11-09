"""The transfer test's shapes for the v2 pipeline: every M of the 12 hipBLASLt anchor
ranges (anchors_hbl*.csv, m_grid in aiter_tune/runs/transfer/make_rows.py), in
shapes.py's format, instead of the shipped tuned rows.

    python anchor_shapes.py runs/<name>     # writes shapes.csv and untuned.csv
"""

import sys
from pathlib import Path

import pandas as pd

from shapes import KEY, label

TRANSFER = Path("/home/jbowley/repos/aiter/streamk_study/aiter_tune/runs/transfer")
sys.path.insert(0, str(TRANSFER))
from make_rows import m_grid  # noqa: E402


def main(out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    anchors = pd.concat(
        [pd.read_csv(TRANSFER / f) for f in ("anchors_hbl.csv", "anchors_hbl_large_m.csv")]
    )
    rows = [dict(M=M, N=a.N, K=a.K, bias=False, anchor_M0=a.M0)
            for a in anchors.itertuples() for M in m_grid(a.M0)]
    d = pd.DataFrame(rows)
    anchors_of = d.groupby(KEY).anchor_M0.apply(lambda s: ";".join(map(str, sorted(s))))
    d = d.drop(columns="anchor_M0").drop_duplicates(KEY).set_index(KEY)
    d["anchor_M0"] = anchors_of
    d = d.reset_index()
    d["shipped_libtype"] = ""
    d["shipped_kernel"] = ""
    d["tuned_us"] = float("nan")
    d["source"] = "transfer_anchors"
    shapes = label(d).sort_values(KEY)
    shapes.to_csv(out / "shapes.csv", index=False)
    shapes[KEY].assign(
        dtype="torch.bfloat16", outdtype="torch.bfloat16", scaleAB=False, bpreshuffle=False
    ).to_csv(out / "untuned.csv", index=False)
    print(f"{len(shapes)} shapes from {len(anchors)} anchors")
    print(shapes.groupby("set").size().to_string())


if __name__ == "__main__":
    main(sys.argv[1])
