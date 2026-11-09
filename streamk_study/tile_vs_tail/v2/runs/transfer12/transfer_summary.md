# Does our runtime Stream-K rescue a tile fixed at M0?

Our Gluon kernels on the 12 hipBLASLt anchors (every candidate timed at every M with the tuner). Ratios are to the best tile per M (base or uneven, no Stream-K), below M0 unless stated. See analyze_transfer.py for the definitions.

## 1. Against the best tile per M

| method                            |   points |   geomean |   worst | slower_5pct   |
|:----------------------------------|---------:|----------:|--------:|:--------------|
| tile fixed at M0                  |      234 |     1.065 |   1.451 | 38%           |
| M0 Stream-K tile + runtime plan   |      234 |     1.041 |   1.408 | 31%           |
| M0 Stream-K tile + best plan at M |      234 |     1.035 |   1.408 | 29%           |
| M0 Stream-K config fixed          |      234 |     1.056 |   1.493 | 33%           |
| best Stream-K config per M        |      234 |     1.027 |   1.296 | 29%           |
| best of everything per M          |      234 |     0.986 |   1     | 0%            |

## 2. Does the runtime plan rescue the fixed tile?

sk_runtime / tile_fixed below M0 (below 1: the Stream-K tile with its runtime plan beats the tile fixed at M0).

| comparison                |   points |   geomean |   worst | slower_5pct   |
|:--------------------------|---------:|----------:|--------:|:--------------|
| sk_runtime / tile_fixed   |      234 |     0.977 |   1.164 | 10%           |
| sk_tile_best / tile_fixed |      234 |     0.972 |   1.164 | 9%            |
| sk_runtime / sk_fixed     |      234 |     0.985 |   1.116 | 3%            |

## 3. At M0 (tuned point)

| comparison                     |   points |   geomean |   worst | slower_5pct   |
|:-------------------------------|---------:|----------:|--------:|:--------------|
| best Stream-K / best tile      |       12 |     0.996 |   1.101 | 8%            |
| best of everything / best tile |       12 |     0.984 |   1     | 0%            |

## 4. Per anchor (geomean below M0)

|     N |    K |    M0 | tile_fixed            | sk_tile    |   fixed_vs_best |   sk_runtime_vs_best |   sk_runtime_vs_fixed |   worst_fixed |   worst_sk_runtime |   runtime_splits |   sk_fixed_fallbacks |
|------:|-----:|------:|:----------------------|:-----------|----------------:|---------------------:|----------------------:|--------------:|-------------------:|-----------------:|---------------------:|
|  1536 | 1536 |  8192 | uneven_256x192        | 256x256x64 |           1.105 |                1.229 |                 1.112 |         1.276 |              1.352 |                5 |                    0 |
|  2560 | 8192 |  8192 | uneven_256x160        | 256x256x64 |           1.16  |                0.994 |                 0.857 |         1.424 |              1.101 |                8 |                    9 |
|  5120 | 5120 |  4096 | uneven_256x160        | 256x256x64 |           1.15  |                1.004 |                 0.873 |         1.451 |              1.099 |                5 |                    4 |
|  5120 | 5120 |  8192 | uneven_256x224        | 256x256x64 |           1.054 |                0.985 |                 0.935 |         1.203 |              1.054 |               11 |                    6 |
|  5120 | 5120 | 16384 | compute_bound_256x256 | 256x256x64 |           1.018 |                0.979 |                 0.962 |         1.066 |              1.028 |               12 |                    0 |
|  6144 | 4608 |  8192 | compute_bound_256x256 | 256x256x64 |           1.033 |                0.997 |                 0.965 |         1.135 |              1.038 |                5 |                    0 |
|  6144 | 4608 | 16384 | compute_bound_256x256 | 256x256x64 |           1.015 |                0.976 |                 0.962 |         1.058 |              0.996 |               16 |                    0 |
| 38720 | 6144 |   256 | uneven_256x192        | 256x128x64 |           1.04  |                1.173 |                 1.127 |         1.117 |              1.227 |                9 |                    0 |
| 38720 | 6144 |   512 | uneven_256x160        | 256x256x64 |           1.154 |                1.15  |                 0.997 |         1.44  |              1.408 |               17 |                    0 |
| 64640 | 4096 |   512 | compute_bound_256x256 | 256x256x64 |           1.114 |                1.117 |                 1.003 |         1.304 |              1.324 |                0 |                    0 |
| 64640 | 4096 |  1024 | compute_bound_256x256 | 256x256x64 |           1.06  |                1.054 |                 0.995 |         1.187 |              1.197 |                0 |                    0 |
| 64640 | 4096 |  1792 | compute_bound_256x256 | 256x256x64 |           1.017 |                1.015 |                 0.998 |         1.083 |              1.08  |                0 |                    0 |

## 5. Checks

- At M0, tile_fixed equals best_tile on 12 of 12 anchors, sk_fixed equals best_sk on 12 of 12.
- Stream-K's dp plan (best BLOCK_K) / base at the same tile, every shape: points 2196, geomean 1.003, worst 1.113, slower_5pct 10%.
- sk_fixed fell back to the runtime plan at 19 of 234 points below M0.


## 6. Replay (--run_config, A/B mean) at every fourth M

405 replayed (method, M) points; median A/B noise 0.5%; replay / tuner time geomean 1.001.

| comparison (below M0, replay times) | points | geomean |
|---|---|---|
| tile_fixed / best_tile | 69 | 1.073 |
| sk_runtime / best_tile | 69 | 1.05 |
| sk_fixed / best_tile | 69 | 1.057 |
| best_sk / best_tile | 69 | 1.035 |
| sk_runtime / tile_fixed | 69 | 0.979 |
