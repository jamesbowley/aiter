# Prefill (M > 512): a tile fixed at M0 against a tile chosen per M (v9)

Our Gluon v9 kernel (gemm_a16w16.py, power-of-two tiles) and the uneven tiles (gemm_a16w16_uneven.py), every tile timed at every M with the tuner (transfer12). 9 anchors with M0 > 512, 191 points with 512 < M < M0. Comparisons: (1) v9_fixed / v9_best, fixing v9's tile with the tiles we have; (2) v9_best / best_uneven, what v9's menu leaves on the table; (3) uneven_fixed / best_uneven, fixing the tile with the finer menu.

## 1. Pooled

| group               | comparison                    |   points |   geomean |   worst | above_5pct   |
|:--------------------|:------------------------------|---------:|----------:|--------:|:-------------|
| all anchors         | 1: v9_fixed / v9_best         |      191 |     1.017 |   1.367 | 7%           |
| all anchors         | 2: v9_best / best_uneven      |      191 |     1.053 |   1.313 | 35%          |
| all anchors         | 3: uneven_fixed / best_uneven |      191 |     1.055 |   1.451 | 33%          |
| all anchors         | v9_fixed / best_uneven        |      191 |     1.07  |   1.442 | 40%          |
| small N (N <= 2560) | 1: v9_fixed / v9_best         |       34 |     1.096 |   1.367 | 41%          |
| small N (N <= 2560) | 2: v9_best / best_uneven      |       34 |     1.129 |   1.313 | 71%          |
| small N (N <= 2560) | 3: uneven_fixed / best_uneven |       34 |     1.132 |   1.424 | 62%          |
| small N (N <= 2560) | v9_fixed / best_uneven        |       34 |     1.238 |   1.442 | 100%         |
| large N (N > 2560)  | 1: v9_fixed / v9_best         |      157 |     1     |   1.008 | 0%           |
| large N (N > 2560)  | 2: v9_best / best_uneven      |      157 |     1.037 |   1.222 | 27%          |
| large N (N > 2560)  | 3: uneven_fixed / best_uneven |      157 |     1.039 |   1.451 | 27%          |
| large N (N > 2560)  | v9_fixed / best_uneven        |      157 |     1.037 |   1.222 | 27%          |

## 2. Per anchor: v9 menu against the menu with uneven tiles

fixed_loss = geomean of (tile fixed at M0 / best tile of the same menu per M); changes = M values where the menu's best tile differs from its M0 pick; distinct = number of different best tiles over the range.

|     N |    K |    M0 | M_range    | v9_tile_M0   |   v9_fixed_loss |   v9_worst | v9_above_5pct   | v9_changes   |   v9_distinct | uneven_tile_M0   |   uneven_fixed_loss |   uneven_worst | uneven_above_5pct   | uneven_changes   |   uneven_distinct |   v9_best_vs_uneven |   v9_fixed_vs_uneven |
|------:|-----:|------:|:-----------|:-------------|----------------:|-----------:|:----------------|:-------------|--------------:|:-----------------|--------------------:|---------------:|:--------------------|:-----------------|------------------:|--------------------:|---------------------:|
|  1536 | 1536 |  8192 | 4352-8191  | 256x256      |           1.086 |      1.367 | 29%             | 5/17         |             3 | uneven_256x192   |               1.105 |          1.276 | 71%                 | 15/17            |                10 |               1.153 |                1.253 |
|  2560 | 8192 |  8192 | 4352-8191  | 128x256      |           1.106 |      1.233 | 53%             | 14/17        |             4 | uneven_256x160   |               1.16  |          1.424 | 53%                 | 12/17            |                11 |               1.106 |                1.223 |
|  5120 | 5120 |  4096 | 2304-4095  | 256x256      |           1.002 |      1.008 | 0%              | 2/9          |             3 | uneven_256x160   |               1.15  |          1.451 | 44%                 | 9/9              |                 7 |               1.121 |                1.123 |
|  5120 | 5120 |  8192 | 4352-8191  | 256x256      |           1     |      1     | 0%              | 0/17         |             1 | uneven_256x224   |               1.054 |          1.203 | 47%                 | 14/17            |                 9 |               1.053 |                1.053 |
|  5120 | 5120 | 16384 | 8448-16383 | 256x256      |           1     |      1     | 0%              | 0/33         |             1 | 256x256          |               1.018 |          1.066 | 9%                  | 19/33            |                 8 |               1.018 |                1.018 |
|  6144 | 4608 |  8192 | 4352-8191  | 256x256      |           1     |      1     | 0%              | 0/17         |             1 | 256x256          |               1.033 |          1.135 | 24%                 | 10/17            |                 7 |               1.033 |                1.033 |
|  6144 | 4608 | 16384 | 8448-16383 | 256x256      |           1     |      1     | 0%              | 0/33         |             1 | 256x256          |               1.015 |          1.058 | 9%                  | 12/33            |                 6 |               1.015 |                1.015 |
| 64640 | 4096 |  1024 | 528-1023   | 256x256      |           1     |      1     | 0%              | 0/33         |             1 | 256x256          |               1.06  |          1.187 | 55%                 | 22/33            |                 5 |               1.06  |                1.06  |
| 64640 | 4096 |  1792 | 960-1791   | 256x256      |           1     |      1     | 0%              | 0/15         |             1 | 256x256          |               1.017 |          1.083 | 13%                 | 5/15             |                 4 |               1.017 |                1.017 |

## 3. Best tiles over the range (in M order)

|     N |    K |    M0 | v9_best_tiles                      | best_uneven_tiles                                                                                                                                                       |
|------:|-----:|------:|:-----------------------------------|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  1536 | 1536 |  8192 | 128x128, 128x256, 256x256          | uneven_192x160, uneven_160x192, uneven_224x160, uneven_160x224, uneven_192x192, uneven_256x160, uneven_160x256, uneven_224x192, uneven_192x256, uneven_256x192          |
|  2560 | 8192 |  8192 | 256x256, 128x128, 128x256, 256x128 | uneven_192x256, uneven_256x192, uneven_160x320, uneven_320x160, uneven_224x256, uneven_320x192, 256x256, uneven_192x320, uneven_160x224, uneven_224x160, uneven_256x160 |
|  5120 | 5120 |  4096 | 256x256, 128x128, 256x128          | uneven_256x192, uneven_320x160, uneven_256x224, uneven_192x320, uneven_224x160, uneven_160x256, uneven_192x224                                                          |
|  5120 | 5120 |  8192 | 256x256                            | uneven_192x256, uneven_224x224, uneven_320x160, uneven_224x256, uneven_256x224, 256x256, uneven_256x192, uneven_192x288, uneven_160x320                                 |
|  5120 | 5120 | 16384 | 256x256                            | uneven_224x256, 256x256, uneven_320x192, uneven_192x320, uneven_160x320, uneven_320x160, uneven_192x288, uneven_256x224                                                 |
|  6144 | 4608 |  8192 | 256x256                            | uneven_192x288, uneven_256x224, 256x256, uneven_256x192, uneven_288x192, uneven_224x256, uneven_320x192                                                                 |
|  6144 | 4608 | 16384 | 256x256                            | uneven_192x288, uneven_256x224, uneven_224x256, uneven_288x192, 256x256, uneven_256x192                                                                                 |
| 64640 | 4096 |  1024 | 256x256                            | uneven_192x256, uneven_224x256, 256x256, uneven_288x192, uneven_320x192                                                                                                 |
| 64640 | 4096 |  1792 | 256x256                            | uneven_320x192, 256x256, uneven_224x256, uneven_288x224                                                                                                                 |


## 4. Replay (--run_config, A/B mean) at every fourth M

252 replayed (method, shape) points; median A/B noise 0.4%; replay / tuner time geomean 1.000.

| comparison (below M0, replay times) | points | geomean | worst |
|---|---|---|---|
| v9_fixed / v9_best | 56 | 1.021 | 1.339 |
| v9_best / best_uneven | 56 | 1.055 | 1.237 |
| uneven_fixed / best_uneven | 56 | 1.055 | 1.452 |
| v9_fixed / best_uneven | 56 | 1.077 | 1.449 |
