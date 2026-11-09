# Grid: 912 shapes (720 factorial, 64 ragged N, 128 ragged M)

## Noise between the two passes (pass B / pass A, each method's best per shape)

median 1.0022, 5-95% 0.985-1.022, median |difference| 0.60% over 3920 pairs.

## Padding gate

| finer tiles when padding share > | geomean loss vs best (grid) | shapes >3% slower |
|---|---|---|
| no gate | 1.0095 | 9.0% |
| 0.020 | 1.0027 | 1.6% |
| 0.050 | 1.0027 | 1.6% |
| 0.075 | 1.0027 | 1.6% |
| 0.100 | 1.0019 | 0.7% |
| 0.150 | 1.0034 | 2.9% |
| 0.200 | 1.0034 | 2.9% |
| 0.300 | 1.0034 | 2.9% |

Chosen: finer tiles when more than 10.0% of the 256x256 tile grid is padding.

## Fitted rule (per K, over all W and M under the gate; finer tiles where no tail method applies)

| K | tail method | use it up to S/P | bootstrap: same method | bootstrap S/P 5-95% | geomean loss vs best |
|---|---|---|---|---|---|
| 1024 | sub-tile tail | 0.5000 | 100% | 0.500-0.500 | 1.0008 |
| 2048 | sub-tile tail | 0.5000 | 100% | 0.500-0.500 | 1.0012 |
| 4096 | sub-tile tail | 0.5000 | 100% | 0.500-0.500 | 1.0026 |
| 8192 | Stream-K | 0.5625 | 100% | 0.500-0.938 | 1.0017 |
| 16384 | Stream-K | 0.9375 | 100% | 0.500-0.938 | 1.0012 |

Fitted per K and W:

| K | W=1 | W=2 | W=4 |
|---|---|---|---|
| 1024 | sub <= 0.500 | sub <= 0.500 | sub <= 0.500 |
| 2048 | sub <= 0.500 | sub <= 0.500 | SK <= 0.750 |
| 4096 | sub <= 0.500 | sub <= 0.500 | SK <= 0.875 |
| 8192 | SK <= 0.562 | SK <= 0.812 | SK <= 0.938 |
| 16384 | SK <= 0.562 | SK <= 0.500 | SK <= 0.938 |

Fitted per K and M:

| K | M=512 | M=1024 | M=4096 | M=700 |
|---|---|---|---|---|
| 1024 | sub <= 0.500 | sub <= 0.500 | sub <= 0.500 | nan |
| 2048 | sub <= 0.500 | sub <= 0.500 | sub <= 0.500 | sub <= 0.500 |
| 4096 | sub <= 0.500 | sub <= 0.500 | sub <= 0.500 | nan |
| 8192 | SK <= 0.500 | SK <= 0.938 | SK <= 0.562 | SK <= 0.500 |
| 16384 | SK <= 0.562 | SK <= 0.938 | SK <= 0.500 | nan |

## Rules on the grid

| rule | vs base | slower than best per shape | shapes >3% slower than best |
|---|---|---|---|
| finer tiles everywhere | 1.051 | 1.0206 | 29% |
| sub-tile tail everywhere | 1.053 | 1.0187 | 18% |
| Stream-K everywhere | 1.051 | 1.0208 | 21% |
| finer tiles; Stream-K for K > 4096, S/P <= 1/2 | 1.056 | 1.0161 | 19% |
| AITER-study rule (+ sub-tile for K <= 2048, S/P <= 1/2) | 1.059 | 1.0124 | 11% |
| fitted rule, no padding gate | 1.062 | 1.0095 | 9% |
| fitted rule | 1.070 | 1.0019 | 1% |
| recommended rule (padding <= 10%, S/P <= 1/2: sub-tile K <= 4096, Stream-K K > 4096) | 1.070 | 1.0021 | 1% |
| best per shape | 1.073 | 1.0000 | 0% |

## Stream-K's split against its persistent loop (at most half a wave left)

| K | persistent loop vs base | split vs persistent loop |
|---|---|---|
| 1024 | 1.037 | 1.016 |
| 2048 | 1.013 | 1.029 |
| 4096 | 1.007 | 1.082 |
| 8192 | 1.004 | 1.112 |
| 16384 | 1.004 | 1.156 |

## Validation on the 491 AITER shapes with M > 128 (main run's replays; speedup over the base)

| rule | all (491) | S/P <= 1/2, K <= 2048 (47) | S/P <= 1/2, 2048 < K <= 4096 (56) | S/P <= 1/2, K > 4096 (156) | S/P > 1/2 (232) | time-weighted, all | shapes >3% slower than best |
|---|---|---|---|---|---|---|---|
| finer tiles everywhere | 1.061 | 1.062 | 1.062 | 1.087 | 1.042 | 1.023 | 23% |
| finer tiles; Stream-K for K > 4096, S/P <= 1/2 | 1.071 | 1.062 | 1.062 | 1.120 | 1.042 | 1.033 | 8% |
| AITER-study rule | 1.073 | 1.089 | 1.062 | 1.120 | 1.042 | 1.033 | 5% |
| fitted rule, no padding gate | 1.068 | 1.082 | 1.060 | 1.107 | 1.042 | 1.030 | 12% |
| fitted rule | 1.075 | 1.089 | 1.076 | 1.122 | 1.041 | 1.034 | 3% |
| recommended rule | 1.075 | 1.089 | 1.076 | 1.122 | 1.042 | 1.034 | 2% |
| best per shape | 1.078 | 1.098 | 1.082 | 1.126 | 1.043 | 1.037 | 0% |

