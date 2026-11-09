# Tables for the tile-quantization issue

Run `runs/full`: 491 shapes with more than one wave of 256x256 tiles and a partial last wave (S/P below is at 256x256). Cells: geomean / faster by >3% / slower by >3%.

## Fixed 256x256 tile (tuning profile): tail method over whole 256x256 tiles

By K, at most half a wave left (tuned columns: no fallback to whole tiles):

| K | n | Stream-K split vs whole 256x256 | sub-tile vs whole 256x256 | Stream-K split, tuned tile, vs base | sub-tile, tuned tile, vs base |
|---|---|---|---|---|---|
| K<=1024 | 12 | 1.009 / 33% / 25% (n=12) | 1.150 / 92% / 0% (n=12) | 0.993 / 33% / 33% (n=12) | 1.105 / 92% / 0% (n=12) |
| K<=2048 | 35 | 1.045 / 51% / 3% (n=35) | 1.088 / 80% / 0% (n=35) | 1.043 / 54% / 3% (n=35) | 1.084 / 80% / 0% (n=35) |
| K<=4096 | 56 | 1.058 / 64% / 0% (n=56) | 1.069 / 68% / 2% (n=56) | 1.053 / 62% / 0% (n=56) | 1.061 / 66% / 4% (n=56) |
| K<=8192 | 130 | 1.122 / 88% / 1% (n=130) | 1.111 / 87% / 0% (n=130) | 1.117 / 88% / 1% (n=130) | 1.107 / 86% / 0% (n=130) |
| K>8192 | 26 | 1.150 / 92% / 0% (n=26) | 1.108 / 85% / 0% (n=26) | 1.149 / 92% / 0% (n=26) | 1.108 / 85% / 0% (n=26) |
| all | 259 | 1.094 / 76% / 2% (n=259) | 1.100 / 82% / 0% (n=259) | 1.090 / 76% / 2% (n=259) | 1.094 / 81% / 1% (n=259) |

By K, more than half a wave left (tuned columns: no fallback to whole tiles):

| K | n | Stream-K split vs whole 256x256 | sub-tile vs whole 256x256 | Stream-K split, tuned tile, vs base | sub-tile, tuned tile, vs base |
|---|---|---|---|---|---|
| K<=1024 | 9 | 0.804 / 0% / 100% (n=9) |  | 0.933 / 0% / 67% (n=9) | 0.893 / 0% / 89% (n=9) |
| K<=2048 | 31 | 0.861 / 0% / 100% (n=31) |  | 0.886 / 0% / 100% (n=31) | 0.854 / 0% / 100% (n=31) |
| K<=4096 | 113 | 0.957 / 0% / 57% (n=113) |  | 0.963 / 0% / 49% (n=113) | 0.673 / 0% / 100% (n=70) |
| K<=8192 | 74 | 0.976 / 3% / 51% (n=74) |  | 0.976 / 4% / 51% (n=74) | 0.820 / 1% / 99% (n=67) |
| K>8192 | 5 | 1.029 / 40% / 0% (n=5) |  | 1.029 / 40% / 0% (n=5) | 0.838 / 0% / 100% (n=5) |
| all | 232 | 0.945 / 2% / 61% (n=232) |  | 0.957 / 2% / 56% (n=232) | 0.769 / 1% / 99% (n=182) |

By last-wave fill:

| fill | n | stream_k_split | sub_tile |
|---|---|---|---|
| S/P<=1/4 | 128 | 1.145 / 85% / 1% (n=128) | 1.153 / 88% / 0% (n=128) |
| S/P<=1/2 | 131 | 1.047 / 66% / 3% (n=131) | 1.051 / 76% / 1% (n=131) |
| S/P<=3/4 | 96 | 0.964 / 4% / 47% (n=96) |  |
| S/P<1 | 136 | 0.931 / 0% / 71% (n=136) |  |
| all | 491 | 1.021 / 41% / 30% (n=491) | 1.100 / 82% / 0% (n=259) |

## End to end, by last-wave fill (each method with whole tiles as fallback)

| comparison | S/P<=1/4 (128) | S/P<=1/2 (131) | S/P<=3/4 (96) | S/P<1 (136) | all |
|---|---|---|---|---|---|
| uneven vs base | 1.118 / 80% / 0% (n=128) | 1.039 / 52% / 0% (n=131) | 1.055 / 55% / 0% (n=96) | 1.033 / 32% / 0% (n=136) | 1.061 / 54% / 0% (n=491) |
| sub-tile vs base | 1.137 / 87% / 0% (n=128) | 1.051 / 73% / 0% (n=131) | 1.001 / 0% / 0% (n=96) | 1.000 / 0% / 0% (n=136) | 1.048 / 42% / 0% (n=491) |
| Stream-K vs base | 1.133 / 81% / 0% (n=128) | 1.050 / 69% / 0% (n=131) | 1.006 / 6% / 0% (n=96) | 1.001 / 0% / 0% (n=136) | 1.048 / 41% / 0% (n=491) |
| uneven vs sub-tile | 0.983 / 6% / 34% (n=128) | 0.989 / 6% / 17% (n=131) | 1.055 / 55% / 0% (n=96) | 1.033 / 32% / 0% (n=136) | 1.012 / 23% / 13% (n=491) |
| uneven vs Stream-K | 0.987 / 16% / 35% (n=128) | 0.989 / 8% / 32% (n=131) | 1.050 / 47% / 0% (n=96) | 1.032 / 32% / 0% (n=136) | 1.012 / 25% / 18% (n=491) |
| sub-tile vs Stream-K | 1.004 / 16% / 15% (n=128) | 1.000 / 10% / 9% (n=131) | 0.995 / 0% / 6% (n=96) | 0.999 / 0% / 0% (n=136) | 1.000 / 7% / 8% (n=491) |
| Stream-K kernel, any plan or tile, vs base | 1.134 / 83% / 0% (n=128) | 1.056 / 75% / 0% (n=131) | 1.016 / 12% / 0% (n=96) | 1.014 / 6% / 0% (n=136) | 1.056 / 46% / 0% (n=491) |

## End to end, by K (each method with whole tiles as fallback)

| comparison | K<=1024 (21) | K<=2048 (66) | K<=4096 (169) | K<=8192 (204) | K>8192 (31) | all |
|---|---|---|---|---|---|---|
| uneven vs base | 1.056 / 57% / 0% (n=21) | 1.034 / 36% / 0% (n=66) | 1.049 / 46% / 0% (n=169) | 1.076 / 64% / 0% (n=204) | 1.084 / 74% / 0% (n=31) | 1.061 / 54% / 0% (n=491) |
| sub-tile vs base | 1.057 / 48% / 0% (n=21) | 1.044 / 42% / 0% (n=66) | 1.019 / 21% / 0% (n=169) | 1.066 / 54% / 0% (n=204) | 1.093 / 71% / 0% (n=31) | 1.048 / 42% / 0% (n=491) |
| Stream-K vs base | 1.020 / 29% / 0% (n=21) | 1.027 / 32% / 0% (n=66) | 1.016 / 20% / 0% (n=169) | 1.073 / 56% / 0% (n=204) | 1.133 / 87% / 0% (n=31) | 1.048 / 41% / 0% (n=491) |
| uneven vs sub-tile | 0.999 / 33% / 19% (n=21) | 0.990 / 8% / 14% (n=66) | 1.029 / 33% / 7% (n=169) | 1.010 / 19% / 17% (n=204) | 0.991 / 23% / 23% (n=31) | 1.012 / 23% / 13% (n=491) |
| uneven vs Stream-K | 1.036 / 48% / 14% (n=21) | 1.007 / 21% / 14% (n=66) | 1.032 / 36% / 3% (n=169) | 1.003 / 17% / 24% (n=204) | 0.957 / 3% / 68% (n=31) | 1.012 / 25% / 18% (n=491) |
| sub-tile vs Stream-K | 1.037 / 38% / 5% (n=21) | 1.017 / 26% / 3% (n=66) | 1.003 / 4% / 1% (n=169) | 0.993 / 1% / 6% (n=204) | 0.965 / 0% / 65% (n=31) | 1.000 / 7% / 8% (n=491) |
| Stream-K kernel, any plan or tile, vs base | 1.077 / 81% / 0% (n=21) | 1.043 / 50% / 0% (n=66) | 1.022 / 20% / 0% (n=169) | 1.074 / 56% / 0% (n=204) | 1.133 / 87% / 0% (n=31) | 1.056 / 46% / 0% (n=491) |

## By number of full waves

| full waves at 256x256 | n | uneven vs base | sub-tile vs base | Stream-K vs base | uneven vs Stream-K |
|---|---|---|---|---|---|
| 1 | 185 | 1.109 / 78% / 0% (n=185) | 1.079 / 45% / 0% (n=185) | 1.077 / 45% / 0% (n=185) | 1.030 / 39% / 12% (n=185) |
| 2 | 76 | 1.062 / 82% / 0% (n=76) | 1.064 / 75% / 0% (n=76) | 1.060 / 66% / 0% (n=76) | 1.002 / 24% / 28% (n=76) |
| 3-4 | 92 | 1.034 / 43% / 0% (n=92) | 1.032 / 51% / 0% (n=92) | 1.036 / 51% / 0% (n=92) | 0.998 / 16% / 30% (n=92) |
| 5-8 | 67 | 1.025 / 25% / 0% (n=67) | 1.016 / 28% / 0% (n=67) | 1.018 / 28% / 0% (n=67) | 1.007 / 18% / 21% (n=67) |
| >8 | 71 | 1.004 / 6% / 0% (n=71) | 1.004 / 1% / 0% (n=71) | 1.006 / 3% / 0% (n=71) | 0.998 / 6% / 1% (n=71) |

## Intensity of the picks

Median BM BN / (BM + BN) of the pick: base 128.0, uneven 119.5 (256x256: 128).

## Worked example: 1024x16896x7168 (264 tiles at 256x256, 8 in the last wave)

| tile | tiles | last_wave | useful_share | intensity | us |
|---|---|---|---|---|---|
| 256x256 | 264 | 8 | 0.52 | 128 | 266.2 |
| 256x224 | 304 | 48 | 0.59 | 119 | 247.1 |
| 256x192 | 352 | 96 | 0.69 | 110 | 226.2 |
| 256x160 | 424 | 168 | 0.82 | 98 | 213.6 |
| 128x288 | 472 | 216 | 0.92 | 89 | 224.1 |

Replay: base 261.0 us, uneven 213.4, sub-tile 193.3, Stream-K 181.8.

## hipBLASLt's choice (torch, this session)

Main shapes: 491. Launch grid against tile count: one workgroup per tile on 416, a whole number of tiles per workgroup on 71, a fractional number (where a Stream-K split can run) on 4. Stream-K mode in the kernel names: {5: 491}. Tiles larger than 256x256 of accumulators on 336; the largest is 224x384 (N x M).

With the last wave at most 1/4 full at 256x256 (128 shapes), hipBLASLt's tile leaves the last wave exactly full on 30, at least 3/4 full on 117, and at most 1/4 full on 3.

Most used tiles (N x M):

| tile | shapes |
|---|---|
| 256x320 | 82 |
| 288x288 | 44 |
| 224x384 | 31 |
| 256x256 | 30 |
| 384x208 | 29 |
| 272x256 | 26 |
| 448x192 | 19 |
| 224x256 | 18 |
| 256x224 | 17 |
| 448x176 | 15 |

## Against hipBLASLt (raw, t_hipBLASLt / t_ours)

| ours | all | hipBLASLt tile <= 256x256 (155) | hipBLASLt tile > 256x256 (336) |
|---|---|---|---|
| base (power-of-two tiles) | 0.977 / 40% / 38% (n=491) | 1.008 / 50% / 31% (n=155) | 0.963 / 35% / 42% (n=336) |
| finer tiles | 1.036 / 54% / 14% (n=491) | 1.066 / 68% / 6% (n=155) | 1.022 / 47% / 18% (n=336) |
| sub-tile tail | 1.024 / 56% / 19% (n=491) | 1.040 / 60% / 15% (n=155) | 1.016 / 54% / 21% (n=336) |
| Stream-K | 1.024 / 57% / 21% (n=491) | 1.038 / 61% / 16% (n=155) | 1.017 / 55% / 23% (n=336) |
| best of the three | 1.053 / 68% / 9% (n=491) | 1.073 / 76% / 6% (n=155) | 1.044 / 64% / 10% (n=336) |

Base re-timed in the torch session against its earlier replay: median 0.999, 5-95% 0.978-1.019.

