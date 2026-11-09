# Uneven tiles against sub-tile tails: full

1522 shapes. Times from aiter's --run_config replay of each method's pick (picked in the tuning run). Speedup = t_other / t_ours; faster / slower beyond +-3%. HBM peak 5.48 TB/s (bw_peak.txt); memory-bound when the minimum traffic 2(MK + NK + MN) at the best time reaches 80% of it, mixed from 50%.

Missing replays per column: pow2_split 180, subtile 755

Methods: `uneven` = best non-power-of-two data-parallel tile; `subtile` = best (main tile, sub-tile tail); `pow2_split` = best power-of-two Stream-K split; `pow2_dp` = best power-of-two data-parallel; `pow2_best` = the best of those and our previous pick (the PR's kernels); `dp_any` = min(uneven, pow2_dp); `v32_any` = min(subtile, pow2_best); `ours_new` = the best of everything.

## Shapes by regime (at our previous tile) and K

| regime | K<=1024 | K<=2048 | K<=4096 | K<=8192 | K>8192 |
|---|---|---|---|---|---|
| below_one_wave | 54 | 103 | 269 | 312 | 46 |
| one_wave_tail | 7 | 29 | 48 | 91 | 12 |
| two_plus_waves_tail | 8 | 25 | 148 | 90 | 17 |
| no_tail | 42 | 40 | 56 | 87 | 38 |

## Q1. Uneven data-parallel against sub-tile tails

Pure methods, on the 767 shapes where both have a candidate (speedup = t_subtile / t_uneven, above 1 the uneven tile wins):

| group | shapes | faster | parity | slower | geomean | median |
|---|---|---|---|---|---|---|
| below_one_wave | 336 | 321 (96%) | 11 (3%) | 4 (1%) | 1.228 | +20.0% |
| one_wave_tail | 161 | 96 (60%) | 26 (16%) | 39 (24%) | 1.105 | +10.5% |
| two_plus_waves_tail | 255 | 109 (43%) | 71 (28%) | 75 (29%) | 1.126 | -0.4% |
| no_tail | 15 | 14 (93%) | 1 (7%) | 0 (0%) | 1.439 | +43.0% |
| all | 767 | 540 (70%) | 109 (14%) | 118 (15%) | 1.171 | +14.7% |

| group | shapes | faster | parity | slower | geomean | median |
|---|---|---|---|---|---|---|
| K<=1024 | 32 | 23 (72%) | 3 (9%) | 6 (19%) | 1.231 | +26.5% |
| K<=2048 | 101 | 80 (79%) | 5 (5%) | 16 (16%) | 1.211 | +24.2% |
| K<=4096 | 250 | 199 (80%) | 29 (12%) | 22 (9%) | 1.219 | +15.6% |
| K<=8192 | 326 | 205 (63%) | 62 (19%) | 59 (18%) | 1.130 | +11.6% |
| K>8192 | 58 | 33 (57%) | 10 (17%) | 15 (26%) | 1.103 | +7.2% |
| all | 767 | 540 (70%) | 109 (14%) | 118 (15%) | 1.171 | +14.7% |

After at least one full wave, by tail fraction S / P at our previous tile:

| group | shapes | faster | parity | slower | geomean | median |
|---|---|---|---|---|---|---|
| S/P<=1/4 | 121 | 23 (19%) | 34 (28%) | 64 (53%) | 0.991 | -3.1% |
| S/P<=1/2 | 124 | 15 (12%) | 61 (49%) | 48 (39%) | 0.991 | -2.1% |
| S/P<=3/4 | 88 | 84 (95%) | 2 (2%) | 2 (2%) | 1.172 | +16.7% |
| S/P<1 | 83 | 83 (100%) | 0 (0%) | 0 (0%) | 1.517 | +40.3% |
| all | 416 | 205 (49%) | 97 (23%) | 114 (27%) | 1.118 | +1.9% |

With fallbacks on every shape (speedup = t_v32_any / t_dp_any): each method may also use its power-of-two kernels.

| group | shapes | faster | parity | slower | geomean | median |
|---|---|---|---|---|---|---|
| below_one_wave | 784 | 162 (21%) | 202 (26%) | 420 (54%) | 0.930 | -4.2% |
| one_wave_tail | 187 | 55 (29%) | 93 (50%) | 39 (21%) | 1.017 | +0.0% |
| two_plus_waves_tail | 288 | 51 (18%) | 170 (59%) | 67 (23%) | 1.001 | -0.4% |
| no_tail | 263 | 9 (3%) | 241 (92%) | 13 (5%) | 0.996 | +0.0% |
| all | 1522 | 277 (18%) | 706 (46%) | 539 (35%) | 0.965 | -0.4% |

Joint search of the sub-tile tail's main tile: the best sub-tile pick's main tile differs from our previous pick's tile on 538 of 767 shapes.

## Q2. Against the shipped kernels

`ours_new` against the shipped kernel on shapes whose shipped tile is not a power of two (`_before` columns: `pow2_best`, the PR's kernels, re-timed in the same session):

| subset | shapes | faster | parity | slower | geomean | median | geomean_before | slower_before |
|---|---|---|---|---|---|---|---|---|
| all non-power-of-two shipped | 603 | 407 (67%) | 110 (18%) | 86 (14%) | 1.070 | +6.9% | 1.033 | 174 (29%) |
| without memory-bound | 552 | 397 (72%) | 107 (19%) | 48 (9%) | 1.081 | +7.5% | 1.043 | 136 (25%) |
| without short (< 20 us) | 547 | 383 (70%) | 91 (17%) | 73 (13%) | 1.075 | +7.4% | 1.041 | 143 (26%) |
| without below one wave | 238 | 132 (55%) | 59 (25%) | 47 (20%) | 1.035 | +3.9% | 0.990 | 88 (37%) |
| without all three | 206 | 122 (59%) | 57 (28%) | 27 (13%) | 1.047 | +4.7% | 1.002 | 67 (33%) |
| shipped tile we run exactly | 381 | 272 (71%) | 55 (14%) | 54 (14%) | 1.080 | +7.5% | 1.042 | 111 (29%) |
| shipped tile we only approximate | 222 | 135 (61%) | 55 (25%) | 32 (14%) | 1.054 | +5.6% | 1.019 | 63 (28%) |

| subset | shapes | faster | slower | geomean | geomean_before |
|---|---|---|---|---|---|
| all | 1522 | 1117 (73%) | 141 (9%) | 1.111 | 1.087 |
| power-of-two shipped | 919 | 710 (77%) | 55 (6%) | 1.138 | 1.123 |
| non-power-of-two shipped | 603 | 407 (67%) | 86 (14%) | 1.070 | 1.033 |

By how memory-bound the shape is (non-power-of-two shipped):

| group | shapes | faster | parity | slower | geomean | median |
|---|---|---|---|---|---|---|
| compute | 480 | 353 (74%) | 89 (19%) | 38 (8%) | 1.087 | +8.0% |
| mixed | 72 | 44 (61%) | 18 (25%) | 10 (14%) | 1.042 | +3.9% |
| memory | 51 | 10 (20%) | 3 (6%) | 38 (75%) | 0.963 | -7.2% |
| all | 603 | 407 (67%) | 110 (18%) | 86 (14%) | 1.070 | +6.9% |

| group | shapes | faster | parity | slower | geomean | median |
|---|---|---|---|---|---|---|
| AI<64 | 18 | 2 (11%) | 3 (17%) | 13 (72%) | 0.944 | -7.6% |
| AI<128 | 27 | 7 (26%) | 4 (15%) | 16 (59%) | 0.966 | -4.0% |
| AI<256 | 119 | 69 (58%) | 25 (21%) | 25 (21%) | 1.042 | +3.9% |
| AI<512 | 187 | 125 (67%) | 37 (20%) | 25 (13%) | 1.077 | +7.1% |
| AI>=512 | 252 | 204 (81%) | 41 (16%) | 7 (3%) | 1.100 | +8.8% |
| all | 603 | 407 (67%) | 110 (18%) | 86 (14%) | 1.070 | +6.9% |

## Q3. Is power-of-two Stream-K needed when a tile can fill the waves?

Speedup = t_pow2_split / t_dp_any: above 1 a data-parallel tile (any tile, uneven included) beats the best power-of-two Stream-K split.

| group | shapes | faster | parity | slower | geomean | median |
|---|---|---|---|---|---|---|
| below_one_wave | 784 | 277 (35%) | 106 (14%) | 401 (51%) | 0.978 | -3.6% |
| one_wave_tail | 187 | 118 (63%) | 57 (30%) | 12 (6%) | 1.079 | +7.0% |
| two_plus_waves_tail | 288 | 110 (38%) | 133 (46%) | 45 (16%) | 1.030 | +1.3% |
| no_tail | 83 | 68 (82%) | 12 (14%) | 3 (4%) | 1.234 | +16.0% |
| all | 1342 | 573 (43%) | 308 (23%) | 461 (34%) | 1.017 | +0.9% |

After at least one full wave, by tail fraction (at our previous tile) and K:

| sp | k | shapes | geomean | split_wins | dp_wins | verdict |
|---|---|---|---|---|---|---|
| S/P<=1/4 | K<=2048 | 13 | 1.187 | 0 | 10 |  |
| S/P<=1/4 | K<=4096 | 31 | 1.034 | 2 | 11 |  |
| S/P<=1/4 | K<=8192 | 66 | 0.989 | 16 | 7 | Stream-K needed |
| S/P<=1/4 | K>8192 | 13 | 0.931 | 10 | 0 | Stream-K needed |
| S/P<=1/2 | K<=1024 | 7 | 1.134 | 1 | 4 |  |
| S/P<=1/2 | K<=2048 | 15 | 1.028 | 0 | 4 |  |
| S/P<=1/2 | K<=4096 | 37 | 1.019 | 4 | 10 |  |
| S/P<=1/2 | K<=8192 | 53 | 0.984 | 18 | 2 | Stream-K needed |
| S/P<=1/2 | K>8192 | 12 | 0.966 | 6 | 0 | Stream-K needed |
| S/P<=3/4 | K<=1024 | 3 | 1.091 | 0 | 3 |  |
| S/P<=3/4 | K<=2048 | 11 | 1.140 | 0 | 11 |  |
| S/P<=3/4 | K<=4096 | 23 | 1.039 | 0 | 10 |  |
| S/P<=3/4 | K<=8192 | 47 | 1.096 | 0 | 36 |  |
| S/P<=3/4 | K>8192 | 4 | 1.005 | 0 | 0 |  |
| S/P<1 | K<=1024 | 5 | 1.192 | 0 | 5 |  |
| S/P<1 | K<=2048 | 15 | 1.203 | 0 | 15 |  |
| S/P<1 | K<=4096 | 105 | 1.089 | 0 | 88 |  |
| S/P<1 | K<=8192 | 15 | 1.061 | 0 | 12 |  |

`ours_new` over `pow2_best` (what the two methods add to the PR's kernels):

| group | shapes | faster | parity | slower | geomean | median |
|---|---|---|---|---|---|---|
| below_one_wave | 784 | 162 (21%) | 622 (79%) | 0 (0%) | 1.023 | +0.0% |
| one_wave_tail | 187 | 72 (39%) | 115 (61%) | 0 (0%) | 1.044 | +2.1% |
| two_plus_waves_tail | 288 | 64 (22%) | 224 (78%) | 0 (0%) | 1.022 | +0.0% |
| no_tail | 263 | 10 (4%) | 253 (96%) | 0 (0%) | 1.004 | +0.0% |
| all | 1522 | 308 (20%) | 1214 (80%) | 0 (0%) | 1.022 | +0.0% |

Winner of `ours_new` per regime:

| regime | pow2_best | subtile | uneven |
|---|---|---|---|
| below_one_wave | 559 | 0 | 225 |
| no_tail | 250 | 1 | 12 |
| one_wave_tail | 58 | 43 | 86 |
| two_plus_waves_tail | 168 | 56 | 64 |

## Below one wave: fill a fraction of the wave, then split (estimate)

400 shapes below one wave at our previous tile have an uneven tile giving 90-100% of a half or quarter wave. Estimated time of that tile split 2 or 4 ways (owner ending, efficiency measured at the previous tile) against the measured `ours_new`: geomean 0.960, estimated faster by >3% on 109, slower on 166. Needs an uneven Stream-K kernel to measure.

| M | N | K | est_tile | est_n | est_fill | est_eta | est_us | ours_new | est_over_new |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 18992 | 4096 | 96x160 | 2.0 | 0.93 | 0.655 | 30.278 | 37.14 | 1.227 |
| 12288 | 384 | 7168 | 192x192 | 2.0 | 1.0 | 0.715 | 68.065 | 81.93 | 1.204 |
| 61 | 19360 | 4096 | 96x160 | 2.0 | 0.945 | 0.646 | 30.768 | 36.99 | 1.202 |
| 1024 | 6144 | 4608 | 256x192 | 2.0 | 1.0 | 0.729 | 50.358 | 59.8 | 1.187 |
| 64 | 19360 | 4096 | 96x160 | 2.0 | 0.945 | 0.649 | 30.948 | 36.66 | 1.185 |
| 57 | 19360 | 4096 | 96x160 | 2.0 | 0.945 | 0.65 | 30.605 | 35.92 | 1.174 |
| 1024 | 6144 | 12288 | 256x192 | 2.0 | 1.0 | 0.805 | 116.72 | 136.86 | 1.173 |
| 304 | 6416 | 4096 | 128x160 | 2.0 | 0.961 | 0.738 | 25.554 | 29.81 | 1.167 |
| 16384 | 160 | 6144 | 128x160 | 2.0 | 1.0 | 0.712 | 51.03 | 58.58 | 1.148 |
| 272 | 6416 | 4096 | 128x160 | 2.0 | 0.961 | 0.721 | 25.354 | 29.06 | 1.146 |
| 16384 | 384 | 5120 | 256x192 | 2.0 | 1.0 | 0.672 | 62.418 | 71.22 | 1.141 |
| 62 | 19360 | 4096 | 96x160 | 2.0 | 0.945 | 0.632 | 31.942 | 36.44 | 1.141 |
| 121 | 19360 | 4096 | 128x160 | 2.0 | 0.945 | 0.654 | 36.991 | 42.19 | 1.141 |
| 512 | 12288 | 6144 | 256x192 | 2.0 | 1.0 | 0.775 | 68.196 | 77.75 | 1.14 |
| 999 | 6144 | 6144 | 256x192 | 2.0 | 1.0 | 0.754 | 64.896 | 73.92 | 1.139 |

