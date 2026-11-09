# Uneven tiles against sub-tile tails: pilot

152 shapes. Times from aiter's --run_config replay of each method's pick (picked in the tuning run). Speedup = t_other / t_ours; faster / slower beyond +-3%. HBM peak 5.48 TB/s (bw_peak.txt); memory-bound when the minimum traffic 2(MK + NK + MN) at the best time reaches 80% of it, mixed from 50%.

Missing replays per column: pow2_split 18, subtile 74

Methods: `uneven` = best non-power-of-two data-parallel tile; `subtile` = best (main tile, sub-tile tail); `pow2_split` = best power-of-two Stream-K split; `pow2_dp` = best power-of-two data-parallel; `pow2_best` = the best of those and our previous pick (the PR's kernels); `dp_any` = min(uneven, pow2_dp); `v32_any` = min(subtile, pow2_best); `ours_new` = the best of everything.

## Shapes by regime (at our previous tile) and K

| regime | K<=1024 | K<=2048 | K<=4096 | K<=8192 | K>8192 |
|---|---|---|---|---|---|
| below_one_wave | 5 | 11 | 25 | 33 | 3 |
| one_wave_tail | 1 | 2 | 6 | 9 | 1 |
| two_plus_waves_tail | 2 | 1 | 16 | 9 | 1 |
| no_tail | 5 | 5 | 5 | 9 | 3 |

## Q1. Uneven data-parallel against sub-tile tails

Pure methods, on the 78 shapes where both have a candidate (speedup = t_subtile / t_uneven, above 1 the uneven tile wins):

| group | shapes | faster | parity | slower | geomean | median |
|---|---|---|---|---|---|---|
| below_one_wave | 33 | 31 (94%) | 2 (6%) | 0 (0%) | 1.209 | +20.2% |
| one_wave_tail | 16 | 11 (69%) | 1 (6%) | 4 (25%) | 1.089 | +8.3% |
| two_plus_waves_tail | 26 | 15 (58%) | 4 (15%) | 7 (27%) | 1.133 | +8.4% |
| no_tail | 3 | 2 (67%) | 1 (33%) | 0 (0%) | 1.237 | +37.9% |
| all | 78 | 59 (76%) | 8 (10%) | 11 (14%) | 1.159 | +14.7% |

| group | shapes | faster | parity | slower | geomean | median |
|---|---|---|---|---|---|---|
| K<=1024 | 6 | 4 (67%) | 1 (17%) | 1 (17%) | 1.255 | +27.0% |
| K<=2048 | 6 | 5 (83%) | 0 (0%) | 1 (17%) | 1.133 | +15.5% |
| K<=4096 | 25 | 21 (84%) | 3 (12%) | 1 (4%) | 1.184 | +13.7% |
| K<=8192 | 36 | 25 (69%) | 4 (11%) | 7 (19%) | 1.131 | +14.9% |
| K>8192 | 5 | 4 (80%) | 0 (0%) | 1 (20%) | 1.162 | +15.9% |
| all | 78 | 59 (76%) | 8 (10%) | 11 (14%) | 1.159 | +14.7% |

After at least one full wave, by tail fraction S / P at our previous tile:

| group | shapes | faster | parity | slower | geomean | median |
|---|---|---|---|---|---|---|
| S/P<=1/4 | 13 | 4 (31%) | 2 (15%) | 7 (54%) | 1.003 | -3.4% |
| S/P<=1/2 | 11 | 4 (36%) | 3 (27%) | 4 (36%) | 1.038 | -2.0% |
| S/P<=3/4 | 11 | 11 (100%) | 0 (0%) | 0 (0%) | 1.174 | +15.6% |
| S/P<1 | 7 | 7 (100%) | 0 (0%) | 0 (0%) | 1.410 | +39.1% |
| all | 42 | 26 (62%) | 5 (12%) | 11 (26%) | 1.116 | +8.4% |

With fallbacks on every shape (speedup = t_v32_any / t_dp_any): each method may also use its power-of-two kernels.

| group | shapes | faster | parity | slower | geomean | median |
|---|---|---|---|---|---|---|
| below_one_wave | 77 | 15 (19%) | 13 (17%) | 49 (64%) | 0.927 | -5.7% |
| one_wave_tail | 19 | 9 (47%) | 5 (26%) | 5 (26%) | 1.028 | +0.0% |
| two_plus_waves_tail | 29 | 4 (14%) | 18 (62%) | 7 (24%) | 0.999 | +0.0% |
| no_tail | 27 | 1 (4%) | 26 (96%) | 0 (0%) | 0.998 | +0.0% |
| all | 152 | 29 (19%) | 62 (41%) | 61 (40%) | 0.965 | -0.6% |

Joint search of the sub-tile tail's main tile: the best sub-tile pick's main tile differs from our previous pick's tile on 55 of 78 shapes.

## Q2. Against the shipped kernels

`ours_new` against the shipped kernel on shapes whose shipped tile is not a power of two (`_before` columns: `pow2_best`, the PR's kernels, re-timed in the same session):

| subset | shapes | faster | parity | slower | geomean | median | geomean_before | slower_before |
|---|---|---|---|---|---|---|---|---|
| all non-power-of-two shipped | 60 | 36 (60%) | 9 (15%) | 15 (25%) | 1.048 | +5.2% | 1.015 | 22 (37%) |
| without memory-bound | 54 | 36 (67%) | 9 (17%) | 9 (17%) | 1.065 | +6.7% | 1.031 | 16 (30%) |
| without short (< 20 us) | 54 | 34 (63%) | 8 (15%) | 12 (22%) | 1.055 | +6.7% | 1.028 | 18 (33%) |
| without below one wave | 24 | 11 (46%) | 6 (25%) | 7 (29%) | 1.016 | +2.1% | 0.968 | 11 (46%) |
| without all three | 20 | 10 (50%) | 6 (30%) | 4 (20%) | 1.036 | +3.3% | 0.986 | 8 (40%) |
| shipped tile we run exactly | 39 | 25 (64%) | 4 (10%) | 10 (26%) | 1.045 | +5.0% | 1.010 | 15 (38%) |
| shipped tile we only approximate | 21 | 11 (52%) | 5 (24%) | 5 (24%) | 1.054 | +7.3% | 1.024 | 7 (33%) |

| subset | shapes | faster | slower | geomean | geomean_before |
|---|---|---|---|---|---|
| all | 152 | 97 (64%) | 23 (15%) | 1.086 | 1.063 |
| power-of-two shipped | 92 | 61 (66%) | 8 (9%) | 1.112 | 1.096 |
| non-power-of-two shipped | 60 | 36 (60%) | 15 (25%) | 1.048 | 1.015 |

By how memory-bound the shape is (non-power-of-two shipped):

| group | shapes | faster | parity | slower | geomean | median |
|---|---|---|---|---|---|---|
| compute | 48 | 34 (71%) | 6 (12%) | 8 (17%) | 1.075 | +8.4% |
| mixed | 6 | 2 (33%) | 3 (50%) | 1 (17%) | 0.994 | +0.1% |
| memory | 6 | 0 (0%) | 0 (0%) | 6 (100%) | 0.905 | -7.8% |
| all | 60 | 36 (60%) | 9 (15%) | 15 (25%) | 1.048 | +5.2% |

| group | shapes | faster | parity | slower | geomean | median |
|---|---|---|---|---|---|---|
| AI<64 | 4 | 0 (0%) | 0 (0%) | 4 (100%) | 0.899 | -7.7% |
| AI<128 | 2 | 1 (50%) | 1 (50%) | 0 (0%) | 1.016 | +1.7% |
| AI<256 | 14 | 6 (43%) | 2 (14%) | 6 (43%) | 0.993 | +0.5% |
| AI<512 | 17 | 10 (59%) | 3 (18%) | 4 (24%) | 1.062 | +6.0% |
| AI>=512 | 23 | 19 (83%) | 3 (13%) | 1 (4%) | 1.104 | +11.0% |
| all | 60 | 36 (60%) | 9 (15%) | 15 (25%) | 1.048 | +5.2% |

## Q3. Is power-of-two Stream-K needed when a tile can fill the waves?

Speedup = t_pow2_split / t_dp_any: above 1 a data-parallel tile (any tile, uneven included) beats the best power-of-two Stream-K split.

| group | shapes | faster | parity | slower | geomean | median |
|---|---|---|---|---|---|---|
| below_one_wave | 77 | 25 (32%) | 7 (9%) | 45 (58%) | 0.968 | -5.5% |
| one_wave_tail | 19 | 13 (68%) | 4 (21%) | 2 (11%) | 1.082 | +7.4% |
| two_plus_waves_tail | 29 | 10 (34%) | 13 (45%) | 6 (21%) | 1.031 | +1.6% |
| no_tail | 9 | 9 (100%) | 0 (0%) | 0 (0%) | 1.238 | +17.8% |
| all | 134 | 57 (43%) | 24 (18%) | 53 (40%) | 1.013 | +0.3% |

After at least one full wave, by tail fraction (at our previous tile) and K:

| sp | k | shapes | geomean | split_wins | dp_wins | verdict |
|---|---|---|---|---|---|---|
| S/P<=1/4 | K<=2048 | 1 | 1.213 | 0 | 1 |  |
| S/P<=1/4 | K<=4096 | 5 | 1.028 | 0 | 2 |  |
| S/P<=1/4 | K<=8192 | 6 | 0.990 | 3 | 1 | Stream-K needed |
| S/P<=1/4 | K>8192 | 1 | 0.949 | 1 | 0 | Stream-K needed |
| S/P<=1/2 | K<=1024 | 2 | 1.245 | 0 | 1 |  |
| S/P<=1/2 | K<=2048 | 1 | 1.045 | 0 | 1 |  |
| S/P<=1/2 | K<=4096 | 2 | 1.014 | 1 | 1 |  |
| S/P<=1/2 | K<=8192 | 5 | 0.976 | 2 | 0 | Stream-K needed |
| S/P<=1/2 | K>8192 | 1 | 0.963 | 1 | 0 | Stream-K needed |
| S/P<=3/4 | K<=2048 | 1 | 1.131 | 0 | 1 |  |
| S/P<=3/4 | K<=4096 | 3 | 1.049 | 0 | 2 |  |
| S/P<=3/4 | K<=8192 | 7 | 1.090 | 0 | 4 |  |
| S/P<1 | K<=1024 | 1 | 1.230 | 0 | 1 |  |
| S/P<1 | K<=4096 | 12 | 1.062 | 0 | 8 |  |

`ours_new` over `pow2_best` (what the two methods add to the PR's kernels):

| group | shapes | faster | parity | slower | geomean | median |
|---|---|---|---|---|---|---|
| below_one_wave | 77 | 15 (19%) | 62 (81%) | 0 (0%) | 1.021 | +0.0% |
| one_wave_tail | 19 | 12 (63%) | 7 (37%) | 0 (0%) | 1.058 | +4.1% |
| two_plus_waves_tail | 29 | 5 (17%) | 24 (83%) | 0 (0%) | 1.014 | +0.0% |
| no_tail | 27 | 2 (7%) | 25 (93%) | 0 (0%) | 1.006 | +0.0% |
| all | 152 | 34 (22%) | 118 (78%) | 0 (0%) | 1.021 | +0.0% |

Winner of `ours_new` per regime:

| regime | pow2_best | subtile | uneven |
|---|---|---|---|
| below_one_wave | 56 | 0 | 21 |
| no_tail | 23 | 1 | 3 |
| one_wave_tail | 5 | 5 | 9 |
| two_plus_waves_tail | 18 | 5 | 6 |

## Below one wave: fill a fraction of the wave, then split (estimate)

44 shapes below one wave at our previous tile have an uneven tile giving 90-100% of a half or quarter wave. Estimated time of that tile split 2 or 4 ways (owner ending, efficiency measured at the previous tile) against the measured `ours_new`: geomean 1.005, estimated faster by >3% on 18, slower on 18. Needs an uneven Stream-K kernel to measure.

| M | N | K | est_tile | est_n | est_fill | est_eta | est_us | ours_new | est_over_new |
|---|---|---|---|---|---|---|---|---|---|
| 64 | 18992 | 4096 | 64x160 | 2.0 | 0.93 | 0.655 | 24.43 | 36.01 | 1.474 |
| 92 | 19360 | 4096 | 96x160 | 2.0 | 0.945 | 0.62 | 33.009 | 40.42 | 1.225 |
| 256 | 2880 | 2048 | 64x96 | 2.0 | 0.938 | 0.611 | 13.186 | 15.28 | 1.159 |
| 768 | 4096 | 8192 | 192x128 | 2.0 | 1.0 | 0.839 | 49.857 | 57.32 | 1.15 |
| 116 | 19360 | 4096 | 128x160 | 2.0 | 0.945 | 0.626 | 38.07 | 43.49 | 1.142 |
| 14336 | 384 | 7168 | 224x192 | 2.0 | 1.0 | 0.68 | 76.423 | 86.82 | 1.136 |
| 128 | 16384 | 2048 | 128x128 | 2.0 | 1.0 | 0.611 | 19.191 | 21.6 | 1.126 |
| 152 | 6416 | 4096 | 64x160 | 2.0 | 0.961 | 0.689 | 22.461 | 25.17 | 1.121 |
| 1792 | 2048 | 4096 | 224x128 | 2.0 | 1.0 | 0.736 | 32.943 | 36.08 | 1.095 |
| 512 | 6144 | 12288 | 256x192 | 4.0 | 1.0 | 0.632 | 75.111 | 82.22 | 1.095 |
| 1536 | 384 | 7168 | 96x96 | 4.0 | 1.0 | 0.626 | 20.147 | 21.81 | 1.083 |
| 256 | 6144 | 4096 | 128x96 | 2.0 | 1.0 | 0.715 | 23.117 | 24.94 | 1.079 |
| 512 | 12800 | 5120 | 256x224 | 2.0 | 0.906 | 0.734 | 64.745 | 69.74 | 1.077 |
| 2048 | 2560 | 6144 | 256x160 | 2.0 | 1.0 | 0.794 | 58.473 | 62.67 | 1.072 |
| 2048 | 2304 | 6144 | 192x224 | 2.0 | 0.945 | 0.803 | 57.207 | 60.79 | 1.063 |

