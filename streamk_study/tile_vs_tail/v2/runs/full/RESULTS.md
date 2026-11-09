# Uneven tiles, sub-tile tails and Stream-K against gemm_a16w16.py: full

610 shapes (550 main, 30 full_waves, 30 below_one_wave). Times from aiter's --run_config replay of each method's pick (picked in the tuning run). Speedup = t_base / t_method, where base is gemm_a16w16.py at its best power-of-two tile; faster / slower beyond +-3%. The claim is relative to that kernel only.

Missing replays on the main set: {'shipped': 0, 'base': 0, 'uneven': 0, 'subtile': 75, 'subtile_rule': 75, 'streamk': 0, 'streamk_dp': 0}

## 1. Each method against the base

Main set (more than one wave of 256x256 tiles, partial last wave), alone:

| method | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| uneven | 550 | 286 (52%) | 166 (30%) | 98 (18%) | 1.043 | +3.3% | 0.831 | 1.332 |
| subtile | 475 | 214 (45%) | 75 (16%) | 186 (39%) | 0.948 | +1.5% | 0.514 | 1.373 |
| subtile, gfx1250's rule | 475 | 200 (42%) | 86 (18%) | 189 (40%) | 0.942 | +1.2% | 0.511 | 1.367 |
| streamk | 550 | 241 (44%) | 305 (55%) | 4 (1%) | 1.051 | +2.0% | 0.960 | 1.478 |
| uneven + base | 550 | 286 (52%) | 264 (48%) | 0 (0%) | 1.057 | +3.3% | 1.000 | 1.332 |
| subtile + base | 550 | 214 (39%) | 336 (61%) | 0 (0%) | 1.044 | +0.0% | 1.000 | 1.373 |
| streamk + base | 550 | 241 (44%) | 309 (56%) | 0 (0%) | 1.052 | +2.0% | 1.000 | 1.478 |
| best of all | 550 | 347 (63%) | 203 (37%) | 0 (0%) | 1.077 | +5.3% | 1.000 | 1.478 |

uneven + base, by K:

| group | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| K<=1024 | 21 | 12 (57%) | 9 (43%) | 0 (0%) | 1.056 | +4.5% | 1.000 | 1.182 |
| K<=2048 | 66 | 24 (36%) | 42 (64%) | 0 (0%) | 1.034 | +0.4% | 1.000 | 1.254 |
| K<=4096 | 212 | 92 (43%) | 120 (57%) | 0 (0%) | 1.044 | +1.7% | 1.000 | 1.317 |
| K<=8192 | 220 | 135 (61%) | 85 (39%) | 0 (0%) | 1.072 | +4.4% | 1.000 | 1.276 |
| K>8192 | 31 | 23 (74%) | 8 (26%) | 0 (0%) | 1.084 | +4.9% | 1.000 | 1.332 |
| all | 550 | 286 (52%) | 264 (48%) | 0 (0%) | 1.057 | +3.3% | 1.000 | 1.332 |

uneven + base, by last-wave fill at 256x256:

| group | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| S/P<=1/4 | 147 | 114 (78%) | 33 (22%) | 0 (0%) | 1.109 | +11.2% | 1.000 | 1.332 |
| S/P<=1/2 | 141 | 71 (50%) | 70 (50%) | 0 (0%) | 1.037 | +3.0% | 1.000 | 1.254 |
| S/P<=3/4 | 100 | 57 (57%) | 43 (43%) | 0 (0%) | 1.057 | +3.8% | 1.000 | 1.237 |
| S/P<1 | 162 | 44 (27%) | 118 (73%) | 0 (0%) | 1.028 | +0.0% | 1.000 | 1.261 |
| all | 550 | 286 (52%) | 264 (48%) | 0 (0%) | 1.057 | +3.3% | 1.000 | 1.332 |

subtile + base, by K:

| group | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| K<=1024 | 21 | 10 (48%) | 11 (52%) | 0 (0%) | 1.057 | +2.6% | 1.000 | 1.304 |
| K<=2048 | 66 | 28 (42%) | 38 (58%) | 0 (0%) | 1.044 | +1.0% | 1.000 | 1.336 |
| K<=4096 | 212 | 40 (19%) | 172 (81%) | 0 (0%) | 1.017 | +0.0% | 1.000 | 1.323 |
| K<=8192 | 220 | 114 (52%) | 106 (48%) | 0 (0%) | 1.062 | +3.3% | 1.000 | 1.372 |
| K>8192 | 31 | 22 (71%) | 9 (29%) | 0 (0%) | 1.093 | +5.3% | 1.000 | 1.373 |
| all | 550 | 214 (39%) | 336 (61%) | 0 (0%) | 1.044 | +0.0% | 1.000 | 1.373 |

subtile + base, by last-wave fill at 256x256:

| group | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| S/P<=1/4 | 147 | 115 (78%) | 32 (22%) | 0 (0%) | 1.120 | +11.0% | 1.000 | 1.373 |
| S/P<=1/2 | 141 | 96 (68%) | 45 (32%) | 0 (0%) | 1.047 | +4.8% | 1.000 | 1.145 |
| S/P<=3/4 | 100 | 3 (3%) | 97 (97%) | 0 (0%) | 1.003 | +0.0% | 1.000 | 1.069 |
| S/P<1 | 162 | 0 (0%) | 162 (100%) | 0 (0%) | 1.000 | +0.0% | 1.000 | 1.000 |
| all | 550 | 214 (39%) | 336 (61%) | 0 (0%) | 1.044 | +0.0% | 1.000 | 1.373 |

streamk + base, by K:

| group | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| K<=1024 | 21 | 17 (81%) | 4 (19%) | 0 (0%) | 1.077 | +7.3% | 1.000 | 1.168 |
| K<=2048 | 66 | 33 (50%) | 33 (50%) | 0 (0%) | 1.043 | +3.0% | 1.000 | 1.260 |
| K<=4096 | 212 | 47 (22%) | 165 (78%) | 0 (0%) | 1.022 | +1.1% | 1.000 | 1.291 |
| K<=8192 | 220 | 117 (53%) | 103 (47%) | 0 (0%) | 1.071 | +4.4% | 1.000 | 1.436 |
| K>8192 | 31 | 27 (87%) | 4 (13%) | 0 (0%) | 1.133 | +9.0% | 1.005 | 1.478 |
| all | 550 | 241 (44%) | 309 (56%) | 0 (0%) | 1.052 | +2.0% | 1.000 | 1.478 |

streamk + base, by last-wave fill at 256x256:

| group | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| S/P<=1/4 | 147 | 114 (78%) | 33 (22%) | 0 (0%) | 1.120 | +10.2% | 1.000 | 1.478 |
| S/P<=1/2 | 141 | 103 (73%) | 38 (27%) | 0 (0%) | 1.053 | +5.0% | 1.000 | 1.160 |
| S/P<=3/4 | 100 | 15 (15%) | 85 (85%) | 0 (0%) | 1.019 | +0.9% | 1.000 | 1.140 |
| S/P<1 | 162 | 9 (6%) | 153 (94%) | 0 (0%) | 1.013 | +0.8% | 1.000 | 1.168 |
| all | 550 | 241 (44%) | 309 (56%) | 0 (0%) | 1.052 | +2.0% | 1.000 | 1.478 |

## 2. Head to head (each with the base as fallback)

uneven over subtile (t_subtile / t_uneven), by K:

| group | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| K<=1024 | 21 | 7 (33%) | 10 (48%) | 4 (19%) | 0.999 | +0.0% | 0.887 | 1.077 |
| K<=2048 | 66 | 5 (8%) | 52 (79%) | 9 (14%) | 0.990 | +0.0% | 0.871 | 1.095 |
| K<=4096 | 212 | 68 (32%) | 132 (62%) | 12 (6%) | 1.027 | +0.0% | 0.913 | 1.269 |
| K<=8192 | 220 | 41 (19%) | 145 (66%) | 34 (15%) | 1.010 | +0.0% | 0.873 | 1.269 |
| K>8192 | 31 | 7 (23%) | 17 (55%) | 7 (23%) | 0.991 | -0.4% | 0.916 | 1.048 |
| all | 550 | 128 (23%) | 356 (65%) | 66 (12%) | 1.013 | +0.0% | 0.871 | 1.269 |

uneven over subtile (t_subtile / t_uneven), by fill:

| group | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| S/P<=1/4 | 147 | 18 (12%) | 85 (58%) | 44 (30%) | 0.989 | -0.9% | 0.871 | 1.269 |
| S/P<=1/2 | 141 | 11 (8%) | 108 (77%) | 22 (16%) | 0.990 | -1.3% | 0.919 | 1.166 |
| S/P<=3/4 | 100 | 55 (55%) | 45 (45%) | 0 (0%) | 1.054 | +3.5% | 1.000 | 1.237 |
| S/P<1 | 162 | 44 (27%) | 118 (73%) | 0 (0%) | 1.028 | +0.0% | 1.000 | 1.261 |
| all | 550 | 128 (23%) | 356 (65%) | 66 (12%) | 1.013 | +0.0% | 0.871 | 1.269 |

uneven over streamk (t_streamk / t_uneven), by K:

| group | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| K<=1024 | 21 | 6 (29%) | 5 (24%) | 10 (48%) | 0.980 | -2.9% | 0.881 | 1.111 |
| K<=2048 | 66 | 9 (14%) | 41 (62%) | 16 (24%) | 0.991 | -1.1% | 0.899 | 1.164 |
| K<=4096 | 212 | 62 (29%) | 143 (67%) | 7 (3%) | 1.022 | -0.0% | 0.932 | 1.250 |
| K<=8192 | 220 | 36 (16%) | 135 (61%) | 49 (22%) | 1.001 | -0.9% | 0.842 | 1.255 |
| K>8192 | 31 | 1 (3%) | 9 (29%) | 21 (68%) | 0.957 | -3.5% | 0.871 | 1.030 |
| all | 550 | 114 (21%) | 333 (61%) | 103 (19%) | 1.005 | -0.7% | 0.842 | 1.255 |

uneven over streamk (t_streamk / t_uneven), by fill:

| group | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| S/P<=1/4 | 147 | 25 (17%) | 77 (52%) | 45 (31%) | 0.990 | -1.2% | 0.842 | 1.255 |
| S/P<=1/2 | 141 | 11 (8%) | 84 (60%) | 46 (33%) | 0.985 | -2.0% | 0.883 | 1.164 |
| S/P<=3/4 | 100 | 39 (39%) | 57 (57%) | 4 (4%) | 1.038 | +1.2% | 0.927 | 1.231 |
| S/P<1 | 162 | 39 (24%) | 115 (71%) | 8 (5%) | 1.015 | -0.2% | 0.881 | 1.239 |
| all | 550 | 114 (21%) | 333 (61%) | 103 (19%) | 1.005 | -0.7% | 0.842 | 1.255 |

subtile over streamk (t_streamk / t_subtile), by K:

| group | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| K<=1024 | 21 | 7 (33%) | 5 (24%) | 9 (43%) | 0.981 | -1.7% | 0.856 | 1.253 |
| K<=2048 | 66 | 16 (24%) | 40 (61%) | 10 (15%) | 1.001 | -0.3% | 0.899 | 1.137 |
| K<=4096 | 212 | 8 (4%) | 194 (92%) | 10 (5%) | 0.995 | -0.7% | 0.927 | 1.079 |
| K<=8192 | 220 | 2 (1%) | 204 (93%) | 14 (6%) | 0.991 | -0.8% | 0.934 | 1.039 |
| K>8192 | 31 | 0 (0%) | 11 (35%) | 20 (65%) | 0.965 | -3.3% | 0.908 | 0.995 |
| all | 550 | 33 (6%) | 454 (83%) | 63 (11%) | 0.992 | -0.8% | 0.856 | 1.253 |

subtile over streamk (t_streamk / t_subtile), by fill:

| group | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| S/P<=1/4 | 147 | 22 (15%) | 101 (69%) | 24 (16%) | 1.001 | -0.5% | 0.908 | 1.253 |
| S/P<=1/2 | 141 | 11 (8%) | 114 (81%) | 16 (11%) | 0.994 | -0.8% | 0.900 | 1.068 |
| S/P<=3/4 | 100 | 0 (0%) | 86 (86%) | 14 (14%) | 0.984 | -0.9% | 0.878 | 1.017 |
| S/P<1 | 162 | 0 (0%) | 153 (94%) | 9 (6%) | 0.987 | -0.8% | 0.856 | 1.000 |
| all | 550 | 33 (6%) | 454 (83%) | 63 (11%) | 0.992 | -0.8% | 0.856 | 1.253 |

Fastest method per shape (no threshold), main set, by K:

| k_bin | base | streamk_any | subtile | uneven |
|---|---|---|---|---|
| K<=1024 | 1 | 12 | 4 | 4 |
| K<=2048 | 5 | 34 | 17 | 10 |
| K<=4096 | 9 | 89 | 24 | 90 |
| K<=8192 | 9 | 123 | 27 | 61 |
| K>8192 | 0 | 27 | 0 | 4 |

By fill:

| fill_bin | base | streamk_any | subtile | uneven |
|---|---|---|---|---|
| S/P<=1/4 | 0 | 71 | 41 | 35 |
| S/P<=1/2 | 1 | 93 | 31 | 16 |
| S/P<=3/4 | 4 | 36 | 0 | 60 |
| S/P<1 | 19 | 85 | 0 | 58 |

## 3. Controls

full_waves:

| method | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| uneven + base | 30 | 0 (0%) | 30 (100%) | 0 (0%) | 1.001 | +0.0% | 1.000 | 1.022 |
| subtile + base | 30 | 0 (0%) | 30 (100%) | 0 (0%) | 1.000 | +0.0% | 1.000 | 1.000 |
| streamk + base | 30 | 5 (17%) | 25 (83%) | 0 (0%) | 1.021 | +0.7% | 1.000 | 1.129 |

below_one_wave:

| method | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| uneven + base | 30 | 6 (20%) | 24 (80%) | 0 (0%) | 1.017 | +0.0% | 1.000 | 1.145 |
| subtile + base | 30 | 3 (10%) | 27 (90%) | 0 (0%) | 1.009 | +0.0% | 1.000 | 1.065 |
| streamk + base | 30 | 25 (83%) | 5 (17%) | 0 (0%) | 1.726 | +48.5% | 1.000 | 4.892 |

## 4. Implementation checks

Stream-K's dp plan (no Stream-K) against the base: the Stream-K kernel is separate, persistent code, so its gains are comparable only if its dp plan matches the base.

| group | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| main | 550 | 44 (8%) | 498 (91%) | 8 (1%) | 1.010 | +0.7% | 0.957 | 1.168 |
| full_waves | 30 | 5 (17%) | 25 (83%) | 0 (0%) | 1.020 | +0.7% | 0.981 | 1.129 |
| below_one_wave | 30 | 25 (83%) | 4 (13%) | 1 (3%) | 1.196 | +20.3% | 0.790 | 1.560 |
| all | 610 | 74 (12%) | 527 (86%) | 9 (1%) | 1.019 | +0.7% | 0.790 | 1.560 |

## 5. Wave fill against padding (uneven)

eff = M N / (waves x 256 x tile area): the useful share of the tile slots, from the ragged-edge padding (pad) and the idle part of the last wave (fill). At the base's and the uneven method's picks, main set:

| pick | eff | pad | fill |
|---|---|---|---|
| base | 0.863 | 0.996 | 0.901 |
| uneven | 0.952 | 0.996 | 0.982 |

Of the 286 main shapes where uneven + base is faster by >3%, the larger improvement over the base's pick is in wave fill on 210 and in padding on 76.

## 6. Against the shipped aiter kernel (reference)

Speedup = t_shipped / t_ours, main set:

| method | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| base | 550 | 233 (42%) | 152 (28%) | 165 (30%) | 1.025 | +2.2% | 0.672 | 6.259 |
| best of all four | 550 | 399 (73%) | 87 (16%) | 64 (12%) | 1.104 | +6.8% | 0.825 | 6.560 |

105 main shapes reach half the HBM peak at the best time (memory-bound or mixed).

