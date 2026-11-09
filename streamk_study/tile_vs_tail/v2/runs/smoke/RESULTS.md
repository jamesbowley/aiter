# Uneven tiles, sub-tile tails and Stream-K against gemm_a16w16.py: smoke

6 shapes (4 main, 1 full_waves, 1 below_one_wave). Times from aiter's --run_config replay of each method's pick (picked in the tuning run). Speedup = t_base / t_method, where base is gemm_a16w16.py at its best power-of-two tile; faster / slower beyond +-3%. The claim is relative to that kernel only.

Missing replays on the main set: {'shipped': 0, 'base': 0, 'uneven': 0, 'subtile': 0, 'subtile_rule': 0, 'streamk': 0, 'streamk_dp': 0}

## 1. Each method against the base

Main set (more than one wave of 256x256 tiles, partial last wave), alone:

| method | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| uneven | 4 | 3 (75%) | 0 (0%) | 1 (25%) | 1.072 | +7.1% | 0.962 | 1.197 |
| subtile | 4 | 2 (50%) | 0 (0%) | 2 (50%) | 0.850 | -4.3% | 0.517 | 1.111 |
| subtile, gfx1250's rule | 4 | 2 (50%) | 0 (0%) | 2 (50%) | 0.845 | -4.1% | 0.518 | 1.077 |
| streamk | 4 | 1 (25%) | 2 (50%) | 1 (25%) | 1.016 | +1.4% | 0.960 | 1.079 |
| uneven + base | 4 | 3 (75%) | 1 (25%) | 0 (0%) | 1.082 | +7.1% | 1.000 | 1.197 |
| subtile + base | 4 | 2 (50%) | 2 (50%) | 0 (0%) | 1.037 | +2.1% | 1.000 | 1.111 |
| streamk + base | 4 | 1 (25%) | 3 (75%) | 0 (0%) | 1.026 | +1.4% | 1.000 | 1.079 |
| best of all | 4 | 3 (75%) | 1 (25%) | 0 (0%) | 1.100 | +9.5% | 1.020 | 1.197 |

uneven + base, by K:

| group | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| K<=1024 | 1 | 1 (100%) | 0 (0%) | 0 (0%) | 1.084 | +8.4% | 1.084 | 1.084 |
| K<=2048 | 0 |  |  |  |  |  |  |  |
| K<=4096 | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 1.000 | +0.0% | 1.000 | 1.000 |
| K<=8192 | 1 | 1 (100%) | 0 (0%) | 0 (0%) | 1.197 | +19.7% | 1.197 | 1.197 |
| K>8192 | 1 | 1 (100%) | 0 (0%) | 0 (0%) | 1.058 | +5.8% | 1.058 | 1.058 |
| all | 4 | 3 (75%) | 1 (25%) | 0 (0%) | 1.082 | +7.1% | 1.000 | 1.197 |

uneven + base, by last-wave fill at 256x256:

| group | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| S/P<=1/4 | 1 | 1 (100%) | 0 (0%) | 0 (0%) | 1.084 | +8.4% | 1.084 | 1.084 |
| S/P<=1/2 | 1 | 1 (100%) | 0 (0%) | 0 (0%) | 1.058 | +5.8% | 1.058 | 1.058 |
| S/P<=3/4 | 1 | 1 (100%) | 0 (0%) | 0 (0%) | 1.197 | +19.7% | 1.197 | 1.197 |
| S/P<1 | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 1.000 | +0.0% | 1.000 | 1.000 |
| all | 4 | 3 (75%) | 1 (25%) | 0 (0%) | 1.082 | +7.1% | 1.000 | 1.197 |

subtile + base, by K:

| group | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| K<=1024 | 1 | 1 (100%) | 0 (0%) | 0 (0%) | 1.111 | +11.1% | 1.111 | 1.111 |
| K<=2048 | 0 |  |  |  |  |  |  |  |
| K<=4096 | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 1.000 | +0.0% | 1.000 | 1.000 |
| K<=8192 | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 1.000 | +0.0% | 1.000 | 1.000 |
| K>8192 | 1 | 1 (100%) | 0 (0%) | 0 (0%) | 1.041 | +4.1% | 1.041 | 1.041 |
| all | 4 | 2 (50%) | 2 (50%) | 0 (0%) | 1.037 | +2.1% | 1.000 | 1.111 |

subtile + base, by last-wave fill at 256x256:

| group | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| S/P<=1/4 | 1 | 1 (100%) | 0 (0%) | 0 (0%) | 1.111 | +11.1% | 1.111 | 1.111 |
| S/P<=1/2 | 1 | 1 (100%) | 0 (0%) | 0 (0%) | 1.041 | +4.1% | 1.041 | 1.041 |
| S/P<=3/4 | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 1.000 | +0.0% | 1.000 | 1.000 |
| S/P<1 | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 1.000 | +0.0% | 1.000 | 1.000 |
| all | 4 | 2 (50%) | 2 (50%) | 0 (0%) | 1.037 | +2.1% | 1.000 | 1.111 |

streamk + base, by K:

| group | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| K<=1024 | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 1.000 | +0.0% | 1.000 | 1.000 |
| K<=2048 | 0 |  |  |  |  |  |  |  |
| K<=4096 | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 1.020 | +2.0% | 1.020 | 1.020 |
| K<=8192 | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 1.008 | +0.8% | 1.008 | 1.008 |
| K>8192 | 1 | 1 (100%) | 0 (0%) | 0 (0%) | 1.079 | +7.9% | 1.079 | 1.079 |
| all | 4 | 1 (25%) | 3 (75%) | 0 (0%) | 1.026 | +1.4% | 1.000 | 1.079 |

streamk + base, by last-wave fill at 256x256:

| group | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| S/P<=1/4 | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 1.000 | +0.0% | 1.000 | 1.000 |
| S/P<=1/2 | 1 | 1 (100%) | 0 (0%) | 0 (0%) | 1.079 | +7.9% | 1.079 | 1.079 |
| S/P<=3/4 | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 1.008 | +0.8% | 1.008 | 1.008 |
| S/P<1 | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 1.020 | +2.0% | 1.020 | 1.020 |
| all | 4 | 1 (25%) | 3 (75%) | 0 (0%) | 1.026 | +1.4% | 1.000 | 1.079 |

## 2. Head to head (each with the base as fallback)

uneven over subtile (t_subtile / t_uneven), by K:

| group | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| K<=1024 | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 0.976 | -2.4% | 0.976 | 0.976 |
| K<=2048 | 0 |  |  |  |  |  |  |  |
| K<=4096 | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 1.000 | +0.0% | 1.000 | 1.000 |
| K<=8192 | 1 | 1 (100%) | 0 (0%) | 0 (0%) | 1.197 | +19.7% | 1.197 | 1.197 |
| K>8192 | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 1.016 | +1.6% | 1.016 | 1.016 |
| all | 4 | 1 (25%) | 3 (75%) | 0 (0%) | 1.044 | +0.8% | 0.976 | 1.197 |

uneven over subtile (t_subtile / t_uneven), by fill:

| group | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| S/P<=1/4 | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 0.976 | -2.4% | 0.976 | 0.976 |
| S/P<=1/2 | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 1.016 | +1.6% | 1.016 | 1.016 |
| S/P<=3/4 | 1 | 1 (100%) | 0 (0%) | 0 (0%) | 1.197 | +19.7% | 1.197 | 1.197 |
| S/P<1 | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 1.000 | +0.0% | 1.000 | 1.000 |
| all | 4 | 1 (25%) | 3 (75%) | 0 (0%) | 1.044 | +0.8% | 0.976 | 1.197 |

uneven over streamk (t_streamk / t_uneven), by K:

| group | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| K<=1024 | 1 | 1 (100%) | 0 (0%) | 0 (0%) | 1.084 | +8.4% | 1.084 | 1.084 |
| K<=2048 | 0 |  |  |  |  |  |  |  |
| K<=4096 | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 0.981 | -1.9% | 0.981 | 0.981 |
| K<=8192 | 1 | 1 (100%) | 0 (0%) | 0 (0%) | 1.187 | +18.7% | 1.187 | 1.187 |
| K>8192 | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 0.980 | -2.0% | 0.980 | 0.980 |
| all | 4 | 2 (50%) | 2 (50%) | 0 (0%) | 1.055 | +3.2% | 0.980 | 1.187 |

uneven over streamk (t_streamk / t_uneven), by fill:

| group | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| S/P<=1/4 | 1 | 1 (100%) | 0 (0%) | 0 (0%) | 1.084 | +8.4% | 1.084 | 1.084 |
| S/P<=1/2 | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 0.980 | -2.0% | 0.980 | 0.980 |
| S/P<=3/4 | 1 | 1 (100%) | 0 (0%) | 0 (0%) | 1.187 | +18.7% | 1.187 | 1.187 |
| S/P<1 | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 0.981 | -1.9% | 0.981 | 0.981 |
| all | 4 | 2 (50%) | 2 (50%) | 0 (0%) | 1.055 | +3.2% | 0.980 | 1.187 |

subtile over streamk (t_streamk / t_subtile), by K:

| group | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| K<=1024 | 1 | 1 (100%) | 0 (0%) | 0 (0%) | 1.111 | +11.1% | 1.111 | 1.111 |
| K<=2048 | 0 |  |  |  |  |  |  |  |
| K<=4096 | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 0.981 | -1.9% | 0.981 | 0.981 |
| K<=8192 | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 0.992 | -0.8% | 0.992 | 0.992 |
| K>8192 | 1 | 0 (0%) | 0 (0%) | 1 (100%) | 0.965 | -3.5% | 0.965 | 0.965 |
| all | 4 | 1 (25%) | 2 (50%) | 1 (25%) | 1.011 | -1.4% | 0.965 | 1.111 |

subtile over streamk (t_streamk / t_subtile), by fill:

| group | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| S/P<=1/4 | 1 | 1 (100%) | 0 (0%) | 0 (0%) | 1.111 | +11.1% | 1.111 | 1.111 |
| S/P<=1/2 | 1 | 0 (0%) | 0 (0%) | 1 (100%) | 0.965 | -3.5% | 0.965 | 0.965 |
| S/P<=3/4 | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 0.992 | -0.8% | 0.992 | 0.992 |
| S/P<1 | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 0.981 | -1.9% | 0.981 | 0.981 |
| all | 4 | 1 (25%) | 2 (50%) | 1 (25%) | 1.011 | -1.4% | 0.965 | 1.111 |

Fastest method per shape (no threshold), main set, by K:

| k_bin | streamk_any | subtile | uneven |
|---|---|---|---|
| K<=1024 | 0 | 1 | 0 |
| K<=2048 | 0 | 0 | 0 |
| K<=4096 | 1 | 0 | 0 |
| K<=8192 | 0 | 0 | 1 |
| K>8192 | 1 | 0 | 0 |

By fill:

| fill_bin | streamk_any | subtile | uneven |
|---|---|---|---|
| S/P<=1/4 | 0 | 1 | 0 |
| S/P<=1/2 | 1 | 0 | 0 |
| S/P<=3/4 | 0 | 0 | 1 |
| S/P<1 | 1 | 0 | 0 |

## 3. Controls

full_waves:

| method | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| uneven + base | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 1.000 | +0.0% | 1.000 | 1.000 |
| subtile + base | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 1.000 | +0.0% | 1.000 | 1.000 |
| streamk + base | 1 | 1 (100%) | 0 (0%) | 0 (0%) | 1.113 | +11.3% | 1.113 | 1.113 |

below_one_wave:

| method | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| uneven + base | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 1.000 | +0.0% | 1.000 | 1.000 |
| subtile + base | 1 | 0 (0%) | 1 (100%) | 0 (0%) | 1.002 | +0.2% | 1.002 | 1.002 |
| streamk + base | 1 | 1 (100%) | 0 (0%) | 0 (0%) | 3.452 | +245.2% | 3.452 | 3.452 |

## 4. Implementation checks

Stream-K's dp plan (no Stream-K) against the base: the Stream-K kernel is separate, persistent code, so its gains are comparable only if its dp plan matches the base.

| group | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| main | 4 | 0 (0%) | 3 (75%) | 1 (25%) | 0.997 | +0.5% | 0.960 | 1.020 |
| full_waves | 1 | 1 (100%) | 0 (0%) | 0 (0%) | 1.113 | +11.3% | 1.113 | 1.113 |
| below_one_wave | 1 | 1 (100%) | 0 (0%) | 0 (0%) | 1.503 | +50.3% | 1.503 | 1.503 |
| all | 6 | 2 (33%) | 3 (50%) | 1 (17%) | 1.088 | +1.4% | 0.960 | 1.503 |

## 5. Wave fill against padding (uneven)

eff = M N / (waves x 256 x tile area): the useful share of the tile slots, from the ragged-edge padding (pad) and the idle part of the last wave (fill). At the base's and the uneven method's picks, main set:

| pick | eff | pad | fill |
|---|---|---|---|
| base | 0.931 | 1.000 | 0.931 |
| uneven | 0.977 | 0.995 | 0.982 |

Of the 3 main shapes where uneven + base is faster by >3%, the larger improvement over the base's pick is in wave fill on 1 and in padding on 2.

## 6. Against the shipped aiter kernel (reference)

Speedup = t_shipped / t_ours, main set:

| method | shapes | faster | parity | slower | geomean | median | worst | best |
|---|---|---|---|---|---|---|---|---|
| base | 4 | 1 (25%) | 2 (50%) | 1 (25%) | 0.964 | -0.2% | 0.797 | 1.086 |
| best of all four | 4 | 3 (75%) | 0 (0%) | 1 (25%) | 1.060 | +9.3% | 0.955 | 1.108 |

0 main shapes reach half the HBM peak at the best time (memory-bound or mixed).

