# Calibration: grid_bench.py against aiter's tuner (--run_config), same session

## The same kernel on the same shape (harness / aiter)

| Method | Pairs | Median | Geomean | 5% | 95% |
|---|---|---|---|---|---|
| base | 60 | 1.0083 | 1.0083 | 0.970 | 1.043 |
| dp | 60 | 1.0031 | 1.0042 | 0.969 | 1.042 |
| split | 60 | 1.0048 | 1.0071 | 0.964 | 1.057 |
| subtile | 58 | 1.0018 | 1.0022 | 0.958 | 1.042 |
| uneven | 60 | 1.0070 | 1.0083 | 0.961 | 1.061 |
| all | 298 | 1.0044 | 1.0061 | 0.963 | 1.050 |

## Winning method

60 shapes. Same winner, or the harness's winner within 1% of the best in aiter's replays: 55 (92%). Identical winner: 47.

Each method's best (harness) over aiter's replay of the main run's pick, geomean: uneven+base 1.0061, subtile+base 1.0032, split+base 1.0060.
