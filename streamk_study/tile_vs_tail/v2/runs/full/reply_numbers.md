# Reply numbers (491 shapes, M > 128)

K <= 1024: 21 shapes; K > 8192: 31; base pick 256x256: 459

## Persistent no-split plan vs whole tiles, both 256x256 (tuning profile)

| K | persistent 256x256 vs whole 256x256 | n |
|---|---|---|
| K <= 1024 | 1.061 | 21 |
| 1024 < K <= 2048 | 1.019 | 66 |
| K > 2048 | 1.005 | 404 |
| 2048 < K <= 4096 | 1.007 | 169 |
| 4096 < K <= 8192 | 1.004 | 204 |
| K > 8192 | 1.003 | 31 |

## Section 2 rules by region

| rule | all (491) | S<=P/2, K<=2048 (47) | of which S<=P/4 (17) | S<=P/2, 2048<K<=4096 (56) | S<=P/2, K>4096 (156) | S>P/2 (232) |
|---|---|---|---|---|---|---|
| finer tiles everywhere | 1.061 | 1.062 | 1.091 | 1.062 | 1.087 | 1.042 |
| sub-tile tail everywhere | 1.048 | 1.089 | 1.138 | 1.060 | 1.106 | 1.000 |
| Stream-K everywhere | 1.048 | 1.045 | 1.062 | 1.048 | 1.120 | 1.003 |
| finer, Stream-K K>4096 S<=P/2 | 1.071 | 1.062 | 1.091 | 1.062 | 1.120 | 1.042 |
| + sub-tile K<=2048 S<=P/2 | 1.073 | 1.089 | 1.138 | 1.062 | 1.120 | 1.042 |
| best per shape | 1.078 | 1.098 | 1.140 | 1.082 | 1.126 | 1.043 |

sub-tile vs finer, S<=P/2 K<=2048: 1.0257; S<=P/4 K<=2048: 1.0430; 2048<K<=4096 S<=P/2 finer vs sub: 1.0020

## Figure 3 cells

| fill | K | n | finer tiles | sub-tile tail | Stream-K | winner | lead |
|---|---|---|---|---|---|---|---|
| S/P<=1/4 | K<=1024 | 3 | 1.129 | 1.173 | 1.023 | sub-tile tail | 3.9% |
| S/P<=1/4 | K<=2048 | 14 | 1.083 | 1.130 | 1.070 | sub-tile tail | 4.4% |
| S/P<=1/4 | K<=4096 | 25 | 1.095 | 1.081 | 1.065 | finer tiles | 1.2% |
| S/P<=1/4 | K<=8192 | 73 | 1.126 | 1.149 | 1.156 | Stream-K | 0.6% |
| S/P<=1/4 | K>8192 | 13 | 1.152 | 1.183 | 1.237 | Stream-K | 4.6% |
| S/P<=1/2 | K<=1024 | 9 | 1.058 | 1.076 | 1.032 | sub-tile tail | 1.8% |
| S/P<=1/2 | K<=2048 | 21 | 1.040 | 1.056 | 1.038 | sub-tile tail | 1.5% |
| S/P<=1/2 | K<=4096 | 31 | 1.036 | 1.043 | 1.034 | sub-tile tail | 0.6% |
| S/P<=1/2 | K<=8192 | 57 | 1.038 | 1.050 | 1.061 | Stream-K | 1.0% |
| S/P<=1/2 | K>8192 | 13 | 1.035 | 1.045 | 1.076 | Stream-K | 2.9% |
| S/P<=3/4 | K<=1024 | 4 | 1.061 | 1.006 | 1.015 | finer tiles | 4.5% |
| S/P<=3/4 | K<=2048 | 15 | 1.015 | 1.000 | 1.001 | finer tiles | 1.4% |
| S/P<=3/4 | K<=4096 | 21 | 1.015 | 1.000 | 1.001 | finer tiles | 1.5% |
| S/P<=3/4 | K<=8192 | 51 | 1.086 | 1.000 | 1.006 | finer tiles | 8.0% |
| S/P<=3/4 | K>8192 | 5 | 1.041 | 1.000 | 1.031 | finer tiles | 0.9% |
| S/P<1 | K<=1024 | 5 | 1.009 | 1.000 | 1.000 | finer tiles | 0.9% |
| S/P<1 | K<=2048 | 16 | 1.001 | 1.000 | 1.000 | finer tiles | 0.1% |
| S/P<1 | K<=4096 | 92 | 1.048 | 1.000 | 1.001 | finer tiles | 4.7% |
| S/P<1 | K<=8192 | 23 | 1.000 | 1.000 | 1.000 | finer tiles | 0.0% |

## Persistent loop vs split, S<=P/2

| K | persistent vs base | split vs persistent | persistent >3% faster than base | n |
|---|---|---|---|---|
| K<=1024 | 1.060 | 0.972 | 67% | 12 |
| K<=2048 | 1.025 | 1.025 | 37% | 35 |
| K<=4096 | 1.008 | 1.039 | 4% | 56 |
| K<=8192 | 1.006 | 1.107 | 0% | 130 |
| K>8192 | 1.005 | 1.148 | 0% | 26 |

## hipBLASLt by region (t_hipBLASLt / t_ours)

| region | n | base | finer | sub-tile | Stream-K |
|---|---|---|---|---|---|
| S<=P/2, K<=4096 | 103 | 0.986 | 1.047 | 1.058 | 1.032 |
| S<=P/2, K>4096 | 156 | 0.929 | 1.010 | 1.027 | 1.040 |
| S>P/2, K<=4096 | 153 | 1.013 | 1.048 | 1.014 | 1.015 |
| S>P/2, K>4096 | 79 | 0.993 | 1.050 | 0.993 | 0.999 |

S<=P/2, K>4096: hipBLASLt tile > 256x256 on 134 of 156; median useful share 0.95 (256x256: 0.77); median intensity 135

Fractional-grid shapes: 512x201088x2880 (S=36, 1347/245); 4096x6144x6144 (S=128, 384/256); 4096x6144x18432 (S=128, 384/256); 16384x6656x2560 (S=128, 1664/256)

Main shapes in hipBLASLt's tuned tables (either key order): at most 31 of 491

## AITER, S<=P/2: finer over tail method by padding share

               sub     sk   n
pb    K                      
none  False  1.026  0.999  60
      True   1.020  1.037  88
<=2%  False  1.014  0.994  33
      True   1.023  1.034  41
2-10% False  0.989  0.912   4
      True   1.009  1.014  23
>10%  False  0.864  0.868   6
      True   0.938  0.929   4

pad > 10%: 10 shapes, finer vs best tail 1.111

