# Results

The study is complete; the terms and the three methods are as defined in the ticket. The study set out to find where each method helps on gfx950 and why. The main finding is that the answer has a simple structure. On a controlled grid of 912 shapes, three quantities decide which method wins: the last-wave fill, K and the padding. A rule based on them comes within about 0.3% of choosing the best method for each shape, and it holds on the 491 AITER shapes it was not fitted on. Each boundary corresponds to a mechanism, so the structure should outlast these particular implementations, even though most of the thresholds will move.

## Summary

**The mechanisms.** Each boundary in the results has a cause that comes from the problem rather than from the code:

- **Half a wave left (S/P = 1/2):** the sub-tile tail cannot split more than half a wave of leftover tiles (S·Q <= P with Q >= 2). Above this point, Stream-K's split costs more than it recovers.
- **K (sub-tile tail against Stream-K):** Stream-K's split has a fixed cost (a reduction and a workspace) that long K amortises. The sub-tile tail's cost hardly depends on K.
- **Small leftover (tail methods against finer tiles):** to fill the waves, a finer tile lowers the intensity of every tile, to fix a leftover that is only a few percent of the work. The tail methods keep the 256x256 main loop and pay only on the leftover tiles.
- **Padding (finer tiles against the tail methods):** the tail methods pay the padding in every 256x256 tile, whereas a finer tile that fits the ragged dimension removes it.

**The recommended rule.** With S the leftover tiles at 256x256 and P = 256:

- if S/P <= 1/2 and at most 10% of the 256x256 tile grid is padding, use the sub-tile tail for K <= 4096 and Stream-K for K > 4096;
- otherwise, use finer tiles.

| Geomean speedup over the base | Grid (912 shapes) | AITER shapes (491) |
|---|---|---|
| Best single method | 1.051-1.053 (all three within 0.2%) | 1.061 (finer tiles) |
| Recommended rule | **1.070** | **1.075** |
| Best method for each shape | 1.073 | 1.078 |

**What carries over.** The rule shows that the differences between the methods are simple, smooth functions of the shape. It does not show that the implementations are optimal: a method that was a few percent slower on every shape would move a boundary but leave the rule just as clean. The structure should carry over as the methods are optimised or combined, because it follows from the mechanisms above. Most of the thresholds will move:

- The half-wave limit of the sub-tile tail is structural and will not move.
- The K crossover will move down if Stream-K's split gets cheaper, and up if the sub-tile tail gets a persistent loop.
- The 10% padding gate depends on the tile menu (sides in multiples of 32, at most 256x256 of accumulators).
- Combining a finer tile with a tail method in one kernel would make the choice a joint one of tile and tail, which a cost model would serve better than thresholds.
- Shapes below one wave, or near the memory-bound region, involve other mechanisms and are outside this grid.

**Answers to the ticket's questions**

1. **How much each method recovers.** On the grid, the best method recovers 12-22% when the last wave is at most a quarter full and 6-11% when it is a quarter to half full, depending on K. With more than half a wave left and no padding, no method gains more than 1.5% (Table 1).
2. **Where each method stops helping.** Both tail methods stop at half a wave. The sub-tile tail cannot go further by construction (S·Q <= P). Beyond half a wave Stream-K gains at most about 1%, and only for K >= 8192. Stream-K's split breaks even at about K = 1024 and gains up to 16% at K = 16384. The sub-tile tail's gain hardly depends on K.
3. **Finer tiles with small S.** Yes, the lower intensity costs more than the idle CUs it saves. With at most half a wave left and no padding, finer tiles are 2-6% behind the better tail method. To fill the waves when the leftover is small, a finer tile must shrink a lot, and every tile then runs at lower MFMA efficiency. On 1024x18432x8192 (32 tiles past a full wave), the best finer tile is 256x160, and its main loop runs at 80% MFMA efficiency, against 93-95% for the 256x256 main loop that the tail methods keep.
4. **Single method or dispatch.** Dispatch clearly gives more, but the rule needs the padding share as well as S/P and K. Without the padding test the rule gets 1.061 on the grid instead of 1.070.
5. **Stream-K's persistent loop.** For K >= 4096, almost all of Stream-K's gain comes from the split (the loop is worth under 1%). At K <= 1024 the persistent loop alone is worth 4-6%, as much as or more than the split.
6. **hipBLASLt.** Our variants are competitive. On the AITER shapes, each one, tuned per shape, is 2-4% faster than hipBLASLt's heuristic kernel choice, and the best of the three is 5.3% faster. This does not mean our kernels are faster: on one sweep where hipBLASLt was tuned exhaustively, it was about 3% faster than our best method. hipBLASLt uses finer and larger tiles and almost never runs a Stream-K split, because its cost model counts a Stream-K kernel as whole waves.
7. **Impact on AITER.** AITER reuses a tile tuned at one M for every M down to half of it. Finer tiles are sensitive to this: reuse costs them 5.5% on average (up to 45%) against the best tile at each M, compared with 1.7% for power-of-two tiles. Finer tiles therefore need a tile selection that follows M. The tail methods are less exposed, because S is computed from the actual M at launch.

## The experiment

**Kernels.** Each method is tuned per shape and, as a tuner would, may fall back to the base's whole tiles.

- **Base:** `gemm_a16w16.py`, the gfx950 gluon compute-bound kernel (PR [#5897](https://github.com/ROCm/aiter/pull/5897)), with power-of-two tiles (sides 64, 128 or 256).
- **Finer tiles:** `gemm_a16w16_uneven.py`, the base with uneven quadrant halves: sides any multiple of 32 from 64 to 320, at most 256x256 of accumulators.
- **Sub-tile tail:** `gemm_a16w16_subtile.py`, the base with each of the S leftover tiles cut into Q sub-tiles of sides 64, 128 or 256, subject to S·Q <= P (gfx1250's rule).
- **Stream-K:** the splitting plans of `gemm_a16w16_streamk.py` (PR [#7391](https://github.com/ROCm/aiter/pull/7391)), a persistent kernel in which each CU runs one workgroup that loops over its tiles. Its data-parallel plan (persistent, no split) is used only to separate the effect of the loop from that of the split.

The finer-tile and sub-tile kernels are small diffs to `gemm_a16w16.py` (attached). With the new feature switched off, both compile to exactly the base's machine code (all 9 power-of-two tiles, with and without bias, ragged M, N and K), so every measured difference comes from the tail method.

**Grid (used to fit the rule).** The grid has 912 shapes in three parts:

- **Factorial (720 shapes, no padding):** every combination of K in {1024, 2048, 4096, 8192, 16384}, W in {1, 2, 4} full waves of 256x256 tiles, S in {0, 16, ..., 240} leftover tiles, and M in {512, 1024, 4096}, with N chosen to give exactly W x 256 + S tiles.
- **Ragged N (64 shapes, padding under 0.5%):** N - 72 instead of N, for M = 1024, W in {1, 2} and K in {2048, 8192}. This gives ragged N edges with almost no padding.
- **Ragged M (128 shapes, 9% to 41% padding):** M in {300, 450, 700, 1100}, for K in {2048, 8192}, W in {1, 2} and S/P <= 1/2. This is the part that tests padding.

The rule is fitted and scored on all 912 shapes. Because M is fixed and N sets the tile count, N is large: 4096 to 161,792 (the smallest at M = 4096 with one full wave). All factorial and ragged-N shapes are compute-bound: at least 338 FLOP per byte of minimum traffic, against a ridge point of about 310 for MI355X. The M = 300 shapes are borderline (260 to 290 FLOP per byte). The S = 0 controls tie, as expected.

**Test set.** The test set is every bf16 shape in AITER's tuned configs (`aiter/configs/bf16_tuned_gemm.csv` and the 26 `model_configs/*bf16_tuned_gemm*.csv` files) that meets three conditions. First, 256x256 tiles give more than one wave with a partial last wave. Second, the kernels support it (N and K multiples of 8, K >= 256). Third, M > 128: the 59 decode-like shapes with smaller M are excluded because the base never uses 256x256 tiles on them. That leaves 491 shapes. They come from real models but are not a designed experiment: K, W, fill and M move together, K is concentrated between 2048 and 8192, and some groups are dominated by near-identical shapes (for example, 18 shapes with N = 37888 and K = 5120). That is why the rule is fitted on the grid. Control sets with exactly full waves and with less than one wave (30 shapes each) tie, as expected.

**Timing.** The test set was tuned with AITER's tuner, and each method's pick was then timed again in a `--run_config` replay. The grid was timed with `grid_bench.py`, a harness that measures each candidate the same way as the tuner but does the per-shape setup only once (6 to 29 seconds per shape, against about 33 for the tuner). It uses the same inputs, rotates up to 101 input copies so that no iteration finds its inputs in L2 or MALL, and uses AITER's `run_iters_rotate` and `get_trace_perf` and the same correctness check. Calibrated against AITER's replays on 60 AITER shapes in the same session, it reads 0.2-0.8% higher at the median for every method, so there is no bias between methods. It picks the same winning method, or one within 1% of it, on 55 of the 60 shapes. The grid was timed twice, in two different random orders. Each method's best time differs between the two passes by 0.6% at the median. All comparisons are within one session.

**Reading the tables.** Speedups are geomeans of t_base / t_method, where the base runs at its best power-of-two tile for each shape. S, S/P and the padding share are always counted at 256x256 tiles. The padding share is the fraction of the 256x256 tile grid that lies outside the M x N output.

## Results on the grid

### Where each method helps

**Table 1.** Factorial part of the grid (no padding), speedup over the base as finer tiles / sub-tile tail / Stream-K. Each cell covers 36 shapes for S/P <= 1/2 and 63 shapes for S/P > 1/2. The best method in each cell is in bold.

| K | S/P <= 1/4 | 1/4 < S/P <= 1/2 | S/P > 1/2 |
|---|---|---|---|
| 1024 | 1.060 / **1.120** / 1.072 | 1.034 / **1.059** / 1.035 | **1.005** / 1.000 / 1.000 |
| 2048 | 1.074 / **1.121** / 1.093 | 1.037 / **1.056** / 1.036 | **1.006** / 1.000 / 1.000 |
| 4096 | 1.094 / **1.131** / 1.121 | 1.040 / 1.058 / **1.059** | **1.007** / 1.000 / 1.001 |
| 8192 | 1.118 / 1.148 / **1.165** | 1.054 / 1.069 / **1.088** | **1.009** / 1.000 / 1.007 |
| 16384 | 1.149 / 1.169 / **1.219** | 1.066 / 1.065 / **1.106** | 1.010 / 1.000 / **1.012** |

- **Sub-tile tail:** its gain is about the same at every K. It is the best method for K <= 4096 (at K = 4096 it ties with Stream-K from a quarter to half full).
- **Stream-K:** its gain grows with K, as expected from the fixed cost of the split. It is the best method for K >= 8192.
- **Finer tiles:** with at most half a wave left, they are 2-6% behind the better tail method at every K.
- **More than half a wave left:** with no padding there is little to recover. Finer tiles help only just above half a wave (Figure 1).

<img width="2250" height="2100" alt="Image" src="https://github.com/user-attachments/assets/10bd2e61-09b8-4fbf-a207-3f3f6c00ffbd" />

**Figure 1.** Best method for every shape in the factorial part of the grid, faceted by W and M. These shapes have no padding, so the rule's padding test never applies and the rule reduces to its S/P threshold. The padded shapes (the ragged-M part) are shown in Figure 2. The black line marks that threshold at S/P = 1/2. Below it the rule uses a tail method (the sub-tile tail for K <= 4096, Stream-K for K >= 8192), and above it the rule uses finer tiles. Solid cells: one method is at least 1% ahead of the next. Hatched cells: the methods are within 1% of each other, and the hatch colour shows the fastest. Grey cells: no method gains 1% over whole 256x256 tiles. The number in each cell is the best method's gain over whole 256x256 tiles, in percent.

### Padding decides between finer tiles and the tail methods

The tail methods keep 256x256 tiles and pay for the padding in every tile, whereas a finer tile can choose a side that fits the ragged dimension. On the ragged-M part of the grid (128 shapes, all with S/P <= 1/2), finer tiles beat the better tail method by 21% at M = 300 (41% padding, where two rows of 160 fit) and by 5.7% at M = 1100 (14% padding, five rows of 224). They lose by 1.3% at M = 450 (12% padding): there the best finer tile (160x288) still leaves 6% padding, so it removes only half, and it costs intensity. The tail methods pay the padding at 256x256, while finer tiles pay their own padding, so what decides between them is the difference: the padding a finer tile removes. The padding share at 256x256 is an upper bound on that difference and usually close to it, but not always (M = 450). The rule uses the padding share at 256x256 because it depends only on M and N, whereas the padding removed depends on which finer tile is chosen.

<img width="2250" height="930" alt="Image" src="https://github.com/user-attachments/assets/47e5ae15-a762-4830-940a-e11744c8de1b" />

**Figure 2.** Every shape with at most half a wave left. The vertical axis is the better tail method's time divided by the finer tiles' time, so values above 1 mean finer tiles are faster. The left panel plots this against the padding share at 256x256, which is the rule's input. The right panel plots it against the padding removed: the padding share at 256x256 minus the padding share of the fastest finer tile's own tile grid. For example, at M = 300, 256x256 tiles cover 512 rows (41% padding), while 160-row tiles cover 320 rows (6%), so the finer tile removes 35 points. At M = 450, 256x256 tiles cover 512 rows (12%), while 160-row tiles cover 480 rows (6%), so it removes only 6 points. Negative values mean the fastest finer tile has more padding than 256x256 (it was chosen for wave fill, not padding). Grey points are grid shapes with no padding, where the tail methods win. The black line is the geomean of each padding band. On the left, the M = 450 shapes (downward triangles) sit below 1 even though they are past the 10% gate. On the right, they sit with the other shapes that remove about 6 points, where the band geomean is close to 1. The finer tiles' advantage therefore follows the padding removed more closely than the padding share at 256x256.

### Fitting the rule

For each candidate padding gate, finer tiles are used above the gate. For each K, the tail method and its S/P threshold were chosen to minimise the geomean loss against the best method for each shape. The whole rule was then scored on all 912 grid shapes:

| Finer tiles when the padding share is above | Geomean loss against the best method per shape | Shapes more than 3% slower than their best |
|---|---|---|
| no gate | 0.95% | 9.0% |
| 2% to 7.5% | 0.27% | 1.6% |
| **10%** | **0.19%** | **0.7%** |
| 15% to 30% | 0.34% | 2.9% |

Under the 10% gate, the fit chooses the sub-tile tail for K <= 4096 and Stream-K for K >= 8192 in all 200 bootstrap resamples of the shapes. The sub-tile tail's threshold is always S/P = 1/2, its structural limit. Stream-K's fitted threshold is 1/2 or higher (up to 15/16 at K = 16384 and with four full waves). Capping it at 1/2 costs 0.02% on the grid, so the rule caps it. Fitting each W and each M separately gives the same split, except with four full waves at K = 2048 and 4096, where Stream-K edges out the sub-tile tail.

### Why the tail methods win when the leftover is small

The cost of finer tiles shows up on 1024x16896x7168. At 256x256 this shape has 264 tiles: 256 run in the first wave and 8 in the second, which is 97% idle. No tile of at most 256x256 fits the GEMM in one wave, so filling two waves needs tiles of about half the area. In the table below, the useful share is M·N / (waves x P x tile area), and the times are from tuning.

| Tile | Tiles | Last wave | Useful share | Intensity | Time (us) |
|---|---|---|---|---|---|
| 256x256 | 264 | 8 | 0.52 | 128 | 266.2 |
| 256x224 | 304 | 48 | 0.59 | 119 | 247.1 |
| 256x192 | 352 | 96 | 0.69 | 110 | 226.2 |
| 256x160 (best finer tile) | 424 | 168 | 0.82 | 98 | 213.6 |
| 128x288 | 472 | 216 | 0.92 | 89 | 224.1 |

The sub-tile tail (193.3 us) and Stream-K (181.8 us) are faster because they keep 256x256 for the first 256 tiles and change only the last 8, which are 3% of the work.

Thread traces (`rocprofv3 --att`, one CU) of each method's winning kernel on 1024x18432x8192 (288 tiles, 32 past a full wave) show the mechanism:

| Kernel | MFMA issue efficiency in the main loop | Main loop's share of instruction time |
|---|---|---|
| Base, 256x256 | 93.0% | 97% |
| Stream-K, 256x256 | 95.3% | 73% (the rest is its partial tile and fix-up) |
| Sub-tile tail, 256x256 main tile | 94.4% | 65% (the rest includes the sub-tiles) |
| Finer tiles, 256x160 | 80.4% | 94% |

The tail methods run the base's main loop at the same efficiency and pay only on the leftover tiles. Finer tiles pay for their lower intensity in every tile, and the cost depends on how far the tile has to shrink. Whole-kernel counters (MFMA busy cycles over busy CU cycles) on 8 grid shapes with K >= 8192 show this. With 32 tiles past a full wave, the best finer tile is 256x160, which runs 10-12% below the base at 256x256. With 192 tiles past a full wave, it is 256x224, which runs only 2-3% below.

### Stream-K: the split against the persistent loop

Both effects are measured within the Stream-K kernel. Its data-parallel plan runs the same persistent loop without a split, so "persistent loop vs base" is t_base / t_data-parallel, and "split vs persistent loop" is t_data-parallel / t_split (each with whole-tile fallback). This assumes the loop is worth the same with and without a split. With at most half a wave left, on the grid:

| K | Persistent loop vs base | Split vs persistent loop |
|---|---|---|
| 1024 | 1.037 | 1.016 |
| 2048 | 1.013 | 1.029 |
| 4096 | 1.007 | 1.082 |
| 8192 | 1.004 | 1.112 |
| 16384 | 1.004 | 1.156 |

The grid starts at K = 1024. Below that, the AITER shapes show the trend continuing: on the 11 with K < 1024 (K from 320 to 768) and at most half a wave left, the loop is worth 6.3% and the split is 3.1% slower than the loop alone. The split therefore breaks even near K = 1024, and at short K most of Stream-K's gain comes from the persistent loop, which the other methods could also use.

## Test on the AITER shapes

No threshold was fitted on these shapes. The table below uses the speedup over the base from AITER's `--run_config` replays, with the rule's choice for each region in brackets:

| Region | Shapes | Finer tiles | Sub-tile tail | Stream-K | Rule | Best per shape |
|---|---|---|---|---|---|---|
| S/P <= 1/2, padding <= 10%, K <= 4096 (sub-tile tail) | 97 | 1.053 | **1.075** | 1.046 | 1.075 | 1.082 |
| S/P <= 1/2, padding <= 10%, K > 4096 (Stream-K) | 152 | 1.084 | 1.105 | **1.120** | 1.120 | 1.123 |
| S/P <= 1/2, padding > 10% (finer tiles) | 10 | **1.220** | 1.090 | 1.088 | 1.220 | 1.220 |
| S/P > 1/2, padding <= 10% (finer tiles) | 167 | **1.011** | 1.000 | 1.003 | 1.011 | 1.012 |
| S/P > 1/2, padding > 10% (finer tiles) | 65 | **1.126** | 1.000 | 1.002 | 1.126 | 1.126 |
| All | 491 | 1.061 | 1.048 | 1.048 | **1.075** | 1.078 |

In every region, the rule chooses the best of the three methods. The AITER shapes differ from the grid mainly in padding: 75 of the 491 have more than 10% (M from 144 to 1600), and those are where finer tiles gain most. With more than half a wave left and at most 10% padding, the best method gains only 1.2% over the base, close to the 0.5-1.5% on the grid. The large gain of finer tiles above half a wave on the AITER shapes comes from the 65 shapes with more than 10% padding.

Each rule on both sets, given as geomean speedup over the base / share of shapes more than 3% slower than their best method:

| Rule | Grid (912) | AITER shapes (491) |
|---|---|---|
| Finer tiles everywhere | 1.051 / 29% | 1.061 / 23% |
| Sub-tile tail everywhere | 1.053 / 18% | 1.048 / 29% |
| Stream-K everywhere | 1.051 / 21% | 1.048 / 27% |
| Recommended rule without the padding gate | 1.061 / 6% | 1.073 / 4% |
| **Recommended rule** | **1.070 / 1%** | **1.075 / 2%** |
| Best method per shape | 1.073 / 0% | 1.078 / 0% |

To check that the grid result does not depend on the harness, each method's pick on 135 grid shapes (covering every K, W and M) was replayed through AITER's tuner. The rule gets 1.089, against 1.093 for the best method per shape. A fresh replay of the AITER shapes in the same session gives 1.077 against 1.081. Weighted by run time, the rule gets 1.034 on the AITER shapes, against 1.037 for the best method per shape.

## Comparison with hipBLASLt

To check that the variants are not weak implementations, they were compared with torch `F.linear`, which runs hipBLASLt's own kernel choice, timed in the same session. hipBLASLt is an optimized library that also uses non-power-of-two tiles. Values are t_hipBLASLt / t_ours, so above 1 means ours is faster:

| Ours | All (491) | hipBLASLt tile at most 256x256 (155) | hipBLASLt tile larger than 256x256 (336) |
|---|---|---|---|
| Base (power-of-two tiles) | 0.977 | 1.008 | 0.963 |
| Finer tiles | 1.036 | 1.066 | 1.022 |
| Sub-tile tail | 1.024 | 1.040 | 1.016 |
| Stream-K | 1.024 | 1.038 | 1.017 |
| Best of the three | 1.053 | 1.073 | 1.044 |

These numbers compare our variants, each tuned per shape over its full menu, with hipBLASLt's heuristic choice: for all but at most 31 of the shapes, hipBLASLt picks its kernel with a cost model, not a tuning run. They show that our tuned variants beat its default selection, not that our kernels are faster than its kernels. One sweep tested this directly: M = 1024, K = 7168 and N from 16384 to 32768, with hipBLASLt tuned exhaustively (every solution under three Stream-K grid policies, with `hipblaslt-bench`). Tuned hipBLASLt was 3.3% faster than our best method on the 15 shapes with a leftover, and 4-5% faster at exactly one and two full waves. The two sides were timed with different harnesses (`hipblaslt-bench` and `do_bench`), so the absolute gap is only indicative. Measured against its own full-wave time, tuned hipBLASLt loses more to the leftover than our best method does (7.4% against 5.1% on average).

What hipBLASLt chooses:

- **Finer and larger tiles.** Almost all of its tiles have a non-power-of-two side, and on 336 shapes they hold more than 256x256 accumulators (for example 256x320, 288x288 and 224x384). On the 128 shapes whose last wave at 256x256 is at most a quarter full, its tile leaves the last wave at least three-quarters full on 117.
- **Almost no Stream-K split.** Every kernel it picks is a Stream-K kernel (`SK5`), but a split needs a fractional number of tiles per workgroup, and the launch grid gives one on only 4 of the 491 shapes.

The reason is how hipBLASLt chooses kernels, not a measured advantage of larger tiles. At most 31 of the shapes are in its tuned tables. For the rest, it ranks kernels with origami's latency model (`shared/origami` in rocm-libraries), which predicts one wave's time multiplied by ceil(tiles / CUs) whenever the Stream-K grid has fewer workgroups than tiles (`compute_launch_parameters` in `gemm.cpp`). The model therefore never credits a split, and it always prefers a tile that fits the waves better. In addition, its default Stream-K grid (`k_split_aware` in `streamk.cpp`) does not spread a small leftover over all the CUs. With between 1 and 1.5 times as many tiles as CUs, it gives each workgroup 1.5 tiles and so uses tiles / 1.5 CUs. That is 181 of the 256 CUs for 272 tiles, and it reaches all 256 only at 384 tiles. From 385 to 511 tiles, no fractional grid fits, and it falls back to one workgroup per tile with no split. For M = 1024, N = 17408, K = 7168 (272 tiles at 256x256), timed with `hipblaslt-bench`:

| hipBLASLt kernel | Grid | Origami's waves | Measured (us) |
|---|---|---|---|
| 256x256, default Stream-K grid | 181 workgroups, 1.5 tiles each | 2 | 243.2 |
| 256x256, no split | 272 workgroups | 2 | 268.2 |
| 256x256, Stream-K over all 256 CUs (`TENSILE_STREAMK_FIXED_GRID=256`) | 256 workgroups, 1.0625 tiles each | (not chosen) | 200.7 |
| 272x256 | 256 workgroups | 1 | 189.3 |

Origami's `compute_total_latency` confirms the whole-wave count. It gives 693,855 for the 272 tiles at 256x256, both with Stream-K over all 256 CUs and with no split. That is exactly twice the 346,927 of one full wave (N = 16384), whereas 272x256 gets 360,805. hipBLASLt's choice is therefore not evidence that larger tiles beat Stream-K. Whether its own split would win in general depends on the split's cost, which this study does not measure in enough cases to say.

## Impact on AITER: a tuned tile is only good for the M it was tuned at

AITER's tuned configs cover only some values of M. Other values reuse a nearby row through `getPaddedM`, which rounds M up to the next fine granularity and then to the next power of two, so a tile tuned at M0 also runs for M0/2 < M < M0. To measure the cost, 9 prefill shapes were taken from AITER's configs (M0 from 1024 to 16384, all from rows where torch was the fastest backend). Every base and finer tile was timed at every M in that range with M > 512 (191 points, confirmed by replay with 0.4% noise):

| Tile menu | Tile tuned at M0 / best tile per M (geomean) | Worst | Points more than 5% slower | Distinct best tiles over the range |
|---|---|---|---|---|
| Base (power-of-two) | 1.017 | 1.37 | 7% | 1-4 |
| Finer tiles | 1.055 | 1.45 | 33% | 4-11 |

<img width="1800" height="1050" alt="Image" src="https://github.com/user-attachments/assets/b4a84a91-418c-4b56-9725-3f6470a7fd53" />

**Figure 3.** One pair of bars per shape, over M0/2 < M < M0. The top panel shows how many different tiles are the best tile somewhere in the range. The bottom panel shows what reusing the tile tuned at M0 costs, against the best tile from the same menu at each M.

- **Power-of-two tiles change in coarse steps,** so reuse costs at most 0.2% on 7 of the 9 shapes. It matters only at small N (1.09 for N = 1536 and 1.11 for N = 2560), where there are few tile columns and the wave fill changes quickly with M.
- **With finer tiles, the best tile changes on every shape.** Reuse costs 1.5% to 16% per shape. Reused finer tiles are still about 1.4% ahead of the base's tuned tile, but most of their gain depends on choosing the tile for the actual M.

The tail methods are less exposed, because they keep the main tile and their tail depends only on S, which the kernel can compute from the actual M at launch (as gfx1250's rule does).

## Next steps

- Implement the rule in the dispatcher (S, K and the padding share are all known at launch).
- Replace the tuned finer tile and the padding test with a launch-time cost model over the tile menu (waves, padding and per-tile throughput). The tile would then follow M, and the choice would depend on the padding a finer tile actually removes rather than the padding share at 256x256.
- Add finer tiles larger than 256x256 of accumulators, as hipBLASLt uses, where LDS allows.
- Try combining a finer tile with a tail method.
- Use the persistent loop at short K. The Stream-K kernel's data-parallel plan already provides it (4-6% for K <= 1024 and 1-2.5% for 1024 < K <= 2048), but the rule does not use it, and the sub-tile tail and finer tiles do not have it. Adding both to the persistent kernel, as gfx1250's Stream-K kernel already does for the sub-tile tail, would combine the gains.

## Caveats

- One workgroup per CU (P = 256). Within each method the tile is tuned per shape, so the results assume a tile selection as good as the tuner's. For finer tiles reused across M, the results are much worse (see Impact on AITER).
- Finer tiles here have sides of at most 320 and at most 256x256 of accumulators.
- The rule's thresholds are specific to these implementations and this tile menu (see What carries over). The grid covers only shapes with at least one full wave of 256x256 tiles, down to N = 4096, K = 1024 and M = 512 (300 in the ragged-M set).
- Stream-K runs in a persistent kernel and the other methods do not. For K <= 2048 the loop alone is worth 1-6%, so short-K comparisons favour Stream-K, and the sub-tile tail's lead there is, if anything, understated. "Stream-K" means its splitting plans only.
- The grid is bf16 without bias, at five values of K; an AITER shape uses the rule for its K group. M is a multiple of 256 except in the ragged-M set, so the 10% gate rests on 128 grid shapes and the 10 AITER shapes with S/P <= 1/2 and more than 10% padding.
- The AITER shapes are not a fully blind test. The ragged-M set was added because a fit without padding disagreed with them for 2048 < K <= 4096, and capping Stream-K's threshold at 1/2 also gains 0.06% on them.
- The thread traces cover one shape on one CU, and the counters cover 8 shapes.
- Including the 59 decode-like shapes with M <= 128 changes the rule's result by under 0.5% (1.070 against 1.073 for the best method per shape).
- hipBLASLt is a different kernel, and except on one sweep it was run with its heuristic kernel choice while our variants were tuned per shape. The comparison therefore shows that our variants are competitive, not that our kernels are faster or which tile choice is better.

## Attached

 - [gemm_a16w16_subtile.py](https://github.com/user-attachments/files/34566503/gemm_a16w16_subtile.py): the sub-tile tail, built on `gemm_a16w16.py`.
 - [gemm_a16w16_uneven.py](https://github.com/user-attachments/files/34566504/gemm_a16w16_uneven.py): finer tiles (uneven quadrant halves), built on `gemm_a16w16.py`.