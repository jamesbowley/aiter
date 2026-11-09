# Results

**Main finding:** Three quantities decide which method wins on gfx950: how full the last wave is, K, and the padding. A simple rule based on them comes within about 0.3% of picking the best method for every shape.

- The rule was fitted on a controlled grid of 912 shapes.
- It holds on 491 AITER shapes it was **not** fitted on.
- Each boundary in the rule matches a mechanism. So the structure should outlast these implementations, even though most thresholds will move.

P, S, S/P, Q, intensity and the three methods are as defined in the ticket. This reply adds these terms:

| Term | Meaning |
|---|---|
| Base | `gemm_a16w16.py` with power-of-two tiles |
| Tail methods | The sub-tile tail and Stream-K together |
| Padding share | Fraction of the **256x256** tile grid that lies outside the M x N output |
| Best finer tile | The **fastest** finer tile for a given shape |
| Padding removed | Padding share minus the padding of the best finer tile's grid |
| Speedup | Geomean of t_base / t_method, with the base at its best power-of-two tile for each shape. Above 1 means faster than the base. |

## Summary

### The rule

1. If S/P > 1/2, **or** the padding share is above 10%: use **finer tiles**.
2. Otherwise, if K ≤ 4096: use the **sub-tile tail**.
3. Otherwise (K > 4096): use **Stream-K**.

| Speedup over the base | Grid (912 shapes) | AITER shapes (491) |
|---|---|---|
| Best single method | 1.051-1.053 (all three within 0.2%) | 1.061 (finer tiles) |
| Recommended rule | **1.070** | **1.075** |
| Best method for each shape | 1.073 | 1.078 |

### Why each boundary exists

Each boundary comes from the problem, not from the code. The first three match the ticket's expected trade-offs. The fourth is new.

- **Half a wave left (S/P = 1/2).** This is the sub-tile tail's limit (S·Q ≤ P with Q ≥ 2). The measurements show Stream-K stops there too: above it, the split costs more than it recovers.
- **K = 4096 (sub-tile tail vs Stream-K).** Stream-K's fixed split cost is paid off by long K. The sub-tile tail's cost hardly depends on K.
- **Small leftover (tail methods vs finer tiles).** Finer tiles lower the intensity of **every** tile to fix a leftover that is only a few percent of the work, and that costs more than it recovers. The tail methods keep the 256x256 main loop and pay only on the leftover tiles.
- **10% padding (finer tiles vs tail methods).** The tail methods pay for the padding in every 256x256 tile. A finer tile that fits the ragged dimension removes it.

### What carries over

The rule shows that the differences between the methods are simple, smooth functions of the shape. It does **not** show that the implementations are optimal. A method that was a few percent slower on every shape would move a boundary, but the rule would stay just as clean.

Expected changes as the methods are optimised or combined:

- **Half-wave limit:** structural. It will not move.
- **K crossover:** moves **down** if Stream-K's split gets cheaper, and **up** if the sub-tile tail gets a persistent loop.
- **10% padding gate:** depends on the tile menu (sides in multiples of 32, at most 256x256 of accumulators).
- **Combined kernels:** combining a finer tile with a tail method would make the choice a joint one of tile and tail. A cost model would suit that better than thresholds.
- **Out of scope:** shapes below one wave, or near the memory-bound region, involve other mechanisms. They are outside this grid.

### Answers to the ticket's questions

1. **How much does each method recover?** On the grid, the best method recovers:
   - 12-22% when the last wave is at most a quarter full,
   - 6-11% when it is a quarter to half full (depending on K),
   - at most 1.5% with more than half a wave left and no padding (Table 1).
2. **Where does each method stop helping?**
   - Both tail methods stop at half a wave. The sub-tile tail cannot go further by construction. Beyond half a wave, Stream-K gains at most about 1%, and only for K ≥ 8192.
   - Stream-K's split breaks even at about K = 1024. It gains up to 16% at K = 16384.
   - The sub-tile tail's gain hardly depends on K.
3. **Do finer tiles lose when S is small?** Yes. Their lower intensity costs more than the idle CUs they save.
   - With at most half a wave left and no padding, finer tiles are 2-6% behind the better tail method.
   - The reason: to fill the waves when the leftover is small, the tile must shrink a lot. Every tile then runs at lower MFMA efficiency.
   - Example: 1024x18432x8192 (32 tiles past a full wave). The best finer tile is 256x160. Its main loop runs at 80% MFMA efficiency, against 93-95% for the 256x256 main loop that the tail methods keep.
4. **Single method or dispatch?** Dispatch clearly gives more. But the rule needs the padding share as well as S/P and K. Without the padding test, the rule gets 1.061 on the grid instead of 1.070.
5. **How much is Stream-K's persistent loop worth?**
   - K ≥ 4096: almost all of Stream-K's gain comes from the split. The loop is worth under 1%.
   - K ≤ 1024: the loop alone is worth 4-6%, as much as or more than the split.
6. **How do we compare with hipBLASLt?** Our variants are competitive. That does **not** mean our kernels are faster.
   - On the AITER shapes, each of our variants (tuned per shape) is 2-4% faster than hipBLASLt's **heuristic** kernel choice. The best of the three is 5.3% faster.
   - On one sweep where hipBLASLt was **tuned exhaustively**, it was about 3% faster than our best method.
   - hipBLASLt uses finer and larger tiles, and almost never runs a Stream-K split. Its cost model counts a Stream-K kernel as whole waves.
7. **What is the impact on AITER?** AITER reuses a tile tuned at one M for every M down to half of it.
   - For finer tiles, reuse costs 5.5% on average (up to 45%) against the best tile at each M.
   - For power-of-two tiles, it costs 1.7%.
   - So finer tiles need a tile selection that follows M. The tail methods are less exposed, because S is computed from the actual M at launch.

## The experiment

### Kernels

Each method is tuned per shape. Like a tuner, it may fall back to the base's whole tiles.

- **Base:** `gemm_a16w16.py`, the gfx950 gluon compute-bound kernel (PR [#5897](https://github.com/ROCm/aiter/pull/5897)). Power-of-two tiles: sides 64, 128 or 256.
- **Finer tiles:** `gemm_a16w16_uneven.py`. This is the base with uneven quadrant halves. Sides are any multiple of 32 from 64 to 320, with at most 256x256 of accumulators.
- **Sub-tile tail:** `gemm_a16w16_subtile.py`. This is the base with the sub-tile tail added. Sub-tile sides are 64, 128 or 256.
- **Stream-K:** the splitting plans of `gemm_a16w16_streamk.py` (PR [#7391](https://github.com/ROCm/aiter/pull/7391)). It is a persistent kernel: each CU runs one workgroup that loops over its tiles. Its data-parallel plan (persistent loop, no split) is used only to separate the effect of the loop from the effect of the split.

The finer-tile and sub-tile kernels are small diffs to `gemm_a16w16.py` (attached). With the new feature switched off, both compile to **exactly** the base's machine code. This was checked for all 9 power-of-two tiles, with and without bias, and with ragged M, N and K. So every measured difference comes from the new feature.

### Grid (used to fit the rule)

The grid has 912 shapes in three parts:

- **Factorial part (720 shapes, no padding).** Every combination of:
  - K in {1024, 2048, 4096, 8192, 16384}
  - W in {1, 2, 4} full waves of 256x256 tiles
  - S in {0, 16, ..., 240}
  - M in {512, 1024, 4096}
  - N is chosen to give exactly W x 256 + S tiles.
- **Ragged-N part (64 shapes, padding share under 0.5%).** Uses N − 72 instead of N, for M = 1024, W in {1, 2} and K in {2048, 8192}. This gives ragged N edges with almost no padding.
- **Ragged-M part (128 shapes, padding share 9% to 41%).** M in {300, 450, 700, 1100}, for K in {2048, 8192}, W in {1, 2} and S/P ≤ 1/2. **This is the part that tests padding.**

Other facts about the grid:

- The rule is fitted and scored on all 912 shapes.
- M is fixed and N sets the tile count, so N is large: 4096 to 161,792. The smallest N is at M = 4096 with one full wave.
- All factorial and ragged-N shapes are compute-bound: at least 338 FLOP per byte of minimum traffic, against a ridge point of about 310 for MI355X.
- The M = 300 shapes are borderline: 260 to 290 FLOP per byte.
- The S = 0 controls tie, as expected.

### Test set (AITER shapes)

The test set is the ticket's shape set, taken from `aiter/configs/bf16_tuned_gemm.csv` and the 26 `model_configs/*bf16_tuned_gemm*.csv` files. Two more conditions apply:

1. The kernels support it: N and K are multiples of 8, and K ≥ 256.
2. M > 128. The 59 decode-like shapes with smaller M are excluded, because the base never uses 256x256 tiles on them.

That leaves **491 shapes**.

The rule is fitted on the grid instead, because these shapes come from real models rather than a designed experiment:

- K, W, fill and M move together.
- K is concentrated between 2048 and 8192.
- Some groups are dominated by near-identical shapes (for example, 18 shapes with N = 37888 and K = 5120).

The ticket's two control sets (30 shapes each) tie, as expected.

### Timing

- **AITER shapes:** tuned with AITER's tuner. Each method's pick was then timed again in a `--run_config` replay.
- **Grid:** timed with `grid_bench.py`. It measures each candidate the same way as the tuner, but does the per-shape setup only once (6 to 29 seconds per shape, against about 33 for the tuner). Like the tuner, it:
  - uses the same inputs,
  - rotates up to 101 input copies, so no iteration finds its inputs in L2 or MALL,
  - uses AITER's `run_iters_rotate` and `get_trace_perf`, and the same correctness check.

Checks on `grid_bench.py`:

- Against AITER's replays on 60 AITER shapes in the same session, it reads 0.2-0.8% higher at the median for **every** method. So there is no bias between methods.
- It picks the same winning method, or one within 1% of it, on 55 of the 60 shapes.
- The grid was timed twice, in two different random orders. Each method's best time differs between the two passes by 0.6% at the median.

All comparisons are within one session.

## Results on the grid

### Where each method helps

**Takeaway:** With at most half a wave left, the sub-tile tail wins for K ≤ 4096 and Stream-K wins for K ≥ 8192. With more than half a wave left and no padding, there is little to recover.

**Table 1.** Factorial part of the grid (no padding).

- Each cell is the speedup over the base, written as **finer tiles / sub-tile tail / Stream-K**.
- The best method in each cell is in bold.
- Each cell covers 36 shapes for S/P ≤ 1/2, and 63 shapes for S/P > 1/2.

| K | S/P ≤ 1/4 | 1/4 < S/P ≤ 1/2 | S/P > 1/2 |
|---|---|---|---|
| 1024 | 1.060 / **1.120** / 1.072 | 1.034 / **1.059** / 1.035 | **1.005** / 1.000 / 1.000 |
| 2048 | 1.074 / **1.121** / 1.093 | 1.037 / **1.056** / 1.036 | **1.006** / 1.000 / 1.000 |
| 4096 | 1.094 / **1.131** / 1.121 | 1.040 / 1.058 / **1.059** | **1.007** / 1.000 / 1.001 |
| 8192 | 1.118 / 1.148 / **1.165** | 1.054 / 1.069 / **1.088** | **1.009** / 1.000 / 1.007 |
| 16384 | 1.149 / 1.169 / **1.219** | 1.066 / 1.065 / **1.106** | 1.010 / 1.000 / **1.012** |

- **Sub-tile tail:** its gain is about the same at every K. It is the best method for K ≤ 4096. (At K = 4096 it ties with Stream-K when the last wave is a quarter to half full.)
- **Stream-K:** its gain grows with K, as expected from the fixed cost of the split. It is the best method for K ≥ 8192.
- **Finer tiles:** with at most half a wave left, they are 2-6% behind the better tail method at every K.
- **More than half a wave left:** with no padding, there is little to recover. Finer tiles help only just above half a wave (Figure 1).

<img width="2250" height="2100" alt="Image" src="https://github.com/user-attachments/assets/10bd2e61-09b8-4fbf-a207-3f3f6c00ffbd" />

**Figure 1.** Best method for every shape in the factorial part of the grid, faceted by W and M.

- These shapes have no padding, so the rule reduces to its S/P threshold. The padded shapes are in Figure 2.
- **Black line:** the rule's threshold at S/P = 1/2.
  - Below it, the rule uses a tail method: the sub-tile tail for K ≤ 4096, Stream-K for K ≥ 8192.
  - Above it, the rule uses finer tiles.
- **Solid cells:** one method is at least 1% ahead of the next.
- **Hatched cells:** the methods are within 1% of each other. The hatch colour shows the fastest.
- **Grey cells:** no method gains 1% over whole 256x256 tiles.
- **Number in each cell:** the best method's gain over whole 256x256 tiles, in percent.

### Padding decides between finer tiles and the tail methods

**Takeaway:** What decides the winner is the padding a finer tile **removes**, not the padding share. The rule still uses the padding share, because it depends only on M and N.

- The tail methods keep 256x256 tiles, so they pay for the padding in every tile.
- A finer tile can pick a side that fits the ragged dimension.

On the ragged-M part of the grid (128 shapes, all with S/P ≤ 1/2):

| M | Padding share | Best finer tile | Finer tiles vs the better tail method |
|---|---|---|---|
| 300 | 41% | Two rows of 160 fit | **21% faster** |
| 1100 | 14% | Five rows of 224 fit | **5.7% faster** |
| 450 | 12% | 160x288, still leaves 6% padding | **1.3% slower** |

At M = 450, the best finer tile removes only half the padding, and it still costs intensity. This is why the padding removed is the better predictor:

- The tail methods pay the padding share.
- Finer tiles pay their own padding.
- So what matters is the difference: the padding removed.

The padding share is an upper bound on the padding removed. It is usually close to it, but not always (M = 450). The rule uses the padding share because it depends only on M and N. The padding removed depends on which finer tile is chosen.

<img width="2250" height="930" alt="Image" src="https://github.com/user-attachments/assets/47e5ae15-a762-4830-940a-e11744c8de1b" />

**Figure 2.** Every shape with at most half a wave left.

- **Vertical axis (both panels):** the better tail method's time ÷ the finer tiles' time. Above 1 means finer tiles are faster.
- **Left panel, horizontal axis:** the padding share. This is the rule's input.
- **Right panel, horizontal axis:** the padding removed.
  - A **negative** value means the best finer tile has **more** padding than 256x256. It was picked for wave fill, not padding.
- **Grey points:** grid shapes with no padding, where the tail methods win.
- **Downward triangles:** M = 450.
- **Black line:** the geomean of each padding band.

How the padding removed is computed:

- M = 300: 256x256 tiles cover 512 rows (41% padding). 160-row tiles cover 320 rows (6%). The padding removed is **35 points**.
- M = 450: 256x256 tiles cover 512 rows (12% padding). 160-row tiles cover 480 rows (6%). The padding removed is **only 6 points**.

How to read it:

- **Left panel:** the M = 450 shapes sit below 1, even though they are past the 10% gate.
- **Right panel:** the M = 450 shapes sit with the other shapes that remove about 6 points, where the band geomean is close to 1.
- So the finer tiles' advantage follows the padding removed more closely than the padding share.

### Fitting the rule

**Takeaway:** A 10% padding gate fits best, and the choice of tail method for each K is stable.

How the rule was fitted:

1. For each candidate padding gate, use finer tiles above the gate.
2. For each K, choose the tail method and its S/P threshold that minimise the geomean loss against the best method for each shape.
3. Score the whole rule on all 912 grid shapes.

| Finer tiles when the padding share is above | Geomean loss against the best method per shape | Shapes more than 3% slower than their best |
|---|---|---|
| no gate | 0.95% | 9.0% |
| 2% to 7.5% | 0.27% | 1.6% |
| **10%** | **0.19%** | **0.7%** |
| 15% to 30% | 0.34% | 2.9% |

Under the 10% gate:

- The fit chooses the sub-tile tail for K ≤ 4096 and Stream-K for K ≥ 8192 in all 200 bootstrap resamples of the shapes.
- The sub-tile tail's threshold is always S/P = 1/2, its structural limit.
- Stream-K's fitted threshold is 1/2 or higher (up to 15/16 at K = 16384 with four full waves). Capping it at 1/2 costs 0.02% on the grid, so the rule caps it.
- Fitting each W and each M separately gives the same split. The one exception is four full waves at K = 2048 and 4096, where Stream-K edges out the sub-tile tail.

### Why the tail methods win when the leftover is small

**Takeaway:** Finer tiles lower the intensity of every tile to fix a leftover that is only a few percent of the work. The tail methods change only the leftover tiles.

Example: 1024x16896x7168.

- At 256x256, this shape has 264 tiles: 256 run in the first wave and 8 in the second. The second wave is 97% idle.
- No tile of at most 256x256 fits the GEMM in one wave. Filling two waves needs tiles of about half the area.

In the table, useful share = M·N / (waves x P x tile area). Times are from tuning.

| Tile | Tiles | Last wave | Useful share | Intensity | Time (us) |
|---|---|---|---|---|---|
| 256x256 | 264 | 8 | 0.52 | 128 | 266.2 |
| 256x224 | 304 | 48 | 0.59 | 119 | 247.1 |
| 256x192 | 352 | 96 | 0.69 | 110 | 226.2 |
| 256x160 (best finer tile) | 424 | 168 | 0.82 | 98 | 213.6 |
| 128x288 | 472 | 216 | 0.92 | 89 | 224.1 |

The sub-tile tail (193.3 us) and Stream-K (181.8 us) are both faster than the best finer tile (213.6 us). They keep 256x256 for the first 256 tiles and change only the last 8, which are 3% of the work.

Thread traces show the mechanism. They were taken with `rocprofv3 --att` on one CU, for each method's winning kernel on 1024x18432x8192 (288 tiles, 32 past a full wave):

| Kernel | MFMA issue efficiency in the main loop | Main loop's share of instruction time |
|---|---|---|
| Base, 256x256 | 93.0% | 97% |
| Stream-K, 256x256 | 95.3% | 73% (the rest is its partial tile and fix-up) |
| Sub-tile tail, 256x256 main tile | 94.4% | 65% (the rest includes the sub-tiles) |
| Finer tiles, 256x160 | 80.4% | 94% |

- The tail methods run the base's main loop at the same efficiency. They pay only on the leftover tiles.
- Finer tiles pay for their lower intensity in **every** tile. The cost depends on how far the tile has to shrink.

Whole-kernel counters confirm this. They measure MFMA busy cycles over busy CU cycles, on 8 grid shapes with K ≥ 8192:

- **32 tiles past a full wave:** the best finer tile is 256x160. It runs 10-12% below the base at 256x256.
- **192 tiles past a full wave:** the best finer tile is 256x224. It runs only 2-3% below.

### Stream-K: the split vs the persistent loop

**Takeaway:** At long K, Stream-K's gain comes from the split. At short K, it comes mostly from the persistent loop, which the other methods could also use.

Both effects are measured within the Stream-K kernel:

- **Persistent loop vs base** = t_base / t_data-parallel. The data-parallel plan runs the same persistent loop without a split.
- **Split vs persistent loop** = t_data-parallel / t_split.
- Both use whole-tile fallback. This assumes the loop is worth the same with and without a split.

With at most half a wave left, on the grid:

| K | Persistent loop vs base | Split vs persistent loop |
|---|---|---|
| 1024 | 1.037 | 1.016 |
| 2048 | 1.013 | 1.029 |
| 4096 | 1.007 | 1.082 |
| 8192 | 1.004 | 1.112 |
| 16384 | 1.004 | 1.156 |

The grid starts at K = 1024. Below that, the AITER shapes show the trend continuing. On the 11 AITER shapes with K < 1024 (K from 320 to 768) and at most half a wave left:

- the loop is worth 6.3%,
- the split is 3.1% **slower** than the loop alone.

So the split breaks even near K = 1024.

## Test on the AITER shapes

**Takeaway:** In every region, the rule chooses the best of the three methods. No threshold was fitted on these shapes.

Speedups are over the base, from AITER's `--run_config` replays. The rule's choice for each region is in brackets.

| Region | Shapes | Finer tiles | Sub-tile tail | Stream-K | Rule | Best per shape |
|---|---|---|---|---|---|---|
| S/P ≤ 1/2, padding ≤ 10%, K ≤ 4096 (sub-tile tail) | 97 | 1.053 | **1.075** | 1.046 | 1.075 | 1.082 |
| S/P ≤ 1/2, padding ≤ 10%, K > 4096 (Stream-K) | 152 | 1.084 | 1.105 | **1.120** | 1.120 | 1.123 |
| S/P ≤ 1/2, padding > 10% (finer tiles) | 10 | **1.220** | 1.090 | 1.088 | 1.220 | 1.220 |
| S/P > 1/2, padding ≤ 10% (finer tiles) | 167 | **1.011** | 1.000 | 1.003 | 1.011 | 1.012 |
| S/P > 1/2, padding > 10% (finer tiles) | 65 | **1.126** | 1.000 | 1.002 | 1.126 | 1.126 |
| All | 491 | 1.061 | 1.048 | 1.048 | **1.075** | 1.078 |

- The AITER shapes differ from the grid mainly in padding. 75 of the 491 have more than 10% padding (M from 144 to 1600). Those are where finer tiles gain most.
- With more than half a wave left and at most 10% padding, the best method gains only 1.2% over the base. That is close to the 0.5-1.5% on the grid.
- So the large gain of finer tiles above half a wave comes from the 65 shapes with more than 10% padding.

Each rule on both sets. Each cell is **geomean speedup over the base / share of shapes more than 3% slower than their best method**:

| Rule | Grid (912) | AITER shapes (491) |
|---|---|---|
| Finer tiles everywhere | 1.051 / 29% | 1.061 / 23% |
| Sub-tile tail everywhere | 1.053 / 18% | 1.048 / 29% |
| Stream-K everywhere | 1.051 / 21% | 1.048 / 27% |
| Recommended rule without the padding gate | 1.061 / 6% | 1.073 / 4% |
| **Recommended rule** | **1.070 / 1%** | **1.075 / 2%** |
| Best method per shape | 1.073 / 0% | 1.078 / 0% |

Cross-checks:

- **Harness check:** each method's pick on 135 grid shapes (covering every K, W and M) was replayed through AITER's tuner. The rule gets 1.089, against 1.093 for the best method per shape. So the grid result does not depend on the harness.
- **Fresh replay** of the AITER shapes in the same session: the rule gets 1.077, against 1.081.
- **Weighted by run time** on the AITER shapes: the rule gets 1.034, against 1.037.

## Comparison with hipBLASLt

**Takeaway:** Our tuned variants beat hipBLASLt's **default** kernel choice. That does not mean our kernels are faster: on one sweep, **tuned** hipBLASLt was about 3% faster than our best method.

Our variants were compared with torch `F.linear`, which runs hipBLASLt's own kernel choice, timed in the same session. hipBLASLt is an optimised library that also uses non-power-of-two tiles. Values are t_hipBLASLt / t_ours, so above 1 means ours is faster.

| Ours | All (491) | hipBLASLt tile at most 256x256 (155) | hipBLASLt tile larger than 256x256 (336) |
|---|---|---|---|
| Base (power-of-two tiles) | 0.977 | 1.008 | 0.963 |
| Finer tiles | 1.036 | 1.066 | 1.022 |
| Sub-tile tail | 1.024 | 1.040 | 1.016 |
| Stream-K | 1.024 | 1.038 | 1.017 |
| Best of the three | 1.053 | 1.073 | 1.044 |

What this table compares:

- **Ours:** each variant tuned per shape over its full menu.
- **hipBLASLt:** its heuristic choice. For all but at most 31 of the shapes, it picks its kernel with a cost model, not a tuning run.

One sweep tested the kernels directly, with hipBLASLt tuned:

- Shapes: M = 1024, K = 7168, N from 16384 to 32768.
- hipBLASLt was tuned exhaustively with `hipblaslt-bench`: every solution under three Stream-K grid policies.
- Tuned hipBLASLt was 3.3% faster than our best method on the 15 shapes with a leftover. It was 4-5% faster at exactly one and two full waves.
- The two sides were timed with different harnesses (`hipblaslt-bench` and `do_bench`), so the absolute gap is only indicative.
- Measured against its own full-wave time, tuned hipBLASLt loses **more** to the leftover than our best method does: 7.4% against 5.1% on average.

### What hipBLASLt chooses

- **Finer and larger tiles.** Almost all of its tiles have a non-power-of-two side. On 336 shapes, they hold more than 256x256 accumulators (for example 256x320, 288x288 and 224x384). On the 128 shapes whose last wave at 256x256 is at most a quarter full, its tile leaves the last wave at least three-quarters full on 117.
- **Almost no Stream-K split.** Every kernel it picks is a Stream-K kernel (`SK5`). But a split needs a fractional number of tiles per workgroup, and the launch grid gives one on only 4 of the 491 shapes.

### Why it chooses this

The reason is how hipBLASLt picks kernels, not a measured advantage of larger tiles.

- **Its cost model never credits a split.**
  - At most 31 of the shapes are in its tuned tables.
  - For the rest, it ranks kernels with origami's latency model (`shared/origami` in rocm-libraries).
  - When the Stream-K grid has fewer workgroups than tiles, the model predicts one wave's time × ceil(tiles / CUs) (`compute_launch_parameters` in `gemm.cpp`).
  - So it always prefers a tile that fits the waves better.
- **Its default Stream-K grid does not spread a small leftover over all the CUs** (`k_split_aware` in `streamk.cpp`).
  - With 1 to 1.5 times as many tiles as CUs, it gives each workgroup 1.5 tiles, so it uses tiles / 1.5 CUs.
  - For 272 tiles, that is 181 of the 256 CUs. It reaches all 256 CUs only at 384 tiles.
  - From 385 to 511 tiles, no fractional grid fits. It falls back to one workgroup per tile, with no split.

Example: M = 1024, N = 17408, K = 7168 (272 tiles at 256x256), timed with `hipblaslt-bench`:

| hipBLASLt kernel | Grid | Origami's waves | Measured (us) |
|---|---|---|---|
| 256x256, default Stream-K grid | 181 workgroups, 1.5 tiles each | 2 | 243.2 |
| 256x256, no split | 272 workgroups | 2 | 268.2 |
| 256x256, Stream-K over all 256 CUs (`TENSILE_STREAMK_FIXED_GRID=256`) | 256 workgroups, 1.0625 tiles each | (not chosen) | 200.7 |
| 272x256 | 256 workgroups | 1 | 189.3 |

Origami's `compute_total_latency` confirms the whole-wave count:

- 272 tiles at 256x256 get 693,855, both with Stream-K over all 256 CUs and with no split.
- That is exactly twice the 346,927 of one full wave (N = 16384).
- 272x256 gets 360,805.

So hipBLASLt's choice is not evidence that larger tiles beat Stream-K. Whether its own split would win in general depends on the split's cost. This study does not measure that in enough cases to say.

## Impact on AITER: a tuned tile is only good for the M it was tuned at

**Takeaway:** Finer tiles need a tile choice that follows M. Reusing a finer tile tuned at a different M loses most of its gain.

How AITER reuses tiles:

- AITER's tuned configs cover only some values of M.
- Other values reuse a nearby row through `getPaddedM`. It rounds M up to the next fine granularity, then to the next power of two.
- So a tile tuned at M0 also runs for every M with M0/2 < M < M0.

How the cost was measured:

- 9 prefill shapes from AITER's configs, with M0 from 1024 to 16384. All come from rows where torch was the fastest backend.
- Every base tile and finer tile was timed at every M in M0/2 < M < M0 with M > 512.
- 191 points in total, confirmed by replay with 0.4% noise.

Reuse cost = time with the tile tuned at M0 ÷ time with the best tile at each M, from the same tile menu.

| Tile menu | Reuse cost (geomean) | Worst | Points more than 5% slower | Distinct best tiles over the range |
|---|---|---|---|---|
| Base (power-of-two) | 1.017 | 1.37 | 7% | 1-4 |
| Finer tiles | 1.055 | 1.45 | 33% | 4-11 |

<img width="1800" height="1050" alt="Image" src="https://github.com/user-attachments/assets/b4a84a91-418c-4b56-9725-3f6470a7fd53" />

**Figure 3.** One pair of bars per shape, over M0/2 < M < M0.

- **Top panel:** how many different tiles are the best tile somewhere in the range.
- **Bottom panel:** the reuse cost, against the best tile from the same menu at each M.

What it shows:

- **Power-of-two tiles change in coarse steps.** Reuse costs at most 0.2% on 7 of the 9 shapes. It matters only at small N (1.09 for N = 1536 and 1.11 for N = 2560). There, there are few tile columns, so the wave fill changes quickly with M.
- **With finer tiles, the best tile changes on every shape.** Reuse costs 1.5% to 16% per shape. Reused finer tiles are still about 1.4% ahead of the base's tuned tile, but most of their gain depends on choosing the tile for the actual M.

The tail methods are less exposed. They keep the main tile, and their tail depends only on S, which the kernel can compute from the actual M at launch (as gfx1250's rule does).

## Next steps

- **Implement the rule in the dispatcher.** S, K and the padding share are all known at launch.
- **Replace the tuned finer tile and the padding test with a launch-time cost model** over the tile menu (waves, padding and per-tile throughput). The tile would then follow M. The choice would depend on the padding removed, rather than the padding share.
- **Add finer tiles larger than 256x256 of accumulators**, as hipBLASLt uses, where LDS allows.
- **Try combining a finer tile with a tail method.**
- **Use the persistent loop at short K.**
  - The Stream-K kernel's data-parallel plan already provides it: 4-6% for K ≤ 1024, and 1-2.5% for 1024 < K ≤ 2048.
  - The rule does not use it, and the sub-tile tail and finer tiles do not have it.
  - Adding both to the persistent kernel would combine the gains. gfx1250's Stream-K kernel already does this for the sub-tile tail.

## Caveats

- **One workgroup per CU (P = 256).**
- **Tuned tiles.** Within each method, the tile is tuned per shape. So the results assume a tile selection as good as the tuner's. For finer tiles reused across M, the results are much worse (see Impact on AITER).
- **Tile menu.** Finer tiles here have sides of at most 320, and at most 256x256 of accumulators.
- **Thresholds are implementation-specific.** They depend on these implementations and this tile menu (see What carries over).
- **Grid coverage.** The grid covers only shapes with at least one full wave of 256x256 tiles, down to N = 4096, K = 1024 and M = 512 (M = 300 in the ragged-M part).
- **Only Stream-K has a persistent loop.** For K ≤ 2048, the loop alone is worth 1-6%. So short-K comparisons favour Stream-K, and the sub-tile tail's lead there is, if anything, understated. "Stream-K" here means its splitting plans only.
- **Limited K and M.** The grid is bf16 without bias, at five values of K. An AITER shape uses the rule for its K group. M is a multiple of 256 except in the ragged-M part. So the 10% gate rests on 128 grid shapes and the 10 AITER shapes with S/P ≤ 1/2 and more than 10% padding.
- **The AITER test is not fully blind.** The ragged-M part was added because a fit without padding disagreed with the AITER shapes for 2048 < K ≤ 4096. Capping Stream-K's threshold at 1/2 also gains 0.06% on them.
- **Small trace sample.** The thread traces cover one shape on one CU. The counters cover 8 shapes.
- **Decode-like shapes.** Including the 59 shapes with M ≤ 128 changes the rule's result by under 0.5% (1.070 against 1.073 for the best method per shape).
- **hipBLASLt is a different kernel.** Except on one sweep, it ran with its heuristic kernel choice, while our variants were tuned per shape. So the comparison shows our variants are competitive. It does not show that our kernels are faster, or which tile choice is better.

## Attached

- [gemm_a16w16_subtile.py](https://github.com/user-attachments/files/34566503/gemm_a16w16_subtile.py): the sub-tile tail, built on `gemm_a16w16.py`.
- [gemm_a16w16_uneven.py](https://github.com/user-attachments/files/34566504/gemm_a16w16_uneven.py): finer tiles (uneven quadrant halves), built on `gemm_a16w16.py`.