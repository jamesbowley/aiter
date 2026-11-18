# Cost model for the gfx950 a16w16 Gluon GEMMs

`gemm_a16w16_streamk_cost_model.py` predicts the time of each complete Gluon candidate for a shape
(compute-bound kernel, or Stream-K tile + plan + split + exchange + partials type) and ranks them.
The lowest prediction is the pick; the lowest few are a shortlist to time instead of the full
menu (about 1,000 kernels per shape).

```python
from aiter.ops.triton._gluon_kernels.gfx950.gemm.basic import gemm_a16w16_streamk_cost_model as cm

cm.shortlist(1256, 32320, 5120, k=5)    # kernel names, lowest predicted time first
cm.predict_kernel_us(1256, 32320, 5120, "gluon_streamk_256x256x64_g4_one_tile_owner_k2_wt_p16")
```

or `python -m aiter.ops.triton._gluon_kernels.gfx950.gemm.basic.gemm_a16w16_streamk_cost_model M N K [k]`.

It is a **ranking model**: its 10 constants are fitted so that candidates of the same shape come
out in the right order. Absolute predictions are only good to about +-25%, and the constants are
effective values, not hardware measurements (see "What the constants are").

## The form

```text
1. Tile grid         T = ceil(M / BM) * ceil(N / BN) tiles on 256 CUs
                     W = floor(T / 256) full waves,  S = T mod 256 leftover tiles

2. Cycles per        cyc = BM*BN*BK/1024 * (1 + c/BM + c/BN)  +  2 * (s0 + s_x * extra_blocks)
   K-pair on one CU        '--- MFMA ---'    '- edge cost -'      '---- fixed per K-step ----'
   (a K-pair = two K-steps of BK; BM*BN*BK/1024 = 4*BM*BN*BK flops at 4,096 bf16 flops per CU
    per cycle; extra_blocks is 0 for power-of-two tiles)

3. One phase         t_compute = K_pairs * cyc / 2.4 GHz
                     t_memory  = unique HBM bytes of the phase / B_hbm
                     t_phase   = (t_compute^p + t_memory^p)^(1/p)  +  e_pf * cyc / 2.4 GHz

4. Total             W full-wave phases + one phase for the S leftover tiles, run by the plan:
                       dp        whole tiles
                       dp_sub    smaller sub-tiles
                       one_tile  each leftover tile split k ways along K (K_pairs / k), then a fixup
                       spread,   the tail's K-work shared by all 256 programs, then a fixup
                       two_tile
                     fixup = sync latency (s_ow owner, s_rs reduce-scatter wt, s_tt two_tile)
                             + max(partial bytes per program / b_cu, all partial bytes / B_hbm)
```

| Constant | Value | Meaning |
|---|---|---|
| c | 89.4 | per-side (edge) overhead of a K-step |
| s0 | 499 cycles | fixed cost per K-step |
| s_x | 27.5 cycles | per extra block of a non-power-of-two tile |
| e_pf | 1.06 K-pairs | pipeline fill and drain per tile |
| B_hbm | 4.86 TB/s | effective HBM bandwidth |
| p | 3.33 | sharpness of max(compute, memory) |
| s_ow | 1.39 us | owner fixup sync |
| s_rs | 0.27 us | reduce-scatter (wt exchange) sync |
| s_tt | 12.2 us | two_tile sync |
| b_cu | 29.5 GB/s | one program's partial-read rate |

The constants were fitted on MI355X (1,400 W) timings of the AITER bf16 shapes where Gluon is
competitive, plus synthetic grid shapes, with a pairwise ranking loss.

### Worked example: M = 1,256, N = 32,320, K = 5,120

```text
256x256x64:  T = 5 * 127 = 635 tiles: 2 full waves + 123 leftover tiles (the third wave 48% full)
             cyc = 4096 (MFMA) + 2862 (edge) + 999 (fixed) = 7956 per K-pair; K = 5120 is 40 K-pairs
             one full wave: compute 132.6 us, memory 211.8 MB / 4.86 TB/s = 43.6 us -> 133.6 + fill 3.5
  dp:                   2 waves 274.2 + last wave whole (40 K-pairs) 136.2                 = 410.4 us
  one_tile owner k2 p16: 2 waves 274.2 + last wave split in two (20 K-pairs each, 246 CUs busy)
                        70.3 + fixup (sync 1.39 + partials 6.6) 8.0                       = 352.6 us
```

| Candidate | Predicted | Measured | Predicted rank | Measured rank (of 760) |
|---|---|---|---|---|
| 256x256, one_tile owner k2, 16-bit partials | 352.6 us | 312.8 us | 1 | 1 |
| 256x256, one_tile owner k2, fp32 partials | 359.2 us | 314.6 us | 4 | 2 |
| 256x256, dp | 410.4 us | 326.7 us | 48 | 18 |
| 128x256, dp | 453.0 us | 367.1 us | 184 | 167 |

## How well it ranks (MI355X)

Regret = the chosen kernel's time / the fastest Gluon kernel's time for the shape (geomean).
"Unseen" = scored by a model fitted without any M of that shape's (N, K).

| Use | Shapes | Truth | Top pick | Best of top 5 timed |
|---|---|---|---|---|
| Tuning AITER's shapes (unseen (N, K)) | 3,425 | full sweep | 1.043 | 1.008 |
| Model families not in AITER (never fitted) | 80 | full sweep | 1.021 | 1.004 |
| Ms AITER has not tuned (unseen (N, K)) | 631 | harness re-time | 1.029 | 1.005 |
| The same, AITER's lookup-row kernel reused | 596 | harness re-time | 1.087 | - |

Tuning only the model's top 10 instead of the full menu changes the Gluon-vs-AITER dispatch
decision on 18 of 3,409 shapes. Seen and unseen (N, K) score within 0.3% of each other.

**Why it ranks well with unrealistic constants:** an error common to all candidates of a shape
(the real clock, absolute bandwidth, launch overhead) does not change their order. What matters
is the error between candidates: the model orders pairs that differ by 10% or more correctly
88-99% of the time, so it reliably discards most of the menu, but it orders kernels within 3%
of each other little better than chance. Hence a top pick about 3% behind, and a top 5 within
about 0.5%.

## What the constants are

- **The clock is a unit, not a measurement.** Ranking cannot separate the clock from the cycle
  overheads: scaling the clock, B_hbm and the sync latencies together changes every prediction of
  a shape by the same factor. 2.4 GHz is fixed; c and s0 absorb the gap between it and the
  sustained clock under load, together with the real in-loop overhead (a full wave of 256x256x64
  measures about 60% of MFMA peak at 2.4 GHz).
- **B_hbm is effective:** the fastest kernels on weight-streaming shapes reach 4.5-5.8 TB/s, and
  the model counts unique bytes only (no per-XCD L2 re-reads).
- **Not modelled:** a partly filled last wave running faster per tile (measured: a 48%-full wave
  costs about 80% of a full one). The fixup constants partly absorb it. A fitted term for it did
  not improve ranking at untuned Ms.

## Adapting it to MI350X (or another gfx950 part)

MI350X is the same gfx950 chip with the same CU count, so AITER's tuned-row lookup (keyed on
`gfx` and `cu_num`) cannot tell it from MI355X; it runs at a lower power limit and clock. The form
should carry over; the constants that encode the compute/memory balance (B_hbm, c, s0) and the
sync latencies are the ones expected to move. In order of cost:

1. **Check the MI355X constants as they are.** Pick a few hundred shapes (for example the AITER
   shapes of a few (N, K) groups, at tuned and intermediate Ms, multiples of 8). For each, time
   in AITER's tuner harness (`csrc/gemm_a16w16/gemm_a16w16_tune.py --run_config`, each row twice)
   the full sweep's 20 fastest kernels, the model's top 5 (`shortlist`) and the kernel AITER's
   lookup would run. Score the model's top pick and top 5 against the fastest row. If they are
   within about 0.5% of the MI355X numbers above, keep the constants.
2. **Measure the sustained clock** on a representative set of kernels (large-M 256x256 `dp`,
   small-M split plans, partly filled last waves) with hardware counters, for example
   `rocprofv3` with the GPU-active cycle counter divided by the kernel's duration. This turns the
   clock from a unit into a measurement: with it fixed, c, s0 and the fixup latencies can take
   physical values, and a partial-wave clock term can be tied to the measured clock instead of
   fitted.
3. **Refit the 10 constants** on an MI350X sweep, keeping the form.
   - Data: every menu candidate (`menu(M, N, K)`) timed for a few hundred to a few thousand AITER
     shapes, as rows (M, N, K, kernelName, us); their `candidate()` geometry gives the inputs.
   - Loss: within each shape, pairs of a near-best candidate (within 10% of the shape's best) and
     a candidate at least 2% slower; logistic loss on the difference of log predictions (scale
     0.03), plus a small term on the absolute log error to pin the scale. Start from the values in
     `C` and keep each constant within a physically plausible range.
   - Validate by cross-validation grouped by (N, K), and on intermediate Ms. Compare the top-pick
     and top-5 regret with the table above. Accept the refit only if it ranks at least as well;
     do not add terms that help on the fitted shapes but not on held-out ones.
4. **Use the measured clock to decide which C to load per part.** Both parts report the same
   `gfx`/`cu_num`, so pick the constants by product name or measured clock, not by architecture.
