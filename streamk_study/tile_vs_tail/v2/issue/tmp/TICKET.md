# [Exploration][MI350] Wave quantization, stream-k, sub-tiles or finer tiles

## Summary

A GEMM is split into output tiles, and each of the P = 256 CUs on MI355X runs one tile at a time, so the tiles run in waves of 256. When the largest tile, 256x256, gives more than one wave and the last wave is only partly full, the GEMM pays for wave quantization: S tiles are left over after the full waves, and P - S CUs sit idle while they run. This mostly affects compute-bound GEMMs, since they are the ones large enough to need more than one wave of the largest tile.

The cost is significant. PR [#7391](https://github.com/ROCm/aiter/pull/7391) (gfx950 gluon Stream-K) measures it in its wave-quantization figure (K = 4096): 16 tiles past a full wave, which leaves 240 CUs idle in the last wave, costs 32% of the throughput, and 22% and 16% when there are more full waves before it.

There are three candidate ways to deal with the S leftover tiles, and it is not clear which works best on gfx950, or when to use each one.

**Terms**

- **P**: number of CUs, 256 (one workgroup per CU).
- **S**: tiles left for the last wave at the largest tile, 256x256 (tiles mod P).
- **S/P**: how full the last wave is, from 0 to 1.
- **Q**: number of sub-tiles each leftover tile is cut into (sub-tile tail only).
- **Intensity**: BM·BN/(BM+BN) for a BM x BN tile, which is proportional to its arithmetic intensity (MACs per byte loaded). 256x256 has 128.

## Three methods

1. **Stream-K** (PR [#7391](https://github.com/ROCm/aiter/pull/7391)): split the S tiles over K across more CUs, then reduce the partial sums. The full waves keep the 256x256 tile, but the split adds a reduction and a workspace, so it has a fixed cost that long K should amortise better than short K.
2. **Sub-tile tail** (as in PR [#7364](https://github.com/ROCm/aiter/pull/7364) for gfx1250): cut each of the S tiles into Q smaller sub-tiles that each run over the whole K, so the last wave has S·Q workgroups instead of S. The full waves keep the 256x256 tile and no reduction is needed, but the rule S·Q <= P with Q >= 2 means it only applies when S <= P/2.
3. **Finer tiles**: allow tile sides that are not powers of two (160, 224, 288, ...), so the whole GEMM divides into waves that are close to full. This works for any S, but every tile is smaller than 256x256 and has lower intensity, so it may cost more than it recovers when S is small.

Expected trade-offs, to be checked:

| | Stream-K | Sub-tile tail | Finer tiles |
|---|---|---|---|
| Full waves keep the largest tile's intensity | yes | yes | no: every tile is smaller |
| Effect of K | gain should grow with K (fixed split cost) | should not depend much on K | should not depend much on K |
| Useful range of S | probably up to half a wave | up to half a wave (S·Q <= P, Q >= 2) | any S |
| Likely weakness | short K; large S | S > P/2 | small S |

## Questions

1. How much does each method recover against the base kernel's best power-of-two tile, as a function of the last-wave fill S/P and of K?
2. Where does each method stop helping? In particular, at what K does Stream-K's split break even, and does it help at all when more than half a wave is left?
3. When S is small, does the lower intensity of finer tiles cost more than the idle CUs it saves?
4. Is there a single best method, or does a dispatch policy that picks one per shape (from S/P and K) give clearly more?
5. How much of Stream-K's result comes from its persistent loop rather than from the split?
6. Are the variants competitive with hipBLASLt, and which approach does hipBLASLt itself take on these shapes?
7. Does any of these methods affect how AITER works today, for example how tuned configs are selected and reused across M?

## Deliverables

- POC versions of finer tiles and the sub-tile tail, built on the gfx950 gluon compute-bound kernel `gemm_a16w16.py` (PR [#5897](https://github.com/ROCm/aiter/pull/5897)).
- Per-shape tuning (AITER's tuner, confirmed by `--run_config` replay) of the base, the two variants and the Stream-K plans of PR #7391, on every bf16 shape in AITER's tuned configs that has more than one wave with a partial last wave, plus control sets with exactly full waves and with less than one wave.
- Results by last-wave fill and by K: each method at a fixed 256x256 tile, and end to end with tuned tiles, including the best method per region.
- A comparison with hipBLASLt (torch `F.linear`) on the same shapes.
- Examples of any impact on how AITER works (not an exhaustive study).
- A recommendation for which method to use where, and a list of follow-up work.
