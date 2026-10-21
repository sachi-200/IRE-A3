# Hypotheses

Each hypothesis is committed to git before the experiment it describes is run.

## Part 3: Compression (SIFT-1M)

**Setup.** IVF-PQ with nlist = 4096 and 8-bit codes, varying the number of PQ
sub-vectors m = 8, 16, 32, 64 (8 to 64 bytes per vector). Each m is run as
plain IVF-PQ and as IVF-PQ whose top 100 candidates are re-ranked with exact
distances. nprobe is swept as in Part 2.

**Hypothesis.**

1. *Plain IVF-PQ has a recall ceiling that rises with m, and no m reaches
   0.95.* PQ ranks candidates by approximate distances. Once the distance
   error is larger than the gap between the true 10th and 11th neighbours,
   their order is effectively random, so increasing nprobe stops helping.
   More sub-vectors mean fewer dimensions per 256-centroid codebook and a
   smaller error, so the ceiling rises with m (Part 2 measured about 0.60 /
   0.75 / 0.88 for m = 16 / 32 / 64). I expect m = 8 to plateau below 0.5.
2. *Re-ranking the top 100 lifts recall to at least 0.95 for m >= 16.* PQ
   errors mostly reorder near neighbours rather than pushing them far down
   the list, so the true top 10 is usually inside PQ's top 100. Re-ranking
   with exact distances then puts them back in order, and the remaining
   limit is the coarse quantizer (nprobe), as for IVF-Flat. For m = 8 the
   error may be so large that true neighbours fall outside the top 100, so
   re-ranking helps but may not reach 0.95.
3. *Re-ranking is cheap in time.* It adds 100 exact distance computations
   per query, which is small next to scanning thousands of codes. So at
   recall 0.95, IVF-PQ + re-ranking with a small m should be about as fast
   as, or faster than, IVF-Flat at the same nlist, because scanning short
   codes is cheaper than scanning full 512-byte vectors.
4. *But re-ranking gives up PQ's memory saving.* It needs the original
   vectors (512 bytes each) in addition to the codes, so its memory is
   slightly above IVF-Flat's. PQ only saves memory if the raw vectors can
   live on slower storage (disk or SSD), which this benchmark does not test.

## Part 4: Experiment on my own data

_Not written yet._
