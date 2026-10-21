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

## Part 4: Drifting data (synthetic)

**Question.** If an index is built on one distribution and new vectors then
arrive from a shifted one, which index degrades, how (recall or speed), and
does rebuilding fix it?

**Setup.** Seeded Gaussian mixture in d = 64. The initial data is 200K vectors
from 100 clusters (centres ~ N(0, I), within-cluster noise sigma = 0.25). Each
index is built and trained on these 200K only. Then 200K new vectors are
inserted, drawn from 100 new clusters generated the same way but with every
centre moved by a fixed vector of length s * R, where R = sqrt(64) = 8 is the
typical distance of a centre from the origin. Queries (2,000 tune + 2,000
test) come from the new clusters, and ground truth is exact search over all
400K vectors.

* Varied property: the shift s = 0.5, 1, 2.
* Control: s = 0 (new vectors come from the same distribution).
* Everything else fixed: HNSW (M = 16, efConstruction = 200), IVF-Flat
  (nlist = 1024), IVF-PQ (nlist = 1024, m = 16, 8 bits). efSearch and nprobe
  are chosen on the control so that tune recall reaches 0.95, then held fixed.
* Each index is measured "stale" (built on the old data, new data inserted)
  and "rebuilt" (trained and built from scratch on all 400K).

**Hypothesis.**

1. *IVF-Flat loses speed, not recall.* Its centroids stay where the old
   data was. New vectors are assigned to the few old centroids closest to
   the shifted region, so those lists become very long. Queries from the new
   region probe exactly those lists, so they still find their neighbours
   (recall stays near the control), but each query scans far more vectors,
   so QPS falls as s grows. Rebuilding spreads the new data over new
   centroids and restores the control's speed.
2. *IVF-PQ loses recall.* Its PQ codebooks were trained on residuals of the
   old data (vector minus its nearby centroid, all small). New vectors are
   far from every old centroid, so their residuals are much larger than
   anything the codebooks have seen, the quantization error grows, and
   neighbours are misordered. I expect recall at fixed nprobe to drop
   sharply with s, and rebuilding to restore it.
3. *HNSW is robust.* Inserting into a graph searches the current graph for
   neighbours, so new vectors link to each other and the new region gets
   its own well-connected part of the graph; the upper layers also receive
   new nodes, so search can reach the region. I expect stale HNSW to stay
   within about 0.02 recall of the rebuilt one at every s.
4. *Control check.* At s = 0, stale and rebuilt indexes should be about the
   same for all three, which shows that any difference at s > 0 comes from
   the shift and not from building the index in two steps.
