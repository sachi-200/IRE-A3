"""Tie-aware recall@k.

A returned item counts as a hit if its *true* distance to the query is no
larger than the true distance of the k-th exact neighbour. So if several base
vectors are tied at the k-th position, returning any of them is correct, even
though only one of them appears in the ground-truth id list.

Distances are recomputed in float64 from the raw vectors rather than taken
from the index, because (a) compressed indexes such as IVF-PQ return
approximate distances, and (b) float32 distances from FAISS can be off by a
little, which would turn a genuine tie into a "miss".
"""
import numpy as np


def exact_distances(queries, base, ids, chunk=1000):
    """Exact squared L2 distance (float64) from each query to each listed id.

    queries: (nq, d), base: (nb, d), ids: (nq, m) int. Ids of -1 (FAISS's
    "no result" marker) get distance +inf, so they never count as hits.
    """
    ids = np.asarray(ids)
    out = np.full(ids.shape, np.inf, dtype=np.float64)
    for s in range(0, len(ids), chunk):
        blk = ids[s:s + chunk]
        valid = blk >= 0
        vecs = base[np.where(valid, blk, 0)].astype(np.float64)        # (b, m, d)
        q = queries[s:s + chunk].astype(np.float64)[:, None, :]        # (b, 1, d)
        d = ((vecs - q) ** 2).sum(axis=-1)
        out[s:s + chunk] = np.where(valid, d, np.inf)
    return out


def recall_at_k(retrieved_dists, gt_dists, k=10, rtol=1e-6, atol=1e-9):
    """Per-query tie-aware recall@k.

    retrieved_dists: (nq, >=k) true distances of the items the index returned
                     (in returned order; only the first k are used). Items
                     are assumed distinct, which holds for FAISS results.
    gt_dists:        (nq, >=k) true distances of the exact nearest neighbours.

    Returns an (nq,) array; take .mean() for the dataset-level number.
    """
    retrieved_dists = np.asarray(retrieved_dists, dtype=np.float64)
    gt_dists = np.asarray(gt_dists, dtype=np.float64)
    if gt_dists.shape[1] < k:
        raise ValueError(f"need at least {k} ground-truth distances per query")
    if retrieved_dists.shape[0] != gt_dists.shape[0]:
        raise ValueError("retrieved and ground truth have different numbers of queries")

    kth = np.sort(gt_dists, axis=1)[:, k - 1]
    threshold = kth + rtol * np.abs(kth) + atol
    top = retrieved_dists[:, :k]
    hits = (top <= threshold[:, None]).sum(axis=1)
    return hits / k


def recall_at_k_from_ids(queries, base, retrieved_ids, gt_ids, k=10):
    """Convenience wrapper: recompute true distances, then call recall_at_k."""
    return recall_at_k(
        exact_distances(queries, base, np.asarray(retrieved_ids)[:, :k]),
        exact_distances(queries, base, np.asarray(gt_ids)[:, :k]),
        k=k,
    )
