import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from recall import exact_distances, recall_at_k, recall_at_k_from_ids  # noqa: E402


def test_perfect_recall():
    gt = np.arange(1, 11, dtype=float)[None, :]
    assert recall_at_k(gt, gt, k=10)[0] == 1.0


def test_order_does_not_matter():
    gt = np.arange(1, 11, dtype=float)[None, :]
    assert recall_at_k(gt[:, ::-1], gt, k=10)[0] == 1.0


def test_zero_recall():
    gt = np.arange(1, 11, dtype=float)[None, :]
    far = np.full((1, 10), 100.0)
    assert recall_at_k(far, gt, k=10)[0] == 0.0


def test_partial_recall():
    gt = np.arange(1, 11, dtype=float)[None, :]
    got = gt.copy()
    got[0, -3:] = 50.0  # three wrong items
    assert recall_at_k(got, gt, k=10)[0] == pytest.approx(0.7)


def test_tie_at_kth_is_not_a_mistake():
    # Exact top-10 distances, 10th is 10.0. Another base point (not in the
    # gt id list) is also at distance 10.0; returning it is correct.
    gt = np.arange(1, 11, dtype=float)[None, :]
    got = gt.copy()
    got[0, 9] = 10.0
    assert recall_at_k(got, gt, k=10)[0] == 1.0


def test_tie_with_float_noise_is_not_a_mistake():
    gt = np.arange(1, 11, dtype=float)[None, :]
    got = gt.copy()
    got[0, 9] = 10.0 * (1 + 1e-9)
    assert recall_at_k(got, gt, k=10)[0] == 1.0


def test_just_beyond_kth_is_a_mistake():
    gt = np.arange(1, 11, dtype=float)[None, :]
    got = gt.copy()
    got[0, 9] = 10.01
    assert recall_at_k(got, gt, k=10)[0] == pytest.approx(0.9)


def test_missing_results_count_as_misses():
    gt = np.arange(1, 11, dtype=float)[None, :]
    got = gt.copy()
    got[0, 5:] = np.inf  # e.g. FAISS returned -1 for the last 5
    assert recall_at_k(got, gt, k=10)[0] == pytest.approx(0.5)


def test_only_first_k_returned_items_are_used():
    gt = np.arange(1, 11, dtype=float)[None, :]
    got = np.concatenate([np.full((1, 10), 99.0), gt], axis=1)  # good ones at 11..20
    assert recall_at_k(got, gt, k=10)[0] == 0.0


def test_multiple_queries_are_independent():
    gt = np.tile(np.arange(1, 11, dtype=float), (2, 1))
    got = gt.copy()
    got[1] = 99.0
    np.testing.assert_allclose(recall_at_k(got, gt, k=10), [1.0, 0.0])


def test_end_to_end_with_vectors_and_tied_ids():
    # 1-D points on a line; the query sits at 0. Points at -3 and +3 are tied.
    base = np.array([[0.], [1.], [-1.], [2.], [3.], [-3.], [10.]], dtype=np.float32)
    query = np.array([[0.]], dtype=np.float32)
    k = 5
    gt_ids = np.array([[0, 1, 2, 3, 4]])      # ground truth picked +3
    tied_ids = np.array([[0, 1, 2, 3, 5]])    # we picked -3 instead: still correct
    wrong_ids = np.array([[0, 1, 2, 3, 6]])   # we picked 10: one mistake
    assert recall_at_k_from_ids(query, base, tied_ids, gt_ids, k=k)[0] == 1.0
    assert recall_at_k_from_ids(query, base, wrong_ids, gt_ids, k=k)[0] == pytest.approx(0.8)


def test_exact_distances_handles_minus_one():
    base = np.array([[0., 0.], [3., 4.]], dtype=np.float32)
    q = np.array([[0., 0.]], dtype=np.float32)
    d = exact_distances(q, base, np.array([[1, -1]]))
    assert d[0, 0] == 25.0 and np.isinf(d[0, 1])


def test_too_few_ground_truth_raises():
    with pytest.raises(ValueError):
        recall_at_k(np.zeros((1, 10)), np.zeros((1, 5)), k=10)
