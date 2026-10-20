"""Download and load the ann-benchmarks datasets (SIFT-1M and GloVe-100)."""
import os
import sys
import urllib.error
import urllib.request

import h5py
import numpy as np

from common import DATA_DIR, QUICK_N_BASE

URLS = {
    "sift": "http://ann-benchmarks.com/sift-128-euclidean.hdf5",
    "glove": "http://ann-benchmarks.com/glove-100-angular.hdf5",
}

N_TUNE = 5000  # first 5,000 queries: tuning; last 5,000: final numbers


def _fetch(url, tmp, timeout=60, retries=10):
    """Download url into tmp, resuming after stalls or dropped connections."""
    # The server returns 403 for Python's default "Python-urllib" User-Agent.
    headers = {"User-Agent": "Mozilla/5.0"}
    total = None
    for attempt in range(retries):
        done = os.path.getsize(tmp) if os.path.exists(tmp) else 0
        if total is not None and done >= total:
            return
        req = urllib.request.Request(url, headers={**headers, "Range": f"bytes={done}-"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                if resp.status != 206:  # server ignored Range: start over
                    done = 0
                total = done + int(resp.headers["Content-Length"])
                with open(tmp, "ab" if done else "wb") as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        sys.stdout.write(f"\r  {done / 1e6:7.1f} / {total / 1e6:.1f} MB "
                                         f"({100 * done / total:5.1f}%)")
                        sys.stdout.flush()
            if done >= total:
                return
        except (OSError, urllib.error.URLError) as e:  # timeouts are OSErrors
            print(f"\n  connection problem ({e}); resuming (attempt {attempt + 2}/{retries})")
    raise RuntimeError(f"could not download {url}")


def download(name):
    """Download the HDF5 file once and return its local path."""
    os.makedirs(DATA_DIR, exist_ok=True)
    url = URLS[name]
    path = os.path.join(DATA_DIR, os.path.basename(url))
    if not os.path.exists(path):
        print(f"Downloading {url}")
        tmp = path + ".part"
        _fetch(url, tmp)
        os.replace(tmp, path)
        print()
    return path


def normalize(x):
    """L2-normalise rows. Zero rows are left as zero instead of becoming NaN."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return np.ascontiguousarray(x / norms, dtype=np.float32)


def load_dataset(name, quick=False):
    """Return dict with base, queries, and the file's gt_ids / gt_dists.

    GloVe is L2-normalised so that Euclidean distance ranks neighbours exactly
    like angular distance: ||a-b||^2 = 2 - 2 cos(a, b) for unit vectors.

    quick=True keeps only the first QUICK_N_BASE base vectors. The file's
    ground truth refers to the full base set, so it is set to None; use
    load_exact() (computed by part1_exact.py) instead.
    """
    path = download(name)
    with h5py.File(path, "r") as f:
        base = f["train"][:QUICK_N_BASE] if quick else f["train"][:]
        base = np.ascontiguousarray(base, dtype=np.float32)
        queries = np.ascontiguousarray(f["test"][:], dtype=np.float32)
        gt_ids = None if quick else f["neighbors"][:].astype(np.int64)
        gt_dists = None if quick else f["distances"][:].astype(np.float64)
    if name == "glove":
        base = normalize(base)
        queries = normalize(queries)
    return {"base": base, "queries": queries, "gt_ids": gt_ids, "gt_dists": gt_dists}


def tune_split(x):
    return x[:N_TUNE]


def test_split(x):
    return x[N_TUNE:]


def exact_paths(name, quick):
    tag = f"{name}_q{QUICK_N_BASE}" if quick else name
    return (os.path.join(DATA_DIR, f"{tag}_exact_I.npy"),
            os.path.join(DATA_DIR, f"{tag}_exact_D.npy"))


def load_exact(name, quick=False):
    """Exact top-100 (ids, float32 squared L2) for all 10,000 queries."""
    i_path, d_path = exact_paths(name, quick)
    if not os.path.exists(i_path):
        raise FileNotFoundError(f"{i_path} missing: run part1_exact.py first")
    return np.load(i_path), np.load(d_path)
