"""Part 2 figures and tables from results/raw/part2_*.csv (and part1_*.csv).

Selection uses the tune half only; every reported number comes from the test
half of the same setting:
  * trade-off curve: the Pareto frontier of each index type (over all its
    build settings) chosen on tune recall/QPS, drawn at test recall/QPS;
  * QPS at recall r: the fastest setting whose tune recall >= r.

Outputs:
  results/figures/part2_tradeoff.png          both datasets side by side
  results/figures/part2_tradeoff_<ds>.png     one per dataset
  results/tables/part2_qps.{md,csv}           QPS at recall 0.90 / 0.95
  results/tables/part2_cost.{md,csv}          build time, size, bytes/vector at 0.95
"""
import argparse
import csv
import glob
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import DATASETS, results_dir, write_csv  # noqa: E402

INDEXES = ["hnsw", "ivfflat", "ivfpq"]
LABEL = {"hnsw": "HNSW", "ivfflat": "IVF-Flat", "ivfpq": "IVF-PQ", "exact": "Exact (Flat)"}
COLOR = {"hnsw": "#2a78d6", "ivfflat": "#eb6834", "ivfpq": "#1baf7a", "exact": "#52514e"}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
TARGETS = [0.90, 0.95]


def read_raw(out, pattern):
    rows = []
    for path in sorted(glob.glob(os.path.join(out, "raw", pattern))):
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def load_points(out):
    """{dataset: {index: [point]}}, a point holding its tune and test rows."""
    pairs = defaultdict(dict)
    for r in read_raw(out, "part2_*.csv"):
        key = (r["dataset"], r["index"], r["build_params"], r["search_params"])
        pairs[key][r["query_set"]] = r
    points = defaultdict(lambda: defaultdict(list))
    for (ds, idx, bp, sp), halves in pairs.items():
        if "tune" not in halves or "test" not in halves:
            continue
        tune, test = halves["tune"], halves["test"]
        points[ds][idx].append({
            "build": bp, "search": sp, "n_base": int(test["n_base"]),
            "tune_recall": float(tune["recall_at_10"]), "tune_qps": float(tune["qps"]),
            "recall": float(test["recall_at_10"]), "qps": float(test["qps"]),
            "build_s": float(test["build_s"]), "index_bytes": int(test["index_bytes"]),
            "bytes_per_vector": float(test["bytes_per_vector"]),
        })
    return points


def load_exact(out):
    exact = {}
    for r in read_raw(out, "part1_*.csv"):
        if r.get("mode") == "single" and r.get("index") == "Flat":
            exact[r["dataset"]] = {
                "recall": float(r["recall_at_10"]), "qps": float(r["qps"]),
                "build_s": float(r["build_s"]), "index_bytes": int(r["index_bytes"]),
                "bytes_per_vector": float(r["bytes_per_vector"]),
            }
    return exact


def pareto(pts):
    """Points not beaten on both tune recall and tune QPS, by increasing recall."""
    front, best_qps = [], -1.0
    for p in sorted(pts, key=lambda p: (-p["tune_recall"], -p["tune_qps"])):
        if p["tune_qps"] > best_qps:
            front.append(p)
            best_qps = p["tune_qps"]
    return front[::-1]


def select(pts, target):
    ok = [p for p in pts if p["tune_recall"] >= target]
    return max(ok, key=lambda p: p["tune_qps"]) if ok else None


def plot_dataset(ax, ds, pts_by_index, exact):
    for idx in INDEXES:
        pts = pts_by_index.get(idx, [])
        if not pts:
            continue
        c = COLOR[idx]
        ax.scatter([p["recall"] for p in pts], [p["qps"] for p in pts],
                   s=10, color=c, alpha=0.25, linewidths=0, zorder=2)
        front = pareto(pts)
        xs, ys = [p["recall"] for p in front], [p["qps"] for p in front]
        ax.plot(xs, ys, "-o", color=c, lw=1.6, ms=4.5, mec="white", mew=0.8,
                label=LABEL[idx], zorder=3)
    if ds in exact:
        e = exact[ds]
        ax.scatter([e["recall"]], [e["qps"]], marker="*", s=110, color=COLOR["exact"],
                   edgecolors="white", linewidths=0.8, label=LABEL["exact"], zorder=4)
    for t in TARGETS:
        ax.axvline(t, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax.set_yscale("log")
    ax.set_xlabel("recall@10 (test queries)", color=INK)
    ax.set_ylabel("queries per second (1 thread, log scale)", color=INK)
    n = pts_by_index[next(iter(pts_by_index))][0]["n_base"]
    name = {"sift": "SIFT", "glove": "GloVe-100"}.get(ds, ds)
    ax.set_title(f"{name} ({n / 1e6:.3g}M vectors)" if n >= 1e6 else f"{name} ({n // 1000}K vectors)",
                 color=INK)
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.legend(fontsize=8, frameon=False, loc="lower left")


def fmt_setting(p):
    return f"{p['build']}, {p['search']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = results_dir(args.quick)
    tables = os.path.join(out, "tables")
    os.makedirs(tables, exist_ok=True)

    points, exact = load_points(out), load_exact(out)
    datasets = [d for d in DATASETS if d in points]
    if not datasets:
        raise SystemExit("no part2 results found: run part2_bench.py first")

    # Figures
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.2 * len(datasets), 4.0), squeeze=False)
    for ax, ds in zip(axes[0], datasets):
        plot_dataset(ax, ds, points[ds], exact)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "figures", "part2_tradeoff.png"), dpi=200)
    plt.close(fig)
    for ds in datasets:
        fig, ax = plt.subplots(figsize=(5.2, 4.0))
        plot_dataset(ax, ds, points[ds], exact)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "figures", f"part2_tradeoff_{ds}.png"), dpi=200)
        plt.close(fig)

    # Tables
    qps_rows, cost_rows = [], []
    md_qps = ["| Dataset | Index | QPS @ 0.90 | setting (test recall) | QPS @ 0.95 | setting (test recall) |",
              "|---|---|---|---|---|---|"]
    md_cost = ["| Dataset | Index | Setting | Build time (s) | Index size (MB) | Bytes/vector | Test recall | QPS |",
               "|---|---|---|---|---|---|---|---|"]
    for ds in datasets:
        if ds in exact:
            e = exact[ds]
            md_qps.append(f"| {ds} | {LABEL['exact']} | {e['qps']:.1f} | recall 1.0 "
                          f"| {e['qps']:.1f} | recall 1.0 |")
            md_cost.append(f"| {ds} | {LABEL['exact']} | - | {e['build_s']:.2f} | "
                           f"{e['index_bytes'] / 1e6:.1f} | {e['bytes_per_vector']:.1f} | 1.0000 | "
                           f"{e['qps']:.1f} |")
            cost_rows.append({"dataset": ds, "index": "exact", "setting": "",
                              "reached_0.95": True, "build_s": e["build_s"],
                              "index_bytes": e["index_bytes"],
                              "bytes_per_vector": e["bytes_per_vector"],
                              "test_recall": e["recall"], "qps": e["qps"]})
        for idx in INDEXES:
            pts = points[ds].get(idx, [])
            if not pts:
                continue
            cells = []
            for t in TARGETS:
                p = select(pts, t)
                row = {"dataset": ds, "index": idx, "target_recall": t}
                if p:
                    row.update(setting=fmt_setting(p), test_recall=p["recall"], qps=p["qps"])
                    cells += [f"{p['qps']:.1f}", f"{fmt_setting(p)} ({p['recall']:.3f})"]
                else:
                    best = max(p["tune_recall"] for p in pts)
                    row.update(setting="", test_recall="", qps="")
                    cells += ["not reached", f"max tune recall {best:.3f}"]
                qps_rows.append(row)
            md_qps.append(f"| {ds} | {LABEL[idx]} | " + " | ".join(cells) + " |")

            p = select(pts, 0.95)
            reached = p is not None
            if not reached:  # report the most accurate setting instead, marked
                p = max(pts, key=lambda p: p["tune_recall"])
            mark = "" if reached else " (best; 0.95 not reached)"
            md_cost.append(f"| {ds} | {LABEL[idx]} | {fmt_setting(p)}{mark} | {p['build_s']:.1f} | "
                           f"{p['index_bytes'] / 1e6:.1f} | {p['bytes_per_vector']:.1f} | "
                           f"{p['recall']:.4f} | {p['qps']:.1f} |")
            cost_rows.append({"dataset": ds, "index": idx, "setting": fmt_setting(p),
                              "reached_0.95": reached, "build_s": p["build_s"],
                              "index_bytes": p["index_bytes"],
                              "bytes_per_vector": p["bytes_per_vector"],
                              "test_recall": p["recall"], "qps": p["qps"]})

    with open(os.path.join(tables, "part2_qps.md"), "w") as f:
        f.write("\n".join(md_qps) + "\n")
    with open(os.path.join(tables, "part2_cost.md"), "w") as f:
        f.write("\n".join(md_cost) + "\n")
    write_csv(os.path.join(tables, "part2_qps.csv"), qps_rows,
              ["dataset", "index", "target_recall", "setting", "test_recall", "qps"])
    write_csv(os.path.join(tables, "part2_cost.csv"), cost_rows,
              ["dataset", "index", "setting", "reached_0.95", "build_s", "index_bytes",
               "bytes_per_vector", "test_recall", "qps"])

    print("\n".join(md_qps))
    print()
    print("\n".join(md_cost))
    print(f"\nfigures in {os.path.join(out, 'figures')}, tables in {tables}")


if __name__ == "__main__":
    main()
