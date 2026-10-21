"""Part 3 figure and table from results/raw/part3_sift_compression.csv.

Left panel:  recall@10 vs QPS for every m, plain (solid) and re-ranked
             (dashed), with IVF-Flat at the same nlist (Part 2) and exact search.
Right panel: memory (bytes per vector) vs the best test recall each setting
             reaches - the recall ceiling.

Settings for the table are chosen on the tune half and reported on the test
half, as in Part 2.

Outputs:
  results/figures/part3_compression.png
  results/tables/part3_compression.{md,csv}
"""
import argparse
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FormatStrFormatter, NullLocator  # noqa: E402

from common import results_dir, write_csv  # noqa: E402
from part2_report import (COLOR, GRID, INK, MUTED, TARGETS, load_exact,  # noqa: E402
                          read_raw, select)

DATASET = "sift"
# One hue, light -> dark, for the ordered PQ code size m.
M_COLORS = ["#86b6ee", "#4f8fe0", "#2a6fc4", "#173f78"]


def load_points(out):
    """{(index, build_params): [point]}, each point in nprobe order."""
    pairs = defaultdict(dict)
    for r in read_raw(out, "part3_*.csv"):
        pairs[(r["index"], r["build_params"], r["search_params"])][r["query_set"]] = r
    series = defaultdict(list)
    for (idx, bp, sp), h in pairs.items():
        if idx == "ivfflat":
            continue  # the full-precision control, loaded by load_ivfflat
        if "tune" in h and "test" in h:
            t, s = h["tune"], h["test"]
            series[(idx, bp)].append({
                "build": bp, "search": sp, "nprobe": int(sp.split("=")[1]),
                "m": int(dict(kv.split("=") for kv in bp.split(","))["m"]),
                "nlist": int(dict(kv.split("=") for kv in bp.split(","))["nlist"]),
                "tune_recall": float(t["recall_at_10"]), "tune_qps": float(t["qps"]),
                "recall": float(s["recall_at_10"]), "qps": float(s["qps"]),
                "build_s": float(s["build_s"]), "index_bytes": int(s["index_bytes"]),
                "bytes_per_vector": float(s["bytes_per_vector"]),
            })
    for pts in series.values():
        pts.sort(key=lambda p: p["nprobe"])
    return series


def load_ivfflat(out, nlist):
    """IVF-Flat at the same nlist, the full-precision reference.

    Prefers the control measured in the Part 3 run itself (same machine
    load); falls back to Part 2's sweep. Returns (points, source label).
    """
    for pattern, rows_filter, label in [
            ("part3_*.csv", lambda r: r["index"] == "ivfflat", "same run"),
            (f"part2_{DATASET}_ivfflat.csv", lambda r: True, "Part 2")]:
        pts = defaultdict(dict)
        for r in read_raw(out, pattern):
            if rows_filter(r) and r["build_params"] == f"nlist={nlist}":
                pts[r["search_params"]][r["query_set"]] = r
        out_pts = [{"nprobe": int(sp.split("=")[1]),
                    "tune_recall": float(h["tune"]["recall_at_10"]), "tune_qps": float(h["tune"]["qps"]),
                    "recall": float(h["test"]["recall_at_10"]), "qps": float(h["test"]["qps"]),
                    "build_s": float(h["test"]["build_s"]),
                    "bytes_per_vector": float(h["test"]["bytes_per_vector"])}
                   for sp, h in pts.items() if "test" in h and "tune" in h]
        if out_pts:
            return sorted(out_pts, key=lambda p: p["nprobe"]), label
    return [], ""


def style_axes(ax):
    ax.grid(True, which="major", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    out = results_dir(args.quick)
    tables = os.path.join(out, "tables")
    os.makedirs(tables, exist_ok=True)

    series = load_points(out)
    if not series:
        raise SystemExit("no part3 results found: run part3_compression.py first")
    ms = sorted({pts[0]["m"] for pts in series.values()})
    color = {m: M_COLORS[min(i, len(M_COLORS) - 1)] for i, m in enumerate(ms)}
    nlist = next(iter(series.values()))[0]["nlist"]
    flat, flat_src = load_ivfflat(out, nlist)
    exact = load_exact(out).get(DATASET)
    n_base = int(read_raw(out, "part3_*.csv")[0]["n_base"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.4, 4.0))

    # Left: recall vs QPS
    for (idx, _), pts in sorted(series.items(), key=lambda kv: (kv[1][0]["m"], kv[0][0])):
        m, rr = pts[0]["m"], "rerank" in idx
        ax1.plot([p["recall"] for p in pts], [p["qps"] for p in pts],
                 ls="--" if rr else "-", marker="s" if rr else "o", color=color[m],
                 lw=1.6, ms=4.5, mec="white", mew=0.8, zorder=3)
    if flat:
        ax1.plot([p["recall"] for p in flat], [p["qps"] for p in flat], "-o",
                 color=COLOR["ivfflat"], lw=1.6, ms=4.5, mec="white", mew=0.8, zorder=2)
    if exact:
        ax1.scatter([exact["recall"]], [exact["qps"]], marker="*", s=110,
                    color=COLOR["exact"], edgecolors="white", linewidths=0.8, zorder=4)
    for t in TARGETS:
        ax1.axvline(t, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax1.set_yscale("log")
    ax1.set_xlabel("recall@10 (test queries)", color=INK)
    ax1.set_ylabel("queries per second (1 thread, log scale)", color=INK)
    ax1.set_title("Speed vs recall (sweeping nprobe)", color=INK, fontsize=10)
    style_axes(ax1)

    # Right: memory vs best recall
    cluster = []  # re-ranked points that reach 0.95: they overlap IVF-Flat/exact
    for (idx, _), pts in series.items():
        m, rr = pts[0]["m"], "rerank" in idx
        best = max(p["recall"] for p in pts)
        x = pts[0]["bytes_per_vector"]
        ax2.scatter([x], [best], marker="s" if rr else "o",
                    s=46, color=color[m], edgecolors="white", linewidths=0.8, zorder=3)
        if not rr:
            ax2.annotate(f"m={m}", (x, best), xytext=(6, -3),
                         textcoords="offset points", fontsize=8, color=INK)
        elif best < TARGETS[-1]:
            ax2.annotate(f"m={m} + re-rank", (x, best), xytext=(-6, -3), ha="right",
                         textcoords="offset points", fontsize=8, color=INK)
        else:
            cluster.append((x, best, m))
    if cluster:
        x, y = min(c[0] for c in cluster), min(c[1] for c in cluster)
        ms_txt = ", ".join(str(c[2]) for c in sorted(cluster, key=lambda c: c[2]))
        ax2.annotate(f"m={ms_txt} + re-rank, IVF-Flat, exact", (x, y), xytext=(-24, 0),
                     ha="right", va="center", textcoords="offset points", fontsize=8, color=INK,
                     arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.8})
    if flat:
        ax2.scatter([flat[0]["bytes_per_vector"]], [max(p["recall"] for p in flat)], marker="o",
                    s=46, color=COLOR["ivfflat"], edgecolors="white", linewidths=0.8, zorder=3)
    if exact:
        ax2.scatter([exact["bytes_per_vector"]], [1.0], marker="*", s=110, color=COLOR["exact"],
                    edgecolors="white", linewidths=0.8, zorder=4)
    for t in TARGETS:
        ax2.axhline(t, color=MUTED, lw=0.8, ls=":", zorder=1)
    ax2.set_xscale("log")
    ax2.xaxis.set_major_locator(FixedLocator([8, 16, 32, 64, 128, 256, 512, 1024]))
    ax2.xaxis.set_major_formatter(FormatStrFormatter("%d"))
    ax2.xaxis.set_minor_locator(NullLocator())
    ax2.set_xlabel("index memory, bytes per vector (log scale)", color=INK)
    ax2.set_ylabel("best recall@10 reached (test queries)", color=INK)
    ax2.set_title("Memory vs recall ceiling", color=INK, fontsize=10)
    style_axes(ax2)

    handles = [Line2D([], [], color=color[m], lw=2, label=f"m = {m} ({m} B code)") for m in ms]
    handles += [Line2D([], [], color=MUTED, ls="-", marker="o", lw=1.6, label="IVF-PQ"),
                Line2D([], [], color=MUTED, ls="--", marker="s", lw=1.6,
                       label="IVF-PQ + re-rank top 100")]
    if flat:
        handles.append(Line2D([], [], color=COLOR["ivfflat"], marker="o", lw=1.6,
                              label=f"IVF-Flat (nlist={nlist}, {flat_src})"))
    if exact:
        handles.append(Line2D([], [], color=COLOR["exact"], marker="*", ls="", ms=10,
                              label="Exact (Flat)"))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    size = f"{n_base / 1e6:.3g}M" if n_base >= 1e6 else f"{n_base // 1000}K"
    fig.suptitle(f"PQ compression on SIFT ({size} vectors, nlist={nlist})", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(out, "figures", "part3_compression.png"), dpi=200)
    plt.close(fig)

    # Table
    md = ["| m | Variant | Bytes/vector | Build time (s) | Best test recall | "
          "QPS @ 0.90 (nprobe, test recall) | QPS @ 0.95 (nprobe, test recall) |",
          "|---|---|---|---|---|---|---|"]
    rows = []
    for (idx, _), pts in sorted(series.items(), key=lambda kv: (kv[1][0]["m"], kv[0][0])):
        p0 = pts[0]
        cells, row = [], {"m": p0["m"], "variant": idx, "bytes_per_vector": p0["bytes_per_vector"],
                          "build_s": p0["build_s"], "best_test_recall": max(p["recall"] for p in pts)}
        for t in TARGETS:
            s = select(pts, t)
            row[f"qps_at_{t}"] = s["qps"] if s else ""
            row[f"nprobe_at_{t}"] = s["nprobe"] if s else ""
            cells.append(f"{s['qps']:.1f} ({s['nprobe']}, {s['recall']:.3f})" if s else "not reached")
        rows.append(row)
        md.append(f"| {p0['m']} | {idx} | {p0['bytes_per_vector']:.1f} | {p0['build_s']:.1f} | "
                  f"{row['best_test_recall']:.4f} | " + " | ".join(cells) + " |")
    if flat:
        cells, row = [], {"m": "", "variant": f"ivfflat ({flat_src})",
                          "bytes_per_vector": flat[0]["bytes_per_vector"],
                          "build_s": flat[0]["build_s"],
                          "best_test_recall": max(p["recall"] for p in flat)}
        for t in TARGETS:
            s = select(flat, t)
            row[f"qps_at_{t}"] = s["qps"] if s else ""
            row[f"nprobe_at_{t}"] = s["nprobe"] if s else ""
            cells.append(f"{s['qps']:.1f} ({s['nprobe']}, {s['recall']:.3f})" if s else "not reached")
        rows.append(row)
        md.append(f"| - | IVF-Flat ({flat_src}) | {flat[0]['bytes_per_vector']:.1f} | "
                  f"{flat[0]['build_s']:.1f} | {row['best_test_recall']:.4f} | " + " | ".join(cells) + " |")
    with open(os.path.join(tables, "part3_compression.md"), "w") as f:
        f.write("\n".join(md) + "\n")
    write_csv(os.path.join(tables, "part3_compression.csv"), rows,
              ["m", "variant", "bytes_per_vector", "build_s", "best_test_recall",
               "qps_at_0.9", "nprobe_at_0.9", "qps_at_0.95", "nprobe_at_0.95"])
    print("\n".join(md))
    print(f"\nfigure in {os.path.join(out, 'figures')}, table in {tables}")


if __name__ == "__main__":
    main()
