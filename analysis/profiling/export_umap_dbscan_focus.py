#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

FAMILY_MARKERS = {
    "go": "o",
    "python": "s",
    "node": "^",
    "twin": "D",
    "synthetic": "P",
}
PREFERENCE_MARKERS = {
    "x86-preferred": "o",
    "arm-preferred": "^",
    "architecture-independent": "s",
}
SELECTED_LABELS = {
    "amd_faster",
    "arm_faster",
    "filehandle",
    "thread",
    "linpack",
    "primenumber",
    "twin-primenumber",
    "twin-chacha20",
    "twin-readmemory",
    "compression-node",
    "graph-mst-py",
    "json-dumps-node",
    "readdisk",
}


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def save(fig, stem: Path):
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"), dpi=180, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def cluster_colors(rows):
    # Keep the same visual convention used by the original UMAP export.
    preferred = {
        -1: "tab:blue",
        0: "tab:orange",
        1: "tab:green",
        2: "tab:red",
        3: "tab:purple",
        4: "tab:brown",
    }
    clusters = sorted({int(r["cluster"]) for r in rows})
    fallback = list(plt.get_cmap("tab20").colors)
    result = {}
    for idx, cluster in enumerate(clusters):
        result[cluster] = preferred.get(cluster, fallback[idx % len(fallback)])
    return result


def compute_bounds(rows, padding_ratio: float):
    xs = [float(r["umap1"]) for r in rows]
    ys = [float(r["umap2"]) for r in rows]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    xspan = max(xmax - xmin, 1e-9)
    yspan = max(ymax - ymin, 1e-9)
    return (
        (xmin - padding_ratio * xspan, xmax + padding_ratio * xspan),
        (ymin - padding_ratio * yspan, ymax + padding_ratio * yspan),
    )


def cluster_handles(colors):
    handles = []
    for cluster in sorted(colors):
        marker = "x" if cluster == -1 else "o"
        label = "noise" if cluster == -1 else f"cluster {cluster}"
        handles.append(
            Line2D(
                [0], [0],
                marker=marker,
                linestyle="",
                color=colors[cluster],
                markerfacecolor=("none" if marker == "x" else colors[cluster]),
                markeredgecolor=colors[cluster],
                markersize=8,
                label=label,
            )
        )
    return handles


def family_handles(rows):
    present = {r["function_family"] for r in rows}
    return [
        Line2D(
            [0], [0],
            marker=FAMILY_MARKERS[f],
            linestyle="",
            color="black",
            markerfacecolor="lightgray",
            markeredgecolor="black",
            markersize=8,
            label=f,
        )
        for f in FAMILY_MARKERS
        if f in present
    ]


def preference_handles(rows):
    present = {r["ground_truth_label"] for r in rows}
    return [
        Line2D(
            [0], [0],
            marker=PREFERENCE_MARKERS[p],
            linestyle="",
            color="black",
            markerfacecolor="lightgray",
            markeredgecolor="black",
            markersize=8,
            label=p,
        )
        for p in PREFERENCE_MARKERS
        if p in present
    ]


def draw_points(ax, rows, colors, mode):
    for row in rows:
        cluster = int(row["cluster"])
        if cluster == -1:
            marker = "x"
        elif mode in {"family", "labels"}:
            marker = FAMILY_MARKERS[row["function_family"]]
        elif mode == "preference":
            marker = PREFERENCE_MARKERS[row["ground_truth_label"]]
        else:
            marker = "o"

        ax.scatter(
            float(row["umap1"]),
            float(row["umap2"]),
            color=colors[cluster],
            marker=marker,
            s=68,
            alpha=0.85,
        )

        if mode == "labels" and row["function_name"] in SELECTED_LABELS:
            ax.annotate(
                row["function_name"],
                (float(row["umap1"]), float(row["umap2"])),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )


def configure_axis(ax, title, bounds=None):
    ax.set_title(title)
    ax.set_xlabel("UMAP-1")
    ax.set_ylabel("UMAP-2")
    ax.grid(True, alpha=0.2)
    if bounds is not None:
        xlim, ylim = bounds
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)


def add_legends(fig, rows, colors, mode):
    handles = cluster_handles(colors)
    labels = [h.get_label() for h in handles]

    if mode in {"family", "labels"}:
        extra = family_handles(rows)
        handles += extra
        labels += [f"family: {h.get_label()}" for h in extra]
    elif mode == "preference":
        extra = preference_handles(rows)
        handles += extra
        labels += [f"ground truth: {h.get_label()}" for h in extra]

    for handle, label in zip(handles, labels):
        handle.set_label(label)

    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.005, 0.5),
        fontsize=8,
        frameon=True,
    )


def plot_full_focus(rows, focus_rows, title, stem, mode, focus_bounds):
    colors = cluster_colors(rows)
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    draw_points(axes[0], rows, colors, mode)
    configure_axis(axes[0], "Full UMAP embedding")

    focus_clusters = sorted({int(r["cluster"]) for r in focus_rows})
    focus_text = ", ".join("noise" if c == -1 else f"C{c}" for c in focus_clusters)
    draw_points(axes[1], focus_rows, colors, mode)
    configure_axis(axes[1], f"Focus view: {focus_text}", focus_bounds)

    fig.suptitle(title, fontsize=14)
    add_legends(fig, rows, colors, mode)
    fig.subplots_adjust(right=0.83, wspace=0.22, top=0.90)
    save(fig, stem)


def plot_focus_only(focus_rows, all_rows, title, stem, mode, focus_bounds):
    colors = cluster_colors(all_rows)
    fig, ax = plt.subplots(figsize=(11, 8))
    draw_points(ax, focus_rows, colors, mode)
    configure_axis(ax, title, focus_bounds)
    add_legends(fig, focus_rows, colors, mode)
    fig.subplots_adjust(right=0.80)
    save(fig, stem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--umap-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument(
        "--exclude-cluster",
        action="append",
        type=int,
        default=None,
        help="DBSCAN cluster(s) omitted only from the focus panel. Default: 0.",
    )
    ap.add_argument("--padding-ratio", type=float, default=0.08)
    args = ap.parse_args()

    umap_dir = args.umap_dir.resolve()
    output_dir = args.output_dir.resolve()
    figures = output_dir / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    coordinates = umap_dir / "dbscan-paper5-robust-cosine-ms4-q80-umap-coordinates.csv"
    rows = read_csv(coordinates)
    if not rows:
        raise RuntimeError(f"No rows in {coordinates}")

    exclude = set(args.exclude_cluster if args.exclude_cluster is not None else [0])
    focus_rows = [r for r in rows if int(r["cluster"]) not in exclude]
    if not focus_rows:
        raise RuntimeError("Focus selection is empty")

    focus_bounds = compute_bounds(focus_rows, args.padding_ratio)

    export_rows = []
    for row in rows:
        export_rows.append(
            {
                **row,
                "included_in_focus": int(row["cluster"]) not in exclude,
            }
        )
    write_csv(output_dir / "dbscan-umap-focus-coordinates.csv", export_rows)
    write_csv(
        output_dir / "dbscan-umap-focus-config.csv",
        [
            {
                "excluded_clusters_from_focus": ";".join(str(x) for x in sorted(exclude)),
                "focus_point_count": len(focus_rows),
                "full_point_count": len(rows),
                "padding_ratio": args.padding_ratio,
                "focus_xlim_min": focus_bounds[0][0],
                "focus_xlim_max": focus_bounds[0][1],
                "focus_ylim_min": focus_bounds[1][0],
                "focus_ylim_max": focus_bounds[1][1],
                "note": "Full view always retains every function; exclusions affect only the complementary focus panel.",
            }
        ],
    )

    prefix = "dbscan-paper5-robust-cosine-ms4-q80-umap"
    modes = [
        ("clusters", "DBSCAN UMAP — clusters", "clusters"),
        ("cluster-family", "DBSCAN UMAP — cluster × function family", "family"),
        ("cluster-preference", "DBSCAN UMAP — cluster × architectural preference", "preference"),
        ("selected-labels", "DBSCAN UMAP — selected labels", "labels"),
    ]

    for suffix, title, mode in modes:
        plot_full_focus(
            rows,
            focus_rows,
            title + " — full + focus",
            figures / f"{prefix}-{suffix}-full-focus",
            mode,
            focus_bounds,
        )
        plot_focus_only(
            focus_rows,
            rows,
            title + " — focus view",
            figures / f"{prefix}-{suffix}-focus",
            mode,
            focus_bounds,
        )

    print(
        f"points={len(rows)} focus_points={len(focus_rows)} "
        f"excluded_clusters={sorted(exclude)} "
        f"focus_xlim=({focus_bounds[0][0]:.4f},{focus_bounds[0][1]:.4f}) "
        f"focus_ylim=({focus_bounds[1][0]:.4f},{focus_bounds[1][1]:.4f}) "
        f"output={output_dir}"
    )


if __name__ == "__main__":
    main()
