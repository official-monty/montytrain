#!/usr/bin/env python3
"""
heatmap_pc_tc.py — Build a 2D heatmap from CSV rows of: piece_count, threat_count

- X axis: threat count (default range 0..256)
- Y axis: piece count  (default range 0..32)

Usage:
  python heatmap_pc_tc.py data.csv --out heatmap.png
  python heatmap_pc_tc.py data.csv --out heatmap.png --log
  python heatmap_pc_tc.py data.csv --header   # if the first row is a header
  python heatmap_pc_tc.py data.csv --xmin 0 --xmax 256 --ymin 0 --ymax 32

CSV format:
  Each line: <piece_count>,<threat_count>
  By default there is no header. If your file has a header row, pass --header.
"""

import argparse
import csv
import gzip
import io
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors


def open_maybe_gzip(path: Path):
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", newline="")
    return open(path, "r", encoding="utf-8", newline="")


def build_counts_matrix(csv_path: Path, xmin: int, xmax: int, ymin: int, ymax: int,
                        sep: str = ",", has_header: bool = False,
                        clamp: bool = False):
    nx = xmax - xmin + 1
    ny = ymax - ymin + 1
    counts = np.zeros((ny, nx), dtype=np.int64)  # index as [y, x] => [piece, threat]

    with open_maybe_gzip(csv_path) as f:
        reader = csv.reader(f, delimiter=sep)
        if has_header:
            next(reader, None)
        bad_rows = 0
        for row in reader:
            if not row:
                continue
            try:
                pc = int(row[0])
                tc = int(row[1])
            except (ValueError, IndexError):
                bad_rows += 1
                continue

            if ymin <= pc <= ymax and xmin <= tc <= xmax:
                counts[pc - ymin, tc - xmin] += 1
            elif clamp:
                pc_clamped = min(max(pc, ymin), ymax)
                tc_clamped = min(max(tc, xmin), xmax)
                counts[pc_clamped - ymin, tc_clamped - xmin] += 1
            else:
                # skip out-of-range rows
                continue

    if bad_rows:
        print(f"[warn] Skipped {bad_rows} unparsable row(s).", file=sys.stderr)
    return counts


def plot_heatmap(counts: np.ndarray, xmin: int, xmax: int, ymin: int, ymax: int,
                 out_path: Path, title: str, log_scale: bool, dpi: int, grid: bool):
    # Align integer bins so cell centers fall on integer values.
    extent = [xmin - 0.5, xmax + 0.5, ymin - 0.5, ymax + 0.5]

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)

    if log_scale:
        data = np.ma.masked_less_equal(counts, 0)  # mask zeros for LogNorm
        norm = mcolors.LogNorm(vmin=1, vmax=counts.max() if counts.max() > 0 else 1)
        img = ax.imshow(data, origin="lower", extent=extent, aspect="auto", norm=norm)
    else:
        img = ax.imshow(counts, origin="lower", extent=extent, aspect="auto")

    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Frequency (count of rows)")

    ax.set_xlabel("Threat count")
    ax.set_ylabel("Piece count")
    if title:
        ax.set_title(title)

    # Ticks at sensible intervals
    ax.xaxis.set_major_locator(mticker.MultipleLocator(16))
    ax.xaxis.set_minor_locator(mticker.MultipleLocator(4))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(4))
    ax.yaxis.set_minor_locator(mticker.MultipleLocator(1))

    if grid:
        ax.grid(which="major", linewidth=0.6, alpha=0.5)
        ax.grid(which="minor", linewidth=0.2, alpha=0.25)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="Generate a 2D heatmap from piece/threat CSV.")
    ap.add_argument("csv", type=Path, help="Input CSV (optionally .gz)")
    ap.add_argument("--out", type=Path, default=Path("heatmap.png"), help="Output image path")
    ap.add_argument("--sep", default=",", help="CSV delimiter (default: ,)")
    ap.add_argument("--header", action="store_true", help="Set if the CSV has a header row")
    ap.add_argument("--xmin", type=int, default=0)
    ap.add_argument("--xmax", type=int, default=72)
    ap.add_argument("--ymin", type=int, default=0)
    ap.add_argument("--ymax", type=int, default=32)
    ap.add_argument("--clamp", action="store_true", help="Clamp out-of-range values into the range (else skip)")
    ap.add_argument("--log", action="store_true", help="Log color scale (good for heavy-tailed counts)")
    ap.add_argument("--title", default="", help="Plot title")
    ap.add_argument("--dpi", type=int, default=150, help="Output image DPI")
    ap.add_argument("--grid", action="store_true", help="Draw major/minor gridlines")
    args = ap.parse_args()

    counts = build_counts_matrix(
        args.csv, args.xmin, args.xmax, args.ymin, args.ymax,
        sep=args.sep, has_header=args.header, clamp=args.clamp
    )

    plot_heatmap(
        counts, args.xmin, args.xmax, args.ymin, args.ymax,
        args.out, args.title, args.log, args.dpi, args.grid
    )
    total = int(counts.sum())
    nz = int((counts > 0).sum())
    print(f"Saved {args.out} | total rows counted: {total} | non-empty cells: {nz}")


if __name__ == "__main__":
    main()
