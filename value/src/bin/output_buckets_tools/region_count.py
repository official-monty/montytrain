#!/usr/bin/env python3
"""
interactive_region_count_pc_tc.py — interactively count rows in rectangles of (threat, piece)

Input CSV rows: piece_count,threat_count
- X (columns) = threat count
- Y (rows)    = piece count

Run:
  python interactive_region_count_pc_tc.py data.csv
  # if your CSV has a header row:
  python interactive_region_count_pc_tc.py data.csv --header
  # adjust histogram bounds if needed:
  python interactive_region_count_pc_tc.py data.csv --xmin 0 --xmax 256 --ymin 0 --ymax 32

At the prompt, enter rectangles in any of these forms (inclusive ranges):
  x1:x2,y1:y2
  x1,x2,y1,y2
  x1 x2 y1 y2
Commands:  help / ?   show help
           q | quit | exit   leave the program
"""

import argparse
import csv
import gzip
import io
from pathlib import Path
import sys
from typing import Tuple, Optional

import numpy as np


def open_maybe_gzip(path: Path):
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", newline="")
    return open(path, "r", encoding="utf-8", newline="")


def build_counts_matrix(csv_path: Path, xmin: int, xmax: int, ymin: int, ymax: int,
                        sep: str = ",", has_header: bool = False) -> np.ndarray:
    nx = xmax - xmin + 1
    ny = ymax - ymin + 1
    counts = np.zeros((ny, nx), dtype=np.int64)  # [piece, threat]
    bad = 0

    with open_maybe_gzip(csv_path) as f:
        reader = csv.reader(f, delimiter=sep)
        if has_header:
            next(reader, None)
        for row in reader:
            if not row:
                continue
            try:
                pc = int(row[0])
                tc = int(row[1])
            except (ValueError, IndexError):
                bad += 1
                continue
            if ymin <= pc <= ymax and xmin <= tc <= xmax:
                counts[pc - ymin, tc - xmin] += 1

    if bad:
        print(f"[warn] Skipped {bad} unparsable row(s).", file=sys.stderr)
    return counts


def prefix_sum_2d(a: np.ndarray) -> np.ndarray:
    return a.cumsum(axis=0).cumsum(axis=1)


def rect_sum(prefix: np.ndarray, x1: int, x2: int, y1: int, y2: int,
             xmin: int, xmax: int, ymin: int, ymax: int) -> Tuple[int, Tuple[int,int,int,int], bool]:
    # Clamp to histogram bounds (inclusive). Empty => 0.
    X1c = max(x1, xmin)
    X2c = min(x2, xmax)
    Y1c = max(y1, ymin)
    Y2c = min(y2, ymax)
    clamped = (X1c != x1 or X2c != x2 or Y1c != y1 or Y2c != y2)

    if X1c > X2c or Y1c > Y2c:
        return 0, (X1c, X2c, Y1c, Y2c), clamped

    X1 = X1c - xmin
    X2 = X2c - xmin
    Y1 = Y1c - ymin
    Y2 = Y2c - ymin

    S = prefix
    total = S[Y2, X2]
    if X1 > 0:
        total -= S[Y2, X1 - 1]
    if Y1 > 0:
        total -= S[Y1 - 1, X2]
    if X1 > 0 and Y1 > 0:
        total += S[Y1 - 1, X1 - 1]
    return int(total), (X1c, X2c, Y1c, Y2c), clamped


def parse_rect_line(line: str) -> Optional[Tuple[int,int,int,int]]:
    s = line.strip()
    if not s:
        return None
    if s.lower() in {"q", "quit", "exit"}:
        raise SystemExit
    if s.lower() in {"help", "?"}:
        print("Enter a rectangle as one of:\n"
              "  x1:x2,y1:y2   (threat, piece)\n"
              "  x1,x2,y1,y2\n"
              "  x1 x2 y1 y2\n"
              "All ranges are inclusive. Type q to quit.")
        return None

    def to_ints(parts):
        try:
            return [int(p) for p in parts]
        except ValueError:
            raise ValueError("Non-integer value found.")

    if "," in s and ":" in s and s.count(",") == 1:
        # x1:x2,y1:y2
        xpart, ypart = s.split(",", 1)
        x1, x2 = to_ints(xpart.split(":"))
        y1, y2 = to_ints(ypart.split(":"))
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        return x1, x2, y1, y2

    if "," in s:
        parts = to_ints([p.strip() for p in s.split(",")])
        if len(parts) != 4:
            raise ValueError("Expected 4 comma-separated integers.")
        x1, x2, y1, y2 = parts
        if x1 > x2: x1, x2 = x2, x1
        if y1 > y2: y1, y2 = y2, y1
        return x1, x2, y1, y2

    # space-separated
    parts = to_ints(s.split())
    if len(parts) != 4:
        raise ValueError("Expected 4 integers: x1 x2 y1 y2.")
    x1, x2, y1, y2 = parts
    if x1 > x2: x1, x2 = x2, x1
    if y1 > y2: y1, y2 = y2, y1
    return x1, x2, y1, y2


def main():
    ap = argparse.ArgumentParser(description="Interactively count rows in (threat, piece) rectangles.")
    ap.add_argument("csv", type=Path, help="Input CSV (optionally .gz)")
    ap.add_argument("--sep", default=",", help="CSV delimiter (default: ,)")
    ap.add_argument("--header", action="store_true", help="Set if CSV has a header row")
    ap.add_argument("--xmin", type=int, default=0)
    ap.add_argument("--xmax", type=int, default=256)
    ap.add_argument("--ymin", type=int, default=0)
    ap.add_argument("--ymax", type=int, default=32)
    args = ap.parse_args()

    counts = build_counts_matrix(args.csv, args.xmin, args.xmax, args.ymin, args.ymax,
                                 sep=args.sep, has_header=args.header)
    total = int(counts.sum())
    S = prefix_sum_2d(counts)

    print(f"Loaded {args.csv}")
    print(f"Histogram bounds: threat [{args.xmin}..{args.xmax}], piece [{args.ymin}..{args.ymax}]")
    print(f"Total rows considered: {total}")
    print("Enter rectangles as x1:x2,y1:y2 (threat, piece). Type 'help' for more. 'q' to quit.\n")

    while True:
        try:
            line = input("> ")
        except EOFError:
            break
        if line is None:
            continue
        try:
            rect = parse_rect_line(line)
        except ValueError as e:
            print(f"[error] {e}")
            continue
        except SystemExit:
            print("bye.")
            return
        if rect is None:
            continue

        x1, x2, y1, y2 = rect
        n, (X1c, X2c, Y1c, Y2c), clamped = rect_sum(S, x1, x2, y1, y2,
                                                    args.xmin, args.xmax, args.ymin, args.ymax)
        frac = (n / total) if total > 0 else 0.0
        clamp_note = " [clamped to bounds]" if clamped else ""
        print(f"x=[{x1}..{x2}], y=[{y1}..{y2}] -> count={n}  frac={frac:.6f}"
              f" (evaluated on x=[{X1c}..{X2c}], y=[{Y1c}..{Y2c}]){clamp_note}")

if __name__ == "__main__":
    main()