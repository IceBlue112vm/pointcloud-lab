import argparse
from pathlib import Path

import laspy
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Randomly subsample a LAS point cloud."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--count", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error("Input file does not exist.")

    if args.output.exists():
        parser.error("Output file already exists.")

    las = laspy.read(args.input)
    total = len(las.points)

    if not 0 < args.count <= total:
        parser.error("Invalid target point count.")

    rng = np.random.default_rng(args.seed)
    indices = np.sort(
        rng.choice(total, size=args.count, replace=False)
    )

    subset = laspy.LasData(las.header)
    subset.points = las.points[indices].copy()
    subset.write(args.output)

    print(f"Input:  {total:,} points")
    print(f"Output: {len(subset.points):,} points")
    print(f"Saved:  {args.output}")


if __name__ == "__main__":
    main()
