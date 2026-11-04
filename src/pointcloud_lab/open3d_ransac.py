import argparse
from pathlib import Path

import laspy
import numpy as np
import open3d as o3d


def main():
    parser = argparse.ArgumentParser(
        description="Detect the largest plane in a LAS point cloud."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--threshold", type=float, default=0.03
    )
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error("Input file does not exist.")

    if args.output.exists():
        parser.error("Output file already exists.")

    # 1. Read LAS
    las = laspy.read(args.input)
    xyz = las.xyz

    print(f"Input: {len(xyz):,} points")

    # 2. Convert to Open3D PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # 3. Detect the largest plane
    o3d.utility.random.seed(42)

    plane_model, inliers = pcd.segment_plane(
        distance_threshold=args.threshold,
        ransac_n=3,
        num_iterations=1000,
        probability=1.0,
    )

    # 4. Calculate plane orientation
    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    dip = np.degrees(
        np.arccos(
            np.clip(abs(c) / np.linalg.norm(normal), 0, 1)
        )
    )

    print(f"Plane: {a:.6f}x + {b:.6f}y + {c:.6f}z + {d:.6f} = 0")
    print(f"Inliers: {len(inliers):,} ({len(inliers) / len(xyz):.2%})")
    print(f"Dip: {dip:.2f} degrees")

    # 5. Save detected plane points as LAS
    subset = laspy.LasData(las.header)
    subset.points = las.points[np.asarray(inliers)].copy()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    subset.write(args.output)

    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
