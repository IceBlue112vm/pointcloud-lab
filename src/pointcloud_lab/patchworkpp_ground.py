import argparse
from pathlib import Path
from time import perf_counter

import laspy
import numpy as np
import pypatchworkpp as pwp


def save_subset(source, indices, path):
    if path.exists():
        raise FileExistsError(f"Output already exists: {path}")

    result = laspy.LasData(source.header)
    result.points = source.points[indices].copy()
    result.write(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--origin-ground",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Ground reference point for the virtual sensor",
    )
    parser.add_argument(
        "--min-range",
        type=float,
        default=2.7,
    )
    parser.add_argument(
        "--zone0-sectors",
        type=int,
        default=16,
    )
    args = parser.parse_args()

    las = laspy.read(args.input)
    xyz = las.xyz

    params = pwp.Parameters()
    params.min_range = args.min_range
    params.num_sectors_each_zone = [
        args.zone0_sectors, 32, 54, 32
    ]
    params.verbose = False
    params.enable_RNR = False

    # Virtual sensor position for this experiment.
    if args.origin_ground is not None:
        ground_point = np.array(args.origin_ground)

        origin = ground_point + np.array([
            0.0,
            0.0,
            params.sensor_height,
        ])
    else:
        origin = np.array([
            xyz[:, 0].min() - 5.0,
            (xyz[:, 1].min() + xyz[:, 1].max()) / 2.0,
            np.percentile(xyz[:, 2], 5) + params.sensor_height,
        ])

    # Patchwork++ input: XYZ + intensity.
    points = np.zeros((len(xyz), 4), dtype=np.float32)
    points[:, :3] = (xyz - origin).astype(np.float32)
    points[:, 3] = np.asarray(las.intensity, dtype=np.float32)

    algorithm = pwp.patchworkpp(params)

    start = perf_counter()
    algorithm.estimateGround(points)
    elapsed = perf_counter() - start

    ground = np.asarray(
        algorithm.getGroundIndices(), dtype=np.int64
    ).ravel()

    nonground = np.asarray(
        algorithm.getNongroundIndices(), dtype=np.int64
    ).ravel()

    # Detect points not returned by either category.
    indices = np.concatenate([ground, nonground])

    if np.any(indices < 0) or np.any(indices >= len(xyz)):
        raise RuntimeError("Invalid point indices returned.")

    if len(np.unique(indices)) != len(indices):
        raise RuntimeError("Duplicate classification indices.")

    classified = np.zeros(len(xyz), dtype=bool)
    classified[indices] = True
    unclassified = np.flatnonzero(~classified)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    save_subset(
        las, ground, args.output_dir / "patchworkpp_ground.las"
    )
    save_subset(
        las, nonground, args.output_dir / "patchworkpp_nonground.las"
    )

    if len(unclassified) > 0:
        save_subset(
            las,
            unclassified,
            args.output_dir / "patchworkpp_unclassified.las",
        )

    print(f"Input: {len(xyz):,}")
    print(f"Virtual sensor origin: {origin}")
    print(f"Ground: {len(ground):,}")
    print(f"Non-ground: {len(nonground):,}")
    print(f"Unclassified: {len(unclassified):,}")
    print(f"Processing time: {elapsed:.3f} seconds")


if __name__ == "__main__":
    main()
