from collections import Counter
from pathlib import Path

import laspy
import numpy as np


RESULT_DIR = Path("outputs/patchworkpp_test_002_center_r05")

ORIGIN_XY = np.array([-25.677, -8.245])
MIN_RANGE = 0.5
MAX_RANGE = 80.0

RINGS = [2, 4, 4, 4]
SECTORS = [16, 32, 54, 32]

BOUNDS = [
    MIN_RANGE,
    (7 * MIN_RANGE + MAX_RANGE) / 8,
    (3 * MIN_RANGE + MAX_RANGE) / 4,
    (MIN_RANGE + MAX_RANGE) / 2,
    MAX_RANGE,
]


def count_patches(path):
    xyz = laspy.read(path).xyz

    dx = xyz[:, 0] - ORIGIN_XY[0]
    dy = xyz[:, 1] - ORIGIN_XY[1]

    radius = np.hypot(dx, dy)
    theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)

    counts = Counter()

    for zone in range(4):
        lower, upper = BOUNDS[zone], BOUNDS[zone + 1]
        mask = (radius >= lower) & (radius < upper)

        ring_width = (upper - lower) / RINGS[zone]
        sector_width = 2 * np.pi / SECTORS[zone]

        ring_ids = np.floor(
            (radius[mask] - lower) / ring_width
        ).astype(int)

        sector_ids = np.floor(
            theta[mask] / sector_width
        ).astype(int)

        counts.update(
            (zone, int(ring), int(sector))
            for ring, sector in zip(ring_ids, sector_ids)
        )

    outside = int(np.count_nonzero(
        (radius < MIN_RANGE) | (radius >= MAX_RANGE)
    ))

    return counts, outside


ground, ground_outside = count_patches(
    RESULT_DIR / "patchworkpp_ground.las"
)

nonground, nonground_outside = count_patches(
    RESULT_DIR / "patchworkpp_nonground.las"
)

patches = set(ground) | set(nonground)

print("Zone Ring Sector | Total | Ground | Non-ground | Ground %")
print("-" * 67)

for patch in sorted(
    patches, key=lambda p: nonground[p], reverse=True
)[:20]:
    g = ground[patch]
    ng = nonground[patch]
    total = g + ng

    print(
        f"{patch[0]:4} {patch[1]:4} {patch[2]:6} | "
        f"{total:5,} | {g:6,} | {ng:10,} | "
        f"{100 * g / total:6.2f}%"
    )

print(f"\nOutside CZM: Ground={ground_outside}, "
      f"Non-ground={nonground_outside}")
