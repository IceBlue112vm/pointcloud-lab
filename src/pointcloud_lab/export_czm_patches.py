from pathlib import Path

import laspy
import numpy as np


source = laspy.read("data/stairs_and_ground_200k.las")
xyz = source.xyz

origin = np.array([-25.677, -8.245])

dx = xyz[:, 0] - origin[0]
dy = xyz[:, 1] - origin[1]

radius = np.hypot(dx, dy)
theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)

# Zone 0, Ring 1, min_range = 0.5
min_range = 0.5
zone_end = (7 * min_range + 80.0) / 8
ring_start = (min_range + zone_end) / 2
ring_end = zone_end

output_dir = Path("outputs/czm_debug")
output_dir.mkdir(parents=True, exist_ok=True)

for sector in (0, 1):
    sector_start = sector * 2 * np.pi / 16
    sector_end = (sector + 1) * 2 * np.pi / 16

    mask = (
        (radius >= ring_start)
        & (radius < ring_end)
        & (theta >= sector_start)
        & (theta < sector_end)
    )

    indices = np.flatnonzero(mask)

    subset = laspy.LasData(source.header)
    subset.points = source.points[indices].copy()

    output = output_dir / f"z0_r1_s{sector}.las"
    subset.write(output)

    print(f"Sector {sector}: {len(indices):,} points")
    print(f"Saved: {output}")
