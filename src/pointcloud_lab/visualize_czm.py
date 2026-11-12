from pathlib import Path

import laspy
import numpy as np
import matplotlib.pyplot as plt


# Test-002: virtual sensor at staircase center
origin = np.array([-25.677, -8.245])

min_range = 0.5
max_range = 80.0

num_rings = [2, 4, 4, 4]
num_sectors = [16, 32, 54, 32]

# Patchwork++ CZM zone boundaries
zone_bounds = [
    min_range,
    (7 * min_range + max_range) / 8,
    (3 * min_range + max_range) / 4,
    (min_range + max_range) / 2,
    max_range,
]

result_dir = Path("outputs/patchworkpp_test_002_center_r05")

ground = laspy.read(
    result_dir / "patchworkpp_ground.las"
).xyz

nonground = laspy.read(
    result_dir / "patchworkpp_nonground.las"
).xyz

fig, ax = plt.subplots(figsize=(11, 9))

# Classification result (XY projection)
ax.scatter(
    nonground[:, 0], nonground[:, 1],
    s=0.4, alpha=0.5, label="Non-ground"
)

ax.scatter(
    ground[:, 0], ground[:, 1],
    s=0.4, alpha=0.7, label="Ground"
)

# Plot CZM ring and sector boundaries
for zone in range(4):
    r_start = zone_bounds[zone]
    r_end = zone_bounds[zone + 1]

    ring_width = (r_end - r_start) / num_rings[zone]

    for i in range(num_rings[zone] + 1):
        radius = r_start + i * ring_width
        circle = plt.Circle(
            origin, radius,
            fill=False,
            edgecolor="black",
            linewidth=0.7,
            alpha=0.55,
        )
        ax.add_patch(circle)

    for j in range(num_sectors[zone]):
        theta = 2 * np.pi * j / num_sectors[zone]

        radii = np.array([r_start, r_end])
        x = origin[0] + radii * np.cos(theta)
        y = origin[1] + radii * np.sin(theta)

        ax.plot(
            x, y, color="black",
            linewidth=0.5, alpha=0.35,
        )

# Virtual sensor
ax.scatter(
    *origin, marker="x", s=100,
    color="red", label="Virtual Sensor",
    zorder=10
)

# Focus on the actual point cloud bounds
all_xy = np.vstack([ground[:, :2], nonground[:, :2]])
mins = all_xy.min(axis=0) - 1.0
maxs = all_xy.max(axis=0) + 1.0

ax.set_xlim(mins[0], maxs[0])
ax.set_ylim(mins[1], maxs[1])
ax.set_aspect("equal", adjustable="box")

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_title("Patchwork++ Ground Segmentation / CZM")
ax.legend()
ax.grid(alpha=0.15)

output = result_dir / "czm_overlay.png"
fig.savefig(output, dpi=200, bbox_inches="tight")
plt.close(fig)

print(f"Saved: {output}")
