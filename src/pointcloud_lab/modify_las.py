import argparse

import laspy


def main():
    parser = argparse.ArgumentParser(
        description="Read, modify, and save a LAS point cloud."
    )
    parser.add_argument("input", help="Input LAS file")
    parser.add_argument("output", help="Output LAS file")
    args = parser.parse_args()

    # Read
    las = laspy.read(args.input)
    print(f"Loaded {len(las.points):,} points")

    # Modify
    las.z = las.z + 1.0

    # Write
    las.write(args.output)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
