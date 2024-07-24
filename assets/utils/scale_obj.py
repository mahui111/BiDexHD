import os
import argparse
from typing import List

from tqdm import tqdm


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scale object")
    parser.add_argument(
        "-d", "--dir", type=str, default="./", help="Directory of object files"
    )
    parser.add_argument("-s", "--scale", type=float, default=1.0, help="Scale factor")
    return parser.parse_args()


def scale(obj: List[str], scale_factor: float) -> List[str]:
    res = []

    for line in obj:
        if line.startswith("v "):
            x, y, z = line.split(" ")[1:]
            x, y, z = (
                float(x) * scale_factor,
                float(y) * scale_factor,
                float(z) * scale_factor,
            )
            line = f"v {x} {y} {z}\n"
        res.append(line)

    return res


def main():
    args = get_args()
    for obj in tqdm(os.listdir(args.dir)):
        if not obj.endswith(".obj"):
            continue
        with open(os.path.join(args.dir, obj), "r") as f:
            obj_content = f.readlines()
            scaled_content = scale(obj_content, args.scale)
        with open(os.path.join(args.dir, "scaled_" + obj), "w") as f:
            f.writelines(scaled_content)


if __name__ == "__main__":
    main()
