#!/usr/bin/env python
# fix_meta_invert.py  --  convert every transform_matrix from w2c to c2w
import json, numpy as np
from pathlib import Path
import argparse

def invert_pose(M: np.ndarray) -> list:
    """Invert a 4x4 pose matrix (world-to-camera → camera-to-world)."""
    return np.linalg.inv(M).tolist()

def process(meta_path: Path, backup=False):
    if backup:
        bak = meta_path.with_suffix(".json.bak")
        bak.write_bytes(meta_path.read_bytes())

    meta = json.loads(meta_path.read_text())
    for loc in meta["locations"]:
        M = np.asarray(loc["transform_matrix"], dtype=np.float64)
        loc["transform_matrix"] = invert_pose(M)

    meta_path.write_text(json.dumps(meta, indent=4))
    print("fixed", meta_path.relative_to(meta_path.parents[2]))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data", help="folder with train/ and val/")
    ap.add_argument("--backup", action="store_true", help="keep *.bak copies")
    args = ap.parse_args()

    for p in Path(args.root).glob("*/**/meta.json"):
        process(p, backup=args.backup)

if __name__ == "__main__":
    main()
