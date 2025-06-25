#!/usr/bin/env python
# fix_meta_exact.py  --  remove exact uniform scale from every transform_matrix
import json, numpy as np
from pathlib import Path
import argparse

def descale(M: np.ndarray) -> list:
    """Return matrix with uniform scale removed."""
    R = M[:3, :3].astype(np.float64)
    s = np.linalg.norm(R[:, 0])          # exact scale
    R /= s                               # pure rotation
    T = M[:3, 3] / s                     # translate once
    out = np.eye(4)
    out[:3, :3] = R
    out[:3, 3]  = T
    return out.tolist()

def process(meta_path: Path, backup=False):
    if backup:
        bak = meta_path.with_suffix(".json.bak")
        bak.write_bytes(meta_path.read_bytes())

    meta = json.loads(meta_path.read_text())
    for loc in meta["locations"]:
        M = np.asarray(loc["transform_matrix"], dtype=np.float64)
        loc["transform_matrix"] = descale(M)

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
