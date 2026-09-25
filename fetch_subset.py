"""
fetch_subset.py -- pull a tiny slice of ROCOv2 for smoke-testing the pipeline.

The HF mirror stores ~450 MB shards, but each has 100-row row groups, so we read
only row group 0 of one shard per split via ranged HTTP -- ~28 MB per split
instead of the full 17 GB. The output shards keep the mirror's exact schema
(image struct<bytes,path>, image_id, caption, cui list<string>) and filenames
that match the data.parquet globs in config.yaml, so every downstream script
runs unmodified.

Writes to a SEPARATE root (default D:/rrg_smoke) so toy shards can never be
confused with, or overwrite, a real download.

Usage:
    python fetch_subset.py
    python fetch_subset.py --train 10 --valid 8 --test 8
    python fetch_subset.py --root D:/rrg_smoke
"""
import argparse
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "datasets/eltorio/ROCOv2-radiology"
SHARD = {"train":      "data/train-00000-of-00027.parquet",
         "validation": "data/validation-00000-of-00006.parquet",
         "test":       "data/test-00000-of-00006.parquet"}


def grab(fs, split, n, out_dir):
    src = f"{REPO}/{SHARD[split]}"
    with fs.open(src, "rb") as f:
        pf = pq.ParquetFile(f)
        rg0 = pf.metadata.row_group(0)
        if n > rg0.num_rows:
            raise SystemExit(
                f"{split}: asked for {n} rows but row group 0 holds only "
                f"{rg0.num_rows}; widen the script to read more row groups.")
        print(f"{split:10s} reading row group 0 "
              f"({rg0.num_rows} rows, {rg0.total_byte_size/1e6:.1f} MB) ...")
        tbl = pf.read_row_group(0).slice(0, n)

    out = out_dir / f"{split}-00000-of-00001.parquet"
    pq.write_table(tbl, out)
    ids = tbl.column("image_id").to_pylist()
    print(f"{split:10s} wrote {tbl.num_rows} rows -> {out.name} "
          f"({out.stat().st_size/1e6:.1f} MB)  first_id={ids[0]}")
    return tbl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="D:/rrg_smoke")
    ap.add_argument("--train", type=int, default=10)
    ap.add_argument("--valid", type=int, default=8)
    ap.add_argument("--test",  type=int, default=8)
    args = ap.parse_args()

    out_dir = Path(args.root) / "raw" / "rocov2" / "data"
    out_dir.mkdir(parents=True, exist_ok=True)
    for sub in ("features", "checkpoints", "cache"):
        (Path(args.root) / sub).mkdir(parents=True, exist_ok=True)

    fs = HfFileSystem()
    first = None
    for split, n in (("train", args.train),
                     ("validation", args.valid),
                     ("test", args.test)):
        t = grab(fs, split, n, out_dir)
        first = first or t

    print("\nschema written (matches the mirror exactly):")
    print(first.schema)
    print(f"\nRoot ready: {args.root}")
    print(f"Next:  RRG_ROOT={args.root} python load_Dataset.py")


if __name__ == "__main__":
    main()
