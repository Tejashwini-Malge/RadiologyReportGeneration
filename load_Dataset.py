import glob
from pathlib import Path

import pyarrow.dataset as ds

from rrg_config import load_config, enable_utf8_stdout

CFG = load_config()
enable_utf8_stdout()
ROCOV2_DIR = CFG["paths"]["rocov2_dir"]
DATA = CFG["data"]

# keys that may be absent from config.yaml; must match extract_features.ID_COL
DEFAULTS = {"image_col": "image", "caption_col": "caption",
            "cui_col": "cui", "image_id_col": "image_id"}


def human(n):
    for u in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {u}"
        n /= 1024
    return f"{n:.2f} PB"


def files_for(pattern):
    return sorted(glob.glob(str(Path(ROCOV2_DIR) / pattern)))


def main():
    print(f"ROCOv2 dir: {ROCOV2_DIR}\n")

    grand_rows, grand_bytes = 0, 0
    first_schema = None

    for split, pattern in DATA["parquet"].items():
        fs = files_for(pattern)
        if not fs:
            print(f"[{split}] NO files match {pattern}  <-- fix config.data.parquet.{split}\n")
            continue

        dataset = ds.dataset(fs, format="parquet")
        n_rows = dataset.count_rows()
        cols = dataset.schema.names
        size = sum(Path(f).stat().st_size for f in fs)
        grand_rows += n_rows
        grand_bytes += size
        first_schema = first_schema or dataset.schema

        print(f"[{split}]  {len(fs)} shards,  {n_rows:,} rows,  {human(size)}")
        print(f"    columns : {cols}")

        # peek one row to reveal real image encoding + caption + cui
        row = next(dataset.scanner(batch_size=1).to_batches()).to_pandas().iloc[0]
        _describe_row(row, cols)
        print()

    print(f"TOTAL: {grand_rows:,} rows, {human(grand_bytes)} on disk")
    print("(expected ~79,789 rows total)\n")

    if first_schema is not None:
        _check_cols(first_schema.names)


def _describe_row(row, cols):
    ic, cc, cui = DATA["image_col"], DATA["caption_col"], DATA["cui_col"]
    if ic in cols:
        v = row[ic]
        if isinstance(v, dict):
            print(f"    image   : dict keys={list(v.keys())}")
        elif isinstance(v, (bytes, bytearray)):
            print(f"    image   : raw bytes, len={len(v)}")
        else:
            print(f"    image   : type={type(v).__name__}  {str(v)[:60]}")
    if cc in cols:
        cap = str(row[cc])
        print(f"    caption : ({len(cap.split())} words) {cap[:90]}")
    if cui in cols:
        print(f"    cui     : {row[cui]}")
    iid = DATA.get("image_id_col", DEFAULTS["image_id_col"])
    if iid in cols:
        print(f"    image_id: {row[iid]}")


def _check_cols(cols):
    """extract_features.py scans image_col/caption_col/cui_col AND image_id_col.
    A missing one kills the multi-hour run at its first batch, so check all four
    here -- where checking is free. Returns True if every column is present."""
    print("Config column check:")
    ok_all = True
    for key in ("image_col", "caption_col", "cui_col", "image_id_col"):
        name = DATA.get(key, DEFAULTS[key])
        ok = name in cols
        ok_all &= ok
        note = "" if ok else f"   <-- NOT in {cols} ; fix config.data.{key}"
        print(f"  {'ok ' if ok else '!! '}config.data.{key} = '{name}'{note}")
    if not ok_all:
        print()
        print("  !! extract_features.py will fail on the first batch. "
              "Fix config.yaml before running it.")
    return ok_all


if __name__ == "__main__":
    main()