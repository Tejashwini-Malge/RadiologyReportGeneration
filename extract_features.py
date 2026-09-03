"""
02_extract_features.py  (v3.1 -- v3 + overwrite guard + isolated dry runs)
Run the FROZEN Swin-base encoder over every split ONCE and write features to .h5.
After this you never touch raw images again -- training runs on these .h5 files.

v3 changes:
  * writes `image_id` -- the join key for the Zenodo concept CSVs. Row order is NOT
    a safe key: the HF mirror has 59,962 train rows, Zenodo's CSV has 59,958.
  * writes h5.attrs["complete"] / ["n_written"] -- because create_dataset PREALLOCATES
    the full array, an interrupted run leaves a file with the right shape and the
    right size, silently full of zero features and empty captions. Neither
    f["features"].shape[0] nor the file size can detect this. attrs can.

v3.1 changes (two only -- everything else is byte-for-byte v3):
  * OVERWRITE GUARD: h5py.File(out, "w") TRUNCATES. A stray re-run of any cell
    silently destroyed a complete 5.8 GB train.h5 and replaced it with a 10 MB
    dry run, reporting COMPLETE. Now a complete file with the expected row count
    is skipped unless --overwrite is passed explicitly.
  * ISOLATED DRY RUNS: --limit writes to <split>_dry.h5, never <split>.h5. A dry
    run can no longer touch real output, no matter which cell you run.

Output per split:  <features_dir>/<split>.h5      (dry runs: <split>_dry.h5)
    features : float16 [N, 49, 1024]   (Swin last_hidden_state, for cross-attention)
    pooler   : float16 [N, 1024]       (pooled, for the CUI classifier)
    caption  : utf-8 strings
    cui      : utf-8 strings           (mirror's manual-only subset -- see concepts CSV)
    image_id : utf-8 strings           (e.g. ROCOv2_2023_train_000001)
    attrs    : complete (bool), n_written (int)

Usage:
    python 02_extract_features.py --split valid --limit 64   # dry run -> valid_dry.h5
    python 02_extract_features.py --split train
    python 02_extract_features.py                            # valid, test, train
    python 02_extract_features.py --split train --overwrite  # force re-extract
    python 02_extract_features.py --verify-only
"""
import io
import glob
import argparse
from pathlib import Path

import numpy as np
import h5py
import torch
from PIL import Image
import pyarrow.dataset as ds
from transformers import AutoImageProcessor, SwinModel

from rrg_config import load_config

CFG = load_config()
P, DATA, EX = CFG["paths"], CFG["data"], CFG["extract"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ID_COL = DATA.get("image_id_col", "image_id")


def files_for(pattern):
    return sorted(glob.glob(str(Path(P["rocov2_dir"]) / pattern)))


def out_path(split, limit=None):
    """Real output is <split>.h5; dry runs are quarantined to <split>_dry.h5."""
    name = f"{split}_dry.h5" if limit is not None else f"{split}.h5"
    return Path(P["features_dir"]) / name


def decode_image(v):
    """ROCOv2 image cell is {'bytes':..,'path':..}. Prefer bytes."""
    if isinstance(v, dict):
        if v.get("bytes") is not None:
            return Image.open(io.BytesIO(v["bytes"])).convert("RGB")
        if v.get("path"):
            return Image.open(v["path"]).convert("RGB")
    if isinstance(v, (bytes, bytearray)):
        return Image.open(io.BytesIO(v)).convert("RGB")
    if isinstance(v, str):
        return Image.open(v).convert("RGB")
    raise ValueError(f"Unrecognized image cell type: {type(v)}")


@torch.no_grad()
def extract_split(split, model, processor, limit=None, overwrite=False):
    fs = files_for(DATA["parquet"][split])
    if not fs:
        print(f"[{split}] no shards match {DATA['parquet'][split]} -- skipping")
        return
    dataset = ds.dataset(fs, format="parquet")
    n = dataset.count_rows() if limit is None else min(limit, dataset.count_rows())

    out = out_path(split, limit)
    out.parent.mkdir(parents=True, exist_ok=True)

    # ---- OVERWRITE GUARD -------------------------------------------------
    # h5py.File(out, "w") truncates. Refuse to open a complete file for
    # writing unless the caller explicitly asked for it.
    if out.exists() and not overwrite:
        try:
            with h5py.File(out, "r") as h5:
                done = bool(h5.attrs.get("complete", False))
                nw = int(h5.attrs.get("n_written", -1))
            if done and nw == n:
                print(f"[{split}] SKIP -- complete file exists "
                      f"({nw:,} rows) at {out}")
                print(f"[{split}]         pass --overwrite to force re-extraction\n")
                return
            if done and nw != n:
                print(f"[{split}] REFUSING to overwrite: existing file is complete "
                      f"with {nw:,} rows but this run expects {n:,}.")
                print(f"[{split}]         if that is intended, pass --overwrite\n")
                return
            print(f"[{split}] existing file is incomplete ({nw:,}/{n:,}) "
                  f"-- re-extracting")
        except OSError as e:
            print(f"[{split}] existing file unreadable ({e}) -- re-extracting")
    # ----------------------------------------------------------------------

    np_dtype = np.float16 if EX["dtype"] == "float16" else np.float32
    bs = EX["batch_size"]
    print(f"[{split}] {n:,} rows ({len(fs)} shards) -> {out}")

    vstr = h5py.string_dtype(encoding="utf-8")
    i = 0
    with h5py.File(out, "w") as h5:
        d_feat = h5.create_dataset("features", (n, 49, 1024), dtype=np_dtype)
        d_pool = h5.create_dataset("pooler",   (n, 1024),     dtype=np_dtype)
        d_cap  = h5.create_dataset("caption",  (n,), dtype=vstr)
        d_cui  = h5.create_dataset("cui",      (n,), dtype=vstr)
        d_iid  = h5.create_dataset("image_id", (n,), dtype=vstr)

        # mark incomplete up front -- if the process dies, the flag stays False
        h5.attrs["complete"] = False
        h5.attrs["n_written"] = 0

        cols = [DATA["image_col"], DATA["caption_col"], DATA["cui_col"], ID_COL]
        for batch in dataset.scanner(columns=cols, batch_size=bs).to_batches():
            df = batch.to_pandas()
            for start in range(0, len(df), bs):
                if i >= n:
                    break
                chunk = df.iloc[start:start + bs]
                imgs = [decode_image(v) for v in chunk[DATA["image_col"]]]
                inp = processor(images=imgs, return_tensors="pt").to(DEVICE)
                with torch.inference_mode():
                    om = model(**inp)
                feat = om.last_hidden_state.to(torch.float16).cpu().numpy()
                pool = om.pooler_output.to(torch.float16).cpu().numpy()

                b = min(len(imgs), n - i)
                d_feat[i:i + b] = feat[:b].astype(np_dtype)
                d_pool[i:i + b] = pool[:b].astype(np_dtype)
                d_cap[i:i + b]  = chunk[DATA["caption_col"]].astype(str).tolist()[:b]
                d_cui[i:i + b]  = [str(x) for x in chunk[DATA["cui_col"]].tolist()][:b]
                d_iid[i:i + b]  = chunk[ID_COL].astype(str).tolist()[:b]
                i += b
                if i % (bs * 20) == 0 or i >= n:
                    print(f"    {i:,}/{n:,}", flush=True)
            if i >= n:
                break

        h5.attrs["n_written"] = i
        h5.attrs["complete"] = bool(i == n)

    mb = out.stat().st_size / (1024 ** 2)
    status = "COMPLETE" if i == n else f"*** INCOMPLETE {i:,}/{n:,} -- RERUN ***"
    print(f"[{split}] {status}. {out.name} = {mb:.0f} MB\n")


def verify(split, limit=None):
    """Cheap post-hoc check. Use this, not the file size."""
    out = out_path(split, limit)
    if not out.exists():
        print(f"[{split}] MISSING ({out.name})")
        return False
    with h5py.File(out, "r") as h5:
        ok = bool(h5.attrs.get("complete", False))
        nw = int(h5.attrs.get("n_written", -1))
        tot = h5["features"].shape[0]
        has_id = "image_id" in h5
        sample = h5["image_id"][0].decode() if has_id and nw else "-"
    print(f"[{split}] {out.name}  complete={ok}  written={nw:,}/{tot:,}  "
          f"image_id={'yes' if has_id else 'NO'}  first_id={sample}")
    return ok and has_id


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "valid", "test"], default=None)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap rows (dry run -- writes <split>_dry.h5)")
    ap.add_argument("--overwrite", action="store_true",
                    help="force re-extraction even if a complete .h5 exists")
    ap.add_argument("--verify-only", action="store_true",
                    help="check existing .h5 files, extract nothing")
    args = ap.parse_args()

    splits = [args.split] if args.split else ["valid", "test", "train"]

    if args.verify_only:
        for s in splits:
            verify(s, args.limit)
        return

    print(f"Device: {DEVICE}   encoder: {CFG['models']['vision_encoder']}")
    if DEVICE == "cpu":
        print("  !! No CUDA -- run this on the GPU machine.")

    processor = AutoImageProcessor.from_pretrained(
        CFG["models"]["vision_encoder"],
        use_fast=True)
    model = SwinModel.from_pretrained(CFG["models"]["vision_encoder"]).to(DEVICE).eval()
    for p in model.parameters():
        p.requires_grad = False

    for s in splits:
        extract_split(s, model, processor, limit=args.limit,
                      overwrite=args.overwrite)

    print("--- verify ---")
    for s in splits:
        verify(s, args.limit)


if __name__ == "__main__":
    main()