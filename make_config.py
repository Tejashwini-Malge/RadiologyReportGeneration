"""
make_config.py  (v2 — handles sharded parquet under data/)
Writes a VALID config.yaml next to this file. Run once:
    python make_config.py
"""
from pathlib import Path
import yaml

CFG = {
    "paths": {
        "d_root":          "D:/rrg",
        "rocov2_dir":      "D:/rrg/raw/rocov2",   # folder that contains the data/ subfolder
        "features_dir":    "D:/rrg/features",
        "checkpoints_dir": "D:/rrg/checkpoints",
        "cache_root":      "D:/rrg/cache",
    },
    "data": {
        # glob patterns (relative to rocov2_dir) -- match ALL shards of each split
        "parquet": {
            "train": "data/train-*.parquet",
            "valid": "data/validation-*.parquet",
            "test":  "data/test-*.parquet",
        },
        # >>> fix these to what 01 prints if they differ <<<
        "image_col":   "image",
        "caption_col": "caption",
        "cui_col":     "cui",
    },
    "models": {
        "vision_encoder":  "microsoft/swin-base-patch4-window7-224",
        "decoder":         "GanjinZero/biobart-v2-base",
        "concept_encoder": "StanfordAIMI/RadBERT",
    },
    "extract": {
        "image_size":  224,
        "batch_size":  32,
        "num_workers": 4,
        "dtype":       "float16",
    },
    "train": {
        "batch_size":         8,
        "lr":                 3.0e-5,
        "epochs":             10,
        "max_caption_tokens": 128,
        "visual_dim":         1024,
        "decoder_dim":        768,
        "grad_accum":         2,
        "warmup_steps":       500,
        "save_every_epoch":   True,
    },
}

out = Path(__file__).parent / "config.yaml"
with open(out, "w", encoding="utf-8") as f:
    yaml.dump(CFG, f, sort_keys=False, default_flow_style=False)

reparsed = yaml.safe_load(open(out, encoding="utf-8"))
print(f"Wrote and validated: {out}")
for split, patt in reparsed["data"]["parquet"].items():
    print(f"  {split:5s} -> {patt}")