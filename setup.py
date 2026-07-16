
import os
import sys
import subprocess
from pathlib import Path

import yaml

CFG = yaml.safe_load(open(Path(__file__).parent / "config.yaml", encoding="utf-8"))
P = CFG["paths"]

# Folders that must exist on D:
DIRS = {
    "root":        P["d_root"],
    "rocov2":      P["rocov2_dir"],
    "features":    P["features_dir"],
    "checkpoints": P["checkpoints_dir"],
    "cache":       P["cache_root"],
    "hf_home":     os.path.join(P["cache_root"], "huggingface"),
    "torch_home":  os.path.join(P["cache_root"], "torch"),
    "pip_cache":   os.path.join(P["cache_root"], "pip"),
    "tmp":         os.path.join(P["cache_root"], "tmp"),
}

# Env vars we want pointed at D:
ENV = {
    "HF_HOME":            DIRS["hf_home"],
    "HUGGINGFACE_HUB_CACHE": os.path.join(DIRS["hf_home"], "hub"),
    "TRANSFORMERS_CACHE": os.path.join(DIRS["hf_home"], "transformers"),
    "HF_DATASETS_CACHE":  os.path.join(DIRS["hf_home"], "datasets"),
    "TORCH_HOME":         DIRS["torch_home"],
    "PIP_CACHE_DIR":      DIRS["pip_cache"],
    "TMP":                DIRS["tmp"],
    "TEMP":               DIRS["tmp"],
}


def main():
    if os.name != "nt":
        print("WARNING: not Windows. Dirs will be created but setx is skipped.")

    # 1. create all directories
    for name, d in DIRS.items():
        Path(d).mkdir(parents=True, exist_ok=True)
        print(f"  ok  {name:12s} -> {d}")

    # 2. set for THIS process (so you can keep working in the same session)
    for k, v in ENV.items():
        os.environ[k] = v

    # 3. persist for future terminals via setx (Windows only)
    if os.name == "nt":
        print("\nPersisting env vars (setx)...")
        for k, v in ENV.items():
            # setx writes to the user registry; value length limit is 1024 chars
            subprocess.run(["setx", k, v], check=False,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(f"  setx {k}={v}")

    # 4. write a .env you can eyeball / source later
    env_path = Path(P["d_root"]) / ".env"
    env_path.write_text("\n".join(f"{k}={v}" for k, v in ENV.items()), encoding="utf-8")

    print(f"\nWrote {env_path}")
    free_gb = _free_gb(P["d_root"])
    print(f"Free space on D: ~ {free_gb:.1f} GB")
    if free_gb < 25:
        print("  !! Under 25 GB free. Features(~8GB)+checkpoints may not fit. Free space first.")
    print("\nDONE. Close and reopen your terminal, then run 01_load_rocov2.py")


def _free_gb(path):
    try:
        import shutil
        return shutil.disk_usage(path).free / (1024 ** 3)
    except Exception:
        return -1.0


if __name__ == "__main__":
    main()