"""
rrg_config.py
One place to load config.yaml, with environment overrides so the same checkout
runs on the Windows box (D:/rrg) and on Colab (/content/drive/MyDrive/rrg)
without editing the file.

Overrides, highest priority last:
    config.yaml
    RRG_ROOT                -> re-roots every path that lives under d_root
    RRG_ROCOV2_DIR / RRG_FEATURES_DIR / RRG_CHECKPOINTS_DIR / RRG_CACHE_ROOT
                            -> replace one path outright

Why re-rooting and not just "set d_root": the other four paths in config.yaml are
written as absolute D:/rrg/... strings, so changing d_root alone would leave
features_dir pointing at a drive that does not exist on Colab.

    export RRG_ROOT=/content/drive/MyDrive/rrg     # everything moves together
    export RRG_FEATURES_DIR=/scratch/features      # ...except this one
"""
import os
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).parent / "config.yaml"

# env var -> key in config["paths"]
PATH_ENV = {
    "RRG_ROCOV2_DIR":      "rocov2_dir",
    "RRG_FEATURES_DIR":    "features_dir",
    "RRG_CHECKPOINTS_DIR": "checkpoints_dir",
    "RRG_CACHE_ROOT":      "cache_root",
}


def _posix(p):
    """Normalise separators so re-rooting compares like with like on Windows."""
    return str(p).replace("\\", "/").rstrip("/")


def _reroot(path, old_root, new_root):
    """If `path` is inside old_root, move it under new_root. Otherwise leave it."""
    p, old = _posix(path), _posix(old_root)
    if p == old:
        return _posix(new_root)
    if p.lower().startswith(old.lower() + "/"):
        return _posix(Path(new_root) / p[len(old) + 1:])
    return p


def apply_env(cfg, env=None):
    """Apply RRG_* overrides to an already-parsed config dict. Returns the dict."""
    env = os.environ if env is None else env
    paths = cfg["paths"]

    new_root = env.get("RRG_ROOT")
    if new_root:
        old_root = paths["d_root"]
        for key, val in list(paths.items()):
            paths[key] = _reroot(val, old_root, new_root)

    for var, key in PATH_ENV.items():
        if env.get(var):
            paths[key] = _posix(env[var])

    return cfg


def load_config(path=None, env=None):
    """Parse config.yaml and apply environment overrides."""
    path = Path(path) if path else CONFIG_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found -- run `python make_config.py` first."
        )
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return apply_env(cfg, env)


def enable_utf8_stdout():
    """Make print() safe for text that is not representable in the console's
    codepage. Windows consoles default to cp1252; a single U+202F in a generated
    caption or a ROCOv2 reference raises UnicodeEncodeError and kills the script
    -- after the expensive work is already done. Replaces unencodable characters
    instead of raising. No-op where stdout does not support reconfigure()."""
    import sys
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def describe(cfg):
    """One-line-per-path summary, for scripts to print at startup."""
    return "\n".join(f"  {k:16s} {v}" for k, v in cfg["paths"].items())
