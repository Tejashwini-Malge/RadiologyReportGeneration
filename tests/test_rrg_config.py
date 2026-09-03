"""config.yaml loading + the RRG_* environment overrides."""
import pytest

import rrg_config as rc


def test_defaults_come_from_the_yaml_file():
    paths = rc.load_config(env={})["paths"]
    assert paths["d_root"] == "D:/rrg"
    assert paths["features_dir"] == "D:/rrg/features"


def test_rrg_root_moves_every_derived_path_together():
    """The regression this exists for: setting only d_root would leave
    features_dir on D:, which does not exist on Colab."""
    paths = rc.load_config(env={"RRG_ROOT": "/content/drive/MyDrive/rrg"})["paths"]
    assert paths == {
        "d_root":          "/content/drive/MyDrive/rrg",
        "rocov2_dir":      "/content/drive/MyDrive/rrg/raw/rocov2",
        "features_dir":    "/content/drive/MyDrive/rrg/features",
        "checkpoints_dir": "/content/drive/MyDrive/rrg/checkpoints",
        "cache_root":      "/content/drive/MyDrive/rrg/cache",
    }


def test_a_single_path_override_wins_over_rrg_root():
    paths = rc.load_config(env={"RRG_ROOT": "/content/rrg",
                                "RRG_FEATURES_DIR": "/scratch/feat"})["paths"]
    assert paths["features_dir"] == "/scratch/feat"
    assert paths["checkpoints_dir"] == "/content/rrg/checkpoints"


def test_override_without_rrg_root_leaves_the_rest_alone():
    paths = rc.load_config(env={"RRG_CHECKPOINTS_DIR": "/ckpt"})["paths"]
    assert paths["checkpoints_dir"] == "/ckpt"
    assert paths["d_root"] == "D:/rrg"


def test_empty_env_var_is_ignored_not_applied_as_empty_path():
    paths = rc.load_config(env={"RRG_FEATURES_DIR": ""})["paths"]
    assert paths["features_dir"] == "D:/rrg/features"


@pytest.mark.parametrize("path,expected", [
    ("D:/rrg",              "/new"),                 # the root itself
    ("D:/rrg/features",     "/new/features"),        # under the root
    ("D:\\rrg\\features",  "/new/features"),      # windows separators
    ("d:/RRG/features",     "/new/features"),        # windows case-insensitivity
    ("E:/elsewhere",        "E:/elsewhere"),         # not under the root: untouched
    ("D:/rrg2/features",    "D:/rrg2/features"),     # prefix but not a path component
])
def test_reroot_boundaries(path, expected):
    assert rc._reroot(path, "D:/rrg", "/new") == expected


def test_missing_config_file_says_how_to_fix_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="make_config.py"):
        rc.load_config(path=tmp_path / "nope.yaml")


def test_config_yaml_matches_what_make_config_would_write():
    """make_config.py is the generator; config.yaml is its committed output.
    If they drift, everyone downstream is reading stale values."""
    import ast
    src = (rc.CONFIG_PATH.parent / "make_config.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    node = next(n for n in tree.body
                if isinstance(n, __import__("ast").Assign)
                and n.targets[0].id == "CFG")
    assert ast.literal_eval(node.value) == rc.load_config(env={})


# ---------------- console encoding ----------------

def test_enable_utf8_stdout_makes_unencodable_text_printable(capsys):
    """Regression: a generated caption containing U+202F crashed evaluate.py on
    a cp1252 Windows console -- after generation had already finished."""
    rc.enable_utf8_stdout()
    print("narrow no-break space")
    assert "no-break" in capsys.readouterr().out


def test_enable_utf8_stdout_is_a_noop_when_streams_cannot_reconfigure(monkeypatch):
    import io
    import sys
    monkeypatch.setattr(sys, "stdout", io.StringIO())   # no reconfigure()
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    rc.enable_utf8_stdout()                             # must not raise
