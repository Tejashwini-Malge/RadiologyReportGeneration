"""The pre-flight check. Its whole job is to fail loudly here rather than
three hours into extract_features.py."""
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import load_Dataset as ld


@pytest.mark.parametrize("n,out", [
    (0, "0.00 B"), (512, "512.00 B"), (2048, "2.00 KB"),
    (1024 ** 2, "1.00 MB"), (3 * 1024 ** 3, "3.00 GB"),
])
def test_human_readable_sizes(n, out):
    assert ld.human(n) == out


def test_check_cols_passes_when_every_configured_column_exists(capsys):
    assert ld._check_cols(["image", "caption", "cui", "image_id", "extra"]) is True
    assert "!!" not in capsys.readouterr().out


def test_check_cols_fails_on_a_missing_image_id(capsys):
    """The regression: image_id was not checked, so a mirror without it passed
    pre-flight and then killed extract_features.py at its first batch."""
    assert ld._check_cols(["image", "caption", "cui"]) is False
    out = capsys.readouterr().out
    assert "config.data.image_id_col" in out
    assert "will fail on the first batch" in out


def test_check_cols_names_the_config_key_to_fix(capsys):
    ld._check_cols(["image", "cui", "image_id"])
    assert "fix config.data.caption_col" in capsys.readouterr().out


def test_check_cols_honours_a_renamed_column_in_config(monkeypatch, capsys):
    monkeypatch.setitem(ld.DATA, "caption_col", "report")
    assert ld._check_cols(["image", "report", "cui", "image_id"]) is True


def test_files_for_globs_shards_under_the_dataset_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(ld, "ROCOV2_DIR", str(tmp_path))
    (tmp_path / "data").mkdir()
    for name in ["train-00000.parquet", "train-00001.parquet", "test-00000.parquet"]:
        (tmp_path / "data" / name).touch()
    found = ld.files_for("data/train-*.parquet")
    assert [p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1] for p in found] == [
        "train-00000.parquet", "train-00001.parquet"]


def test_files_for_returns_empty_when_nothing_matches(tmp_path, monkeypatch):
    monkeypatch.setattr(ld, "ROCOV2_DIR", str(tmp_path))
    assert ld.files_for("data/nope-*.parquet") == []


def test_describe_row_survives_each_image_encoding(capsys):
    cols = ["image", "caption", "cui", "image_id"]
    for cell, expect in [({"bytes": b"x", "path": None}, "dict keys"),
                         (b"rawbytes", "raw bytes, len=8"),
                         ("/some/path.png", "type=str")]:
        ld._describe_row({"image": cell, "caption": "a b c",
                          "cui": "['C000001']", "image_id": "ROCOv2_x"}, cols)
        out = capsys.readouterr().out
        assert expect in out
        assert "(3 words)" in out
        assert "image_id: ROCOv2_x" in out


def test_main_reports_a_real_parquet_split(tmp_path, monkeypatch, capsys):
    """End-to-end over a tiny on-disk split -- catches schema/API breakage."""
    monkeypatch.setattr(ld, "ROCOV2_DIR", str(tmp_path))
    monkeypatch.setitem(ld.DATA, "parquet", {"train": "data/train-*.parquet"})
    (tmp_path / "data").mkdir()
    pq.write_table(pa.table({
        "image":    pa.array([{"bytes": b"png", "path": None}] * 2),
        "caption":  pa.array(["one two", "three"]),
        "cui":      pa.array(["['C000001']", "[]"]),
        "image_id": pa.array(["ROCOv2_a", "ROCOv2_b"]),
    }), tmp_path / "data" / "train-00000.parquet")

    ld.main()
    out = capsys.readouterr().out
    assert "1 shards,  2 rows" in out
    assert "TOTAL: 2 rows" in out
    assert "!!" not in out


def test_main_flags_a_split_with_no_shards(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(ld, "ROCOV2_DIR", str(tmp_path))
    monkeypatch.setitem(ld.DATA, "parquet", {"train": "data/missing-*.parquet"})
    ld.main()
    assert "NO files match" in capsys.readouterr().out
