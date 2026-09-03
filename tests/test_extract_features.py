"""The overwrite guard and dry-run isolation in extract_features.py.

These two behaviours exist because a stray re-run once truncated a complete
5.8 GB train.h5 and replaced it with a 10 MB dry run that reported COMPLETE.
Nothing else in the repo protects the most expensive artefact it produces.

The Swin encoder is replaced with a stub -- this exercises the file-handling
logic, not the model.
"""
import io
import types

import h5py
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from PIL import Image

import extract_features as ef


# ---------------- fixtures ----------------

def _png_bytes(colour):
    buf = io.BytesIO()
    Image.new("RGB", (8, 8), colour).save(buf, format="PNG")
    return buf.getvalue()


class _StubInputs(dict):
    """Stands in for the processor's BatchFeature: dict-like with .to()."""
    def to(self, device):
        return self


class _StubProcessor:
    def __call__(self, images, return_tensors=None):
        return _StubInputs(pixel_values=torch.zeros(len(images), 3, 224, 224))


class _StubSwin:
    """Returns the real output shapes; values encode the batch position so a
    truncated or misaligned write is visible in the assertions."""
    def __call__(self, pixel_values):
        b = pixel_values.shape[0]
        return types.SimpleNamespace(
            last_hidden_state=torch.ones(b, 49, 1024),
            pooler_output=torch.ones(b, 1024),
        )


@pytest.fixture
def rig(tmp_path, monkeypatch):
    """A 4-row parquet 'split' plus module paths pointed at tmp_path."""
    rocov2, feats = tmp_path / "rocov2", tmp_path / "features"
    (rocov2 / "data").mkdir(parents=True)
    feats.mkdir()

    n = 4
    table = pa.table({
        "image":    pa.array([{"bytes": _png_bytes((i * 40, 0, 0)), "path": None}
                              for i in range(n)]),
        "caption":  pa.array([f"caption {i}" for i in range(n)]),
        "cui":      pa.array([f"['C000{i:03d}']" for i in range(n)]),
        "image_id": pa.array([f"ROCOv2_2023_train_{i:06d}" for i in range(n)]),
    })
    pq.write_table(table, rocov2 / "data" / "train-00000.parquet")

    monkeypatch.setattr(ef, "P", {"rocov2_dir": str(rocov2),
                                  "features_dir": str(feats)})
    monkeypatch.setattr(ef, "DATA", {"parquet": {"train": "data/train-*.parquet"},
                                     "image_col": "image",
                                     "caption_col": "caption",
                                     "cui_col": "cui"})
    monkeypatch.setattr(ef, "EX", {"dtype": "float16", "batch_size": 2})
    monkeypatch.setattr(ef, "ID_COL", "image_id")
    monkeypatch.setattr(ef, "DEVICE", "cpu")
    return types.SimpleNamespace(features=feats, n=n,
                                 model=_StubSwin(), processor=_StubProcessor())


def run(rig, **kw):
    ef.extract_split("train", rig.model, rig.processor, **kw)


# ---------------- decode_image ----------------

def test_decode_image_prefers_the_bytes_field():
    assert ef.decode_image({"bytes": _png_bytes("red"), "path": "/does/not/exist"}).size == (8, 8)


def test_decode_image_falls_back_to_path(tmp_path):
    f = tmp_path / "x.png"
    Image.new("RGB", (8, 8), "blue").save(f)
    assert ef.decode_image({"bytes": None, "path": str(f)}).size == (8, 8)


def test_decode_image_accepts_raw_bytes():
    assert ef.decode_image(_png_bytes("green")).mode == "RGB"


def test_decode_image_rejects_unknown_cell_types():
    with pytest.raises(ValueError, match="Unrecognized image cell type"):
        ef.decode_image(42)


# ---------------- out_path ----------------

def test_dry_runs_get_their_own_filename(rig):
    assert ef.out_path("train").name == "train.h5"
    assert ef.out_path("train", limit=64).name == "train_dry.h5"


# ---------------- extraction ----------------

def test_a_full_run_writes_every_row_and_marks_the_file_complete(rig):
    run(rig)
    with h5py.File(rig.features / "train.h5", "r") as h5:
        assert h5.attrs["complete"] is np.True_ or bool(h5.attrs["complete"])
        assert int(h5.attrs["n_written"]) == rig.n
        assert h5["features"].shape == (rig.n, 49, 1024)
        assert h5["pooler"].shape == (rig.n, 1024)
        assert [c.decode() for c in h5["caption"][:]] == [f"caption {i}" for i in range(rig.n)]
        assert h5["image_id"][0].decode() == "ROCOv2_2023_train_000000"
        assert h5["image_id"][-1].decode() == "ROCOv2_2023_train_000003"


def test_rerunning_skips_a_complete_file_instead_of_truncating_it(rig, capsys):
    run(rig)
    before = (rig.features / "train.h5").read_bytes()
    run(rig)                                    # the stray re-run
    assert (rig.features / "train.h5").read_bytes() == before
    assert "SKIP" in capsys.readouterr().out


def test_overwrite_flag_does_re_extract(rig, capsys):
    run(rig)
    run(rig, overwrite=True)
    out = capsys.readouterr().out
    assert "SKIP" not in out and "COMPLETE" in out


def test_a_dry_run_cannot_touch_the_real_file(rig):
    """The exact failure mode: --limit must never open <split>.h5 for writing."""
    run(rig)
    real = (rig.features / "train.h5").read_bytes()
    run(rig, limit=2)
    assert (rig.features / "train.h5").read_bytes() == real
    assert (rig.features / "train_dry.h5").exists()
    with h5py.File(rig.features / "train_dry.h5", "r") as h5:
        assert int(h5.attrs["n_written"]) == 2


def test_an_interrupted_file_is_re_extracted_not_skipped(rig, capsys):
    """A killed run leaves the right shape and the right size, full of zeros.
    Only the attrs can tell -- so the guard must trust them, not the shape."""
    out = rig.features / "train.h5"
    with h5py.File(out, "w") as h5:
        h5.create_dataset("features", (rig.n, 49, 1024), dtype=np.float16)
        h5.attrs["complete"] = False
        h5.attrs["n_written"] = 1
    run(rig)
    assert "incomplete" in capsys.readouterr().out
    with h5py.File(out, "r") as h5:
        assert int(h5.attrs["n_written"]) == rig.n
        assert np.asarray(h5["features"][0]).any()      # real values, not zeros


def test_a_complete_file_with_the_wrong_row_count_is_refused(rig, capsys):
    """Guards against extracting against a different dataset revision."""
    out = rig.features / "train.h5"
    with h5py.File(out, "w") as h5:
        h5.create_dataset("features", (99, 49, 1024), dtype=np.float16)
        h5.attrs["complete"] = True
        h5.attrs["n_written"] = 99
    run(rig)
    assert "REFUSING to overwrite" in capsys.readouterr().out
    with h5py.File(out, "r") as h5:
        assert int(h5.attrs["n_written"]) == 99      # untouched


def test_missing_shards_are_skipped_without_creating_a_file(rig, capsys):
    ef.DATA["parquet"]["train"] = "data/nothing-*.parquet"
    run(rig)
    assert "no shards match" in capsys.readouterr().out
    assert not (rig.features / "train.h5").exists()


# ---------------- verify ----------------

def test_verify_reports_true_only_for_a_complete_file_with_image_ids(rig):
    assert ef.verify("train") is False          # nothing extracted yet
    run(rig)
    assert ef.verify("train") is True


def test_verify_reports_false_for_an_interrupted_file(rig):
    with h5py.File(rig.features / "train.h5", "w") as h5:
        h5.create_dataset("features", (rig.n, 49, 1024), dtype=np.float16)
        h5.create_dataset("image_id", (rig.n,), dtype=h5py.string_dtype("utf-8"))
        h5.attrs["complete"] = False
        h5.attrs["n_written"] = 2
    assert ef.verify("train") is False
