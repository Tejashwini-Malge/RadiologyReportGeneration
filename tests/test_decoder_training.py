"""Collation, checkpoint round-trips, and the dataset reader.

BioBART is not downloaded here -- a stub tokenizer exercises the label-masking
logic, which is the part that silently corrupts training when it is wrong.
"""
import h5py
import numpy as np
import pytest
import torch
import torch.nn as nn
from transformers import get_linear_schedule_with_warmup

import decoder_training as dt


class StubTokenizer:
    """Word-per-token, id 0 is pad. Mirrors the HF call signature used."""
    pad_token_id = 0

    def __call__(self, caps, padding=None, truncation=None,
                 max_length=None, return_tensors=None):
        seqs = [[5 + i for i, _ in enumerate(c.split())][:max_length] for c in caps]
        width = max(len(s) for s in seqs)
        ids = [s + [self.pad_token_id] * (width - len(s)) for s in seqs]
        return {"input_ids": torch.tensor(ids)}


# ---------------- collation ----------------

def test_collate_stacks_features_and_masks_padding_out_of_the_loss():
    collate = dt.build_collate(StubTokenizer())
    batch = [(torch.zeros(49, 1024), "one two three"),
             (torch.ones(49, 1024), "one")]
    feats, labels = collate(batch)

    assert feats.shape == (2, 49, 1024)
    assert labels.shape == (2, 3)
    # row 1 was padded to width 3; those two positions must be ignored
    assert labels[0].tolist() == [5, 6, 7]
    assert labels[1].tolist() == [5, -100, -100]


def test_collate_never_leaves_a_raw_pad_id_in_the_labels():
    collate = dt.build_collate(StubTokenizer())
    _, labels = collate([(torch.zeros(49, 1024), "a b"), (torch.zeros(49, 1024), "a")])
    assert (labels == StubTokenizer.pad_token_id).sum() == 0


def test_collate_truncates_at_max_caption_tokens(monkeypatch):
    monkeypatch.setitem(dt.T, "max_caption_tokens", 4)
    collate = dt.build_collate(StubTokenizer())
    _, labels = collate([(torch.zeros(49, 1024), "a b c d e f g")])
    assert labels.shape[1] == 4


# ---------------- dataset ----------------

@pytest.fixture
def features_h5(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "P", {"features_dir": str(tmp_path),
                                  "checkpoints_dir": str(tmp_path / "ckpt")})
    with h5py.File(tmp_path / "train.h5", "w") as h5:
        h5.create_dataset("features", data=np.arange(3 * 49 * 1024, dtype=np.float16)
                          .reshape(3, 49, 1024))
        h5.create_dataset("caption", data=["a b", "c", "d e f"],
                          dtype=h5py.string_dtype("utf-8"))
    return tmp_path


def test_dataset_length_and_decoding(features_h5):
    ds = dt.FeatureCaptionDS("train")
    assert len(ds) == 3
    feat, cap = ds[1]
    assert feat.shape == (49, 1024)
    assert feat.dtype == torch.float32       # h5 stores fp16, training wants fp32
    assert cap == "c"


def test_dataset_opens_the_h5_lazily_so_it_survives_dataloader_workers(features_h5):
    ds = dt.FeatureCaptionDS("train")
    assert ds.h5 is None
    ds[0]
    assert ds.h5 is not None


# ---------------- checkpointing ----------------

def _trainable(tmp_path):
    model = nn.Linear(4, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = get_linear_schedule_with_warmup(opt, 1, 10)
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    return model, opt, sched, scaler


def test_checkpoint_round_trip_restores_model_optimizer_and_schedule(features_h5):
    model, opt, sched, scaler = _trainable(features_h5)
    # take a real step so optimizer and scheduler state is non-trivial
    model(torch.ones(2, 4)).sum().backward()
    opt.step(); sched.step()
    lr_before = sched.get_last_lr()
    weights_before = model.weight.detach().clone()

    dt.save_ckpt(model, opt, sched, scaler, 3, 1.23, 1.23, [{"epoch": 3}], "last")

    model2, opt2, sched2, scaler2 = _trainable(features_h5)
    epoch, best, history = dt.load_ckpt(dt.ckpt_dir() / "last.pt",
                                        model2, opt2, sched2, scaler2)
    assert epoch == 3 and best == 1.23 and history == [{"epoch": 3}]
    assert torch.equal(model2.weight, weights_before)
    assert sched2.get_last_lr() == lr_before


def test_epoch_checkpoints_are_named_separately_from_last_and_best(features_h5):
    model, opt, sched, scaler = _trainable(features_h5)
    for tag in ("last", "best", "epoch07"):
        dt.save_ckpt(model, opt, sched, scaler, 7, 0.5, 0.5, [], tag)
    names = {p.name for p in dt.ckpt_dir().glob("*.pt")}
    assert names == {"last.pt", "best.pt", "decoder_epoch07.pt"}


def test_checkpoints_embed_the_config_so_evaluate_can_detect_drift(features_h5):
    model, opt, sched, scaler = _trainable(features_h5)
    dt.save_ckpt(model, opt, sched, scaler, 1, 0.5, 0.5, [], "last")
    ck = torch.load(dt.ckpt_dir() / "last.pt", map_location="cpu", weights_only=False)
    assert ck["config"]["train"]["visual_dim"] == dt.T["visual_dim"]


def test_history_is_written_as_json(features_h5):
    import json
    dt.write_history([{"epoch": 1, "val_loss": 2.0}])
    assert json.loads((dt.ckpt_dir() / "history.json").read_text()) == [
        {"epoch": 1, "val_loss": 2.0}]


# ---------------- misc ----------------

@pytest.mark.parametrize("sec,out", [(0, "0h00m"), (90, "0h01m"), (7200, "2h00m"),
                                     (3660, "1h01m")])
def test_fmt_renders_elapsed_time(sec, out):
    assert dt.fmt(sec) == out


# ---------------- --limit subset ----------------

@pytest.fixture
def big_features_h5(tmp_path, monkeypatch):
    monkeypatch.setattr(dt, "P", {"features_dir": str(tmp_path),
                                  "checkpoints_dir": str(tmp_path / "ckpt")})
    monkeypatch.setattr(dt, "RUN_LIMIT", None)
    n = 40
    with h5py.File(tmp_path / "train.h5", "w") as h5:
        # feature value encodes the row index so we can tell which rows were picked
        feats = np.zeros((n, 49, 1024), dtype=np.float16)
        for i in range(n):
            feats[i, 0, 0] = i
        h5.create_dataset("features", data=feats)
        h5.create_dataset("caption", data=[f"cap {i}" for i in range(n)],
                          dtype=h5py.string_dtype("utf-8"))
    return tmp_path


def test_limit_caps_the_number_of_rows(big_features_h5):
    assert len(dt.FeatureCaptionDS("train", limit=10)) == 10


def test_limit_larger_than_the_split_reads_everything(big_features_h5):
    ds = dt.FeatureCaptionDS("train", limit=999)
    assert len(ds) == 40 and ds.index is None


def test_no_limit_reads_everything(big_features_h5):
    assert dt.FeatureCaptionDS("train").index is None


def test_the_subset_is_random_not_the_first_n_rows(big_features_h5):
    """ROCOv2 shards are not shuffled, so head-of-file rows can be skewed."""
    ds = dt.FeatureCaptionDS("train", limit=10, seed=0)
    picked = [ds[i][0][0, 0].item() for i in range(len(ds))]
    assert picked != list(range(10))
    assert all(0 <= p < 40 for p in picked)


def test_the_same_seed_gives_the_same_subset(big_features_h5):
    a = dt.FeatureCaptionDS("train", limit=10, seed=7)
    b = dt.FeatureCaptionDS("train", limit=10, seed=7)
    assert a.index.tolist() == b.index.tolist()


def test_a_different_seed_gives_a_different_subset(big_features_h5):
    a = dt.FeatureCaptionDS("train", limit=10, seed=0)
    b = dt.FeatureCaptionDS("train", limit=10, seed=1)
    assert a.index.tolist() != b.index.tolist()


def test_the_subset_has_no_duplicate_rows(big_features_h5):
    idx = dt.FeatureCaptionDS("train", limit=10, seed=3).index
    assert len(set(idx.tolist())) == 10


def test_subset_indices_are_sorted_for_h5py_fancy_indexing(big_features_h5):
    idx = dt.FeatureCaptionDS("train", limit=10, seed=3).index
    assert idx.tolist() == sorted(idx.tolist())


def test_features_and_captions_stay_aligned_under_subsetting(big_features_h5):
    ds = dt.FeatureCaptionDS("train", limit=10, seed=5)
    for i in range(len(ds)):
        feat, cap = ds[i]
        assert cap == f"cap {int(feat[0, 0].item())}"


def test_a_subset_run_cannot_overwrite_real_checkpoints(big_features_h5, monkeypatch):
    """A 10-row smoke run must not be able to clobber best.pt from a real run --
    the same quarantine as _dry.h5 in extract_features.py."""
    real = dt.ckpt_dir()
    monkeypatch.setattr(dt, "RUN_LIMIT", 10)
    subset = dt.ckpt_dir()
    assert subset != real
    assert subset.name == "limit10"
    assert subset.parent == real
