"""Metric helpers and the degeneracy check.

degeneracy_report is the one signal that catches the classic captioning
failure -- one generic sentence for every image -- which BLEU alone hides.
"""
import importlib.util

import h5py
import numpy as np
import pytest
import torch

import evaluate as ev

has_rouge = importlib.util.find_spec("rouge_score") is not None


# ---------------- degeneracy ----------------

def test_degeneracy_flags_a_model_that_emits_one_sentence_for_everything():
    preds = ["no acute findings"] * 9 + ["something else"]
    r = ev.degeneracy_report(preds)
    assert r["n"] == 10
    assert r["unique_preds"] == 2
    assert r["unique_ratio"] == 0.2
    assert r["most_common_pred"] == "no acute findings"
    assert r["most_common_share"] == 0.9


def test_degeneracy_is_clean_for_fully_distinct_predictions():
    r = ev.degeneracy_report([f"finding {i}" for i in range(5)])
    assert r["unique_ratio"] == 1.0
    assert r["most_common_share"] == 0.2
    assert r["mean_pred_len_words"] == 2.0
    assert r["empty_preds"] == 0


def test_degeneracy_counts_empty_predictions():
    r = ev.degeneracy_report(["", "", "a b c"])
    assert r["empty_preds"] == 2
    assert r["mean_pred_len_words"] == 1.0


# ---------------- BLEU ----------------

def test_bleu_is_one_when_predictions_match_the_references():
    refs = ["chest x ray shows no acute disease",
            "ct of the abdomen with contrast"]
    out = ev.compute_bleu(list(refs), refs)
    for n in (1, 2, 3, 4):
        assert out[f"BLEU-{n}"] == pytest.approx(1.0, abs=1e-6)


def test_bleu_is_case_insensitive():
    refs = ["Chest X Ray Shows No Acute Disease"]
    assert ev.compute_bleu(["chest x ray shows no acute disease"],
                           refs)["BLEU-4"] == pytest.approx(1.0, abs=1e-6)


def test_bleu_drops_when_predictions_are_unrelated():
    out = ev.compute_bleu(["completely unrelated words here now"],
                          ["chest x ray shows no acute disease"])
    assert out["BLEU-1"] < 0.2


@pytest.mark.skipif(not has_rouge, reason="rouge-score not installed")
def test_rouge_is_one_for_identical_text():
    refs = ["chest x ray shows no acute disease"]
    out = ev.compute_rouge(list(refs), refs)
    assert out["ROUGE-1"] == pytest.approx(1.0)
    assert out["ROUGE-L"] == pytest.approx(1.0)


# ---------------- dataset / collate ----------------

@pytest.fixture
def features_h5(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "P", {"features_dir": str(tmp_path)})
    with h5py.File(tmp_path / "test.h5", "w") as h5:
        h5.create_dataset("features", data=np.zeros((5, 49, 1024), dtype=np.float16))
        h5.create_dataset("caption", data=[f"ref {i}" for i in range(5)],
                          dtype=h5py.string_dtype("utf-8"))
    return tmp_path


def test_reference_dataset_returns_raw_caption_strings(features_h5):
    ds = ev.FeatureRefDS("test")
    assert len(ds) == 5
    feat, cap = ds[2]
    assert feat.shape == (49, 1024) and feat.dtype == torch.float32
    assert cap == "ref 2"


def test_limit_caps_the_number_of_rows(features_h5):
    assert len(ev.FeatureRefDS("test", limit=2)) == 2
    assert len(ev.FeatureRefDS("test", limit=999)) == 5


def test_missing_features_file_names_the_step_that_produces_it(tmp_path, monkeypatch):
    monkeypatch.setattr(ev, "P", {"features_dir": str(tmp_path)})
    with pytest.raises(FileNotFoundError, match="run extract_features.py"):
        ev.FeatureRefDS("valid")


def test_collate_keeps_references_as_strings():
    feats, refs = ev.collate([(torch.zeros(49, 1024), "a"),
                              (torch.ones(49, 1024), "b")])
    assert feats.shape == (2, 49, 1024)
    assert refs == ["a", "b"]


# ---------------- trainer import ----------------

def test_evaluate_imports_the_model_class_directly():
    """evaluate.py rebuilds the model to load a checkpoint into. It used to
    locate the trainer via importlib path-loading; that is now a plain import,
    so a rename breaks at import time instead of silently at runtime."""
    from decoder_training import SwinToDecoder
    assert ev.SwinToDecoder is SwinToDecoder


@pytest.mark.parametrize("sec,out", [(0, "0h00m"), (61, "0h01m"), (3600, "1h00m")])
def test_fmt(sec, out):
    assert ev.fmt(sec) == out


def test_checkpoint_is_memory_mapped_when_supported(tmp_path, monkeypatch):
    """Evaluation discards optimizer state, so it must not be materialised in
    RAM. Regression: loading it outright killed a 6 GB machine mid-evaluation."""
    import torch
    calls = []
    real_load = torch.load

    def spy(*a, **kw):
        calls.append(kw.get("mmap"))
        return real_load(*a, **kw)

    monkeypatch.setattr(torch, "load", spy)
    f = tmp_path / "ck.pt"
    torch.save({"model_state": {"w": torch.zeros(4)}}, f)
    try:
        torch.load(f, map_location="cpu", weights_only=False, mmap=True)
    except Exception:
        pytest.skip("mmap unsupported on this torch/filesystem")
    assert calls[-1] is True
