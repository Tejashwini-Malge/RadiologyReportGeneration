"""CUI parsing, label encoding, and the metric math the experiment's
conclusion rests on. If prf() or prior_baseline() are wrong, the
'do frozen Swin features carry concept signal' answer is wrong."""
import numpy as np
import pytest

import cui_classifier as cui


# ---------------- parse_cuis ----------------

def test_parses_the_python_list_repr():
    assert cui.parse_cuis("['C0004238', 'C0817096']") == ["C0004238", "C0817096"]


def test_parses_the_numpy_array_repr():
    """Same data, different str() -- no commas. This is why it is a regex."""
    assert cui.parse_cuis("['C0004238' 'C0817096']") == ["C0004238", "C0817096"]


def test_parses_bytes_from_h5():
    assert cui.parse_cuis(b"['C0004238']") == ["C0004238"]


@pytest.mark.parametrize("junk", ["", "nan", "None", "[]", "no codes here"])
def test_unparsable_cells_give_an_empty_list_not_an_error(junk):
    assert cui.parse_cuis(junk) == []


def test_codes_shorter_than_six_digits_are_not_matched():
    """Documents actual behaviour: CUI_RE requires six or more digits, so the
    'C1' in the parse_cuis docstring example is dropped, not parsed."""
    assert cui.parse_cuis("['C0004238' 'C1']") == ["C0004238"]


# ---------------- vocab / encoding ----------------

def test_build_vocab_drops_rare_codes_and_indexes_in_sorted_order():
    Y = [["C000002"], ["C000002"], ["C000001"], ["C000003", "C000002"]]
    vocab, freq = cui.build_vocab(Y, min_freq=2)
    assert vocab == {"C000002": 0}
    assert freq["C000001"] == 1 and freq["C000002"] == 3


def test_build_vocab_indexes_multiple_codes_in_sorted_order():
    vocab, _ = cui.build_vocab([["C000003", "C000001", "C000002"]], min_freq=1)
    assert vocab == {"C000001": 0, "C000002": 1, "C000003": 2}


def test_build_vocab_can_be_empty():
    vocab, _ = cui.build_vocab([["C000001"]], min_freq=99)
    assert vocab == {}


def test_encode_sets_one_hot_columns_and_ignores_out_of_vocab_codes():
    vocab = {"C000001": 0, "C000002": 1}
    M = cui.encode([["C000001"], ["C000002", "C999999"], []], vocab)
    assert M.tolist() == [[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]


def test_encode_is_idempotent_for_duplicate_codes():
    assert cui.encode([["C000001", "C000001"]], {"C000001": 0}).tolist() == [[1.0]]


# ---------------- prf ----------------

def test_prf_against_hand_computed_values():
    pred = np.array([[1, 0], [1, 1]], dtype=np.float32)
    true = np.array([[1, 1], [0, 1]], dtype=np.float32)
    # tp=[1,1] fp=[1,0] fn=[0,1] -> micro P=R=F1=2/3
    # per-label f=[2/3, 2/3], both supported -> macro 2/3
    m = cui.prf(pred, true)
    assert m["micro_P"] == pytest.approx(2 / 3)
    assert m["micro_R"] == pytest.approx(2 / 3)
    assert m["micro_F1"] == pytest.approx(2 / 3)
    assert m["macro_F1"] == pytest.approx(2 / 3)
    assert m["n_labels_with_support"] == 2


def test_prf_is_perfect_when_pred_equals_true():
    true = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    m = cui.prf(true.copy(), true)
    assert m["micro_F1"] == pytest.approx(1.0)
    assert m["macro_F1"] == pytest.approx(1.0)


def test_prf_predicting_nothing_is_zero_not_nan():
    true = np.array([[1, 1]], dtype=np.float32)
    m = cui.prf(np.zeros_like(true), true)
    assert m["micro_F1"] == pytest.approx(0.0)
    assert not np.isnan(m["macro_F1"])


def test_macro_f1_only_averages_labels_that_have_support():
    """Label 1 never appears in truth. Including it would drag macro_F1 down
    with a meaningless zero."""
    pred = np.array([[1, 0]], dtype=np.float32)
    true = np.array([[1, 0]], dtype=np.float32)
    m = cui.prf(pred, true)
    assert m["n_labels_with_support"] == 1
    assert m["macro_F1"] == pytest.approx(1.0)


# ---------------- threshold / baseline ----------------

def test_tune_threshold_picks_a_cut_that_separates_the_classes():
    logits = np.array([[2.0], [-2.0]], dtype=np.float32)   # probs ~ .88 / .12
    true = np.array([[1.0], [0.0]], dtype=np.float32)
    thr, f1 = cui.tune_threshold(logits, true)
    assert f1 == pytest.approx(1.0)
    assert 0.12 < thr < 0.88


def test_prior_baseline_ignores_the_image_and_emits_the_top_k_train_cuis():
    Mtr = np.array([[1, 0]] * 8 + [[1, 1]] * 2, dtype=np.float32)
    Mva = np.array([[1, 0]] * 4, dtype=np.float32)
    base = cui.prior_baseline(Mtr, Mva)
    assert base["k"] == 1
    assert base["micro_F1"] == pytest.approx(1.0)


def test_prior_baseline_is_beatable_when_labels_are_image_dependent():
    """The whole experiment is 'model micro_F1 minus this'. If a per-sample
    label pattern still scored 1.0 here, the comparison would be meaningless."""
    Mtr = np.array([[1, 0], [0, 1]] * 5, dtype=np.float32)
    Mva = np.array([[1, 0], [0, 1]] * 2, dtype=np.float32)
    assert cui.prior_baseline(Mtr, Mva)["micro_F1"] < 1.0


# ---------------- head ----------------

def test_cui_head_maps_pooler_dim_to_vocab_dim():
    import torch
    head = cui.CuiHead(in_dim=1024, n_cls=7, hidden=16, dropout=0.0)
    assert head(torch.zeros(3, 1024)).shape == (3, 7)
