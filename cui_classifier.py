"""
cui_classifier.py
Stage 1 of concept conditioning: predict UMLS CUIs from the FROZEN Swin pooler feature.

This exists to answer ONE question before any fusion code gets written:
    do the frozen ImageNet-Swin features carry medical concept signal at all?

The answer is the gap between the model and the PRIOR BASELINE (always predict the
k most frequent CUIs, k tuned on val). If the model does not clearly beat that,
the features are the bottleneck -- not the decoder, not the fusion module -- and
CUI conditioning built on these features cannot work.

Needs only the CUI *codes*. The UMLS code->term mapping (for RadBERT) is a separate
problem and is not required here.

Usage:
    python cui_classifier.py --limit 2000     # dry run
    python cui_classifier.py
    python cui_classifier.py --min-freq 50 --hidden 1024

Outputs (to <checkpoints_dir>/):
    cui_classifier.pt        weights + vocab + tuned threshold
    cui_vocab.json           code -> index, with train frequencies
    cui_metrics.json         model vs baseline -- READ THIS
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn as nn

from rrg_config import load_config

CFG = load_config()
P, T = CFG["paths"], CFG["train"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CUI_RE = re.compile(r"C\d{6,}")


def parse_cuis(s):
    """The cui column was written with str() over whatever pandas held -- a python
    list gives "['C0004238', 'C1']" and a numpy array gives "['C0004238' 'C1']".
    Regex sidesteps both. Returns [] if nothing matches."""
    if isinstance(s, bytes):
        s = s.decode("utf-8")
    return CUI_RE.findall(str(s))


def load_split(split, limit=None):
    path = Path(P["features_dir"]) / f"{split}.h5"
    if not path.exists():
        raise FileNotFoundError(f"{path} -- run extract_features.py for split '{split}' first")
    with h5py.File(path, "r") as h5:
        n = h5["pooler"].shape[0]
        if limit:
            n = min(limit, n)
        X = h5["pooler"][:n].astype(np.float32)          # [n,1024]
        cui_raw = h5["cui"][:n]
    Y = [parse_cuis(c) for c in cui_raw]
    empty = sum(1 for y in Y if not y)
    print(f"{split}: {n:,} rows, {empty:,} with no parsable CUI "
          f"({empty / max(n,1):.1%})")
    return X, Y


def build_vocab(Y_train, min_freq):
    freq = Counter(c for y in Y_train for c in y)
    keep = sorted([c for c, f in freq.items() if f >= min_freq])
    print(f"vocab: {len(keep):,} CUIs with freq >= {min_freq} "
          f"(of {len(freq):,} distinct)")
    return {c: i for i, c in enumerate(keep)}, freq


def encode(Y, vocab):
    M = np.zeros((len(Y), len(vocab)), dtype=np.float32)
    for i, y in enumerate(Y):
        for c in y:
            j = vocab.get(c)
            if j is not None:
                M[i, j] = 1.0
    return M


class CuiHead(nn.Module):
    def __init__(self, in_dim, n_cls, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_cls),
        )

    def forward(self, x):
        return self.net(x)


# ---------------- metrics ----------------

def prf(pred, true):
    """Micro and macro P/R/F1 for binary multi-label matrices."""
    tp = (pred * true).sum(0)
    fp = (pred * (1 - true)).sum(0)
    fn = ((1 - pred) * true).sum(0)

    mi_p = tp.sum() / max(tp.sum() + fp.sum(), 1e-9)
    mi_r = tp.sum() / max(tp.sum() + fn.sum(), 1e-9)
    mi_f = 2 * mi_p * mi_r / max(mi_p + mi_r, 1e-9)

    p = tp / np.maximum(tp + fp, 1e-9)
    r = tp / np.maximum(tp + fn, 1e-9)
    f = 2 * p * r / np.maximum(p + r, 1e-9)
    support = true.sum(0) > 0

    return {
        "micro_P": float(mi_p), "micro_R": float(mi_r), "micro_F1": float(mi_f),
        "macro_F1": float(f[support].mean()) if support.any() else 0.0,
        "n_labels_with_support": int(support.sum()),
    }


def tune_threshold(logits, true):
    best = (0.5, -1.0)
    probs = 1 / (1 + np.exp(-logits))
    for t in np.arange(0.05, 0.75, 0.05):
        f = prf((probs >= t).astype(np.float32), true)["micro_F1"]
        if f > best[1]:
            best = (float(t), f)
    return best


def prior_baseline(Y_tr_mat, Y_val_mat):
    """Ignore the image entirely: always emit the k most frequent train CUIs.
    Sweep k, keep the best. This is the number the model has to beat."""
    order = np.argsort(-Y_tr_mat.sum(0))
    best = {"k": 0, "micro_F1": -1.0}
    for k in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]:
        pred = np.zeros_like(Y_val_mat)
        pred[:, order[:k]] = 1.0
        m = prf(pred, Y_val_mat)
        if m["micro_F1"] > best["micro_F1"]:
            best = {"k": k, **m}
    return best


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-freq", type=int, default=20,
                    help="drop CUIs rarer than this in train")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    print(f"Device: {DEVICE}")
    Xtr, Ytr = load_split("train", args.limit)
    Xva, Yva = load_split("valid", args.limit)

    vocab, freq = build_vocab(Ytr, args.min_freq)
    if not vocab:
        raise SystemExit("empty vocab -- lower --min-freq, or CUI parsing failed")

    Mtr, Mva = encode(Ytr, vocab), encode(Yva, vocab)
    print(f"labels/sample: train {Mtr.sum(1).mean():.2f}, val {Mva.sum(1).mean():.2f}")

    base = prior_baseline(Mtr, Mva)
    print(f"\nPRIOR BASELINE (no image): k={base['k']}  "
          f"micro_F1={base['micro_F1']:.4f}\n")

    model = CuiHead(Xtr.shape[1], len(vocab), args.hidden, args.dropout).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()

    Xtr_t = torch.from_numpy(Xtr)
    Mtr_t = torch.from_numpy(Mtr)
    Xva_t = torch.from_numpy(Xva).to(DEVICE)

    best_f1, best_state, best_thr, bad = -1.0, None, 0.5, 0
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(Xtr_t))
        tot = 0.0
        for i in range(0, len(perm), args.batch_size):
            idx = perm[i:i + args.batch_size]
            xb = Xtr_t[idx].to(DEVICE)
            yb = Mtr_t[idx].to(DEVICE)
            loss = lossf(model(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)

        model.eval()
        with torch.no_grad():
            logits = model(Xva_t).cpu().numpy()
        thr, f1 = tune_threshold(logits, Mva)
        print(f"  epoch {ep:2d}  train_loss {tot/len(perm):.4f}  "
              f"val_micro_F1 {f1:.4f} @thr {thr:.2f}")

        if f1 > best_f1:
            best_f1, best_thr, bad = f1, thr, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= args.patience:
                print(f"early stop: no gain for {args.patience} epochs")
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(Xva_t).cpu().numpy()
    probs = 1 / (1 + np.exp(-logits))
    metrics = prf((probs >= best_thr).astype(np.float32), Mva)
    metrics.update({"threshold": best_thr, "vocab_size": len(vocab),
                    "min_freq": args.min_freq,
                    "mean_preds_per_sample": float((probs >= best_thr).sum(1).mean())})

    out = {"model": metrics, "prior_baseline": base,
           "micro_F1_gain_over_prior": metrics["micro_F1"] - base["micro_F1"]}

    d = Path(P["checkpoints_dir"]); d.mkdir(parents=True, exist_ok=True)
    torch.save({"state": best_state, "vocab": vocab, "threshold": best_thr,
                "in_dim": Xtr.shape[1], "hidden": args.hidden,
                "dropout": args.dropout, "metrics": out}, d / "cui_classifier.pt")
    json.dump({"vocab": vocab, "train_freq": {c: freq[c] for c in vocab}},
              open(d / "cui_vocab.json", "w"), indent=2)
    json.dump(out, open(d / "cui_metrics.json", "w"), indent=2)

    print("\n--- result ---")
    print(f"  model  micro_F1  {metrics['micro_F1']:.4f}  "
          f"(P {metrics['micro_P']:.4f} R {metrics['micro_R']:.4f})")
    print(f"  model  macro_F1  {metrics['macro_F1']:.4f}")
    print(f"  prior  micro_F1  {base['micro_F1']:.4f}  (k={base['k']})")
    print(f"  GAIN             {out['micro_F1_gain_over_prior']:+.4f}")
    print(f"\nwrote -> {d}")


if __name__ == "__main__":
    main()