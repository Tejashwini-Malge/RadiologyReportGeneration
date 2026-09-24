"""
evaluate.py
Load a trained checkpoint, generate captions for a split, dump predictions,
and score them with BLEU-1..4 / ROUGE-1,2,L / BERTScore.

GREEN is NOT here on purpose -- it is a 7B LLM and needs its own run. See notes.

Usage:
    python evaluate.py --ckpt /content/drive/MyDrive/rrg/checkpoints/best.pt --limit 64
    python evaluate.py --ckpt /content/drive/MyDrive/rrg/checkpoints/best.pt
    python evaluate.py --ckpt best.pt --split valid --no-bertscore

Outputs (to <checkpoints_dir>/eval/):
    predictions_<split>.json   every (pred, ref) pair
    samples_<split>.txt        first 30 pairs, human-readable -- READ THIS FIRST
    metrics_<split>.json       the numbers

Deps:
    pip install -r requirements.txt
"""
import argparse
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

from decoder_training import SwinToDecoder
from rrg_config import load_config, enable_utf8_stdout

CFG = load_config()
enable_utf8_stdout()
P, T = CFG["paths"], CFG["train"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class FeatureRefDS(Dataset):
    """Like the training dataset, but returns the caption as a raw string (the reference)."""
    def __init__(self, split, limit=None):
        self.path = Path(P["features_dir"]) / f"{split}.h5"
        if not self.path.exists():
            raise FileNotFoundError(f"{self.path} -- run extract_features.py for split '{split}' first")
        with h5py.File(self.path, "r") as h5:
            self.n = h5["features"].shape[0]
        if limit:
            self.n = min(limit, self.n)
        self.h5 = None

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.h5 is None:
            self.h5 = h5py.File(self.path, "r")
        feat = torch.from_numpy(self.h5["features"][i].astype(np.float32))
        cap = self.h5["caption"][i]
        cap = cap.decode("utf-8") if isinstance(cap, bytes) else str(cap)
        return feat, cap


def collate(batch):
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def fmt(sec):
    h, m = divmod(int(sec) // 60, 60)
    return f"{h}h{m:02d}m"


# ---------------- generation ----------------

@torch.no_grad()
def generate_all(model, tokenizer, loader, beams, max_new):
    model.eval()
    preds, refs = [], []
    t0 = time.time()
    for bi, (feats, caps) in enumerate(loader):
        feats = feats.to(DEVICE)
        out = model.generate(feats, max_new_tokens=max_new, num_beams=beams)
        preds += tokenizer.batch_decode(out, skip_special_tokens=True)
        refs += caps
        if bi % 10 == 0:
            el = time.time() - t0
            eta = el / max(bi, 1) * (len(loader) - bi)
            print(f"  batch {bi}/{len(loader)}  elapsed {fmt(el)} eta {fmt(eta)}")
    return [p.strip() for p in preds], [r.strip() for r in refs]


# ---------------- metrics ----------------

def compute_bleu(preds, refs):
    """BLEU-1..4, corpus level. This is what the RRG literature reports."""
    from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
    hyp = [p.lower().split() for p in preds]
    ref = [[r.lower().split()] for r in refs]
    sm = SmoothingFunction().method1
    out = {}
    for n in (1, 2, 3, 4):
        w = tuple([1.0 / n] * n)
        out[f"BLEU-{n}"] = corpus_bleu(ref, hyp, weights=w, smoothing_function=sm)
    return out


def compute_rouge(preds, refs):
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
    acc = {"ROUGE-1": [], "ROUGE-2": [], "ROUGE-L": []}
    for p, r in zip(preds, refs):
        s = sc.score(r, p)
        acc["ROUGE-1"].append(s["rouge1"].fmeasure)
        acc["ROUGE-2"].append(s["rouge2"].fmeasure)
        acc["ROUGE-L"].append(s["rougeL"].fmeasure)
    return {k: float(np.mean(v)) for k, v in acc.items()}


def compute_bertscore(preds, refs, model_type):
    from bert_score import score
    Pr, Rc, F1 = score(preds, refs, lang="en", model_type=model_type,
                       verbose=False, batch_size=32,
                       device=DEVICE, rescale_with_baseline=False)
    return {"BERTScore-P": float(Pr.mean()),
            "BERTScore-R": float(Rc.mean()),
            "BERTScore-F1": float(F1.mean())}


def degeneracy_report(preds):
    """Cheap sanity signals. A high top-1 share means the model emits one generic
    sentence for everything -- BLEU can still look OK when this happens."""
    from collections import Counter
    c = Counter(preds)
    top, topn = c.most_common(1)[0]
    lens = [len(p.split()) for p in preds]
    return {
        "n": len(preds),
        "unique_preds": len(c),
        "unique_ratio": round(len(c) / max(len(preds), 1), 4),
        "most_common_pred": top,
        "most_common_share": round(topn / max(len(preds), 1), 4),
        "mean_pred_len_words": round(float(np.mean(lens)), 2),
        "empty_preds": sum(1 for p in preds if not p),
    }


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["test", "valid", "train"])
    ap.add_argument("--limit", type=int, default=None, help="cap rows (dry run)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--beams", type=int, default=4)
    ap.add_argument("--no-bertscore", action="store_true")
    ap.add_argument("--bertscore-model", default="roberta-large")
    args = ap.parse_args()

    # Evaluation only needs model_state, but the checkpoint also carries the
    # optimizer/scheduler state needed for --resume (~1.1 GB for BioBART-base).
    # mmap leaves those on disk instead of materialising them in RAM; without it
    # a low-memory machine can die loading a checkpoint it is about to discard.
    try:
        ck = torch.load(args.ckpt, map_location=DEVICE, weights_only=False, mmap=True)
    except (TypeError, RuntimeError, ValueError):
        ck = torch.load(args.ckpt, map_location=DEVICE, weights_only=False)
    tcfg = ck.get("config", {}).get("train", {})
    print(f"ckpt: {Path(args.ckpt).name}  epoch={ck.get('epoch')}  "
          f"val_loss={ck.get('val_loss'):.4f}")

    # Guard: the model is rebuilt from THIS config.yaml. If it disagrees with the
    # config the checkpoint was trained under, load_state_dict fails or silently
    # mismatches. Check the keys that determine tensor shapes.
    for k in ("visual_dim", "decoder_dim"):
        if k in tcfg and tcfg[k] != T[k]:
            raise SystemExit(
                f"config mismatch: {k} is {T[k]} here but {tcfg[k]} in the ckpt. "
                f"Fix config.yaml before evaluating."
            )
    max_new = tcfg.get("max_caption_tokens", T["max_caption_tokens"])
    if tcfg.get("max_caption_tokens") != T.get("max_caption_tokens"):
        print(f"  note: using ckpt's max_caption_tokens={max_new}")

    tokenizer = AutoTokenizer.from_pretrained(CFG["models"]["decoder"])
    model = SwinToDecoder().to(DEVICE)
    model.load_state_dict(ck["model_state"])

    ds = FeatureRefDS(args.split, limit=args.limit)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                    collate_fn=collate, num_workers=T.get("num_workers", 0))
    print(f"{args.split}: {len(ds):,} rows, beams={args.beams}, max_new={max_new}")

    preds, refs = generate_all(model, tokenizer, dl, args.beams, max_new)

    outdir = Path(P["checkpoints_dir"]) / "eval"
    outdir.mkdir(parents=True, exist_ok=True)

    with open(outdir / f"predictions_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump([{"pred": p, "ref": r} for p, r in zip(preds, refs)], f, indent=2)

    with open(outdir / f"samples_{args.split}.txt", "w", encoding="utf-8") as f:
        for i in range(min(30, len(preds))):
            f.write(f"[{i}]\nPRED: {preds[i]}\nREF : {refs[i]}\n\n")

    print("\n--- first 5 ---")
    for i in range(min(5, len(preds))):
        print(f"PRED: {preds[i]}\nREF : {refs[i]}\n")

    metrics = {"checkpoint": str(args.ckpt), "split": args.split,
               "n": len(preds), "beams": args.beams,
               "val_loss_at_ckpt": ck.get("val_loss")}
    metrics["degeneracy"] = degeneracy_report(preds)
    print("degeneracy:", json.dumps(metrics["degeneracy"], indent=2))

    metrics.update(compute_bleu(preds, refs))
    metrics.update(compute_rouge(preds, refs))
    if not args.no_bertscore:
        metrics.update(compute_bertscore(preds, refs, args.bertscore_model))
        metrics["bertscore_model"] = args.bertscore_model

    with open(outdir / f"metrics_{args.split}.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("\n--- metrics ---")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:16s} {v:.4f}")
    print(f"\nwrote -> {outdir}")


if __name__ == "__main__":
    main()