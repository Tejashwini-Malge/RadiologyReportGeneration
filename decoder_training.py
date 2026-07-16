"""
03_train_decoder.py
Train the BioBART decoder to generate captions from FROZEN Swin features.
A small projection (1024 -> 768) + BioBART decoder are the ONLY trainable parts.

Resume-capable: checkpoints store optimizer / scheduler / scaler state, so a run
stopped after epoch N continues correctly from epoch N+1.

Usage:
    python 03_train_decoder.py
    python 03_train_decoder.py --smoke --steps 500
    python 03_train_decoder.py --resume D:/rrg/checkpoints/last.pt
    python 03_train_decoder.py --resume auto        # picks up last.pt if it exists
"""
import argparse
import json
import time
from pathlib import Path

import yaml
import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          get_linear_schedule_with_warmup)
from transformers.modeling_outputs import BaseModelOutput

CFG = yaml.safe_load(open(Path(__file__).parent / "config.yaml", encoding="utf-8"))
P, T = CFG["paths"], CFG["train"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# read with fallbacks so config.yaml / make_config.py need no changes
PATIENCE = T.get("early_stop_patience", 2)
NUM_WORKERS = T.get("num_workers", 0)  # 0 on Windows, 2-4 on Colab/Linux


class FeatureCaptionDS(Dataset):
    """Reads one split's .h5 (features + caption). Everything it needs is in the file."""
    def __init__(self, split):
        self.path = Path(P["features_dir"]) / f"{split}.h5"
        with h5py.File(self.path, "r") as h5:
            self.n = h5["features"].shape[0]
        self.h5 = None  # opened lazily per worker

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        if self.h5 is None:
            self.h5 = h5py.File(self.path, "r")
        feat = torch.from_numpy(self.h5["features"][i].astype(np.float32))  # [49,1024]
        cap = self.h5["caption"][i]
        cap = cap.decode("utf-8") if isinstance(cap, bytes) else str(cap)
        return feat, cap


def build_collate(tokenizer):
    def collate(batch):
        feats = torch.stack([b[0] for b in batch])          # [B,49,1024]
        caps = [b[1] for b in batch]
        tok = tokenizer(caps, padding=True, truncation=True,
                        max_length=T["max_caption_tokens"], return_tensors="pt")
        labels = tok["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100      # ignore pad in loss
        return feats, labels
    return collate


class SwinToBioBART(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(T["visual_dim"], T["decoder_dim"])   # 1024 -> 768
        self.bart = AutoModelForSeq2SeqLM.from_pretrained(CFG["models"]["decoder"])

    def forward(self, feats, labels=None):
        enc = self.proj(feats)                                # [B,49,768]
        attn = torch.ones(enc.shape[:2], dtype=torch.long, device=enc.device)
        enc_out = BaseModelOutput(last_hidden_state=enc)
        return self.bart(encoder_outputs=enc_out, attention_mask=attn, labels=labels)

    @torch.no_grad()
    def generate(self, feats, max_new_tokens=None):
        if max_new_tokens is None:
            max_new_tokens = T["max_caption_tokens"]
        enc = self.proj(feats)
        attn = torch.ones(enc.shape[:2], dtype=torch.long, device=enc.device)
        enc_out = BaseModelOutput(last_hidden_state=enc)
        return self.bart.generate(encoder_outputs=enc_out, attention_mask=attn,
                                  max_new_tokens=max_new_tokens, num_beams=4)


# ---------------- checkpointing ----------------

def ckpt_dir():
    d = Path(P["checkpoints_dir"])
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_ckpt(model, opt, sched, scaler, epoch, val_loss, best_val, history, tag):
    """tag: 'epoch03' -> decoder_epoch03.pt, or 'last' / 'best'."""
    path = ckpt_dir() / (f"decoder_{tag}.pt" if tag.startswith("epoch") else f"{tag}.pt")
    torch.save({
        "epoch": epoch,
        "val_loss": val_loss,
        "best_val": best_val,
        "history": history,
        "model_state": model.state_dict(),
        "opt_state": opt.state_dict(),
        "sched_state": sched.state_dict(),
        "scaler_state": scaler.state_dict(),
        "config": CFG,
    }, path)
    print(f"    saved {path.name}  (val_loss={val_loss:.4f})")


def load_ckpt(path, model, opt, sched, scaler):
    ck = torch.load(path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ck["model_state"])
    opt.load_state_dict(ck["opt_state"])
    sched.load_state_dict(ck["sched_state"])
    scaler.load_state_dict(ck["scaler_state"])
    print(f"resumed from {Path(path).name}: "
          f"epoch {ck['epoch']} done, val_loss={ck['val_loss']:.4f}")
    return ck["epoch"], ck.get("best_val", float("inf")), ck.get("history", [])


def write_history(history):
    with open(ckpt_dir() / "history.json", "w") as f:
        json.dump(history, f, indent=2)


# ---------------- eval ----------------

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    tot, n = 0.0, 0
    for feats, labels in loader:
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        with torch.autocast("cuda", enabled=(DEVICE == "cuda")):
            loss = model(feats, labels=labels).loss
        tot += loss.item() * feats.size(0)
        n += feats.size(0)
    return tot / max(n, 1)


def fmt(sec):
    h, m = divmod(int(sec) // 60, 60)
    return f"{h}h{m:02d}m"


# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=50, help="steps for --smoke")
    ap.add_argument("--resume", type=str, default=None,
                    help="path to a .pt, or 'auto' to use checkpoints/last.pt")
    args = ap.parse_args()

    print(f"Device: {DEVICE}   decoder: {CFG['models']['decoder']}")
    tokenizer = AutoTokenizer.from_pretrained(CFG["models"]["decoder"])
    collate = build_collate(tokenizer)

    train_ds = FeatureCaptionDS("train")
    val_ds = FeatureCaptionDS("valid")
    train_dl = DataLoader(train_ds, batch_size=T["batch_size"], shuffle=True,
                          collate_fn=collate, num_workers=NUM_WORKERS)
    val_dl = DataLoader(val_ds, batch_size=T["batch_size"], shuffle=False,
                        collate_fn=collate, num_workers=NUM_WORKERS)

    model = SwinToBioBART().to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=T["lr"])
    total_steps = len(train_dl) * T["epochs"] // T["grad_accum"]
    sched = get_linear_schedule_with_warmup(opt, T["warmup_steps"], total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

    start_epoch, best_val, history = 1, float("inf"), []
    if args.resume:
        path = ckpt_dir() / "last.pt" if args.resume == "auto" else Path(args.resume)
        if path.exists():
            done, best_val, history = load_ckpt(path, model, opt, sched, scaler)
            start_epoch = done + 1
        elif args.resume != "auto":
            raise FileNotFoundError(path)
        else:
            print("resume auto: no last.pt found, starting fresh.")

    if start_epoch > T["epochs"]:
        print(f"nothing to do: {T['epochs']} epochs already complete.")
        return

    bad_epochs = 0
    for epoch in range(start_epoch, T["epochs"] + 1):
        model.train()
        opt.zero_grad()
        t0 = time.time()
        for step, (feats, labels) in enumerate(train_dl):
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            with torch.autocast("cuda", enabled=(DEVICE == "cuda")):
                loss = model(feats, labels=labels).loss / T["grad_accum"]
            scaler.scale(loss).backward()

            if (step + 1) % T["grad_accum"] == 0:
                scaler.step(opt); scaler.update(); sched.step(); opt.zero_grad()

            if step % 50 == 0:
                el = time.time() - t0
                eta = el / max(step, 1) * (len(train_dl) - step)
                print(f"  epoch {epoch} step {step}/{len(train_dl)} "
                      f"loss {loss.item() * T['grad_accum']:.4f} "
                      f"| elapsed {fmt(el)} eta {fmt(eta)}")
            if args.smoke and step >= args.steps:
                print(f"  smoke run: stopping after {args.steps} steps."); return

        val_loss = evaluate(model, val_dl)
        train_min = (time.time() - t0) / 60
        print(f"epoch {epoch} done. val_loss={val_loss:.4f} ({train_min:.1f} min)")

        history.append({"epoch": epoch, "val_loss": val_loss, "minutes": round(train_min, 1)})
        write_history(history)

        save_ckpt(model, opt, sched, scaler, epoch, val_loss, best_val, history, "last")
        if T.get("save_every_epoch", True):
            save_ckpt(model, opt, sched, scaler, epoch, val_loss, best_val, history,
                      f"epoch{epoch:02d}")

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            save_ckpt(model, opt, sched, scaler, epoch, val_loss, best_val, history, "best")
        else:
            bad_epochs += 1
            print(f"    no improvement ({bad_epochs}/{PATIENCE})")
            if bad_epochs >= PATIENCE:
                print(f"early stop: val_loss has not improved for {PATIENCE} epochs.")
                break

    print(f"\nDone. best val_loss={best_val:.4f}  ->  {ckpt_dir() / 'best.pt'}")
    print("Copy the checkpoints dir off the machine NOW.")


if __name__ == "__main__":
    main()