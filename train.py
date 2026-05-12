"""
Training loop cho M1 (HaLong + CrossTurnAttention + Noisy-OR Focal Loss).

- Binary classification: scam vs harmless
- Single schedule + optional encoder unfreeze at unfreeze_epoch
"""

import argparse
import dataclasses
import json
import os
import random
import sys
import time

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from config import M1Config
from dataset import DialogueDataset, collate_fn, load_json, truncate_augment
from model import M1Classifier
from metrics import compute_streaming_metrics, print_streaming_report


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Evaluation ─────────────────────────────────────────────────────

@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()
    total_loss, n_dlg = 0.0, 0
    all_labels, all_d_probs, all_p_agg = [], [], []

    for batch in loader:
        input_ids  = batch["input_ids"].to(device)
        attn_masks = batch["attn_masks"].to(device)
        turn_mask  = batch["turn_mask"].to(device)
        labels     = batch["labels"].to(device)
        n_turns    = batch["n_turns"]

        output = model(input_ids, attn_masks, turn_mask, labels=labels)
        B = labels.shape[0]
        if output["loss"] is not None:
            total_loss += output["loss"].item() * B
        n_dlg += B

        for b in range(B):
            n = int(n_turns[b].item())
            all_labels.append(int(labels[b].item()))
            all_d_probs.append(float(output["dialogue_probs"][b].item()))
            all_p_agg.append(output["p_agg"][b, :n].cpu().numpy())

        del output
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = compute_streaming_metrics(all_labels, all_d_probs, all_p_agg, threshold)
    metrics["loss"] = total_loss / max(n_dlg, 1)
    return metrics


# ── Training epoch ─────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, scheduler, device, cfg, global_step: int):
    model.train()
    total_loss, n_dlg = 0.0, 0
    optimizer.zero_grad()

    for step, batch in enumerate(loader):
        input_ids  = batch["input_ids"].to(device)
        attn_masks = batch["attn_masks"].to(device)
        turn_mask  = batch["turn_mask"].to(device)
        labels     = batch["labels"].to(device)

        output = model(input_ids, attn_masks, turn_mask, labels=labels)
        loss   = output["loss"] / cfg.grad_accum_steps
        loss.backward()

        B = labels.shape[0]
        total_loss += output["loss"].item() * B
        n_dlg      += B

        if (step + 1) % cfg.grad_accum_steps == 0 or (step + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

        del output, loss
        if step % 20 == 0 and device.type == "cuda":
            torch.cuda.empty_cache()

        if (step + 1) % 5 == 0 or (step + 1) == len(loader):
            avg = total_loss / max(n_dlg, 1)
            lr  = scheduler.get_last_lr()[0]
            print(f"  [{step+1}/{len(loader)}] loss={avg:.4f} lr={lr:.2e}")

    return total_loss / max(n_dlg, 1), global_step


# ── Utilities ──────────────────────────────────────────────────────

def _save_model(model, tokenizer, cfg, val_metrics):
    save_path = os.path.join(cfg.output_dir, "best_model")
    os.makedirs(save_path, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_path, "model.pt"))
    with open(os.path.join(save_path, "config.json"), "w") as f:
        json.dump(dataclasses.asdict(cfg), f, indent=2)
    tokenizer.save_pretrained(save_path)
    with open(os.path.join(save_path, "val_metrics.json"), "w") as f:
        json.dump({k: (float(v) if isinstance(v, (float, np.floating)) else v)
                   for k, v in val_metrics.items()}, f, indent=2)


def _print_val_metrics(epoch, elapsed, tr_loss, m):
    print(f"\n  Epoch {epoch} ({elapsed:.1f}s):")
    print(f"    Train loss:       {tr_loss:.4f}")
    print(f"    Val loss:         {m['loss']:.4f}")
    print(f"    Val acc:          {m['dialogue_accuracy']:.4f}")
    print(f"    Val F1:           {m['dialogue_f1']:.4f}")
    if not np.isnan(m.get("auroc", float("nan"))):
        print(f"    Val AUROC:        {m['auroc']:.4f}")
    print(f"    Detection rate:   {m['detection_rate']:.4f}  ({m['num_detected']}/{m['num_scam']})")
    if not np.isnan(m["avg_detection_delay"]):
        print(f"    Avg turn:         {m['avg_detection_delay']:.2f}")
    if "alert_at_half" in m:
        print(f"    Alert in 1st half:{m['alert_at_half']:.3f}")
    print(f"    False alarm rate: {m['false_alarm_rate']:.4f}  ({m['num_false_alarms']}/{m['num_harmless']})")


def _dataset_overview(train_dlg, val_dlg):
    print(f"\n{'='*60}")
    print(f"DATASET OVERVIEW")
    print(f"{'='*60}")
    for name, data in [("Train", train_dlg), ("Test", val_dlg)]:
        scam_n    = sum(1 for d in data if d["label"] == "scam")
        harm_n    = sum(1 for d in data if d["label"] == "harmless")
        turn_lens = [len(d["turns"]) for d in data]
        avg_turns = np.mean(turn_lens) if turn_lens else 0
        min_turns = min(turn_lens) if turn_lens else 0
        max_turns = max(turn_lens) if turn_lens else 0
        print(f"  {name:6s}: {len(data):5d} dialogues  "
              f"(scam={scam_n}, harmless={harm_n})  "
              f"turns: avg={avg_turns:.1f}, min={min_turns}, max={max_turns}")
    print(f"{'='*60}")


@torch.no_grad()
def _preview_stream(model, val_ds, val_dlg, device, cfg, max_samples=2):
    model.eval()
    if len(val_dlg) == 0:
        return

    scam_indices     = [i for i, d in enumerate(val_dlg) if d["label"] == "scam"]
    harmless_indices = [i for i, d in enumerate(val_dlg) if d["label"] == "harmless"]
    selected = []
    if scam_indices:
        selected.append(random.choice(scam_indices))
    if harmless_indices:
        selected.append(random.choice(harmless_indices))
    while len(selected) < max_samples and len(val_dlg) > len(selected):
        idx = random.randint(0, len(val_dlg) - 1)
        if idx not in selected:
            selected.append(idx)

    print(f"\n  -- Streaming Inference Preview --")
    for idx in selected:
        dlg   = val_dlg[idx]
        label = dlg["label"]
        turns = dlg["turns"]
        n     = min(len(turns), cfg.max_turns)

        item  = val_ds[idx]
        batch = {
            "input_ids":  item["input_ids"].unsqueeze(0).to(device),
            "attn_masks": item["attn_masks"].unsqueeze(0).to(device),
            "turn_mask":  item["turn_mask"].unsqueeze(0).to(device),
        }
        output = model(batch["input_ids"], batch["attn_masks"], batch["turn_mask"])
        probs  = output["turn_probs"][0, :n].cpu().tolist()
        p_agg_list = output["p_agg"][0, :n].cpu().tolist()
        d_prob = output["dialogue_probs"][0].item()
        pred   = "scam" if d_prob >= cfg.threshold else "harmless"

        status = "✓" if pred == label else "✗"
        print(f"    [{status}] true={label:8s} pred={pred:8s} p_agg={d_prob:.3f} ({n} turns)")
        for t in range(n):
            q  = probs[t]
            pa = p_agg_list[t]
            alert = " ← ALERT" if pa >= cfg.threshold else ""
            text  = turns[t][:50]
            print(f"      T{t+1:02d} q={q:.3f} p_agg={pa:.3f}{alert}  \"{text}{'...' if len(turns[t]) > 50 else ''}\"")
        print()


# ── Build optimizer ────────────────────────────────────────────────

def _build_optimizer_and_scheduler(model, cfg, total_steps, with_encoder=False):
    if with_encoder and cfg.unfreeze_layers > 0:
        encoder_params = [p for n, p in model.named_parameters()
                          if p.requires_grad and 'encoder' in n]
        head_params    = [p for n, p in model.named_parameters()
                          if p.requires_grad and 'encoder' not in n]
        optimizer = AdamW([
            {"params": head_params,    "lr": cfg.lr},
            {"params": encoder_params, "lr": cfg.encoder_lr},
        ], weight_decay=cfg.weight_decay)
    else:
        optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.lr, weight_decay=cfg.weight_decay,
        )

    warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    return optimizer, scheduler


# ── Dataset option mapping ─────────────────────────────────────────

DATASET_OPTIONS = {
    "real":      {"train": "real_1.json",         "test": "real_2.json"},
    "synthetic": {"train": "synthetic_data.json",  "test": "real_2.json"},
    "tele":      {"train": "tele_data.json",       "test": "real_2.json"},
    "real_syn":  {"train": "real_syn.json",        "test": "real_2.json"},
    "real_tele": {"train": "real_tele.json",       "test": "real_2.json"},
}


# ── Main training ──────────────────────────────────────────────────

def train(cfg: M1Config = None, dataset_option: str = None,
          train_file: str = None, test_file: str = None):
    if cfg is None:
        cfg = M1Config()

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load data ──
    if train_file and test_file:
        train_path = train_file
        val_path   = test_file
        print(f"\nUsing custom dataset paths:")
        print(f"  Train: {train_path}")
        print(f"  Test:  {val_path}")
    elif dataset_option and dataset_option in DATASET_OPTIONS:
        ds_files  = DATASET_OPTIONS[dataset_option]
        train_path = os.path.join(cfg.dataset_dir, ds_files["train"])
        val_path   = os.path.join(cfg.dataset_dir, ds_files["test"])
        print(f"\nDataset option: {dataset_option}")
        print(f"  Train: {ds_files['train']}")
        print(f"  Test:  {ds_files['test']}")
    else:
        train_path = os.path.join(cfg.data_dir, "train.json")
        val_path   = os.path.join(cfg.data_dir, "test.json")
        print(f"\nUsing default data from {cfg.data_dir}")

    for p in [train_path, val_path]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found.")
            return

    print(f"\nLoading data...")
    train_dlg = load_json(train_path)
    val_dlg   = load_json(val_path)

    if cfg.truncate_aug:
        before    = len(train_dlg)
        train_dlg = truncate_augment(train_dlg, cfg.aug_k, cfg.aug_min_turns)
        scam_n    = sum(1 for d in train_dlg if d["label"] == "scam")
        harm_n    = sum(1 for d in train_dlg if d["label"] == "harmless")
        print(f"Truncate aug: {before} → {len(train_dlg)} (scam={scam_n}, harmless={harm_n})")

    print(f"Train: {len(train_dlg)} | Test: {len(val_dlg)}")
    _dataset_overview(train_dlg, val_dlg)

    print(f"\nLoading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_ds = DialogueDataset(train_dlg, tokenizer, cfg.max_turn_len, cfg.max_turns)
    val_ds   = DialogueDataset(val_dlg,   tokenizer, cfg.max_turn_len, cfg.max_turns)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)

    print(f"\nLoading model: {cfg.model_name}")
    model = M1Classifier(cfg).to(device)

    steps_per_epoch = max(1, len(train_loader) // cfg.grad_accum_steps)
    total_steps = steps_per_epoch * cfg.num_epochs

    print(f"\n{'='*60}")
    print(f"TRAINING CONFIG")
    print(f"{'='*60}")
    print(f"  lr              = {cfg.lr}")
    print(f"  harm_weight     = {cfg.class_weight_harmless}")
    print(f"  lambda_aux      = {cfg.weighted_lambda}")
    print(f"  unfreeze_epoch  = {cfg.unfreeze_epoch}")
    print(f"  unfreeze_layers = {cfg.unfreeze_layers}")
    print(f"  encoder_lr      = {cfg.encoder_lr}")
    print(f"  Trainable params: {model.count_trainable_params():,}")

    optimizer, scheduler = _build_optimizer_and_scheduler(model, cfg, total_steps)

    os.makedirs(cfg.output_dir, exist_ok=True)
    best_val_acc  = 0.0
    best_epoch    = 0
    global_step   = 0
    encoder_unfrozen = False

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()

        # ── Unfreeze encoder ──
        if (epoch == cfg.unfreeze_epoch and not encoder_unfrozen
                and cfg.unfreeze_layers > 0):
            encoder_unfrozen = True
            model.unfreeze_last_n_layers(cfg.unfreeze_layers)
            print(f"\n  *** Encoder unfrozen at epoch {epoch} ***")
            print(f"  Trainable params: {model.count_trainable_params():,}")

            remaining_steps = steps_per_epoch * (cfg.num_epochs - epoch + 1)
            optimizer, scheduler = _build_optimizer_and_scheduler(
                model, cfg, remaining_steps, with_encoder=True
            )

        print(f"\nEpoch {epoch}/{cfg.num_epochs}")
        tr_loss, global_step = run_epoch(
            model, train_loader, optimizer, scheduler, device, cfg, global_step
        )

        val_metrics = evaluate(model, val_loader, device, cfg.threshold)
        _print_val_metrics(epoch, time.time() - t0, tr_loss, val_metrics)
        _preview_stream(model, val_ds, val_dlg, device, cfg)

        val_acc = val_metrics["dialogue_accuracy"]
        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_epoch    = epoch
            _save_model(model, tokenizer, cfg, val_metrics)
            print(f"    * Best model saved (val_acc={best_val_acc:.4f})")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nBest epoch: {best_epoch} | val_acc={best_val_acc:.4f}")

    best_pt = os.path.join(cfg.output_dir, "best_model", "model.pt")
    if os.path.exists(best_pt):
        model.load_state_dict(torch.load(best_pt, map_location=device, weights_only=True))
        final_metrics = evaluate(model, val_loader, device, cfg.threshold)
        print_streaming_report(final_metrics)

    return model


# ── CLI ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train M1: HaLong + CrossTurnAttention")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-file", default=None)
    parser.add_argument("--test-file",  default=None)
    parser.add_argument("--dataset",    default=None,
                        choices=list(DATASET_OPTIONS.keys()))
    parser.add_argument("--debug",      action="store_true",
                        help="2 epochs, batch_size=4, no aug")
    parser.add_argument("--small",      action="store_true",
                        help="5 epochs, batch_size=2")
    return parser.parse_args()


if __name__ == "__main__":
    cfg  = M1Config()
    args = parse_args()

    if args.output_dir:
        cfg.output_dir = args.output_dir
    elif args.dataset:
        cfg.output_dir = os.path.join(_dir, "outputs", args.dataset)

    if args.debug:
        cfg.num_epochs       = 2
        cfg.batch_size       = 4
        cfg.grad_accum_steps = 4
        cfg.warmup_ratio     = 0.03
        cfg.truncate_aug     = False
        cfg.unfreeze_epoch   = 99  # never unfreeze in debug
        print("DEBUG MODE: 2 epochs, batch_size=4, no aug")
    if args.small:
        cfg.num_epochs = 5
        cfg.batch_size = 2
        print("SMALL MODE: 5 epochs, batch_size=2")

    train(cfg, dataset_option=args.dataset,
          train_file=args.train_file, test_file=args.test_file)
