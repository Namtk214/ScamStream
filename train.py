"""
Training loop cho M1 Multi-Class (HaLong + CrossTurnAttention + Noisy-OR Focal CE).

- 5 classes: harmless (0), A (1), B (2), C (3), D (4)
- 3-phase training schedule with per-class weights
- Val = test file (dùng chung vì data hạn chế)
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

from config import M1Config, CLASS_NAMES, SCENARIO_TO_IDX
from dataset import DialogueDataset, collate_fn, load_json, truncate_augment, get_class_label
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
def evaluate(model, loader, device, cfg):
    model.eval()
    total_loss, n_dlg = 0.0, 0
    all_labels, all_preds, all_probs, all_p_agg = [], [], [], []

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
            all_preds.append(int(output["dialogue_preds"][b].item()))
            all_probs.append(output["dialogue_probs"][b].cpu().numpy())
            all_p_agg.append(output["p_agg"][b, :n].cpu().numpy())

        del output
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = compute_streaming_metrics(
        all_labels, all_preds, all_probs, all_p_agg,
        class_names=cfg.class_names,
        scam_alert_thresh=cfg.scam_alert_thresh,
    )
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
    # Save val metrics (convert numpy types)
    safe_metrics = {}
    for k, v in val_metrics.items():
        if isinstance(v, (float, np.floating)):
            safe_metrics[k] = float(v)
        elif isinstance(v, (int, np.integer)):
            safe_metrics[k] = int(v)
        elif isinstance(v, dict):
            safe_metrics[k] = {
                sk: float(sv) if isinstance(sv, (float, np.floating)) else sv
                for sk, sv in v.items()
            }
        else:
            safe_metrics[k] = v
    with open(os.path.join(save_path, "val_metrics.json"), "w") as f:
        json.dump(safe_metrics, f, indent=2, default=str)


def _print_val_metrics(epoch, elapsed, tr_loss, m):
    print(f"\n  Epoch {epoch} ({elapsed:.1f}s):")
    print(f"    Train loss:       {tr_loss:.4f}")
    print(f"    Val loss:         {m['loss']:.4f}")
    print(f"    Val accuracy:     {m['accuracy']:.4f}")
    print(f"    Val macro-F1:     {m['macro_f1']:.4f}")
    print(f"    Val weighted-F1:  {m['weighted_f1']:.4f}")
    print(f"    Per-class F1:     ", end="")
    for c in CLASS_NAMES:
        print(f"{c}={m['per_class_f1'][c]:.3f} ", end="")
    print()
    print(f"    Detection rate:   {m['overall_detection_rate']:.4f} "
          f"({m['num_scam_detected']}/{m['num_scam']})")
    print(f"    False alarm rate: {m['false_alarm_rate']:.4f} "
          f"({m['num_false_alarms']}/{m['num_harmless']})")
    if "alert_at_half" in m:
        print(f"    Alert in 1st half: {m['alert_at_half']:.3f}")


def _dataset_overview(train_dlg, val_dlg):
    """Print dataset overview before training."""
    print(f"\n{'='*60}")
    print(f"DATASET OVERVIEW (Multi-Class)")
    print(f"{'='*60}")
    for name, data in [("Train", train_dlg), ("Test", val_dlg)]:
        from collections import Counter
        class_dist = Counter(get_class_label(d) for d in data)
        turn_lens = [len(d["turns"]) for d in data]
        avg_turns = np.mean(turn_lens) if turn_lens else 0
        print(f"  {name:6s}: {len(data):5d} dialogues  "
              f"turns: avg={avg_turns:.1f}")
        for c_idx, c_name in enumerate(CLASS_NAMES):
            count = class_dist.get(c_idx, 0)
            print(f"    {c_name:10s}: {count:5d}")
    print(f"{'='*60}")


@torch.no_grad()
def _preview_stream(model, val_ds, val_dlg, device, cfg, max_samples=2):
    """Streaming inference preview on random validation samples after each epoch."""
    model.eval()
    if len(val_dlg) == 0:
        return

    # Try to pick 1 scam + 1 harmless
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

    print(f"\n  -- Streaming Inference Preview (Multi-Class) --")
    for idx in selected:
        dlg   = val_dlg[idx]
        label = dlg["label"]
        scenario = dlg.get("scenario", "none")
        true_class = get_class_label(dlg)
        turns = dlg["turns"]
        n     = min(len(turns), cfg.max_turns)

        item  = val_ds[idx]
        batch = {
            "input_ids":  item["input_ids"].unsqueeze(0).to(device),
            "attn_masks": item["attn_masks"].unsqueeze(0).to(device),
            "turn_mask":  item["turn_mask"].unsqueeze(0).to(device),
        }
        output = model(batch["input_ids"], batch["attn_masks"], batch["turn_mask"])
        pred_class = output["dialogue_preds"][0].item()
        d_probs = output["dialogue_probs"][0].cpu().numpy()

        true_name = cfg.class_names[true_class]
        pred_name = cfg.class_names[pred_class]
        status = "✓" if pred_class == true_class else "✗"
        print(f"    [{status}] true={true_name:10s} pred={pred_name:10s} "
              f"conf={d_probs[pred_class]:.3f} ({n} turns)")

        # Show per-turn probabilities
        for t in range(n):
            p_agg_t = output["p_agg"][0, t].cpu().numpy()
            p_norm = p_agg_t / (p_agg_t.sum() + 1e-8)
            top_class = int(p_norm.argmax())
            top_name = cfg.class_names[top_class]
            top_prob = p_norm[top_class]
            text = turns[t][:50]
            print(f"      T{t+1:02d} → {top_name}({top_prob:.3f})  "
                  f"\"{text}{'...' if len(turns[t]) > 50 else ''}\"")
        print()


# ── Main training ──────────────────────────────────────────────────

def train(cfg: M1Config = None, train_file: str = None, test_file: str = None):
    if cfg is None:
        cfg = M1Config()

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Load data ──
    if not train_file or not test_file:
        print("ERROR: --train-file and --test-file are required.")
        return

    for p in [train_file, test_file]:
        if not os.path.exists(p):
            print(f"ERROR: {p} not found.")
            return

    print(f"\nLoading data...")
    print(f"  Train: {train_file}")
    print(f"  Test:  {test_file}")
    train_dlg = load_json(train_file)
    val_dlg   = load_json(test_file)

    if cfg.truncate_aug:
        before    = len(train_dlg)
        train_dlg = truncate_augment(train_dlg, cfg.aug_k, cfg.aug_min_turns)
        from collections import Counter
        class_dist = Counter(get_class_label(d) for d in train_dlg)
        print(f"Truncate aug: {before} → {len(train_dlg)}")
        for c_idx, c_name in enumerate(CLASS_NAMES):
            print(f"  {c_name}: {class_dist.get(c_idx, 0)}")

    print(f"Train: {len(train_dlg)} | Test: {len(val_dlg)}")
    _dataset_overview(train_dlg, val_dlg)

    # ── Tokenizer ──
    print(f"\nLoading tokenizer: {cfg.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)

    train_ds = DialogueDataset(train_dlg, tokenizer, cfg.max_turn_len, cfg.max_turns)
    val_ds   = DialogueDataset(val_dlg,   tokenizer, cfg.max_turn_len, cfg.max_turns)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=2, pin_memory=True)

    # ── Model ──
    print(f"\nLoading model: {cfg.model_name}")
    print(f"  Num classes: {cfg.num_classes}")
    print(f"  Classes: {cfg.class_names}")
    model = M1Classifier(cfg).to(device)

    # Apply Phase 1 overrides
    cfg.class_weights = list(cfg.phase1_class_weights)
    cfg.weighted_lambda = cfg.phase1_lambda_aux
    current_phase = 1

    print(f"\n{'='*60}")
    print(f"PHASE 1 (epoch 1-{cfg.phase2_epoch-1}): encoder frozen, light auxiliary")
    print(f"{'='*60}")
    print(f"  class_weights = {cfg.class_weights}")
    print(f"  lambda_aux    = {cfg.weighted_lambda}")
    print(f"  lr            = {cfg.phase1_lr}")
    print(f"  Trainable params: {model.count_trainable_params():,}")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.phase1_lr, weight_decay=cfg.weight_decay,
    )

    steps_per_epoch  = len(train_loader) // cfg.grad_accum_steps
    phase1_total = steps_per_epoch * (cfg.phase2_epoch - 1)
    warmup_steps = max(1, int(phase1_total * cfg.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=phase1_total,
    )

    # ── Training loop ──
    os.makedirs(cfg.output_dir, exist_ok=True)
    best_macro_f1 = 0.0
    best_epoch    = 0
    global_step   = 0

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()

        # ── Phase 2 transition ──
        if epoch == cfg.phase2_epoch and current_phase < 2:
            current_phase = 2
            cfg.class_weights = list(cfg.phase2_class_weights)
            cfg.weighted_lambda = cfg.phase2_lambda_aux

            print(f"\n{'='*60}")
            print(f"PHASE 2 (epoch {cfg.phase2_epoch}-{cfg.phase3_epoch-1}): frozen + auxiliary")
            print(f"{'='*60}")
            print(f"  class_weights = {cfg.class_weights}")
            print(f"  lambda_aux    = {cfg.weighted_lambda}")
            print(f"  lr            = {cfg.phase2_lr}")

            optimizer = AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg.phase2_lr, weight_decay=cfg.weight_decay,
            )
            phase2_total = steps_per_epoch * (cfg.phase3_epoch - cfg.phase2_epoch)
            warmup_steps = max(1, int(phase2_total * cfg.warmup_ratio))
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=phase2_total,
            )

        # ── Phase 3 transition ──
        if epoch == cfg.phase3_epoch and current_phase < 3:
            current_phase = 3
            cfg.class_weights = list(cfg.phase3_class_weights)
            cfg.weighted_lambda = cfg.phase3_lambda_aux

            if cfg.phase3_unfreeze_layers > 0:
                model.unfreeze_last_n_layers(cfg.phase3_unfreeze_layers)

            print(f"\n{'='*60}")
            if cfg.phase3_unfreeze_layers > 0:
                print(f"PHASE 3 (epoch {cfg.phase3_epoch}+): unfreeze last {cfg.phase3_unfreeze_layers} layers")
            else:
                print(f"PHASE 3 (epoch {cfg.phase3_epoch}+): encoder frozen, loss schedule only")
            print(f"{'='*60}")
            print(f"  class_weights = {cfg.class_weights}")
            print(f"  lambda_aux    = {cfg.weighted_lambda}")
            print(f"  head_lr       = {cfg.phase3_head_lr}")
            if cfg.phase3_unfreeze_layers > 0:
                print(f"  encoder_lr    = {cfg.phase3_encoder_lr}")
            print(f"  Trainable params: {model.count_trainable_params():,}")

            if cfg.phase3_unfreeze_layers > 0:
                encoder_params = [p for n, p in model.named_parameters()
                                  if p.requires_grad and 'encoder' in n]
                head_params    = [p for n, p in model.named_parameters()
                                  if p.requires_grad and 'encoder' not in n]
                optimizer = AdamW([
                    {"params": head_params,    "lr": cfg.phase3_head_lr},
                    {"params": encoder_params, "lr": cfg.phase3_encoder_lr},
                ], weight_decay=cfg.weight_decay)
            else:
                optimizer = AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=cfg.phase3_head_lr, weight_decay=cfg.weight_decay,
                )

            remaining = cfg.num_epochs - cfg.phase3_epoch + 1
            phase3_total = steps_per_epoch * remaining
            warmup_steps = max(1, int(phase3_total * cfg.warmup_ratio))
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=phase3_total,
            )

        print(f"\nEpoch {epoch}/{cfg.num_epochs} [Phase {current_phase}]")
        tr_loss, global_step = run_epoch(
            model, train_loader, optimizer, scheduler, device, cfg, global_step
        )

        val_metrics = evaluate(model, val_loader, device, cfg)
        _print_val_metrics(epoch, time.time() - t0, tr_loss, val_metrics)

        # Streaming preview
        _preview_stream(model, val_ds, val_dlg, device, cfg)

        # Best model selection by macro-F1
        val_f1 = val_metrics["macro_f1"]
        if val_f1 > best_macro_f1:
            best_macro_f1 = val_f1
            best_epoch    = epoch
            _save_model(model, tokenizer, cfg, val_metrics)
            print(f"    * Best model saved (macro_f1={best_macro_f1:.4f})")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nBest epoch: {best_epoch} | macro_f1={best_macro_f1:.4f}")

    # ── Final report trên val (test) set với best model ──
    best_pt = os.path.join(cfg.output_dir, "best_model", "model.pt")
    if os.path.exists(best_pt):
        model.load_state_dict(torch.load(best_pt, map_location=device, weights_only=True))
        final_metrics = evaluate(model, val_loader, device, cfg)
        print_streaming_report(final_metrics)

    return model


# ── CLI ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train M1 Multi-Class: HaLong + CrossTurnAttention")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--train-file", required=True,
                        help="Path to train JSON file")
    parser.add_argument("--test-file",  required=True,
                        help="Path to test JSON file")
    parser.add_argument("--debug",      action="store_true",
                        help="2 epochs, no aug")
    parser.add_argument("--small",      action="store_true",
                        help="5 epochs")
    return parser.parse_args()


if __name__ == "__main__":
    cfg  = M1Config()
    args = parse_args()

    if args.output_dir:
        cfg.output_dir = args.output_dir

    if args.debug:
        cfg.num_epochs       = 2
        cfg.batch_size       = 4
        cfg.grad_accum_steps = 4
        cfg.warmup_ratio     = 0.03
        cfg.truncate_aug     = False
        cfg.phase2_epoch     = 99   # skip phase 2/3
        cfg.phase3_epoch     = 99
        print("DEBUG MODE: 2 epochs, batch_size=4, no aug, single phase")
    if args.small:
        cfg.num_epochs = 5
        cfg.batch_size = 4
        print("SMALL MODE: 5 epochs, batch_size=4")

    train(cfg, train_file=args.train_file, test_file=args.test_file)
