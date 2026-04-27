"""
Training loop cho M1 (HaLong + CrossTurnAttention + Weighted CE).

- Val = test.json (dùng chung vì data hạn chế)
- Wandb logging: train loss/lr per step, full val metrics per epoch
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

# WANDB_API_KEY = os.environ.get(
#     "WANDB_API_KEY",
#     "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",
# )
WANDB_API_KEY = None


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
    all_labels, all_d_probs, all_t_probs = [], [], []

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
            all_t_probs.append(output["turn_probs"][b, :n].cpu().numpy())

        del output
        if device.type == "cuda":
            torch.cuda.empty_cache()

    metrics = compute_streaming_metrics(all_labels, all_d_probs, all_t_probs, threshold)
    metrics["loss"] = total_loss / max(n_dlg, 1)
    return metrics


# ── Training epoch ─────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, scheduler, device, cfg, global_step: int, wandb_run):
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
            scheduler.step()          # cosine+warmup: step per optimizer update
            optimizer.zero_grad()
            global_step += 1

            # if wandb_run is not None:
            #     wandb_run.log({
            #         "train/loss": total_loss / max(n_dlg, 1),
            #         "train/lr":   scheduler.get_last_lr()[0],
            #     }, step=global_step)

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
    """Print dataset overview before training."""
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
    """Streaming inference preview on random validation samples after each epoch."""
    model.eval()
    if len(val_dlg) == 0:
        return

    # Try to pick 1 scam + 1 harmless
    scam_indices    = [i for i, d in enumerate(val_dlg) if d["label"] == "scam"]
    harmless_indices = [i for i, d in enumerate(val_dlg) if d["label"] == "harmless"]
    selected = []
    if scam_indices:
        selected.append(random.choice(scam_indices))
    if harmless_indices:
        selected.append(random.choice(harmless_indices))
    # Fill remaining slots
    while len(selected) < max_samples and len(val_dlg) > len(selected):
        idx = random.randint(0, len(val_dlg) - 1)
        if idx not in selected:
            selected.append(idx)

    print(f"\n  -- Streaming Inference Preview --")
    for idx in selected:
        dlg   = val_dlg[idx]
        label = dlg["label"]
        turns = dlg["turns"]
        n     = len(turns)

        # Build full input for all turns at once
        item  = val_ds[idx]
        batch = {
            "input_ids":  item["input_ids"].unsqueeze(0).to(device),
            "attn_masks": item["attn_masks"].unsqueeze(0).to(device),
            "turn_mask":  item["turn_mask"].unsqueeze(0).to(device),
        }
        output = model(batch["input_ids"], batch["attn_masks"], batch["turn_mask"])
        probs  = output["turn_probs"][0, :n].cpu().tolist()
        d_prob = output["dialogue_probs"][0].item()
        pred   = "scam" if d_prob >= cfg.threshold else "harmless"

        status = "✓" if pred == label else "✗"
        print(f"    [{status}] true={label:8s} pred={pred:8s} p_final={d_prob:.3f} ({n} turns)")
        for t in range(n):
            p = probs[t]
            alert = " ← ALERT" if p >= cfg.threshold else ""
            text  = turns[t][:50]
            print(f"      T{t+1:02d} p={p:.3f}{alert}  \"{text}{'...' if len(turns[t]) > 50 else ''}\"")
        print()


def _wandb_val_log(wandb_run, m, epoch, tr_loss):
    pass
    # if wandb_run is None:
    #     return
    # log = {
    #     "epoch":                  epoch,
    #     "train/loss_epoch":       tr_loss,
    #     "val/loss":               m["loss"],
    #     "val/f1":                 m["dialogue_f1"],
    #     "val/accuracy":           m["dialogue_accuracy"],
    #     "val/detection_rate":     m["detection_rate"],
    #     "val/false_alarm_rate":   m["false_alarm_rate"],
    # }
    # if not np.isnan(m.get("auroc", float("nan"))):
    #     log["val/auroc"] = m["auroc"]
    # if not np.isnan(m["avg_detection_delay"]):
    #     log["val/avg_turn"] = m["avg_detection_delay"]    # avg turn phát hiện scam
    # if "alert_at_half" in m:
    #     log["val/alert_at_half"] = m["alert_at_half"]
    # if "median_lead_frac" in m:
    #     log["val/median_lead_frac"] = m["median_lead_frac"]
    # wandb_run.log(log)


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
        # Direct file paths provided by user
        train_path = train_file
        val_path   = test_file
        print(f"\nUsing custom dataset paths:")
        print(f"  Train: {train_path}")
        print(f"  Test:  {val_path}")
    elif dataset_option and dataset_option in DATASET_OPTIONS:
        # Use dataset/ directory with the selected option
        ds_files  = DATASET_OPTIONS[dataset_option]
        train_path = os.path.join(cfg.dataset_dir, ds_files["train"])
        val_path   = os.path.join(cfg.dataset_dir, ds_files["test"])
        print(f"\nDataset option: {dataset_option}")
        print(f"  Train: {ds_files['train']}")
        print(f"  Test:  {ds_files['test']}")
    else:
        # Default: use data_dir/train.json + test.json (legacy behavior)
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

    # ── Dataset overview ──
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
    model = M1Classifier(cfg).to(device)

    print(f"Loss: Focal(γ={cfg.focal_gamma}) × U-shape(floor={cfg.w_floor}) × class_w(harm={cfg.class_weight_harmless})")
    print(f"Trainable params (encoder frozen): {model.count_trainable_params():,}")

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay,
    )

    # Số optimizer steps trước khi unfreeze
    steps_per_epoch  = len(train_loader) // cfg.grad_accum_steps
    frozen_epochs    = cfg.unfreeze_epoch - 1          # epochs chạy với encoder frozen
    total_steps_frozen = steps_per_epoch * frozen_epochs or steps_per_epoch * cfg.num_epochs
    warmup_steps_frozen = max(1, int(total_steps_frozen * cfg.warmup_ratio))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps_frozen,
        num_training_steps=total_steps_frozen,
    )

    # ── Wandb ──
    # try:
    #     import wandb
    #     wandb.login(key=WANDB_API_KEY)
    #     wandb_run = wandb.init(
    #         project="viscam-m1",
    #         name=f"m1-halong-ep{cfg.num_epochs}-bs{cfg.batch_size}",
    #         config=dataclasses.asdict(cfg),
    #     )
    #     print(f"Wandb run: {wandb_run.url}")
    # except Exception as e:
    #     print(f"[WARN] Wandb init failed: {e} — continuing without wandb")
    #     wandb_run = None
    wandb_run = None

    # ── Training loop ──
    os.makedirs(cfg.output_dir, exist_ok=True)
    best_val_loss = float("inf")
    best_epoch    = 0
    no_improve    = 0
    patience      = 3
    global_step   = 0

    for epoch in range(1, cfg.num_epochs + 1):
        t0 = time.time()

        if epoch == cfg.unfreeze_epoch:
            model.unfreeze_encoder()
            optimizer = AdamW(
                model.parameters(),
                lr=cfg.lr * 0.1, weight_decay=cfg.weight_decay,
            )
            remaining_epochs = cfg.num_epochs - epoch + 1
            total_steps_unfreeze  = steps_per_epoch * remaining_epochs
            warmup_steps_unfreeze = max(1, int(total_steps_unfreeze * cfg.warmup_ratio))
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps_unfreeze,
                num_training_steps=total_steps_unfreeze,
            )
            print(f"\nEpoch {epoch}: encoder unfrozen, lr={cfg.lr * 0.1:.2e}")
            print(f"Trainable params: {model.count_trainable_params():,}")

        print(f"\nEpoch {epoch}/{cfg.num_epochs}")
        tr_loss, global_step = run_epoch(
            model, train_loader, optimizer, scheduler, device, cfg, global_step, wandb_run
        )

        val_metrics = evaluate(model, val_loader, device, cfg.threshold)
        _print_val_metrics(epoch, time.time() - t0, tr_loss, val_metrics)
        _wandb_val_log(wandb_run, val_metrics, epoch, tr_loss)

        # Streaming preview
        _preview_stream(model, val_ds, val_dlg, device, cfg)

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_epoch    = epoch
            no_improve    = 0
            _save_model(model, tokenizer, cfg, val_metrics)
            print(f"    * Best model saved (val_loss={best_val_loss:.4f})")
            # if wandb_run is not None:
            #     wandb_run.summary["best_val_loss"] = best_val_loss
            #     wandb_run.summary["best_epoch"]    = best_epoch
            #     wandb_run.summary["best_val_f1"]   = val_metrics["dialogue_f1"]
        else:
            no_improve += 1
            print(f"    No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"\nBest epoch: {best_epoch} | val_loss={best_val_loss:.4f}")

    # ── Final report trên val (test) set với best model ──
    best_pt = os.path.join(cfg.output_dir, "best_model", "model.pt")
    if os.path.exists(best_pt):
        model.load_state_dict(torch.load(best_pt, map_location=device, weights_only=True))
        final_metrics = evaluate(model, val_loader, device, cfg.threshold)
        print_streaming_report(final_metrics)
        # if wandb_run is not None:
        #     wandb_run.summary.update({
        #         "final/loss":              final_metrics["loss"],
        #         "final/f1":               final_metrics["dialogue_f1"],
        #         "final/accuracy":         final_metrics["dialogue_accuracy"],
        #         "final/avg_turn":         final_metrics.get("avg_detection_delay", float("nan")),
        #         "final/auroc":            final_metrics.get("auroc", float("nan")),
        #         "final/detection_rate":   final_metrics["detection_rate"],
        #         "final/false_alarm_rate": final_metrics["false_alarm_rate"],
        #     })

    # if wandb_run is not None:
    #     wandb_run.finish()

    return model


# ── CLI ────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train M1: HaLong + CrossTurnAttention")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--run-name",   default=None, help="Wandb run name")
    parser.add_argument("--train-file", default=None,
                        help="Path to custom train JSON file")
    parser.add_argument("--test-file",  default=None,
                        help="Path to custom test JSON file")
    parser.add_argument("--dataset",    default=None,
                        choices=list(DATASET_OPTIONS.keys()),
                        help="Dataset option: "
                             "real (Real-1→Real-2), "
                             "synthetic (Synthetic→Real-2), "
                             "tele (Tele→Real-2), "
                             "real_syn (Real-1+Synthetic→Real-2), "
                             "real_tele (Real-1+Tele→Real-2)")
    parser.add_argument("--debug",      action="store_true",
                        help="2 epochs, batch_size=1, accum=4, no aug")
    parser.add_argument("--small",      action="store_true",
                        help="5 epochs, batch_size=2")
    return parser.parse_args()


if __name__ == "__main__":
    cfg  = M1Config()
    args = parse_args()

    if args.output_dir:
        cfg.output_dir = args.output_dir
    elif args.dataset:
        # Auto-set output dir based on dataset option
        cfg.output_dir = os.path.join(_dir, "outputs", args.dataset)

    if args.debug:
        cfg.num_epochs       = 2
        cfg.batch_size       = 4
        cfg.grad_accum_steps = 4    # effective batch = 16
        cfg.warmup_ratio     = 0.03  # warmup ngắn hơn cho debug
        cfg.truncate_aug     = False
        print("DEBUG MODE: 2 epochs, batch_size=4, no aug")
    if args.small:
        cfg.num_epochs = 5
        cfg.batch_size = 2
        print("SMALL MODE: 5 epochs, batch_size=2")

    train(cfg, dataset_option=args.dataset,
          train_file=args.train_file, test_file=args.test_file)
