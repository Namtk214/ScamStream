"""
Test M1 Classifier trên dữ liệu test.

Usage:
    python test.py
    python test.py --data ../data/test.json --threshold 0.4
    python test.py --out-dir outputs/eval
"""

import argparse
import dataclasses
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from config import M1Config, DATA_DIR
from dataset import DialogueDataset, collate_fn, load_json
from model import M1Classifier
from metrics import compute_streaming_metrics, print_streaming_report, _first_alert_turn


@torch.no_grad()
def run_inference(model, dataset, raw, device, threshold):
    model.eval()
    loader = DataLoader(dataset, batch_size=2, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    all_labels, all_d_probs, all_t_probs = [], [], []
    total_loss, n_dlg = 0.0, 0

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

    metrics = compute_streaming_metrics(all_labels, all_d_probs, all_t_probs, threshold)
    metrics["loss"] = total_loss / max(n_dlg, 1)
    return metrics, all_labels, all_d_probs, all_t_probs


def save_metrics_json(metrics, out_path):
    summary = {
        "accuracy":          round(metrics["dialogue_accuracy"], 4),
        "f1":                round(metrics["dialogue_f1"], 4),
        "auroc":             round(metrics.get("auroc", float("nan")), 4),
        "avg_delay_turns":   round(metrics.get("avg_detection_delay", float("nan")), 4),
        "detection_rate":    round(metrics["detection_rate"], 4),
        "false_alarm_rate":  round(metrics["false_alarm_rate"], 4),
        "loss":              round(metrics.get("loss", float("nan")), 4),
        "num_scam":          int(metrics["num_scam"]),
        "num_harmless":      int(metrics["num_harmless"]),
        "num_detected":      int(metrics["num_detected"]),
        "num_false_alarms":  int(metrics["num_false_alarms"]),
    }
    if "mean_alert_turn" in metrics:
        summary["mean_alert_turn"]   = round(metrics["mean_alert_turn"], 4)
        summary["median_alert_turn"] = round(metrics["median_alert_turn"], 4)
        summary["alert_at_half"]     = round(metrics["alert_at_half"], 4)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Metrics saved → {out_path}")


def save_errors_excel(raw, all_labels, all_d_probs, all_t_probs, threshold, out_path):
    try:
        import openpyxl
        from openpyxl.styles import PatternFill, Font, Alignment
        from openpyxl.utils import get_column_letter
    except ImportError:
        print("  [WARN] openpyxl not installed — skipping Excel output")
        return

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Wrong Predictions"

    # Header
    headers = ["#", "Error Type", "True Label", "Pred Label",
               "Confidence", "Turn #", "Turn Prob", "Turn Text"]
    ws.append(headers)
    hdr_fill = PatternFill("solid", fgColor="1F4E79")
    hdr_font = Font(bold=True, color="FFFFFF")
    for col, _ in enumerate(headers, 1):
        cell = ws.cell(1, col)
        cell.fill = hdr_fill
        cell.font = hdr_font
        cell.alignment = Alignment(horizontal="center", wrap_text=True)

    fill_fp = PatternFill("solid", fgColor="FFE0E0")   # light red — false positive
    fill_fn = PatternFill("solid", fgColor="FFF2CC")   # light yellow — false negative
    fill_fp_turn = PatternFill("solid", fgColor="FFF5F5")
    fill_fn_turn = PatternFill("solid", fgColor="FFFBE6")

    row_idx = 2
    err_count = 0
    for idx, (label, d_prob, t_probs) in enumerate(zip(all_labels, all_d_probs, all_t_probs)):
        pred = 1 if d_prob >= threshold else 0
        if pred == label:
            continue

        err_count += 1
        dlg = raw[idx]
        turns = dlg["turns"][:len(t_probs)]
        true_lbl = "scam" if label == 1 else "harmless"
        pred_lbl = "scam" if pred == 1 else "harmless"
        err_type = "FP (harmless→scam)" if label == 0 else "FN (scam→harmless)"
        row_fill      = fill_fp if label == 0 else fill_fn
        row_fill_turn = fill_fp_turn if label == 0 else fill_fn_turn

        first_alert = _first_alert_turn(t_probs, threshold)

        for t, (turn_text, t_prob) in enumerate(zip(turns, t_probs)):
            is_first = (t == 0)
            ws.append([
                err_count if is_first else "",
                err_type  if is_first else "",
                true_lbl  if is_first else "",
                pred_lbl  if is_first else "",
                round(d_prob, 4) if is_first else "",
                t + 1,
                round(float(t_prob), 4),
                str(turn_text),
            ])
            cur_fill = row_fill if is_first else row_fill_turn
            for col in range(1, len(headers) + 1):
                cell = ws.cell(row_idx, col)
                cell.fill = cur_fill
                cell.alignment = Alignment(wrap_text=True, vertical="top")
                if col == 7:  # turn prob
                    cell.number_format = "0.0000"
                if col == 5 and is_first:  # confidence
                    cell.number_format = "0.0000"
            # Mark alerted turns
            if first_alert is not None and t == first_alert:
                ws.cell(row_idx, 7).font = Font(bold=True, color="C00000")
            row_idx += 1

        # Blank separator row
        ws.append([""] * len(headers))
        row_idx += 1

    # Column widths
    col_widths = [5, 20, 12, 12, 12, 8, 10, 80]
    for col, w in enumerate(col_widths, 1):
        ws.column_dimensions[get_column_letter(col)].width = w

    ws.freeze_panes = "A2"

    # Summary sheet
    ws2 = wb.create_sheet("Summary")
    ws2.append(["Metric", "Value"])
    ws2.append(["Total wrong", err_count])
    fp_n = sum(1 for l, p in zip(all_labels, all_d_probs)
                if l == 0 and (p >= threshold))
    fn_n = sum(1 for l, p in zip(all_labels, all_d_probs)
                if l == 1 and (p < threshold))
    ws2.append(["False Positives (harmless→scam)", fp_n])
    ws2.append(["False Negatives (scam→harmless)", fn_n])

    wb.save(out_path)
    print(f"  Errors saved  → {out_path}  ({err_count} wrong predictions)")


def parse_args():
    cfg = M1Config()
    parser = argparse.ArgumentParser(description="Test M1 Scam Classifier")
    parser.add_argument("--data",      default=os.path.join(DATA_DIR, "test.json"))
    parser.add_argument("--model",     default=os.path.join(cfg.output_dir, "best_model"))
    parser.add_argument("--threshold", type=float, default=cfg.threshold)
    parser.add_argument("--out-dir",   default=None,
                        help="Output directory (default: same as --model)")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = args.out_dir or args.model

    print("=" * 65)
    print("M1 SCAM CLASSIFIER — TEST")
    print("=" * 65)
    print(f"  Data:      {args.data}")
    print(f"  Model:     {args.model}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Out dir:   {out_dir}")

    if not os.path.exists(args.data):
        print(f"\n  [ERROR] Not found: {args.data}")
        sys.exit(1)

    model_pt = os.path.join(args.model, "model.pt")
    if not os.path.exists(model_pt):
        print(f"\n  [ERROR] Model not found: {model_pt}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device:    {device}")

    config_json = os.path.join(args.model, "config.json")
    if os.path.exists(config_json):
        with open(config_json) as f:
            cfg_dict = json.load(f)
        cfg = M1Config(**{k: v for k, v in cfg_dict.items()
                          if k in {f.name for f in dataclasses.fields(M1Config)}})
    else:
        cfg = M1Config()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model     = M1Classifier(cfg).to(device)
    model.load_state_dict(torch.load(model_pt, map_location=device, weights_only=True))
    model.eval()

    raw = load_json(args.data)
    scam_n = sum(1 for d in raw if d["label"] == "scam")
    harm_n = sum(1 for d in raw if d["label"] == "harmless")
    print(f"\n  {len(raw)} conversations (scam={scam_n}, harmless={harm_n})")

    dataset = DialogueDataset(raw, tokenizer, cfg.max_turn_len, cfg.max_turns)

    print(f"\nEvaluating...")
    metrics, all_labels, all_d_probs, all_t_probs = run_inference(
        model, dataset, raw, device, args.threshold
    )
    print_streaming_report(metrics)

    os.makedirs(out_dir, exist_ok=True)
    save_metrics_json(metrics, os.path.join(out_dir, "test_metrics.json"))
    save_errors_excel(raw, all_labels, all_d_probs, all_t_probs, args.threshold,
                      os.path.join(out_dir, "test_errors.xlsx"))

    print("\nDone!")


if __name__ == "__main__":
    main()
