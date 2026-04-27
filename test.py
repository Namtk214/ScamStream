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


def _compute_p_agg(turn_probs):
    """Compute Noisy-OR cumulative probability from per-turn probs."""
    import numpy as np
    p_agg = 0.0
    p_agg_list = []
    for q in turn_probs:
        p_agg = 1.0 - (1.0 - p_agg) * (1.0 - float(q))
        p_agg_list.append(p_agg)
    return np.array(p_agg_list)


@torch.no_grad()
def run_inference(model, dataset, raw, device, threshold):
    model.eval()
    loader = DataLoader(dataset, batch_size=2, shuffle=False,
                        collate_fn=collate_fn, num_workers=0)

    all_labels, all_d_probs, all_t_probs, all_p_agg = [], [], [], []
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
            t_probs = output["turn_probs"][b, :n].cpu().numpy()
            all_t_probs.append(t_probs)
            all_p_agg.append(_compute_p_agg(t_probs))

    metrics = compute_streaming_metrics(all_labels, all_d_probs, all_t_probs, threshold)
    metrics["loss"] = total_loss / max(n_dlg, 1)
    return metrics, all_labels, all_d_probs, all_t_probs, all_p_agg


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
    if "p_agg_detection_rate" in metrics:
        summary["p_agg_detection_rate"]  = round(metrics["p_agg_detection_rate"], 4)
        summary["p_agg_false_alarm_rate"] = round(metrics["p_agg_false_alarm_rate"], 4)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"  Metrics saved → {out_path}")


def save_errors_excel(raw, all_labels, all_d_probs, all_t_probs, all_p_agg,
                      threshold, out_path):
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
               "Confidence", "Turn #", "q_t (evidence)", "p_agg (Noisy-OR)", "Turn Text"]
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
    for idx, (label, d_prob, t_probs, p_agg) in enumerate(
            zip(all_labels, all_d_probs, all_t_probs, all_p_agg)):
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
            pa = float(p_agg[t]) if t < len(p_agg) else 0.0
            ws.append([
                err_count if is_first else "",
                err_type  if is_first else "",
                true_lbl  if is_first else "",
                pred_lbl  if is_first else "",
                round(d_prob, 4) if is_first else "",
                t + 1,
                round(float(t_prob), 4),
                round(pa, 4),
                str(turn_text),
            ])
            cur_fill = row_fill if is_first else row_fill_turn
            for col in range(1, len(headers) + 1):
                cell = ws.cell(row_idx, col)
                cell.fill = cur_fill
                cell.alignment = Alignment(wrap_text=True, vertical="top")
                if col in (7, 8):  # q_t and p_agg
                    cell.number_format = "0.0000"
                if col == 5 and is_first:  # confidence
                    cell.number_format = "0.0000"
            # Mark alerted turns (by p_agg)
            if pa >= threshold:
                ws.cell(row_idx, 8).font = Font(bold=True, color="C00000")
            if first_alert is not None and t == first_alert:
                ws.cell(row_idx, 7).font = Font(bold=True, color="C00000")
            row_idx += 1

        # Blank separator row
        ws.append([""] * len(headers))
        row_idx += 1

    # Column widths
    col_widths = [5, 20, 12, 12, 12, 8, 14, 14, 80]
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
    metrics, all_labels, all_d_probs, all_t_probs, all_p_agg = run_inference(
        model, dataset, raw, device, args.threshold
    )

    # Add Noisy-OR p_agg based streaming metrics
    p_agg_detected, p_agg_fa = 0, 0
    num_scam = sum(1 for l in all_labels if l == 1)
    num_harm = sum(1 for l in all_labels if l == 0)
    p_agg_alert_turns = []
    for label, pa_list in zip(all_labels, all_p_agg):
        first_alert = None
        for t, pa in enumerate(pa_list):
            if pa >= args.threshold:
                first_alert = t
                break
        if label == 1 and first_alert is not None:
            p_agg_detected += 1
            p_agg_alert_turns.append(first_alert)
        elif label == 0 and first_alert is not None:
            p_agg_fa += 1
    metrics["p_agg_detection_rate"]   = p_agg_detected / max(num_scam, 1)
    metrics["p_agg_false_alarm_rate"] = p_agg_fa / max(num_harm, 1)
    if p_agg_alert_turns:
        metrics["p_agg_avg_alert_turn"] = float(np.mean(p_agg_alert_turns))

    print_streaming_report(metrics)

    # Print Noisy-OR p_agg specific metrics
    print(f"\n  Noisy-OR p_agg Streaming Metrics:")
    print(f"    Detection rate (p_agg):   {metrics['p_agg_detection_rate']:.4f} ({p_agg_detected}/{num_scam})")
    print(f"    False alarm (p_agg):      {metrics['p_agg_false_alarm_rate']:.4f} ({p_agg_fa}/{num_harm})")
    if "p_agg_avg_alert_turn" in metrics:
        print(f"    Avg alert turn (p_agg):   {metrics['p_agg_avg_alert_turn']:.2f}")

    os.makedirs(out_dir, exist_ok=True)
    save_metrics_json(metrics, os.path.join(out_dir, "test_metrics.json"))
    save_errors_excel(raw, all_labels, all_d_probs, all_t_probs, all_p_agg,
                      args.threshold, os.path.join(out_dir, "test_errors.xlsx"))

    # Error analysis — 20 samples per type
    print_error_samples(raw, all_labels, all_d_probs, all_t_probs, all_p_agg,
                        args.threshold, max_per_type=20)

    print("\nDone!")


def print_error_samples(raw, all_labels, all_d_probs, all_t_probs, all_p_agg,
                        threshold, max_per_type=20):
    """Print top False Positives and False Negatives with turn-by-turn probs + p_agg."""
    FP, FN = [], []
    for idx, (label, d_prob, t_probs, p_agg) in enumerate(
            zip(all_labels, all_d_probs, all_t_probs, all_p_agg)):
        pred = 1 if d_prob >= threshold else 0
        if label == 0 and pred == 1:
            FP.append((idx, d_prob, t_probs, p_agg, raw[idx]["turns"][:len(t_probs)]))
        elif label == 1 and pred == 0:
            FN.append((idx, d_prob, t_probs, p_agg, raw[idx]["turns"][:len(t_probs)]))

    print(f"\n  Total errors: FP={len(FP)}, FN={len(FN)}")

    print(f"\n{'='*65}")
    print(f"  FALSE POSITIVES (harmless → predicted scam) — top {min(len(FP), max_per_type)}")
    print(f"{'='*65}")
    for idx, dp, tps, pag, turns in sorted(FP, key=lambda x: -x[1])[:max_per_type]:
        print(f"\n  [FP #{idx}] p_final={dp:.4f} p_agg_final={pag[-1]:.4f}  ({len(turns)} turns)")
        for t, (q, pa, turn) in enumerate(zip(tps, pag, turns), 1):
            alert = " [!]" if pa >= threshold else ""
            print(f"    T{t}: q={q:.4f} p_agg={pa:.4f}{alert}  {str(turn)[:80]}")

    print(f"\n{'='*65}")
    print(f"  FALSE NEGATIVES (scam → predicted harmless) — top {min(len(FN), max_per_type)}")
    print(f"{'='*65}")
    for idx, dp, tps, pag, turns in sorted(FN, key=lambda x: x[1])[:max_per_type]:
        print(f"\n  [FN #{idx}] p_final={dp:.4f} p_agg_final={pag[-1]:.4f}  ({len(turns)} turns)")
        for t, (q, pa, turn) in enumerate(zip(tps, pag, turns), 1):
            alert = " [!]" if pa >= threshold else ""
            print(f"    T{t}: q={q:.4f} p_agg={pa:.4f}{alert}  {str(turn)[:80]}")


if __name__ == "__main__":
    main()
