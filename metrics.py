"""
Multi-class evaluation metrics cho M1 (HaLong + CrossTurnAttention).

5 classes: harmless (0), A (1), B (2), C (3), D (4)

Metrics:
  - Dialogue-level: accuracy, macro-F1, weighted-F1, per-class P/R/F1
  - Streaming: per-class detection rate, avg delay, false alarm rate
  - Early detection: median/mean alert turn (on TPs)
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix,
)
from typing import Dict, List, Optional

from config import CLASS_NAMES


def compute_streaming_metrics(
    all_dialogue_labels: List[int],
    all_dialogue_preds: List[int],
    all_dialogue_probs: List[np.ndarray],   # [C] normalized p_agg at last turn
    all_p_agg: List[np.ndarray],            # [T, C] Noisy-OR per turn per class
    class_names: List[str] = None,
    scam_alert_thresh: float = 0.80,
) -> Dict:
    """
    Compute multi-class dialogue-level and streaming metrics.

    all_dialogue_probs: [N][C] normalized p_agg at last turn
    all_dialogue_preds: [N] predicted class
    all_p_agg:          [N][T, C] Noisy-OR cumulative per turn per class
    """
    if class_names is None:
        class_names = CLASS_NAMES

    d_labels = np.array(all_dialogue_labels)
    d_preds  = np.array(all_dialogue_preds)
    n_classes = len(class_names)

    # Per-class precision, recall, F1
    prec, rec, f1_per, support = precision_recall_fscore_support(
        d_labels, d_preds, labels=list(range(n_classes)), zero_division=0
    )

    # Confusion matrix
    cm = confusion_matrix(d_labels, d_preds, labels=list(range(n_classes)))

    # ── Streaming detection metrics ──
    # For scam classes (1-4): detection = predict correct class at any turn via p_agg
    # "Alert" = p_agg for the correct class crosses a threshold
    scam_detection = {}
    all_alert_turns = []
    all_n_turns = []
    for label, p_agg_seq in zip(all_dialogue_labels, all_p_agg):
        n = p_agg_seq.shape[0]
        all_n_turns.append(n)
        if label > 0:  # scam class
            alert_turn = _first_alert_turn_multiclass(p_agg_seq, label, scam_alert_thresh)
            all_alert_turns.append(alert_turn)
        else:
            # For harmless: check if any scam class triggers
            any_scam_alert = _first_any_scam_alert(p_agg_seq, scam_alert_thresh)
            all_alert_turns.append(any_scam_alert)

    # Per-class detection stats
    for c in range(1, n_classes):
        c_indices = [i for i, l in enumerate(all_dialogue_labels) if l == c]
        n_total = len(c_indices)
        n_detected = sum(1 for i in c_indices if all_alert_turns[i] is not None)
        delays = [all_alert_turns[i] for i in c_indices if all_alert_turns[i] is not None]
        scam_detection[class_names[c]] = {
            "total": n_total,
            "detected": n_detected,
            "detection_rate": n_detected / max(n_total, 1),
            "avg_delay": float(np.mean(delays)) if delays else float("nan"),
        }

    # Harmless false alarm rate (any scam class triggered)
    harmless_indices = [i for i, l in enumerate(all_dialogue_labels) if l == 0]
    n_harmless = len(harmless_indices)
    n_false_alarms = sum(1 for i in harmless_indices if all_alert_turns[i] is not None)

    # Overall scam detection (any scam class)
    all_scam_indices = [i for i, l in enumerate(all_dialogue_labels) if l > 0]
    n_scam = len(all_scam_indices)
    n_scam_detected = sum(1 for i in all_scam_indices if all_alert_turns[i] is not None)

    metrics = {
        # Dialogue-level
        "accuracy":     float(accuracy_score(d_labels, d_preds)),
        "macro_f1":     float(f1_score(d_labels, d_preds, average="macro", zero_division=0)),
        "weighted_f1":  float(f1_score(d_labels, d_preds, average="weighted", zero_division=0)),
        "confusion_matrix": cm.tolist(),
        # Per-class
        "per_class_precision": {class_names[c]: float(prec[c]) for c in range(n_classes)},
        "per_class_recall":    {class_names[c]: float(rec[c]) for c in range(n_classes)},
        "per_class_f1":        {class_names[c]: float(f1_per[c]) for c in range(n_classes)},
        "per_class_support":   {class_names[c]: int(support[c]) for c in range(n_classes)},
        # Streaming overall
        "overall_detection_rate": n_scam_detected / max(n_scam, 1),
        "false_alarm_rate":       n_false_alarms / max(n_harmless, 1),
        "num_scam":               n_scam,
        "num_harmless":           n_harmless,
        "num_scam_detected":      n_scam_detected,
        "num_false_alarms":       n_false_alarms,
        # Per-class detection
        "scam_detection": scam_detection,
        # Early detection stats
        **early_detection_stats(all_alert_turns, all_n_turns, all_dialogue_labels),
    }
    return metrics


def early_detection_stats(all_alert_turns: List[Optional[int]],
                           all_n_turns: List[int],
                           all_labels: List[int]) -> Dict[str, float]:
    """
    Chỉ tính trên True Positive (scam bị detect đúng).
    alert_turn: 0-based index của turn đầu tiên vượt threshold.
    """
    tp_turns, tp_fracs = [], []
    for alert_t, n, label in zip(all_alert_turns, all_n_turns, all_labels):
        if label > 0 and alert_t is not None:  # scam detected
            tp_turns.append(alert_t)
            tp_fracs.append(alert_t / max(n, 1))
    if not tp_turns:
        return {}
    return {
        "median_alert_turn": float(np.median(tp_turns)),
        "mean_alert_turn":   float(np.mean(tp_turns)),
        "median_lead_frac":  float(np.median(tp_fracs)),
        "alert_at_half":     float(np.mean([f <= 0.5 for f in tp_fracs])),
    }


def print_streaming_report(metrics: Dict, class_names: List[str] = None):
    if class_names is None:
        class_names = CLASS_NAMES

    print("\n" + "=" * 65)
    print("M1 MULTI-CLASS STREAMING EVALUATION REPORT")
    print("=" * 65)

    print("\n  Dialogue-Level Metrics:")
    print(f"    Accuracy:    {metrics['accuracy']:.4f}")
    print(f"    Macro F1:    {metrics['macro_f1']:.4f}")
    print(f"    Weighted F1: {metrics['weighted_f1']:.4f}")

    print("\n  Per-Class F1 (precision / recall / F1 / support):")
    for c in class_names:
        p = metrics["per_class_precision"][c]
        r = metrics["per_class_recall"][c]
        f = metrics["per_class_f1"][c]
        s = metrics["per_class_support"][c]
        print(f"    {c:10s}: P={p:.3f}  R={r:.3f}  F1={f:.3f}  n={s}")

    print("\n  Confusion Matrix:")
    cm = metrics["confusion_matrix"]
    header = "          " + "".join(f"{c:>8s}" for c in class_names)
    print(f"    {header}")
    for i, row in enumerate(cm):
        row_str = "".join(f"{v:8d}" for v in row)
        print(f"    {class_names[i]:>8s}  {row_str}")

    print("\n  Streaming Detection (overall):")
    print(
        f"    Detection rate:   {metrics['overall_detection_rate']:.4f} "
        f"({metrics['num_scam_detected']}/{metrics['num_scam']})"
    )
    print(
        f"    False alarm rate: {metrics['false_alarm_rate']:.4f} "
        f"({metrics['num_false_alarms']}/{metrics['num_harmless']})"
    )

    print("\n  Per-Class Detection (streaming):")
    for c_name, stats in metrics.get("scam_detection", {}).items():
        dr = stats["detection_rate"]
        ad = stats["avg_delay"]
        ad_str = f"{ad:.1f}" if not np.isnan(ad) else "N/A"
        print(f"    {c_name:10s}: {stats['detected']}/{stats['total']} "
              f"({dr:.3f})  avg_delay={ad_str}")

    if "median_alert_turn" in metrics:
        print("\n  Early Detection Stats (all scam TPs):")
        print(f"    Median alert turn: {metrics['median_alert_turn']:.1f}")
        print(f"    Mean alert turn:   {metrics['mean_alert_turn']:.1f}")
        print(f"    Median lead frac:  {metrics['median_lead_frac']:.3f}")
        print(f"    Alert in 1st half: {metrics['alert_at_half']:.3f}")

    if "loss" in metrics:
        print(f"\n  Loss: {metrics['loss']:.4f}")
    print("=" * 65)


def _first_alert_turn_multiclass(p_agg_seq: np.ndarray, target_class: int,
                                  threshold: float) -> Optional[int]:
    """Find first turn where p_agg for target class >= threshold."""
    for t in range(p_agg_seq.shape[0]):
        # Normalize p_agg at this turn
        p_row = p_agg_seq[t]
        p_norm = p_row / (p_row.sum() + 1e-8)
        if p_norm[target_class] >= threshold:
            return t
    return None


def _first_any_scam_alert(p_agg_seq: np.ndarray, threshold: float) -> Optional[int]:
    """Find first turn where any scam class p_agg_normalized >= threshold (for false alarm)."""
    for t in range(p_agg_seq.shape[0]):
        p_row = p_agg_seq[t]
        p_norm = p_row / (p_row.sum() + 1e-8)
        scam_total = p_norm[1:].sum()  # sum of all scam classes
        if scam_total >= threshold:
            return t
    return None
