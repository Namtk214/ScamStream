"""
Evaluation metrics cho M1 (HaLong + CrossTurnAttention).

Giống baseline 1 nhưng thêm early_detection_stats() từ notebook:
  - median_alert_turn  : turn giữa (0-based) model fire alert đúng trên TP
  - mean_alert_turn    : trung bình turn fire alert trên TP
  - median_lead_frac   : alert_turn / n_turns (càng nhỏ = phát hiện càng sớm)
  - alert_at_half      : % TP được phát hiện trong nửa đầu cuộc gọi
"""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from typing import Dict, List, Optional


def compute_streaming_metrics(
    all_dialogue_labels: List[int],
    all_dialogue_probs: List[float],
    all_p_agg: List[np.ndarray],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute dialogue-level and streaming metrics.

    all_dialogue_probs: p_agg at last turn (dialogue-level prediction)
    all_p_agg:          Noisy-OR cumulative per turn [T] — dùng cho streaming detection
    """
    d_labels = np.array(all_dialogue_labels)
    d_probs  = np.array(all_dialogue_probs)
    d_preds  = (d_probs >= threshold).astype(int)

    try:
        auroc = float(roc_auc_score(d_labels, d_probs))
    except ValueError:
        auroc = float("nan")

    detection_delays = []
    num_scam, num_detected   = 0, 0
    num_harmless, num_false_alarms = 0, 0
    all_alert_turns, all_n_turns   = [], []

    for label, p_agg in zip(all_dialogue_labels, all_p_agg):
        first_alert = _first_alert_turn(p_agg, threshold)
        all_alert_turns.append(first_alert)
        all_n_turns.append(len(p_agg))

        if label == 1:
            num_scam += 1
            if first_alert is not None:
                num_detected += 1
                detection_delays.append(first_alert)
        else:
            num_harmless += 1
            if first_alert is not None:
                num_false_alarms += 1

    metrics = {
        # Dialogue-level
        "dialogue_accuracy": float(accuracy_score(d_labels, d_preds)),
        "dialogue_f1":       float(f1_score(d_labels, d_preds, zero_division=0)),
        "auroc":             auroc,
        # Streaming (based on p_agg threshold crossing)
        "detection_rate":      num_detected / max(num_scam, 1),
        "avg_detection_delay": float(np.mean(detection_delays)) if detection_delays else float("nan"),
        "false_alarm_rate":    num_false_alarms / max(num_harmless, 1),
        "num_scam":            num_scam,
        "num_harmless":        num_harmless,
        "num_detected":        num_detected,
        "num_false_alarms":    num_false_alarms,
        # Early-detection stats (từ notebook)
        **early_detection_stats(all_alert_turns, all_n_turns, all_dialogue_labels),
        # Aliases cho train.py early stopping
        "accuracy": float(accuracy_score(d_labels, d_preds)),
        "f1":       float(f1_score(d_labels, d_preds, zero_division=0)),
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
        if label == 1 and alert_t is not None:
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


def print_streaming_report(metrics: Dict[str, float]):
    print("\n" + "=" * 60)
    print("M1 STREAMING EVALUATION REPORT")
    print("=" * 60)

    print("\n  Dialogue-Level Metrics (last-turn prediction):")
    print(f"    Accuracy: {metrics['dialogue_accuracy']:.4f}")
    print(f"    F1:       {metrics['dialogue_f1']:.4f}")
    if not np.isnan(metrics.get("auroc", float("nan"))):
        print(f"    AUROC:    {metrics['auroc']:.4f}")

    print("\n  Streaming Detection (p_agg threshold crossing):")
    print(
        f"    Detection rate:   {metrics['detection_rate']:.4f} "
        f"({metrics['num_detected']}/{metrics['num_scam']})"
    )
    if not np.isnan(metrics["avg_detection_delay"]):
        print(f"    Avg delay:        {metrics['avg_detection_delay']:.2f} turns")
    print(
        f"    False alarm rate: {metrics['false_alarm_rate']:.4f} "
        f"({metrics['num_false_alarms']}/{metrics['num_harmless']})"
    )

    if "median_alert_turn" in metrics:
        print("\n  Early Detection Stats (TP only):")
        print(f"    Median alert turn: {metrics['median_alert_turn']:.1f}")
        print(f"    Mean alert turn:   {metrics['mean_alert_turn']:.1f}")
        print(f"    Median lead frac:  {metrics['median_lead_frac']:.3f}")
        print(f"    Alert in 1st half: {metrics['alert_at_half']:.3f}")

    if "loss" in metrics:
        print(f"\n  Loss: {metrics['loss']:.4f}")
    print("=" * 60)


def _first_alert_turn(turn_probs: np.ndarray, threshold: float) -> Optional[int]:
    for i, p in enumerate(turn_probs):
        if float(p) >= threshold:
            return i
    return None
