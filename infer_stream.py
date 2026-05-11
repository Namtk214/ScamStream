"""
Streaming inference cho M1 Multi-Class Classifier (HaLong + CrossTurnAttention).

Multi-class Noisy-OR online per class:
  p_agg_t^c = 1 − (1 − p_agg_{t-1}^c) × (1 − q_t^c)
  O(1) per turn per class, monotonically non-decreasing per class.

5 classes: harmless (0), A (1), B (2), C (3), D (4)
"""

import dataclasses
import json
import os
import sys

import numpy as np
import torch
from transformers import AutoTokenizer

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from config import M1Config, CLASS_NAMES
from model import M1Classifier


class StreamingInferenceEngine:
    """
    Stateful streaming inference — theo dõi từng dialogue theo dialogue_id.
    Multi-class: returns per-class probabilities.
    """

    def __init__(self, model_path: str, scam_alert_thresh: float = None):
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device_str)

        # Load config
        config_json = os.path.join(model_path, "config.json")
        if os.path.exists(config_json):
            with open(config_json) as f:
                cfg_dict = json.load(f)
            self.cfg = M1Config(**{k: v for k, v in cfg_dict.items()
                                   if k in {f.name for f in dataclasses.fields(M1Config)}})
        else:
            self.cfg = M1Config()

        if scam_alert_thresh is not None:
            self.cfg.scam_alert_thresh = scam_alert_thresh

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = M1Classifier(self.cfg).to(self.device)
        self.model.load_state_dict(
            torch.load(os.path.join(model_path, "model.pt"),
                       map_location=self.device, weights_only=True)
        )
        self.model.eval()

        # Buffer per dialogue
        self._buffers: dict = {}

    def reset(self, dialogue_id: str):
        self._buffers.pop(dialogue_id, None)

    def reset_all(self):
        self._buffers.clear()

    @torch.no_grad()
    def predict_turn(self, dialogue_id: str, text: str) -> dict:
        """
        Nhận 1 turn mới, trả về multi-class probabilities.

        Returns:
          {
            "turn_index":      int,
            "predicted_class": str   (class name),
            "predicted_idx":   int   (class index),
            "class_probs":     dict  {class_name: float},
            "is_scam":         bool  sum(scam probs) >= thresh,
            "scam_prob":       float sum of all scam class probs,
          }
        """
        enc = self.tokenizer(
            text,
            max_length=self.cfg.max_turn_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids_t  = enc["input_ids"]
        mask_t = enc["attention_mask"]

        buf = self._buffers.setdefault(dialogue_id, {"ids": [], "masks": []})
        buf["ids"].append(ids_t.squeeze(0))
        buf["masks"].append(mask_t.squeeze(0))

        T = len(buf["ids"])
        if T > self.cfg.max_turns:
            buf["ids"]   = buf["ids"][-self.cfg.max_turns:]
            buf["masks"] = buf["masks"][-self.cfg.max_turns:]
            T = self.cfg.max_turns

        L = self.cfg.max_turn_len
        input_ids  = torch.zeros(1, self.cfg.max_turns, L, dtype=torch.long)
        attn_masks = torch.zeros(1, self.cfg.max_turns, L, dtype=torch.long)
        turn_mask  = torch.zeros(1, self.cfg.max_turns, dtype=torch.bool)

        for i, (ids, mask) in enumerate(zip(buf["ids"], buf["masks"])):
            input_ids[0, i]  = ids
            attn_masks[0, i] = mask
            turn_mask[0, i]  = True

        input_ids  = input_ids.to(self.device)
        attn_masks = attn_masks.to(self.device)
        turn_mask  = turn_mask.to(self.device)

        output = self.model(input_ids, attn_masks, turn_mask)

        # Normalized p_agg at current turn
        p_agg_t = output["p_agg"][0, T - 1].cpu().numpy()   # [C]
        p_norm = p_agg_t / (p_agg_t.sum() + 1e-8)

        pred_idx = int(p_norm.argmax())
        pred_name = self.cfg.class_names[pred_idx]
        scam_prob = float(p_norm[1:].sum())

        class_probs = {self.cfg.class_names[c]: float(p_norm[c])
                       for c in range(self.cfg.num_classes)}

        return {
            "turn_index":      T - 1,
            "predicted_class": pred_name,
            "predicted_idx":   pred_idx,
            "class_probs":     class_probs,
            "is_scam":         scam_prob >= self.cfg.scam_alert_thresh,
            "scam_prob":       scam_prob,
        }

    def predict_conversation(self, turns: list, dialogue_id: str = "default") -> list:
        """
        turns: list of str
        Trả về list results (1 per turn).
        """
        self.reset(dialogue_id)
        results = []
        for turn in turns:
            text = turn if isinstance(turn, str) else turn.get("text", turn.get("content", ""))
            result = self.predict_turn(dialogue_id, text)
            results.append(result)
        return results


# ── Demo CLI ──────────────────────────────────────────────────────

def _demo():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="outputs/best_model")
    parser.add_argument("--threshold", type=float, default=0.80)
    args = parser.parse_args()

    model_path = os.path.join(os.path.dirname(__file__), args.model)
    if not os.path.exists(os.path.join(model_path, "model.pt")):
        print(f"Model not found: {model_path}")
        return

    engine = StreamingInferenceEngine(model_path, scam_alert_thresh=args.threshold)

    conversations = [
        {
            "id": "scam_A_demo",
            "label": "SCAM-A (giả danh cơ quan)",
            "turns": [
                "Alo, tôi đang gọi từ công an tỉnh, cần xác minh một số thông tin của bạn.",
                "Dạ vâng, tôi nghe ạ.",
                "Tài khoản của bạn có liên quan đến đường dây rửa tiền. Bạn cần chuyển tiền vào tài khoản an toàn ngay.",
                "Ơ... chuyển tiền như thế nào ạ?",
                "Chuyển hết 50 triệu vào tài khoản này: 0123456789, ngân hàng Vietcombank. Khẩn cấp, đừng nói với ai.",
            ],
        },
        {
            "id": "harmless_demo",
            "label": "HARMLESS",
            "turns": [
                "Alo, bạn ơi mình gọi để hỏi về lịch họp ngày mai.",
                "Ừ, mình nghe. Họp lúc 9 giờ sáng phải không?",
                "Đúng rồi, phòng họp tầng 3. Bạn nhớ mang tài liệu nhé.",
                "Ok, mình sẽ chuẩn bị. Còn cần gì nữa không?",
                "Không, vậy thôi. Hẹn gặp ngày mai nhé.",
            ],
        },
    ]

    for conv in conversations:
        print(f"\n{'='*60}")
        print(f"Demo: {conv['id']} (Ground truth: {conv['label']})")
        print(f"{'='*60}")

        results = engine.predict_conversation(conv["turns"], dialogue_id=conv["id"])
        for r in results:
            scam_bar = "█" * int(r["scam_prob"] * 20) + "░" * (20 - int(r["scam_prob"] * 20))
            alert = " ← ALERT" if r["is_scam"] else ""
            text  = conv["turns"][r["turn_index"]][:55]
            # Top 2 classes
            sorted_probs = sorted(r["class_probs"].items(), key=lambda x: -x[1])
            top2 = " ".join(f"{k}={v:.3f}" for k, v in sorted_probs[:2])
            print(f"  T{r['turn_index']+1:02d} [{scam_bar}] {top2}{alert}")
            print(f"       \"{text}{'...' if len(text) >= 55 else ''}\"")


if __name__ == "__main__":
    _demo()
