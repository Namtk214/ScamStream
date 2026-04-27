"""
Streaming inference cho M1 Classifier (HaLong + CrossTurnAttention).

Cơ chế: duy trì buffer input_ids/attn_masks của các turns đã đến.
Mỗi turn mới: thêm vào buffer → chạy full forward pass trên prefix → lấy p_t ở vị trí cuối.

Khác baseline 1 (GRU): không có stateful hidden; thay vào đó re-encode toàn bộ prefix.
Đây là chi phí của cross-attention: O(T^2) nhưng không có information leak từ tương lai.
"""

import dataclasses
import json
import os
import sys

import torch
from transformers import AutoTokenizer

_dir = os.path.dirname(os.path.abspath(__file__))
if _dir not in sys.path:
    sys.path.insert(0, _dir)

from config import M1Config
from model import M1Classifier


class StreamingInferenceEngine:
    """
    Stateful streaming inference — theo dõi từng dialogue theo dialogue_id.
    """

    def __init__(self, model_path: str, threshold: float = None):
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

        if threshold is not None:
            self.cfg.alert_thresh = threshold

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.model = M1Classifier(self.cfg).to(self.device)
        self.model.load_state_dict(
            torch.load(os.path.join(model_path, "model.pt"),
                       map_location=self.device, weights_only=True)
        )
        self.model.eval()

        # Buffer per dialogue: {dialogue_id: {"input_ids": [T, L], "attn_masks": [T, L]}}
        self._buffers: dict = {}

    def reset(self, dialogue_id: str):
        self._buffers.pop(dialogue_id, None)

    def reset_all(self):
        self._buffers.clear()

    @torch.no_grad()
    def predict_turn(self, dialogue_id: str, text: str, speaker: int = None) -> dict:
        """
        Nhận 1 turn mới, trả về probability tại turn đó.

        Returns:
          {
            "turn_index":  int   (0-based),
            "q_t":         float  per-turn evidence P(SCAM),
            "p_agg":       float  Noisy-OR cumulative (monotonically non-decreasing),
            "is_scam":     bool   p_agg >= alert_thresh,
            "probability": float  (alias for p_agg, backward compat),
          }
        """
        # Tokenize turn mới
        enc = self.tokenizer(
            text,
            max_length=self.cfg.max_turn_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ids_t  = enc["input_ids"]    # [1, L]
        mask_t = enc["attention_mask"]  # [1, L]

        # Thêm vào buffer
        buf = self._buffers.setdefault(dialogue_id, {"ids": [], "masks": [], "p_agg": 0.0})
        buf["ids"].append(ids_t.squeeze(0))
        buf["masks"].append(mask_t.squeeze(0))

        T = len(buf["ids"])
        if T > self.cfg.max_turns:
            # Giữ max_turns turns gần nhất
            buf["ids"]   = buf["ids"][-self.cfg.max_turns:]
            buf["masks"] = buf["masks"][-self.cfg.max_turns:]
            T = self.cfg.max_turns

        # Build padded tensors [1, max_turns, L]
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
        turn_probs = output["turn_probs"][0, :T].cpu().tolist()  # [T]
        q_t = turn_probs[-1]   # per-turn evidence

        # Online Noisy-OR update: p_agg = 1 - (1 - p_prev) * (1 - q_t)
        p_agg_prev = buf["p_agg"]
        p_agg = 1.0 - (1.0 - p_agg_prev) * (1.0 - q_t)
        buf["p_agg"] = p_agg

        turn_index = T - 1

        return {
            "turn_index":     turn_index,
            "q_t":            q_t,
            "p_agg":          p_agg,
            "is_scam":        p_agg >= self.cfg.alert_thresh,
            "probability":    p_agg,       # backward compat alias
            "all_turn_probs": turn_probs,
        }

    def predict_conversation(self, turns: list, dialogue_id: str = "default") -> list:
        """
        turns: list of str  (schema mới)
               hoặc list of dict {"text": str, ...}  (schema cũ, backwards compat)
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

    engine = StreamingInferenceEngine(model_path, threshold=args.threshold)

    conversations = [
        {
            "id": "scam_demo",
            "label": "SCAM",
            "turns": [
                {"text": "Alo, tôi đang gọi từ công an tỉnh, cần xác minh một số thông tin của bạn."},
                {"text": "Dạ vâng, tôi nghe ạ."},
                {"text": "Tài khoản của bạn có liên quan đến đường dây rửa tiền. Bạn cần chuyển tiền vào tài khoản an toàn ngay."},
                {"text": "Ơ... chuyển tiền như thế nào ạ?"},
                {"text": "Chuyển hết 50 triệu vào tài khoản này: 0123456789, ngân hàng Vietcombank. Khẩn cấp, đừng nói với ai."},
            ],
        },
        {
            "id": "harmless_demo",
            "label": "HARMLESS",
            "turns": [
                {"text": "Alo, bạn ơi mình gọi để hỏi về lịch họp ngày mai."},
                {"text": "Ừ, mình nghe. Họp lúc 9 giờ sáng phải không?"},
                {"text": "Đúng rồi, phòng họp tầng 3. Bạn nhớ mang tài liệu nhé."},
                {"text": "Ok, mình sẽ chuẩn bị. Còn cần gì nữa không?"},
                {"text": "Không, vậy thôi. Hẹn gặp ngày mai nhé."},
            ],
        },
    ]

    for conv in conversations:
        print(f"\n{'='*60}")
        print(f"Demo: {conv['id']} (Ground truth: {conv['label']})")
        print(f"{'='*60}")

        results = engine.predict_conversation(conv["turns"], dialogue_id=conv["id"])
        for r in results:
            bar    = "█" * int(r["p_agg"] * 20) + "░" * (20 - int(r["p_agg"] * 20))
            status = " ← ALERT" if r["is_scam"] else ""
            text   = conv["turns"][r["turn_index"]]["text"][:55]
            print(f"  T{r['turn_index']+1:02d} [{bar}] q={r['q_t']:.3f} p_agg={r['p_agg']:.3f}{status}")
            print(f"       \"{text}...\'" if len(text) >= 55 else f"       \"{text}\"")


if __name__ == "__main__":
    _demo()
