"""
Dataset cho M1 Multi-Class (HaLong + CrossTurnAttention).

Schema data:
  {"label": "scam"|"harmless", "scenario": "A"|"B"|"C"|"D"|None, "turns": ["text1", ...]}

Multi-class mapping (dùng scenario field):
  harmless (no scenario) → 0
  scam + scenario A      → 1
  scam + scenario B      → 2
  scam + scenario C      → 3
  scam + scenario D      → 4
"""

import json
import random
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from config import SCENARIO_TO_IDX


def load_json(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def get_class_label(dlg: Dict) -> int:
    """Extract multi-class label from dialogue dict."""
    if dlg["label"] == "harmless":
        return 0
    scenario = dlg.get("scenario", "A")  # default A nếu scam mà thiếu scenario
    return SCENARIO_TO_IDX.get(scenario, 1)


def split_data(dialogues: List[Dict], val_ratio: float, test_ratio: float,
               seed: int = 42) -> Tuple[List, List, List]:
    rng  = random.Random(seed)
    data = dialogues.copy()
    rng.shuffle(data)
    n        = len(data)
    n_test   = int(n * test_ratio)
    n_val    = int(n * val_ratio)
    return data[n_test + n_val:], data[n_test:n_test + n_val], data[:n_test]


def truncate_augment(dialogues: List[Dict], k: int, min_turns: int) -> List[Dict]:
    """
    Tạo k bản truncate của mỗi SCAM dialogue (bất kể scenario) để buộc model
    học từ partial context.
    HARMLESS không augment để tránh imbalance thêm.
    """
    augmented = []
    for dlg in dialogues:
        augmented.append(dlg)
        if dlg["label"] != "scam":
            continue
        n = len(dlg["turns"])
        if n <= min_turns:
            continue
        possible = list(range(min_turns, n))
        lengths  = random.sample(possible, min(k, len(possible)))
        for trunc_len in sorted(lengths):
            augmented.append({
                "label":    dlg["label"],
                "scenario": dlg.get("scenario"),
                "turns":    dlg["turns"][:trunc_len],
            })
    return augmented


class DialogueDataset(Dataset):
    """
    Mỗi sample = 1 dialogue, padded tới max_turns.
    Multi-class label dựa trên scenario field.
    """

    def __init__(self, dialogues: List[Dict], tokenizer, max_turn_len: int, max_turns: int):
        self.dialogues    = dialogues
        self.tok          = tokenizer
        self.max_turn_len = max_turn_len
        self.max_turns    = max_turns

    def __len__(self) -> int:
        return len(self.dialogues)

    def __getitem__(self, idx: int) -> Dict:
        dlg    = self.dialogues[idx]
        turns  = dlg["turns"][:self.max_turns]
        n_real = len(turns)

        input_ids_list, attn_mask_list = [], []
        for turn in turns:
            enc = self.tok(
                turn,
                max_length=self.max_turn_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids_list.append(enc["input_ids"].squeeze(0))
            attn_mask_list.append(enc["attention_mask"].squeeze(0))

        pad_ids  = torch.zeros(self.max_turn_len, dtype=torch.long)
        pad_mask = torch.zeros(self.max_turn_len, dtype=torch.long)
        for _ in range(self.max_turns - n_real):
            input_ids_list.append(pad_ids)
            attn_mask_list.append(pad_mask)

        turn_mask = torch.zeros(self.max_turns, dtype=torch.bool)
        turn_mask[:n_real] = True

        label = get_class_label(dlg)
        return {
            "input_ids":  torch.stack(input_ids_list),   # [max_turns, max_turn_len]
            "attn_masks": torch.stack(attn_mask_list),   # [max_turns, max_turn_len]
            "turn_mask":  turn_mask,                      # [max_turns]
            "n_turns":    torch.tensor(n_real),
            "label":      torch.tensor(label, dtype=torch.long),
        }


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Padding đã làm trong __getitem__, chỉ stack lại."""
    return {
        "input_ids":  torch.stack([b["input_ids"]  for b in batch]),
        "attn_masks": torch.stack([b["attn_masks"] for b in batch]),
        "turn_mask":  torch.stack([b["turn_mask"]  for b in batch]),
        "n_turns":    torch.stack([b["n_turns"]    for b in batch]),
        "labels":     torch.stack([b["label"]      for b in batch]),
    }
