import sys, os, json, dataclasses, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from transformers import AutoTokenizer
from dataset import DialogueDataset, collate_fn, load_json
from model import M1Classifier
from config import M1Config

model_path = os.path.join(os.path.dirname(__file__), "outputs", "best_model")
device = torch.device("cpu")

with open(os.path.join(model_path, "config.json")) as f:
    cfg_dict = json.load(f)
cfg = M1Config(**{k: v for k, v in cfg_dict.items()
                  if k in {f.name for f in dataclasses.fields(M1Config)}})

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = M1Classifier(cfg).to(device)
model.load_state_dict(torch.load(os.path.join(model_path, "model.pt"),
                                  map_location=device, weights_only=True))
model.eval()

data_path = os.path.join(os.path.dirname(__file__), "..", "data", "data_processed", "test.json")
raw = load_json(data_path)
dataset = DialogueDataset(raw, tokenizer, cfg.max_turn_len, cfg.max_turns)

threshold = 0.5
FP, FN = [], []

with torch.no_grad():
    for idx in range(len(dataset)):
        item = dataset[idx]
        dlg = raw[idx]
        batch = collate_fn([item])
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["input_ids"], batch["attn_masks"], batch["turn_mask"])
        d_prob = float(out["dialogue_probs"][0])
        vlen = int(batch["turn_mask"][0].sum())
        t_probs = out["turn_probs"][0, :vlen].tolist()
        true_lbl = dlg["label"]
        pred_lbl = "scam" if d_prob >= threshold else "harmless"
        if true_lbl == "harmless" and pred_lbl == "scam":
            FP.append((idx, d_prob, t_probs, dlg["turns"][:vlen]))
        elif true_lbl == "scam" and pred_lbl == "harmless":
            FN.append((idx, d_prob, t_probs, dlg["turns"][:vlen]))

print(f"Total FP: {len(FP)}, FN: {len(FN)}")

print("\n===== TOP 10 FALSE POSITIVES (harmless → predicted scam) =====")
for idx, dp, tps, turns in sorted(FP, key=lambda x: -x[1])[:10]:
    print(f"\n  [FP #{idx}] p_final={dp:.4f}  ({len(turns)} turns)")
    for t, (p, turn) in enumerate(zip(tps, turns), 1):
        alert = " [!]" if p >= threshold else ""
        print(f"    T{t}: p={p:.4f}{alert}  {str(turn)[:90]}")

print("\n===== TOP 5 FALSE NEGATIVES (scam → predicted harmless) =====")
for idx, dp, tps, turns in sorted(FN, key=lambda x: x[1])[:5]:
    print(f"\n  [FN #{idx}] p_final={dp:.4f}  ({len(turns)} turns)")
    for t, (p, turn) in enumerate(zip(tps, turns), 1):
        alert = " [!]" if p >= threshold else ""
        print(f"    T{t}: p={p:.4f}{alert}  {str(turn)[:90]}")
