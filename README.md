# ScamStream Multi-Class — Streaming Scam Call Detection by Scenario

Mô hình phát hiện cuộc gọi lừa đảo theo thời gian thực với phân loại **5 classes** theo kịch bản lừa đảo, sử dụng **HaLong Embedding** + **Causal CrossTurnAttention** + **Multi-Head Noisy-OR Focal CE Loss** + **U-shape Weighted Prefix Auxiliary Loss**.

## Classes

| Class | Scenario | Mô tả | Ví dụ |
|-------|----------|-------|-------|
| `harmless` (0) | — | Cuộc gọi bình thường | Giao hàng, hẹn họp, hỏi thăm |
| `A` (1) | Giả danh cơ quan | Mạo danh công an, ngân hàng, đăng kiểm | "Em gọi từ trung tâm đăng kiểm" |
| `B` (2) | Lừa tài khoản/giao dịch | Kích hoạt dịch vụ, phí tự động, hoàn trả | "Đăng ký thành viên thành công, phí trừ tự động" |
| `C` (3) | Lừa qua người quen/ship | Giả vờ quen biết, lừa chuyển tiền ship | "Anh chuyển tiền ship vào đó rồi sao?" |
| `D` (4) | Đe dọa/tống tiền | Đe dọa lộ thông tin, tống tiền | "Mày có muốn xử lý cho im lặng không?" |

## Architecture

```
Turn text → HaLong Encoder ([CLS]) → Linear Proj → CrossTurnAttention (causal) → MLP Head → C logits
         → softmax → q_t^c (per-turn, per-class evidence)
         → Noisy-OR per-class → p_agg^c (cumulative, monotonically non-decreasing)
         → normalize → class prediction
```

- **Encoder**: `contextboxai/halong_embedding` — Vietnamese pretrained embedding, không cần word segmentation
- **CrossTurnAttention**: Causal attention — turn `t` chỉ attend tới các turn trước đó `0..t-1`
- **Multi-Head Output**: C=5 logits per turn → softmax → per-class evidence

### Multi-Head Noisy-OR

Khác với binary version (1 Noisy-OR head), multi-class version có **1 Noisy-OR head per class**:

```
q_t^c    = softmax(logit_t)[c]                          ← per-turn evidence cho class c
p_agg_t^c = 1 − ∏_{i=0}^{t} (1 − q_i^c)               ← cumulative per class, monotonically non-decreasing
prediction = argmax(normalize(p_agg_final))              ← tại turn cuối
```

Mỗi class tích lũy evidence **độc lập** qua Noisy-OR → p_agg per class luôn **không giảm** → phù hợp cho streaming detection.

### Loss: L_total = L_main + λ × L_aux

#### 1) Focal Cross-Entropy (Main)

```
p_norm_final = normalize(p_agg_final)                    ← pseudo-probabilities tại turn cuối
L_main = Focal_CE(p_norm_final, y) × class_weight[y]
```

| Cơ chế | Công thức | Vai trò |
|--------|-----------|---------|
| **Focal** | `(1-p_t)^γ` | Focus vào hard examples, down-weight easy |
| **Noisy-OR** | `p_agg^c = 1 − ∏(1 − q_i^c)` | Tích lũy evidence per-class qua các turns |
| **Class weight** | `class_weights[c]` | Cân bằng imbalance (D có ít data nhất → weight cao nhất) |

#### 2) U-shape Weighted Prefix Auxiliary Loss

Apply **Focal CE** trên normalized `p_agg` tại **mỗi prefix** với trọng số U-shape:

```
w(t, N) = (2t/N - 1)² × (1 - w_floor) + w_floor
L_aux  = Σ w(t,N) × Focal_CE(normalize(p_agg_t), y)  /  Σ w(t,N)
```

| Vị trí | Weight | Ý nghĩa |
|--------|--------|---------|
| **Turn đầu** (t=0) | 1.0 (max) | Phạt nặng nếu miss sớm |
| **Turn giữa** (t=N/2) | w_floor (0.1) | Phạt nhẹ — model thiếu info |
| **Turn cuối** (t=N) | 1.0 (max) | Phạt nặng nếu đủ info mà vẫn sai |

### Streaming Inference — Online Noisy-OR

Khi inference streaming, `p_agg` per-class được tính **online** O(1) per turn:

```
p_agg_0^c = 0
p_agg_t^c = 1 − (1 − p_agg_{t-1}^c) × (1 − q_t^c)    ← chỉ cần q_t^c mới + p_agg^c cũ
```

## 3-Phase Training Schedule

| Phase | Epochs | Encoder | class_weights | λ_aux | Mục tiêu |
|-------|--------|---------|---------------|-------|----------|
| **Phase 1** | 1-2 | Frozen | [1, 1, 1, 1.5, 2] | 0.1 | Head học phân biệt 5 class |
| **Phase 2** | 3-4 | Frozen | [1.5, 1, 1, 2, 3] | 0.4 | Cải thiện minority classes + early detection |
| **Phase 3** | 5-15 | Unfreeze last 3 layers | [2, 1, 1, 2.5, 4] | 0.2 | Fine-tune encoder, squeeze accuracy |

Mỗi phase tạo **optimizer + cosine scheduler mới** với LR và class weights riêng.

## Project Structure

```
ScamStream/
├── config.py              # M1Config — hyperparameters, class definitions, phase schedule
├── model.py               # M1Classifier, CrossTurnAttention, Multi-Head Noisy-OR Focal CE
├── dataset.py             # DialogueDataset, collate_fn, truncate_augment (scenario-aware)
├── train.py               # 3-phase training loop + streaming preview (multi-class)
├── test.py                # Multi-class evaluation & error analysis
├── metrics.py             # Per-class P/R/F1, confusion matrix, streaming detection metrics
├── infer_stream.py        # Real-time streaming inference (per-class Noisy-OR online)
└── prepare_datasets.py    # Data preprocessing & conversion pipeline
```

## Data Format

Dataset phải là file JSON với format sau (field `scenario` bắt buộc cho scam):

```json
[
  {
    "label": "scam",
    "scenario": "A",
    "turns": [
      "Alo, tôi đang gọi từ công an...",
      "Dạ vâng, tôi nghe ạ.",
      "..."
    ]
  },
  {
    "label": "harmless",
    "turns": [
      "Alo, chị ơi em giao hàng...",
      "..."
    ]
  }
]
```

## Data Sources

| Source | Mô tả | Dialogues |
|--------|--------|-----------|
| **Real.xlsx** | Dữ liệu thực — chia 50/50 → real1, real2 | ~830 |
| **Tele-data** | Dữ liệu tele (Tele28k, etc.) | ~3,600 |
| **Synthetic-data** | Dữ liệu synthetic (Vanilla, Viscam, Bothbosu) | Varies |

Test set luôn là **real2.json** (416 dialogues: 237 harmless, 56 A, 61 B, 39 C, 23 D).

## Requirements

```bash
pip install torch transformers numpy scikit-learn openpyxl
```

- **Python 3.8+**
- **PyTorch** & **Transformers**: Core models
- **numpy** & **scikit-learn**: Metrics (F1, Precision, Recall, Confusion Matrix)
- **openpyxl**: Export error analysis Excel

## Quick Start

### 1. Training

```bash
# Train với real1 data
python train.py \
  --train-file ../dataset_scamstream/exp3_real+syn/train/real1.json \
  --test-file ../dataset_scamstream/exp1_prompt_vs_viscam/test/real2.json

# Debug mode: 2 epochs, no augmentation
python train.py \
  --train-file ../dataset_scamstream/exp3_real+syn/train/real1.json \
  --test-file ../dataset_scamstream/exp1_prompt_vs_viscam/test/real2.json \
  --debug

# Train với Viscam synthetic data
python train.py \
  --train-file ../dataset_scamstream/exp1_prompt_vs_viscam/train/Viscam.json \
  --test-file ../dataset_scamstream/exp1_prompt_vs_viscam/test/real2.json \
  --output-dir outputs/viscam
```

### 2. Evaluation

```bash
python test.py \
  --data ../dataset_scamstream/exp1_prompt_vs_viscam/test/real2.json \
  --model outputs/best_model
```

Output bao gồm:
- **Per-class metrics**: Precision, Recall, F1 cho mỗi class (harmless, A, B, C, D)
- **Confusion matrix**: 5×5 matrix
- **Streaming detection**: Per-class detection rate, avg delay
- **Early detection stats**: Avg alert turn, Alert in 1st half
- **Error analysis Excel**: Từng turn hiển thị `p_agg` per class

### 3. Streaming Inference

```bash
python infer_stream.py --model outputs/best_model
```

Output mỗi turn:
```
T01 [████░░░░░░░░░░░░░░░░] harmless=0.850 A=0.120
     "Alo, tôi đang gọi từ công an tỉnh..."
T02 [████████░░░░░░░░░░░░] A=0.452 harmless=0.350
     "Tài khoản của bạn có liên quan đến..."
T03 [████████████████░░░░] A=0.786 harmless=0.120 ← ALERT
     "Chuyển hết 50 triệu vào tài khoản..."
```

## Key Hyperparameters

| Parameter | Default | Mô tả |
|-----------|---------|-------|
| `model_name` | `contextboxai/halong_embedding` | Backbone encoder |
| `num_classes` | 5 | Số classes [harmless, A, B, C, D] |
| `max_turn_len` | 96 | Max tokens per turn |
| `max_turns` | 20 | Max turns per dialogue |
| `hidden_dim` | 256 | Hidden dimension |
| `attn_heads` | 4 | Attention heads in CrossTurnAttention |
| `focal_gamma` | 2.0 | Focal loss gamma |
| `w_floor` | 0.1 | U-shape weight minimum |
| `batch_size` | 8 | Batch size |
| `grad_accum_steps` | 4 | Gradient accumulation (effective batch = 32) |
| `num_epochs` | 15 | Total epochs |
| `scam_alert_thresh` | 0.80 | Streaming alert threshold (sum scam probs) |

### Phase Schedule

| Phase | Epochs | LR | class_weights | λ_aux |
|-------|--------|----|---------------|-------|
| 1 | 1-2 | 2e-4 | [1, 1, 1, 1.5, 2] | 0.1 |
| 2 | 3-4 | 2e-4 | [1.5, 1, 1, 2, 3] | 0.4 |
| 3 | 5-15 | head=8e-5, enc=1e-5 | [2, 1, 1, 2.5, 4] | 0.2 |

## Experiment Datasets

Dùng data từ `dataset_scamstream/`:

| Experiment | Train File | Test File | Mô tả |
|------------|-----------|-----------|-------|
| **exp1** (Vanilla) | `exp1.../train/Vanilla.json` | `exp1.../test/real2.json` | Vanilla prompt synthetic |
| **exp1** (Viscam) | `exp1.../train/Viscam.json` | `exp1.../test/real2.json` | Viscam synthetic |
| **exp2** (Bothbosu) | `exp2.../train/Bothbosu.json` | `exp2.../test/real2.json` | Adversarial Bothbosu |
| **exp2** (Tele28k) | `exp2.../train/Tele28k.json` | `exp2.../test/real2.json` | Tele 28k subset |
| **exp3** (real1) | `exp3.../train/real1.json` | `exp3.../test/real2.json` | Real data only |
| **exp3** (Viscam) | `exp3.../train/Viscam.json` | `exp3.../test/real2.json` | Real + Viscam synthetic |
| **exp4** | — | `exp4.../test/real2.json` | LLM evaluation only |
