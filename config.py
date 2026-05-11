from dataclasses import dataclass, field
import os

BASELINE2_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASELINE2_ROOT), "data", "data_processed")
DATASET_DIR = os.path.join(BASELINE2_ROOT, "dataset")


@dataclass
class M1Config:
    # Backbone — HaLong Embedding (không cần word segmentation)
    model_name: str = "contextboxai/halong_embedding"
    max_turn_len: int = 96
    max_turns: int = 20

    # Model
    hidden_dim: int = 256
    attn_heads: int = 4
    dropout: float = 0.2

    # Loss weighting — Focal × U-shape temporal weight
    # w(t,N) = (2t/N - 1)^2 * (1 - w_floor) + w_floor
    w_floor: float = 0.1              # minimum weight ở giữa dialogue
    focal_gamma: float = 2.0          # focusing: 0=BCE thuần, 2=focal chuẩn
    class_weight_harmless: float = 5.0   # runtime default (overridden by phase schedule)
    weighted_lambda: float = 0.0 

    # Augmentation (Fix 02 from HSM-Net §03)
    truncate_aug: bool = True
    aug_k: int = 2            # số bản truncate mỗi scam dialogue
    aug_min_turns: int = 2    # số turns tối thiểu sau truncate

    # Training
    batch_size: int = 8
    grad_accum_steps: int = 4  # effective batch = batch_size * grad_accum_steps
    use_grad_ckpt: bool = True
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    warmup_ratio: float = 0.1  # % tổng steps dùng để warmup
    num_epochs: int = 15

    # ── Phase schedule ──────────────────────────────────────────
    # Phase 1 (epoch 1 → phase2_epoch-1): encoder frozen, pure Noisy-OR Focal
    #   → Mục tiêu: học phân biệt, chống báo scam hết, AUROC tăng
    phase1_lr: float = 2e-4
    phase1_harm_weight: float = 2.0
    phase1_lambda_aux: float = 0.1     # tắt auxiliary

    # Phase 2 (phase2_epoch → phase3_epoch-1): encoder frozen, thêm auxiliary nhẹ
    #   → Mục tiêu: giữ false alarm, thêm early detection
    phase2_epoch: int = 3
    phase2_lr: float = 2e-4
    phase2_harm_weight: float = 3.0
    phase2_lambda_aux: float = 0.4

    # Phase 3 (phase3_epoch → end): unfreeze last N layers, fine-tune
    #   → Mục tiêu: squeeze thêm accuracy
    phase3_epoch: int = 5
    phase3_head_lr: float = 8e-5       # hơi cao hơn P2 drop, co-adapt với encoder mới unfreeze
    phase3_encoder_lr: float = 1e-5    # 8× nhỏ hơn head — discriminative fine-tuning chuẩn
    phase3_harm_weight: float = 4.0
    phase3_lambda_aux: float = 0.2
    phase3_unfreeze_layers: int = 3    # unfreeze last N transformer layers

    # Inference
    alert_thresh: float = 0.80
    threshold: float = 0.5
    seed: int = 42

    val_ratio: float = 0.3
    test_ratio: float = 0.10

    # Paths
    data_dir: str = field(default_factory=lambda: DATA_DIR)
    dataset_dir: str = field(default_factory=lambda: DATASET_DIR)
    output_dir: str = field(default_factory=lambda: os.path.join(BASELINE2_ROOT, "outputs"))
