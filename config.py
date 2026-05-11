from dataclasses import dataclass, field
from typing import List
import os

BASELINE2_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASELINE2_ROOT), "data", "data_processed")
DATASET_DIR = os.path.join(os.path.dirname(BASELINE2_ROOT), "dataset_scamstream")


# ── Class definitions ────────────────────────────────────────────
# 0=harmless, 1=A, 2=B, 3=C, 4=D
CLASS_NAMES = ["harmless", "A", "B", "C", "D"]
SCENARIO_TO_IDX = {"harmless": 0, "none": 0, "A": 1, "B": 2, "C": 3, "D": 4}
NUM_CLASSES = len(CLASS_NAMES)


@dataclass
class M1Config:
    # Backbone — HaLong Embedding (không cần word segmentation)
    model_name: str = "contextboxai/halong_embedding"
    max_turn_len: int = 96
    max_turns: int = 20

    # Multi-class
    num_classes: int = NUM_CLASSES
    class_names: List[str] = field(default_factory=lambda: list(CLASS_NAMES))

    # Model
    hidden_dim: int = 256
    attn_heads: int = 4
    dropout: float = 0.2

    # Loss weighting — Focal × U-shape temporal weight
    # w(t,N) = (2t/N - 1)^2 * (1 - w_floor) + w_floor
    w_floor: float = 0.1              # minimum weight ở giữa dialogue
    focal_gamma: float = 2.0          # focusing: 0=CE thuần, 2=focal chuẩn
    weighted_lambda: float = 0.0      # runtime (overridden by phase schedule)

    # Per-class weights [harmless, A, B, C, D]
    # Overridden by phase schedule at runtime
    class_weights: List[float] = field(default_factory=lambda: [1.0, 1.0, 1.0, 1.0, 1.0])

    # Augmentation
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
    # Phase 1 (epoch 1-2): encoder frozen, nhẹ auxiliary
    #   → Mục tiêu: head học phân biệt 5 class cơ bản
    phase1_lr: float = 2e-4
    phase1_class_weights: List[float] = field(
        default_factory=lambda: [1.0, 1.0, 1.0, 1.5, 2.0]
        # harmless=1, A=1, B=1, C=1.5 (ít hơn), D=2.0 (ít nhất)
    )
    phase1_lambda_aux: float = 0.1

    # Phase 2 (epoch 3-4): encoder frozen, thêm auxiliary
    #   → Mục tiêu: cải thiện per-class balance, early detection
    phase2_epoch: int = 3
    phase2_lr: float = 2e-4
    phase2_class_weights: List[float] = field(
        default_factory=lambda: [1.5, 1.0, 1.0, 2.0, 3.0]
    )
    phase2_lambda_aux: float = 0.4

    # Phase 3 (epoch 5-15): unfreeze last N layers, fine-tune
    #   → Mục tiêu: squeeze accuracy, đặc biệt minority classes
    phase3_epoch: int = 5
    phase3_head_lr: float = 8e-5
    phase3_encoder_lr: float = 1e-5
    phase3_class_weights: List[float] = field(
        default_factory=lambda: [2.0, 1.0, 1.0, 2.5, 4.0]
    )
    phase3_lambda_aux: float = 0.2
    phase3_unfreeze_layers: int = 3    # unfreeze last N transformer layers

    # Inference
    scam_alert_thresh: float = 0.80  # p(any scam class) >= thresh → alert
    seed: int = 42

    # Paths
    data_dir: str = field(default_factory=lambda: DATA_DIR)
    dataset_dir: str = field(default_factory=lambda: DATASET_DIR)
    output_dir: str = field(default_factory=lambda: os.path.join(BASELINE2_ROOT, "outputs"))
