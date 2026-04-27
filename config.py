from dataclasses import dataclass, field
import os

BASELINE2_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(os.path.dirname(BASELINE2_ROOT), "data", "data_processed")


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
    class_weight_harmless: float = 8.0   # sample-level balance — 2× ratio to suppress FA

    # Augmentation (Fix 02 from HSM-Net §03)
    truncate_aug: bool = True
    aug_k: int = 3            # số bản truncate mỗi scam dialogue
    aug_min_turns: int = 2    # số turns tối thiểu sau truncate

    # Training
    batch_size: int = 2
    grad_accum_steps: int = 8  # effective batch = batch_size * grad_accum_steps
    use_grad_ckpt: bool = True
    lr: float = 2e-5
    weight_decay: float = 1e-2
    grad_clip: float = 1.0
    warmup_ratio: float = 0.1  # % tổng steps dùng để warmup
    num_epochs: int = 10
    unfreeze_epoch: int = 3    # epoch bắt đầu unfreeze encoder (lr * 0.1)

    # Inference
    alert_thresh: float = 0.80
    threshold: float = 0.5
    seed: int = 42

    val_ratio: float = 0.15
    test_ratio: float = 0.10

    # Paths
    data_dir: str = field(default_factory=lambda: DATA_DIR)
    output_dir: str = field(default_factory=lambda: os.path.join(BASELINE2_ROOT, "outputs"))
