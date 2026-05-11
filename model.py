"""
M1 Multi-Class Classifier — HaLong Embedding + Causal CrossTurnAttention
              + Multi-Head Noisy-OR Focal Loss.

Architecture:
  turn text → HaLong encoder ([CLS] token) → e_t [embed_dim]
            → Linear projection → h_t [hidden_dim]
            → CrossTurnAttention (causal: turn_t attends to h_0..h_{t-1})
            → MLP head → C logits per turn
            → softmax → q_t^c (per-turn, per-class evidence)
            → Noisy-OR per-class → p_agg^c (cumulative, monotonically non-decreasing)

Multi-class: 5 classes [harmless, A, B, C, D]
  q_t^c  = softmax(logit_t)[c]           ← per-turn evidence cho class c
  p_agg_t^c = 1 − ∏(1 − q_i^c)          ← Noisy-OR cumulative per class
  Prediction = argmax(p_agg_final)

Training Loss:
  L_total = L_main + λ × L_aux

  1) Main: Focal CE(p_agg_final_normalized, y) × class_weight
     p_agg_final = Noisy-OR tại turn cuối, normalize lại → pseudo-probabilities
     Focal (1-p_t)^γ focus hard examples, class_weight cân bằng imbalance

  2) Auxiliary: Σ w(t,N) × Focal CE(p_agg_t_normalized, y)
     U-shape weighted prefix trên cumulative Noisy-OR tại mỗi prefix
     Thúc đẩy model predict đúng class sớm dần

Streaming inference:
  Online Noisy-OR per class: p_agg_t^c = 1 − (1 − p_agg_{t-1}^c) × (1 − q_t^c)
  O(1) per turn per class, monotonically non-decreasing per class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# ─────────────────────────────────────────────────────────────────
# Noisy-OR utilities (multi-class)
# ─────────────────────────────────────────────────────────────────

def _compute_p_agg_multiclass(turn_probs: torch.Tensor, turn_mask: torch.Tensor,
                               eps: float = 1e-6) -> torch.Tensor:
    """
    Tính Noisy-OR cumulative p_agg cho toàn batch, per-class.

    Input:
      turn_probs : [B, T, C]  — per-turn evidence q_t^c (softmax)
      turn_mask  : [B, T]     — True = real turn
    Output:
      p_agg      : [B, T, C]  — cumulative Noisy-OR per class, padding giữ nguyên
    """
    q = turn_probs.clamp(eps, 1 - eps)                          # [B, T, C]
    mask_3d = turn_mask.float().unsqueeze(-1)                   # [B, T, 1]
    log_not_q = torch.log1p(-q) * mask_3d                      # zero padding
    cumsum_log = torch.cumsum(log_not_q, dim=1)                 # [B, T, C]
    p_agg = (1.0 - torch.exp(cumsum_log)).clamp(eps, 1 - eps)  # [B, T, C]
    return p_agg


def _normalize_p_agg(p_agg: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normalize p_agg per-class thành probabilities (sum to 1).
    p_agg: [B, T, C] hoặc [B, C]
    """
    return p_agg / (p_agg.sum(dim=-1, keepdim=True) + eps)


def _focal_ce(p_norm: torch.Tensor, targets: torch.Tensor,
              gamma: float = 2.0, eps: float = 1e-6) -> torch.Tensor:
    """
    Focal Cross-Entropy (element-wise, no reduction).

    p_norm  : [..., C]  — normalized probabilities
    targets : [...]     — class indices (long)
    Returns : [...]     — focal CE per sample

    FocalCE = -(1-p_t)^γ × log(p_t) × (γ+1)
    với p_t = probability của class đúng.
    """
    p_norm = p_norm.clamp(eps, 1 - eps)
    # Gather probability of correct class
    p_t = p_norm.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    focal_mod = (1 - p_t) ** gamma
    ce = -torch.log(p_t)
    return focal_mod * ce * (gamma + 1)


# ─────────────────────────────────────────────────────────────────
# Loss functions (multi-class)
# ─────────────────────────────────────────────────────────────────

def noisy_or_focal_ce_loss(turn_probs: torch.Tensor, labels: torch.Tensor,
                            turn_mask: torch.Tensor, p_agg: torch.Tensor,
                            gamma: float = 2.0,
                            class_weights: torch.Tensor = None,
                            weighted_lambda: float = 0.5,
                            w_floor: float = 0.1,
                            eps: float = 1e-6) -> torch.Tensor:
    """
    Multi-class Noisy-OR Focal CE Loss.

    1) Main: Focal CE(p_agg_final_normalized, y) × class_weight[y]
    2) Auxiliary: Σ w(t,N) × Focal CE(p_agg_t_normalized, y)

    Total: L = L_main + λ × L_aux
    """
    B, T, C = turn_probs.shape
    y = labels.long()                                           # [B]

    # ── 1) Main: Focal CE trên p_agg_final (dialogue-level) ──
    n_turns = turn_mask.sum(dim=1).long()                       # [B]
    p_agg_final = torch.stack([
        p_agg[b, n_turns[b] - 1] for b in range(B)
    ])                                                          # [B, C]
    p_norm_final = _normalize_p_agg(p_agg_final)                # [B, C]

    focal_main = _focal_ce(p_norm_final, y, gamma=gamma, eps=eps)  # [B]

    # Class balance tại sample level
    if class_weights is not None:
        cw = class_weights[y]                                   # [B]
    else:
        cw = torch.ones_like(focal_main)
    loss_main = (focal_main * cw).mean()

    # ── 2) Auxiliary: U-shape Weighted Prefix Focal CE ──
    loss_aux = torch.tensor(0.0, device=turn_probs.device)
    if weighted_lambda > 0:
        # Normalize p_agg at every prefix
        p_norm_all = _normalize_p_agg(p_agg)                    # [B, T, C]

        y_exp = y.unsqueeze(1).expand(B, T)                     # [B, T]
        focal_prefix = _focal_ce(p_norm_all, y_exp, gamma=gamma, eps=eps)  # [B, T]
        focal_prefix = focal_prefix * turn_mask.float()         # zero padding

        # U-shape temporal weight: w(t,N) = (2t/N - 1)² × (1 - w_floor) + w_floor
        n = n_turns.float().clamp(min=1)                        # [B]
        t_idx = torch.arange(T, dtype=torch.float32,
                             device=turn_probs.device)          # [T]
        t_norm = t_idx.unsqueeze(0) / n.unsqueeze(1)            # [B, T]
        weights = (2 * t_norm - 1) ** 2 * (1 - w_floor) + w_floor  # [B, T]
        weights = weights * turn_mask.float()                   # zero padding

        # Weighted avg (normalize bởi sum weights)
        loss_per_sample = (weights * focal_prefix).sum(dim=1) / weights.sum(dim=1).clamp(min=1e-8)
        loss_aux = loss_per_sample.mean()

    return loss_main + weighted_lambda * loss_aux


class CrossTurnAttention(nn.Module):
    """Causal attention: turn_t attends to h_0..h_{t-1}."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=n_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm = nn.LayerNorm(d_model)
        self.proj = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, h_seq: torch.Tensor, turn_mask: torch.Tensor) -> torch.Tensor:
        """
        h_seq    : [B, T, D]
        turn_mask: [B, T] bool — True = real turn
        Returns  : [B, T, D]
        """
        B, T, D = h_seq.shape
        out = torch.zeros_like(h_seq)
        out[:, 0, :] = h_seq[:, 0, :]
        for t in range(1, T):
            query  = h_seq[:, t:t+1, :]      # [B, 1, D]
            keys   = h_seq[:, :t, :]          # [B, t, D]
            key_pm = ~turn_mask[:, :t]        # True = ignore (padding)
            ctx, _ = self.mha(query, keys, keys, key_padding_mask=key_pm)
            fused  = self.proj(torch.cat([h_seq[:, t, :], ctx.squeeze(1)], dim=-1))
            out[:, t, :] = self.norm(fused + h_seq[:, t, :])
        return out


class M1Classifier(nn.Module):
    """HaLong per-turn encoder + causal CrossTurnAttention + Multi-Head Noisy-OR head."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = AutoModel.from_pretrained(cfg.model_name)
        self._encoder_frozen = True
        for p in self.encoder.parameters():
            p.requires_grad = False

        embed_dim = self.encoder.config.hidden_size
        d = cfg.hidden_dim
        self.proj = nn.Linear(embed_dim, d)
        self.attn = CrossTurnAttention(d, cfg.attn_heads, cfg.dropout)
        self.head = nn.Sequential(
            nn.Linear(d, d // 2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(d // 2, cfg.num_classes),   # C logits per turn
        )

    def unfreeze_encoder(self):
        """Unfreeze toàn bộ encoder."""
        for p in self.encoder.parameters():
            p.requires_grad = True
        self._encoder_frozen = False
        self._enable_grad_ckpt()

    def unfreeze_last_n_layers(self, n: int):
        """Unfreeze last N transformer layers + pooler, giữ layers trước frozen."""
        # Freeze everything first
        for p in self.encoder.parameters():
            p.requires_grad = False

        # Find transformer layers (works for BERT-like models)
        layers = None
        if hasattr(self.encoder, 'encoder') and hasattr(self.encoder.encoder, 'layer'):
            layers = self.encoder.encoder.layer
        elif hasattr(self.encoder, 'transformer') and hasattr(self.encoder.transformer, 'layer'):
            layers = self.encoder.transformer.layer

        if layers is not None:
            total = len(layers)
            for i in range(max(0, total - n), total):
                for p in layers[i].parameters():
                    p.requires_grad = True
            print(f"  Unfroze encoder layers [{total-n}..{total-1}] (last {n} of {total})")
        else:
            # Fallback: unfreeze toàn bộ
            print("  [WARN] Cannot find layer structure — unfreezing all encoder params")
            for p in self.encoder.parameters():
                p.requires_grad = True

        # Unfreeze pooler if exists
        if hasattr(self.encoder, 'pooler') and self.encoder.pooler is not None:
            for p in self.encoder.pooler.parameters():
                p.requires_grad = True

        self._encoder_frozen = False
        self._enable_grad_ckpt()

    def _enable_grad_ckpt(self):
        if self.cfg.use_grad_ckpt and hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False}
            )
            print("  Gradient checkpointing enabled on encoder.")

    def _encode_turns(self, input_ids: torch.Tensor, attn_masks: torch.Tensor) -> torch.Tensor:
        """
        Encode turns sequentially để tránh OOM từ B*T forward passes cùng lúc.
        input_ids : [B, T, L]
        Returns   : [B, T, embed_dim]
        """
        B, T, L = input_ids.shape
        cls_list = []
        for t in range(T):
            ids_t  = input_ids[:, t, :]
            mask_t = attn_masks[:, t, :]
            if self._encoder_frozen:
                with torch.no_grad():
                    out_t = self.encoder(input_ids=ids_t, attention_mask=mask_t)
            else:
                out_t = self.encoder(input_ids=ids_t, attention_mask=mask_t)
            cls_list.append(out_t.last_hidden_state[:, 0, :])   # [B, E]
            del out_t
        return torch.stack(cls_list, dim=1)   # [B, T, E]

    def forward(self, input_ids: torch.Tensor, attn_masks: torch.Tensor,
                turn_mask: torch.Tensor, labels: torch.Tensor = None):
        """
        input_ids  : [B, T, L]
        attn_masks : [B, T, L]
        turn_mask  : [B, T] bool — True = real turn
        labels     : [B] long — 0..C-1 class label (training only)

        Returns dict:
          loss           : scalar | None
          turn_probs     : [B, T, C]  per-turn evidence q_t^c (softmax)
          p_agg          : [B, T, C]  Noisy-OR cumulative per-class tại mỗi prefix
          dialogue_probs : [B, C]     p_agg tại last real turn (normalized)
          dialogue_preds : [B]        predicted class (argmax)
        """
        turn_mask = turn_mask.bool()
        B, T, L   = input_ids.shape
        C         = self.cfg.num_classes
        n_turns   = turn_mask.sum(dim=1).long()    # [B]

        cls_seq = self._encode_turns(input_ids, attn_masks)   # [B, T, E]
        h_seq   = F.gelu(self.proj(cls_seq))                   # [B, T, D]
        h_ctx   = self.attn(h_seq, turn_mask)                  # [B, T, D]
        logits  = self.head(h_ctx)                             # [B, T, C]

        # Per-turn evidence: softmax over classes
        turn_probs = torch.softmax(logits, dim=-1)             # [B, T, C]

        # ── Noisy-OR cumulative per-class ──
        p_agg = _compute_p_agg_multiclass(turn_probs, turn_mask)  # [B, T, C]

        # dialogue_probs = normalized p_agg tại last real turn
        p_agg_final = torch.stack([
            p_agg[b, n_turns[b] - 1] for b in range(B)
        ])                                                     # [B, C]
        dialogue_probs = _normalize_p_agg(p_agg_final)         # [B, C]
        dialogue_preds = dialogue_probs.argmax(dim=-1)         # [B]

        loss = None
        if labels is not None:
            # Build class_weights tensor
            cw = torch.tensor(self.cfg.class_weights,
                              dtype=torch.float32,
                              device=input_ids.device)
            loss = noisy_or_focal_ce_loss(
                turn_probs, labels, turn_mask, p_agg,
                gamma=self.cfg.focal_gamma,
                class_weights=cw,
                weighted_lambda=self.cfg.weighted_lambda,
                w_floor=self.cfg.w_floor,
            )

        return {
            "loss":           loss,
            "turn_probs":     turn_probs,       # [B, T, C] per-turn evidence
            "p_agg":          p_agg,             # [B, T, C] Noisy-OR cumulative
            "dialogue_probs": dialogue_probs,    # [B, C]    normalized p_agg at last turn
            "dialogue_preds": dialogue_preds,    # [B]       predicted class
        }

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
