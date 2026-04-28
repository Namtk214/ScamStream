"""
M1 Classifier — HaLong Embedding + Causal CrossTurnAttention
              + Noisy-OR Focal Loss (adapted from Streaming-Bert).

Architecture:
  turn text → HaLong encoder ([CLS] token) → e_t [embed_dim]
            → Linear projection → h_t [hidden_dim]
            → CrossTurnAttention (causal: turn_t attends to h_0..h_{t-1})
            → MLP head → sigmoid → q_t (per-turn evidence)
            → Noisy-OR aggregation → p_agg (cumulative, monotonically non-decreasing)

Training Loss:
  L_total = L_main + λ × L_aux

  1) Main: FocalBCE(p_agg_final, y) × class_weight
     p_agg_final = Noisy-OR tại turn cuối (dialogue-level prediction)
     Focal (1-p)^γ focus hard examples, class_weight cân bằng imbalance

  2) Auxiliary: Σ (2t/N) × FocalBCE(p_t_agg, y)
     Weighted prefix trên cumulative Noisy-OR tại mỗi prefix
     Thúc đẩy model predict đúng sớm dần

Streaming inference:
  Online Noisy-OR: p_agg_t = 1 − (1 − p_agg_{t-1}) × (1 − q_t)
  O(1) per turn, monotonically non-decreasing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


# ─────────────────────────────────────────────────────────────────
# Noisy-OR utilities
# ─────────────────────────────────────────────────────────────────

def _compute_p_agg(turn_probs: torch.Tensor, turn_mask: torch.Tensor,
                   eps: float = 1e-6) -> torch.Tensor:
    """
    Tính Noisy-OR cumulative p_t_agg cho toàn batch.

    Input:
      turn_probs : [B, T]  — per-turn evidence q_t (sigmoid)
      turn_mask  : [B, T]  — True = real turn
    Output:
      p_agg      : [B, T]  — cumulative Noisy-OR, padding giữ nguyên
    """
    q = turn_probs.clamp(eps, 1 - eps)
    log_not_q = torch.log1p(-q) * turn_mask.float()    # zero padding trước cumsum
    cumsum_log = torch.cumsum(log_not_q, dim=1)
    p_agg = (1.0 - torch.exp(cumsum_log)).clamp(eps, 1 - eps)
    return p_agg


def _focal_bce(p: torch.Tensor, y: torch.Tensor,
               gamma: float = 2.0, eps: float = 1e-6) -> torch.Tensor:
    """
    Focal Binary Cross-Entropy (element-wise, no reduction).

    FocalBCE = -(1-p_t)^γ × log(p_t)  ×  (γ+1)
    với p_t = xác suất của class đúng.
    Nhân (γ+1) để normalize magnitude ~ BCE.
    """
    p = p.clamp(eps, 1 - eps)
    p_t = torch.where(y.bool(), p, 1 - p)   # prob of correct class
    focal_mod = (1 - p_t) ** gamma
    bce = -torch.log(p_t)
    return focal_mod * bce * (gamma + 1)


# ─────────────────────────────────────────────────────────────────
# Loss functions
# ─────────────────────────────────────────────────────────────────

def noisy_or_focal_loss(turn_probs: torch.Tensor, labels: torch.Tensor,
                        turn_mask: torch.Tensor, p_agg: torch.Tensor,
                        gamma: float = 2.0,
                        class_weight_harmless: float = 8.0,
                        weighted_lambda: float = 0.5,
                        w_floor: float = 0.1,
                        eps: float = 1e-6) -> torch.Tensor:
    """
    Noisy-OR Focal Loss (adapted from Streaming-Bert).

    1) Main: FocalBCE(p_agg_final, y) × class_weight
       → Focal modulation focus hard examples
       → class_weight cân bằng scam/harmless imbalance

    2) Auxiliary: Σ w(t,N) × FocalBCE(p_t_agg, y)
       w(t,N) = (2t/N - 1)² × (1 - w_floor) + w_floor
       → U-shape: lớn ở đầu/cuối dialogue, đáy ở giữa
       → Phạt cả early miss lẫn late miss

    Total: L = L_main + λ × L_aux
    """
    B, T = turn_probs.shape
    y = labels.float()                                         # [B]

    # ── 1) Main: FocalBCE trên p_agg_final (dialogue-level) ──
    n_turns = turn_mask.sum(dim=1).long()                      # [B]
    p_dialogue = torch.stack([
        p_agg[b, n_turns[b] - 1] for b in range(B)
    ])                                                         # [B]

    focal_main = _focal_bce(p_dialogue, y, gamma=gamma, eps=eps)  # [B]

    # Class balance tại sample level
    cw = torch.where(labels == 0,
                     torch.full_like(focal_main, class_weight_harmless),
                     torch.ones_like(focal_main))
    loss_main = (focal_main * cw).mean()

    # ── 2) Auxiliary: U-shape Weighted Prefix FocalBCE ──
    loss_aux = torch.tensor(0.0, device=turn_probs.device)
    if weighted_lambda > 0:
        y_exp = y.unsqueeze(1).expand(B, T)                    # [B, T]
        focal_prefix = _focal_bce(p_agg, y_exp, gamma=gamma, eps=eps)  # [B, T]
        focal_prefix = focal_prefix * turn_mask.float()        # zero padding

        # U-shape temporal weight: w(t,N) = (2t/N - 1)² × (1 - w_floor) + w_floor
        # Lớn ở hai đầu (turn 0 và turn N), đáy ở giữa (turn N/2)
        n = n_turns.float().clamp(min=1)                       # [B]
        t_idx = torch.arange(T, dtype=torch.float32,
                             device=turn_probs.device)         # [T], 0-based
        t_norm = t_idx.unsqueeze(0) / n.unsqueeze(1)           # [B, T]
        weights = (2 * t_norm - 1) ** 2 * (1 - w_floor) + w_floor  # [B, T]
        weights = weights * turn_mask.float()                  # zero padding

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
    """HaLong per-turn encoder + causal CrossTurnAttention + Noisy-OR Focal head."""

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
            nn.Linear(d // 2, 1),   # sigmoid output: q_t evidence
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
        labels     : [B] long — 0/1 dialogue label (training only)

        Returns dict:
          loss           : scalar | None
          turn_probs     : [B, T]  per-turn evidence q_t
          p_agg          : [B, T]  Noisy-OR cumulative tại mỗi prefix
          dialogue_probs : [B]     p_agg tại last real turn (= dialogue prediction)
        """
        turn_mask = turn_mask.bool()
        B, T, L   = input_ids.shape
        n_turns   = turn_mask.sum(dim=1).long()    # [B]

        cls_seq = self._encode_turns(input_ids, attn_masks)   # [B, T, E]
        h_seq   = F.gelu(self.proj(cls_seq))                   # [B, T, D]
        h_ctx   = self.attn(h_seq, turn_mask)                  # [B, T, D]
        logits  = self.head(h_ctx).squeeze(-1)                 # [B, T]

        turn_probs = torch.sigmoid(logits)                     # [B, T]  evidence q_t

        # ── Noisy-OR cumulative ──
        p_agg = _compute_p_agg(turn_probs, turn_mask)          # [B, T]

        # dialogue_probs = p_agg tại last real turn
        dialogue_probs = torch.stack([
            p_agg[b, n_turns[b] - 1] for b in range(B)
        ])                                                     # [B]

        loss = None
        if labels is not None:
            loss = noisy_or_focal_loss(
                turn_probs, labels, turn_mask, p_agg,
                gamma=self.cfg.focal_gamma,
                class_weight_harmless=self.cfg.class_weight_harmless,
                weighted_lambda=self.cfg.weighted_lambda,
                w_floor=self.cfg.w_floor,
            )

        return {
            "loss":           loss,
            "turn_probs":     turn_probs,       # per-turn evidence q_t
            "p_agg":          p_agg,             # Noisy-OR cumulative
            "dialogue_probs": dialogue_probs,    # p_agg at last turn
        }

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
