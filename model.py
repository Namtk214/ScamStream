"""
M1 Classifier — HaLong Embedding + Causal CrossTurnAttention + U-shape Loss.

Architecture:
  turn text → HaLong encoder ([CLS] token) → e_t [embed_dim]
            → Linear projection → h_t [hidden_dim]
            → CrossTurnAttention (causal: turn_t attends to h_0..h_{t-1})
            → MLP head → sigmoid → turn_probs [B, T]

Training:
  U-shape temporal weight + class balance:
    w(t, N) = (2t/N - 1)^2 * (1 - w_floor) + w_floor
  Lớn ở hai đầu (turn 0 và turn N), đáy ở giữa (turn N/2).
  Phạt cả early miss lẫn late miss.

Streaming inference:
  Duy trì buffer turns đã encode; mỗi turn mới chạy full forward
  trên prefix, lấy p_t tại vị trí cuối cùng.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


def focal_u_shape_loss(turn_probs: torch.Tensor, labels: torch.Tensor,
                       turn_mask: torch.Tensor,
                       w_floor: float = 0.1,
                       gamma: float = 2.0,
                       class_weight_harmless: float = 4.25) -> torch.Tensor:
    """
    Focal loss × U-shape temporal weight.

    Mỗi cơ chế xử lý một vấn đề riêng:
      - Focal (1-p_t)^γ : down-weight easy examples, focus on hard boundary cases
      - U-shape w(t)    : điều chỉnh tầm quan trọng theo vị trí turn trong dialogue
      - class_weight    : cân bằng imbalance scam/harmless ở sample level

    Loss magnitude được chuẩn hóa về cùng bậc với BCE bằng cách chia
    cho normalization constant E[(1-p)^γ] ≈ 1/(γ+1) tại p~Uniform.
    """
    B, T = turn_probs.shape
    p   = turn_probs.clamp(1e-6, 1 - 1e-6)
    y   = labels.float().unsqueeze(1).expand(B, T)   # [B, T]

    # p_t = xác suất của class đúng tại mỗi turn
    p_t = torch.where(y.bool(), p, 1 - p)            # [B, T]

    # Focal modulator — KHÔNG có alpha để tránh double-scale
    focal_mod = (1 - p_t) ** gamma                   # [B, T]

    # BCE per turn
    bce = -torch.log(p_t)                            # [B, T]

    # Chuẩn hóa focal về cùng magnitude với BCE: nhân (γ+1)
    focal_normalized = focal_mod * bce * (gamma + 1) # [B, T]

    # U-shape temporal weight: w(t,N) = (2t/N - 1)^2 * (1-w_floor) + w_floor
    # Lớn ở hai đầu (turn 0 và turn N), đáy ở giữa (turn N/2)
    n      = turn_mask.sum(dim=1).clamp(min=1).float()         # [B]
    t_idx  = torch.arange(T, device=p.device).float()          # [T]
    t_norm = t_idx.unsqueeze(0) / n.unsqueeze(1)               # [B, T]
    w_u    = (2 * t_norm - 1) ** 2 * (1 - w_floor) + w_floor  # [B, T]
    w_u    = w_u * turn_mask.float()                            # zero padding

    # Sample-level loss (weighted avg over turns)
    loss_per_sample = (focal_normalized * w_u).sum(dim=1) / w_u.sum(dim=1).clamp(min=1e-8)

    # Class balance tại sample level
    cw = torch.where(labels == 0,
                     torch.full_like(loss_per_sample, class_weight_harmless),
                     torch.ones_like(loss_per_sample))
    loss_per_sample = loss_per_sample * cw

    return loss_per_sample.mean()


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
    """HaLong per-turn encoder + causal CrossTurnAttention + weighted CE head."""

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
            nn.Linear(d // 2, 1),   # sigmoid output: P(scam)
        )

    def unfreeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = True
        self._encoder_frozen = False
        if self.cfg.use_grad_ckpt and hasattr(self.encoder, 'gradient_checkpointing_enable'):
            self.encoder.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={'use_reentrant': False}
            )
            print("Gradient checkpointing enabled on encoder.")

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
          turn_probs     : [B, T]  P(SCAM) per turn
          dialogue_probs : [B]     P(SCAM) tại last real turn (để evaluate)
        """
        turn_mask = turn_mask.bool()
        B, T, L   = input_ids.shape
        n_turns   = turn_mask.sum(dim=1).long()    # [B]

        cls_seq = self._encode_turns(input_ids, attn_masks)   # [B, T, E]
        h_seq   = F.gelu(self.proj(cls_seq))                   # [B, T, D]
        h_ctx   = self.attn(h_seq, turn_mask)                  # [B, T, D]
        logits  = self.head(h_ctx).squeeze(-1)                 # [B, T]

        turn_probs     = torch.sigmoid(logits)                 # [B, T]
        dialogue_probs = torch.stack([
            turn_probs[b, n_turns[b] - 1] for b in range(B)
        ])                                                     # [B]

        loss = None
        if labels is not None:
            loss = focal_u_shape_loss(turn_probs, labels, turn_mask,
                                      w_floor=self.cfg.w_floor,
                                      gamma=self.cfg.focal_gamma,
                                      class_weight_harmless=self.cfg.class_weight_harmless)

        return {
            "loss":           loss,
            "turn_probs":     turn_probs,
            "dialogue_probs": dialogue_probs,
        }

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
