"""Dual-Head Neural Reward Model & Constrained Lagrangian Loss Engine.

Implements the neural counterpart to the decoupled clinical dual reward architecture:
1. DualHeadRewardModel:
   - Shared causal transformer backbone (MiniGPT or PEFT/LoRA backbone).
   - Independent linear reward heads with LayerNorm pre-conditioning:
     - Safety Head: r_safety in [-inf, +inf] (projects to sigmoid probability).
     - Helpfulness Head: r_helpful in [-inf, +inf].
   - Differentiable Asymmetric Safety Gating forward pass.
2. ConstrainedDualBTLoss:
   - Joint Bradley-Terry loss optimizing safety and helpfulness preference margins.
   - Non-compensatory safety barrier penalty penalizing chosen responses that breach
     the safety threshold tau_safe.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.m1_transformer.model import MiniGPT


class DualHeadRewardModel(nn.Module):
    """Causal Transformer backbone augmented with decoupled Safety and Helpfulness heads."""

    def __init__(self, backbone: MiniGPT):
        super().__init__()
        self.backbone = backbone
        d_model = backbone.config.d_model

        # LayerNorm pre-conditioning before linear projections
        self.ln_safe = nn.LayerNorm(d_model)
        self.ln_help = nn.LayerNorm(d_model)

        # Independent linear heads for Safety and Helpfulness
        self.safety_head = nn.Linear(d_model, 1, bias=False)
        self.helpful_head = nn.Linear(d_model, 1, bias=False)

        # Small normal weight initialization for training stability
        nn.init.normal_(self.safety_head.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.helpful_head.weight, mean=0.0, std=0.02)

    @property
    def device(self) -> torch.device:
        return self.backbone.device

    def extract_hidden_states(self, idx: torch.Tensor) -> torch.Tensor:
        """Forward pass through transformer layers to obtain final hidden states (B, T, d_model)."""
        idx = idx.to(self.device)
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=self.device)

        tok_emb = self.backbone.transformer.wte(idx)
        pos_emb = self.backbone.transformer.wpe(pos)
        x = self.backbone.transformer.drop(tok_emb + pos_emb)

        for block in self.backbone.transformer.h:
            x, _ = block(x)

        x = self.backbone.transformer.ln_f(x)
        return x

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes decoupled (r_safety, r_helpful) scalar tensors for input batch.

        Args:
            input_ids: Token ID tensor of shape (B, T).
            attention_mask: Optional binary mask of shape (B, T).

        Returns:
            Tuple of (safety_rewards, helpful_rewards), each of shape (B,).
        """
        input_ids = input_ids.to(self.device)
        B, T = input_ids.size()
        hidden = self.extract_hidden_states(input_ids)  # Shape: (B, T, d_model)

        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
            indices = torch.arange(T, device=self.device).unsqueeze(0).expand(B, -1)
            masked_indices = torch.where(attention_mask.bool(), indices, torch.full_like(indices, -1))
            last_token_idx = masked_indices.max(dim=-1).values.clamp(min=0)  # Shape: (B,)
            batch_idx = torch.arange(B, device=self.device)
            pooled_hidden = hidden[batch_idx, last_token_idx]  # Shape: (B, d_model)
        else:
            pooled_hidden = hidden[:, -1, :]  # Shape: (B, d_model)

        # Compute decoupled rewards
        h_safe = self.ln_safe(pooled_hidden)
        h_help = self.ln_help(pooled_hidden)

        r_safety = self.safety_head(h_safe).squeeze(-1)   # Shape: (B,)
        r_helpful = self.helpful_head(h_help).squeeze(-1) # Shape: (B,)

        return r_safety, r_helpful

    def forward_composite(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        tau_safe: float = 0.40,
        alpha: float = 0.50,
        w_s: float = 0.60,
        w_h: float = 0.40,
    ) -> torch.Tensor:
        """Computes gated composite scalar reward using the non-compensatory safety gate."""
        r_safe, r_help = self.forward(input_ids, attention_mask)
        # Normalize to [0, 1] via sigmoid for gating calculation
        s_norm = torch.sigmoid(r_safe)
        h_norm = torch.sigmoid(r_help)

        # Differentiable soft-gating or piece-wise gating:
        # If s_norm < tau_safe: alpha * s_norm, else: w_s * s_norm + w_h * h_norm
        mask_safe = (s_norm >= tau_safe).float()
        composite = (1.0 - mask_safe) * (alpha * s_norm) + mask_safe * (w_s * s_norm + w_h * h_norm)
        return composite


class ConstrainedDualBTLoss(nn.Module):
    """Numerically stable Constrained Dual Bradley-Terry loss with Lagrangian safety penalty."""

    def __init__(
        self,
        margin_safe: float = 0.20,
        margin_help: float = 0.10,
        weight_safe: float = 0.60,
        weight_help: float = 0.40,
        tau_safe: float = 0.40,
        barrier_weight: float = 1.0,
        l2_reg: float = 0.001,
    ):
        super().__init__()
        self.margin_safe = margin_safe
        self.margin_help = margin_help
        self.weight_safe = weight_safe
        self.weight_help = weight_help
        self.tau_safe = tau_safe
        self.barrier_weight = barrier_weight
        self.l2_reg = l2_reg

    def forward(
        self,
        r_safe_w: torch.Tensor,
        r_safe_l: torch.Tensor,
        r_help_w: torch.Tensor,
        r_help_l: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Calculates combined dual Bradley-Terry loss across safety and helpfulness.

        Returns dictionary of loss components:
            {"total_loss", "loss_safe", "loss_help", "loss_barrier", "reg_loss"}
        """
        # 1. Safety Pairwise Loss: r_safe_w should exceed r_safe_l by margin_safe
        diff_safe = r_safe_w - r_safe_l - self.margin_safe
        loss_safe = F.softplus(-diff_safe).mean()

        # 2. Helpfulness Pairwise Loss: r_help_w should exceed r_help_l by margin_help
        diff_help = r_help_w - r_help_l - self.margin_help
        loss_help = F.softplus(-diff_help).mean()

        # 3. Non-Compensatory Safety Barrier Penalty:
        # Penalizes chosen candidates if their normalized safety falls below tau_safe
        s_norm_w = torch.sigmoid(r_safe_w)
        safety_deficit = F.relu(self.tau_safe - s_norm_w)
        loss_barrier = (safety_deficit ** 2).mean() * self.barrier_weight

        # 4. L2 Regularization on raw reward magnitudes
        reg_loss = 0.5 * self.l2_reg * (
            r_safe_w.pow(2).mean() + r_safe_l.pow(2).mean() +
            r_help_w.pow(2).mean() + r_help_l.pow(2).mean()
        )

        total_loss = (self.weight_safe * loss_safe) + (self.weight_help * loss_help) + loss_barrier + reg_loss

        return {
            "total_loss": total_loss,
            "loss_safe": loss_safe,
            "loss_help": loss_help,
            "loss_barrier": loss_barrier,
            "reg_loss": reg_loss,
        }
