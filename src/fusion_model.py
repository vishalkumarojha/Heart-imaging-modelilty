"""Phase 4 — the fusion layer.

FusionLayer: up to 3 frozen-encoder embeddings ([B,1024] each) + a presence mask
-> one shared fused representation [B, out_dim].

  * Each present embedding is per-modality LayerNorm'd first — X-ray embeddings
    are ~15x larger in raw magnitude than the echo/MRI ones (DenseNet ReLU+pool
    vs LSTM tanh range); without this, X-ray dominates by scale alone.
  * Each ABSENT modality is replaced by a LEARNED missing-modality token
    (a trainable [1024] parameter per modality) — NOT a zero vector.
  * Combination = concat(3 x [B,1024]) -> MLP.  Concat+MLP (not attention)
    because with only 3 fixed modality slots there is nothing for attention to
    route over that a 2-layer MLP can't learn directly; it keeps the parameter
    count and failure modes minimal, which matters given the pathway is only
    ever validated single-modality-present.

TaskHeads: three independent linear heads on the SHARED fused rep — X-ray
(2 logits, multi-label), echo (3 logits), MRI (5 logits). Real label spaces,
nothing unified/invented.

>>> There is NO real tri-modal patient data. This model is trained & validated
>>> single-modality-present only. Multi-modal use is SYNTHETIC (see fusion_demo).
"""
from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from . import fusion_config as FC


class FusionLayer(nn.Module):
    def __init__(self, embed_dim: int = FC.EMBED_DIM, n_modalities: int = 3,
                 hidden: int = FC.FUSION_HIDDEN, out_dim: int = FC.FUSION_OUT_DIM,
                 dropout: float = FC.FUSION_DROPOUT) -> None:
        super().__init__()
        self.n = n_modalities
        self.out_dim = out_dim
        self.norms = nn.ModuleList([nn.LayerNorm(embed_dim) for _ in range(n_modalities)])
        # learned missing-modality tokens (one per modality), small init
        self.missing_token = nn.ParameterList(
            [nn.Parameter(torch.randn(embed_dim) * 0.02) for _ in range(n_modalities)]
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * n_modalities, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
            nn.GELU(),
        )

    def forward(self, embeds: List[Optional[torch.Tensor]], mask: torch.Tensor) -> torch.Tensor:
        """embeds: list of length n; entry i is [B,1024] or None.
        mask: [B, n] (1 = modality present, 0 = missing -> use learned token)."""
        bsz = mask.shape[0]
        parts = []
        for i in range(self.n):
            token = self.missing_token[i].unsqueeze(0).expand(bsz, -1)
            e = embeds[i]
            if e is None:
                parts.append(token)
                continue
            e = self.norms[i](e)
            m = mask[:, i:i + 1].to(e.dtype)
            parts.append(m * e + (1.0 - m) * token)
        return self.mlp(torch.cat(parts, dim=1))          # [B, out_dim]


class TaskHeads(nn.Module):
    """One linear head per modality's ORIGINAL label space (logits, no activation)."""

    def __init__(self, in_dim: int) -> None:
        super().__init__()
        self.heads = nn.ModuleDict(
            {name: nn.Linear(in_dim, spec["n_out"]) for name, spec in FC.TASK_SPECS.items()}
        )

    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        return {name: head(z) for name, head in self.heads.items()}


class FusionModel(nn.Module):
    """FusionLayer + TaskHeads. Encoders are NOT part of this module — they are
    frozen and produce the [B,1024] embeddings that get fed in."""

    def __init__(self) -> None:
        super().__init__()
        self.fusion = FusionLayer()
        self.heads = TaskHeads(self.fusion.out_dim)

    def forward(self, embeds: List[Optional[torch.Tensor]], mask: torch.Tensor):
        z = self.fusion(embeds, mask)
        return self.heads(z), z

    @staticmethod
    def mask_for(present: List[str], batch: int, device) -> torch.Tensor:
        """Build a [batch, 3] presence mask from a list of present modality names."""
        idx = {m: i for i, m in enumerate(FC.MODALITIES)}
        row = torch.zeros(len(FC.MODALITIES))
        for m in present:
            row[idx[m]] = 1.0
        return row.unsqueeze(0).expand(batch, -1).to(device)


def count_parameters(model: nn.Module):
    trn = sum(p.numel() for p in model.parameters() if p.requires_grad)
    tot = sum(p.numel() for p in model.parameters())
    return trn, tot
