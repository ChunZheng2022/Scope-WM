import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .vit import ViTPredictor


def tile_condition(condition: torch.Tensor, num_tokens: int) -> torch.Tensor:
    return repeat(condition.unsqueeze(2), "b t 1 d -> b t n d", n=num_tokens)


def compose_patch_tokens(
    visual_history: torch.Tensor,
    proprio_history: Optional[torch.Tensor],
    action_history: Optional[torch.Tensor],
) -> torch.Tensor:
    parts = [visual_history]
    num_tokens = int(visual_history.shape[-2])
    if proprio_history is not None and proprio_history.shape[-1] > 0:
        parts.append(tile_condition(proprio_history, num_tokens))
    if action_history is not None and action_history.shape[-1] > 0:
        parts.append(tile_condition(action_history, num_tokens))
    return torch.cat(parts, dim=-1)


def deterministic_force_min_mask(
    mask: torch.Tensor,
    scores: torch.Tensor,
    k_min: int,
) -> torch.Tensor:
    if k_min <= 0:
        return mask.bool()
    num_tokens = int(mask.shape[-1])
    k_min = min(int(k_min), num_tokens)
    current_k = mask.sum(dim=-1)
    if bool((current_k >= k_min).all()):
        return mask.bool()

    forced = mask.bool().clone()
    need_fill = current_k < k_min
    topk_idx = torch.topk(scores, k=k_min, dim=-1).indices
    fill_mask = torch.zeros_like(mask, dtype=torch.bool)
    fill_mask.scatter_(-1, topk_idx, True)
    forced[need_fill] = torch.logical_or(forced[need_fill], fill_mask[need_fill])
    return forced


def ddpwm_pixel_change_scores_and_mask(
    current: torch.Tensor,
    next_frame: torch.Tensor,
    grid_h: int = 14,
    grid_w: int = 14,
    threshold: float = 0.1,
    partition_precision: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DDP-WM LabelGenerator-style pixel-change target.

    Returns patch scores with shape [B, grid_h * grid_w] and a boolean foreground mask.
    The DDP code computes a finer partition per patch and marks a patch foreground if
    any sub-partition exceeds the threshold.
    """
    sqrt_precision = int(math.sqrt(partition_precision))
    if sqrt_precision * sqrt_precision != int(partition_precision):
        raise ValueError(f"partition_precision must be a square number, got {partition_precision}")

    if current.max() > 2:
        current = current / 255.0
        next_frame = next_frame / 255.0

    pixel_diff = next_frame - current
    pixel_norms_sq = torch.sum(pixel_diff.pow(2), dim=1, keepdim=True)
    patch_size_h = max(int(current.shape[2] // grid_h // sqrt_precision), 1)
    patch_size_w = max(int(current.shape[3] // grid_w // sqrt_precision), 1)
    patch_norms_sq = F.avg_pool2d(
        pixel_norms_sq,
        kernel_size=(patch_size_h, patch_size_w),
        stride=(patch_size_h, patch_size_w),
    )
    norms = torch.sqrt(patch_norms_sq.clamp_min(0.0))
    fine_h = grid_h * sqrt_precision
    fine_w = grid_w * sqrt_precision
    if norms.shape[-2:] != (fine_h, fine_w):
        norms = F.interpolate(norms, size=(fine_h, fine_w), mode="nearest")

    bsz = int(norms.shape[0])
    norms = norms[:, 0].reshape(bsz, grid_h, sqrt_precision, grid_w, sqrt_precision)
    norms = norms.permute(0, 1, 3, 2, 4).reshape(
        bsz, grid_h * grid_w, partition_precision
    )
    scores = norms.max(dim=-1).values
    mask = (norms > float(threshold)).any(dim=-1)
    return scores, mask


class HistoricalInformationFusion(nn.Module):
    def __init__(
        self,
        token_dim: int,
        num_hist: int,
        num_tokens: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.token_dim = int(token_dim)
        self.num_hist = int(num_hist)
        self.num_tokens = int(num_tokens)
        self.query_pos = nn.Parameter(torch.randn(num_tokens, token_dim) * 0.02)
        self.memory_pos = nn.Parameter(torch.randn(num_tokens, token_dim) * 0.02)
        self.time_pos = nn.Parameter(torch.zeros(max(num_hist - 1, 1), token_dim))
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=token_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False,
        )
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(token_dim)

    def forward(self, token_history: torch.Tensor) -> torch.Tensor:
        bsz, hist_len, num_tokens, token_dim = token_history.shape
        if hist_len <= 1:
            return token_history[:, -1:]
        if num_tokens != self.num_tokens or token_dim != self.token_dim:
            raise ValueError(
                "History fusion got unexpected token shape: "
                f"{token_history.shape}, expected N={self.num_tokens}, D={self.token_dim}"
            )

        current = token_history[:, -1].permute(1, 0, 2)
        query_pos = self.query_pos[:, None, :].expand(-1, bsz, -1)
        history = token_history[:, :-1]
        mem_len = int(history.shape[1])
        memory = rearrange(history, "b t n d -> (t n) b d")

        spatial_pos = self.memory_pos[None, :, :].expand(mem_len, -1, -1)
        time_pos = self.time_pos[:mem_len, None, :]
        mem_pos = rearrange(spatial_pos + time_pos, "t n d -> (t n) d")
        mem_pos = mem_pos[:, None, :].expand(-1, bsz, -1)

        attn_out, _ = self.cross_attn(
            query=current + query_pos,
            key=memory + mem_pos,
            value=memory,
            need_weights=False,
        )
        fused = self.norm(current + self.dropout(attn_out))
        return fused.permute(1, 0, 2).unsqueeze(1)


class HistoryFusionDynamicLocalizer(nn.Module):
    def __init__(
        self,
        visual_dim: int,
        action_dim: int,
        proprio_dim: int,
        num_hist: int,
        num_tokens: int,
        reduced_dim: int = 192,
        history_heads: int = 4,
        localizer_layers: int = 3,
        localizer_heads: int = 4,
        mlp_dim: int = 768,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.visual_dim = int(visual_dim)
        self.action_dim = int(action_dim)
        self.proprio_dim = int(proprio_dim)
        self.num_hist = int(num_hist)
        self.num_tokens = int(num_tokens)
        self.reduced_dim = int(reduced_dim)
        full_dim = self.visual_dim + self.proprio_dim + self.action_dim
        self.history_fusion = HistoricalInformationFusion(
            token_dim=full_dim,
            num_hist=num_hist,
            num_tokens=num_tokens,
            num_heads=history_heads,
            dropout=dropout,
        )
        localizer_dim = self.reduced_dim + self.proprio_dim + self.action_dim
        self.visual_reduction = nn.Linear(self.visual_dim, self.reduced_dim)
        self.localizer_vit = ViTPredictor(
            dim=localizer_dim,
            depth=localizer_layers,
            heads=localizer_heads,
            mlp_dim=mlp_dim,
            num_frames=1,
            num_patches=num_tokens,
            dropout=dropout,
            emb_dropout=dropout,
        )
        self.cls_head = nn.Sequential(
            nn.LayerNorm(localizer_dim),
            nn.GELU(),
            nn.Linear(localizer_dim, localizer_dim),
            nn.GELU(),
            nn.Linear(localizer_dim, 1),
        )

    def forward(
        self,
        visual_history: torch.Tensor,
        action_history: Optional[torch.Tensor] = None,
        proprio_history: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        full_history = compose_patch_tokens(
            visual_history=visual_history,
            proprio_history=proprio_history,
            action_history=action_history,
        )
        fused = self.history_fusion(full_history)
        fused = fused.squeeze(1)

        visual = fused[..., : self.visual_dim]
        offset = self.visual_dim
        if self.proprio_dim > 0:
            proprio = fused[..., offset : offset + self.proprio_dim]
            offset += self.proprio_dim
        else:
            proprio = fused.new_zeros(*fused.shape[:-1], 0)
        if self.action_dim > 0:
            action = fused[..., offset : offset + self.action_dim]
        else:
            action = fused.new_zeros(*fused.shape[:-1], 0)

        local_tokens = torch.cat([self.visual_reduction(visual), proprio, action], dim=-1)
        local_tokens = self.localizer_vit(local_tokens)
        return self.cls_head(local_tokens).squeeze(-1)

    def predict_scores_and_mask(
        self,
        visual_history: torch.Tensor,
        action_history: Optional[torch.Tensor] = None,
        proprio_history: Optional[torch.Tensor] = None,
        threshold: float = 0.5,
        min_tokens: int = 32,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self(
            visual_history=visual_history,
            action_history=action_history,
            proprio_history=proprio_history,
        )
        scores = torch.sigmoid(logits)
        mask = scores > float(threshold)
        mask = deterministic_force_min_mask(mask, scores, min_tokens)
        return scores, mask


def build_dynamic_localizer_from_metadata(metadata: Dict) -> HistoryFusionDynamicLocalizer:
    return HistoryFusionDynamicLocalizer(
        visual_dim=int(metadata["visual_dim"]),
        action_dim=int(metadata.get("action_dim", 0)),
        proprio_dim=int(metadata.get("proprio_dim", 0)),
        num_hist=int(metadata["num_hist"]),
        num_tokens=int(metadata["num_tokens"]),
        reduced_dim=int(metadata.get("reduced_dim", 192)),
        history_heads=int(metadata.get("history_heads", 4)),
        localizer_layers=int(metadata.get("localizer_layers", 3)),
        localizer_heads=int(metadata.get("localizer_heads", 4)),
        mlp_dim=int(metadata.get("mlp_dim", 768)),
        dropout=float(metadata.get("dropout", 0.1)),
    )


def load_dynamic_localizer_checkpoint(
    checkpoint_path: str,
    map_location: Optional[torch.device] = None,
) -> Tuple[HistoryFusionDynamicLocalizer, Dict]:
    ckpt = torch.load(checkpoint_path, map_location=map_location or "cpu")
    metadata = ckpt.get("metadata", {})
    localizer = build_dynamic_localizer_from_metadata(metadata)
    state = ckpt.get("dynamic_localizer", ckpt.get("state_dict", ckpt))
    localizer.load_state_dict(state)
    return localizer, metadata
