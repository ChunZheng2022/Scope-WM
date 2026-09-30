import math
from contextlib import contextmanager
from typing import Dict, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from eval_logging import elapsed_since, perf_counter


class GroupedTokenProcessor(nn.Module):
    """Run the predictor on ROI tokens plus grouped background tokens.

    The module is intentionally parameter-free. It keeps grouped-token mechanics
    out of VWorldModel while preserving a full-grid output shape for the
    existing losses, rollout code, and planning objective.
    """

    def __init__(self, group_size: int = 2):
        super().__init__()
        self.group_size = max(1, int(group_size))
        self.reset_timing_stats()

    def reset_timing_stats(self):
        self._last_timing = {}

    def get_last_timing(self):
        return dict(getattr(self, "_last_timing", {}))

    def _add_timing(self, name, elapsed):
        self._last_timing[name] = self._last_timing.get(name, 0.0) + float(elapsed)

    @contextmanager
    def _time_section(self, name):
        start = perf_counter()
        try:
            yield
        finally:
            self._add_timing(name, elapsed_since(start))

    def _spatial_group_ids(self, num_tokens: int, device) -> Tuple[torch.Tensor, int]:
        if self.group_size <= 1:
            group_ids = torch.arange(num_tokens, device=device)
            return group_ids, num_tokens

        side = int(math.sqrt(num_tokens))
        if side * side == num_tokens:
            rows = torch.arange(side, device=device)[:, None].expand(side, side)
            cols = torch.arange(side, device=device)[None, :].expand(side, side)
            group_side = int(math.ceil(side / self.group_size))
            group_ids = (rows // self.group_size) * group_side + (cols // self.group_size)
            group_ids = group_ids.reshape(-1).long()
        else:
            group_ids = torch.arange(num_tokens, device=device) // self.group_size
        return group_ids, int(group_ids.max().item()) + 1

    @staticmethod
    def _selected_token_indices(mask: torch.Tensor) -> torch.Tensor:
        num_tokens = mask.shape[-1]
        keep_k = int(mask.sum(dim=-1).min().item())
        if keep_k <= 0:
            return torch.empty(*mask.shape[:-1], 0, dtype=torch.long, device=mask.device)
        base_idx = torch.arange(num_tokens, device=mask.device)
        view_shape = (1,) * (mask.ndim - 1) + (num_tokens,)
        base_idx = base_idx.view(view_shape).expand(mask.shape)
        fallback = torch.full_like(base_idx, num_tokens)
        selected = torch.where(mask, base_idx, fallback)
        return selected.sort(dim=-1).values[..., :keep_k]

    @staticmethod
    def _gather_patch_tokens(tokens: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        if idx.shape[-1] == 0:
            return tokens[..., :0, :]
        gather_idx = idx.unsqueeze(-1).expand(*idx.shape, tokens.shape[-1])
        return torch.gather(tokens, dim=2, index=gather_idx)

    def _group_z_tokens(self, z: torch.Tensor, roi_mask: torch.Tensor):
        b, t, num_tokens, dim = z.shape
        roi_idx = self._selected_token_indices(roi_mask)
        roi_tokens = self._gather_patch_tokens(z, roi_idx)
        keep_k = roi_idx.shape[-1]

        group_ids, num_groups = self._spatial_group_ids(num_tokens, z.device)
        flat_z = z.reshape(b * t, num_tokens, dim)
        flat_bg = (~roi_mask).reshape(b * t, num_tokens).to(dtype=z.dtype)
        scatter_idx = group_ids.view(1, num_tokens).expand(b * t, num_tokens)

        bg_sums = torch.zeros(b * t, num_groups, dim, device=z.device, dtype=z.dtype)
        bg_sums.scatter_add_(
            1,
            scatter_idx.unsqueeze(-1).expand(-1, -1, dim),
            flat_z * flat_bg.unsqueeze(-1),
        )
        bg_counts = torch.zeros(b * t, num_groups, 1, device=z.device, dtype=z.dtype)
        bg_counts.scatter_add_(1, scatter_idx.unsqueeze(-1), flat_bg.unsqueeze(-1))

        all_sums = torch.zeros(b * t, num_groups, dim, device=z.device, dtype=z.dtype)
        all_sums.scatter_add_(
            1,
            scatter_idx.unsqueeze(-1).expand(-1, -1, dim),
            flat_z,
        )
        all_counts = torch.zeros(b * t, num_groups, 1, device=z.device, dtype=z.dtype)
        all_counts.scatter_add_(
            1,
            scatter_idx.unsqueeze(-1),
            torch.ones(b * t, num_tokens, 1, device=z.device, dtype=z.dtype),
        )

        bg_mean = bg_sums / bg_counts.clamp_min(1.0)
        all_mean = all_sums / all_counts.clamp_min(1.0)
        group_tokens = torch.where(bg_counts > 0, bg_mean, all_mean)
        group_tokens = group_tokens.reshape(b, t, num_groups, dim)

        grouped_z = torch.cat([roi_tokens, group_tokens], dim=2)
        meta = {
            "roi_idx": roi_idx,
            "group_ids": group_ids,
            "keep_k": keep_k,
            "num_groups": num_groups,
            "num_tokens": num_tokens,
        }
        return grouped_z, meta

    def _build_pos_embeddings(self, predictor, meta, num_frames, dim, batch_size):
        predictor = getattr(predictor, "module", predictor)
        pos_embedding = getattr(predictor, "pos_embedding", None)
        if pos_embedding is None:
            return None

        num_tokens = meta["num_tokens"]
        needed = num_frames * num_tokens
        if pos_embedding.shape[1] < needed or pos_embedding.shape[-1] != dim:
            return None

        pos = pos_embedding[:, :needed].reshape(1, num_frames, num_tokens, dim)
        pos = pos.to(device=meta["roi_idx"].device)
        roi_pos = self._gather_patch_tokens(
            pos.expand(batch_size, -1, -1, -1),
            meta["roi_idx"],
        )

        group_ids = meta["group_ids"]
        num_groups = meta["num_groups"]
        scatter_idx = group_ids.view(1, num_tokens).expand(num_frames, num_tokens)
        frame_pos = pos[0]
        group_sums = torch.zeros(
            num_frames,
            num_groups,
            dim,
            device=pos.device,
            dtype=pos.dtype,
        )
        group_sums.scatter_add_(
            1,
            scatter_idx.unsqueeze(-1).expand(-1, -1, dim),
            frame_pos,
        )
        group_counts = torch.zeros(
            num_frames,
            num_groups,
            1,
            device=pos.device,
            dtype=pos.dtype,
        )
        group_counts.scatter_add_(
            1,
            scatter_idx.unsqueeze(-1),
            torch.ones(num_frames, num_tokens, 1, device=pos.device, dtype=pos.dtype),
        )
        group_pos = group_sums / group_counts.clamp_min(1.0)
        group_pos = group_pos.unsqueeze(0).expand(batch_size, -1, -1, -1)
        grouped_pos = torch.cat([roi_pos, group_pos], dim=2)
        return rearrange(grouped_pos, "b t p d -> b (t p) d")

    @staticmethod
    def _causal_mask(num_frames: int, tokens_per_frame: int, device):
        frame_ids = torch.arange(num_frames, device=device).repeat_interleave(
            tokens_per_frame
        )
        return frame_ids[:, None] >= frame_ids[None, :]

    @staticmethod
    def _run_predictor(predictor, z, pos_embeddings=None, attn_mask=None):
        num_frames = z.shape[1]
        z = rearrange(z, "b t p d -> b (t p) d")
        z = predictor(z, pos_embeddings=pos_embeddings, attn_mask=attn_mask)
        return rearrange(z, "b (t p) d -> b t p d", t=num_frames)

    @staticmethod
    def _expand_prediction(grouped_pred: torch.Tensor, meta: Dict):
        b, t, _, dim = grouped_pred.shape
        keep_k = meta["keep_k"]
        num_tokens = meta["num_tokens"]
        group_ids = meta["group_ids"]
        roi_idx = meta["roi_idx"]

        roi_pred = grouped_pred[:, :, :keep_k]
        group_pred = grouped_pred[:, :, keep_k:]
        gather_idx = group_ids.view(1, 1, num_tokens, 1).expand(b, t, num_tokens, dim)
        full_pred = torch.gather(group_pred, dim=2, index=gather_idx)
        if keep_k > 0:
            scatter_idx = roi_idx.unsqueeze(-1).expand(b, t, keep_k, dim)
            full_pred = full_pred.scatter(2, scatter_idx, roi_pred)
        return full_pred

    def forward(self, z: torch.Tensor, roi_mask: torch.Tensor, predictor):
        self.reset_timing_stats()
        with self._time_section("total"):
            num_visual_tokens = z.shape[2]
            if int(roi_mask.sum(dim=-1).min().item()) >= num_visual_tokens:
                with self._time_section("predictor_forward"):
                    pred = self._run_predictor(predictor, z)
                return pred, {
                    "effective_token_count": float(num_visual_tokens),
                    "roi_group_size": int(self.group_size),
                    "roi_num_group_tokens": 0,
                    "roi_grouped_token_count": float(num_visual_tokens),
                }

            with self._time_section("group_z_tokens"):
                grouped_z, meta = self._group_z_tokens(z, roi_mask)
            b, num_frames, grouped_tokens, dim = grouped_z.shape
            with self._time_section("build_pos_embeddings"):
                pos_embeddings = self._build_pos_embeddings(
                    predictor,
                    meta,
                    num_frames=num_frames,
                    dim=dim,
                    batch_size=b,
                )
            with self._time_section("build_attn_mask"):
                attn_mask = self._causal_mask(num_frames, grouped_tokens, grouped_z.device)
            with self._time_section("predictor_forward"):
                grouped_pred = self._run_predictor(
                    predictor,
                    grouped_z,
                    pos_embeddings=pos_embeddings,
                    attn_mask=attn_mask,
                )
            stats = {
                "roi_group_size": int(self.group_size),
                "roi_num_roi_tokens": float(meta["keep_k"]),
                "roi_num_group_tokens": float(meta["num_groups"]),
                "roi_grouped_token_count": float(grouped_tokens),
                "effective_token_count": float(grouped_tokens),
            }
            with self._time_section("expand_prediction"):
                full_pred = self._expand_prediction(grouped_pred, meta)
            return full_pred, stats
