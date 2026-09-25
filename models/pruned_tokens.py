from contextlib import contextmanager
from typing import Dict

import torch
import torch.nn as nn
from einops import rearrange
from eval_logging import elapsed_since, perf_counter

from models.grouped_tokens import GroupedTokenProcessor


class PrunedTokenProcessor(nn.Module):
    """Run the predictor only on ROI-selected patch tokens.

    This is a parameter-free companion to GroupedTokenProcessor. It preserves
    the full-grid output shape expected by the rest of DINO-WM, but predictor
    compute is spent only on selected ROI tokens. Non-selected tokens are filled
    from the input embedding by default, which keeps rollout shapes stable while
    making the compute path a true token-pruned path.
    """

    def __init__(self, fill_mode: str = "carry_forward"):
        super().__init__()
        if fill_mode != "carry_forward":
            raise ValueError("PrunedTokenProcessor currently supports fill_mode='carry_forward'.")
        self.fill_mode = fill_mode
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

    @staticmethod
    def _prune_z_tokens(z: torch.Tensor, roi_mask: torch.Tensor):
        roi_idx = GroupedTokenProcessor._selected_token_indices(roi_mask)
        pruned_z = GroupedTokenProcessor._gather_patch_tokens(z, roi_idx)
        return pruned_z, {
            "roi_idx": roi_idx,
            "keep_k": roi_idx.shape[-1],
            "num_tokens": z.shape[2],
        }

    @staticmethod
    def _build_pos_embeddings(predictor, meta, num_frames, dim, batch_size):
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
        roi_pos = GroupedTokenProcessor._gather_patch_tokens(
            pos.expand(batch_size, -1, -1, -1),
            meta["roi_idx"],
        )
        return rearrange(roi_pos, "b t p d -> b (t p) d")

    @staticmethod
    def _run_predictor(predictor, z, pos_embeddings=None, attn_mask=None):
        num_frames = z.shape[1]
        z = rearrange(z, "b t p d -> b (t p) d")
        z = predictor(z, pos_embeddings=pos_embeddings, attn_mask=attn_mask)
        return rearrange(z, "b (t p) d -> b t p d", t=num_frames)

    @staticmethod
    def _expand_prediction(pruned_pred: torch.Tensor, meta: Dict, fill: torch.Tensor):
        full_pred = fill.clone()
        keep_k = meta["keep_k"]
        if keep_k <= 0:
            return full_pred
        scatter_idx = meta["roi_idx"].unsqueeze(-1).expand(
            *meta["roi_idx"].shape,
            pruned_pred.shape[-1],
        )
        return full_pred.scatter(2, scatter_idx, pruned_pred)

    def forward(self, z: torch.Tensor, roi_mask: torch.Tensor, predictor):
        self.reset_timing_stats()
        with self._time_section("total"):
            num_visual_tokens = z.shape[2]
            min_kept = int(roi_mask.sum(dim=-1).min().item())
            if min_kept >= num_visual_tokens:
                with self._time_section("predictor_forward"):
                    pred = self._run_predictor(predictor, z)
                return pred, {
                    "effective_token_count": float(num_visual_tokens),
                    "roi_num_roi_tokens": float(num_visual_tokens),
                    "roi_pruned_token_count": float(num_visual_tokens),
                    "roi_num_pruned_tokens": 0,
                    "roi_prune_fill": self.fill_mode,
                }
            if min_kept <= 0:
                return z.clone(), {
                    "roi_num_roi_tokens": 0.0,
                    "roi_pruned_token_count": 0.0,
                    "roi_num_pruned_tokens": float(num_visual_tokens),
                    "roi_prune_fill": self.fill_mode,
                    "effective_token_count": 0.0,
                }

            with self._time_section("prune_z_tokens"):
                pruned_z, meta = self._prune_z_tokens(z, roi_mask)
            b, num_frames, pruned_tokens, dim = pruned_z.shape
            with self._time_section("build_pos_embeddings"):
                pos_embeddings = self._build_pos_embeddings(
                    predictor,
                    meta,
                    num_frames=num_frames,
                    dim=dim,
                    batch_size=b,
                )
            with self._time_section("build_attn_mask"):
                attn_mask = GroupedTokenProcessor._causal_mask(
                    num_frames,
                    pruned_tokens,
                    pruned_z.device,
                )
            with self._time_section("predictor_forward"):
                pruned_pred = self._run_predictor(
                    predictor,
                    pruned_z,
                    pos_embeddings=pos_embeddings,
                    attn_mask=attn_mask,
                )
            stats = {
                "roi_num_roi_tokens": float(meta["keep_k"]),
                "roi_pruned_token_count": float(pruned_tokens),
                "roi_num_pruned_tokens": float(num_visual_tokens - meta["keep_k"]),
                "roi_prune_fill": self.fill_mode,
                "effective_token_count": float(pruned_tokens),
            }
            with self._time_section("expand_prediction"):
                full_pred = self._expand_prediction(pruned_pred, meta, fill=z)
            return full_pred, stats
