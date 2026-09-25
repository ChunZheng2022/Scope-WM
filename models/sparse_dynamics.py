from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from einops import rearrange

from eval_logging import elapsed_since, perf_counter
from models.grouped_tokens import GroupedTokenProcessor


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except Exception:
            pass
    return getattr(cfg, key, default)


@dataclass
class SparseDynamicsConfig:
    enabled: bool = False
    mode: str = "sparse_primary"
    mask_source: str = "drs"  # drs/roi, dynamic_localizer, drs_union_dynamic, or drs_intersect_dynamic
    background_processor: str = "none"  # none, ignore, grouped, ru, lrm, or fdbu
    background_group_size: int = 2
    foreground_keep_ratio: float = 1.0
    foreground_topk: Optional[int] = None
    fill_mode: str = "carry_forward"
    loss_on_foreground_only: bool = True
    ru_rank: int = 16
    ru_alpha: float = 1.0
    ru_dropout: float = 0.0
    lrm_heads: int = 4
    lrm_num_tokens: int = 196
    lrm_alpha: float = 1.0
    lrm_dropout: float = 0.0
    lrm_rank: int = 16  # legacy alias for ru_rank
    fdbu_hidden_dim: Optional[int] = None
    fdbu_hidden_mult: float = 2.0
    fdbu_alpha: float = 1.0
    fdbu_dropout: float = 0.0
    cbu_hidden_dim: Optional[int] = None  # legacy alias for fdbu_hidden_dim
    cbu_hidden_mult: float = 2.0  # legacy alias for fdbu_hidden_mult
    cbu_alpha: float = 1.0  # legacy alias for fdbu_alpha
    cbu_dropout: float = 0.0  # legacy alias for fdbu_dropout
    dynamic_localizer_checkpoint: Optional[str] = None
    dynamic_localizer_threshold: float = 0.5
    dynamic_localizer_min_tokens: int = 32

    @classmethod
    def from_any(cls, cfg: Any) -> "SparseDynamicsConfig":
        if cfg is None:
            return cls()
        section = _cfg_get(cfg, "sparse_dynamics", cfg)
        defaults = cls()
        kwargs = {
            name: _cfg_get(section, name, getattr(defaults, name))
            for name in cls.__dataclass_fields__
        }
        ru_rank = _cfg_get(section, "ru_rank", None)
        legacy_lrm_rank = _cfg_get(section, "lrm_rank", None)
        if legacy_lrm_rank is not None and (
            ru_rank is None
            or (ru_rank == defaults.ru_rank and legacy_lrm_rank != defaults.lrm_rank)
        ):
            kwargs["ru_rank"] = legacy_lrm_rank
        source_aliases = {
            "roi": "drs",
            "roi_union_dynamic": "drs_union_dynamic",
            "roi_intersect_dynamic": "drs_intersect_dynamic",
        }
        kwargs["mask_source"] = source_aliases.get(kwargs["mask_source"], kwargs["mask_source"])
        if kwargs["background_processor"] == "cbu":
            kwargs["background_processor"] = "fdbu"
        for fdbu_name, cbu_name in [
            ("fdbu_hidden_dim", "cbu_hidden_dim"),
            ("fdbu_hidden_mult", "cbu_hidden_mult"),
            ("fdbu_alpha", "cbu_alpha"),
            ("fdbu_dropout", "cbu_dropout"),
        ]:
            if _cfg_get(section, fdbu_name, None) is None:
                legacy_value = _cfg_get(section, cbu_name, None)
                if legacy_value is not None:
                    kwargs[fdbu_name] = legacy_value
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_sparse_dynamics_config_dict(cfg: Any) -> Dict[str, Any]:
    return SparseDynamicsConfig.from_any(cfg).to_dict()


class ResidualUpdateModule(nn.Module):
    """Lightweight residual update for non-foreground tokens.

    The foreground sparse predictor produces a mean foreground delta per frame.
    RU projects that delta through a rank bottleneck and broadcasts a small
    residual to background tokens. A token-wise gate keeps the correction
    contextual without running full self-attention over background tokens.
    """

    def __init__(
        self,
        dim: int,
        rank: int = 16,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.rank = max(1, int(rank))
        self.alpha = float(alpha)
        self.down = nn.Linear(self.dim, self.rank)
        self.up = nn.Linear(self.rank, self.dim)
        self.gate = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, 1),
            nn.Sigmoid(),
        )
        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        z: torch.Tensor,
        sparse_z: torch.Tensor,
        sparse_pred: torch.Tensor,
        foreground_mask: torch.Tensor,
    ) -> torch.Tensor:
        # FDBU: summarize predicted foreground state and foreground delta.
        fg_delta = sparse_pred - sparse_z
        fg_context = fg_delta.mean(dim=2, keepdim=True)
        correction = self.up(self.dropout(self.down(fg_context)))
        correction = correction * self.alpha
        gate = self.gate(z)
        background_mask = (~foreground_mask).unsqueeze(-1).to(dtype=z.dtype)
        return z + correction * gate * background_mask


class LowRankCorrectionModule(ResidualUpdateModule):
    """Backward-compatible alias for checkpoints saved before LRM was corrected.

    New experiments should use ``background_processor='ru'`` for this behavior.
    The runtime ``background_processor='lrm'`` path instantiates
    ``DDPLowRankCorrectionModule`` below.
    """


class DDPLowRankCorrectionModule(nn.Module):
    """DDP-WM-style background-query / foreground-memory LRM.

    Background tokens query the predicted foreground tokens through one
    cross-attention layer. This matches the public DDP-WM implementation more
    closely than the cheaper pooled residual update in ``ResidualUpdateModule``.
    """

    def __init__(
        self,
        dim: int,
        heads: int = 4,
        num_tokens: int = 196,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.heads = int(heads)
        self.num_tokens = int(num_tokens)
        self.alpha = float(alpha)
        if self.dim % self.heads != 0:
            raise ValueError(
                f"LRM requires dim divisible by lrm_heads; got dim={self.dim}, heads={self.heads}."
            )
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=self.dim,
            num_heads=self.heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm = nn.LayerNorm(self.dim)
        self.dropout = nn.Dropout(float(dropout))
        self.ape = nn.Parameter(torch.randn(self.num_tokens, self.dim))

    @staticmethod
    def _selected_mask_from_indices(mask_shape, idx):
        selected = torch.zeros(mask_shape, dtype=torch.bool, device=idx.device)
        if idx.shape[-1] > 0:
            selected.scatter_(-1, idx, True)
        return selected

    def forward(
        self,
        z: torch.Tensor,
        sparse_pred: torch.Tensor,
        meta: Dict,
    ) -> torch.Tensor:
        bsz, num_frames, num_tokens, dim = z.shape
        keep_k = int(meta["keep_k"])
        if keep_k <= 0 or keep_k >= num_tokens:
            return z.clone()
        if num_tokens > self.num_tokens:
            raise ValueError(
                f"LRM absolute position table has {self.num_tokens} tokens, but input has {num_tokens}."
            )

        fg_idx = meta["fg_idx"]
        fg_selected = self._selected_mask_from_indices(
            (bsz, num_frames, num_tokens),
            fg_idx,
        )
        bg_idx = GroupedTokenProcessor._selected_token_indices(~fg_selected)
        bg_tokens = GroupedTokenProcessor._gather_patch_tokens(z, bg_idx)

        ape = self.ape[:num_tokens].to(device=z.device, dtype=z.dtype)
        ape = ape.view(1, 1, num_tokens, dim).expand(bsz, num_frames, -1, -1)
        fg_pos = GroupedTokenProcessor._gather_patch_tokens(ape, fg_idx)
        bg_pos = GroupedTokenProcessor._gather_patch_tokens(ape, bg_idx)

        flat_bg = rearrange(bg_tokens, "b t p d -> (b t) p d")
        flat_query = rearrange(bg_tokens + bg_pos, "b t p d -> (b t) p d")
        flat_key = rearrange(sparse_pred + fg_pos, "b t p d -> (b t) p d")
        flat_value = rearrange(sparse_pred, "b t p d -> (b t) p d")

        attn_out = self.cross_attn(
            query=flat_query,
            key=flat_key,
            value=flat_value,
            need_weights=False,
        )[0]
        updated_bg = self.norm(flat_bg + self.alpha * self.dropout(attn_out))
        updated_bg = rearrange(updated_bg, "(b t) p d -> b t p d", b=bsz, t=num_frames)

        fill = z.clone()
        scatter_idx = bg_idx.unsqueeze(-1).expand(*bg_idx.shape, dim)
        return fill.scatter(2, scatter_idx, updated_bg)


class ForegroundDeltaBackgroundUpdateModule(nn.Module):
    """Conditionally update background tokens from foreground dynamics.

    Sparse Primary predicts foreground tokens. FDBU summarizes the foreground
    prediction and foreground delta, then uses that context to produce a gated
    residual update only on non-foreground tokens.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        hidden_mult: float = 2.0,
        alpha: float = 1.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.dim = int(dim)
        self.hidden_dim = int(hidden_dim or max(1, round(self.dim * float(hidden_mult))))
        self.alpha = float(alpha)
        context_dim = self.dim * 2
        token_context_dim = self.dim * 2
        self.context_proj = nn.Sequential(
            nn.LayerNorm(context_dim),
            nn.Linear(context_dim, self.dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
        )
        self.update = nn.Sequential(
            nn.LayerNorm(token_context_dim),
            nn.Linear(token_context_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.dim),
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(token_context_dim),
            nn.Linear(token_context_dim, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        z: torch.Tensor,
        sparse_z: torch.Tensor,
        sparse_pred: torch.Tensor,
        foreground_mask: torch.Tensor,
    ) -> torch.Tensor:
        fg_delta = sparse_pred - sparse_z
        fg_pred_context = sparse_pred.mean(dim=2, keepdim=True)
        fg_delta_context = fg_delta.mean(dim=2, keepdim=True)
        context = self.context_proj(torch.cat([fg_pred_context, fg_delta_context], dim=-1))
        context = context.expand(-1, -1, z.shape[2], -1)
        token_context = torch.cat([z, context], dim=-1)
        # Token-wise gated residual background update.
        update = self.update(token_context)
        gate = self.gate(token_context)
        background_mask = (~foreground_mask).unsqueeze(-1).to(dtype=z.dtype)
        return z + self.alpha * gate * update * background_mask


ContextualBackgroundUpdateModule = ForegroundDeltaBackgroundUpdateModule


class SparsePrimaryDynamics(nn.Module):
    """Run the predictor only on foreground patch tokens.

    This is the first, parameter-free stage of the pluggable sparse dynamics
    stack. It preserves the full-grid latent shape expected by the rest of
    DINO-WM: foreground predictions are scattered back, while non-foreground
    tokens are carried forward from the input latent.
    """

    def __init__(
        self,
        dim: int,
        fill_mode: str = "carry_forward",
        background_processor: str = "none",
        background_group_size: int = 2,
        ru_rank: int = 16,
        ru_alpha: float = 1.0,
        ru_dropout: float = 0.0,
        lrm_heads: int = 4,
        lrm_num_tokens: int = 196,
        lrm_alpha: float = 1.0,
        lrm_dropout: float = 0.0,
        fdbu_hidden_dim: Optional[int] = None,
        fdbu_hidden_mult: float = 2.0,
        fdbu_alpha: float = 1.0,
        fdbu_dropout: float = 0.0,
        cbu_hidden_dim: Optional[int] = None,
        cbu_hidden_mult: float = 2.0,
        cbu_alpha: float = 1.0,
        cbu_dropout: float = 0.0,
    ):
        super().__init__()
        if fill_mode != "carry_forward":
            raise ValueError(
                "SparsePrimaryDynamics currently supports fill_mode='carry_forward'."
            )
        if background_processor == "cbu":
            background_processor = "fdbu"
        if background_processor not in {"none", "ignore", "grouped", "ru", "lrm", "fdbu"}:
            raise ValueError(
                "SparsePrimaryDynamics background_processor must be one of: none, ignore, grouped, ru, lrm, fdbu."
            )
        self.dim = int(dim)
        self.fill_mode = fill_mode
        self.background_processor = background_processor
        self.grouped_processor = (
            GroupedTokenProcessor(group_size=background_group_size)
            if background_processor == "grouped"
            else None
        )
        self.ru = (
            ResidualUpdateModule(
                dim=self.dim,
                rank=ru_rank,
                alpha=ru_alpha,
                dropout=ru_dropout,
            )
            if background_processor == "ru"
            else None
        )
        self.lrm = (
            DDPLowRankCorrectionModule(
                dim=self.dim,
                heads=lrm_heads,
                num_tokens=lrm_num_tokens,
                alpha=lrm_alpha,
                dropout=lrm_dropout,
            )
            if background_processor == "lrm"
            else None
        )
        if fdbu_hidden_dim is None and cbu_hidden_dim is not None:
            fdbu_hidden_dim = cbu_hidden_dim
        if fdbu_hidden_mult == 2.0 and cbu_hidden_mult != 2.0:
            fdbu_hidden_mult = cbu_hidden_mult
        if fdbu_alpha == 1.0 and cbu_alpha != 1.0:
            fdbu_alpha = cbu_alpha
        if fdbu_dropout == 0.0 and cbu_dropout != 0.0:
            fdbu_dropout = cbu_dropout
        self.fdbu = (
            ForegroundDeltaBackgroundUpdateModule(
                dim=self.dim,
                hidden_dim=fdbu_hidden_dim,
                hidden_mult=fdbu_hidden_mult,
                alpha=fdbu_alpha,
                dropout=fdbu_dropout,
            )
            if background_processor == "fdbu"
            else None
        )
        self.cbu = self.fdbu
        self.reset_timing_stats()

    def reset_timing_stats(self):
        self._last_timing = {}

    def get_last_timing(self):
        timing = dict(getattr(self, "_last_timing", {}))
        if self.grouped_processor is not None:
            for key, value in self.grouped_processor.get_last_timing().items():
                timing[f"grouped_{key}"] = value
        return timing

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
        pos = pos.to(device=meta["fg_idx"].device)
        fg_pos = GroupedTokenProcessor._gather_patch_tokens(
            pos.expand(batch_size, -1, -1, -1),
            meta["fg_idx"],
        )
        return rearrange(fg_pos, "b t p d -> b (t p) d")

    @staticmethod
    def _run_predictor(predictor, z, pos_embeddings=None, attn_mask=None):
        num_frames = z.shape[1]
        z = rearrange(z, "b t p d -> b (t p) d")
        z = predictor(z, pos_embeddings=pos_embeddings, attn_mask=attn_mask)
        return rearrange(z, "b (t p) d -> b t p d", t=num_frames)

    @staticmethod
    def _sparse_z_tokens(z: torch.Tensor, foreground_mask: torch.Tensor):
        fg_idx = GroupedTokenProcessor._selected_token_indices(foreground_mask)
        sparse_z = GroupedTokenProcessor._gather_patch_tokens(z, fg_idx)
        return sparse_z, {
            "fg_idx": fg_idx,
            "keep_k": fg_idx.shape[-1],
            "num_tokens": z.shape[2],
        }

    @staticmethod
    def _expand_prediction(sparse_pred: torch.Tensor, meta: Dict, fill: torch.Tensor):
        full_pred = fill.clone()
        keep_k = meta["keep_k"]
        if keep_k <= 0:
            return full_pred
        scatter_idx = meta["fg_idx"].unsqueeze(-1).expand(
            *meta["fg_idx"].shape,
            sparse_pred.shape[-1],
        )
        return full_pred.scatter(2, scatter_idx, sparse_pred)

    def forward(self, z: torch.Tensor, foreground_mask: torch.Tensor, predictor):
        self.reset_timing_stats()
        with self._time_section("total"):
            num_tokens = z.shape[2]
            min_kept = int(foreground_mask.sum(dim=-1).min().item())
            if self.background_processor == "grouped":
                with self._time_section("grouped_background_processor"):
                    pred, grouped_stats = self.grouped_processor(
                        z,
                        foreground_mask,
                        predictor,
                    )
                grouped_stats.update(
                    {
                        "sparse_dynamics_enabled": True,
                        "sparse_dynamics_mode": "sparse_primary",
                        "sparse_background_processor": "grouped",
                        "sparse_num_tokens": float(num_tokens),
                        "sparse_num_foreground_tokens": grouped_stats.get(
                            "roi_num_roi_tokens",
                            float(foreground_mask.sum(dim=-1).float().mean().item()),
                        ),
                        "sparse_num_background_tokens": float(
                            num_tokens
                            - foreground_mask.sum(dim=-1).float().mean().item()
                        ),
                        "sparse_fill_mode": self.fill_mode,
                    }
                )
                return pred, grouped_stats

            if min_kept >= num_tokens:
                with self._time_section("predictor_forward"):
                    pred = self._run_predictor(predictor, z)
                return pred, {
                    "sparse_dynamics_enabled": True,
                    "sparse_dynamics_mode": "sparse_primary",
                    "sparse_background_processor": self.background_processor,
                    "sparse_num_tokens": float(num_tokens),
                    "sparse_num_foreground_tokens": float(num_tokens),
                    "sparse_num_background_tokens": 0.0,
                    "effective_token_count": float(num_tokens),
                    "sparse_fill_mode": self.fill_mode,
                }
            if min_kept <= 0:
                fill = torch.zeros_like(z) if self.background_processor == "ignore" else z.clone()
                return fill, {
                    "sparse_dynamics_enabled": True,
                    "sparse_dynamics_mode": "sparse_primary",
                    "sparse_background_processor": self.background_processor,
                    "sparse_num_tokens": float(num_tokens),
                    "sparse_num_foreground_tokens": 0.0,
                    "sparse_num_background_tokens": float(num_tokens),
                    "effective_token_count": 0.0,
                    "sparse_fill_mode": self.fill_mode,
                }

            with self._time_section("gather_foreground_tokens"):
                # Sparse Primary: run the world-model predictor only on selected foreground tokens.
                sparse_z, meta = self._sparse_z_tokens(z, foreground_mask)
            bsz, num_frames, sparse_tokens, dim = sparse_z.shape
            with self._time_section("build_pos_embeddings"):
                pos_embeddings = self._build_pos_embeddings(
                    predictor,
                    meta,
                    num_frames=num_frames,
                    dim=dim,
                    batch_size=bsz,
                )
            with self._time_section("build_attn_mask"):
                attn_mask = GroupedTokenProcessor._causal_mask(
                    num_frames,
                    sparse_tokens,
                    sparse_z.device,
                )
            with self._time_section("predictor_forward"):
                sparse_pred = self._run_predictor(
                    predictor,
                    sparse_z,
                    pos_embeddings=pos_embeddings,
                    attn_mask=attn_mask,
                )
            with self._time_section("background_update"):
                if self.background_processor == "ignore":
                    fill = torch.zeros_like(z)
                elif self.background_processor == "ru":
                    fill = self.ru(
                        z,
                        sparse_z,
                        sparse_pred,
                        foreground_mask,
                    )
                elif self.background_processor == "lrm":
                    fill = self.lrm(
                        z,
                        sparse_pred,
                        meta,
                    )
                elif self.background_processor == "fdbu":
                    fill = self.fdbu(
                        z,
                        sparse_z,
                        sparse_pred,
                        foreground_mask,
                    )
                else:
                    fill = z
            with self._time_section("expand_prediction"):
                # Scatter predicted foreground tokens back to the full token grid.
                full_pred = self._expand_prediction(sparse_pred, meta, fill=fill)
            return full_pred, {
                "sparse_dynamics_enabled": True,
                "sparse_dynamics_mode": "sparse_primary",
                "sparse_background_processor": self.background_processor,
                "sparse_num_tokens": float(num_tokens),
                "sparse_num_foreground_tokens": float(meta["keep_k"]),
                "sparse_num_background_tokens": float(num_tokens - meta["keep_k"]),
                "effective_token_count": float(sparse_tokens),
                "sparse_fill_mode": self.fill_mode,
            }
