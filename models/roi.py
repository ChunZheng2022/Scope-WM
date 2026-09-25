import math
import time
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Distilled Relevance Selection (DRS) is the public name for this module.
# The older ROI names are kept as compatibility aliases for existing configs,
# checkpoints, and logs.
NO_ROI_MODES = {"none", "full", None}
NO_DRS_MODES = NO_ROI_MODES
LEARNED_ROI_MODES = {"distilled_topk", "learned_topk", "topk"}
LEARNED_DRS_MODES = LEARNED_ROI_MODES
SPARSE_BASELINE_MODES = {
    "random",
    "fixed_random",
    "lhs",
    "attention_encoder",
    "attention_wm",
}


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
class ROIDistillConfig:
    enabled: bool = False
    teacher_checkpoint: Optional[str] = None
    target_path: Optional[str] = None
    output_dir: Optional[str] = None
    target_type: str = "grad_x_input"
    loss_type: str = "kl"
    temperature: float = 1.0
    budget_loss_weight: float = 0.0
    action_conditioned: bool = True
    freeze_world_model: bool = True
    train_roi_head_only: bool = True

    @classmethod
    def from_any(cls, cfg: Any) -> "ROIDistillConfig":
        if cfg is None:
            return cls()
        return cls(**{field: _cfg_get(cfg, field, getattr(cls(), field)) for field in cls.__dataclass_fields__})


DRSDistillConfig = ROIDistillConfig


@dataclass
class SparseBaselineConfig:
    enabled: bool = False
    modes: Sequence[str] = field(
        default_factory=lambda: [
            "random",
            "fixed_random",
            "lhs",
            "attention_encoder",
            "attention_wm",
        ]
    )
    drop_ratios: Sequence[float] = field(
        default_factory=lambda: [0.1, 0.3, 0.5, 0.7, 0.9]
    )
    resample_each_mpc_iter: bool = True
    save_masks: bool = False

    @classmethod
    def from_any(cls, cfg: Any) -> "SparseBaselineConfig":
        if cfg is None:
            return cls()
        return cls(**{field: _cfg_get(cfg, field, getattr(cls(), field)) for field in cls.__dataclass_fields__})


@dataclass
class ROIConfig:
    use_roi: bool = False
    roi_mode: str = "none"
    roi_keep_ratio: float = 1.0
    roi_topk: Optional[int] = None
    roi_action_aware: bool = True
    roi_mask_type: str = "mask"
    roi_detach_score: bool = True
    roi_debug: bool = False
    roi_head_checkpoint: Optional[str] = None
    roi_hidden_dim: int = 128
    roi_background_token: str = "zero"
    roi_group_size: int = 2
    fixed_random_seed: int = 0
    roi_distill: ROIDistillConfig = field(default_factory=ROIDistillConfig)
    sparse_baseline: SparseBaselineConfig = field(default_factory=SparseBaselineConfig)

    @classmethod
    def from_any(cls, cfg: Any) -> "ROIConfig":
        if cfg is None:
            return cls()
        kwargs = {}
        defaults = cls()
        for name in cls.__dataclass_fields__:
            if name in {"roi_distill", "sparse_baseline"}:
                continue
            if name == "use_roi":
                value = _cfg_get(cfg, "use_drs", _cfg_get(cfg, "drs_enabled", None))
                if value is None:
                    value = _cfg_get(cfg, name, getattr(defaults, name))
            else:
                drs_name = name.replace("roi", "drs", 1)
                value = _cfg_get(cfg, drs_name, None)
                if value is None:
                    value = _cfg_get(cfg, name, getattr(defaults, name))
            kwargs[name] = value
        kwargs["roi_distill"] = ROIDistillConfig.from_any(
            _cfg_get(cfg, "drs_distill", _cfg_get(cfg, "roi_distill", None))
        )
        kwargs["sparse_baseline"] = SparseBaselineConfig.from_any(
            _cfg_get(cfg, "sparse_baseline", None)
        )
        return cls(**kwargs)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def extract_roi_config_dict(cfg: Any) -> Dict[str, Any]:
    """Return only DRS/legacy ROI fields so Hydra does not recurse into planner/model targets."""
    return ROIConfig.from_any(cfg).to_dict()


DRSConfig = ROIConfig


def extract_drs_config_dict(cfg: Any) -> Dict[str, Any]:
    """Return DRS-related config fields.

    This is the preferred name. ``extract_roi_config_dict`` is kept for
    backwards compatibility with older experiment scripts.
    """
    return extract_roi_config_dict(cfg)


def compute_keep_k(
    num_tokens: int, keep_ratio: float = 1.0, topk: Optional[int] = None
) -> int:
    if num_tokens < 0:
        raise ValueError(f"num_tokens must be non-negative, got {num_tokens}")
    if topk is not None:
        keep_k = int(topk)
    else:
        keep_ratio = float(keep_ratio)
        keep_k = int(math.ceil(num_tokens * keep_ratio))
        if keep_ratio > 0 and keep_k == 0 and num_tokens > 0:
            keep_k = 1
    return max(0, min(int(num_tokens), keep_k))


def _as_batch_shape(batch_shape: Any) -> Tuple[int, ...]:
    if batch_shape is None:
        return ()
    if isinstance(batch_shape, int):
        return (batch_shape,)
    return tuple(batch_shape)


def random_token_mask(
    num_tokens: int,
    keep_ratio: float = 1.0,
    topk: Optional[int] = None,
    batch_shape: Any = (),
    device: Optional[torch.device] = None,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    batch_shape = _as_batch_shape(batch_shape)
    keep_k = compute_keep_k(num_tokens, keep_ratio, topk)
    if keep_k >= num_tokens:
        return torch.ones(*batch_shape, num_tokens, dtype=torch.bool, device=device)
    if keep_k <= 0:
        return torch.zeros(*batch_shape, num_tokens, dtype=torch.bool, device=device)
    noise = torch.rand(*batch_shape, num_tokens, device=device, generator=generator)
    idx = torch.topk(noise, k=keep_k, dim=-1).indices
    mask = torch.zeros(*batch_shape, num_tokens, dtype=torch.bool, device=device)
    return mask.scatter(-1, idx, True)


def fixed_random_token_mask(
    num_tokens: int,
    keep_ratio: float = 1.0,
    topk: Optional[int] = None,
    batch_shape: Any = (),
    device: Optional[torch.device] = None,
    seed: int = 0,
) -> torch.Tensor:
    generator = torch.Generator(device=device if device is not None else "cpu")
    generator.manual_seed(int(seed))
    return random_token_mask(
        num_tokens=num_tokens,
        keep_ratio=keep_ratio,
        topk=topk,
        batch_shape=batch_shape,
        device=device,
        generator=generator,
    )


def _lhs_indices(num_tokens: int, keep_k: int, generator: torch.Generator) -> torch.Tensor:
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        return torch.randperm(num_tokens, generator=generator)[:keep_k]

    # Latin-hypercube style: stratify both axes, pair one axis with a permutation,
    # then fill collisions deterministically from a random permutation.
    u = (torch.arange(keep_k, dtype=torch.float32) + torch.rand(keep_k, generator=generator)) / max(keep_k, 1)
    perm = torch.randperm(keep_k, generator=generator)
    v = (perm.float() + torch.rand(keep_k, generator=generator)) / max(keep_k, 1)
    rows = torch.clamp((u * side).long(), 0, side - 1)
    cols = torch.clamp((v * side).long(), 0, side - 1)
    idx = torch.unique(rows * side + cols, sorted=False)
    if idx.numel() >= keep_k:
        return idx[:keep_k]

    used = torch.zeros(num_tokens, dtype=torch.bool)
    used[idx] = True
    fill = torch.randperm(num_tokens, generator=generator)
    fill = fill[~used[fill]]
    return torch.cat([idx, fill[: keep_k - idx.numel()]], dim=0)


def lhs_token_mask(
    num_tokens: int,
    keep_ratio: float = 1.0,
    topk: Optional[int] = None,
    batch_shape: Any = (),
    device: Optional[torch.device] = None,
    seed: Optional[int] = None,
) -> torch.Tensor:
    batch_shape = _as_batch_shape(batch_shape)
    keep_k = compute_keep_k(num_tokens, keep_ratio, topk)
    if keep_k >= num_tokens:
        return torch.ones(*batch_shape, num_tokens, dtype=torch.bool, device=device)
    if keep_k <= 0:
        return torch.zeros(*batch_shape, num_tokens, dtype=torch.bool, device=device)

    flat_batch = int(math.prod(batch_shape)) if batch_shape else 1
    generator = torch.Generator(device="cpu")
    if seed is not None:
        generator.manual_seed(int(seed))
    mask = torch.zeros(flat_batch, num_tokens, dtype=torch.bool)
    for row in range(flat_batch):
        idx = _lhs_indices(num_tokens, keep_k, generator)
        mask[row, idx] = True
    mask = mask.reshape(*batch_shape, num_tokens)
    return mask.to(device=device)


def topk_token_mask(
    scores: torch.Tensor,
    keep_ratio: float = 1.0,
    topk: Optional[int] = None,
) -> torch.Tensor:
    if scores.ndim < 2:
        raise ValueError(f"scores must end with token dimension, got {scores.shape}")
    num_tokens = scores.shape[-1]
    keep_k = compute_keep_k(num_tokens, keep_ratio, topk)
    if keep_k >= num_tokens:
        return torch.ones_like(scores, dtype=torch.bool)
    if keep_k <= 0:
        return torch.zeros_like(scores, dtype=torch.bool)
    idx = torch.topk(scores, k=keep_k, dim=-1).indices
    mask = torch.zeros_like(scores, dtype=torch.bool)
    return mask.scatter(-1, idx, True)


def apply_token_mask_or_selection(
    visual_tokens: torch.Tensor,
    mask: torch.Tensor,
    mask_type: str = "mask",
    background_token: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if mask_type == "mask":
        expanded_mask = mask.unsqueeze(-1).to(dtype=visual_tokens.dtype)
        if background_token is None:
            return visual_tokens * expanded_mask
        return torch.where(mask.unsqueeze(-1), visual_tokens, background_token)

    if mask_type == "grouped":
        # Grouped-token execution is handled by VWorldModel because it must
        # jointly group visual/action/proprio channels and then scatter predictor
        # outputs back to the full patch grid. ROISelector only owns the mask.
        return visual_tokens

    if mask_type == "prune":
        keep_k = int(mask.sum(dim=-1).min().item())
        idx = torch.topk(mask.to(torch.int64), k=keep_k, dim=-1).indices
        idx = torch.sort(idx, dim=-1).values
        if visual_tokens.ndim == 3:
            gather_idx = idx.unsqueeze(-1).expand(-1, -1, visual_tokens.shape[-1])
            return torch.gather(visual_tokens, dim=1, index=gather_idx)
        if visual_tokens.ndim == 4:
            gather_idx = idx.unsqueeze(-1).expand(
                -1, -1, -1, visual_tokens.shape[-1]
            )
            return torch.gather(visual_tokens, dim=2, index=gather_idx)
        raise ValueError(f"Unsupported token shape for prune: {visual_tokens.shape}")

    raise ValueError(f"Unsupported roi_mask_type: {mask_type}")


def _flatten_visual_tokens(
    visual_tokens: torch.Tensor,
) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    if visual_tokens.ndim == 3:
        b, n, c = visual_tokens.shape
        return visual_tokens, (b, n, c)
    if visual_tokens.ndim == 4:
        b, t, n, c = visual_tokens.shape
        return visual_tokens.reshape(b * t, n, c), (b, t, n, c)
    raise ValueError(
        f"visual_tokens must be B x N x C or B x T x N x C, got {visual_tokens.shape}"
    )


def _unflatten_scores(scores: torch.Tensor, original_shape: Tuple[int, ...]) -> torch.Tensor:
    if len(original_shape) == 3:
        return scores
    b, t, n, _ = original_shape
    return scores.reshape(b, t, n)


def _flatten_condition(
    value: Optional[torch.Tensor], original_shape: Tuple[int, ...]
) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if len(original_shape) == 4:
        b, t, _, _ = original_shape
        if value.ndim == 3 and value.shape[0] == b and value.shape[1] == t:
            return value.reshape(b * t, -1)
        if value.ndim == 2 and value.shape[0] == b:
            return value[:, None, :].expand(b, t, value.shape[-1]).reshape(b * t, -1)
    else:
        b, _, _ = original_shape
        if value.ndim == 2 and value.shape[0] == b:
            return value
        if value.ndim == 3 and value.shape[0] == b:
            return value[:, -1, :].reshape(b, -1)
    return value.reshape(value.shape[0], -1)


class ActionAwareROIHead(nn.Module):
    def __init__(
        self,
        token_dim: int,
        hidden_dim: int = 128,
        action_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
        use_timestep: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.token_dim = token_dim
        self.hidden_dim = hidden_dim
        self.token_proj = nn.Linear(token_dim, hidden_dim)
        self.action_proj = (
            nn.Linear(action_dim, hidden_dim) if action_dim is not None and action_dim > 0 else None
        )
        self.proprio_proj = (
            nn.Linear(proprio_dim, hidden_dim)
            if proprio_dim is not None and proprio_dim > 0
            else None
        )
        self.context_proj = (
            nn.Linear(context_dim, hidden_dim)
            if context_dim is not None and context_dim > 0
            else None
        )
        self.time_proj = nn.Linear(1, hidden_dim) if use_timestep else None
        self.scorer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def _project_condition(
        self,
        value: Optional[torch.Tensor],
        projector: Optional[nn.Linear],
        original_shape: Tuple[int, ...],
    ) -> Optional[torch.Tensor]:
        if value is None or projector is None:
            return None
        value = _flatten_condition(value, original_shape)
        if value is None:
            return None
        if value.shape[-1] != projector.in_features:
            warnings.warn(
                "Skipping ROI condition with mismatched feature dim "
                f"{value.shape[-1]} != {projector.in_features}.",
                RuntimeWarning,
            )
            return None
        return projector(value)

    def _project_timestep(
        self, timestep: Optional[torch.Tensor], original_shape: Tuple[int, ...], device
    ) -> Optional[torch.Tensor]:
        if timestep is None or self.time_proj is None:
            return None
        value = timestep
        if not isinstance(value, torch.Tensor):
            value = torch.tensor(value, dtype=torch.float32, device=device)
        value = value.to(device=device, dtype=torch.float32)
        if len(original_shape) == 4:
            b, t, _, _ = original_shape
            if value.ndim == 0:
                value = value.repeat(b * t)
            elif value.ndim == 1 and value.shape[0] == t:
                value = value[None, :].expand(b, t).reshape(b * t)
            else:
                value = value.reshape(b * t)
        else:
            b, _, _ = original_shape
            if value.ndim == 0:
                value = value.repeat(b)
            else:
                value = value.reshape(b)
        return self.time_proj(value[:, None])

    def forward(
        self,
        visual_tokens: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        flat_tokens, original_shape = _flatten_visual_tokens(visual_tokens)
        token_h = self.token_proj(flat_tokens)
        cond_h = torch.zeros(
            flat_tokens.shape[0],
            self.hidden_dim,
            dtype=token_h.dtype,
            device=token_h.device,
        )

        for value, projector in (
            (action, self.action_proj),
            (proprio, self.proprio_proj),
            (context, self.context_proj),
        ):
            projected = self._project_condition(value, projector, original_shape)
            if projected is not None:
                cond_h = cond_h + projected
        projected_time = self._project_timestep(timestep, original_shape, flat_tokens.device)
        if projected_time is not None:
            cond_h = cond_h + projected_time

        fused = token_h + cond_h[:, None, :]
        scores = self.scorer(fused).squeeze(-1)
        return _unflatten_scores(scores, original_shape)


class ROISelector(nn.Module):
    def __init__(
        self,
        config: Optional[Any] = None,
        token_dim: Optional[int] = None,
        action_dim: Optional[int] = None,
        proprio_dim: Optional[int] = None,
        context_dim: Optional[int] = None,
    ):
        super().__init__()
        self.config = ROIConfig.from_any(config)
        self.token_dim = token_dim
        self.action_dim = action_dim
        self.proprio_dim = proprio_dim
        self.context_dim = context_dim
        self.last_stats: Dict[str, Any] = {}
        self.roi_head: Optional[ActionAwareROIHead] = None

        if (
            self.config.use_roi
            and self.config.roi_mode in LEARNED_ROI_MODES
            and token_dim is not None
        ):
            self.roi_head = ActionAwareROIHead(
                token_dim=token_dim,
                hidden_dim=self.config.roi_hidden_dim,
                action_dim=action_dim if self.config.roi_action_aware else None,
                proprio_dim=proprio_dim if self.config.roi_action_aware else None,
                context_dim=context_dim if self.config.roi_action_aware else None,
            )

    def load_checkpoint(self, checkpoint_path: str, map_location: Optional[str] = None):
        if self.roi_head is None:
            if self.token_dim is None:
                raise ValueError("token_dim is required to load a ROI head checkpoint")
            self.roi_head = ActionAwareROIHead(
                token_dim=self.token_dim,
                hidden_dim=self.config.roi_hidden_dim,
                action_dim=self.action_dim if self.config.roi_action_aware else None,
                proprio_dim=self.proprio_dim if self.config.roi_action_aware else None,
                context_dim=self.context_dim if self.config.roi_action_aware else None,
            )
        ckpt = torch.load(checkpoint_path, map_location=map_location or "cpu")
        state = ckpt.get("roi_head", ckpt.get("state_dict", ckpt))
        self.roi_head.load_state_dict(state)

    def _full_mask(self, visual_tokens: torch.Tensor) -> torch.Tensor:
        return torch.ones(
            *visual_tokens.shape[:-1],
            dtype=torch.bool,
            device=visual_tokens.device,
        )

    def _placeholder_stats(
        self, mode: str, visual_tokens: torch.Tensor, elapsed: float, reason: str
    ) -> Dict[str, Any]:
        mask = self._full_mask(visual_tokens)
        stats = self._build_stats(
            mode=mode,
            visual_tokens=visual_tokens,
            mask=mask,
            scores=None,
            elapsed=elapsed,
        )
        stats["roi_placeholder"] = True
        stats["roi_placeholder_reason"] = reason
        stats["drs_placeholder"] = True
        stats["drs_placeholder_reason"] = reason
        return stats

    @staticmethod
    def _add_drs_aliases(stats: Dict[str, Any]) -> Dict[str, Any]:
        for key, value in list(stats.items()):
            if key.startswith("roi_"):
                stats.setdefault("drs_" + key[len("roi_") :], value)
        if "roi_enabled" in stats:
            stats.setdefault("drs_enabled", stats["roi_enabled"])
        if "roi_mode" in stats:
            stats.setdefault("drs_mode", stats["roi_mode"])
        return stats

    def _build_stats(
        self,
        mode: str,
        visual_tokens: torch.Tensor,
        mask: torch.Tensor,
        scores: Optional[torch.Tensor],
        elapsed: float,
    ) -> Dict[str, Any]:
        num_tokens = int(visual_tokens.shape[-2])
        kept = mask.sum(dim=-1).float()
        stats: Dict[str, Any] = {
            "roi_enabled": bool(self.config.use_roi and mode not in NO_ROI_MODES),
            "roi_mode": "none" if mode is None else mode,
            "roi_keep_ratio": float(self.config.roi_keep_ratio),
            "roi_num_tokens": num_tokens,
            "roi_num_kept_tokens": float(kept.mean().item()),
            "effective_token_count": float(kept.mean().item()),
            "roi_mask_time_sec": float(elapsed),
            "roi_target_type": self.config.roi_distill.target_type,
            "roi_action_aware": bool(self.config.roi_action_aware),
        }
        if scores is not None:
            scores_detached = scores.detach().float()
            stats.update(
                {
                    "roi_score_mean": float(scores_detached.mean().item()),
                    "roi_score_std": float(scores_detached.std(unbiased=False).item()),
                    "roi_score_min": float(scores_detached.min().item()),
                    "roi_score_max": float(scores_detached.max().item()),
                }
            )
        else:
            stats.update(
                {
                    "roi_score_mean": None,
                    "roi_score_std": None,
                    "roi_score_min": None,
                    "roi_score_max": None,
                }
            )
        return self._add_drs_aliases(stats)

    def forward(
        self,
        visual_tokens: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        proprio: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        context: Optional[torch.Tensor] = None,
        mode: Optional[str] = None,
        scores: Optional[torch.Tensor] = None,
        encoder_attention: Optional[torch.Tensor] = None,
        wm_attention: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        mode = self.config.roi_mode if mode is None else mode
        start = time.perf_counter()
        if (not self.config.use_roi) or mode in NO_ROI_MODES:
            mask = self._full_mask(visual_tokens)
            stats = self._build_stats(mode or "none", visual_tokens, mask, None, 0.0)
            self.last_stats = stats
            return visual_tokens, mask, stats

        batch_shape = visual_tokens.shape[:-2]
        num_tokens = visual_tokens.shape[-2]

        if mode == "random":
            # Ablation: random sparse token set with the same budget as DRS.
            mask = random_token_mask(
                num_tokens,
                keep_ratio=self.config.roi_keep_ratio,
                topk=self.config.roi_topk,
                batch_shape=batch_shape,
                device=visual_tokens.device,
            )
        elif mode == "fixed_random":
            mask = fixed_random_token_mask(
                num_tokens,
                keep_ratio=self.config.roi_keep_ratio,
                topk=self.config.roi_topk,
                batch_shape=batch_shape,
                device=visual_tokens.device,
                seed=self.config.fixed_random_seed,
            )
        elif mode == "lhs":
            mask = lhs_token_mask(
                num_tokens,
                keep_ratio=self.config.roi_keep_ratio,
                topk=self.config.roi_topk,
                batch_shape=batch_shape,
                device=visual_tokens.device,
                seed=None,
            )
        elif mode in LEARNED_ROI_MODES:
            # DRS: distilled action-conditioned token relevance prediction.
            if scores is None:
                if self.roi_head is None:
                    raise ValueError(
                        f"roi_mode={mode} requires ActionAwareROIHead or scores"
                    )
                scores = self.roi_head(
                    visual_tokens,
                    action=action,
                    proprio=proprio,
                    timestep=timestep,
                    context=context,
                )
            select_scores = scores.detach() if self.config.roi_detach_score else scores
            # Select Top-K tokens according to distilled relevance scores.
            mask = topk_token_mask(
                select_scores,
                keep_ratio=self.config.roi_keep_ratio,
                topk=self.config.roi_topk,
            )
        elif mode == "attention_encoder":
            if encoder_attention is None:
                elapsed = time.perf_counter() - start
                stats = self._placeholder_stats(
                    mode,
                    visual_tokens,
                    elapsed,
                    "DINO wrapper currently returns patch tokens but does not expose CLS-to-patch attention.",
                )
                self.last_stats = stats
                return visual_tokens, self._full_mask(visual_tokens), stats
            mask = topk_token_mask(
                encoder_attention,
                keep_ratio=self.config.roi_keep_ratio,
                topk=self.config.roi_topk,
            )
            scores = encoder_attention
        elif mode == "attention_wm":
            if wm_attention is None:
                elapsed = time.perf_counter() - start
                stats = self._placeholder_stats(
                    mode,
                    visual_tokens,
                    elapsed,
                    "World-model attention selection needs a two-pass predictor call and attention hooks; this prototype leaves it as an explicit placeholder.",
                )
                self.last_stats = stats
                return visual_tokens, self._full_mask(visual_tokens), stats
            mask = topk_token_mask(
                wm_attention,
                keep_ratio=self.config.roi_keep_ratio,
                topk=self.config.roi_topk,
            )
            scores = wm_attention
        else:
            raise ValueError(f"Unsupported roi_mode: {mode}")

        masked_tokens = apply_token_mask_or_selection(
            visual_tokens,
            mask,
            mask_type=self.config.roi_mask_type,
        )
        elapsed = time.perf_counter() - start
        stats = self._build_stats(mode, visual_tokens, mask, scores, elapsed)
        self.last_stats = stats
        return masked_tokens, mask, stats


def normalize_token_scores(scores: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    scores = torch.clamp(scores, min=0)
    denom = scores.sum(dim=-1, keepdim=True).clamp_min(eps)
    return scores / denom


def grad_x_input_token_target(
    visual_tokens: torch.Tensor,
    loss: torch.Tensor,
    token_slice: Optional[Any] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    grad = torch.autograd.grad(
        loss,
        visual_tokens,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0]
    importance = (grad * visual_tokens).norm(dim=-1)
    if token_slice is not None:
        importance = importance[token_slice]
    return normalize_token_scores(importance, eps=eps)


def roi_distillation_loss(
    pred_scores: torch.Tensor,
    target_distribution: torch.Tensor,
    loss_type: str = "kl",
    temperature: float = 1.0,
) -> torch.Tensor:
    target_distribution = normalize_token_scores(target_distribution)
    temperature = float(temperature)
    if loss_type == "kl":
        pred_log_prob = F.log_softmax(pred_scores / temperature, dim=-1)
        return F.kl_div(pred_log_prob, target_distribution, reduction="batchmean")
    if loss_type == "mse":
        pred_prob = F.softmax(pred_scores / temperature, dim=-1)
        return F.mse_loss(pred_prob, target_distribution)
    if loss_type == "bce":
        target = target_distribution / target_distribution.max(dim=-1, keepdim=True).values.clamp_min(1e-8)
        return F.binary_cross_entropy_with_logits(pred_scores, target)
    if loss_type == "rank":
        pred_prob = F.softmax(pred_scores / temperature, dim=-1)
        target_order = torch.argsort(target_distribution, dim=-1, descending=True)
        pred_sorted = torch.gather(pred_prob, dim=-1, index=target_order)
        margin = pred_sorted[..., 1:] - pred_sorted[..., :-1]
        return F.relu(margin).mean()
    raise ValueError(f"Unsupported DRS distillation loss_type: {loss_type}")


ActionAwareDRSHead = ActionAwareROIHead
DRSSelector = ROISelector
drs_distillation_loss = roi_distillation_loss
