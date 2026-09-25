import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
from PIL import Image, ImageDraw, ImageFont

from models.dynamic_localizer import load_dynamic_localizer_checkpoint
from models.roi import LEARNED_ROI_MODES, ROISelector, topk_token_mask
from plan import load_model


PANEL_SIZE = 224
PANEL_GAP = 14
LABEL_H = 28


REGION_NAMES = {
    0: "background/static non-roi",
    1: "roi static",
    2: "dynamic non-roi",
    3: "dynamic roi core",
}

REGION_COLORS = {
    0: (42, 42, 42),
    1: (52, 211, 153),
    2: (59, 130, 246),
    3: (244, 63, 94),
}


def _checkpoint_path(model_path: Path, model_epoch: str) -> Path:
    name = "model_latest.pth" if str(model_epoch) == "latest" else f"model_{model_epoch}.pth"
    return model_path / "checkpoints" / name


def _parse_sample_indices(value: str) -> List[int]:
    items: List[int] = []
    for part in str(value).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = [p.strip() for p in part.split(":")]
            if len(pieces) not in (2, 3):
                raise ValueError(f"Bad sample range: {part}")
            start = int(pieces[0])
            stop = int(pieces[1])
            step = int(pieces[2]) if len(pieces) == 3 else 1
            items.extend(range(start, stop, step))
        else:
            items.append(int(part))
    if not items:
        raise ValueError("--sample-indices produced an empty list")
    return items


def _patch_side(num_tokens: int) -> int:
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"Only square token grids are supported, got {num_tokens} tokens")
    return side


def _normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    scores = scores.detach().float()
    flat = scores.reshape(-1, scores.shape[-1])
    lo = flat.min(dim=-1, keepdim=True).values
    hi = flat.max(dim=-1, keepdim=True).values
    norm = (flat - lo) / (hi - lo).clamp_min(1e-8)
    return norm.reshape_as(scores)


def _tensor_image_to_pil(image: torch.Tensor, size: int = PANEL_SIZE) -> Image.Image:
    img = image.detach().cpu().float()
    if img.ndim != 3:
        raise ValueError(f"Expected image tensor with 3 dims, got {tuple(img.shape)}")
    if img.shape[0] in (1, 3):
        chw = img
    elif img.shape[-1] in (1, 3):
        chw = img.permute(2, 0, 1)
    else:
        raise ValueError(f"Cannot infer channel dimension from {tuple(img.shape)}")
    if chw.max() > 2:
        chw = chw / 255.0
    chw = chw.clamp(0, 1)
    arr = (chw.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr).convert("RGB").resize((size, size), Image.BILINEAR)


def _score_grid(score: torch.Tensor) -> np.ndarray:
    score = score.detach().cpu().float()
    if score.ndim > 1:
        score = score.reshape(-1, score.shape[-1])[0]
    side = _patch_side(int(score.shape[-1]))
    return score.reshape(side, side).numpy()


def _mask_grid(mask: torch.Tensor) -> np.ndarray:
    mask = mask.detach().cpu().bool()
    if mask.ndim > 1:
        mask = mask.reshape(-1, mask.shape[-1])[0]
    side = _patch_side(int(mask.shape[-1]))
    return mask.reshape(side, side).numpy()


def _heat_color(value: float) -> Tuple[int, int, int]:
    value = float(np.clip(value, 0.0, 1.0))
    if value < 0.5:
        t = value / 0.5
        r = int(38 + 31 * t)
        g = int(70 + 160 * t)
        b = int(180 - 120 * t)
    else:
        t = (value - 0.5) / 0.5
        r = int(69 + 186 * t)
        g = int(230 - 70 * t)
        b = int(60 - 40 * t)
    return r, g, b


def _draw_grid_lines(img: Image.Image, side: int, color=(255, 255, 255, 92)) -> Image.Image:
    out = img.convert("RGB")
    draw = ImageDraw.Draw(out, "RGBA")
    patch = PANEL_SIZE // side
    for i in range(side + 1):
        xy = i * patch
        draw.line([(xy, 0), (xy, PANEL_SIZE)], fill=color, width=1)
        draw.line([(0, xy), (PANEL_SIZE, xy)], fill=color, width=1)
    return out


def _draw_score_heatmap(score_grid: np.ndarray, mask_grid: Optional[np.ndarray] = None) -> Image.Image:
    side = int(score_grid.shape[0])
    patch = PANEL_SIZE // side
    arr = np.zeros((PANEL_SIZE, PANEL_SIZE, 3), dtype=np.uint8)
    for r in range(side):
        for c in range(side):
            arr[r * patch : (r + 1) * patch, c * patch : (c + 1) * patch] = _heat_color(
                float(score_grid[r, c])
            )
    img = Image.fromarray(arr).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    if mask_grid is not None:
        for r in range(side):
            for c in range(side):
                if bool(mask_grid[r, c]):
                    x0, y0 = c * patch, r * patch
                    draw.rectangle(
                        [x0, y0, x0 + patch - 1, y0 + patch - 1],
                        outline=(255, 255, 255, 230),
                        width=2,
                    )
    return _draw_grid_lines(img, side, color=(255, 255, 255, 70))


def _draw_region_map(region_grid: np.ndarray) -> Image.Image:
    side = int(region_grid.shape[0])
    patch = PANEL_SIZE // side
    arr = np.zeros((PANEL_SIZE, PANEL_SIZE, 3), dtype=np.uint8)
    for r in range(side):
        for c in range(side):
            arr[r * patch : (r + 1) * patch, c * patch : (c + 1) * patch] = REGION_COLORS[
                int(region_grid[r, c])
            ]
    return _draw_grid_lines(Image.fromarray(arr).convert("RGB"), side, color=(255, 255, 255, 82))


def _draw_region_overlay(image: Image.Image, region_grid: np.ndarray) -> Image.Image:
    out = image.convert("RGB").resize((PANEL_SIZE, PANEL_SIZE), Image.BILINEAR)
    pixels = np.asarray(out).astype(np.float32)
    side = int(region_grid.shape[0])
    patch = PANEL_SIZE // side
    for r in range(side):
        for c in range(side):
            color = np.array(REGION_COLORS[int(region_grid[r, c])], dtype=np.float32)
            alpha = 0.18 if int(region_grid[r, c]) == 0 else 0.46
            y0, y1 = r * patch, (r + 1) * patch
            x0, x1 = c * patch, (c + 1) * patch
            pixels[y0:y1, x0:x1] = pixels[y0:y1, x0:x1] * (1.0 - alpha) + color * alpha
    return _draw_grid_lines(Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8)), side)


def _draw_legend() -> Image.Image:
    img = Image.new("RGB", (PANEL_SIZE, PANEL_SIZE), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    y = 22
    for idx in [3, 1, 2, 0]:
        color = REGION_COLORS[idx]
        draw.rectangle([18, y, 42, y + 24], fill=color, outline=(20, 20, 20))
        draw.text((52, y + 5), REGION_NAMES[idx], fill=(20, 20, 20), font=font)
        y += 42
    draw.text((18, y + 4), "region = ROI mask x dynamic mask", fill=(20, 20, 20), font=font)
    return img


def _make_panel_grid(
    panels: List[Tuple[str, Image.Image]],
    output_path: Path,
    title: str,
    cols: int = 4,
) -> None:
    font = ImageFont.load_default()
    rows = int(math.ceil(len(panels) / cols))
    width = cols * PANEL_SIZE + (cols + 1) * PANEL_GAP
    height = 44 + rows * (PANEL_SIZE + LABEL_H + PANEL_GAP) + PANEL_GAP
    canvas = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    draw.text((PANEL_GAP, 14), title, fill=(20, 20, 20), font=font)
    for idx, (label, image) in enumerate(panels):
        row = idx // cols
        col = idx % cols
        x = PANEL_GAP + col * (PANEL_SIZE + PANEL_GAP)
        y = 42 + row * (PANEL_SIZE + LABEL_H + PANEL_GAP)
        draw.text((x, y), label, fill=(20, 20, 20), font=font)
        canvas.paste(image.convert("RGB").resize((PANEL_SIZE, PANEL_SIZE)), (x, y + LABEL_H))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def _corr(a: np.ndarray, b: np.ndarray, rank: bool = False) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    if rank:
        a = _rankdata(a)
        b = _rankdata(b)
    if float(np.std(a)) < 1e-12 or float(np.std(b)) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _deterministic_force_min_mask(
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


def _pixel_delta_score(obs_raw: Dict[str, torch.Tensor], t_a: int, t_b: int, side: int) -> torch.Tensor:
    img_a = obs_raw["visual"][t_a].detach().float()
    img_b = obs_raw["visual"][t_b].detach().float()
    if img_a.shape[0] not in (1, 3):
        img_a = img_a.permute(2, 0, 1)
        img_b = img_b.permute(2, 0, 1)
    if img_a.max() > 2:
        img_a = img_a / 255.0
        img_b = img_b / 255.0
    delta = (img_b - img_a).pow(2).mean(dim=0, keepdim=True).unsqueeze(0)
    pooled = F.adaptive_avg_pool2d(delta, (side, side))[0, 0]
    return pooled.reshape(1, side * side)


def _ddpwm_pixel_delta_score_and_mask(
    obs_raw: Dict[str, torch.Tensor],
    model,
    t_a: int,
    t_b: int,
    side: int,
    threshold: float,
    partition_precision: int,
    min_dynamic_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    sqrt_precision = int(math.sqrt(partition_precision))
    if sqrt_precision * sqrt_precision != int(partition_precision):
        raise ValueError(f"partition_precision must be a square number, got {partition_precision}")

    img_a = obs_raw["visual"][t_a].detach().float()
    img_b = obs_raw["visual"][t_b].detach().float()
    if img_a.shape[0] not in (1, 3):
        img_a = img_a.permute(2, 0, 1)
        img_b = img_b.permute(2, 0, 1)
    if img_a.max() > 2:
        img_a = img_a / 255.0
        img_b = img_b / 255.0

    device = next(model.parameters()).device
    images = torch.stack([img_a, img_b], dim=0).to(device)
    transform = getattr(model, "encoder_transform", None)
    if transform is not None:
        images = transform(images)
    current = images[0:1]
    reference = images[1:2]

    pixel_diff = reference - current
    pixel_norms_sq = torch.sum(pixel_diff.pow(2), dim=1, keepdim=True)
    patch_size_h = current.shape[2] // side // sqrt_precision
    patch_size_w = current.shape[3] // side // sqrt_precision
    patch_size_h = max(int(patch_size_h), 1)
    patch_size_w = max(int(patch_size_w), 1)
    patch_norms_sq = F.avg_pool2d(
        pixel_norms_sq,
        kernel_size=(patch_size_h, patch_size_w),
        stride=(patch_size_h, patch_size_w),
    )
    norms = torch.sqrt(patch_norms_sq.clamp_min(0.0))
    fine_h = side * sqrt_precision
    fine_w = side * sqrt_precision
    if norms.shape[-2:] != (fine_h, fine_w):
        norms = F.interpolate(norms, size=(fine_h, fine_w), mode="nearest")

    norms = norms[:, 0].reshape(1, side, sqrt_precision, side, sqrt_precision)
    norms = norms.permute(0, 1, 3, 2, 4).reshape(1, side * side, partition_precision)
    score = norms.max(dim=-1).values
    mask = (norms > float(threshold)).any(dim=-1)
    mask = _deterministic_force_min_mask(mask, score, min_dynamic_tokens)
    return _normalize_scores(score), mask


def _ddpwm_feature_delta_score_and_mask(
    z_obs: Dict[str, torch.Tensor],
    t_a: int,
    t_b: int,
    threshold: float,
    d_feature: int,
    min_dynamic_tokens: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    delta_z = z_obs["visual"][:, t_b, :, :d_feature] - z_obs["visual"][:, t_a, :, :d_feature]
    score = torch.norm(delta_z, p=2, dim=-1)
    mask = score > float(threshold)
    mask = _deterministic_force_min_mask(mask, score, min_dynamic_tokens)
    return _normalize_scores(score), mask


def _history_window(x: torch.Tensor, t_cur: int, num_hist: int) -> torch.Tensor:
    end = int(t_cur) + 1
    start = end - int(num_hist)
    if start >= 0:
        return x[:, start:end]
    prefix = x[:, :1].expand(-1, -start, *x.shape[2:])
    return torch.cat([prefix, x[:, :end]], dim=1)


def _dynamic_score(
    mode: str,
    z_obs: Dict[str, torch.Tensor],
    act_emb: torch.Tensor,
    obs_raw: Dict[str, torch.Tensor],
    model,
    args,
    t_cur: int,
    side: int,
    dynamic_localizer=None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
    if mode == "localizer":
        if dynamic_localizer is None:
            raise ValueError("--dynamic-localizer-checkpoint is required for --dyn-score-mode localizer")
        t_ref = min(t_cur + 1, int(z_obs["visual"].shape[1]) - 1)
        num_hist = int(getattr(dynamic_localizer, "num_hist", getattr(model, "num_hist", 1)))
        visual_history = _history_window(z_obs["visual"], t_cur, num_hist)
        action_history = _history_window(act_emb, t_cur, num_hist)
        proprio_history = _history_window(z_obs["proprio"], t_cur, num_hist)
        score, mask = dynamic_localizer.predict_scores_and_mask(
            visual_history=visual_history,
            action_history=action_history,
            proprio_history=proprio_history,
            threshold=args.dynamic_localizer_threshold,
            min_tokens=args.dynamic_localizer_min_tokens,
        )
        return _normalize_scores(score), mask, t_ref

    if mode.endswith("_next"):
        t_ref = min(t_cur + 1, int(z_obs["visual"].shape[1]) - 1)
    elif mode.endswith("_prev"):
        t_ref = max(t_cur - 1, 0)
    else:
        raise ValueError(f"Unsupported dynamic score mode: {mode}")

    if mode.startswith("feature_delta"):
        score = (z_obs["visual"][:, t_ref] - z_obs["visual"][:, t_cur]).pow(2).mean(dim=-1)
        mask = None
    elif mode.startswith("pixel_delta"):
        score = _pixel_delta_score(obs_raw, t_cur, t_ref, side).to(z_obs["visual"].device)
        mask = None
    elif mode.startswith("ddpwm_pixel"):
        score, mask = _ddpwm_pixel_delta_score_and_mask(
            obs_raw=obs_raw,
            model=model,
            t_a=t_cur,
            t_b=t_ref,
            side=side,
            threshold=args.ddpwm_pixel_threshold,
            partition_precision=args.ddpwm_partition_precision,
            min_dynamic_tokens=args.ddpwm_min_dynamic_tokens,
        )
    elif mode.startswith("ddpwm_feature"):
        score, mask = _ddpwm_feature_delta_score_and_mask(
            z_obs=z_obs,
            t_a=t_cur,
            t_b=t_ref,
            threshold=args.ddpwm_feature_threshold,
            d_feature=args.ddpwm_feature_dim,
            min_dynamic_tokens=args.ddpwm_min_dynamic_tokens,
        )
    else:
        raise ValueError(f"Unsupported dynamic score mode: {mode}")
    return _normalize_scores(score), mask, t_ref


def _load_model_and_dataset(args):
    model_path = Path(args.ckpt_base_path) / "outputs" / args.model_name
    train_cfg = OmegaConf.load(model_path / "hydra.yaml")
    if args.data_path is not None:
        with open_dict(train_cfg):
            train_cfg.env.dataset.data_path = args.data_path

    device = torch.device(args.device)
    model = load_model(
        _checkpoint_path(model_path, args.model_epoch),
        train_cfg,
        train_cfg.num_action_repeat,
        device=device,
        roi_runtime_cfg={
            "use_roi": False,
            "roi_mode": "none",
            "load_decoder": False,
            "profile_wm_timing": False,
        },
    )
    model.eval()

    datasets, _ = hydra.utils.call(
        train_cfg.env.dataset,
        num_hist=train_cfg.num_hist,
        num_pred=train_cfg.num_pred,
        frameskip=train_cfg.frameskip,
    )
    return model, train_cfg, datasets


def _build_roi_selector(args, token_dim: int, action_dim: int, proprio_dim: int, device: torch.device):
    cfg = {
        "use_roi": True,
        "roi_mode": args.roi_mode,
        "roi_keep_ratio": args.roi_keep_ratio,
        "roi_topk": args.roi_topk,
        "roi_action_aware": True,
        "roi_mask_type": "mask",
        "roi_hidden_dim": args.roi_hidden_dim,
        "roi_detach_score": True,
    }
    selector = ROISelector(
        config=cfg,
        token_dim=token_dim,
        action_dim=action_dim,
        proprio_dim=proprio_dim,
    ).to(device)
    selector.eval()
    if args.roi_mode in LEARNED_ROI_MODES:
        if not args.roi_head_checkpoint:
            raise ValueError(
                f"--roi-head-checkpoint is required for learned ROI mode {args.roi_mode}"
            )
        selector.load_checkpoint(args.roi_head_checkpoint, map_location=device)
    return selector


def _region_stats(region_grid: np.ndarray, roi_grid: np.ndarray, dyn_grid: np.ndarray) -> Dict:
    total = int(region_grid.size)
    stats = {
        "total_tokens": total,
        "roi_tokens": int(roi_grid.sum()),
        "dynamic_tokens": int(dyn_grid.sum()),
    }
    for idx, name in REGION_NAMES.items():
        count = int((region_grid == idx).sum())
        stats[name.replace(" ", "_").replace("/", "_")] = {
            "count": count,
            "ratio": count / max(total, 1),
        }
    return stats


def _visualize_one_sample(
    args,
    model,
    dataset,
    sample_index: int,
    selector_cache: Dict[Tuple[int, int, int], ROISelector],
    summary_rows: List[Dict],
    dynamic_localizer=None,
) -> Path:
    device = next(model.parameters()).device
    obs_raw, act, state = dataset[sample_index]
    obs_b = {key: value.unsqueeze(0).to(device) for key, value in obs_raw.items()}
    act_b = act.unsqueeze(0).to(device)

    with torch.no_grad():
        z_obs = model.encode_obs(obs_b)
        act_emb = model.encode_act(act_b)

    default_t = max(0, min(int(getattr(model, "num_hist", 1)) - 1, int(z_obs["visual"].shape[1]) - 1))
    t_cur = default_t if args.timestep is None else int(args.timestep)
    t_cur = max(0, min(t_cur, int(z_obs["visual"].shape[1]) - 1))

    visual_tokens = z_obs["visual"][:, t_cur]
    proprio = z_obs["proprio"][:, t_cur] if "proprio" in z_obs else None
    action_t = act_emb[:, min(t_cur, int(act_emb.shape[1]) - 1)]
    side = _patch_side(int(visual_tokens.shape[-2]))

    key = (int(visual_tokens.shape[-1]), int(action_t.shape[-1]), int(proprio.shape[-1]) if proprio is not None else 0)
    if key not in selector_cache:
        selector_cache[key] = _build_roi_selector(
            args,
            token_dim=key[0],
            action_dim=key[1],
            proprio_dim=key[2],
            device=device,
        )
    selector = selector_cache[key]

    with torch.no_grad():
        if args.roi_mode in LEARNED_ROI_MODES:
            roi_score = selector.roi_head(visual_tokens, action=action_t, proprio=proprio)
        else:
            _, _, roi_stats = selector(
                visual_tokens,
                action=action_t,
                proprio=proprio,
                mode=args.roi_mode,
            )
            roi_score = roi_stats.get("scores")
            if roi_score is None:
                roi_score = torch.ones(visual_tokens.shape[:-1], device=device)
        roi_score = _normalize_scores(roi_score)
        roi_mask = topk_token_mask(roi_score, keep_ratio=args.roi_keep_ratio, topk=args.roi_topk)
        dyn_score, ddpwm_mask, t_ref = _dynamic_score(
            args.dyn_score_mode,
            z_obs,
            act_emb,
            obs_raw,
            model,
            args,
            t_cur,
            side,
            dynamic_localizer=dynamic_localizer,
        )
        if args.dyn_mask_mode == "threshold":
            if ddpwm_mask is None:
                raise ValueError(
                    "--dyn-mask-mode threshold is only supported for ddpwm_* dynamic score modes"
                )
            dyn_mask = ddpwm_mask
        elif args.dyn_mask_mode == "topk":
            dyn_mask = topk_token_mask(dyn_score, keep_ratio=args.dyn_keep_ratio, topk=args.dyn_topk)
        elif args.dyn_mask_mode == "auto":
            if ddpwm_mask is not None:
                dyn_mask = ddpwm_mask
            else:
                dyn_mask = topk_token_mask(dyn_score, keep_ratio=args.dyn_keep_ratio, topk=args.dyn_topk)
        else:
            raise ValueError(f"Unsupported dyn_mask_mode: {args.dyn_mask_mode}")

    roi_grid = _mask_grid(roi_mask)
    dyn_grid = _mask_grid(dyn_mask)
    roi_score_grid = _score_grid(roi_score)
    dyn_score_grid = _score_grid(dyn_score)

    region_grid = np.zeros_like(roi_grid, dtype=np.int64)
    region_grid[np.logical_and(roi_grid, np.logical_not(dyn_grid))] = 1
    region_grid[np.logical_and(np.logical_not(roi_grid), dyn_grid)] = 2
    region_grid[np.logical_and(roi_grid, dyn_grid)] = 3

    cur_img = _tensor_image_to_pil(obs_raw["visual"][t_cur])
    ref_img = _tensor_image_to_pil(obs_raw["visual"][t_ref])
    panels = [
        (f"current frame t={t_cur}", cur_img),
        (f"reference frame t={t_ref}", ref_img),
        ("roi_score + top-k", _draw_score_heatmap(roi_score_grid, roi_grid)),
        ("dyn_score + top-k", _draw_score_heatmap(dyn_score_grid, dyn_grid)),
        ("four regions", _draw_region_map(region_grid)),
        ("regions on image", _draw_region_overlay(cur_img, region_grid)),
        ("legend", _draw_legend()),
    ]

    stats = _region_stats(region_grid, roi_grid, dyn_grid)
    stats.update(
        {
            "sample_index": int(sample_index),
            "timestep": int(t_cur),
            "reference_timestep": int(t_ref),
            "roi_dyn_pearson": _corr(roi_score_grid, dyn_score_grid, rank=False),
            "roi_dyn_spearman": _corr(roi_score_grid, dyn_score_grid, rank=True),
            "roi_mode": args.roi_mode,
            "dyn_score_mode": args.dyn_score_mode,
            "roi_keep_ratio": args.roi_keep_ratio,
            "dyn_keep_ratio": args.dyn_keep_ratio,
            "dynamic_localizer_checkpoint": args.dynamic_localizer_checkpoint,
        }
    )
    summary_rows.append(stats)

    output_dir = Path(args.output_dir)
    stem = (
        f"roi_dyn_regions_sample{sample_index}_t{t_cur}_"
        f"roi{args.roi_keep_ratio:g}_dyn{args.dyn_keep_ratio:g}"
    )
    png_path = output_dir / f"{stem}.png"
    npz_path = output_dir / f"{stem}.npz"
    _make_panel_grid(
        panels,
        png_path,
        title=(
            "ROI x dynamic region visualization "
            f"(sample={sample_index}, roi={args.roi_mode}, dyn={args.dyn_score_mode})"
        ),
        cols=4,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        npz_path,
        roi_score=roi_score_grid,
        dyn_score=dyn_score_grid,
        roi_mask=roi_grid,
        dyn_mask=dyn_grid,
        region=region_grid,
    )
    return png_path


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualize ROI score, dynamic score, and their four semantic token regions."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--ckpt-base-path", required=True)
    parser.add_argument("--model-epoch", default="latest")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output-dir", default="roi_dynamic_regions_viz")
    parser.add_argument("--split", default="valid", choices=["train", "valid"])
    parser.add_argument("--sample-indices", default="0")
    parser.add_argument("--timestep", type=int, default=None)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--roi-mode", default="distilled_topk")
    parser.add_argument("--roi-keep-ratio", type=float, default=0.5)
    parser.add_argument("--roi-topk", type=int, default=None)
    parser.add_argument("--roi-hidden-dim", type=int, default=128)
    parser.add_argument("--roi-head-checkpoint", default=None)

    parser.add_argument(
        "--dyn-score-mode",
        default="ddpwm_pixel_prev",
        choices=[
            "feature_delta_next",
            "feature_delta_prev",
            "pixel_delta_next",
            "pixel_delta_prev",
            "ddpwm_pixel_next",
            "ddpwm_pixel_prev",
            "ddpwm_feature_next",
            "ddpwm_feature_prev",
            "localizer",
        ],
    )
    parser.add_argument("--dyn-keep-ratio", type=float, default=0.5)
    parser.add_argument("--dyn-topk", type=int, default=None)
    parser.add_argument("--dyn-mask-mode", default="auto", choices=["auto", "topk", "threshold"])
    parser.add_argument("--ddpwm-pixel-threshold", type=float, default=0.1)
    parser.add_argument("--ddpwm-feature-threshold", type=float, default=45.0)
    parser.add_argument("--ddpwm-feature-dim", type=int, default=384)
    parser.add_argument("--ddpwm-partition-precision", type=int, default=4)
    parser.add_argument("--ddpwm-min-dynamic-tokens", type=int, default=32)
    parser.add_argument("--dynamic-localizer-checkpoint", default=None)
    parser.add_argument("--dynamic-localizer-threshold", type=float, default=0.5)
    parser.add_argument("--dynamic-localizer-min-tokens", type=int, default=32)
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    model, _, datasets = _load_model_and_dataset(args)
    dataset = datasets[args.split]
    summary_rows: List[Dict] = []
    selector_cache: Dict[Tuple[int, int, int], ROISelector] = {}
    dynamic_localizer = None
    if args.dyn_score_mode == "localizer":
        if args.dynamic_localizer_checkpoint is None:
            raise ValueError("--dynamic-localizer-checkpoint is required for --dyn-score-mode localizer")
        dynamic_localizer, _ = load_dynamic_localizer_checkpoint(
            args.dynamic_localizer_checkpoint,
            map_location=torch.device(args.device),
        )
        dynamic_localizer.to(next(model.parameters()).device)
        dynamic_localizer.eval()

    for sample_index in _parse_sample_indices(args.sample_indices):
        png_path = _visualize_one_sample(
            args,
            model,
            dataset,
            sample_index,
            selector_cache,
            summary_rows,
            dynamic_localizer=dynamic_localizer,
        )
        print(f"Saved {png_path}")

    summary_path = Path(args.output_dir) / "summary.jsonl"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        for row in summary_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Saved {summary_path}")


if __name__ == "__main__":
    main()
