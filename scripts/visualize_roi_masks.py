import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

from models.roi import (
    LEARNED_ROI_MODES,
    ROISelector,
    fixed_random_token_mask,
    lhs_token_mask,
    random_token_mask,
)


PANEL_SIZE = 224
PATCH_SIDE = 14


def _parse_modes(value: str) -> List[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _resize_chw_image(image: torch.Tensor, size: int = PANEL_SIZE) -> np.ndarray:
    if image.ndim == 3 and image.shape[0] in (1, 3):
        chw = image.float()
    elif image.ndim == 3 and image.shape[-1] in (1, 3):
        chw = image.permute(2, 0, 1).float()
    else:
        raise ValueError(f"Expected CHW or HWC image tensor, got {tuple(image.shape)}")
    if chw.max() > 2:
        chw = chw / 255.0
    chw = chw.clamp(0, 1).unsqueeze(0)
    resized = F.interpolate(chw, size=(size, size), mode="bilinear", align_corners=False)
    hwc = resized[0].permute(1, 2, 0).detach().cpu().numpy()
    return (hwc * 255).round().astype(np.uint8)


def _load_raw_pointmaze_frame(data_path: Path, episode: int, frame: int) -> np.ndarray:
    obs_path = data_path / "obses" / f"episode_{episode:03d}.pth"
    if not obs_path.exists():
        raise FileNotFoundError(f"PointMaze frame file not found: {obs_path}")
    frames = torch.load(obs_path, map_location="cpu")
    frame = min(max(int(frame), 0), int(frames.shape[0]) - 1)
    return _resize_chw_image(frames[frame])


def _mask_to_numpy(mask: torch.Tensor) -> np.ndarray:
    mask = mask.detach().cpu().bool()
    if mask.ndim > 1:
        mask = mask.reshape(-1, mask.shape[-1])[0]
    num_tokens = int(mask.shape[-1])
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"Only square patch grids are supported, got {num_tokens} tokens")
    return mask.reshape(side, side).numpy()


def _scores_to_numpy(scores: Optional[torch.Tensor]) -> Optional[np.ndarray]:
    if scores is None:
        return None
    scores = scores.detach().cpu().float()
    if scores.ndim > 1:
        scores = scores.reshape(-1, scores.shape[-1])[0]
    side = int(math.sqrt(scores.shape[-1]))
    if side * side != scores.shape[-1]:
        return None
    values = scores.reshape(side, side).numpy()
    values = values - values.min()
    denom = max(float(values.max()), 1e-8)
    return values / denom


def _draw_mask_overlay(image: np.ndarray, mask_grid: np.ndarray) -> Image.Image:
    out = Image.fromarray(image).convert("RGB")
    patch = out.width // mask_grid.shape[1]
    pixels = np.asarray(out).copy()
    for r in range(mask_grid.shape[0]):
        for c in range(mask_grid.shape[1]):
            y0, y1 = r * patch, (r + 1) * patch
            x0, x1 = c * patch, (c + 1) * patch
            if not bool(mask_grid[r, c]):
                pixels[y0:y1, x0:x1] = (pixels[y0:y1, x0:x1] * 0.18).astype(np.uint8)
            else:
                pixels[y0:y1, x0:x1] = (
                    pixels[y0:y1, x0:x1] * 0.72
                    + np.array([55, 220, 120], dtype=np.float32) * 0.28
                ).astype(np.uint8)
    out = Image.fromarray(pixels).convert("RGB")
    draw = ImageDraw.Draw(out, "RGBA")
    for r in range(mask_grid.shape[0]):
        for c in range(mask_grid.shape[1]):
            x0, y0 = c * patch, r * patch
            x1, y1 = (c + 1) * patch, (r + 1) * patch
            if bool(mask_grid[r, c]):
                draw.rectangle([x0, y0, x1 - 1, y1 - 1], outline=(60, 255, 130, 210), width=2)
    return out


def _draw_mask_grid(mask_grid: np.ndarray) -> Image.Image:
    side = mask_grid.shape[0]
    patch = PANEL_SIZE // side
    arr = np.zeros((PANEL_SIZE, PANEL_SIZE, 3), dtype=np.uint8)
    for r in range(side):
        for c in range(side):
            color = np.array([245, 245, 245], dtype=np.uint8) if mask_grid[r, c] else np.array([15, 15, 15], dtype=np.uint8)
            arr[r * patch : (r + 1) * patch, c * patch : (c + 1) * patch] = color
    img = Image.fromarray(arr).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    for pos in range(side + 1):
        xy = pos * patch
        draw.line([(xy, 0), (xy, PANEL_SIZE)], fill=(150, 150, 150, 80), width=1)
        draw.line([(0, xy), (PANEL_SIZE, xy)], fill=(150, 150, 150, 80), width=1)
    return img


def _draw_score_heatmap(scores_grid: Optional[np.ndarray], mask_grid: np.ndarray) -> Image.Image:
    if scores_grid is None:
        return _draw_mask_grid(mask_grid)
    side = scores_grid.shape[0]
    patch = PANEL_SIZE // side
    arr = np.zeros((PANEL_SIZE, PANEL_SIZE, 3), dtype=np.uint8)
    for r in range(side):
        for c in range(side):
            value = float(scores_grid[r, c])
            color = np.array(
                [
                    int(30 + 225 * value),
                    int(35 + 120 * (1.0 - abs(value - 0.5) * 2.0)),
                    int(160 * (1.0 - value)),
                ],
                dtype=np.uint8,
            )
            arr[r * patch : (r + 1) * patch, c * patch : (c + 1) * patch] = color
    img = Image.fromarray(arr).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    for r in range(side):
        for c in range(side):
            if mask_grid[r, c]:
                x0, y0 = c * patch, r * patch
                draw.rectangle([x0, y0, x0 + patch - 1, y0 + patch - 1], outline=(255, 255, 255, 230), width=2)
    return img


def _draw_title(draw: ImageDraw.ImageDraw, xy, text: str, font):
    draw.text(xy, text, fill=(20, 20, 20), font=font)


def _make_grid(rows: List[Dict], output_path: Path, title: str):
    font = ImageFont.load_default()
    label_h = 28
    row_h = PANEL_SIZE + label_h + 14
    gap = 18
    cols = 3
    width = cols * PANEL_SIZE + (cols + 1) * gap
    height = 42 + len(rows) * row_h + 8
    canvas = Image.new("RGB", (width, height), (250, 250, 250))
    draw = ImageDraw.Draw(canvas)
    _draw_title(draw, (gap, 12), title, font)
    headers = ["original", "mask / score", "masked input view"]
    for col, header in enumerate(headers):
        x = gap + col * (PANEL_SIZE + gap)
        _draw_title(draw, (x, 34), header, font)

    y = 58
    for row in rows:
        x0 = gap
        canvas.paste(Image.fromarray(row["image"]).convert("RGB"), (x0, y + label_h))
        x1 = gap + PANEL_SIZE + gap
        canvas.paste(row["score_panel"], (x1, y + label_h))
        x2 = gap + 2 * (PANEL_SIZE + gap)
        canvas.paste(row["overlay"], (x2, y + label_h))
        draw.text((gap, y), row["label"], fill=(20, 20, 20), font=font)
        kept = int(row["mask_grid"].sum())
        total = int(row["mask_grid"].size)
        draw.text((x2, y), f"kept {kept}/{total} ({kept / total:.2f})", fill=(20, 20, 20), font=font)
        y += row_h

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _slugify(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in ("-", "_", ".") else "_" for ch in value).strip("_")


def _save_single_panels(rows: List[Dict], output_dir: Path, prefix: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    for row_idx, row in enumerate(rows):
        label = _slugify(row["label"]) or f"row{row_idx}"
        stem = f"{prefix}_{row_idx:02d}_{label}"
        Image.fromarray(row["image"]).convert("RGB").save(output_dir / f"{stem}_input.png")
        if row.get("goal_image") is not None:
            Image.fromarray(row["goal_image"]).convert("RGB").save(output_dir / f"{stem}_goal.png")
        row["score_panel"].save(output_dir / f"{stem}_score.png")
        row["overlay"].save(output_dir / f"{stem}_overlay.png")


def _make_selector_mask(
    mode: str,
    keep_ratio: float,
    seed: int,
    num_tokens: int,
    sample_idx: int,
) -> torch.Tensor:
    batch_shape = (1,)
    if mode == "random":
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed + sample_idx)
        return random_token_mask(num_tokens, keep_ratio=keep_ratio, batch_shape=batch_shape, generator=generator)
    if mode == "fixed_random":
        return fixed_random_token_mask(num_tokens, keep_ratio=keep_ratio, batch_shape=batch_shape, seed=seed)
    if mode == "lhs":
        return lhs_token_mask(num_tokens, keep_ratio=keep_ratio, batch_shape=batch_shape, seed=seed + sample_idx)
    raise ValueError(f"Mode {mode} needs model features; provide --model-name and --ckpt-base-path.")


def _visualize_without_model(args) -> List[Dict]:
    data_path = Path(args.data_path or (Path(os.environ.get("DATASET_DIR", "data")) / "point_maze"))
    image = _load_raw_pointmaze_frame(data_path, args.episode, args.frame)
    rows = []
    for mode in _parse_modes(args.modes):
        repeats = args.num_samples if mode == "random" else 1
        for sample_idx in range(repeats):
            mask = _make_selector_mask(mode, args.keep_ratio, args.seed, PATCH_SIDE * PATCH_SIDE, sample_idx)
            mask_grid = _mask_to_numpy(mask)
            rows.append(
                {
                    "label": f"{mode} sample={sample_idx}",
                    "image": image,
                    "mask_grid": mask_grid,
                    "score_panel": _draw_mask_grid(mask_grid),
                    "overlay": _draw_mask_overlay(image, mask_grid),
                }
            )
    return rows


def _load_model_sample(args):
    import hydra
    from omegaconf import OmegaConf, open_dict

    from plan import load_model
    from utils import move_to_device

    model_path = Path(args.ckpt_base_path) / "outputs" / args.model_name
    train_cfg = OmegaConf.load(model_path / "hydra.yaml")
    if args.data_path is not None:
        with open_dict(train_cfg):
            train_cfg.env.dataset.data_path = args.data_path

    ckpt_name = "model_latest.pth" if str(args.model_epoch) == "latest" else f"model_{args.model_epoch}.pth"
    model_ckpt = model_path / "checkpoints" / ckpt_name
    device = torch.device(args.device)
    model = load_model(
        model_ckpt,
        train_cfg,
        train_cfg.num_action_repeat,
        device=device,
        roi_runtime_cfg={"use_roi": False, "roi_mode": "none", "load_decoder": False},
    )
    model.eval()

    datasets, _ = hydra.utils.call(
        train_cfg.env.dataset,
        num_hist=train_cfg.num_hist,
        num_pred=train_cfg.num_pred,
        frameskip=train_cfg.frameskip,
    )
    sample = datasets[args.split][args.sample_index]
    obs, act, _ = sample
    num_obs_frames = int(obs["visual"].shape[0])
    timestep = args.timestep
    if timestep < 0:
        timestep = num_obs_frames + timestep
    timestep = min(max(int(timestep), 0), num_obs_frames - 1)
    goal_timestep = args.goal_timestep
    if goal_timestep < 0:
        goal_timestep = num_obs_frames + goal_timestep
    goal_timestep = min(max(int(goal_timestep), 0), num_obs_frames - 1)
    image = _resize_chw_image(obs["visual"][timestep])
    goal_image = _resize_chw_image(obs["visual"][goal_timestep])
    obs = {k: v.unsqueeze(0) for k, v in obs.items()}
    act = act.unsqueeze(0)
    obs = move_to_device(obs, device)
    act = act.to(device)
    with torch.no_grad():
        z_dct = model.encode_obs(obs)
        act_emb = model.encode_act(act)
    timestep = min(timestep, int(z_dct["visual"].shape[1]) - 1)
    return model, image, goal_image, z_dct, act_emb, timestep, goal_timestep


def _visualize_with_model(args) -> List[Dict]:
    model, image, goal_image, z_dct, act_emb, timestep, goal_timestep = _load_model_sample(args)
    visual_tokens = z_dct["visual"][:, timestep]
    proprio = z_dct["proprio"][:, timestep]
    action = act_emb[:, timestep]
    rows = []
    for mode in _parse_modes(args.modes):
        repeats = args.num_samples if mode == "random" else 1
        for sample_idx in range(repeats):
            cfg = {
                "use_roi": True,
                "roi_mode": mode,
                "roi_keep_ratio": args.keep_ratio,
                "roi_topk": args.roi_topk,
                "roi_action_aware": True,
                "roi_mask_type": "mask",
                "roi_detach_score": True,
                "roi_hidden_dim": args.roi_hidden_dim,
                "fixed_random_seed": args.seed,
            }
            selector = ROISelector(
                config=cfg,
                token_dim=visual_tokens.shape[-1],
                action_dim=action.shape[-1],
                proprio_dim=proprio.shape[-1],
            ).to(visual_tokens.device)
            scores = None
            if mode in LEARNED_ROI_MODES:
                if args.roi_head_checkpoint is None:
                    raise ValueError(f"{mode} requires --roi-head-checkpoint")
                selector.load_checkpoint(args.roi_head_checkpoint, map_location=args.device)
                selector.to(visual_tokens.device)
                with torch.no_grad():
                    scores = selector.roi_head(visual_tokens, action=action, proprio=proprio)
            elif mode == "random":
                torch.manual_seed(args.seed + sample_idx)
            with torch.no_grad():
                _, mask, _ = selector(
                    visual_tokens,
                    action=action,
                    proprio=proprio,
                    scores=scores,
                    mode=mode,
                )
            mask_grid = _mask_to_numpy(mask)
            score_grid = _scores_to_numpy(scores)
            rows.append(
                {
                    "label": f"{mode} sample={sample_idx} split={args.split} idx={args.sample_index} t={timestep}",
                    "image": image,
                    "goal_image": goal_image,
                    "goal_timestep": goal_timestep,
                    "mask_grid": mask_grid,
                    "score_panel": _draw_score_heatmap(score_grid, mask_grid),
                    "overlay": _draw_mask_overlay(image, mask_grid),
                }
            )
    return rows


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Visualize ROI token masks on a 14x14 DINO patch grid."
    )
    parser.add_argument("--output-dir", default="roi_mask_viz")
    parser.add_argument("--data-path", default=None, help="Dataset root override, e.g. <DATA_ROOT>/pusht_noise or deformable root")
    parser.add_argument("--modes", default="random,fixed_random,lhs")
    parser.add_argument("--keep-ratio", type=float, default=0.5)
    parser.add_argument("--roi-topk", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=3, help="Number of dynamic random masks to show.")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--frame", type=int, default=0)

    parser.add_argument("--ckpt-base-path", default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--model-epoch", default="latest")
    parser.add_argument("--roi-head-checkpoint", default=None)
    parser.add_argument("--drs-head-checkpoint", default=None, help="Alias for --roi-head-checkpoint.")
    parser.add_argument("--split", default="valid", choices=["train", "valid"])
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--timestep", type=int, default=0)
    parser.add_argument("--goal-timestep", type=int, default=-1, help="Frame index in the sliced sample to save as the goal image.")
    parser.add_argument("--roi-hidden-dim", type=int, default=128)
    parser.add_argument("--drs-topk", type=int, default=None, help="Alias for --roi-topk.")
    parser.add_argument("--save-single-panels", action="store_true", help="Also save input/score/overlay as separate PNG files.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    return parser


def main():
    args = build_argparser().parse_args()
    if args.drs_head_checkpoint is not None:
        args.roi_head_checkpoint = args.drs_head_checkpoint
    if args.drs_topk is not None:
        args.roi_topk = args.drs_topk

    use_model = args.ckpt_base_path is not None and args.model_name is not None
    rows = _visualize_with_model(args) if use_model else _visualize_without_model(args)
    output_dir = Path(args.output_dir)
    label_name = "drs" if "drs" in Path(sys.argv[0]).stem or args.drs_head_checkpoint is not None else "roi"
    output_name = f"{label_name}_masks_keep{args.keep_ratio:g}_{args.modes.replace(',', '_')}.png"
    title = (
        f"{label_name.upper()} token mask visualization "
        f"(keep_ratio={args.keep_ratio:g}, modes={args.modes})"
    )
    _make_grid(rows, output_dir / output_name, title)
    if args.save_single_panels:
        _save_single_panels(rows, output_dir, prefix=label_name)
    print(f"Saved {label_name.upper()} mask visualization to {output_dir / output_name}")


if __name__ == "__main__":
    main()
