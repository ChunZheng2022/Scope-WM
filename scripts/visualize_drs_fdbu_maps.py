import argparse
import json
import math
import os
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from omegaconf import OmegaConf, open_dict
from PIL import Image, ImageDraw

from datasets.pusht_dset import ACTION_MEAN, ACTION_STD
from env.pusht.pusht_wrapper import PushTWrapper
from models.grouped_tokens import GroupedTokenProcessor
from plan import load_model
from preprocessor import Preprocessor
from utils import move_to_device


PATCH_SIDE = 14
PANEL_SIZE = 224


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


def _resize_image_array(image, size: int = PANEL_SIZE) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim != 3 or arr.shape[-1] not in (1, 3):
        raise ValueError(f"Expected HWC/CHW image array, got {arr.shape}")
    arr = arr.astype(np.float32)
    if arr.max() <= 2.0:
        arr = arr * 255.0
    arr = np.clip(arr, 0, 255).astype(np.uint8)
    out = Image.fromarray(arr.squeeze(-1) if arr.shape[-1] == 1 else arr).convert("RGB")
    out = out.resize((size, size), Image.BILINEAR)
    return np.asarray(out)


def _as_14x14(values: torch.Tensor) -> np.ndarray:
    values = values.detach().cpu().float().reshape(-1)
    side = int(math.sqrt(values.numel()))
    if side * side != values.numel():
        raise ValueError(f"Expected a square token grid, got {values.numel()} tokens")
    return values.reshape(side, side).numpy()


def _normalize_grid(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    values = values - float(values.min())
    denom = float(values.max())
    if denom < 1e-8:
        return np.zeros_like(values)
    return values / denom


def _json_safe(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return _json_safe(value.detach().cpu().item())
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _draw_drs_overlay(image: np.ndarray, mask_grid: np.ndarray, output_path: Path):
    out = Image.fromarray(image).convert("RGB")
    pixels = np.asarray(out).copy()
    side = mask_grid.shape[0]
    patch = out.width // side

    for r in range(side):
        for c in range(side):
            y0, y1 = r * patch, (r + 1) * patch
            x0, x1 = c * patch, (c + 1) * patch
            if mask_grid[r, c]:
                pixels[y0:y1, x0:x1] = (
                    pixels[y0:y1, x0:x1] * 0.66
                    + np.array([35, 235, 115], dtype=np.float32) * 0.34
                ).astype(np.uint8)
            else:
                pixels[y0:y1, x0:x1] = (pixels[y0:y1, x0:x1] * 0.58).astype(np.uint8)

    out = Image.fromarray(pixels).convert("RGB")
    draw = ImageDraw.Draw(out, "RGBA")
    for pos in range(side + 1):
        xy = pos * patch
        draw.line([(xy, 0), (xy, out.height)], fill=(255, 255, 255, 92), width=1)
        draw.line([(0, xy), (out.width, xy)], fill=(255, 255, 255, 92), width=1)
    for r in range(side):
        for c in range(side):
            if not mask_grid[r, c]:
                continue
            x0, y0 = c * patch, r * patch
            draw.rectangle([x0, y0, x0 + patch - 1, y0 + patch - 1], outline=(0, 255, 85, 230), width=2)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def _draw_update_heatmap(
    delta_grid: np.ndarray,
    foreground_grid: np.ndarray,
    output_path: Path,
    title: str,
):
    delta_norm = _normalize_grid(delta_grid)
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2), dpi=220)
    heat = ax.imshow(delta_norm, cmap="magma", interpolation="nearest", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=9)
    ax.set_xticks(np.arange(-0.5, PATCH_SIDE, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, PATCH_SIDE, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.35, alpha=0.55)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ys, xs = np.where(foreground_grid)
    if len(xs) > 0:
        ax.scatter(xs, ys, s=18, facecolors="none", edgecolors="#34ff7a", linewidths=0.7)
    fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _draw_positive_heatmap(
    value_grid: np.ndarray,
    foreground_grid: np.ndarray,
    output_path: Path,
    title: str,
    vmax: float = None,
    cmap: str = "magma",
):
    values = value_grid.astype(np.float32)
    if vmax is None:
        vmax = float(np.max(values))
    if vmax < 1e-8:
        vmax = 1.0
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2), dpi=220)
    heat = ax.imshow(values, cmap=cmap, interpolation="nearest", vmin=0.0, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks(np.arange(-0.5, PATCH_SIDE, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, PATCH_SIDE, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.35, alpha=0.55)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ys, xs = np.where(foreground_grid)
    if len(xs) > 0:
        ax.scatter(xs, ys, s=18, facecolors="none", edgecolors="#34ff7a", linewidths=0.7)
    fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _draw_difference_heatmap(
    diff_grid: np.ndarray,
    foreground_grid: np.ndarray,
    output_path: Path,
    title: str,
    cmap: str = "coolwarm",
    symmetric: bool = True,
):
    diff = diff_grid.astype(np.float32)
    if symmetric:
        vmax = float(np.max(np.abs(diff)))
        if vmax < 1e-8:
            vmax = 1.0
        vmin = -vmax
    else:
        vmin = float(np.min(diff))
        vmax = float(np.max(diff))
        if abs(vmax - vmin) < 1e-8:
            vmax = vmin + 1.0
    fig, ax = plt.subplots(1, 1, figsize=(3.2, 3.2), dpi=220)
    heat = ax.imshow(diff, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=9)
    ax.set_xticks(np.arange(-0.5, PATCH_SIDE, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, PATCH_SIDE, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.35, alpha=0.55)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    ys, xs = np.where(foreground_grid)
    if len(xs) > 0:
        ax.scatter(xs, ys, s=18, facecolors="none", edgecolors="#34ff7a", linewidths=0.7)
    fig.colorbar(heat, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0.2)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _draw_fdbu_heatmap(
    image: np.ndarray,
    delta_grid: np.ndarray,
    foreground_grid: np.ndarray,
    output_path: Path,
):
    delta_norm = _normalize_grid(delta_grid)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), dpi=180)

    axes[0].imshow(image)
    axes[0].set_title("PushT frame")
    axes[0].axis("off")

    heat = axes[1].imshow(delta_norm, cmap="magma", interpolation="nearest", vmin=0.0, vmax=1.0)
    axes[1].set_title("FDBU bg update |Delta z|")
    axes[1].set_xticks(np.arange(-0.5, PATCH_SIDE, 1), minor=True)
    axes[1].set_yticks(np.arange(-0.5, PATCH_SIDE, 1), minor=True)
    axes[1].grid(which="minor", color="white", linewidth=0.35, alpha=0.55)
    axes[1].tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)

    ys, xs = np.where(foreground_grid)
    if len(xs) > 0:
        axes[1].scatter(xs, ys, s=16, facecolors="none", edgecolors="#34ff7a", linewidths=0.8, label="DRS foreground")
        axes[1].legend(loc="lower right", fontsize=5, frameon=True)

    fig.colorbar(heat, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _model_checkpoint_path(base_path: str, model_name: str, model_epoch: str) -> Path:
    ckpt_name = "model_latest.pth" if str(model_epoch) == "latest" else f"model_{model_epoch}.pth"
    return Path(base_path) / "outputs" / model_name / "checkpoints" / ckpt_name


def _load_dense_reference_model(args, device):
    if not args.dense_model_name and not args.dense_checkpoint:
        return None, None
    if args.dense_checkpoint:
        dense_ckpt = Path(args.dense_checkpoint)
        if args.dense_train_config:
            dense_cfg_path = Path(args.dense_train_config)
        else:
            dense_cfg_path = dense_ckpt.parent.parent / "hydra.yaml"
        dense_base = None
        dense_epoch = None
        dense_model_name = dense_ckpt.parent.parent.name
    else:
        dense_base = args.dense_ckpt_base_path or args.ckpt_base_path
        dense_epoch = args.dense_model_epoch or args.model_epoch
        dense_model_name = args.dense_model_name
        dense_path = Path(dense_base) / "outputs" / dense_model_name
        dense_cfg_path = dense_path / "hydra.yaml"
        dense_ckpt = _model_checkpoint_path(dense_base, dense_model_name, dense_epoch)

    if not dense_cfg_path.exists():
        raise FileNotFoundError(f"Dense model hydra config not found: {dense_cfg_path}")
    if not dense_ckpt.exists():
        raise FileNotFoundError(f"Dense model checkpoint not found: {dense_ckpt}")

    dense_cfg = OmegaConf.load(dense_cfg_path)
    if args.data_path is not None:
        with open_dict(dense_cfg):
            dense_cfg.env.dataset.data_path = args.data_path
    dense_runtime_cfg = {
        "load_decoder": False,
        "use_drs": False,
        "drs_mode": "none",
        "use_roi": False,
        "roi_mode": "none",
        "sparse_dynamics": {
            "enabled": False,
        },
    }
    dense_model = load_model(
        dense_ckpt,
        dense_cfg,
        dense_cfg.num_action_repeat,
        device=device,
        roi_runtime_cfg=dense_runtime_cfg,
    )
    dense_model.eval()
    return dense_model, {
        "dense_model_name": dense_model_name,
        "dense_ckpt_base_path": None if dense_base is None else str(dense_base),
        "dense_model_epoch": None if dense_epoch is None else str(dense_epoch),
        "dense_checkpoint": str(dense_ckpt),
        "dense_train_config": str(dense_cfg_path),
    }


def _find_dataset_with_stats(dataset):
    cur = dataset
    for _ in range(8):
        if all(hasattr(cur, name) for name in ["action_mean", "state_mean", "proprio_mean", "transform"]):
            return cur
        cur = getattr(cur, "dataset", None)
        if cur is None:
            break
    raise AttributeError("Could not find dataset stats needed to transform plan observations.")


def _make_preprocessor(dataset) -> Preprocessor:
    dset = _find_dataset_with_stats(dataset)
    return Preprocessor(
        action_mean=dset.action_mean,
        action_std=dset.action_std,
        state_mean=dset.state_mean,
        state_std=dset.state_std,
        proprio_mean=dset.proprio_mean,
        proprio_std=dset.proprio_std,
        transform=dset.transform,
    )


def _denormalize_pusht_actions(actions: torch.Tensor, frameskip: int) -> np.ndarray:
    if actions.ndim == 2:
        actions = actions.unsqueeze(0)
    exec_actions = rearrange(actions.cpu(), "b t (f d) -> b (t f) d", f=frameskip)
    mean = ACTION_MEAN.view(1, 1, -1)
    std = ACTION_STD.view(1, 1, -1)
    return (exec_actions * std + mean).numpy()[0]


def _rollout_pusht_final_obs(seed: int, init_state: np.ndarray, normalized_actions: torch.Tensor, frameskip: int):
    exec_actions = _denormalize_pusht_actions(normalized_actions, frameskip)
    env = PushTWrapper(with_velocity=True, with_target=True)
    obs, _ = env.prepare(seed, np.asarray(init_state, dtype=np.float32).copy())
    final_obs = obs
    for action in exec_actions:
        final_obs, _, _, _ = env.step(action)
    env.close()
    return {
        key: np.expand_dims(np.expand_dims(np.asarray(value), axis=0), axis=1)
        for key, value in final_obs.items()
    }


def _load_plan_inputs(args, datasets, device):
    plan_dir = Path(args.plan_dir)
    with (plan_dir / "plan_targets.pkl").open("rb") as f:
        targets = pickle.load(f)
    actions_dir = Path(args.actions_plan_dir) if args.actions_plan_dir else plan_dir
    actions_path = actions_dir / "planned_actions.pt"
    if not actions_path.exists():
        raise FileNotFoundError(f"Missing planned_actions.pt in {actions_dir}")
    actions_payload = torch.load(actions_path, map_location="cpu")

    eval_id = int(args.eval_id)
    raw_obs = {
        key: np.asarray(value[eval_id : eval_id + 1])
        for key, value in targets["obs_0"].items()
    }
    raw_obs_g = {
        key: np.asarray(value[eval_id : eval_id + 1])
        for key, value in targets["obs_g"].items()
    }
    timestep = args.timestep
    num_obs = int(raw_obs["visual"].shape[1])
    if timestep < 0:
        timestep = num_obs + timestep
    timestep = max(0, min(int(timestep), num_obs - 1))
    image = _resize_image_array(raw_obs["visual"][0, timestep])

    preprocessor = _make_preprocessor(datasets[args.split])
    obs_b = move_to_device(preprocessor.transform_obs(raw_obs), device)
    obs_g_b = move_to_device(preprocessor.transform_obs(raw_obs_g), device)

    planned_actions = actions_payload["actions"][eval_id : eval_id + 1].float()
    goal_h = int(targets.get("goal_H", planned_actions.shape[1]))
    rollout_len = max(goal_h, num_obs)
    if planned_actions.shape[1] < rollout_len:
        pad = torch.zeros(
            planned_actions.shape[0],
            rollout_len - planned_actions.shape[1],
            planned_actions.shape[2],
            dtype=planned_actions.dtype,
        )
        planned_actions = torch.cat([planned_actions, pad], dim=1)
    act_b = planned_actions[:, :num_obs].to(device)
    rollout_act_b = planned_actions[:, :rollout_len].to(device)
    init_state = np.asarray(targets["state_0"][eval_id])
    if args.future_error_target == "env_rollout":
        raw_future_obs = _rollout_pusht_final_obs(
            seed=int(1 + 99 * eval_id),
            init_state=init_state,
            normalized_actions=planned_actions[:, :goal_h],
            frameskip=int(args.frameskip),
        )
        future_obs_b = move_to_device(preprocessor.transform_obs(raw_future_obs), device)
        future_target_source = "env_rollout"
    else:
        future_obs_b = obs_g_b
        future_target_source = "plan_obs_g"
    return image, obs_b, act_b, timestep, targets, {
        "plan_dir": str(plan_dir),
        "actions_plan_dir": str(actions_dir),
        "eval_id": eval_id,
        "eval_seed": int(1 + 99 * eval_id),
        "goal_H": goal_h,
        "rollout_action_steps": rollout_len,
        "future_target_source": future_target_source,
        "_obs_g": future_obs_b,
        "_rollout_act": rollout_act_b,
    }


def _load_model_and_sample(args):
    model_path = Path(args.ckpt_base_path) / "outputs" / args.model_name
    train_cfg = OmegaConf.load(model_path / "hydra.yaml")
    if args.data_path is not None:
        with open_dict(train_cfg):
            train_cfg.env.dataset.data_path = args.data_path

    runtime_cfg = {
        "use_drs": True,
        "drs_mode": args.drs_mode,
        "drs_keep_ratio": args.drs_keep_ratio,
        "drs_topk": args.drs_topk,
        "drs_mask_type": "mask",
        "drs_head_checkpoint": args.drs_head_checkpoint,
        "load_decoder": False,
        "sparse_dynamics": {
            "enabled": True,
            "mode": "sparse_primary",
            "mask_source": args.sparse_mask_source,
            "background_processor": args.background_processor,
            "fdbu_hidden_mult": args.fdbu_hidden_mult,
            "fdbu_alpha": args.fdbu_alpha,
            "fdbu_dropout": args.fdbu_dropout,
            "loss_on_foreground_only": False,
        },
    }
    ckpt_path = _model_checkpoint_path(args.ckpt_base_path, args.model_name, args.model_epoch)
    device = torch.device(args.device)
    model = load_model(
        ckpt_path,
        train_cfg,
        train_cfg.num_action_repeat,
        device=device,
        roi_runtime_cfg=runtime_cfg,
    )
    model.eval()
    dense_model, dense_meta = _load_dense_reference_model(args, device)

    datasets, _ = hydra.utils.call(
        train_cfg.env.dataset,
        num_hist=train_cfg.num_hist,
        num_pred=train_cfg.num_pred,
        frameskip=train_cfg.frameskip,
    )
    if args.plan_dir:
        image, obs_b, act_b, timestep, targets, case_meta = _load_plan_inputs(
            args,
            datasets,
            device,
        )
        state = np.asarray(targets["state_0"][int(args.eval_id)])
        return model, dense_model, dense_meta, image, obs_b, act_b, timestep, state, case_meta

    sample = datasets[args.split][args.sample_index]
    obs, act, state = sample
    timestep = args.timestep
    if timestep < 0:
        timestep = int(obs["visual"].shape[0]) + timestep
    timestep = max(0, min(int(timestep), int(obs["visual"].shape[0]) - 1))
    image = _resize_chw_image(obs["visual"][timestep])

    obs_b = {k: v.unsqueeze(0) for k, v in obs.items()}
    act_b = act.unsqueeze(0)
    obs_b = move_to_device(obs_b, device)
    act_b = act_b.to(device)
    return model, dense_model, dense_meta, image, obs_b, act_b, timestep, state, {
        "split": args.split,
        "sample_index": int(args.sample_index),
        "obs_g_available": False,
    }


def _random_mask_grid(num_tokens: int, num_selected: int, seed: int) -> np.ndarray:
    side = int(math.sqrt(num_tokens))
    if side * side != num_tokens:
        raise ValueError(f"Expected square token grid, got {num_tokens} tokens")
    num_selected = max(0, min(int(num_selected), int(num_tokens)))
    rng = np.random.default_rng(int(seed))
    mask = np.zeros(num_tokens, dtype=bool)
    if num_selected > 0:
        mask[rng.choice(num_tokens, size=num_selected, replace=False)] = True
    return mask.reshape(side, side)


def _compute_drs_and_fdbu(args):
    model, dense_model, dense_meta, image, obs, act, timestep, state, case_meta = _load_model_and_sample(args)
    obs_g = case_meta.pop("_obs_g", None)
    rollout_act = case_meta.pop("_rollout_act", None)
    processor = getattr(model, "sparse_primary_dynamics", None)
    if processor is None:
        raise ValueError("Model has no sparse_primary_dynamics module; enable sparse_dynamics in the checkpoint/runtime config.")
    if getattr(processor, "fdbu", None) is None:
        raise ValueError("sparse_primary_dynamics.background_processor must be fdbu to visualize FDBU updates.")

    with torch.no_grad():
        z_obs = model.encode_obs(obs)
        z_act = model.encode_act(act)
        z = model.compose_z(z_obs["visual"], z_obs["proprio"], z_act)
        foreground_mask, sparse_stats = model._select_sparse_foreground_mask(z)
        sparse_z, meta = processor._sparse_z_tokens(z, foreground_mask)
        pos_embeddings = processor._build_pos_embeddings(
            model.predictor,
            meta,
            num_frames=z.shape[1],
            dim=z.shape[-1],
            batch_size=z.shape[0],
        )
        attn_mask = GroupedTokenProcessor._causal_mask(
            z.shape[1],
            sparse_z.shape[2],
            sparse_z.device,
        )
        sparse_pred = processor._run_predictor(
            model.predictor,
            sparse_z,
            pos_embeddings=pos_embeddings,
            attn_mask=attn_mask,
        )
        fdbu_fill = processor.fdbu(z, sparse_z, sparse_pred, foreground_mask)
        delta_mag = (fdbu_fill - z).norm(dim=-1)
        bg_delta_mag = delta_mag.masked_fill(foreground_mask, 0.0)
        if dense_model is None:
            dense_z = z
            dense_pred = model._run_predictor_on_z(z)
            dense_source = "current_model_full_token_pass"
        else:
            dense_z = dense_model.encode(obs, act)
            dense_pred = dense_model._run_predictor_on_z(dense_z)
            dense_source = "dense_reference_model"
        dense_delta_mag = (dense_pred - dense_z).norm(dim=-1)
        dense_bg_delta_mag = dense_delta_mag.masked_fill(foreground_mask, 0.0)
        gt_future = {}
        if obs_g is not None and rollout_act is not None:
            final_bg_mask = foreground_mask[:, timestep]
            fdbu_rollout_obses, _ = model.rollout(obs, rollout_act)
            fdbu_pred_future = fdbu_rollout_obses["visual"][:, -1]
            fdbu_gt_future = model.encode_obs(obs_g)["visual"][:, -1]
            fdbu_future_error = (fdbu_pred_future - fdbu_gt_future).norm(dim=-1)
            fdbu_future_bg_error = fdbu_future_error.masked_fill(final_bg_mask, 0.0)

            if dense_model is not None:
                dense_rollout_obses, _ = dense_model.rollout(obs, rollout_act)
                dense_pred_future = dense_rollout_obses["visual"][:, -1]
                dense_gt_future = dense_model.encode_obs(obs_g)["visual"][:, -1]
            else:
                dense_rollout_obses, _ = model.rollout(obs, rollout_act)
                dense_pred_future = dense_rollout_obses["visual"][:, -1]
                dense_gt_future = model.encode_obs(obs_g)["visual"][:, -1]
            dense_future_error = (dense_pred_future - dense_gt_future).norm(dim=-1)
            dense_future_bg_error = dense_future_error.masked_fill(final_bg_mask, 0.0)
            gt_future = {
                "fdbu_future_bg_error_grid": _as_14x14(fdbu_future_bg_error[0]),
                "dense_future_bg_error_grid": _as_14x14(dense_future_bg_error[0]),
            }

    drs_mask_grid = _as_14x14(foreground_mask[0, timestep]).astype(bool)
    random_grid = _random_mask_grid(
        drs_mask_grid.size,
        int(drs_mask_grid.sum()),
        seed=args.random_mask_seed,
    )
    fdbu_grid = _as_14x14(bg_delta_mag[0, timestep])
    dense_grid = _as_14x14(dense_bg_delta_mag[0, timestep])
    diff_grid = fdbu_grid - dense_grid
    abs_diff_grid = np.abs(diff_grid)
    future_error = {}
    if gt_future:
        future_diff_grid = (
            gt_future["fdbu_future_bg_error_grid"]
            - gt_future["dense_future_bg_error_grid"]
        )
        future_error = {
            **gt_future,
            "fdbu_minus_dense_future_bg_error_grid": future_diff_grid,
            "abs_future_error_diff_grid": np.abs(future_diff_grid),
        }
    return {
        "image": image,
        "random_mask_grid": random_grid,
        "drs_mask_grid": drs_mask_grid,
        "dense_delta_grid": dense_grid,
        "fdbu_delta_grid": fdbu_grid,
        "fdbu_minus_dense_grid": diff_grid,
        "abs_diff_grid": abs_diff_grid,
        **future_error,
        "state": state.detach().cpu().numpy() if torch.is_tensor(state) else np.asarray(state),
        "sparse_stats": sparse_stats,
        "timestep": timestep,
        "case_meta": case_meta,
        "dense_meta": dense_meta,
        "dense_source": dense_source,
    }


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Visualize DRS-selected PushT tokens and actual FDBU background update magnitudes."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--ckpt-base-path", required=True)
    parser.add_argument("--model-epoch", default="latest")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--drs-head-checkpoint", required=True)
    parser.add_argument("--dense-model-name", default=None, help="Optional dense baseline model name for dense predictor update visualization.")
    parser.add_argument("--dense-ckpt-base-path", default=None, help="Optional ckpt base path for --dense-model-name; defaults to --ckpt-base-path.")
    parser.add_argument("--dense-model-epoch", default=None, help="Optional dense model epoch; defaults to --model-epoch.")
    parser.add_argument("--dense-checkpoint", default=None, help="Optional exact dense baseline checkpoint path. Takes precedence over --dense-model-name.")
    parser.add_argument("--dense-train-config", default=None, help="Optional exact hydra.yaml path for --dense-checkpoint; defaults to checkpoint parent parent / hydra.yaml.")
    parser.add_argument("--output-dir", default="viz_drs_fdbu")
    parser.add_argument("--split", default="valid", choices=["train", "valid"])
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--plan-dir", default=None, help="Optional plan output dir; if set, visualize this plan target instead of a dataset sample.")
    parser.add_argument("--actions-plan-dir", default=None, help="Optional plan output dir supplying planned_actions.pt for action-conditioned DRS scores.")
    parser.add_argument("--eval-id", type=int, default=0, help="Eval index to visualize when --plan-dir is set.")
    parser.add_argument("--timestep", type=int, default=-1, help="History timestep to visualize; -1 means the last history frame.")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--random-mask-seed", type=int, default=0)
    parser.add_argument("--frameskip", type=int, default=5, help="PushT frameskip used to unpack WM actions for --future-error-target env_rollout.")
    parser.add_argument(
        "--future-error-target",
        choices=["plan_obs_g", "env_rollout"],
        default="plan_obs_g",
        help="Target used for future latent prediction error heatmaps.",
    )

    parser.add_argument("--drs-mode", default="distilled_topk")
    parser.add_argument("--drs-keep-ratio", type=float, default=1.0)
    parser.add_argument("--drs-topk", type=int, default=32)
    parser.add_argument("--sparse-mask-source", default="drs")
    parser.add_argument("--background-processor", default="fdbu")
    parser.add_argument("--fdbu-hidden-mult", type=float, default=2.0)
    parser.add_argument("--fdbu-alpha", type=float, default=1.0)
    parser.add_argument("--fdbu-dropout", type=float, default=0.0)
    return parser


def main():
    args = build_argparser().parse_args()
    output_dir = Path(args.output_dir)
    result = _compute_drs_and_fdbu(args)

    if args.plan_dir:
        stem = f"{args.model_name.replace('/', '_')}_eval{args.eval_id:03d}_t{result['timestep']}"
    else:
        stem = f"{args.model_name.replace('/', '_')}_sample{args.sample_index}_t{result['timestep']}"
    random_path = output_dir / f"{stem}_random_mask.png"
    drs_path = output_dir / f"{stem}_drs_mask.png"
    dense_path = output_dir / f"{stem}_dense_bg_update.png"
    fdbu_path = output_dir / f"{stem}_fdbu_bg_update.png"
    diff_path = output_dir / f"{stem}_fdbu_minus_dense_bg_update.png"
    abs_diff_path = output_dir / f"{stem}_abs_diff_bg_update.png"
    dense_future_error_path = output_dir / f"{stem}_dense_future_bg_error.png"
    fdbu_future_error_path = output_dir / f"{stem}_fdbu_future_bg_error.png"
    future_error_diff_path = output_dir / f"{stem}_fdbu_minus_dense_future_bg_error.png"
    fdbu_pair_path = output_dir / f"{stem}_fdbu_bg_update_pair.png"
    raw_path = output_dir / f"{stem}_raw.npz"
    meta_path = output_dir / f"{stem}_meta.json"

    _draw_drs_overlay(result["image"], result["random_mask_grid"], random_path)
    _draw_drs_overlay(result["image"], result["drs_mask_grid"], drs_path)
    _draw_update_heatmap(
        result["dense_delta_grid"],
        result["drs_mask_grid"],
        dense_path,
        title="Dense bg update |Delta z|",
    )
    _draw_update_heatmap(
        result["fdbu_delta_grid"],
        result["drs_mask_grid"],
        fdbu_path,
        title="FDBU bg update |Delta z|",
    )
    _draw_difference_heatmap(
        result["fdbu_minus_dense_grid"],
        result["drs_mask_grid"],
        diff_path,
        title="FDBU - Dense bg update",
        cmap="coolwarm",
        symmetric=True,
    )
    _draw_difference_heatmap(
        result["abs_diff_grid"],
        result["drs_mask_grid"],
        abs_diff_path,
        title="|FDBU - Dense| bg update",
        cmap="viridis",
        symmetric=False,
    )
    has_future_error = (
        "dense_future_bg_error_grid" in result
        and "fdbu_future_bg_error_grid" in result
    )
    if has_future_error:
        future_vmax = float(
            max(
                np.max(result["dense_future_bg_error_grid"]),
                np.max(result["fdbu_future_bg_error_grid"]),
            )
        )
        _draw_positive_heatmap(
            result["dense_future_bg_error_grid"],
            result["drs_mask_grid"],
            dense_future_error_path,
            title="Dense future bg error",
            vmax=future_vmax,
        )
        _draw_positive_heatmap(
            result["fdbu_future_bg_error_grid"],
            result["drs_mask_grid"],
            fdbu_future_error_path,
            title="FDBU future bg error",
            vmax=future_vmax,
        )
        _draw_difference_heatmap(
            result["fdbu_minus_dense_future_bg_error_grid"],
            result["drs_mask_grid"],
            future_error_diff_path,
            title="FDBU - Dense future error",
            cmap="coolwarm",
            symmetric=True,
        )
    _draw_fdbu_heatmap(
        result["image"],
        result["fdbu_delta_grid"],
        result["drs_mask_grid"],
        fdbu_pair_path,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_arrays = {
        "random_mask": result["random_mask_grid"].astype(np.uint8),
        "drs_mask": result["drs_mask_grid"].astype(np.uint8),
        "dense_bg_delta": result["dense_delta_grid"].astype(np.float32),
        "fdbu_bg_delta": result["fdbu_delta_grid"].astype(np.float32),
        "fdbu_minus_dense_bg_delta": result["fdbu_minus_dense_grid"].astype(np.float32),
        "abs_diff_bg_delta": result["abs_diff_grid"].astype(np.float32),
        "state": result["state"],
    }
    if has_future_error:
        raw_arrays.update(
            {
                "dense_future_bg_error": result["dense_future_bg_error_grid"].astype(np.float32),
                "fdbu_future_bg_error": result["fdbu_future_bg_error_grid"].astype(np.float32),
                "fdbu_minus_dense_future_bg_error": result[
                    "fdbu_minus_dense_future_bg_error_grid"
                ].astype(np.float32),
                "abs_future_error_diff": result["abs_future_error_diff_grid"].astype(np.float32),
            }
        )
    np.savez_compressed(raw_path, **raw_arrays)
    future_outputs = {}
    if has_future_error:
        future_outputs = {
            "dense_future_bg_error": str(dense_future_error_path),
            "fdbu_future_bg_error": str(fdbu_future_error_path),
            "fdbu_minus_dense_future_bg_error": str(future_error_diff_path),
        }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "model_name": args.model_name,
                "model_epoch": args.model_epoch,
                "split": args.split,
                "sample_index": args.sample_index,
                "plan_dir": args.plan_dir,
                "actions_plan_dir": args.actions_plan_dir,
                "eval_id": args.eval_id,
                "case_meta": result["case_meta"],
                "dense_source": result["dense_source"],
                "dense_model": result["dense_meta"],
                "timestep": result["timestep"],
                "drs_topk": args.drs_topk,
                "random_mask_seed": args.random_mask_seed,
                "sparse_stats": _json_safe(result["sparse_stats"]),
                "outputs": {
                    "random_mask": str(random_path),
                    "drs_mask": str(drs_path),
                    "dense_bg_update": str(dense_path),
                    "fdbu_bg_update": str(fdbu_path),
                    "fdbu_minus_dense_bg_update": str(diff_path),
                    "abs_diff_bg_update": str(abs_diff_path),
                    **future_outputs,
                    "fdbu_bg_update_pair": str(fdbu_pair_path),
                    "raw_npz": str(raw_path),
                },
            },
            f,
            indent=2,
        )

    print(f"Saved random mask overlay: {random_path}")
    print(f"Saved DRS mask overlay: {drs_path}")
    print(f"Saved dense background update heatmap: {dense_path}")
    print(f"Saved FDBU background update heatmap: {fdbu_path}")
    print(f"Saved FDBU-minus-dense difference heatmap: {diff_path}")
    print(f"Saved absolute difference heatmap: {abs_diff_path}")
    if has_future_error:
        print(f"Saved dense future background error heatmap: {dense_future_error_path}")
        print(f"Saved FDBU future background error heatmap: {fdbu_future_error_path}")
        print(f"Saved FDBU-minus-dense future error heatmap: {future_error_diff_path}")
    print(f"Saved raw arrays: {raw_path}")


if __name__ == "__main__":
    main()
