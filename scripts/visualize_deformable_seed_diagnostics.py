import argparse
import csv
import json
import math
import os
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


def _read_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_targets(plan_dir: Path) -> Dict:
    path = plan_dir / "plan_targets.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Missing plan_targets.pkl: {path}")
    with path.open("rb") as f:
        return pickle.load(f)


def _load_planned_actions(plan_dir: Path) -> Optional[Dict]:
    import torch

    path = plan_dir / "planned_actions.pt"
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu")


def _as_float_list(value) -> Optional[List[float]]:
    if value is None:
        return None
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, list):
        return None
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return None


def _extract_cd_from_mpc_jsonl(plan_dir: Path) -> Tuple[List[Dict], Optional[int]]:
    rows = []
    for row in _read_jsonl(plan_dir / "mpc_iterations.jsonl"):
        task_metrics = row.get("eval_task_metrics") or {}
        cd = None
        for key in ["chamfer_distance", "task_chamfer_distance"]:
            cd = _as_float_list(task_metrics.get(key))
            if cd is not None:
                break
        if cd is None:
            continue
        rows.append({"mpc_iter": int(row["mpc_iter"]), "cd": cd})
    n_evals = None
    if rows:
        n_evals = len(rows[0]["cd"])
    return rows, n_evals


def _parse_eval_seeds_from_log(text: str) -> Optional[List[int]]:
    match = re.search(r"eval_seed:\s*\[([^\]]+)\]", text, flags=re.S)
    if not match:
        return None
    seeds = re.findall(r"-?\d+", match.group(1))
    return [int(seed) for seed in seeds]


def _extract_cd_from_stdout_log(log_file: Path) -> Tuple[List[Dict], Optional[List[int]]]:
    text = log_file.read_text(encoding="utf-8", errors="ignore")
    eval_seeds = _parse_eval_seeds_from_log(text)
    rows = []
    current_iter = None
    current_values: List[float] = []
    iter_re = re.compile(r"MPC iter\s+(\d+)\s+Eval")
    cd_re = re.compile(r"CD:\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)")
    for raw_line in text.splitlines():
        iter_match = iter_re.search(raw_line)
        if iter_match:
            if current_iter is not None and current_values:
                rows.append({"mpc_iter": current_iter, "cd": current_values})
            current_iter = int(iter_match.group(1))
            current_values = []
            continue
        cd_match = cd_re.search(raw_line)
        if cd_match and current_iter is not None:
            current_values.append(float(cd_match.group(1)))
    if current_iter is not None and current_values:
        rows.append({"mpc_iter": current_iter, "cd": current_values})
    return rows, eval_seeds


def _to_particle_state(value) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim == 1:
        if arr.size % 4 != 0:
            raise ValueError(f"Cannot reshape flat state of size {arr.size} to particles")
        arr = arr.reshape(-1, 4)
    elif arr.ndim == 2 and arr.shape[-1] != 4:
        if arr.size % 4 != 0:
            raise ValueError(f"Cannot reshape state with shape {arr.shape} to particles")
        arr = arr.reshape(-1, 4)
    elif arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    if arr.shape[-1] < 3:
        raise ValueError(f"State must have at least 3 coordinates, got {arr.shape}")
    return arr.astype(np.float32)


def _xz(state: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    particles = _to_particle_state(state)
    return particles[:, 0], particles[:, 2]


def _chamfer_np(a: np.ndarray, b: np.ndarray, max_points: int = 6000) -> float:
    a = _to_particle_state(a)[:, :3]
    b = _to_particle_state(b)[:, :3]
    if len(a) > max_points:
        idx = np.linspace(0, len(a) - 1, max_points).astype(np.int64)
        a = a[idx]
    if len(b) > max_points:
        idx = np.linspace(0, len(b) - 1, max_points).astype(np.int64)
        b = b[idx]
    try:
        from scipy.spatial import cKDTree

        tree_b = cKDTree(b)
        tree_a = cKDTree(a)
        return float(tree_b.query(a, k=1)[0].mean() + tree_a.query(b, k=1)[0].mean())
    except Exception:
        import torch

        a_t = torch.tensor(a, dtype=torch.float32)
        b_t = torch.tensor(b, dtype=torch.float32)
        dist = torch.cdist(a_t[None], b_t[None])[0]
        return float(dist.min(dim=1).values.mean() + dist.min(dim=0).values.mean())


def _axis_equal_2d(ax, xs: Sequence[np.ndarray], ys: Sequence[np.ndarray]) -> None:
    x = np.concatenate([np.asarray(v) for v in xs if len(v)])
    y = np.concatenate([np.asarray(v) for v in ys if len(v)])
    x_min, x_max = float(x.min()), float(x.max())
    y_min, y_max = float(y.min()), float(y.max())
    cx, cy = (x_min + x_max) / 2, (y_min + y_max) / 2
    radius = max(x_max - x_min, y_max - y_min) / 2
    radius = max(radius * 1.15, 1e-3)
    ax.set_xlim(cx - radius, cx + radius)
    ax.set_ylim(cy - radius, cy + radius)
    ax.set_aspect("equal", adjustable="box")


def _plot_state_panel(
    ax,
    state: np.ndarray,
    title: str,
    color: str = "tab:blue",
    goal: Optional[np.ndarray] = None,
):
    x, z = _xz(state)
    marker_size = 4 if len(x) < 3000 else 1
    if goal is not None:
        gx, gz = _xz(goal)
        ax.scatter(gx, gz, s=marker_size, c="tab:orange", alpha=0.25, label="goal")
    ax.scatter(x, z, s=marker_size, c=color, alpha=0.75, label="state")
    if goal is not None:
        _axis_equal_2d(ax, [x, gx], [z, gz])
    else:
        _axis_equal_2d(ax, [x], [z])
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("z")
    ax.grid(alpha=0.2)


def _plot_cd_curves(rows: List[Dict], eval_id: int, eval_seed, out_path: Path) -> None:
    matrix = np.asarray([row["cd"] for row in rows], dtype=np.float32)
    mpc_iters = np.asarray([row["mpc_iter"] for row in rows], dtype=np.int32)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for idx in range(matrix.shape[1]):
        ax.plot(mpc_iters, matrix[:, idx], color="0.75", linewidth=1.0, alpha=0.8)
    ax.plot(
        mpc_iters,
        matrix.mean(axis=1),
        color="black",
        linestyle="--",
        linewidth=1.6,
        label="mean over evals",
    )
    label = f"eval_id={eval_id}"
    if eval_seed is not None:
        label += f", seed={eval_seed}"
    ax.plot(
        mpc_iters,
        matrix[:, eval_id],
        color="crimson",
        marker="o",
        linewidth=2.4,
        label=label,
    )
    best_idx = int(np.argmin(matrix[:, eval_id]))
    ax.scatter(
        [mpc_iters[best_idx]],
        [matrix[best_idx, eval_id]],
        color="crimson",
        s=80,
        zorder=4,
    )
    ax.annotate(
        f"best {matrix[best_idx, eval_id]:.3f} @ iter {mpc_iters[best_idx]}",
        xy=(mpc_iters[best_idx], matrix[best_idx, eval_id]),
        xytext=(8, 10),
        textcoords="offset points",
        color="crimson",
    )
    ax.set_xlabel("MPC iteration")
    ax.set_ylabel("Chamfer distance")
    ax.set_title("Per-seed CD trajectory")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _plot_cd_heatmap(rows: List[Dict], out_path: Path, eval_id: int) -> None:
    matrix = np.asarray([row["cd"] for row in rows], dtype=np.float32).T
    fig, ax = plt.subplots(figsize=(8, 4.8))
    im = ax.imshow(matrix, aspect="auto", interpolation="nearest", cmap="magma")
    ax.axhline(eval_id, color="cyan", linewidth=1.5)
    ax.set_xlabel("MPC iteration index")
    ax.set_ylabel("eval_id")
    ax.set_title("Chamfer distance heatmap")
    fig.colorbar(im, ax=ax, label="CD")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _write_seed_summary(
    rows: List[Dict],
    targets: Dict,
    eval_seeds: Optional[List[int]],
    out_path: Path,
) -> List[Dict]:
    matrix = np.asarray([row["cd"] for row in rows], dtype=np.float32) if rows else None
    n = matrix.shape[1] if matrix is not None else len(targets["state_0"])
    summaries = []
    for idx in range(n):
        init = _to_particle_state(targets["state_0"][idx])
        goal = _to_particle_state(targets["state_g"][idx])
        init_goal_cd = _chamfer_np(init, goal)
        init_center = init[:, :3].mean(axis=0)
        goal_center = goal[:, :3].mean(axis=0)
        center_dist = float(np.linalg.norm(init_center - goal_center))
        row = {
            "eval_id": idx,
            "eval_seed": eval_seeds[idx] if eval_seeds and idx < len(eval_seeds) else "",
            "initial_goal_cd": init_goal_cd,
            "center_dist": center_dist,
            "num_particles": int(init.shape[0]),
        }
        if matrix is not None:
            values = matrix[:, idx]
            best_pos = int(np.argmin(values))
            row.update(
                {
                    "mean_mpc_cd": float(np.mean(values)),
                    "min_mpc_cd": float(values[best_pos]),
                    "best_mpc_iter": int(rows[best_pos]["mpc_iter"]),
                    "final_logged_cd": float(values[-1]),
                    "cd_improvement_from_iter0": float(values[0] - values[-1]),
                }
            )
        summaries.append(row)
    fieldnames = sorted({key for row in summaries for key in row.keys()})
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summaries)
    return summaries


def _plot_initial_goal(
    init: np.ndarray,
    goal: np.ndarray,
    eval_id: int,
    eval_seed,
    out_path: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
    _plot_state_panel(axes[0], init, "initial", color="tab:blue")
    _plot_state_panel(axes[1], goal, "goal", color="tab:orange")
    _plot_state_panel(axes[2], init, "overlay", color="tab:blue", goal=goal)
    title = f"eval_id={eval_id}"
    if eval_seed is not None:
        title += f", seed={eval_seed}"
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _load_deform_action_stats(data_path: str, object_name: str, normalize_action: bool):
    from datasets.deformable_env_dset import DeformDataset

    dset = DeformDataset(
        data_path=data_path,
        object_name=object_name,
        normalize_action=normalize_action,
    )
    return dset.action_mean.float(), dset.action_std.float()


def _denormalize_deform_actions(
    actions,
    frameskip: int,
    action_mean,
    action_std,
) -> np.ndarray:
    from einops import rearrange

    if actions.ndim != 2:
        raise ValueError(f"Expected per-eval actions with shape (T, D), got {actions.shape}")
    exec_actions = rearrange(actions.cpu(), "t (f d) -> (t f) d", f=frameskip)
    exec_actions = exec_actions * action_std.view(1, -1) + action_mean.view(1, -1)
    return exec_actions.numpy()


def _replay_deformable(
    object_name: str,
    seed: int,
    init_state: np.ndarray,
    exec_actions: np.ndarray,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    from env.deformable_env.FlexEnvWrapper import FlexEnvWrapper

    env = FlexEnvWrapper(object_name=object_name)
    try:
        obses, states = env.rollout(seed, _to_particle_state(init_state), exec_actions)
    finally:
        try:
            env.close()
        except Exception:
            pass
    return obses, states


def _plot_replay_states(
    states: np.ndarray,
    goal: np.ndarray,
    rows: List[Dict],
    n_taken_actions: int,
    frameskip: int,
    out_path: Path,
    save_frames_dir: Optional[Path] = None,
) -> None:
    panels: List[Tuple[str, np.ndarray]] = [("initial", states[0])]
    for row in rows:
        step = min((int(row["mpc_iter"]) + 1) * n_taken_actions * frameskip, len(states) - 1)
        panels.append((f"mpc {row['mpc_iter']}", states[step]))
    panels.append(("goal", goal))

    n_cols = 4
    n_rows = math.ceil(len(panels) / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.2 * n_cols, 4.0 * n_rows))
    axes = np.asarray(axes).reshape(-1)
    for ax, (title, state) in zip(axes, panels):
        if title == "goal":
            _plot_state_panel(ax, state, title, color="tab:orange")
        else:
            cd = _chamfer_np(state, goal)
            _plot_state_panel(ax, state, f"{title} | CD={cd:.3f}", color="tab:blue", goal=goal)
    for ax in axes[len(panels) :]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)

    if save_frames_dir is not None:
        save_frames_dir.mkdir(parents=True, exist_ok=True)
        for title, state in panels:
            fig, ax = plt.subplots(figsize=(4.2, 4.0))
            if title == "goal":
                _plot_state_panel(ax, state, title, color="tab:orange")
            else:
                cd = _chamfer_np(state, goal)
                _plot_state_panel(ax, state, f"{title} | CD={cd:.3f}", color="tab:blue", goal=goal)
            fig.tight_layout()
            stem = title.replace(" ", "_")
            fig.savefig(save_frames_dir / f"{stem}.png", dpi=200)
            plt.close(fig)


def _plot_action_norms(exec_actions: np.ndarray, n_taken_actions: int, frameskip: int, out_path: Path) -> None:
    norms = np.linalg.norm(exec_actions, axis=1)
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    ax.plot(np.arange(len(norms)), norms, marker="o", linewidth=1.4)
    stride = max(1, n_taken_actions * frameskip)
    for step in range(stride, len(norms) + 1, stride):
        ax.axvline(step - 0.5, color="0.75", linewidth=0.8, linestyle="--")
    ax.set_xlabel("environment action step")
    ax.set_ylabel("action L2 norm")
    ax.set_title("Executed action magnitude")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def _resize_image_array(image, size: int = 224) -> np.ndarray:
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


def _as_square_grid(values) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu()
    if hasattr(values, "numpy"):
        values = values.float().reshape(-1).numpy()
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    side = int(math.sqrt(values.size))
    if side * side != values.size:
        raise ValueError(f"Expected square token grid, got {values.size} tokens")
    return values.reshape(side, side)


def _normalize_grid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    values = values - float(np.nanmin(values))
    denom = float(np.nanmax(values))
    if denom < 1e-8:
        return np.zeros_like(values)
    return values / denom


def _draw_token_mask_overlay(image: np.ndarray, mask_grid: np.ndarray, output_path: Path) -> None:
    out = Image.fromarray(_resize_image_array(image)).convert("RGB")
    pixels = np.asarray(out).copy()
    side = int(mask_grid.shape[0])
    patch = out.width // side
    for r in range(side):
        for c in range(side):
            y0, y1 = r * patch, (r + 1) * patch
            x0, x1 = c * patch, (c + 1) * patch
            if mask_grid[r, c]:
                pixels[y0:y1, x0:x1] = (
                    pixels[y0:y1, x0:x1] * 0.62
                    + np.array([35, 235, 115], dtype=np.float32) * 0.38
                ).astype(np.uint8)
            else:
                pixels[y0:y1, x0:x1] = (pixels[y0:y1, x0:x1] * 0.55).astype(np.uint8)
    out = Image.fromarray(pixels).convert("RGB")
    draw = ImageDraw.Draw(out, "RGBA")
    for i in range(side + 1):
        v = i * patch
        draw.line([(0, v), (out.width, v)], fill=(255, 255, 255, 105), width=1)
        draw.line([(v, 0), (v, out.height)], fill=(255, 255, 255, 105), width=1)
    for r in range(side):
        for c in range(side):
            if mask_grid[r, c]:
                cx = int((c + 0.5) * patch)
                cy = int((r + 0.5) * patch)
                rad = max(3, patch // 4)
                draw.ellipse(
                    [(cx - rad, cy - rad), (cx + rad, cy + rad)],
                    outline=(0, 255, 120, 255),
                    width=2,
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def _draw_token_heatmap(
    values_grid: np.ndarray,
    mask_grid: Optional[np.ndarray],
    output_path: Path,
    title: str,
    cmap: str = "magma",
    center_zero: bool = False,
) -> None:
    values_grid = np.asarray(values_grid, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    if center_zero:
        vmax = float(np.nanmax(np.abs(values_grid)))
        vmin = -vmax
    else:
        vmin = float(np.nanmin(values_grid))
        vmax = float(np.nanmax(values_grid))
    im = ax.imshow(values_grid, cmap=cmap, interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xticks(np.arange(-0.5, values_grid.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, values_grid.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.4, alpha=0.55)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    if mask_grid is not None:
        yy, xx = np.where(mask_grid)
        ax.scatter(xx, yy, s=32, facecolors="none", edgecolors="#00ff7f", linewidths=1.2)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _model_checkpoint_path(ckpt_base_path: str, model_name: str, model_epoch: str) -> Path:
    ckpt_dir = Path(ckpt_base_path) / "outputs" / model_name / "checkpoints"
    if str(model_epoch) == "latest":
        return ckpt_dir / "model_latest.pth"
    return ckpt_dir / f"model_{model_epoch}.pth"


def _make_deformable_preprocessor(train_cfg, data_path: Optional[str], object_name: Optional[str]):
    import hydra
    from omegaconf import open_dict
    from preprocessor import Preprocessor

    cfg = train_cfg.copy()
    with open_dict(cfg):
        if data_path is not None:
            cfg.env.dataset.data_path = data_path
        if object_name is not None:
            cfg.env.dataset.object_name = object_name
    datasets, _ = hydra.utils.call(
        cfg.env.dataset,
        num_hist=cfg.num_hist,
        num_pred=cfg.num_pred,
        frameskip=cfg.frameskip,
    )
    dset = datasets["train"]
    for _ in range(8):
        if all(hasattr(dset, name) for name in ["action_mean", "state_mean", "proprio_mean", "transform"]):
            break
        dset = getattr(dset, "dataset", None)
        if dset is None:
            raise AttributeError("Could not find dataset stats for deformable preprocessing.")
    return Preprocessor(
        action_mean=dset.action_mean,
        action_std=dset.action_std,
        state_mean=dset.state_mean,
        state_std=dset.state_std,
        proprio_mean=dset.proprio_mean,
        proprio_std=dset.proprio_std,
        transform=dset.transform,
    )


def _load_sparse_model_for_maps(args):
    import torch
    from omegaconf import OmegaConf, open_dict
    from plan import load_model

    if args.ckpt_base_path is None or args.model_name is None:
        raise ValueError("--ckpt-base-path and --model-name are required with --visualize-drs-fdbu")
    if args.drs_head_checkpoint is None:
        raise ValueError("--drs-head-checkpoint is required with --visualize-drs-fdbu")

    model_dir = Path(args.ckpt_base_path) / "outputs" / args.model_name
    cfg_path = model_dir / "hydra.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Model hydra config not found: {cfg_path}")
    train_cfg = OmegaConf.load(cfg_path)
    with open_dict(train_cfg):
        if args.data_path is not None:
            train_cfg.env.dataset.data_path = args.data_path
        if args.object_name is not None:
            train_cfg.env.kwargs.object_name = args.object_name
            train_cfg.env.dataset.object_name = args.object_name
    runtime_cfg = {
        "load_decoder": False,
        "use_drs": True,
        "drs_mode": args.drs_mode,
        "drs_keep_ratio": args.drs_keep_ratio,
        "drs_topk": args.drs_topk,
        "drs_mask_type": "mask",
        "drs_head_checkpoint": args.drs_head_checkpoint,
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
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")
    device = torch.device(args.device)
    model = load_model(
        ckpt_path,
        train_cfg,
        train_cfg.num_action_repeat,
        device=device,
        roi_runtime_cfg=runtime_cfg,
    )
    model.eval()
    preprocessor = _make_deformable_preprocessor(
        train_cfg=train_cfg,
        data_path=args.data_path,
        object_name=args.object_name,
    )
    return model, preprocessor, device, train_cfg


def _transform_single_obs(preprocessor, raw_obs: Dict[str, np.ndarray], device):
    import torch
    from utils import move_to_device

    batched = {
        key: np.expand_dims(np.expand_dims(np.asarray(value), axis=0), axis=1)
        for key, value in raw_obs.items()
    }
    return move_to_device(preprocessor.transform_obs(batched), device)


def _compute_drs_fdbu_maps_for_obs(model, obs, actions, timestep: int = 0) -> Dict:
    import torch
    from models.grouped_tokens import GroupedTokenProcessor

    processor = getattr(model, "sparse_primary_dynamics", None)
    if processor is None:
        raise ValueError("Model has no sparse_primary_dynamics module.")
    if getattr(processor, "fdbu", None) is None:
        raise ValueError("background_processor must be fdbu to visualize FDBU updates.")
    with torch.no_grad():
        z_obs = model.encode_obs(obs)
        z_act = model.encode_act(actions)
        z = model.compose_z(z_obs["visual"], z_obs["proprio"], z_act)
        foreground_mask, sparse_stats = model._select_sparse_foreground_mask(z)
        sep_obs, sep_act = model.separate_emb(z)
        score_grid = None
        selector = getattr(model, "roi_selector", None)
        if selector is not None and getattr(selector, "roi_head", None) is not None:
            scores = selector.roi_head(
                sep_obs["visual"],
                action=sep_act,
                proprio=sep_obs["proprio"],
            )
            score_grid = _as_square_grid(scores[0, timestep])
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
        bg_delta_mag = (fdbu_fill - z).norm(dim=-1).masked_fill(foreground_mask, 0.0)
        return {
            "mask_grid": _as_square_grid(foreground_mask[0, timestep].float()).astype(bool),
            "score_grid": score_grid,
            "fdbu_delta_grid": _as_square_grid(bg_delta_mag[0, timestep]),
            "sparse_stats": sparse_stats,
        }


def _visualize_drs_fdbu_for_selected_seed(
    args,
    targets: Dict,
    eval_id: int,
    out_dir: Path,
    replay_obses: Optional[Dict] = None,
) -> Optional[Dict]:
    if not args.visualize_drs_fdbu:
        return None
    import torch

    if "obs_0" not in targets:
        raise KeyError("plan_targets.pkl does not contain obs_0; cannot render DRS/FDBU maps.")
    model, preprocessor, device, _ = _load_sparse_model_for_maps(args)
    payload = _load_planned_actions(args.plan_dir)
    if payload is None:
        raise FileNotFoundError(f"Missing planned_actions.pt in {args.plan_dir}")
    planned_actions = payload["actions"][eval_id : eval_id + 1].float()
    raw_initial_obs = {
        key: np.asarray(value[eval_id, 0]) for key, value in targets["obs_0"].items()
    }

    map_iters = []
    for value in str(args.map_mpc_iters).split(","):
        value = value.strip()
        if value:
            map_iters.append(int(value))
    if not map_iters:
        map_iters = [0]
    if replay_obses is None and any(mpc_iter > 0 for mpc_iter in map_iters):
        raise ValueError("--map-mpc-iters beyond 0 requires --replay-actions.")

    map_dir = out_dir / "drs_fdbu_maps"
    map_dir.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for mpc_iter in map_iters:
        action_start = int(mpc_iter) * int(args.n_taken_actions)
        if action_start >= planned_actions.shape[1]:
            continue
        action_end = min(action_start + 1, planned_actions.shape[1])
        act = planned_actions[:, action_start:action_end].to(device)
        if replay_obses is not None and mpc_iter > 0:
            obs_step = min(
                action_start * int(args.frameskip),
                int(np.asarray(replay_obses["visual"]).shape[0]) - 1,
            )
            raw_obs = {
                key: np.asarray(value[obs_step])
                for key, value in replay_obses.items()
                if key in {"visual", "proprio"}
            }
        else:
            raw_obs = raw_initial_obs
        obs = _transform_single_obs(preprocessor, raw_obs, device)
        maps = _compute_drs_fdbu_maps_for_obs(model, obs, act, timestep=0)
        image = raw_obs["visual"]
        prefix = f"eval{eval_id:03d}_mpc{mpc_iter:02d}"
        mask_path = map_dir / f"{prefix}_drs_mask_overlay.png"
        fdbu_path = map_dir / f"{prefix}_fdbu_bg_update.png"
        _draw_token_mask_overlay(image, maps["mask_grid"], mask_path)
        _draw_token_heatmap(
            maps["fdbu_delta_grid"],
            maps["mask_grid"],
            fdbu_path,
            title="FDBU background update |Delta z|",
            cmap="magma",
        )
        entry = {
            "drs_mask_overlay": str(mask_path),
            "fdbu_bg_update": str(fdbu_path),
            "sparse_stats": maps["sparse_stats"],
            "num_selected_tokens": int(maps["mask_grid"].sum()),
        }
        if maps["score_grid"] is not None:
            score_path = map_dir / f"{prefix}_drs_score_heatmap.png"
            _draw_token_heatmap(
                _normalize_grid(maps["score_grid"]),
                maps["mask_grid"],
                score_path,
                title="DRS score",
                cmap="viridis",
            )
            entry["drs_score_heatmap"] = str(score_path)
        outputs[str(mpc_iter)] = entry
    return outputs


def _select_eval_id(args, rows: List[Dict], targets: Dict) -> int:
    if args.eval_id is not None:
        return int(args.eval_id)
    n_evals = len(targets["state_0"])
    if args.select_worst and rows:
        matrix = np.asarray([row["cd"] for row in rows], dtype=np.float32)
        return int(np.argmax(matrix.mean(axis=0)))
    return int(n_evals - args.rank_from_end)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize deformable Rope/Granular per-seed planning diagnostics. "
            "New runs can be read from mpc_iterations.jsonl; old runs can pass --log-file."
        )
    )
    parser.add_argument("--plan-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--log-file", type=Path, default=None, help="Optional stdout log from an old run.")
    parser.add_argument("--eval-id", type=int, default=None)
    parser.add_argument("--rank-from-end", type=int, default=4)
    parser.add_argument("--select-worst", action="store_true")
    parser.add_argument("--eval-seeds", type=str, default=None, help="Comma-separated seed list.")
    parser.add_argument("--object-name", choices=["rope", "granular"], default=None)
    parser.add_argument("--replay-actions", action="store_true")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument("--n-taken-actions", type=int, default=5)
    parser.add_argument("--normalize-action", action="store_true", default=True)
    parser.add_argument("--no-normalize-action", dest="normalize_action", action="store_false")
    parser.add_argument("--save-replay-frames", action="store_true")
    parser.add_argument(
        "--visualize-drs-fdbu",
        action="store_true",
        help="Also render DRS mask/score and FDBU background update maps for the selected seed.",
    )
    parser.add_argument("--ckpt-base-path", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--model-epoch", type=str, default="latest")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--drs-head-checkpoint", type=str, default=None)
    parser.add_argument("--drs-mode", type=str, default="distilled_topk")
    parser.add_argument("--drs-keep-ratio", type=float, default=1.0)
    parser.add_argument("--drs-topk", type=int, default=32)
    parser.add_argument("--sparse-mask-source", type=str, default="drs")
    parser.add_argument("--background-processor", type=str, default="fdbu")
    parser.add_argument("--fdbu-hidden-mult", type=float, default=2.0)
    parser.add_argument("--fdbu-alpha", type=float, default=1.0)
    parser.add_argument("--fdbu-dropout", type=float, default=0.0)
    parser.add_argument(
        "--map-mpc-iters",
        type=str,
        default="0",
        help="Comma-separated MPC iters for DRS/FDBU maps. Later iters need --replay-actions.",
    )
    args = parser.parse_args()

    plan_dir = args.plan_dir
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = _load_targets(plan_dir)
    rows, n_evals = _extract_cd_from_mpc_jsonl(plan_dir)
    eval_seeds = None
    if args.log_file is not None:
        log_rows, log_seeds = _extract_cd_from_stdout_log(args.log_file)
        if log_rows:
            rows = log_rows
            n_evals = len(log_rows[0]["cd"])
        eval_seeds = log_seeds
    if args.eval_seeds:
        eval_seeds = [int(x.strip()) for x in args.eval_seeds.split(",") if x.strip()]
    if eval_seeds is None and "eval_seed" in targets:
        saved_seeds = targets.get("eval_seed")
        if isinstance(saved_seeds, (list, tuple)):
            eval_seeds = [int(seed) for seed in saved_seeds]

    eval_id = _select_eval_id(args, rows, targets)
    if eval_id < 0 or eval_id >= len(targets["state_0"]):
        raise ValueError(
            f"eval_id={eval_id} out of range for {len(targets['state_0'])} eval targets"
        )
    eval_seed = eval_seeds[eval_id] if eval_seeds and eval_id < len(eval_seeds) else None

    init = _to_particle_state(targets["state_0"][eval_id])
    goal = _to_particle_state(targets["state_g"][eval_id])

    if rows:
        _plot_cd_curves(rows, eval_id, eval_seed, out_dir / "cd_curves.png")
        _plot_cd_heatmap(rows, out_dir / "cd_heatmap.png", eval_id)
    else:
        print(
            "No per-seed CD trajectory found. Pass --log-file for old runs, "
            "or rerun plan after this patch."
        )

    summaries = _write_seed_summary(rows, targets, eval_seeds, out_dir / "seed_diagnostics.csv")
    _plot_initial_goal(init, goal, eval_id, eval_seed, out_dir / "initial_goal_overlay.png")

    replay_summary = None
    replay_obses = None
    if args.replay_actions:
        if args.object_name is None:
            raise ValueError("--object-name is required with --replay-actions")
        if args.data_path is None:
            raise ValueError("--data-path is required with --replay-actions")
        if eval_seed is None:
            raise ValueError("--eval-seeds or a log with eval_seed is required with --replay-actions")
        payload = _load_planned_actions(plan_dir)
        if payload is None:
            raise FileNotFoundError(f"Missing planned_actions.pt in {plan_dir}")
        actions = payload["actions"][eval_id]
        action_mean, action_std = _load_deform_action_stats(
            data_path=args.data_path,
            object_name=args.object_name,
            normalize_action=args.normalize_action,
        )
        exec_actions = _denormalize_deform_actions(
            actions,
            frameskip=args.frameskip,
            action_mean=action_mean,
            action_std=action_std,
        )
        replay_obses, states = _replay_deformable(
            object_name=args.object_name,
            seed=int(eval_seed),
            init_state=init,
            exec_actions=exec_actions,
        )
        frames_dir = out_dir / "replay_frames" if args.save_replay_frames else None
        _plot_replay_states(
            states=states,
            goal=goal,
            rows=rows,
            n_taken_actions=args.n_taken_actions,
            frameskip=args.frameskip,
            out_path=out_dir / "replay_states.png",
            save_frames_dir=frames_dir,
        )
        _plot_action_norms(
            exec_actions=exec_actions,
            n_taken_actions=args.n_taken_actions,
            frameskip=args.frameskip,
            out_path=out_dir / "action_norms.png",
        )
        replay_summary = {
            "num_replay_states": int(len(states)),
            "num_exec_actions": int(len(exec_actions)),
            "replay_final_cd": _chamfer_np(states[-1], goal),
        }

    drs_fdbu_outputs = _visualize_drs_fdbu_for_selected_seed(
        args=args,
        targets=targets,
        eval_id=eval_id,
        out_dir=out_dir,
        replay_obses=replay_obses,
    )

    target_row = summaries[eval_id]
    report = {
        "plan_dir": str(plan_dir),
        "eval_id": int(eval_id),
        "eval_seed": eval_seed,
        "num_logged_mpc_iters": len(rows),
        "num_logged_evals": n_evals,
        "target_summary": target_row,
        "replay_summary": replay_summary,
        "drs_fdbu_outputs": drs_fdbu_outputs,
        "outputs": {
            "cd_curves": str(out_dir / "cd_curves.png") if rows else None,
            "cd_heatmap": str(out_dir / "cd_heatmap.png") if rows else None,
            "initial_goal_overlay": str(out_dir / "initial_goal_overlay.png"),
            "seed_diagnostics_csv": str(out_dir / "seed_diagnostics.csv"),
            "replay_states": str(out_dir / "replay_states.png") if replay_summary else None,
            "action_norms": str(out_dir / "action_norms.png") if replay_summary else None,
        },
    }
    with (out_dir / "diagnostic_summary.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
