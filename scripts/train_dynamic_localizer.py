import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.dynamic_localizer import (
    HistoryFusionDynamicLocalizer,
    ddpwm_pixel_change_scores_and_mask,
)
from plan import load_model
from utils import move_to_device


def _checkpoint_path(base_path: str, model_name: str, model_epoch: str) -> Path:
    ckpt_name = "model_latest.pth" if str(model_epoch) == "latest" else f"model_{model_epoch}.pth"
    return Path(base_path) / "outputs" / model_name / "checkpoints" / ckpt_name


def _train_config_path(base_path: str, model_name: str) -> Path:
    return Path(base_path) / "outputs" / model_name / "hydra.yaml"


def _prepare_images_for_label(model, visual: torch.Tensor, current_idx: int, next_idx: int):
    current = visual[:, current_idx]
    next_frame = visual[:, next_idx]
    bsz = int(current.shape[0])
    images = torch.cat([current, next_frame], dim=0)
    images = model.encoder_transform(images)
    current_t, next_t = images[:bsz], images[bsz:]
    return current_t, next_t


@torch.no_grad()
def _encode_batch(model, obs: Dict[str, torch.Tensor], act: torch.Tensor):
    z_obs = model.encode_obs(obs)
    act_emb = model.encode_act(act)
    visual_history = z_obs["visual"][:, : model.num_hist]
    action_history = act_emb[:, : model.num_hist]
    proprio_history = z_obs["proprio"][:, : model.num_hist]
    return visual_history, action_history, proprio_history


def _batch_target(model, obs: Dict[str, torch.Tensor], args) -> torch.Tensor:
    current_idx = int(model.num_hist) - 1
    next_idx = current_idx + int(model.num_pred)
    if obs["visual"].shape[1] <= next_idx:
        raise ValueError(
            "Dynamic localizer training requires a sample with at least "
            f"{next_idx + 1} visual frames, got {obs['visual'].shape[1]}."
        )
    current_t, next_t = _prepare_images_for_label(
        model,
        obs["visual"],
        current_idx=current_idx,
        next_idx=next_idx,
    )
    side = int(args.grid_size)
    _, target = ddpwm_pixel_change_scores_and_mask(
        current_t,
        next_t,
        grid_h=side,
        grid_w=side,
        threshold=args.pixel_threshold,
        partition_precision=args.partition_precision,
    )
    return target


def _metrics_from_logits(logits: torch.Tensor, target: torch.Tensor, threshold: float) -> Dict[str, float]:
    pred = torch.sigmoid(logits) > float(threshold)
    target = target.bool()
    tp = torch.logical_and(pred, target).sum().float()
    fp = torch.logical_and(pred, ~target).sum().float()
    fn = torch.logical_and(~pred, target).sum().float()
    union = torch.logical_or(pred, target).sum().float()
    return {
        "pos_ratio": float(target.float().mean().detach().cpu().item()),
        "pred_pos_ratio": float(pred.float().mean().detach().cpu().item()),
        "iou": float((tp / union.clamp_min(1.0)).detach().cpu().item()),
        "precision": float((tp / (tp + fp).clamp_min(1.0)).detach().cpu().item()),
        "recall": float((tp / (tp + fn).clamp_min(1.0)).detach().cpu().item()),
    }


def _mean_metrics(rows: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(rows)
    if not rows:
        return {}
    keys = rows[0].keys()
    return {key: sum(float(row[key]) for row in rows) / len(rows) for key in keys}


def _make_loader(dataset, args, split: str, shuffle: bool):
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        collate_fn=None,
        pin_memory=args.pin_memory,
    )


def _run_epoch(
    *,
    split: str,
    localizer: HistoryFusionDynamicLocalizer,
    teacher_model,
    loader,
    optimizer: Optional[torch.optim.Optimizer],
    args,
    device: torch.device,
):
    is_train = optimizer is not None
    localizer.train(is_train)
    losses = []
    metric_rows = []
    iterator = tqdm(loader, desc=f"{split}", leave=False)
    for batch_idx, data in enumerate(iterator):
        max_batches = args.max_batches if is_train else args.val_max_batches
        if max_batches is not None and batch_idx >= max_batches:
            break
        obs, act, _ = data
        obs = move_to_device(obs, device)
        act = act.to(device)
        with torch.no_grad():
            visual_history, action_history, proprio_history = _encode_batch(
                teacher_model,
                obs,
                act,
            )
            target = _batch_target(teacher_model, obs, args).to(device)

        logits = localizer(
            visual_history=visual_history,
            action_history=action_history,
            proprio_history=proprio_history,
        )
        pos_weight = torch.tensor(float(args.positive_weight), device=device)
        loss = F.binary_cross_entropy_with_logits(
            logits,
            target.float(),
            pos_weight=pos_weight,
        )
        if is_train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(localizer.parameters(), args.grad_clip)
            optimizer.step()

        losses.append(float(loss.detach().cpu().item()))
        metrics = _metrics_from_logits(logits.detach(), target, args.pred_threshold)
        metric_rows.append(metrics)
        iterator.set_postfix(loss=f"{losses[-1]:.4f}", iou=f"{metrics['iou']:.3f}")

    mean = _mean_metrics(metric_rows)
    mean["loss"] = sum(losses) / max(len(losses), 1)
    return mean


def _save_checkpoint(path: Path, localizer, optimizer, metadata, history, epoch):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dynamic_localizer": localizer.cpu().state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "metadata": metadata,
            "history": history,
            "epoch": int(epoch),
        },
        path,
    )


def train_dynamic_localizer(args):
    device = torch.device(args.device)
    train_cfg = OmegaConf.load(args.train_config or _train_config_path(args.ckpt_base_path, args.model_name))
    if args.data_path is not None:
        with open_dict(train_cfg):
            train_cfg.env.dataset.data_path = args.data_path

    teacher_ckpt = args.teacher_checkpoint or _checkpoint_path(
        args.ckpt_base_path,
        args.model_name,
        args.model_epoch,
    )
    teacher_model = load_model(
        Path(teacher_ckpt),
        train_cfg,
        train_cfg.num_action_repeat,
        device=device,
        roi_runtime_cfg={
            "use_roi": False,
            "roi_mode": "none",
            "load_decoder": args.load_decoder,
            "profile_wm_timing": False,
        },
    )
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    datasets, _ = hydra.utils.call(
        train_cfg.env.dataset,
        num_hist=train_cfg.num_hist,
        num_pred=train_cfg.num_pred,
        frameskip=train_cfg.frameskip,
    )
    train_loader = _make_loader(datasets[args.split], args, args.split, shuffle=args.shuffle)
    val_loader = _make_loader(datasets[args.val_split], args, args.val_split, shuffle=False)

    visual_dim = int(getattr(teacher_model.encoder, "emb_dim"))
    with torch.no_grad():
        sample_obs, sample_act, _ = datasets[args.split][0]
        sample_obs_b = {key: value.unsqueeze(0).to(device) for key, value in sample_obs.items()}
        sample_act_b = sample_act.unsqueeze(0).to(device)
        sample_z = teacher_model.encode_obs(sample_obs_b)
        sample_action = teacher_model.encode_act(sample_act_b)
    action_dim = int(sample_action.shape[-1])
    proprio_dim = int(sample_z["proprio"].shape[-1])
    num_tokens = int(sample_z["visual"].shape[-2])

    localizer = HistoryFusionDynamicLocalizer(
        visual_dim=visual_dim,
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        num_hist=int(teacher_model.num_hist),
        num_tokens=num_tokens,
        reduced_dim=args.reduced_dim,
        history_heads=args.history_heads,
        localizer_layers=args.localizer_layers,
        localizer_heads=args.localizer_heads,
        mlp_dim=args.mlp_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        localizer.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    metadata = {
        "type": "history_fusion_dynamic_localizer",
        "teacher_checkpoint": str(teacher_ckpt),
        "train_config": str(args.train_config or _train_config_path(args.ckpt_base_path, args.model_name)),
        "model_name": args.model_name,
        "model_epoch": args.model_epoch,
        "split": args.split,
        "val_split": args.val_split,
        "visual_dim": visual_dim,
        "action_dim": action_dim,
        "proprio_dim": proprio_dim,
        "num_hist": int(teacher_model.num_hist),
        "num_pred": int(teacher_model.num_pred),
        "num_tokens": num_tokens,
        "grid_size": int(args.grid_size),
        "reduced_dim": int(args.reduced_dim),
        "history_heads": int(args.history_heads),
        "localizer_layers": int(args.localizer_layers),
        "localizer_heads": int(args.localizer_heads),
        "mlp_dim": int(args.mlp_dim),
        "dropout": float(args.dropout),
        "pixel_threshold": float(args.pixel_threshold),
        "partition_precision": int(args.partition_precision),
        "positive_weight": float(args.positive_weight),
    }
    output_path = Path(args.output)
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = _run_epoch(
            split="train",
            localizer=localizer,
            teacher_model=teacher_model,
            loader=train_loader,
            optimizer=optimizer,
            args=args,
            device=device,
        )
        val_metrics = _run_epoch(
            split="valid",
            localizer=localizer,
            teacher_model=teacher_model,
            loader=val_loader,
            optimizer=None,
            args=args,
            device=device,
        )
        row = {"epoch": epoch, "train": train_metrics, "valid": val_metrics}
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

        _save_checkpoint(output_path, localizer, optimizer, metadata, history, epoch)
        epoch_path = output_path.with_name(f"{output_path.stem}_epoch_{epoch}{output_path.suffix}")
        _save_checkpoint(epoch_path, localizer, optimizer, metadata, history, epoch)
        localizer.to(device)

    print(f"Saved dynamic localizer checkpoint to {output_path}")


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Train a DDP-WM-style history-fusion dynamic localizer on DINO-WM features."
    )
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--ckpt-base-path", required=True)
    parser.add_argument("--model-epoch", default="latest")
    parser.add_argument("--teacher-checkpoint", default=None)
    parser.add_argument("--train-config", default=None)
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--val-split", default="valid")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--val-max-batches", type=int, default=100)
    parser.add_argument("--shuffle", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--load-decoder", action="store_true")

    parser.add_argument("--pixel-threshold", type=float, default=0.1)
    parser.add_argument("--partition-precision", type=int, default=4)
    parser.add_argument("--positive-weight", type=float, default=10.0)
    parser.add_argument("--pred-threshold", type=float, default=0.5)
    parser.add_argument("--grid-size", type=int, default=14)

    parser.add_argument("--reduced-dim", type=int, default=192)
    parser.add_argument("--history-heads", type=int, default=4)
    parser.add_argument("--localizer-layers", type=int, default=3)
    parser.add_argument("--localizer-heads", type=int, default=4)
    parser.add_argument("--mlp-dim", type=int, default=768)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    return parser


def main():
    train_dynamic_localizer(build_argparser().parse_args())


if __name__ == "__main__":
    main()
