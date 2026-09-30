import argparse
import sys
from itertools import islice
from pathlib import Path
from typing import Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf, open_dict
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.roi import grad_x_input_token_target
from plan import load_model
from utils import move_to_device


def compute_grad_x_input_target_from_loss(
    visual_tokens: torch.Tensor,
    loss: torch.Tensor,
    token_slice: Optional[slice] = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    return grad_x_input_token_target(
        visual_tokens,
        loss,
        token_slice=token_slice,
        eps=eps,
    )


def _teacher_z_loss(model, z_pred: torch.Tensor, z_tgt: torch.Tensor) -> torch.Tensor:
    if model.concat_dim == 0:
        return F.mse_loss(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
    return F.mse_loss(z_pred[:, :, :, :-model.action_dim], z_tgt[:, :, :, :-model.action_dim].detach())


def grad_x_input_targets_for_batch(
    model,
    obs: Dict[str, torch.Tensor],
    act: torch.Tensor,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
    z_dct = model.encode_obs(obs)
    visual_tokens = z_dct["visual"].detach().requires_grad_(True)
    proprio_emb = z_dct["proprio"].detach()
    act_emb = model.encode_act(act).detach()
    z = model.compose_z(visual_tokens, proprio_emb, act_emb)
    z_src = z[:, : model.num_hist, :, :]
    z_tgt = z[:, model.num_pred :, :, :]

    roi_selector = getattr(model, "roi_selector", None)
    drs_selector = getattr(model, "drs_selector", None)
    model.roi_selector = None
    model.drs_selector = None
    try:
        z_pred = model.predict(z_src)
    finally:
        model.roi_selector = roi_selector
        model.drs_selector = drs_selector

    loss = _teacher_z_loss(model, z_pred, z_tgt)
    target = compute_grad_x_input_target_from_loss(
        visual_tokens,
        loss,
        token_slice=(slice(None), slice(0, model.num_hist), slice(None)),
    )
    features = {
        "visual_tokens": visual_tokens[:, : model.num_hist].detach(),
        "action": act_emb[:, : model.num_hist].detach(),
        "proprio": proprio_emb[:, : model.num_hist].detach(),
    }
    return target.detach(), features, loss.detach()


def _save_records(records, output_path, metadata):
    payload = {"metadata": metadata, "records": records}
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def export_targets(args):
    device = torch.device(args.device)
    print(f"Loading train config from {args.train_config}", flush=True)
    train_cfg = OmegaConf.load(args.train_config)
    if args.data_path is not None:
        with open_dict(train_cfg):
            train_cfg.env.dataset.data_path = args.data_path
    print(f"Loading teacher checkpoint from {args.teacher_checkpoint}", flush=True)
    model = load_model(
        Path(args.teacher_checkpoint),
        train_cfg,
        train_cfg.num_action_repeat,
        device=device,
        roi_runtime_cfg={
            "use_drs": False,
            "drs_mode": "none",
            "use_roi": False,
            "roi_mode": "none",
            "load_decoder": args.load_decoder,
        },
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    print(f"Loading dataset from {train_cfg.env.dataset.data_path}", flush=True)
    datasets, _ = hydra.utils.call(
        train_cfg.env.dataset,
        num_hist=train_cfg.num_hist,
        num_pred=train_cfg.num_pred,
        frameskip=train_cfg.frameskip,
    )
    dset = datasets[args.split]
    loader = DataLoader(
        dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=None,
    )

    total_batches = len(loader)
    if args.max_batches is not None:
        total_batches = min(total_batches, args.max_batches)
    print(
        f"Exporting DRS targets: split={args.split}, samples={len(dset)}, "
        f"batch_size={args.batch_size}, batches={total_batches}",
        flush=True,
    )

    records = []
    batch_iter = loader if args.max_batches is None else islice(loader, args.max_batches)
    for batch_idx, data in tqdm(
        enumerate(batch_iter),
        total=total_batches,
        desc="Export DRS targets",
    ):
        obs, act, _ = data
        obs = move_to_device(obs, device)
        act = act.to(device)
        target, features, loss = grad_x_input_targets_for_batch(model, obs, act)
        if args.log_every and (batch_idx + 1) % args.log_every == 0:
            tqdm.write(f"batch={batch_idx} teacher_loss={float(loss.cpu().item()):.6f}")
        records.append(
            {
                "sample_id": f"{args.split}_batch_{batch_idx}",
                "batch_idx": batch_idx,
                "drs_target": target.cpu(),
                "timestep": None,
                "roi_target": target.cpu(),  # legacy key consumed by older scripts
                "visual_tokens": features["visual_tokens"].cpu(),
                "action": features["action"].cpu(),
                "proprio": features["proprio"].cpu(),
                "token_shape": list(target.shape),
                "target_type": "grad_x_input",
                "normalization": "sum_to_one_per_sample_timestep",
                "teacher_loss": float(loss.cpu().item()),
            }
        )

    metadata = {
        "target_type": "grad_x_input",
        "teacher_checkpoint": str(args.teacher_checkpoint),
        "train_config": str(args.train_config),
        "data_path": args.data_path or str(train_cfg.env.dataset.data_path),
        "split": args.split,
        "num_records": len(records),
    }
    _save_records(records, args.output, metadata)
    print(f"Saved {len(records)} DRS target batches to {args.output}")


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Export action-conditioned DRS distillation targets from a full-token DINO-WM teacher."
    )
    parser.add_argument("--teacher-checkpoint", required=True)
    parser.add_argument("--train-config", required=True)
    parser.add_argument(
        "--data-path",
        default=None,
        help="Override train_config env.dataset.data_path, useful for subset datasets.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Print teacher loss every N exported batches. 0 disables per-batch logging.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--load-decoder",
        action="store_true",
        help="Load the decoder if the checkpoint/config provides one. Not needed for grad_x_input targets.",
    )
    parser.add_argument(
        "--target-type",
        default="grad_x_input",
        choices=["grad_x_input", "occlusion", "rollout_cost"],
    )
    return parser


def main():
    args = build_argparser().parse_args()
    if args.target_type != "grad_x_input":
        raise NotImplementedError(
            f"{args.target_type} target export is reserved for a later pass; grad_x_input is implemented."
        )
    export_targets(args)


if __name__ == "__main__":
    main()
