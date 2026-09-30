import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

from models.roi import ActionAwareDRSHead, drs_distillation_loss


def _flatten_time(x: torch.Tensor) -> torch.Tensor:
    if x.ndim >= 3:
        return x.reshape(x.shape[0] * x.shape[1], *x.shape[2:])
    return x


def load_drs_target_dataset(target_path: str) -> Tuple[TensorDataset, Dict]:
    payload = torch.load(target_path, map_location="cpu")
    records = payload.get("records", payload if isinstance(payload, list) else [])
    if not records:
        raise ValueError(f"No DRS target records found in {target_path}")

    visual_tokens = torch.cat([record["visual_tokens"] for record in records], dim=0)
    target_tensors = []
    for record in records:
        if "drs_target" in record:
            target_tensors.append(record["drs_target"])
        else:
            target_tensors.append(record["roi_target"])
    targets = torch.cat(target_tensors, dim=0)
    actions = torch.cat([record["action"] for record in records], dim=0)
    proprios = torch.cat([record["proprio"] for record in records], dim=0)

    visual_tokens = _flatten_time(visual_tokens).float()
    targets = _flatten_time(targets).float()
    actions = _flatten_time(actions).float()
    proprios = _flatten_time(proprios).float()

    dataset = TensorDataset(visual_tokens, actions, proprios, targets)
    metadata = payload.get("metadata", {})
    metadata.update(
        {
            "num_samples": len(dataset),
            "token_dim": visual_tokens.shape[-1],
            "action_dim": actions.shape[-1],
            "proprio_dim": proprios.shape[-1],
            "num_tokens": visual_tokens.shape[-2],
        }
    )
    return dataset, metadata


load_roi_target_dataset = load_drs_target_dataset


def seed_everything(seed: int) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def drs_head_train_step(
    drs_head: ActionAwareDRSHead,
    batch,
    optimizer=None,
    loss_type: str = "kl",
    temperature: float = 1.0,
    device: str = "cpu",
) -> torch.Tensor:
    visual_tokens, action, proprio, target = [x.to(device) for x in batch]
    pred_scores = drs_head(visual_tokens, action=action, proprio=proprio)
    loss = drs_distillation_loss(
        pred_scores,
        target,
        loss_type=loss_type,
        temperature=temperature,
    )
    if optimizer is not None:
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return loss.detach()


roi_head_train_step = drs_head_train_step


def train_drs_head(args):
    device = torch.device(args.device)
    generator = seed_everything(int(args.seed))
    dataset, metadata = load_drs_target_dataset(args.targets)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=generator,
    )
    metadata["seed"] = int(args.seed)

    drs_head = ActionAwareDRSHead(
        token_dim=metadata["token_dim"],
        hidden_dim=args.hidden_dim,
        action_dim=metadata["action_dim"] if args.action_conditioned else None,
        proprio_dim=metadata["proprio_dim"] if args.action_conditioned else None,
    ).to(device)
    optimizer = torch.optim.AdamW(
        drs_head.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    history = []
    for epoch in range(args.epochs):
        losses = []
        drs_head.train()
        for batch in loader:
            loss = drs_head_train_step(
                drs_head,
                batch,
                optimizer=optimizer,
                loss_type=args.loss_type,
                temperature=args.temperature,
                device=device,
            )
            losses.append(float(loss.cpu().item()))
        mean_loss = sum(losses) / max(len(losses), 1)
        history.append({"epoch": epoch + 1, "loss": mean_loss})
        print(f"epoch={epoch + 1} drs_head_loss={mean_loss:.6f}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state_dict = drs_head.cpu().state_dict()
    torch.save(
        {
            "drs_head": state_dict,
            "roi_head": state_dict,  # legacy key consumed by older checkpoints/loaders
            "metadata": metadata,
            "history": history,
            "loss_type": args.loss_type,
            "temperature": args.temperature,
            "action_conditioned": args.action_conditioned,
            "seed": int(args.seed),
        },
        output_path,
    )
    print(f"Saved DRS head checkpoint to {output_path}")


train_roi_head = train_drs_head


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Train ActionAwareDRSHead from exported teacher DRS targets."
    )
    parser.add_argument("--targets", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--loss-type", default="kl", choices=["kl", "bce", "mse", "rank"])
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--action-conditioned",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device", default="cpu")
    return parser


def main():
    train_drs_head(build_argparser().parse_args())


if __name__ == "__main__":
    main()
