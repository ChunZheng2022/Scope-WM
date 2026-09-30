import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch import nn

from models.visual_world_model import VWorldModel
from models.vit import ViTPredictor


class DummyEncoder(nn.Module):
    name = "dummy"
    emb_dim = 384
    patch_size = 16

    def forward(self, x):
        return torch.zeros(x.shape[0], 196, self.emb_dim, device=x.device)


class DummyEmbedding(nn.Module):
    emb_dim = 10

    def forward(self, x):
        return x[..., : self.emb_dim]


def build_model(background_processor: str, device):
    predictor = ViTPredictor(
        num_patches=196,
        num_frames=3,
        dim=404,
        depth=1,
        heads=4,
        mlp_dim=128,
        dim_head=16,
    )
    return VWorldModel(
        image_size=224,
        num_hist=3,
        num_pred=1,
        encoder=DummyEncoder(),
        proprio_encoder=DummyEmbedding(),
        action_encoder=DummyEmbedding(),
        decoder=None,
        predictor=predictor,
        proprio_dim=10,
        action_dim=10,
        concat_dim=1,
        num_action_repeat=1,
        num_proprio_repeat=1,
        train_encoder=False,
        train_predictor=True,
        train_decoder=False,
        use_roi=True,
        roi_mode="random",
        roi_config={
            "use_drs": True,
            "drs_mode": "random",
            "drs_keep_ratio": 0.5,
            "drs_mask_type": "mask",
        },
        sparse_dynamics_config={
            "enabled": True,
            "mode": "sparse_primary",
            "mask_source": "drs",
            "background_processor": background_processor,
            "background_group_size": 2,
            "ru_rank": 8,
            "lrm_heads": 4,
            "loss_on_foreground_only": True,
        },
    ).to(device)


def main():
    torch.manual_seed(0)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for background_processor in ["none", "ignore", "grouped", "ru", "lrm", "fdbu"]:
        model = build_model(background_processor, device)
        z = torch.randn(2, 3, 196, 404, device=device)
        out = model.predict(z)
        stats = model.get_last_roi_stats()
        assert out.shape == z.shape, (background_processor, out.shape, z.shape)
        assert stats["sparse_dynamics_enabled"] is True, stats
        assert stats["sparse_background_processor"] == background_processor, stats
        assert stats["sparse_num_foreground_tokens"] > 0, stats
        if background_processor in {"ru", "lrm", "fdbu"}:
            trainable = sum(
                p.numel()
                for p in model.sparse_primary_dynamics.parameters()
                if p.requires_grad
            )
            assert trainable > 0, trainable
    print("sparse primary dynamics smoke test passed for none/ignore/grouped/ru/lrm/fdbu")


if __name__ == "__main__":
    main()
