import os
from pathlib import Path
import torch
import torch.nn as nn

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

def _resolve_dinov2_hub_source():
    repo_override = os.environ.get("DINOV2_LOCAL_PATH")
    source_override = os.environ.get("DINOV2_SOURCE")
    if repo_override:
        return repo_override, source_override or "local"

    project_root = Path(__file__).resolve().parents[1]
    local_repo = project_root / "dinov2_local"
    if local_repo.exists():
        return str(local_repo), "local"

    raise FileNotFoundError(
        "DINOv2 local source not found. Place it at dinov2_local/ or set "
        "DINOV2_LOCAL_PATH to a local torch.hub-compatible directory."
    )


class DinoV2Encoder(nn.Module):
    def __init__(self, name, feature_key):
        super().__init__()
        self.name = name
        repo_or_dir, source = _resolve_dinov2_hub_source()
        hub_kwargs = {"pretrained": True}
        if source is not None:
            hub_kwargs["source"] = source
        self.base_model = torch.hub.load(
            repo_or_dir,
            name,
            **hub_kwargs,
        )
        self.feature_key = feature_key
        self.emb_dim = self.base_model.num_features
        if feature_key == "x_norm_patchtokens":
            self.latent_ndim = 2
        elif feature_key == "x_norm_clstoken":
            self.latent_ndim = 1
        else:
            raise ValueError(f"Invalid feature key: {feature_key}")

        self.patch_size = self.base_model.patch_size

    def forward(self, x):
        emb = self.base_model.forward_features(x)[self.feature_key]
        if self.latent_ndim == 1:
            emb = emb.unsqueeze(1) # dummy patch dim
        return emb
