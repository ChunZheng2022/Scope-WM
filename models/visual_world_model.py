import torch
import torch.nn as nn
from torchvision import transforms
from contextlib import contextmanager
from einops import rearrange, repeat
from eval_logging import elapsed_since, perf_counter, sync_cuda_if_available
from models.dynamic_localizer import load_dynamic_localizer_checkpoint
from models.grouped_tokens import GroupedTokenProcessor
from models.pruned_tokens import PrunedTokenProcessor
from models.roi import DRSConfig, DRSSelector, topk_token_mask
from models.sparse_dynamics import SparseDynamicsConfig, SparsePrimaryDynamics

class VWorldModel(nn.Module):
    def __init__(
        self,
        image_size,  # 224
        num_hist,
        num_pred,
        encoder,
        proprio_encoder,
        action_encoder,
        decoder,
        predictor,
        proprio_dim=0,
        action_dim=0,
        concat_dim=0,
        num_action_repeat=7,
        num_proprio_repeat=7,
        train_encoder=True,
        train_predictor=False,
        train_decoder=True,
        use_roi=False,
        roi_mode="none",
        roi_config=None,
        roi_head_checkpoint=None,
        sparse_dynamics_config=None,
        profile_wm_timing=True,
        profile_wm_timing_sync_cuda=False,
    ):
        super().__init__()
        self.num_hist = num_hist
        self.num_pred = num_pred
        self.encoder = encoder
        self.proprio_encoder = proprio_encoder
        self.action_encoder = action_encoder
        self.decoder = decoder  # decoder could be None
        self.predictor = predictor  # predictor could be None
        self.train_encoder = train_encoder
        self.train_predictor = train_predictor
        self.train_decoder = train_decoder
        self.num_action_repeat = num_action_repeat
        self.num_proprio_repeat = num_proprio_repeat
        self.proprio_dim = proprio_dim * num_proprio_repeat 
        self.action_dim = action_dim * num_action_repeat 
        self.emb_dim = self.encoder.emb_dim + (self.action_dim + self.proprio_dim) * (concat_dim) # Not used

        print(f"num_action_repeat: {self.num_action_repeat}")
        print(f"num_proprio_repeat: {self.num_proprio_repeat}")
        print(f"proprio encoder: {proprio_encoder}")
        print(f"action encoder: {action_encoder}")
        print(f"proprio_dim: {proprio_dim}, after repeat: {self.proprio_dim}")
        print(f"action_dim: {action_dim}, after repeat: {self.action_dim}")
        print(f"emb_dim: {self.emb_dim}")

        self.concat_dim = concat_dim # 0 or 1
        assert concat_dim == 0 or concat_dim == 1, f"concat_dim {concat_dim} not supported."
        print("Model emb_dim: ", self.emb_dim)

        if "dino" in self.encoder.name:
            decoder_scale = 16  # from vqvae
            num_side_patches = image_size // decoder_scale
            self.encoder_image_size = num_side_patches * encoder.patch_size
            self.encoder_transform = transforms.Compose(
                [transforms.Resize(self.encoder_image_size)]
            )
        else:
            # set self.encoder_transform to identity transform
            self.encoder_transform = lambda x: x

        self.decoder_criterion = nn.MSELoss()
        self.decoder_latent_loss_weight = 0.25
        self.emb_criterion = nn.MSELoss()
        self.roi_config = DRSConfig.from_any(roi_config)
        self.drs_config = self.roi_config
        self.roi_config.use_roi = bool(use_roi)
        self.roi_config.roi_mode = roi_mode or self.roi_config.roi_mode
        self.roi_selector = None
        self.drs_selector = None
        self.grouped_token_processor = None
        self.pruned_token_processor = None
        self.sparse_dynamics_config = SparseDynamicsConfig.from_any(sparse_dynamics_config)
        self.sparse_primary_dynamics = None
        self.sparse_dynamic_localizer = None
        self._roi_stats_buffer = []
        self.last_prediction_token_mask = None
        self.last_roi_stats = self._default_roi_stats()
        self.profile_wm_timing = bool(profile_wm_timing)
        self.profile_wm_timing_sync_cuda = bool(profile_wm_timing_sync_cuda)
        self.reset_timing_stats()
        if self.roi_config.use_roi and self.roi_config.roi_mode not in {"none", "full"}:
            if self.roi_config.roi_mask_type not in {"mask", "grouped", "prune"}:
                raise ValueError(
                    "VWorldModel DRS integration supports roi_mask_type/drs_mask_type='mask', 'grouped', or 'prune'."
                )
            self.roi_selector = DRSSelector(
                self.roi_config,
                token_dim=self.encoder.emb_dim,
                action_dim=action_dim,
                proprio_dim=proprio_dim,
            )
            self.drs_selector = self.roi_selector
            checkpoint_path = roi_head_checkpoint or self.roi_config.roi_head_checkpoint
            if checkpoint_path is not None:
                self.roi_selector.load_checkpoint(checkpoint_path)
            if self.roi_config.roi_mask_type == "grouped":
                self.grouped_token_processor = GroupedTokenProcessor(
                    group_size=self.roi_config.roi_group_size
                )
            elif self.roi_config.roi_mask_type == "prune":
                self.pruned_token_processor = PrunedTokenProcessor()
        if self.sparse_dynamics_config.enabled:
            if self.concat_dim != 1:
                raise ValueError(
                    "sparse_dynamics currently supports concat_dim=1, where action/proprio are concatenated to each visual patch token."
                )
            if self.sparse_dynamics_config.mode != "sparse_primary":
                raise ValueError(
                    "Only sparse_dynamics.mode='sparse_primary' is implemented in this stage."
                )
            self.sparse_primary_dynamics = SparsePrimaryDynamics(
                dim=self.emb_dim,
                fill_mode=self.sparse_dynamics_config.fill_mode,
                background_processor=self.sparse_dynamics_config.background_processor,
                background_group_size=self.sparse_dynamics_config.background_group_size,
                ru_rank=self.sparse_dynamics_config.ru_rank,
                ru_alpha=self.sparse_dynamics_config.ru_alpha,
                ru_dropout=self.sparse_dynamics_config.ru_dropout,
                lrm_heads=self.sparse_dynamics_config.lrm_heads,
                lrm_num_tokens=self.sparse_dynamics_config.lrm_num_tokens,
                lrm_alpha=self.sparse_dynamics_config.lrm_alpha,
                lrm_dropout=self.sparse_dynamics_config.lrm_dropout,
                fdbu_hidden_dim=self.sparse_dynamics_config.fdbu_hidden_dim,
                fdbu_hidden_mult=self.sparse_dynamics_config.fdbu_hidden_mult,
                fdbu_alpha=self.sparse_dynamics_config.fdbu_alpha,
                fdbu_dropout=self.sparse_dynamics_config.fdbu_dropout,
                cbu_hidden_dim=self.sparse_dynamics_config.cbu_hidden_dim,
                cbu_hidden_mult=self.sparse_dynamics_config.cbu_hidden_mult,
                cbu_alpha=self.sparse_dynamics_config.cbu_alpha,
                cbu_dropout=self.sparse_dynamics_config.cbu_dropout,
            )
            if self._sparse_uses_dynamic_localizer():
                checkpoint_path = self.sparse_dynamics_config.dynamic_localizer_checkpoint
                if checkpoint_path is None:
                    raise ValueError(
                        "sparse_dynamics mask_source requires dynamic_localizer_checkpoint."
                    )
                self.sparse_dynamic_localizer, _ = load_dynamic_localizer_checkpoint(
                    checkpoint_path
                )
                for param in self.sparse_dynamic_localizer.parameters():
                    param.requires_grad = False
                self.sparse_dynamic_localizer.eval()

    def train(self, mode=True):
        super().train(mode)
        if self.train_encoder:
            self.encoder.train(mode)
        if self.predictor is not None and self.train_predictor:
            self.predictor.train(mode)
        self.proprio_encoder.train(mode)
        self.action_encoder.train(mode)
        if self.decoder is not None and self.train_decoder:
            self.decoder.train(mode)
        if self.sparse_dynamic_localizer is not None:
            self.sparse_dynamic_localizer.eval()

    def eval(self):
        super().eval()
        self.encoder.eval()
        if self.predictor is not None:
            self.predictor.eval()
        self.proprio_encoder.eval()
        self.action_encoder.eval()
        if self.decoder is not None:
            self.decoder.eval()
        if self.sparse_dynamic_localizer is not None:
            self.sparse_dynamic_localizer.eval()

    def reset_timing_stats(self):
        self._timing_stats = {}
        self._timing_counts = {}

    def get_timing_stats(self):
        return {
            "times_sec": dict(getattr(self, "_timing_stats", {})),
            "counts": dict(getattr(self, "_timing_counts", {})),
        }

    def _add_timing(self, name, elapsed):
        if elapsed is None:
            return
        self._timing_stats[name] = self._timing_stats.get(name, 0.0) + float(elapsed)
        self._timing_counts[name] = self._timing_counts.get(name, 0) + 1

    def _record_external_timing(self, prefix, timing):
        for key, value in (timing or {}).items():
            if isinstance(value, (int, float)):
                self._add_timing(f"{prefix}_{key}", value)

    @contextmanager
    def _time_section(self, name):
        if not getattr(self, "profile_wm_timing", False):
            yield
            return
        if getattr(self, "profile_wm_timing_sync_cuda", False):
            sync_cuda_if_available()
        start = perf_counter()
        try:
            yield
        finally:
            if getattr(self, "profile_wm_timing_sync_cuda", False):
                sync_cuda_if_available()
            self._add_timing(name, elapsed_since(start))

    def encode(self, obs, act): 
        """
        input :  obs (dict): "visual", "proprio", (b, num_frames, 3, img_size, img_size) 
        output:    z (tensor): (b, num_frames, num_patches, emb_dim)
        """
        with self._time_section("encode_total"):
            z_dct = self.encode_obs(obs)
            act_emb = self.encode_act(act)
            return self.compose_z(z_dct["visual"], z_dct["proprio"], act_emb)

    def compose_z(self, visual_emb, proprio_emb, act_emb):
        with self._time_section("compose_z"):
            if self.concat_dim == 0:
                z = torch.cat(
                        [visual_emb, proprio_emb.unsqueeze(2), act_emb.unsqueeze(2)], dim=2 # add as an extra token
                    )  # (b, num_frames, num_patches + 2, dim)
            if self.concat_dim == 1:
                proprio_tiled = repeat(proprio_emb.unsqueeze(2), "b t 1 a -> b t f a", f=visual_emb.shape[2])
                proprio_repeated = proprio_tiled.repeat(1, 1, 1, self.num_proprio_repeat)
                act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=visual_emb.shape[2])
                act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
                z = torch.cat(
                    [visual_emb, proprio_repeated, act_repeated], dim=3
                )  # (b, num_frames, num_patches, dim + action_dim)
            return z
    
    def encode_act(self, act):
        with self._time_section("action_encoder"):
            act = self.action_encoder(act) # (b, num_frames, action_emb_dim)
            return act
    
    def encode_proprio(self, proprio):
        with self._time_section("proprio_encoder"):
            proprio = self.proprio_encoder(proprio)
            return proprio

    def encode_obs(self, obs):
        """
        input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)
        output:   z (dict): "visual", "proprio" (b, t, num_patches, encoder_emb_dim)
        """
        with self._time_section("encode_obs_total"):
            visual = obs['visual']
            b = visual.shape[0]
            visual = rearrange(visual, "b t ... -> (b t) ...")
            with self._time_section("encoder_transform"):
                visual = self.encoder_transform(visual)
            with self._time_section("visual_encoder"):
                visual_embs = self.encoder.forward(visual)
            visual_embs = rearrange(visual_embs, "(b t) p d -> b t p d", b=b)

            proprio = obs['proprio']
            proprio_emb = self.encode_proprio(proprio)
            return {"visual": visual_embs, "proprio": proprio_emb}

    def predict(self, z):  # in embedding space
        """
        input : z: (b, num_hist, num_patches, emb_dim)
        output: z: (b, num_hist, num_patches, emb_dim)
        """
        with self._time_section("predict_total"):
            if self._use_sparse_primary_dynamics():
                return self.predict_with_sparse_primary_dynamics(z)
            if self._use_grouped_roi():
                return self.predict_with_grouped_roi(z)
            if self._use_pruned_roi():
                return self.predict_with_pruned_roi(z)
            z = self.apply_roi_to_z(z)
            return self._run_predictor_on_z(z)

    def _run_predictor_on_z(self, z, pos_embeddings=None, attn_mask=None):
        with self._time_section("predictor_forward"):
            T = z.shape[1]
            # reshape to a batch of windows of inputs
            z = rearrange(z, "b t p d -> b (t p) d")
            # (b, num_hist * num_patches per img, emb_dim)
            if pos_embeddings is None and attn_mask is None:
                z = self.predictor(z)
            else:
                z = self.predictor(z, pos_embeddings=pos_embeddings, attn_mask=attn_mask)
            z = rearrange(z, "b (t p) d -> b t p d", t=T)
            return z

    def decode(self, z):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        with self._time_section("decode_total"):
            z_obs, z_act = self.separate_emb(z)
            obs, diff = self.decode_obs(z_obs)
            return obs, diff

    def decode_obs(self, z_obs):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        with self._time_section("decode_obs_total"):
            b, num_frames, num_patches, emb_dim = z_obs["visual"].shape
            with self._time_section("decoder_forward"):
                visual, diff = self.decoder(z_obs["visual"])  # (b*num_frames, 3, 224, 224)
            visual = rearrange(visual, "(b t) c h w -> b t c h w", t=num_frames)
            obs = {
                "visual": visual,
                "proprio": z_obs["proprio"], # Note: no decoder for proprio for now!
            }
            return obs, diff
    
    def separate_emb(self, z):
        """
        input: z (tensor)
        output: z_obs (dict), z_act (tensor)
        """
        if self.concat_dim == 0:
            z_visual, z_proprio, z_act = z[:, :, :-2, :], z[:, :, -2, :], z[:, :, -1, :]
        elif self.concat_dim == 1:
            z_visual, z_proprio, z_act = z[..., :-(self.proprio_dim + self.action_dim)], \
                                         z[..., -(self.proprio_dim + self.action_dim) :-self.action_dim],  \
                                         z[..., -self.action_dim:]
            # remove tiled dimensions
            z_proprio = z_proprio[:, :, 0, : self.proprio_dim // self.num_proprio_repeat]
            z_act = z_act[:, :, 0, : self.action_dim // self.num_action_repeat]
        z_obs = {"visual": z_visual, "proprio": z_proprio}
        return z_obs, z_act

    def _default_roi_stats(self):
        return {
            "roi_enabled": False,
            "drs_enabled": False,
            "roi_mode": "none",
            "drs_mode": "none",
            "roi_keep_ratio": None,
            "drs_keep_ratio": None,
            "roi_num_tokens": None,
            "drs_num_tokens": None,
            "roi_num_kept_tokens": None,
            "drs_num_kept_tokens": None,
            "roi_num_roi_tokens": None,
            "roi_num_group_tokens": None,
            "roi_grouped_token_count": None,
            "roi_pruned_token_count": None,
            "roi_num_pruned_tokens": None,
            "roi_prune_fill": None,
            "effective_token_count": None,
            "roi_mask_time_sec": None,
            "drs_mask_time_sec": None,
            "roi_score_mean": None,
            "drs_score_mean": None,
            "roi_score_std": None,
            "drs_score_std": None,
            "roi_score_min": None,
            "drs_score_min": None,
            "roi_score_max": None,
            "drs_score_max": None,
            "roi_target_type": None,
            "drs_target_type": None,
            "roi_action_aware": None,
            "drs_action_aware": None,
            "sparse_dynamics_enabled": False,
            "sparse_dynamics_mode": None,
            "sparse_mask_source": None,
            "sparse_background_processor": None,
            "sparse_num_tokens": None,
            "sparse_num_foreground_tokens": None,
            "sparse_num_background_tokens": None,
            "sparse_fill_mode": None,
        }

    def reset_roi_stats(self):
        self._roi_stats_buffer = []
        self.last_prediction_token_mask = None
        self.last_roi_stats = self._default_roi_stats()

    def _record_roi_stats(self, stats):
        self.last_roi_stats = stats
        self._roi_stats_buffer.append(stats)

    def get_last_roi_stats(self):
        if not self._roi_stats_buffer:
            return dict(self.last_roi_stats)
        numeric_keys = [
            "roi_keep_ratio",
            "roi_num_tokens",
            "roi_num_kept_tokens",
            "roi_num_roi_tokens",
            "roi_num_group_tokens",
            "roi_grouped_token_count",
            "roi_pruned_token_count",
            "roi_num_pruned_tokens",
            "effective_token_count",
            "roi_mask_time_sec",
            "roi_score_mean",
            "roi_score_std",
            "roi_score_min",
            "roi_score_max",
            "sparse_num_tokens",
            "sparse_num_foreground_tokens",
            "sparse_num_background_tokens",
        ]
        summary = dict(self._roi_stats_buffer[-1])
        for key in numeric_keys:
            values = [
                stats.get(key)
                for stats in self._roi_stats_buffer
                if isinstance(stats.get(key), (int, float))
            ]
            if values:
                summary[key] = sum(values) / len(values)
        for key, value in list(summary.items()):
            if key.startswith("roi_"):
                summary.setdefault("drs_" + key[len("roi_") :], value)
        return summary

    def _use_grouped_roi(self):
        return (
            self.roi_selector is not None
            and self.grouped_token_processor is not None
            and self.roi_config.roi_mask_type == "grouped"
        )

    def _use_pruned_roi(self):
        return (
            self.roi_selector is not None
            and self.pruned_token_processor is not None
            and self.roi_config.roi_mask_type == "prune"
        )

    def _use_sparse_primary_dynamics(self):
        return (
            self.sparse_dynamics_config.enabled
            and self.sparse_primary_dynamics is not None
        )

    def _sparse_uses_roi(self):
        return self.sparse_dynamics_config.mask_source in {
            "drs",
            "roi",
            "drs_union_dynamic",
            "drs_intersect_dynamic",
            "roi_union_dynamic",
            "roi_intersect_dynamic",
        }

    def _sparse_uses_dynamic_localizer(self):
        return self.sparse_dynamics_config.mask_source in {
            "dynamic_localizer",
            "drs_union_dynamic",
            "drs_intersect_dynamic",
            "roi_union_dynamic",
            "roi_intersect_dynamic",
        }

    def _ensure_roi_selector_device(self, device):
        if self.roi_selector is None:
            return
        param = next(self.roi_selector.parameters(), None)
        if param is not None and param.device != device:
            self.roi_selector.to(device)

    def _ensure_dynamic_localizer_device(self, device):
        if self.sparse_dynamic_localizer is None:
            return
        param = next(self.sparse_dynamic_localizer.parameters(), None)
        if param is not None and param.device != device:
            self.sparse_dynamic_localizer.to(device)

    def _sparse_roi_mask(self, z_obs, z_act):
        if self.roi_selector is None:
            raise ValueError(
                "sparse_dynamics mask_source uses DRS, but DRSSelector is not initialized. "
                "Set use_roi=true/drs_enabled=true and roi_mode/drs_mode to the desired foreground selector."
            )
        self._ensure_roi_selector_device(z_obs["visual"].device)
        _, roi_mask, roi_stats = self.roi_selector(
            visual_tokens=z_obs["visual"],
            action=z_act,
            proprio=z_obs["proprio"],
        )
        return roi_mask.bool(), roi_stats

    def _sparse_dynamic_mask(self, z_obs, z_act, num_frames):
        if self.sparse_dynamic_localizer is None:
            raise ValueError(
                "sparse_dynamics mask_source uses dynamic_localizer, but no localizer is loaded."
            )
        self._ensure_dynamic_localizer_device(z_obs["visual"].device)
        scores, mask = self.sparse_dynamic_localizer.predict_scores_and_mask(
            visual_history=z_obs["visual"],
            action_history=z_act,
            proprio_history=z_obs["proprio"],
            threshold=self.sparse_dynamics_config.dynamic_localizer_threshold,
            min_tokens=self.sparse_dynamics_config.dynamic_localizer_min_tokens,
        )
        if (
            self.sparse_dynamics_config.foreground_topk is not None
            or float(self.sparse_dynamics_config.foreground_keep_ratio) < 1.0
        ):
            mask = topk_token_mask(
                scores,
                keep_ratio=self.sparse_dynamics_config.foreground_keep_ratio,
                topk=self.sparse_dynamics_config.foreground_topk,
            )
        mask = mask.bool().unsqueeze(1).expand(-1, num_frames, -1)
        stats = {
            "roi_enabled": True,
            "roi_mode": "dynamic_localizer",
            "roi_keep_ratio": float(mask.float().mean().detach().cpu().item()),
            "roi_num_tokens": float(mask.shape[-1]),
            "roi_num_kept_tokens": float(mask.sum(dim=-1).float().mean().detach().cpu().item()),
            "roi_num_roi_tokens": float(mask.sum(dim=-1).float().mean().detach().cpu().item()),
            "roi_mask_time_sec": None,
            "roi_score_mean": float(scores.detach().float().mean().cpu().item()),
            "roi_score_std": float(scores.detach().float().std().cpu().item()),
            "roi_score_min": float(scores.detach().float().min().cpu().item()),
            "roi_score_max": float(scores.detach().float().max().cpu().item()),
        }
        for key, value in list(stats.items()):
            if key.startswith("roi_"):
                stats.setdefault("drs_" + key[len("roi_") :], value)
        return mask, stats

    def _select_sparse_foreground_mask(self, z):
        z_obs, z_act = self.separate_emb(z)
        source = self.sparse_dynamics_config.mask_source
        roi_mask = None
        dyn_mask = None
        stats = {}
        if self._sparse_uses_roi():
            roi_mask, stats = self._sparse_roi_mask(z_obs, z_act)
        if self._sparse_uses_dynamic_localizer():
            dyn_mask, dyn_stats = self._sparse_dynamic_mask(z_obs, z_act, z.shape[1])
            if not stats:
                stats = dyn_stats
            else:
                stats.update(
                    {
                        "sparse_dynamic_score_mean": dyn_stats.get("roi_score_mean"),
                        "sparse_dynamic_score_std": dyn_stats.get("roi_score_std"),
                    }
                )

        if source in {"drs", "roi"}:
            mask = roi_mask
        elif source == "dynamic_localizer":
            mask = dyn_mask
        elif source in {"drs_union_dynamic", "roi_union_dynamic"}:
            mask = roi_mask | dyn_mask
        elif source in {"drs_intersect_dynamic", "roi_intersect_dynamic"}:
            mask = roi_mask & dyn_mask
        else:
            raise ValueError(f"Unsupported sparse_dynamics.mask_source: {source}")
        stats.update(
            {
                "sparse_dynamics_enabled": True,
                "sparse_dynamics_mode": self.sparse_dynamics_config.mode,
                "sparse_mask_source": source,
            }
        )
        return mask.bool(), stats

    def predict_with_sparse_primary_dynamics(self, z):
        with self._time_section("sparse_foreground_select"):
            foreground_mask, sparse_stats = self._select_sparse_foreground_mask(z)
        self.last_prediction_token_mask = (
            foreground_mask
            if self.sparse_dynamics_config.loss_on_foreground_only
            else None
        )
        with self._time_section("sparse_primary_total"):
            sparse_pred, processor_stats = self.sparse_primary_dynamics(
                z,
                foreground_mask,
                self.predictor,
            )
        timing_getter = getattr(self.sparse_primary_dynamics, "get_last_timing", None)
        if timing_getter is not None:
            self._record_external_timing("sparse_primary", timing_getter())
        sparse_stats.update(processor_stats)
        self._record_roi_stats(sparse_stats)
        return sparse_pred


    def predict_with_grouped_roi(self, z):
        if self.concat_dim != 1:
            raise ValueError(
                "roi_mask_type='grouped' currently supports concat_dim=1, where action/proprio are concatenated to each visual patch token."
            )
        self._ensure_roi_selector_device(z.device)
        z_obs, z_act = self.separate_emb(z)
        with self._time_section("grouped_roi_select"):
            _, roi_mask, roi_stats = self.roi_selector(
                visual_tokens=z_obs["visual"],
                action=z_act,
                proprio=z_obs["proprio"],
            )
        with self._time_section("grouped_processor_total"):
            grouped_pred, grouped_stats = self.grouped_token_processor(
                z,
                roi_mask,
                self.predictor,
            )
        timing_getter = getattr(self.grouped_token_processor, "get_last_timing", None)
        if timing_getter is not None:
            self._record_external_timing("grouped", timing_getter())
        roi_stats.update(grouped_stats)
        self._record_roi_stats(roi_stats)
        return grouped_pred

    def predict_with_pruned_roi(self, z):
        if self.concat_dim != 1:
            raise ValueError(
                "roi_mask_type='prune' currently supports concat_dim=1, where action/proprio are concatenated to each visual patch token."
            )
        self._ensure_roi_selector_device(z.device)
        z_obs, z_act = self.separate_emb(z)
        with self._time_section("prune_roi_select"):
            _, roi_mask, roi_stats = self.roi_selector(
                visual_tokens=z_obs["visual"],
                action=z_act,
                proprio=z_obs["proprio"],
            )
        self.last_prediction_token_mask = roi_mask
        with self._time_section("prune_processor_total"):
            pruned_pred, pruned_stats = self.pruned_token_processor(
                z,
                roi_mask,
                self.predictor,
            )
        timing_getter = getattr(self.pruned_token_processor, "get_last_timing", None)
        if timing_getter is not None:
            self._record_external_timing("prune", timing_getter())
        roi_stats.update(pruned_stats)
        self._record_roi_stats(roi_stats)
        return pruned_pred

    def apply_roi_to_z(self, z):
        if self.roi_selector is None:
            return z
        self._ensure_roi_selector_device(z.device)
        z_obs, z_act = self.separate_emb(z)
        with self._time_section("roi_mask_select"):
            visual_tokens, roi_mask, roi_stats = self.roi_selector(
                visual_tokens=z_obs["visual"],
                action=z_act,
                proprio=z_obs["proprio"],
            )
        self._record_roi_stats(roi_stats)
        return self.compose_z(visual_tokens, z_obs["proprio"], z_act)

    @staticmethod
    def _masked_emb_criterion(pred, target, token_mask):
        mask = token_mask.to(device=pred.device, dtype=pred.dtype).unsqueeze(-1)
        denom = (mask.sum() * pred.shape[-1]).clamp_min(1.0)
        return (((pred - target.detach()) ** 2) * mask).sum() / denom

    def forward(self, obs, act):
        """
        input:  obs (dict):  "visual", "proprio" (b, num_frames, 3, img_size, img_size)
                act: (b, num_frames, action_dim)
        output: z_pred: (b, num_hist, num_patches, emb_dim)
                visual_pred: (b, num_hist, 3, img_size, img_size)
                visual_reconstructed: (b, num_frames, 3, img_size, img_size)
        """
        with self._time_section("forward_total"):
            self.reset_roi_stats()
            loss = 0
            loss_components = {}
            z = self.encode(obs, act)
            z_src = z[:, : self.num_hist, :, :]  # (b, num_hist, num_patches, dim)
            z_tgt = z[:, self.num_pred :, :, :]  # (b, num_hist, num_patches, dim)
            visual_src = obs['visual'][:, : self.num_hist, ...]  # (b, num_hist, 3, img_size, img_size)
            visual_tgt = obs['visual'][:, self.num_pred :, ...]  # (b, num_hist, 3, img_size, img_size)

            if self.predictor is not None:
                z_pred = self.predict(z_src)
                if self.decoder is not None:
                    obs_pred, diff_pred = self.decode(
                        z_pred.detach()
                    )  # recon loss should only affect decoder
                    visual_pred = obs_pred['visual']
                    recon_loss_pred = self.decoder_criterion(visual_pred, visual_tgt)
                    decoder_loss_pred = (
                        recon_loss_pred + self.decoder_latent_loss_weight * diff_pred
                    )
                    loss_components["decoder_recon_loss_pred"] = recon_loss_pred
                    loss_components["decoder_vq_loss_pred"] = diff_pred
                    loss_components["decoder_loss_pred"] = decoder_loss_pred
                else:
                    visual_pred = None

                # Compute loss for visual, proprio dims (i.e. exclude action dims)
                if self.concat_dim == 0:
                    z_visual_loss = self.emb_criterion(z_pred[:, :, :-2, :], z_tgt[:, :, :-2, :].detach())
                    z_proprio_loss = self.emb_criterion(z_pred[:, :, -2, :], z_tgt[:, :, -2, :].detach())
                    z_loss = self.emb_criterion(z_pred[:, :, :-1, :], z_tgt[:, :, :-1, :].detach())
                elif self.concat_dim == 1:
                    token_mask = (
                        self.last_prediction_token_mask
                        if (
                            self._use_pruned_roi()
                            or (
                                self._use_sparse_primary_dynamics()
                                and self.sparse_dynamics_config.loss_on_foreground_only
                            )
                        )
                        else None
                    )
                    if token_mask is not None:
                        z_visual_loss = self._masked_emb_criterion(
                            z_pred[:, :, :, :-(self.proprio_dim + self.action_dim)],
                            z_tgt[:, :, :, :-(self.proprio_dim + self.action_dim)],
                            token_mask,
                        )
                        z_proprio_loss = self._masked_emb_criterion(
                            z_pred[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim],
                            z_tgt[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim],
                            token_mask,
                        )
                        z_loss = self._masked_emb_criterion(
                            z_pred[:, :, :, :-self.action_dim],
                            z_tgt[:, :, :, :-self.action_dim],
                            token_mask,
                        )
                    else:
                        z_visual_loss = self.emb_criterion(
                            z_pred[:, :, :, :-(self.proprio_dim + self.action_dim)], \
                            z_tgt[:, :, :, :-(self.proprio_dim + self.action_dim)].detach()
                        )
                        z_proprio_loss = self.emb_criterion(
                            z_pred[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim],
                            z_tgt[:, :, :, -(self.proprio_dim + self.action_dim): -self.action_dim].detach()
                        )
                        z_loss = self.emb_criterion(
                            z_pred[:, :, :, :-self.action_dim],
                            z_tgt[:, :, :, :-self.action_dim].detach()
                        )

                loss = loss + z_loss
                loss_components["z_loss"] = z_loss
                loss_components["z_visual_loss"] = z_visual_loss
                loss_components["z_proprio_loss"] = z_proprio_loss
            else:
                visual_pred = None
                z_pred = None

            if self.decoder is not None:
                obs_reconstructed, diff_reconstructed = self.decode(
                    z.detach()
                )  # recon loss should only affect decoder
                visual_reconstructed = obs_reconstructed["visual"]
                recon_loss_reconstructed = self.decoder_criterion(visual_reconstructed, obs['visual'])
                decoder_loss_reconstructed = (
                    recon_loss_reconstructed
                    + self.decoder_latent_loss_weight * diff_reconstructed
                )

                loss_components["decoder_recon_loss_reconstructed"] = (
                    recon_loss_reconstructed
                )
                loss_components["decoder_vq_loss_reconstructed"] = diff_reconstructed
                loss_components["decoder_loss_reconstructed"] = (
                    decoder_loss_reconstructed
                )
                loss = loss + decoder_loss_reconstructed
            else:
                visual_reconstructed = None
            loss_components["loss"] = loss
            return z_pred, visual_pred, visual_reconstructed, loss, loss_components

    def replace_actions_from_z(self, z, act):
        with self._time_section("replace_actions"):
            act_emb = self.encode_act(act)
            if self.concat_dim == 0:
                z[:, :, -1, :] = act_emb
            elif self.concat_dim == 1:
                act_tiled = repeat(act_emb.unsqueeze(2), "b t 1 a -> b t f a", f=z.shape[2])
                act_repeated = act_tiled.repeat(1, 1, 1, self.num_action_repeat)
                z[..., -self.action_dim:] = act_repeated
            return z


    def rollout(self, obs_0, act):
        """
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, action_dim)
        output: embeddings of rollout obs
                visuals: (b, t+n+1, 3, img_size, img_size)
                z: (b, t+n+1, num_patches, emb_dim)
        """
        self.reset_timing_stats()
        with self._time_section("rollout_total"):
            self.reset_roi_stats()
            num_obs_init = obs_0['visual'].shape[1]
            act_0 = act[:, :num_obs_init]
            action = act[:, num_obs_init:]
            z = self.encode(obs_0, act_0)
            t = 0
            inc = 1
            while t < action.shape[1]:
                z_pred = self.predict(z[:, -self.num_hist :])
                z_new = z_pred[:, -inc:, ...]
                z_new = self.replace_actions_from_z(z_new, action[:, t : t + inc, :])
                with self._time_section("rollout_concat"):
                    z = torch.cat([z, z_new], dim=1)
                t += inc

            z_pred = self.predict(z[:, -self.num_hist :])
            z_new = z_pred[:, -1 :, ...] # take only the next pred
            with self._time_section("rollout_concat"):
                z = torch.cat([z, z_new], dim=1)
            z_obses, z_acts = self.separate_emb(z)
            return z_obses, z
