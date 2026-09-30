import os
os.environ.setdefault("WANDB_MODE", "disabled")

import time
import hydra
import torch
import wandb
import logging
import warnings
import threading
import itertools
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict
from einops import rearrange
from accelerate import Accelerator
from torchvision import utils
import torch.distributed as dist
from pathlib import Path
from collections import OrderedDict
from hydra.types import RunMode
from hydra.core.hydra_config import HydraConfig
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from metrics.image_metrics import eval_images
from models.roi import extract_drs_config_dict
from models.sparse_dynamics import extract_sparse_dynamics_config_dict
from utils import slice_trajdict_with_t, cfg_to_dict, seed, sample_tensors
from eval_logging import (
    FUTURE_LIGHTWEIGHT_FIELDS,
    SafeMetricsLogger,
    build_run_meta,
    count_parameters,
    cuda_peak_memory_mb,
    elapsed_since,
    perf_counter,
    reset_cuda_peak_memory,
)

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        with open_dict(cfg):
            cfg["saved_folder"] = os.getcwd()
            log.info(f"Model saved dir: {cfg['saved_folder']}")
        cfg_dict = cfg_to_dict(cfg)
        model_name = cfg_dict["saved_folder"].split("outputs/")[-1]
        model_name += f"_{self.cfg.env.name}_f{self.cfg.frameskip}_h{self.cfg.num_hist}_p{self.cfg.num_pred}"

        if HydraConfig.get().mode == RunMode.MULTIRUN:
            log.info(" Multirun setup begin...")
            log.info(f"SLURM_JOB_NODELIST={os.environ['SLURM_JOB_NODELIST']}")
            log.info(f"DEBUGVAR={os.environ['DEBUGVAR']}")
            # ==== init ddp process group ====
            os.environ["RANK"] = os.environ["SLURM_PROCID"]
            os.environ["WORLD_SIZE"] = os.environ["SLURM_NTASKS"]
            os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
            try:
                dist.init_process_group(
                    backend="nccl",
                    init_method="env://",
                    timeout=timedelta(minutes=5),  # Set a 5-minute timeout
                )
                log.info("Multirun setup completed.")
            except Exception as e:
                log.error(f"DDP setup failed: {e}")
                raise
            torch.distributed.barrier()
            # # ==== /init ddp process group ====

        self.accelerator = Accelerator(log_with="wandb")
        log.info(
            f"rank: {self.accelerator.local_process_index}  model_name: {model_name}"
        )
        self.device = self.accelerator.device
        log.info(f"device: {self.device}   model_name: {model_name}")
        self.base_path = os.path.dirname(os.path.abspath(__file__))

        self.num_reconstruct_samples = self.cfg.training.num_reconstruct_samples
        self.total_epochs = self.cfg.training.epochs
        self.epoch = 0
        self.resume_batch_idx = 0
        self._resume_from_batch_ckpt = False
        self._pending_optimizer_states = {}

        assert cfg.training.batch_size % self.accelerator.num_processes == 0, (
            "Batch size must be divisible by the number of processes. "
            f"Batch_size: {cfg.training.batch_size} num_processes: {self.accelerator.num_processes}."
        )

        OmegaConf.set_struct(cfg, False)
        cfg.effective_batch_size = cfg.training.batch_size
        cfg.gpu_batch_size = cfg.training.batch_size // self.accelerator.num_processes
        OmegaConf.set_struct(cfg, True)
        if self.cfg.get("data_path", None) is not None:
            with open_dict(cfg):
                cfg.env.dataset.data_path = self.cfg.data_path

        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            wandb_run_id = None
            if os.path.exists("hydra.yaml"):
                existing_cfg = OmegaConf.load("hydra.yaml")
                wandb_run_id = existing_cfg["wandb_run_id"]
                log.info(f"Resuming Wandb run {wandb_run_id}")

            wandb_dict = OmegaConf.to_container(cfg, resolve=True)
            if self.cfg.debug:
                log.info("WARNING: Running in debug mode...")
                self.wandb_run = wandb.init(
                    project="dino_wm_debug",
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                )
            else:
                self.wandb_run = wandb.init(
                    project="dino_wm",
                    config=wandb_dict,
                    id=wandb_run_id,
                    resume="allow",
                )
            OmegaConf.set_struct(cfg, False)
            cfg.wandb_run_id = self.wandb_run.id
            OmegaConf.set_struct(cfg, True)
            wandb.run.name = "{}".format(model_name)
            with open(os.path.join(os.getcwd(), "hydra.yaml"), "w") as f:
                f.write(OmegaConf.to_yaml(cfg, resolve=True))

        seed(cfg.training.seed)
        log.info(f"Loading dataset from {self.cfg.env.dataset.data_path} ...")
        self.datasets, traj_dsets = hydra.utils.call(
            self.cfg.env.dataset,
            num_hist=self.cfg.num_hist,
            num_pred=self.cfg.num_pred,
            frameskip=self.cfg.frameskip,
        )

        self.train_traj_dset = traj_dsets["train"]
        self.val_traj_dset = traj_dsets["valid"]

        self.dataloaders = {
            x: torch.utils.data.DataLoader(
                self.datasets[x],
                batch_size=self.cfg.gpu_batch_size,
                shuffle=False, # already shuffled in TrajSlicerDataset
                num_workers=self.cfg.env.num_workers,
                collate_fn=None,
            )
            for x in ["train", "valid"]
        }

        log.info(f"dataloader batch size: {self.cfg.gpu_batch_size}")

        self.dataloaders["train"], self.dataloaders["valid"] = self.accelerator.prepare(
            self.dataloaders["train"], self.dataloaders["valid"]
        )

        self.encoder = None
        self.action_encoder = None
        self.proprio_encoder = None
        self.predictor = None
        self.decoder = None
        self.train_encoder = self.cfg.model.train_encoder
        self.train_predictor = self.cfg.model.train_predictor
        self.train_decoder = self.cfg.model.train_decoder
        log.info(f"Train encoder, predictor, decoder:\
            {self.cfg.model.train_encoder}\
            {self.cfg.model.train_predictor}\
            {self.cfg.model.train_decoder}")

        self._keys_to_save = [
            "epoch",
        ]
        self._keys_to_save += (
            ["encoder", "encoder_optimizer"] if self.train_encoder else []
        )
        self._keys_to_save += (
            ["predictor", "predictor_optimizer", "action_encoder_optimizer"]
            if self.train_predictor and self.cfg.has_predictor
            else []
        )
        self._keys_to_save += (
            ["sparse_primary_dynamics"]
            if self.cfg.get("sparse_dynamics", {}).get("enabled", False)
            else []
        )
        self._keys_to_save += (
            ["decoder", "decoder_optimizer"] if self.train_decoder else []
        )
        self._keys_to_save += ["action_encoder", "proprio_encoder"]

        self.init_models()
        self.init_optimizers()

        self.metrics_logger = (
            SafeMetricsLogger(os.getcwd())
            if self.accelerator.is_main_process
            else None
        )
        self.train_metric_history = []
        self.run_start_time = perf_counter()
        self.param_counts = self._collect_param_counts()
        try:
            self._write_run_meta()
        except Exception as exc:
            log.warning("Failed to write training run metadata: %s", exc)
        self.epoch_log = OrderedDict()

    @staticmethod
    def _is_optimizer_key(key):
        return key.endswith("_optimizer")

    def _serialize_ckpt_value(self, key):
        value = self.__dict__[key]
        if self._is_optimizer_key(key) and hasattr(value, "state_dict"):
            return value.state_dict()
        if hasattr(value, "module"):
            return self.accelerator.unwrap_model(value)
        return value

    def _build_ckpt(self, checkpoint_kind="epoch", batch_idx=None):
        ckpt = {}
        for k in self._keys_to_save:
            if k not in self.__dict__ or self.__dict__[k] is None:
                continue
            ckpt[k] = self._serialize_ckpt_value(k)
        ckpt["checkpoint_kind"] = checkpoint_kind
        if batch_idx is not None:
            ckpt["batch_idx"] = int(batch_idx)
            ckpt["next_batch_idx"] = int(batch_idx) + 1
        return ckpt

    def _atomic_torch_save(self, obj, path):
        path = Path(path)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        torch.save(obj, tmp_path)
        os.replace(tmp_path, path)

    def save_ckpt(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            ckpt_dir = Path("checkpoints")
            ckpt_dir.mkdir(exist_ok=True)
            ckpt = self._build_ckpt(checkpoint_kind="epoch")
            self._atomic_torch_save(ckpt, ckpt_dir / "model_latest.pth")
            self._atomic_torch_save(ckpt, ckpt_dir / f"model_{self.epoch}.pth")
            log.info("Saved model to {}".format(os.getcwd()))
            ckpt_path = os.path.join(os.getcwd(), f"checkpoints/model_{self.epoch}.pth")
        else:
            ckpt_path = None
        model_name = self.cfg["saved_folder"].split("outputs/")[-1]
        model_epoch = self.epoch
        return ckpt_path, model_name, model_epoch

    def save_batch_ckpt(self, batch_idx):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            ckpt_dir = Path("checkpoints")
            ckpt_dir.mkdir(exist_ok=True)
            for old_path in ckpt_dir.glob("model_batch_epoch_*_idx_*.pth"):
                old_path.unlink(missing_ok=True)
            ckpt = self._build_ckpt(
                checkpoint_kind="batch",
                batch_idx=batch_idx,
            )
            latest_path = ckpt_dir / "model_batch_latest.pth"
            batch_path = ckpt_dir / f"model_batch_epoch_{self.epoch}_idx_{batch_idx}.pth"
            self._atomic_torch_save(ckpt, latest_path)
            self._atomic_torch_save(ckpt, batch_path)
            log.info(
                "Saved batch checkpoint to %s (epoch=%s, next_batch=%s)",
                latest_path,
                self.epoch,
                int(batch_idx) + 1,
            )

    def cleanup_batch_ckpts(self):
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            ckpt_dir = Path("checkpoints")
            if not ckpt_dir.exists():
                return
            for old_path in ckpt_dir.glob("model_batch_epoch_*_idx_*.pth"):
                old_path.unlink(missing_ok=True)
            (ckpt_dir / "model_batch_latest.pth").unlink(missing_ok=True)

    def _extract_optimizer_state(self, value):
        if hasattr(value, "state_dict"):
            return value.state_dict()
        return value

    def _restore_optimizer_state(self, name, value):
        optimizer = getattr(self, name, None)
        if optimizer is None:
            self._pending_optimizer_states[name] = self._extract_optimizer_state(value)
            return
        optimizer.load_state_dict(self._extract_optimizer_state(value))

    def _restore_pending_optimizer_states(self):
        if not self._pending_optimizer_states:
            return
        for name, value in list(self._pending_optimizer_states.items()):
            optimizer = getattr(self, name, None)
            if optimizer is None:
                continue
            optimizer.load_state_dict(value)
            del self._pending_optimizer_states[name]
            log.info("Restored optimizer state: %s", name)

    @staticmethod
    def _embedding_dim(module, fallback=None):
        if hasattr(module, "emb_dim"):
            return module.emb_dim
        patch_embed = getattr(module, "patch_embed", None)
        if patch_embed is not None and hasattr(patch_embed, "out_channels"):
            emb_dim = patch_embed.out_channels
            setattr(module, "emb_dim", emb_dim)
            return emb_dim
        if fallback is not None:
            return fallback
        raise AttributeError(
            f"Cannot infer embedding dimension for {type(module).__name__}; "
            "expected .emb_dim or .patch_embed.out_channels."
        )

    def load_ckpt(
        self,
        filename="model_latest.pth",
        load_epoch=True,
        load_decoder=True,
        load_optimizers=True,
    ):
        ckpt = torch.load(filename, map_location="cpu")
        skip_keys = set()
        if not load_epoch:
            skip_keys.add("epoch")
        if not load_decoder:
            skip_keys.update({"decoder", "decoder_optimizer"})
        if not load_optimizers:
            skip_keys.update(
                {
                    "encoder_optimizer",
                    "predictor_optimizer",
                    "decoder_optimizer",
                    "action_encoder_optimizer",
                }
            )
        for k, v in ckpt.items():
            if k in skip_keys:
                continue
            if self._is_optimizer_key(k):
                self._restore_optimizer_state(k, v)
                continue
            self.__dict__[k] = v
        metadata_keys = {"checkpoint_kind", "batch_idx", "next_batch_idx"}
        not_in_ckpt = set(self._keys_to_save) - (set(ckpt.keys()) - metadata_keys)
        if len(not_in_ckpt):
            log.warning("Keys not found in ckpt: %s", not_in_ckpt)
        return ckpt

    def init_models(self):
        resume_checkpoint = self.cfg.get("resume_checkpoint", None)
        batch_ckpt = Path(self.cfg.saved_folder) / "checkpoints" / "model_batch_latest.pth"
        model_ckpt = Path(self.cfg.saved_folder) / "checkpoints" / "model_latest.pth"
        resume_ckpt = None
        if resume_checkpoint:
            resume_ckpt = Path(resume_checkpoint)
            if not resume_ckpt.exists():
                raise FileNotFoundError(f"resume_checkpoint not found: {resume_ckpt}")
        elif batch_ckpt.exists():
            resume_ckpt = batch_ckpt
        elif model_ckpt.exists():
            resume_ckpt = model_ckpt

        if resume_ckpt is not None:
            ckpt = self.load_ckpt(resume_ckpt)
            checkpoint_kind = ckpt.get("checkpoint_kind", "epoch")
            if "batch" in resume_ckpt.name and checkpoint_kind == "epoch":
                checkpoint_kind = "batch"
            if checkpoint_kind == "batch":
                self._resume_from_batch_ckpt = True
                self.resume_batch_idx = int(
                    ckpt.get("next_batch_idx", int(ckpt.get("batch_idx", -1)) + 1)
                )
                log.info(
                    "Resuming from batch checkpoint %s: epoch=%s next_batch=%s",
                    resume_ckpt,
                    self.epoch,
                    self.resume_batch_idx,
                )
            else:
                self.resume_batch_idx = 0
                self._resume_from_batch_ckpt = False
                log.info(f"Resuming from epoch {self.epoch}: {resume_ckpt}")
        elif self.cfg.get("init_checkpoint", None):
            init_checkpoint = Path(self.cfg.init_checkpoint)
            self.load_ckpt(
                init_checkpoint,
                load_epoch=False,
                load_decoder=bool(self.cfg.get("init_checkpoint_load_decoder", False)),
                load_optimizers=False,
            )
            log.info(f"Initialized model components from checkpoint: {init_checkpoint}")

        # initialize encoder
        if self.encoder is None:
            self.encoder = hydra.utils.instantiate(
                self.cfg.encoder,
            )
        if not self.train_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False

        if self.proprio_encoder is None:
            self.proprio_encoder = hydra.utils.instantiate(
                self.cfg.proprio_encoder,
                in_chans=self.datasets["train"].proprio_dim,
                emb_dim=self.cfg.proprio_emb_dim,
            )
        proprio_emb_dim = self._embedding_dim(
            self.proprio_encoder, fallback=self.cfg.proprio_emb_dim
        )
        print(f"Proprio encoder type: {type(self.proprio_encoder)}")
        self.proprio_encoder = self.accelerator.prepare(self.proprio_encoder)

        if self.action_encoder is None:
            self.action_encoder = hydra.utils.instantiate(
                self.cfg.action_encoder,
                in_chans=self.datasets["train"].action_dim,
                emb_dim=self.cfg.action_emb_dim,
            )
        action_emb_dim = self._embedding_dim(
            self.action_encoder, fallback=self.cfg.action_emb_dim
        )
        print(f"Action encoder type: {type(self.action_encoder)}")

        self.action_encoder = self.accelerator.prepare(self.action_encoder)

        if self.accelerator.is_main_process:
            self.wandb_run.watch(self.action_encoder)
            self.wandb_run.watch(self.proprio_encoder)

        # initialize predictor
        if self.encoder.latent_ndim == 1:  # if feature is 1D
            num_patches = 1
        else:
            decoder_scale = 16  # from vqvae
            num_side_patches = self.cfg.img_size // decoder_scale
            num_patches = num_side_patches**2

        if self.cfg.concat_dim == 0:
            num_patches += 2

        if self.cfg.has_predictor:
            if self.predictor is None:
                self.predictor = hydra.utils.instantiate(
                    self.cfg.predictor,
                    num_patches=num_patches,
                    num_frames=self.cfg.num_hist,
                    dim=self.encoder.emb_dim
                    + (
                        proprio_emb_dim * self.cfg.num_proprio_repeat
                        + action_emb_dim * self.cfg.num_action_repeat
                    )
                    * (self.cfg.concat_dim),
                )
            if not self.train_predictor:
                for param in self.predictor.parameters():
                    param.requires_grad = False

        # initialize decoder
        if self.cfg.has_decoder:
            if self.decoder is None:
                if self.cfg.env.decoder_path is not None:
                    decoder_path = os.path.join(
                        self.base_path, self.cfg.env.decoder_path
                    )
                    ckpt = torch.load(decoder_path)
                    if isinstance(ckpt, dict):
                        self.decoder = ckpt["decoder"]
                    else:
                        self.decoder = torch.load(decoder_path)
                    log.info(f"Loaded decoder from {decoder_path}")
                else:
                    self.decoder = hydra.utils.instantiate(
                        self.cfg.decoder,
                        emb_dim=self.encoder.emb_dim,  # 384
                    )
            if not self.train_decoder:
                for param in self.decoder.parameters():
                    param.requires_grad = False
        self.encoder, self.predictor, self.decoder = self.accelerator.prepare(
            self.encoder, self.predictor, self.decoder
        )
        self.model = hydra.utils.instantiate(
            self.cfg.model,
            encoder=self.encoder,
            proprio_encoder=self.proprio_encoder,
            action_encoder=self.action_encoder,
            predictor=self.predictor,
            decoder=self.decoder,
            proprio_dim=proprio_emb_dim,
            action_dim=action_emb_dim,
            concat_dim=self.cfg.concat_dim,
            num_action_repeat=self.cfg.num_action_repeat,
            num_proprio_repeat=self.cfg.num_proprio_repeat,
            use_roi=self.cfg.get(
                "use_drs",
                self.cfg.get("drs_enabled", self.cfg.get("use_roi", False)),
            ),
            roi_mode=self.cfg.get("drs_mode", self.cfg.get("roi_mode", "none")),
            roi_config=extract_drs_config_dict(self.cfg),
            roi_head_checkpoint=self.cfg.get(
                "drs_head_checkpoint", self.cfg.get("roi_head_checkpoint", None)
            ),
            sparse_dynamics_config=extract_sparse_dynamics_config_dict(self.cfg),
        )
        if getattr(self.model, "roi_selector", None) is not None:
            self.model.roi_selector.to(self.device)
        self.sparse_primary_dynamics = getattr(
            self,
            "sparse_primary_dynamics",
            None,
        )
        if self.sparse_primary_dynamics is not None and getattr(
            self.model,
            "sparse_primary_dynamics",
            None,
        ) is not None:
            self.model.sparse_primary_dynamics = self.sparse_primary_dynamics
        self.sparse_primary_dynamics = getattr(
            self.model,
            "sparse_primary_dynamics",
            None,
        )
        if self.sparse_primary_dynamics is not None:
            self.sparse_primary_dynamics = self.accelerator.prepare(
                self.sparse_primary_dynamics.to(self.device)
            )
            self.model.sparse_primary_dynamics = self.sparse_primary_dynamics
        if getattr(self.model, "sparse_dynamic_localizer", None) is not None:
            self.model.sparse_dynamic_localizer.to(self.device)

    def init_optimizers(self):
        self.encoder_optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=self.cfg.training.encoder_lr,
        )
        self.encoder_optimizer = self.accelerator.prepare(self.encoder_optimizer)
        if self.cfg.has_predictor:
            predictor_params = list(self.predictor.parameters())
            sparse_params = []
            if getattr(self, "sparse_primary_dynamics", None) is not None:
                sparse_params = [
                    p
                    for p in self.sparse_primary_dynamics.parameters()
                    if p.requires_grad
                ]
            self.predictor_optimizer = torch.optim.AdamW(
                itertools.chain(predictor_params, sparse_params),
                lr=self.cfg.training.predictor_lr,
            )
            self.predictor_optimizer = self.accelerator.prepare(
                self.predictor_optimizer
            )

            self.action_encoder_optimizer = torch.optim.AdamW(
                itertools.chain(
                    self.action_encoder.parameters(), self.proprio_encoder.parameters()
                ),
                lr=self.cfg.training.action_encoder_lr,
            )
            self.action_encoder_optimizer = self.accelerator.prepare(
                self.action_encoder_optimizer
            )

        if self.cfg.has_decoder:
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(), lr=self.cfg.training.decoder_lr
            )
            self.decoder_optimizer = self.accelerator.prepare(self.decoder_optimizer)
        self._restore_pending_optimizer_states()

    def _unwrap_for_metrics(self, module):
        if module is None:
            return None
        try:
            return self.accelerator.unwrap_model(module)
        except Exception:
            return getattr(module, "module", module)

    def _collect_param_counts(self):
        components = {
            "encoder": self._unwrap_for_metrics(self.encoder),
            "predictor": self._unwrap_for_metrics(self.predictor),
            "decoder": self._unwrap_for_metrics(self.decoder),
            "proprio_encoder": self._unwrap_for_metrics(self.proprio_encoder),
            "action_encoder": self._unwrap_for_metrics(self.action_encoder),
            "drs_selector": self._unwrap_for_metrics(
                getattr(self.model, "drs_selector", None)
            ),
            "roi_selector": self._unwrap_for_metrics(
                getattr(self.model, "roi_selector", None)
            ),
            "sparse_dynamic_localizer": self._unwrap_for_metrics(
                getattr(self.model, "sparse_dynamic_localizer", None)
            ),
            "sparse_primary_dynamics": self._unwrap_for_metrics(
                getattr(self.model, "sparse_primary_dynamics", None)
            ),
        }
        component_counts = {
            name: count_parameters(module) for name, module in components.items()
        }
        model_counts = count_parameters(self._unwrap_for_metrics(self.model))
        return {
            "total_params": model_counts["total_params"],
            "trainable_params": model_counts["trainable_params"],
            "model": model_counts,
            "components": component_counts,
        }

    def _optimizer_lrs(self):
        lrs = {}
        optimizer_names = [
            "encoder_optimizer",
            "predictor_optimizer",
            "action_encoder_optimizer",
            "decoder_optimizer",
        ]
        for name in optimizer_names:
            optimizer = getattr(self, name, None)
            if optimizer is None:
                continue
            try:
                base_optimizer = getattr(optimizer, "optimizer", optimizer)
                lr_values = [
                    group.get("lr")
                    for group in base_optimizer.param_groups
                    if "lr" in group
                ]
                if lr_values:
                    lrs[f"{name}_lr"] = lr_values[0]
            except Exception:
                continue
        return lrs

    def _write_run_meta(self):
        if self.metrics_logger is None:
            return
        try:
            cfg_payload = OmegaConf.to_container(self.cfg, resolve=True)
        except Exception:
            cfg_payload = cfg_to_dict(self.cfg)
        meta = build_run_meta(
            kind="train",
            output_dir=os.getcwd(),
            cfg=cfg_payload,
            extra={
                "device": str(self.device),
                "accelerator_num_processes": self.accelerator.num_processes,
                "effective_batch_size": self.cfg.effective_batch_size,
                "gpu_batch_size": self.cfg.gpu_batch_size,
                "train_dataset_size": len(self.datasets["train"]),
                "valid_dataset_size": len(self.datasets["valid"]),
                "train_traj_count": len(self.train_traj_dset),
                "valid_traj_count": len(self.val_traj_dset),
                "params": self.param_counts,
            },
        )
        self.metrics_logger.write_json("run_meta.json", meta)

    def _write_train_metrics(
        self,
        epoch_log,
        epoch_time,
        train_time,
        validation_time,
    ):
        if self.metrics_logger is None:
            return
        train_iters = len(self.dataloaders["train"])
        valid_iters = len(self.dataloaders["valid"])
        record = OrderedDict(epoch_log)
        record.update(
            {
                "kind": "train",
                "epoch": self.epoch,
                "epoch_time_sec": epoch_time,
                "train_time_sec": train_time,
                "validation_time_sec": validation_time,
                "train_iterations_per_sec": (
                    train_iters / train_time if train_time > 0 else None
                ),
                "validation_iterations_per_sec": (
                    valid_iters / validation_time if validation_time > 0 else None
                ),
                "train_samples_per_sec": (
                    len(self.datasets["train"]) / train_time
                    if train_time > 0
                    else None
                ),
                "validation_samples_per_sec": (
                    len(self.datasets["valid"]) / validation_time
                    if validation_time > 0
                    else None
                ),
                "cuda_peak_memory_mb": cuda_peak_memory_mb(self.device),
                "total_params": self.param_counts["total_params"],
                "trainable_params": self.param_counts["trainable_params"],
                "model_component_params": self.param_counts["components"],
            }
        )
        record.update(self._optimizer_lrs())
        record.update(FUTURE_LIGHTWEIGHT_FIELDS)
        self.metrics_logger.append_jsonl("train_metrics.jsonl", record)
        self.train_metric_history.append(dict(record))

        summary = {
            "kind": "train",
            "output_dir": os.path.abspath(os.getcwd()),
            "epochs_completed": len(self.train_metric_history),
            "last_epoch": self.epoch,
            "last_metrics": record,
            "best_val_loss": self._best_metric("val_loss"),
            "best_train_loss": self._best_metric("train_loss"),
            "total_runtime_sec": elapsed_since(self.run_start_time),
            "params": self.param_counts,
        }
        summary.update(FUTURE_LIGHTWEIGHT_FIELDS)
        self.metrics_logger.write_json("summary.json", summary)

    def _best_metric(self, key):
        values = [
            metrics.get(key)
            for metrics in self.train_metric_history
            if metrics.get(key) is not None
        ]
        return min(values) if values else None

    def monitor_jobs(self, lock):
        """
        check planning eval jobs' status and update logs
        """
        while True:
            with lock:
                finished_jobs = [
                    job_tuple for job_tuple in self.job_set if job_tuple[2].done()
                ]
                for epoch, job_name, job in finished_jobs:
                    result = job.result()
                    print(f"Logging result for {job_name} at epoch {epoch}: {result}")
                    log_data = {
                        f"{job_name}/{key}": value for key, value in result.items()
                    }
                    log_data["epoch"] = epoch
                    self.wandb_run.log(log_data)
                    self.job_set.remove((epoch, job_name, job))
            time.sleep(1)

    def run(self):
        if self.accelerator.is_main_process:
            executor = ThreadPoolExecutor(max_workers=4)
            self.job_set = set()
            lock = threading.Lock()

            self.monitor_thread = threading.Thread(
                target=self.monitor_jobs, args=(lock,), daemon=True
            )
            self.monitor_thread.start()

        init_epoch = self.epoch if self._resume_from_batch_ckpt else self.epoch + 1
        if init_epoch > self.total_epochs:
            log.info(
                "Checkpoint is already at epoch %s; training.epochs=%s, nothing to run.",
                self.epoch,
                self.total_epochs,
            )
            return
        for epoch in range(init_epoch, self.total_epochs + 1):
            self.epoch = epoch
            self.accelerator.wait_for_everyone()
            reset_cuda_peak_memory(self.device)
            epoch_start = perf_counter()
            train_start = perf_counter()
            self.train()
            train_time = elapsed_since(train_start)
            self.accelerator.wait_for_everyone()
            val_start = perf_counter()
            self.val()
            validation_time = elapsed_since(val_start)
            epoch_time = elapsed_since(epoch_start)
            epoch_log = self.logs_flash(step=self.epoch)
            try:
                self._write_train_metrics(
                    epoch_log=epoch_log,
                    epoch_time=epoch_time,
                    train_time=train_time,
                    validation_time=validation_time,
                )
            except Exception as exc:
                log.warning("Failed to write training metrics: %s", exc)
            if self.epoch % self.cfg.training.save_every_x_epoch == 0:
                ckpt_path, model_name, model_epoch = self.save_ckpt()
                self.cleanup_batch_ckpts()
                # main thread only: launch planning jobs on the saved ckpt
                if (
                    self.cfg.plan_settings.plan_cfg_path is not None
                    and ckpt_path is not None
                ):  # ckpt_path is only not None for main process
                    from plan import build_plan_cfg_dicts, launch_plan_jobs

                    cfg_dicts = build_plan_cfg_dicts(
                        plan_cfg_path=os.path.join(
                            self.base_path, self.cfg.plan_settings.plan_cfg_path
                        ),
                        ckpt_base_path=self.cfg.ckpt_base_path,
                        model_name=model_name,
                        model_epoch=model_epoch,
                        planner=self.cfg.plan_settings.planner,
                        goal_source=self.cfg.plan_settings.goal_source,
                        goal_H=self.cfg.plan_settings.goal_H,
                        alpha=self.cfg.plan_settings.alpha,
                    )
                    jobs = launch_plan_jobs(
                        epoch=self.epoch,
                        cfg_dicts=cfg_dicts,
                        plan_output_dir=os.path.join(
                            os.getcwd(), "submitit-evals", f"epoch_{self.epoch}"
                        ),
                    )
                    with lock:
                        self.job_set.update(jobs)

    def err_eval_single(self, z_pred, z_tgt):
        logs = {}
        for k in z_pred.keys():
            loss = self.model.emb_criterion(z_pred[k], z_tgt[k])
            logs[k] = loss
        return logs

    def err_eval(self, z_out, z_tgt, state_tgt=None):
        """
        z_pred: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        z_tgt: (b, n_hist, n_patches, emb_dim), doesn't include action dims
        state:  (b, n_hist, dim)
        """
        logs = {}
        slices = {
            "full": (None, None),
            "pred": (-self.model.num_pred, None),
            "next1": (-self.model.num_pred, -self.model.num_pred + 1),
        }
        for name, (start_idx, end_idx) in slices.items():
            z_out_slice = slice_trajdict_with_t(
                z_out, start_idx=start_idx, end_idx=end_idx
            )
            z_tgt_slice = slice_trajdict_with_t(
                z_tgt, start_idx=start_idx, end_idx=end_idx
            )
            z_err = self.err_eval_single(z_out_slice, z_tgt_slice)

            logs.update({f"z_{k}_err_{name}": v for k, v in z_err.items()})

        return logs

    def train(self):
        save_every_x_batch = int(self.cfg.training.get("save_every_x_batch", 0) or 0)
        start_batch_idx = (
            int(self.resume_batch_idx) if self._resume_from_batch_ckpt else 0
        )
        train_loader = self.dataloaders["train"]
        total_batches = len(train_loader) if hasattr(train_loader, "__len__") else None
        if start_batch_idx > 0:
            log.info(
                "Skipping %s already-finished train batches for epoch %s",
                start_batch_idx,
                self.epoch,
            )
        if total_batches is not None and start_batch_idx >= total_batches:
            train_iter = iter(())
        elif start_batch_idx > 0:
            train_iter = itertools.islice(train_loader, start_batch_idx, None)
        else:
            train_iter = train_loader
        progress_kwargs = {"desc": f"Epoch {self.epoch} Train"}
        if total_batches is not None:
            progress_kwargs.update({"total": total_batches, "initial": start_batch_idx})

        last_batch_idx = None
        for i, data in enumerate(tqdm(train_iter, **progress_kwargs), start=start_batch_idx):
            last_batch_idx = i
            obs, act, state = data
            plot = i == 0  # only plot from the first batch
            self.model.train()
            z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                obs, act
            )

            self.encoder_optimizer.zero_grad()
            if self.cfg.has_decoder:
                self.decoder_optimizer.zero_grad()
            if self.cfg.has_predictor:
                self.predictor_optimizer.zero_grad()
                self.action_encoder_optimizer.zero_grad()

            self.accelerator.backward(loss)

            if self.model.train_encoder:
                self.encoder_optimizer.step()
            if self.cfg.has_decoder and self.model.train_decoder:
                self.decoder_optimizer.step()
            if self.cfg.has_predictor and self.model.train_predictor:
                self.predictor_optimizer.step()
                self.action_encoder_optimizer.step()

            loss = self.accelerator.gather_for_metrics(loss).mean()

            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }
            if self.cfg.has_decoder and plot:
                # only eval images when plotting due to speed
                if self.cfg.has_predictor:
                    z_obs_out, z_act_out = self.model.separate_emb(z_out)
                    z_gt = self.model.encode_obs(obs)
                    z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)

                    state_tgt = state[:, -self.model.num_hist :]  # (b, num_hist, dim)
                    err_logs = self.err_eval(z_obs_out, z_tgt)

                    err_logs = self.accelerator.gather_for_metrics(err_logs)
                    err_logs = {
                        key: value.mean().item() for key, value in err_logs.items()
                    }
                    err_logs = {f"train_{k}": [v] for k, v in err_logs.items()}

                    self.logs_update(err_logs)

                if visual_out is not None:
                    for t in range(
                        self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred
                    ):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.cfg.num_pred], obs["visual"][:, t]
                        )
                        img_pred_scores = self.accelerator.gather_for_metrics(
                            img_pred_scores
                        )
                        img_pred_scores = {
                            f"train_img_{k}_pred": [v.mean().item()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                if visual_reconstructed is not None:
                    for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t]
                        )
                        img_reconstruction_scores = self.accelerator.gather_for_metrics(
                            img_reconstruction_scores
                        )
                        img_reconstruction_scores = {
                            f"train_img_{k}_reconstructed": [v.mean().item()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="train",
                )

            loss_components = {f"train_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)

            if save_every_x_batch > 0 and (i + 1) % save_every_x_batch == 0:
                self.save_batch_ckpt(i)

        if save_every_x_batch > 0 and last_batch_idx is not None:
            if (last_batch_idx + 1) % save_every_x_batch != 0:
                self.save_batch_ckpt(last_batch_idx)
        self.resume_batch_idx = 0
        self._resume_from_batch_ckpt = False

    def val(self):
        self.model.eval()
        if len(self.train_traj_dset) > 0 and self.cfg.has_predictor:
            with torch.no_grad():
                train_rollout_logs = self.openloop_rollout(
                    self.train_traj_dset, mode="train"
                )
                train_rollout_logs = {
                    f"train_{k}": [v] for k, v in train_rollout_logs.items()
                }
                self.logs_update(train_rollout_logs)
                val_rollout_logs = self.openloop_rollout(self.val_traj_dset, mode="val")
                val_rollout_logs = {
                    f"val_{k}": [v] for k, v in val_rollout_logs.items()
                }
                self.logs_update(val_rollout_logs)

        self.accelerator.wait_for_everyone()
        for i, data in enumerate(
            tqdm(self.dataloaders["valid"], desc=f"Epoch {self.epoch} Valid")
        ):
            obs, act, state = data
            plot = i == 0
            self.model.eval()
            z_out, visual_out, visual_reconstructed, loss, loss_components = self.model(
                obs, act
            )

            loss = self.accelerator.gather_for_metrics(loss).mean()

            loss_components = self.accelerator.gather_for_metrics(loss_components)
            loss_components = {
                key: value.mean().item() for key, value in loss_components.items()
            }

            if self.cfg.has_decoder and plot:
                # only eval images when plotting due to speed
                if self.cfg.has_predictor:
                    z_obs_out, z_act_out = self.model.separate_emb(z_out)
                    z_gt = self.model.encode_obs(obs)
                    z_tgt = slice_trajdict_with_t(z_gt, start_idx=self.model.num_pred)

                    state_tgt = state[:, -self.model.num_hist :]  # (b, num_hist, dim)
                    err_logs = self.err_eval(z_obs_out, z_tgt)

                    err_logs = self.accelerator.gather_for_metrics(err_logs)
                    err_logs = {
                        key: value.mean().item() for key, value in err_logs.items()
                    }
                    err_logs = {f"val_{k}": [v] for k, v in err_logs.items()}

                    self.logs_update(err_logs)

                if visual_out is not None:
                    for t in range(
                        self.cfg.num_hist, self.cfg.num_hist + self.cfg.num_pred
                    ):
                        img_pred_scores = eval_images(
                            visual_out[:, t - self.cfg.num_pred], obs["visual"][:, t]
                        )
                        img_pred_scores = self.accelerator.gather_for_metrics(
                            img_pred_scores
                        )
                        img_pred_scores = {
                            f"val_img_{k}_pred": [v.mean().item()]
                            for k, v in img_pred_scores.items()
                        }
                        self.logs_update(img_pred_scores)

                if visual_reconstructed is not None:
                    for t in range(obs["visual"].shape[1]):
                        img_reconstruction_scores = eval_images(
                            visual_reconstructed[:, t], obs["visual"][:, t]
                        )
                        img_reconstruction_scores = self.accelerator.gather_for_metrics(
                            img_reconstruction_scores
                        )
                        img_reconstruction_scores = {
                            f"val_img_{k}_reconstructed": [v.mean().item()]
                            for k, v in img_reconstruction_scores.items()
                        }
                        self.logs_update(img_reconstruction_scores)

                self.plot_samples(
                    obs["visual"],
                    visual_out,
                    visual_reconstructed,
                    self.epoch,
                    batch=i,
                    num_samples=self.num_reconstruct_samples,
                    phase="valid",
                )
            loss_components = {f"val_{k}": [v] for k, v in loss_components.items()}
            self.logs_update(loss_components)

    def openloop_rollout(
        self, dset, num_rollout=10, rand_start_end=True, min_horizon=2, mode="train"
    ):
        np.random.seed(self.cfg.training.seed)
        min_horizon = min_horizon + self.cfg.num_hist
        plotting_dir = f"rollout_plots/e{self.epoch}_rollout"
        if self.accelerator.is_main_process:
            os.makedirs(plotting_dir, exist_ok=True)
        self.accelerator.wait_for_everyone()
        logs = {}

        # rollout with both num_hist and 1 frame as context
        num_past = [(self.cfg.num_hist, ""), (1, "_1framestart")]
        min_valid_horizon = max(1, max(n_past for n_past, _ in num_past) - 1)
        max_sample_attempts = max(100, len(dset) * 2)

        # sample traj
        for idx in range(num_rollout):
            valid_traj = False
            attempts = 0
            while not valid_traj and attempts < max_sample_attempts:
                attempts += 1
                traj_idx = np.random.randint(0, len(dset))
                obs, act, state, _ = dset[traj_idx]
                act = act.to(self.device)
                if rand_start_end:
                    if obs["visual"].shape[0] > min_horizon * self.cfg.frameskip + 1:
                        start = np.random.randint(
                            0,
                            obs["visual"].shape[0] - min_horizon * self.cfg.frameskip - 1,
                        )
                    else:
                        start = 0
                    max_horizon = (obs["visual"].shape[0] - start - 1) // self.cfg.frameskip
                    if max_horizon >= min_valid_horizon:
                        valid_traj = True
                        low_horizon = min(min_horizon, max_horizon)
                        if max_horizon > low_horizon:
                            horizon = np.random.randint(low_horizon, max_horizon + 1)
                        else:
                            horizon = max_horizon
                else:
                    valid_traj = True
                    start = 0
                    horizon = (obs["visual"].shape[0] - 1) // self.cfg.frameskip
            if not valid_traj:
                log.warning(
                    "Skipping openloop rollout sample %s for %s: no trajectory "
                    "supports min_valid_horizon=%s with frameskip=%s.",
                    idx,
                    mode,
                    min_valid_horizon,
                    self.cfg.frameskip,
                )
                continue

            for k in obs.keys():
                obs[k] = obs[k][
                    start : 
                    start + horizon * self.cfg.frameskip + 1 : 
                    self.cfg.frameskip
                ]
            act = act[start : start + horizon * self.cfg.frameskip]
            act = rearrange(act, "(h f) d -> h (f d)", f=self.cfg.frameskip)

            obs_g = {}
            for k in obs.keys():
                obs_g[k] = obs[k][-1].unsqueeze(0).unsqueeze(0).to(self.device)
            z_g = self.model.encode_obs(obs_g)
            actions = act.unsqueeze(0)

            for past in num_past:
                n_past, postfix = past

                obs_0 = {}
                for k in obs.keys():
                    obs_0[k] = (
                        obs[k][:n_past].unsqueeze(0).to(self.device)
                    )  # unsqueeze for batch, (b, t, c, h, w)

                z_obses, z = self.model.rollout(obs_0, actions)
                z_obs_last = slice_trajdict_with_t(z_obses, start_idx=-1, end_idx=None)
                div_loss = self.err_eval_single(z_obs_last, z_g)

                for k in div_loss.keys():
                    log_key = f"z_{k}_err_rollout{postfix}"
                    if log_key in logs:
                        logs[f"z_{k}_err_rollout{postfix}"].append(
                            div_loss[k]
                        )
                    else:
                        logs[f"z_{k}_err_rollout{postfix}"] = [
                            div_loss[k]
                        ]

                if self.cfg.has_decoder:
                    visuals = self.model.decode_obs(z_obses)[0]["visual"]
                    imgs = torch.cat([obs["visual"], visuals[0].cpu()], dim=0)
                    self.plot_imgs(
                        imgs,
                        obs["visual"].shape[0],
                        f"{plotting_dir}/e{self.epoch}_{mode}_{idx}{postfix}.png",
                    )
        logs = {
            key: sum(values) / len(values) for key, values in logs.items() if values
        }
        return logs

    def logs_update(self, logs):
        for key, value in logs.items():
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().item()
            length = len(value)
            count, total = self.epoch_log.get(key, (0, 0.0))
            self.epoch_log[key] = (
                count + length,
                total + sum(value),
            )

    def logs_flash(self, step):
        epoch_log = OrderedDict()
        for key, value in self.epoch_log.items():
            count, sum = value
            to_log = sum / count
            epoch_log[key] = to_log
        epoch_log["epoch"] = step
        if "train_loss" in epoch_log and "val_loss" in epoch_log:
            log.info(f"Epoch {self.epoch}  Training loss: {epoch_log['train_loss']:.4f}  \
                    Validation loss: {epoch_log['val_loss']:.4f}")
        else:
            log.info(f"Epoch {self.epoch} metrics: {epoch_log}")

        if self.accelerator.is_main_process:
            self.wandb_run.log(epoch_log)
        self.epoch_log = OrderedDict()
        return epoch_log

    def plot_samples(
        self,
        gt_imgs,
        pred_imgs,
        reconstructed_gt_imgs,
        epoch,
        batch,
        num_samples=2,
        phase="train",
    ):
        """
        input:  gt_imgs, reconstructed_gt_imgs: (b, num_hist + num_pred, 3, img_size, img_size)
                pred_imgs: (b, num_hist, 3, img_size, img_size)
        output:   imgs: (b, num_frames, 3, img_size, img_size)
        """
        num_frames = gt_imgs.shape[1]
        # sample num_samples images
        gt_imgs, pred_imgs, reconstructed_gt_imgs = sample_tensors(
            [gt_imgs, pred_imgs, reconstructed_gt_imgs],
            num_samples,
            indices=list(range(num_samples))[: gt_imgs.shape[0]],
        )

        num_samples = min(num_samples, gt_imgs.shape[0])

        # fill in blank images for frameskips
        if pred_imgs is not None:
            pred_imgs = torch.cat(
                (
                    torch.full(
                        (num_samples, self.model.num_pred, *pred_imgs.shape[2:]),
                        -1,
                        device=self.device,
                    ),
                    pred_imgs,
                ),
                dim=1,
            )
        else:
            pred_imgs = torch.full(gt_imgs.shape, -1, device=self.device)

        pred_imgs = rearrange(pred_imgs, "b t c h w -> (b t) c h w")
        gt_imgs = rearrange(gt_imgs, "b t c h w -> (b t) c h w")
        reconstructed_gt_imgs = rearrange(
            reconstructed_gt_imgs, "b t c h w -> (b t) c h w"
        )
        imgs = torch.cat([gt_imgs, pred_imgs, reconstructed_gt_imgs], dim=0)

        if self.accelerator.is_main_process:
            os.makedirs(phase, exist_ok=True)
        self.accelerator.wait_for_everyone()

        self.plot_imgs(
            imgs,
            num_columns=num_samples * num_frames,
            img_name=f"{phase}/{phase}_e{str(epoch).zfill(5)}_b{batch}.png",
        )

    def plot_imgs(self, imgs, num_columns, img_name):
        utils.save_image(
            imgs,
            img_name,
            nrow=num_columns,
            normalize=True,
            value_range=(-1, 1),
        )


@hydra.main(config_path="conf", config_name="train")
def main(cfg: OmegaConf):
    trainer = Trainer(cfg)
    trainer.run()


if __name__ == "__main__":
    main()
