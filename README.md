# Scope-WM

Scope-WM is a sparse visual world-model framework built on pretrained visual
features. The core idea is to keep only task-relevant visual tokens for the
primary dynamics rollout, recover useful background context with a lightweight
foreground-conditioned update, and improve low-budget MPC with an elite-bank CEM
proposal prior.

This repository contains the implementation used in our experiments, including:

- **DRS**: Distilled Relevance Selection for action-conditioned token selection.
- **FDBU**: Foreground-Delta Background Update for lightweight background
  consistency.
- **EB-CEM**: Elite-Bank CEM for reusing good action-sequence proposals under a
  small sampling budget.

We are also preparing a new implementation based on `stable-worldmodel`, which
we plan to release soon.

## Checkpoints

Pretrained Scope-WM checkpoints and task-specific DRS heads are available here:

- [Baidu Netdisk](https://pan.baidu.com/s/1duYoX5bi9J49IaVIhg94Qg?pwd=SCWM)
- Extraction code: `SCWM`

The package includes checkpoints for `topk=32` and `topk=98`, plus DRS heads for
the evaluated tasks.

## Code Map

- `models/roi.py`: DRS selector and action-conditioned relevance head.
- `models/sparse_dynamics.py`: sparse rollout and FDBU modules.
- `models/visual_world_model.py`: world-model integration.
- `planning/cem.py`: CEM and EB-CEM planning.
- `planning/mpc.py`: MPC orchestration.
- `scripts/export_drs_targets.py`: export DRS distillation targets.
- `scripts/train_drs_head.py`: train DRS heads.
- `scripts/profile_wm_compute.py`: FLOPs and throughput profiling.
- `scripts/visualize_drs_fdbu_maps.py`: DRS/FDBU visualization utilities.

## Installation

```bash
conda env create -f environment.yml
conda activate scope_wm

export PYTHONPATH=.
export D4RL_SUPPRESS_IMPORT_ERROR=1
export WANDB_MODE=disabled
export OMP_NUM_THREADS=4
export DINOV2_LOCAL_PATH=/path/to/dinov2
export DINOV2_SOURCE=local
```

Rope and Granular require PyFleX. See `PYFLEX_SETUP.md` for notes.

## Dataset Layout

Place datasets under a local root and pass paths explicitly:

```text
DATA_ROOT/
├── deformable/
│   ├── granular/
│   └── rope/
├── point_maze/
├── pusht_noise/
└── wall_single/
```

Example environment variables:

```bash
export DATA_ROOT=/path/to/data
export CKPT_ROOT=/path/to/checkpoints
export OUTPUT_ROOT=/path/to/outputs
export DRS_OUT=/path/to/drs_outputs

export PUSHT_DATA=$DATA_ROOT/pusht_noise
export POINTMAZE_DATA=$DATA_ROOT/point_maze
export WALL_DATA=$DATA_ROOT/wall_single
export DEFORM_DATA=$DATA_ROOT/deformable
```

## Basic Usage

### 1. Train a dense teacher

```bash
python train.py --config-name train.yaml \
  env=pusht frameskip=5 num_hist=3 data_path=$PUSHT_DATA \
  ckpt_base_path=$CKPT_ROOT/pusht_dense \
  has_decoder=false model.train_decoder=false \
  use_drs=false drs_mode=none sparse_dynamics.enabled=false
```

### 2. Export DRS targets

```bash
python scripts/export_drs_targets.py \
  --teacher-checkpoint $TEACHER_CKPT \
  --train-config $TEACHER_HYDRA_YAML \
  --data-path $PUSHT_DATA \
  --output $DRS_OUT/pusht_drs_targets.pt \
  --split train --batch-size 8 --max-batches 400 --device cuda:0
```

### 3. Train a DRS head

```bash
python scripts/train_drs_head.py \
  --targets $DRS_OUT/pusht_drs_targets.pt \
  --output $DRS_OUT/pusht_drs_head.pt \
  --epochs 80 --batch-size 256 --loss-type kl --device cuda:0 --seed 0
```

### 4. Train Scope-WM

Use `drs_topk=98` for the **Fast** setting and `drs_topk=32` for the
**Faster** setting.

```bash
python train.py --config-name train.yaml \
  env=pusht frameskip=5 num_hist=3 data_path=$PUSHT_DATA \
  ckpt_base_path=$OUTPUT_ROOT/pusht_scopewm_fast \
  has_decoder=false model.train_decoder=false \
  use_drs=true drs_mode=distilled_topk drs_topk=98 \
  drs_head_checkpoint=$DRS_OUT/pusht_drs_head.pt \
  sparse_dynamics.enabled=true sparse_dynamics.mode=sparse_primary \
  sparse_dynamics.mask_source=drs \
  sparse_dynamics.background_processor=fdbu
```

### 5. Plan with EB-CEM

```bash
python plan.py --config-name plan_pusht.yaml \
  model_name=$MODEL_NAME ckpt_base_path=$OUTPUT_ROOT/pusht_scopewm_fast \
  model_epoch=$SELECTED_EPOCH data_path=$PUSHT_DATA \
  n_evals=50 planner.max_iter=15 \
  planner.frontloaded_cem_samples.enabled=true \
  planner.sub_planner.eb_cem_enabled=true \
  use_drs=true drs_mode=distilled_topk drs_topk=98 \
  drs_head_checkpoint=$DRS_OUT/pusht_drs_head.pt \
  sparse_dynamics.enabled=true sparse_dynamics.background_processor=fdbu
```

EB-CEM uses a 300-sample first MPC iteration and 100-sample later iterations in
our main configuration. Reproduction records for task-specific settings are in
`conf/reproduce/`.

## Task Settings

| Task | frameskip | num_hist | eval instances | max MPC steps |
| --- | ---: | ---: | ---: | ---: |
| PointMaze | 5 | 3 | 50 | 15 |
| Wall | 5 | 1 | 50 | 15 |
| Push-T | 5 | 3 | 50 | 15 |
| Rope | 1 | 1 | 10 | 7 |
| Granular | 1 | 1 | 10 | 7 |

## Profiling

```bash
python scripts/profile_wm_compute.py \
  --ckpt-base-path $OUTPUT_ROOT/pusht_scopewm_fast \
  --model-name $MODEL_NAME --model-epoch $SELECTED_EPOCH \
  --batch-size 128 --device cuda:0 \
  use_drs=true drs_mode=distilled_topk drs_topk=98 \
  drs_head_checkpoint=$DRS_OUT/pusht_drs_head.pt \
  sparse_dynamics.enabled=true sparse_dynamics.background_processor=fdbu
```

## Acknowledgements

Scope-WM builds on and is inspired by several excellent projects and papers:

- [DINO-WM](https://github.com/gaoyuezhou/dino_wm)
- [DDP-WM](https://github.com/HCPLab-SYSU/DDP-WM)
- [DINOv2](https://github.com/facebookresearch/dinov2)

Please also see `THIRD_PARTY_NOTICES.md` for third-party components retained in
this codebase.
