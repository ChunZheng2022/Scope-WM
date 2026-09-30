import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def parse_seed_range(value):
    text = str(value).strip()
    if ":" in text:
        start, end = text.split(":", 1)
        return list(range(int(start), int(end) + 1))
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run one PointMaze planning job with 60 seeds, decoder disabled."
    )
    parser.add_argument("--model-name", default="point_maze")
    parser.add_argument("--ckpt-base-path", required=True)
    parser.add_argument("--model-epoch", default="latest")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--eval-seeds", default="1:60")
    parser.add_argument("--config-name", default="plan_point_maze.yaml")
    parser.add_argument("--goal-H", type=int, default=5)
    parser.add_argument("--max-mpc-iters", type=int, default=15)
    parser.add_argument("--num-samples", type=int, default=300)
    parser.add_argument("--topk", type=int, default=30)
    parser.add_argument("--opt-steps", type=int, default=10)
    parser.add_argument("--use-roi", action="store_true")
    parser.add_argument("--roi-mode", default="none")
    parser.add_argument("--roi-keep-ratio", type=float, default=1.0)
    parser.add_argument("--roi-head-checkpoint", default=None)
    parser.add_argument(
        "--wandb-logging",
        action="store_true",
        help="Enable wandb logging. Disabled by default to avoid wandb startup stalls.",
    )
    parser.add_argument(
        "--extra-override",
        action="append",
        default=[],
        help="Additional Hydra override. Can be passed multiple times.",
    )
    return parser


def main():
    args = build_parser().parse_args()
    root = Path(__file__).resolve().parents[1]
    plan_py = root / "plan.py"
    seeds = parse_seed_range(args.eval_seeds)
    seed_list = json.dumps(seeds, separators=(",", ":"))

    cmd = [
        sys.executable,
        str(plan_py),
        "--config-name",
        args.config_name,
        f"model_name={args.model_name}",
        f"ckpt_base_path={args.ckpt_base_path}",
        f"model_epoch={args.model_epoch}",
        f"goal_H={args.goal_H}",
        f"n_evals={len(seeds)}",
        f"+eval_seed_list={seed_list}",
        "load_decoder=false",
        "n_plot_samples=0",
        f"wandb_logging={str(args.wandb_logging).lower()}",
        f"planner.max_iter={args.max_mpc_iters}",
        f"planner.sub_planner.num_samples={args.num_samples}",
        f"planner.sub_planner.topk={args.topk}",
        f"planner.sub_planner.opt_steps={args.opt_steps}",
        f"use_roi={str(args.use_roi).lower()}",
        f"roi_mode={args.roi_mode}",
        f"roi_keep_ratio={args.roi_keep_ratio}",
    ]
    if args.roi_head_checkpoint is not None:
        cmd.append(f"roi_head_checkpoint={args.roi_head_checkpoint}")
    cmd.extend(args.extra_override)

    env = os.environ.copy()
    env.setdefault("D4RL_SUPPRESS_IMPORT_ERROR", "1")
    env.setdefault("WANDB_MODE", "disabled")
    if args.dataset_dir is not None:
        env["DATASET_DIR"] = args.dataset_dir

    print("Running one no-decoder PointMaze planning job.")
    print(f"Seeds: {seeds}")
    print("Command:")
    print(" ".join(cmd))
    raise SystemExit(subprocess.call(cmd, cwd=str(root), env=env))


if __name__ == "__main__":
    main()
