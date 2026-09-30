import argparse
import csv
import json
import math
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


DEFAULT_MODEL_NAME = "2026-04-27/17-38-58"
DEFAULT_CKPT_BASE_PATH = "<CKPT_ROOT>"
DEFAULT_MODEL_EPOCHS = "10,20,30,40,50,60,70,80,90,100"
DEFAULT_EVAL_SEEDS = "1:60"


def parse_int_list(value):
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    value = str(value).strip()
    if not value:
        return []
    if ":" in value:
        start, end = value.split(":", 1)
        return list(range(int(start), int(end) + 1))
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_epoch_list(value):
    if isinstance(value, (list, tuple)):
        values = value
    else:
        value = str(value).strip()
        if not value:
            return []
        if ":" in value:
            start, end = value.split(":", 1)
            return list(range(int(start), int(end) + 1))
        values = [x.strip() for x in value.split(",") if x.strip()]

    epochs = []
    for item in values:
        text = str(item).strip()
        if text.lower() == "latest":
            epochs.append("latest")
        else:
            epochs.append(int(text))
    return epochs


def chunks(values, chunk_size):
    return [values[i : i + chunk_size] for i in range(0, len(values), chunk_size)]


def fmt(value):
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return "nan"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def read_json(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def read_jsonl(path):
    rows = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    except Exception:
        pass
    return rows


def parse_log(log_file):
    text = log_file.read_text(errors="ignore")
    plan_dir = ""
    oom = 0
    success_list = []

    for line in text.splitlines():
        if "Planning result saved dir:" in line:
            plan_dir = line.split("Planning result saved dir:", 1)[1].strip()
        if "CUDA out of memory" in line or "OutOfMemoryError" in line:
            oom = 1
        match = re.search(r"Success rate:\s*([0-9]*\.?[0-9]+)", line)
        if match:
            success_list.append(float(match.group(1)))

    return {
        "oom": oom,
        "last_success_from_log": success_list[-1] if success_list else float("nan"),
        "plan_output_dir": plan_dir,
    }


def load_run_outputs(plan_output_dir):
    if not plan_output_dir:
        return {}, []
    run_dir = Path(plan_output_dir)
    summary = read_json(run_dir / "summary.json") or {}
    per_eval = read_jsonl(run_dir / "per_eval_results.jsonl")
    return summary, per_eval


def summarize_chunk(epoch, chunk_id, seed_chunk, exit_code, log_file):
    parsed = parse_log(log_file)
    summary, per_eval = load_run_outputs(parsed["plan_output_dir"])

    if per_eval:
        success_count = sum(1 for row in per_eval if row.get("success"))
        eval_count = len(per_eval)
        success_rate = success_count / eval_count if eval_count else float("nan")
    else:
        success_rate = summary.get("success_rate", parsed["last_success_from_log"])
        eval_count = len(seed_chunk)
        success_count = (
            round(success_rate * eval_count)
            if isinstance(success_rate, float) and not math.isnan(success_rate)
            else None
        )

    return {
        "epoch": epoch,
        "chunk_id": chunk_id,
        "eval_seeds": json.dumps(seed_chunk, separators=(",", ":")),
        "exit_code": exit_code,
        "oom": parsed["oom"],
        "eval_count": eval_count,
        "success_count": success_count,
        "success_rate": success_rate,
        "mean_final_state_distance": summary.get("mean_final_state_distance"),
        "total_episode_time_sec": summary.get("total_episode_time_sec"),
        "total_planning_time_sec": summary.get("total_planning_time_sec"),
        "total_world_model_rollout_time_sec": summary.get(
            "total_world_model_rollout_time_sec"
        ),
        "environment_step_time_sec": summary.get("environment_step_time_sec"),
        "decoder_time_sec": summary.get("decoder_time_sec"),
        "visualization_time_sec": summary.get("visualization_time_sec"),
        "cuda_peak_memory_mb": summary.get("cuda_peak_memory_mb"),
        "mpc_iterations": summary.get("mpc_iterations"),
        "termination_reason": summary.get("termination_reason"),
        "plan_output_dir": parsed["plan_output_dir"],
        "log_file": str(log_file),
    }


def write_csv(path, rows, fieldnames):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: fmt(row.get(key)) for key in fieldnames})


def aggregate_epoch(rows, epochs, total_seed_count):
    aggregate_rows = []
    for epoch in epochs:
        epoch_rows = [row for row in rows if row["epoch"] == epoch]
        valid_rows = [
            row
            for row in epoch_rows
            if isinstance(row.get("success_count"), int)
            and row.get("exit_code") == 0
            and row.get("oom") == 0
        ]
        success_count = sum(row["success_count"] for row in valid_rows)
        eval_count = sum(row["eval_count"] for row in valid_rows)
        success_rate = success_count / total_seed_count if total_seed_count else math.nan

        def mean_metric(name):
            values = [
                row.get(name)
                for row in valid_rows
                if isinstance(row.get(name), (int, float))
            ]
            return sum(values) / len(values) if values else None

        aggregate_rows.append(
            {
                "epoch": epoch,
                "num_chunks": len(epoch_rows),
                "num_valid_chunks": len(valid_rows),
                "eval_count": eval_count,
                "success_count": success_count,
                "success_rate_over_requested_seeds": success_rate,
                "oom_chunks": sum(int(row["oom"]) for row in epoch_rows),
                "mean_final_state_distance": mean_metric("mean_final_state_distance"),
                "mean_total_episode_time_sec": mean_metric("total_episode_time_sec"),
                "mean_total_planning_time_sec": mean_metric(
                    "total_planning_time_sec"
                ),
                "mean_total_world_model_rollout_time_sec": mean_metric(
                    "total_world_model_rollout_time_sec"
                ),
                "mean_environment_step_time_sec": mean_metric(
                    "environment_step_time_sec"
                ),
                "mean_decoder_time_sec": mean_metric("decoder_time_sec"),
                "mean_visualization_time_sec": mean_metric("visualization_time_sec"),
                "mean_cuda_peak_memory_mb": mean_metric("cuda_peak_memory_mb"),
            }
        )
    return aggregate_rows


def build_parser():
    parser = argparse.ArgumentParser(
        description="Run PointMaze planning evaluation in seed chunks."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL_NAME)
    parser.add_argument("--ckpt-base-path", default=DEFAULT_CKPT_BASE_PATH)
    parser.add_argument("--model-epochs", default=DEFAULT_MODEL_EPOCHS)
    parser.add_argument("--eval-seeds", default=DEFAULT_EVAL_SEEDS)
    parser.add_argument("--chunk-size", type=int, default=10)
    parser.add_argument("--config-name", default="plan_point_maze.yaml")
    parser.add_argument("--goal-H", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--opt-steps", type=int, default=1)
    parser.add_argument("--max-mpc-iters", type=int, default=15)
    parser.add_argument("--n-plot-samples", type=int, default=1)
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
    epochs = parse_epoch_list(args.model_epochs)
    eval_seeds = parse_int_list(args.eval_seeds)
    seed_chunks = chunks(eval_seeds, args.chunk_size)

    run_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = root / "eval_suite_results" / run_tag
    log_dir = out_dir / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"Output dir: {out_dir}")
    print(f"Epochs: {epochs}")
    print(f"Seed chunks: {seed_chunks}")

    chunk_rows = []
    for epoch in epochs:
        for chunk_id, seed_chunk in enumerate(seed_chunks, start=1):
            log_file = log_dir / f"epoch_{epoch}_chunk_{chunk_id}.log"
            seed_list = json.dumps(seed_chunk, separators=(",", ":"))
            cmd = [
                sys.executable,
                str(plan_py),
                "--config-name",
                args.config_name,
                f"model_name={args.model_name}",
                f"ckpt_base_path={args.ckpt_base_path}",
                f"model_epoch={epoch}",
                f"goal_H={args.goal_H}",
                f"n_evals={len(seed_chunk)}",
                f"+eval_seed_list={seed_list}",
                f"n_plot_samples={args.n_plot_samples}",
                f"planner.max_iter={args.max_mpc_iters}",
                f"planner.sub_planner.num_samples={args.num_samples}",
                f"planner.sub_planner.topk={args.topk}",
                f"planner.sub_planner.opt_steps={args.opt_steps}",
                "use_roi=false",
                "roi_mode=none",
            ]
            cmd.extend(args.extra_override)

            print("=" * 80)
            print(f"epoch={epoch}, chunk={chunk_id}, seeds={seed_chunk}")
            print(" ".join(cmd))
            print("=" * 80)

            env = os.environ.copy()
            env.setdefault("WANDB_MODE", "disabled")
            with log_file.open("w", encoding="utf-8") as lf:
                proc = subprocess.run(
                    cmd,
                    cwd=str(root),
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    env=env,
                )

            row = summarize_chunk(epoch, chunk_id, seed_chunk, proc.returncode, log_file)
            chunk_rows.append(row)
            print(
                f"done: epoch={epoch}, chunk={chunk_id}, "
                f"exit={row['exit_code']}, oom={row['oom']}, "
                f"success={fmt(row['success_rate'])}"
            )
            time.sleep(2)

    chunk_fields = [
        "epoch",
        "chunk_id",
        "eval_seeds",
        "exit_code",
        "oom",
        "eval_count",
        "success_count",
        "success_rate",
        "mean_final_state_distance",
        "total_episode_time_sec",
        "total_planning_time_sec",
        "total_world_model_rollout_time_sec",
        "environment_step_time_sec",
        "decoder_time_sec",
        "visualization_time_sec",
        "cuda_peak_memory_mb",
        "mpc_iterations",
        "termination_reason",
        "plan_output_dir",
        "log_file",
    ]
    aggregate_rows = aggregate_epoch(chunk_rows, epochs, len(eval_seeds))
    aggregate_fields = [
        "epoch",
        "num_chunks",
        "num_valid_chunks",
        "eval_count",
        "success_count",
        "success_rate_over_requested_seeds",
        "oom_chunks",
        "mean_final_state_distance",
        "mean_total_episode_time_sec",
        "mean_total_planning_time_sec",
        "mean_total_world_model_rollout_time_sec",
        "mean_environment_step_time_sec",
        "mean_decoder_time_sec",
        "mean_visualization_time_sec",
        "mean_cuda_peak_memory_mb",
    ]

    chunk_csv = out_dir / "chunk_summary.csv"
    aggregate_csv = out_dir / "epoch_aggregate.csv"
    write_csv(chunk_csv, chunk_rows, chunk_fields)
    write_csv(aggregate_csv, aggregate_rows, aggregate_fields)
    print("All done.")
    print(f"Chunk summary: {chunk_csv}")
    print(f"Epoch aggregate: {aggregate_csv}")


if __name__ == "__main__":
    main()
