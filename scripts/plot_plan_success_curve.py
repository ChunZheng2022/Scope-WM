import argparse
import csv
import json
import math
import re
from pathlib import Path


def _load_json(path):
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _load_last_jsonl(path):
    last = None
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    last = json.loads(line)
                except Exception:
                    continue
    except Exception:
        return None
    return last


def _as_float(value):
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _as_epoch(value):
    if value is None:
        return None
    if isinstance(value, int):
        return value
    text = str(value)
    match = re.search(r"\d+", text)
    return int(match.group(0)) if match else None


def _timestamp_key(path):
    match = re.match(r"(\d{14})", path.name)
    return match.group(1) if match else path.name


def _read_plan_record(run_dir):
    summary = _load_json(run_dir / "summary.json")
    if summary:
        success = _as_float(summary.get("success_rate"))
        epoch = _as_epoch(summary.get("model_epoch"))
        if success is not None:
            return {
                "run_dir": str(run_dir),
                "epoch": epoch,
                "success_rate": success,
                "source": "summary.json",
                "model_epoch_raw": summary.get("model_epoch"),
                "model_name": summary.get("model_name"),
            }

    plan_metrics = _load_last_jsonl(run_dir / "plan_metrics.jsonl")
    if plan_metrics:
        success = _as_float(plan_metrics.get("success_rate"))
        if success is not None:
            return {
                "run_dir": str(run_dir),
                "epoch": _as_epoch(plan_metrics.get("model_epoch")),
                "success_rate": success,
                "source": "plan_metrics.jsonl",
                "model_epoch_raw": plan_metrics.get("model_epoch"),
                "model_name": plan_metrics.get("model_name"),
            }

    mpc_iter = _load_last_jsonl(run_dir / "mpc_iterations.jsonl")
    if mpc_iter:
        success = _as_float(
            mpc_iter.get("cumulative_success_rate")
            if "cumulative_success_rate" in mpc_iter
            else mpc_iter.get("iteration_success_rate")
        )
        if success is not None:
            return {
                "run_dir": str(run_dir),
                "epoch": None,
                "success_rate": success,
                "source": "mpc_iterations.jsonl",
                "model_epoch_raw": None,
                "model_name": None,
            }

    return None


def _resolve_run_name(root, run):
    if run is None:
        return None
    run_path = Path(run)
    if run_path.is_absolute():
        return run_path.name
    if run_path.parent != Path("."):
        try:
            return run_path.relative_to(root).parts[0]
        except ValueError:
            return run_path.name
    return run


def collect_records(
    root,
    infer_epoch_from_order=True,
    start_run=None,
    start_after_run=None,
):
    root = Path(root)
    run_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir()],
        key=_timestamp_key,
    )
    indexed_run_dirs = list(enumerate(run_dirs, start=1))

    start_name = _resolve_run_name(root, start_run)
    start_after_name = _resolve_run_name(root, start_after_run)
    if start_name and start_after_name:
        raise ValueError("Use only one of start_run or start_after_run.")

    if start_name or start_after_name:
        target_name = start_name or start_after_name
        target_index = None
        for order, run_dir in indexed_run_dirs:
            if run_dir.name == target_name:
                target_index = order
                break
        if target_index is None:
            raise ValueError(
                f"Start run {target_name!r} was not found under {root}."
            )
        min_order = target_index if start_name else target_index + 1
        indexed_run_dirs = [
            (order, run_dir)
            for order, run_dir in indexed_run_dirs
            if order >= min_order
        ]

    records = []
    for order, run_dir in indexed_run_dirs:
        record = _read_plan_record(run_dir)
        if record is None:
            continue
        record["order"] = order
        record["run_name"] = run_dir.name
        if record["epoch"] is None and infer_epoch_from_order:
            record["epoch"] = order
            record["epoch_inferred"] = True
        else:
            record["epoch_inferred"] = False
        records.append(record)

    records.sort(key=lambda row: (row["epoch"] is None, row["epoch"] or row["order"]))
    return records


def write_csv(records, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "epoch",
        "success_rate",
        "epoch_inferred",
        "source",
        "run_name",
        "run_dir",
        "model_epoch_raw",
        "model_name",
        "order",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def plot_records(records, out_path, title):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [row["epoch"] for row in records]
    success = [row["success_rate"] for row in records]
    best_idx = max(range(len(records)), key=lambda i: success[i])
    best_epoch = epochs[best_idx]
    best_success = success[best_idx]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(epochs, success, marker="o", linewidth=1.8, markersize=4)
    ax.scatter([best_epoch], [best_success], color="crimson", s=70, zorder=4)
    ax.annotate(
        f"best: epoch {best_epoch}, SR={best_success:.3f}",
        xy=(best_epoch, best_success),
        xytext=(10, 18),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "crimson"},
        color="crimson",
        fontsize=10,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.set_title(title)
    fig.tight_layout()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return records[best_idx]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Read DINO-WM plan_outputs/* plan summaries and plot success rate "
            "over epochs."
        )
    )
    parser.add_argument(
        "--root",
        default="plan_outputs",
        help="Directory containing per-run plan output folders.",
    )
    parser.add_argument(
        "--out",
        default="plan_outputs_success_curve.png",
        help="Output PNG path.",
    )
    parser.add_argument(
        "--csv",
        default="plan_outputs_success_curve.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--title",
        default="Plan Success Rate by Epoch",
        help="Plot title.",
    )
    parser.add_argument(
        "--no-infer-epoch",
        action="store_true",
        help="Do not infer missing epoch numbers from directory timestamp order.",
    )
    parser.add_argument(
        "--start-run",
        default=None,
        help=(
            "Only include this run folder and later folders in timestamp order. "
            "Accepts either the folder name or a path under --root."
        ),
    )
    parser.add_argument(
        "--start-after-run",
        default=None,
        help=(
            "Only include folders after this run folder in timestamp order. "
            "Accepts either the folder name or a path under --root."
        ),
    )
    args = parser.parse_args()

    records = collect_records(
        args.root,
        infer_epoch_from_order=not args.no_infer_epoch,
        start_run=args.start_run,
        start_after_run=args.start_after_run,
    )
    if not records:
        raise SystemExit(f"No readable plan results found under {args.root}")
    if any(row["epoch"] is None for row in records):
        missing = [row["run_name"] for row in records if row["epoch"] is None]
        raise SystemExit(
            "Some records do not have an epoch. Re-run without --no-infer-epoch "
            f"or fix summaries: {missing[:5]}"
        )

    write_csv(records, args.csv)
    best = plot_records(records, args.out, args.title)

    print(f"Read {len(records)} plan runs from {args.root}")
    print(f"Wrote CSV: {args.csv}")
    print(f"Wrote plot: {args.out}")
    print(
        "Best: "
        f"epoch={best['epoch']} "
        f"success_rate={best['success_rate']:.6f} "
        f"run={best['run_name']}"
    )


if __name__ == "__main__":
    main()
