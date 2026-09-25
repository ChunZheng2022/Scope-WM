import argparse
import csv
import json
from pathlib import Path


def flatten_dict(data, prefix=""):
    flat = {}
    for key, value in data.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            flat.update(flatten_dict(value, name))
        elif isinstance(value, (list, tuple)):
            flat[name] = json.dumps(value, sort_keys=True)
        else:
            flat[name] = value
    return flat


def load_summary(path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def collect_summaries(root, kind):
    rows = []
    for summary_path in Path(root).rglob("summary.json"):
        try:
            summary = load_summary(summary_path)
        except Exception:
            continue
        if kind is not None and summary.get("kind") != kind:
            continue
        row = flatten_dict(summary)
        row["summary_path"] = str(summary_path)
        row["run_dir"] = str(summary_path.parent)
        rows.append(row)
    return rows


def write_csv(rows, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row.keys()})
    if not fieldnames:
        fieldnames = ["run_dir", "summary_path"]
    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Collect DINO-WM train/plan summary.json files into a CSV."
    )
    parser.add_argument("--root", required=True, help="Root directory to scan.")
    parser.add_argument("--out", required=True, help="Output CSV path.")
    parser.add_argument(
        "--kind",
        choices=["train", "plan"],
        required=True,
        help="Only collect summaries of this run kind.",
    )
    args = parser.parse_args()

    rows = collect_summaries(args.root, args.kind)
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} {args.kind} run summaries to {args.out}")


if __name__ == "__main__":
    main()
