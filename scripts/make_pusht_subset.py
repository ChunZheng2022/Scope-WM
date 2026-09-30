import argparse
import json
import os
import pickle
import random
import shutil
import sys
from pathlib import Path
from typing import Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch


TENSOR_FILES = [
    "states.pth",
    "rel_actions.pth",
    "abs_actions.pth",
    "velocities.pth",
]

PKL_FILES = [
    "seq_lengths.pkl",
    "shapes.pkl",
]


def _load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def _save_pickle(obj, path: Path) -> None:
    with path.open("wb") as f:
        pickle.dump(obj, f)


def _safe_output_path(source: Path, output: Path, overwrite: bool) -> None:
    source = source.resolve()
    output = output.resolve()
    if output == source:
        raise ValueError("Output path must be different from source path.")
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output path already exists: {output}. Use --overwrite to replace it.")
        shutil.rmtree(output)


def _episode_count(split_dir: Path) -> int:
    seq_path = split_dir / "seq_lengths.pkl"
    if not seq_path.exists():
        raise FileNotFoundError(f"Missing {seq_path}")
    return len(_load_pickle(seq_path))


def _select_indices(num_episodes: int, ratio: float, seed: int, explicit_count: int = None) -> List[int]:
    if explicit_count is None:
        count = max(1, int(round(num_episodes * ratio)))
    else:
        count = int(explicit_count)
    count = min(max(count, 1), num_episodes)
    rng = random.Random(seed)
    return sorted(rng.sample(range(num_episodes), count))


def _subset_tensor_file(src_path: Path, dst_path: Path, indices: List[int]) -> None:
    if not src_path.exists():
        return
    tensor = torch.load(src_path, map_location="cpu")
    index_tensor = torch.tensor(indices, dtype=torch.long)
    subset = tensor.index_select(0, index_tensor)
    torch.save(subset, dst_path)


def _subset_pickle_file(src_path: Path, dst_path: Path, indices: List[int]) -> None:
    if not src_path.exists():
        return
    values = _load_pickle(src_path)
    subset = [values[i] for i in indices]
    _save_pickle(subset, dst_path)


def _link_or_copy(src: Path, dst: Path, mode: str, fallback_copy: bool = True) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        if mode == "copy":
            shutil.copy2(src, dst)
        elif mode == "hardlink":
            os.link(src, dst)
        elif mode == "symlink":
            os.symlink(src, dst)
        else:
            raise ValueError(f"Unsupported link mode: {mode}")
    except OSError:
        if not fallback_copy or mode == "copy":
            raise
        shutil.copy2(src, dst)


def _copy_episode_videos(
    src_split: Path,
    dst_split: Path,
    indices: List[int],
    link_mode: str,
    fallback_copy: bool,
) -> List[Dict]:
    src_obs = src_split / "obses"
    dst_obs = dst_split / "obses"
    if not src_obs.exists():
        raise FileNotFoundError(f"Missing PushT video directory: {src_obs}")
    dst_obs.mkdir(parents=True, exist_ok=True)

    records = []
    for new_idx, old_idx in enumerate(indices):
        src = src_obs / f"episode_{old_idx:03d}.mp4"
        if not src.exists():
            raise FileNotFoundError(f"Missing episode video: {src}")
        dst = dst_obs / f"episode_{new_idx:03d}.mp4"
        _link_or_copy(src, dst, link_mode, fallback_copy=fallback_copy)
        records.append(
            {
                "new_episode": new_idx,
                "source_episode": old_idx,
                "source_video": str(src),
                "target_video": str(dst),
            }
        )
    return records


def _copy_extra_root_files(src_split: Path, dst_split: Path, known_names) -> None:
    known = set(known_names) | {"obses"}
    for item in src_split.iterdir():
        if item.name in known or item.is_dir():
            continue
        dst = dst_split / item.name
        shutil.copy2(item, dst)


def _make_split_subset(
    source: Path,
    output: Path,
    split: str,
    ratio: float,
    seed: int,
    link_mode: str,
    fallback_copy: bool,
    explicit_count: int = None,
    copy_extra_files: bool = True,
) -> Dict:
    src_split = source / split
    dst_split = output / split
    if not src_split.exists():
        raise FileNotFoundError(f"Missing split directory: {src_split}")
    dst_split.mkdir(parents=True, exist_ok=True)

    num_episodes = _episode_count(src_split)
    indices = _select_indices(num_episodes, ratio, seed, explicit_count=explicit_count)

    for name in TENSOR_FILES:
        _subset_tensor_file(src_split / name, dst_split / name, indices)
    for name in PKL_FILES:
        _subset_pickle_file(src_split / name, dst_split / name, indices)
    video_records = _copy_episode_videos(
        src_split,
        dst_split,
        indices,
        link_mode=link_mode,
        fallback_copy=fallback_copy,
    )
    if copy_extra_files:
        _copy_extra_root_files(src_split, dst_split, known_names=TENSOR_FILES + PKL_FILES)

    metadata = {
        "split": split,
        "source_split": str(src_split),
        "output_split": str(dst_split),
        "ratio": ratio,
        "seed": seed,
        "source_num_episodes": num_episodes,
        "subset_num_episodes": len(indices),
        "selected_source_indices": indices,
        "video_records": video_records,
    }
    with (dst_split / "subset_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    return metadata


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Create a trajectory-level PushT subset without breaking temporal continuity."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="Source PushT dataset directory, e.g. <DATA_ROOT>/pusht_noise",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output dataset directory. Defaults to a sibling named <source>_subset_1over8.",
    )
    parser.add_argument("--ratio", type=float, default=0.125)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--train-count", type=int, default=None)
    parser.add_argument("--val-count", type=int, default=None)
    parser.add_argument("--splits", default="train,val")
    parser.add_argument(
        "--link-mode",
        default="symlink",
        choices=["symlink", "hardlink", "copy"],
        help="How to materialize episode mp4 files. symlink is fastest and saves disk.",
    )
    parser.add_argument(
        "--fallback-copy",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If symlink/hardlink fails, copy videos instead.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--copy-extra-files",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Copy non-standard root files in each split directory.",
    )
    return parser


def main():
    args = build_argparser().parse_args()
    source = Path(args.source).expanduser()
    if args.output is None:
        output = source.parent / f"{source.name}_subset_1over8"
    else:
        output = Path(args.output).expanduser()
    _safe_output_path(source, output, overwrite=args.overwrite)
    output.mkdir(parents=True, exist_ok=True)

    split_names = [item.strip() for item in args.splits.split(",") if item.strip()]
    all_metadata = {
        "source": str(source),
        "output": str(output),
        "ratio": args.ratio,
        "seed": args.seed,
        "link_mode": args.link_mode,
        "splits": {},
    }
    for split_offset, split in enumerate(split_names):
        explicit_count = args.train_count if split == "train" else args.val_count if split in {"val", "valid"} else None
        metadata = _make_split_subset(
            source=source,
            output=output,
            split=split,
            ratio=args.ratio,
            seed=args.seed + split_offset,
            link_mode=args.link_mode,
            fallback_copy=args.fallback_copy,
            explicit_count=explicit_count,
            copy_extra_files=args.copy_extra_files,
        )
        all_metadata["splits"][split] = metadata
        print(
            f"{split}: selected {metadata['subset_num_episodes']}/"
            f"{metadata['source_num_episodes']} episodes -> {output / split}"
        )

    with (output / "subset_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
    print(f"Saved subset dataset to {output}")


if __name__ == "__main__":
    main()
