import argparse
import csv
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from plan import load_model  # noqa: E402


def _parse_hydra_overrides(items: List[str]) -> Dict[str, Any]:
    dotlist = []
    skip_next = False
    for item in items:
        if skip_next:
            skip_next = False
            continue
        if item.startswith("--"):
            skip_next = "=" not in item
            continue
        if "=" not in item:
            continue
        dotlist.append(item[1:] if item.startswith("+") else item)
    if not dotlist:
        return {}
    cfg = OmegaConf.from_dotlist(dotlist)
    return OmegaConf.to_container(cfg, resolve=True)


def _deep_update(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _json_safe(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


def _checkpoint_path(model_path: Path, model_epoch: str) -> Path:
    name = "model_latest.pth" if str(model_epoch) == "latest" else f"model_{model_epoch}.pth"
    return model_path / "checkpoints" / name


def _runtime_dense(runtime_cfg: Dict[str, Any]) -> Dict[str, Any]:
    dense = deepcopy(runtime_cfg)
    dense["load_decoder"] = False
    dense["use_drs"] = False
    dense["drs_mode"] = "none"
    dense["use_roi"] = False
    dense["roi_mode"] = "none"
    sparse = deepcopy(dense.get("sparse_dynamics", {}))
    sparse["enabled"] = False
    dense["sparse_dynamics"] = sparse
    return dense


def _make_dummy_z(model, batch_size: int, num_hist: int, num_tokens: int, dtype) -> torch.Tensor:
    device = next(model.parameters()).device
    return torch.randn(
        int(batch_size),
        int(num_hist),
        int(num_tokens),
        int(model.emb_dim),
        device=device,
        dtype=dtype,
    )


def _cuda_sync(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _profile_flops(model, z: torch.Tensor) -> int:
    activities = [torch.profiler.ProfilerActivity.CPU]
    if z.device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.no_grad():
        with torch.profiler.profile(
            activities=activities,
            with_flops=True,
            record_shapes=False,
            profile_memory=False,
        ) as prof:
            _ = model.predict(z)
            _cuda_sync(z.device)
    total = 0
    for event in prof.key_averages():
        flops = getattr(event, "flops", 0) or 0
        total += int(flops)
    return total


def _measure_throughput(model, z: torch.Tensor, warmup: int, iters: int) -> Dict[str, float]:
    with torch.no_grad():
        for _ in range(max(0, int(warmup))):
            _ = model.predict(z)
        _cuda_sync(z.device)
        if z.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(z.device)
        start = time.perf_counter()
        for _ in range(max(1, int(iters))):
            _ = model.predict(z)
        _cuda_sync(z.device)
        elapsed = time.perf_counter() - start
    num_iters = max(1, int(iters))
    batch_size = int(z.shape[0])
    peak_memory_mb = None
    if z.device.type == "cuda":
        peak_memory_mb = torch.cuda.max_memory_allocated(z.device) / (1024**2)
    return {
        "elapsed_sec": float(elapsed),
        "latency_ms_per_batch": float(elapsed * 1000.0 / num_iters),
        "throughput_samples_per_sec": float(batch_size * num_iters / elapsed),
        "peak_memory_mb": None if peak_memory_mb is None else float(peak_memory_mb),
    }


def _first_number(*values, default=None):
    for value in values:
        if isinstance(value, (int, float)):
            return float(value)
    return default


def _collect_token_stats(model, z: torch.Tensor, input_num_tokens: int, num_hist: int) -> Dict[str, Any]:
    reset = getattr(model, "reset_roi_stats", None)
    if reset is not None:
        reset()
    with torch.no_grad():
        _ = model.predict(z)
        _cuda_sync(z.device)

    getter = getattr(model, "get_last_roi_stats", None)
    stats = getter() if getter is not None else {}
    stats = stats or {}
    effective_tokens = _first_number(
        stats.get("effective_token_count"),
        stats.get("sparse_num_foreground_tokens"),
        stats.get("drs_num_kept_tokens"),
        stats.get("roi_num_kept_tokens"),
        default=float(input_num_tokens),
    )
    foreground_tokens = _first_number(
        stats.get("sparse_num_foreground_tokens"),
        stats.get("drs_num_kept_tokens"),
        stats.get("roi_num_kept_tokens"),
        default=effective_tokens,
    )
    background_tokens = _first_number(
        stats.get("sparse_num_background_tokens"),
        default=max(0.0, float(input_num_tokens) - float(foreground_tokens)),
    )
    return {
        "input_num_tokens": int(input_num_tokens),
        "input_sequence_tokens": int(input_num_tokens) * int(num_hist),
        "effective_predictor_tokens": float(effective_tokens),
        "effective_predictor_sequence_tokens": float(effective_tokens) * int(num_hist),
        "foreground_tokens": float(foreground_tokens),
        "background_tokens": float(background_tokens),
        "token_keep_ratio": float(effective_tokens) / max(1.0, float(input_num_tokens)),
        "token_reduction_ratio": float(input_num_tokens) / max(1.0, float(effective_tokens)),
        "sparse_background_processor": stats.get("sparse_background_processor"),
        "sparse_mask_source": stats.get("sparse_mask_source"),
        "drs_mode": stats.get("drs_mode"),
        "drs_num_kept_tokens": stats.get("drs_num_kept_tokens"),
    }


def _profile_variant(
    label: str,
    model_ckpt: Path,
    train_cfg,
    runtime_cfg: Dict[str, Any],
    device: torch.device,
    batch_size: int,
    num_hist: int,
    num_tokens: int,
    warmup: int,
    iters: int,
    dtype,
) -> Dict[str, Any]:
    model = load_model(
        model_ckpt,
        train_cfg,
        train_cfg.num_action_repeat,
        device=device,
        roi_runtime_cfg=runtime_cfg,
    )
    model.eval()
    z = _make_dummy_z(model, batch_size, num_hist, num_tokens, dtype=dtype)
    _cuda_sync(device)
    token_stats = _collect_token_stats(
        model,
        z,
        input_num_tokens=num_tokens,
        num_hist=num_hist,
    )
    flops_total = _profile_flops(model, z)
    timing = _measure_throughput(model, z, warmup=warmup, iters=iters)
    per_sample = flops_total / max(1, int(batch_size))
    record = {
        "label": label,
        "batch_size": int(batch_size),
        "num_hist": int(num_hist),
        "num_tokens": int(num_tokens),
        "emb_dim": int(model.emb_dim),
        "flops_per_batch": int(flops_total),
        "flops_per_sample": float(per_sample),
        "gflops_per_sample": float(per_sample / 1e9),
    }
    record.update(token_stats)
    record.update(timing)
    del model, z
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return record


def build_argparser():
    parser = argparse.ArgumentParser(
        description=(
            "Profile one-step world-model prediction FLOPs and throughput. "
            "Unknown key=value arguments are treated like Hydra runtime overrides."
        )
    )
    parser.add_argument("--ckpt-base-path", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-epoch", default="latest")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-hist", type=int, default=None)
    parser.add_argument("--num-tokens", type=int, default=196)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    return parser


def main():
    parser = build_argparser()
    args, unknown = parser.parse_known_args()
    overrides = _parse_hydra_overrides(unknown)

    model_path = Path(args.ckpt_base_path) / "outputs" / args.model_name
    train_cfg = OmegaConf.load(model_path / "hydra.yaml")
    base_runtime = OmegaConf.to_container(train_cfg, resolve=True)
    runtime_cfg = _deep_update(deepcopy(base_runtime), overrides)
    runtime_cfg["load_decoder"] = False
    runtime_cfg["profile_wm_timing"] = False
    runtime_cfg["profile_wm_timing_sync_cuda"] = False

    model_ckpt = _checkpoint_path(model_path, args.model_epoch)
    device = torch.device(args.device)
    dtype = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    num_hist = int(args.num_hist or train_cfg.num_hist)

    records = []
    if not args.skip_dense:
        records.append(
            _profile_variant(
                "DINO-WM dense",
                model_ckpt,
                train_cfg,
                _runtime_dense(runtime_cfg),
                device,
                args.batch_size,
                num_hist,
                args.num_tokens,
                args.warmup,
                args.iters,
                dtype,
            )
        )
    records.append(
        _profile_variant(
            "Ours runtime",
            model_ckpt,
            train_cfg,
            runtime_cfg,
            device,
            args.batch_size,
            num_hist,
            args.num_tokens,
            args.warmup,
            args.iters,
            dtype,
        )
    )

    if len(records) >= 2:
        dense, ours = records[0], records[-1]
        if ours["flops_per_sample"] > 0:
            ours["flops_reduction_vs_dense"] = float(
                dense["flops_per_sample"] / ours["flops_per_sample"]
            )
        if dense["throughput_samples_per_sec"] > 0:
            ours["throughput_speedup_vs_dense"] = float(
                ours["throughput_samples_per_sec"]
                / dense["throughput_samples_per_sec"]
            )

    payload = {
        "model_name": args.model_name,
        "model_epoch": args.model_epoch,
        "checkpoint": str(model_ckpt),
        "profile_kind": "single_step_wm_predict",
        "records": records,
    }
    print(json.dumps(_json_safe(payload), indent=2))

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(_json_safe(payload), indent=2) + "\n")
    if args.output_csv:
        out = Path(args.output_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        keys = sorted({key for record in records for key in record.keys()})
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(records)


if __name__ == "__main__":
    main()
