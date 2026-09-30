import json
import logging
import math
import os
import socket
import sys
import time
from datetime import datetime
from pathlib import Path


log = logging.getLogger(__name__)

METRIC_FLOAT_PRECISION = 3


FUTURE_LIGHTWEIGHT_FIELDS = {
    "drs_enabled": None,
    "drs_mode": None,
    "drs_keep_ratio": None,
    "drs_num_tokens": None,
    "drs_num_kept_tokens": None,
    "drs_mask_time_sec": None,
    "drs_score_mean": None,
    "drs_score_std": None,
    "drs_score_min": None,
    "drs_score_max": None,
    "drs_target_type": None,
    "drs_action_aware": None,
    "drs_group_size": None,
    "drs_num_roi_tokens": None,
    "drs_num_group_tokens": None,
    "drs_grouped_token_count": None,
    "drs_pruned_token_count": None,
    "drs_num_pruned_tokens": None,
    "drs_prune_fill": None,
    "roi_enabled": None,
    "roi_mode": None,
    "roi_keep_ratio": None,
    "roi_num_tokens": None,
    "roi_num_kept_tokens": None,
    "roi_mask_time_sec": None,
    "roi_score_mean": None,
    "roi_score_std": None,
    "roi_score_min": None,
    "roi_score_max": None,
    "roi_target_type": None,
    "roi_action_aware": None,
    "roi_group_size": None,
    "roi_num_roi_tokens": None,
    "roi_num_group_tokens": None,
    "roi_grouped_token_count": None,
    "roi_pruned_token_count": None,
    "roi_num_pruned_tokens": None,
    "roi_prune_fill": None,
    "effective_token_count": None,
    "visual_token_count": None,
    "kept_visual_token_count": None,
    "token_reduction_ratio": None,
    "attention_compute_ratio": None,
    "attention_compute_reduction": None,
    "wm_rollout_token_steps": None,
    "candidate_token_steps": None,
    "eval_preprocess_time_sec": None,
    "eval_metric_compute_time_sec": None,
    "eval_wm_timing_breakdown": None,
    "eval_wm_timing_counts": None,
    "final_eval_wm_timing_breakdown": None,
    "final_eval_wm_timing_counts": None,
    "planner_timing_breakdown": None,
    "planner_timing_counts": None,
    "wm_timing_breakdown": None,
    "wm_timing_counts": None,
    "sub_planner_timing_breakdown": None,
    "sub_planner_timing_counts": None,
    "sub_planner_wm_timing_breakdown": None,
    "sub_planner_wm_timing_counts": None,
    "estimated_flops": None,
    "estimated_predictor_flops": None,
    "corridor_loss": None,
    "corridor_cost": None,
    "corridor_violation": None,
    "eb_cem_enabled": None,
    "eb_cem_size": None,
    "eb_cem_num_entries": None,
    "eb_cem_topm": None,
    "eb_cem_local_count": None,
    "eb_cem_global_count": None,
    "eb_cem_saved_count": None,
    "eb_cem_sample_fraction": None,
    "eb_cem_noise_scale": None,
    "eb_cem_shift_steps": None,
}


def utc_now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def perf_counter():
    return time.perf_counter()


def elapsed_since(start_time):
    return perf_counter() - start_time


def safe_float(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(value):
        return None
    return value


def round_metric_float(value):
    if not math.isfinite(value):
        return None
    return round(value, METRIC_FLOAT_PRECISION)


def json_safe(value):
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return round_metric_float(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "detach"):
        try:
            value = value.detach().cpu()
            if value.numel() == 1:
                return json_safe(value.item())
            return json_safe(value.tolist())
        except Exception:
            return str(value)
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(v) for v in value]
    if hasattr(value, "tolist"):
        try:
            return json_safe(value.tolist())
        except Exception:
            pass
    return str(value)


def add_lightweight_defaults(payload):
    for key, value in FUTURE_LIGHTWEIGHT_FIELDS.items():
        payload.setdefault(key, value)
    return payload


def sync_cuda_if_available():
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        return


def reset_cuda_peak_memory(device=None):
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
    except Exception:
        return


def cuda_peak_memory_mb(device=None):
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated(device) / (1024 * 1024)
    except Exception:
        return None
    return None


class Timer:
    def __init__(self):
        self.elapsed = 0.0

    def __enter__(self):
        sync_cuda_if_available()
        self._start = perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        sync_cuda_if_available()
        self.elapsed = elapsed_since(self._start)
        return False


class SafeMetricsLogger:
    def __init__(self, output_dir="."):
        self.output_dir = Path(output_dir)

    def _safe_write(self, description, writer):
        try:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            writer()
        except Exception as exc:
            log.warning("Metrics logging failed while %s: %s", description, exc)

    def write_json(self, filename, payload):
        def writer():
            path = self.output_dir / filename
            with path.open("w", encoding="utf-8") as f:
                json.dump(json_safe(payload), f, indent=2, sort_keys=True)
                f.write("\n")

        self._safe_write(f"writing {filename}", writer)

    def append_jsonl(self, filename, payload):
        def writer():
            path = self.output_dir / filename
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(json_safe(payload), sort_keys=True) + "\n")

        self._safe_write(f"appending {filename}", writer)


def build_run_meta(kind, output_dir, cfg=None, extra=None):
    meta = {
        "kind": kind,
        "created_at": utc_now_iso(),
        "output_dir": os.path.abspath(str(output_dir)),
        "command": sys.argv,
        "hostname": "<host>",
        "pid": os.getpid(),
    }
    if cfg is not None:
        meta["config"] = cfg
    if extra:
        meta.update(extra)
    add_lightweight_defaults(meta)
    return meta


def count_parameters(module):
    if module is None:
        return {"total_params": 0, "trainable_params": 0}
    total = 0
    trainable = 0
    try:
        for param in module.parameters():
            n_param = param.numel()
            total += n_param
            if param.requires_grad:
                trainable += n_param
    except Exception:
        return {"total_params": None, "trainable_params": None}
    return {"total_params": total, "trainable_params": trainable}


def count_named_parameters(named_modules):
    counts = {}
    total = 0
    trainable = 0
    for name, module in named_modules.items():
        module_counts = count_parameters(module)
        counts[name] = module_counts
        if module_counts["total_params"] is not None:
            total += module_counts["total_params"]
        if module_counts["trainable_params"] is not None:
            trainable += module_counts["trainable_params"]
    counts["total_params"] = total
    counts["trainable_params"] = trainable
    return counts


def mean_or_none(values):
    cleaned = [safe_float(v) for v in values]
    cleaned = [v for v in cleaned if v is not None]
    if not cleaned:
        return None
    return sum(cleaned) / len(cleaned)


def min_or_none(values):
    cleaned = [safe_float(v) for v in values]
    cleaned = [v for v in cleaned if v is not None]
    if not cleaned:
        return None
    return min(cleaned)


def max_or_none(values):
    cleaned = [safe_float(v) for v in values]
    cleaned = [v for v in cleaned if v is not None]
    if not cleaned:
        return None
    return max(cleaned)
