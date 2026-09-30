import json
import logging
import numpy as np
from abc import ABC, abstractmethod
from eval_logging import add_lightweight_defaults, elapsed_since, json_safe, perf_counter


log = logging.getLogger(__name__)


class BasePlanner(ABC):
    def __init__(
        self,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        log_filename,
        metrics_logger=None,
        **kwargs,
    ):
        self.wm = wm
        self.action_dim = action_dim
        self.objective_fn = objective_fn
        self.preprocessor = preprocessor
        self.device = next(wm.parameters()).device

        self.evaluator = evaluator
        self.wandb_run = wandb_run
        self.log_filename = log_filename  # do not log if None
        self.metrics_logger = metrics_logger
        self.last_plan_iterations = None
        self.last_termination_reason = None
        self.last_wm_rollout_time_sec = None
        self.last_candidate_stats = None
        self.last_eval_world_model_rollout_time_sec = 0.0
        self.last_eval_environment_step_time_sec = 0.0
        self.last_eval_decoder_time_sec = 0.0
        self.last_eval_visualization_time_sec = 0.0
        self.last_eval_total_time_sec = 0.0
        self.reset_planner_timing()

    def reset_planner_timing(self):
        self.last_planner_timing_breakdown = {}
        self.last_planner_timing_counts = {}
        self.last_wm_timing_breakdown = {}
        self.last_wm_timing_counts = {}

    def _refresh_objective_context(self, obs_0, obs_g):
        setter = getattr(self.objective_fn, "set_context", None)
        if setter is None:
            return
        start = self._planner_time_start()
        setter(
            wm=self.wm,
            preprocessor=self.preprocessor,
            obs_0=obs_0,
            obs_g=obs_g,
            device=self.device,
        )
        self._record_planner_elapsed("objective_context", start)

    def _add_planner_timing(self, name, elapsed):
        if elapsed is None:
            return
        self.last_planner_timing_breakdown[name] = (
            self.last_planner_timing_breakdown.get(name, 0.0) + float(elapsed)
        )
        self.last_planner_timing_counts[name] = (
            self.last_planner_timing_counts.get(name, 0) + 1
        )

    def _planner_time_start(self):
        return perf_counter()

    def _record_planner_elapsed(self, name, start_time):
        self._add_planner_timing(name, elapsed_since(start_time))

    def _accumulate_wm_timing(self, timing_stats, prefix=None):
        if not timing_stats:
            return
        times = timing_stats.get("times_sec", timing_stats)
        counts = timing_stats.get("counts", {})
        if not isinstance(times, dict):
            return
        for key, value in times.items():
            if not isinstance(value, (int, float)):
                continue
            name = f"{prefix}_{key}" if prefix else key
            self.last_wm_timing_breakdown[name] = (
                self.last_wm_timing_breakdown.get(name, 0.0) + float(value)
            )
            count_value = counts.get(key, 1) if isinstance(counts, dict) else 1
            try:
                count_value = int(count_value)
            except Exception:
                count_value = 1
            self.last_wm_timing_counts[name] = (
                self.last_wm_timing_counts.get(name, 0) + count_value
            )

    def _merge_timing_from_planner(self, planner, prefix=None):
        timing = getattr(planner, "last_planner_timing_breakdown", None) or {}
        counts = getattr(planner, "last_planner_timing_counts", None) or {}
        for key, value in timing.items():
            name = f"{prefix}_{key}" if prefix else key
            self.last_planner_timing_breakdown[name] = (
                self.last_planner_timing_breakdown.get(name, 0.0) + float(value)
            )
            self.last_planner_timing_counts[name] = (
                self.last_planner_timing_counts.get(name, 0) + int(counts.get(key, 1))
            )

        wm_timing = getattr(planner, "last_wm_timing_breakdown", None) or {}
        wm_counts = getattr(planner, "last_wm_timing_counts", None) or {}
        self._accumulate_wm_timing(
            {"times_sec": wm_timing, "counts": wm_counts},
            prefix=prefix,
        )

    def _timing_metrics(self):
        return {
            "planner_timing_breakdown": dict(self.last_planner_timing_breakdown),
            "planner_timing_counts": dict(self.last_planner_timing_counts),
            "wm_timing_breakdown": dict(self.last_wm_timing_breakdown),
            "wm_timing_counts": dict(self.last_wm_timing_counts),
        }

    def reset_eval_timing(self):
        self.last_eval_world_model_rollout_time_sec = 0.0
        self.last_eval_environment_step_time_sec = 0.0
        self.last_eval_decoder_time_sec = 0.0
        self.last_eval_visualization_time_sec = 0.0
        self.last_eval_total_time_sec = 0.0

    def record_evaluator_timing(self):
        eval_info = getattr(self.evaluator, "last_eval_info", None)
        if not eval_info:
            return
        timings = eval_info.get("timings", {})
        self.last_eval_world_model_rollout_time_sec += (
            timings.get("world_model_rollout_time_sec") or 0.0
        )
        self.last_eval_environment_step_time_sec += (
            timings.get("environment_step_time_sec") or 0.0
        )
        self.last_eval_decoder_time_sec += timings.get("decoder_time_sec") or 0.0
        self.last_eval_visualization_time_sec += (
            timings.get("visualization_time_sec") or 0.0
        )
        self.last_eval_total_time_sec += timings.get("eval_total_time_sec") or 0.0
        self._accumulate_wm_timing(
            {
                "times_sec": timings.get("wm_timing_breakdown") or {},
                "counts": timings.get("wm_timing_counts") or {},
            },
            prefix="eval",
        )

    def _current_eval_timing_metrics(self):
        eval_info = getattr(self.evaluator, "last_eval_info", None)
        timings = eval_info.get("timings", {}) if eval_info else {}
        return {
            "eval_total_time_sec": timings.get("eval_total_time_sec"),
            "eval_preprocess_time_sec": timings.get("preprocess_time_sec"),
            "world_model_rollout_time_sec": timings.get(
                "world_model_rollout_time_sec"
            ),
            "environment_step_time_sec": timings.get("environment_step_time_sec"),
            "eval_metric_compute_time_sec": timings.get("metric_compute_time_sec"),
            "decoder_time_sec": timings.get("decoder_time_sec"),
            "visualization_time_sec": timings.get("visualization_time_sec"),
            "eval_wm_timing_breakdown": timings.get("wm_timing_breakdown"),
            "eval_wm_timing_counts": timings.get("wm_timing_counts"),
            "cumulative_eval_total_time_sec": self.last_eval_total_time_sec,
            "cumulative_eval_world_model_rollout_time_sec": (
                self.last_eval_world_model_rollout_time_sec
            ),
            "cumulative_eval_environment_step_time_sec": (
                self.last_eval_environment_step_time_sec
            ),
            "cumulative_eval_decoder_time_sec": self.last_eval_decoder_time_sec,
            "cumulative_eval_visualization_time_sec": (
                self.last_eval_visualization_time_sec
            ),
        }

    def _planner_budget_metrics(self):
        metrics = {
            "planner_iterations": self.last_plan_iterations,
            "termination_reason": self.last_termination_reason,
            "planner_world_model_rollout_time_sec": self.last_wm_rollout_time_sec,
            "action_dim": self.action_dim,
            "horizon": getattr(self, "horizon", None),
            "num_samples": getattr(self, "num_samples", None),
            "topk": getattr(self, "topk", None),
            "opt_steps": getattr(self, "opt_steps", None),
            "eval_every": getattr(self, "eval_every", None),
            "max_iter": getattr(self, "max_iter", None),
            "n_taken_actions": getattr(self, "n_taken_actions", None),
        }
        metrics.update(self._timing_metrics())
        if metrics["num_samples"] is not None and metrics["planner_iterations"] is not None:
            n_evals = None
            try:
                n_evals = self.evaluator.obs_0["visual"].shape[0]
            except Exception:
                pass
            if n_evals is not None:
                metrics["cem_candidate_count"] = (
                    n_evals * metrics["num_samples"] * metrics["planner_iterations"]
                )
                if metrics["topk"] is not None:
                    metrics["cem_elite_count"] = (
                        n_evals * metrics["topk"] * metrics["planner_iterations"]
                    )
                eval_info = getattr(self.evaluator, "last_eval_info", None)
                roi_stats = eval_info.get("roi_stats", {}) if eval_info else {}
                effective_tokens = roi_stats.get("effective_token_count")
                if effective_tokens is not None and metrics["horizon"] is not None:
                    metrics["candidate_token_steps"] = (
                        metrics["cem_candidate_count"]
                        * metrics["horizon"]
                        * effective_tokens
                    )
        return metrics

    def append_metrics_record(self, filename, record, mirror_to_plan_metrics=True):
        if self.metrics_logger is None:
            return
        payload = {
            "planner_class": type(self).__name__,
            "logging_prefix": getattr(self, "logging_prefix", None),
        }
        payload.update(record)
        add_lightweight_defaults(payload)
        self.metrics_logger.append_jsonl(filename, payload)
        if mirror_to_plan_metrics and filename != "plan_metrics.jsonl":
            self.metrics_logger.append_jsonl("plan_metrics.jsonl", payload)

    def _last_eval_record(self):
        eval_info = getattr(self.evaluator, "last_eval_info", None) or {}
        successes = eval_info.get("successes")
        state_distances = eval_info.get("state_distances")
        action_len = eval_info.get("action_len")
        metrics = eval_info.get("metrics", {})
        eval_results = eval_info.get("eval_results", {})
        timings = eval_info.get("timings", {})
        roi_stats = eval_info.get("roi_stats", {})
        task_metrics = {}
        if isinstance(eval_results, dict):
            task_metrics = {
                key: value for key, value in eval_results.items() if key != "success"
            }
        record = {
            "eval_filename": eval_info.get("filename"),
            "eval_metrics": metrics,
            "eval_successes": successes,
            "eval_task_metrics": task_metrics,
            "eval_state_distances": state_distances,
            "eval_action_len": action_len,
            "eval_timings": timings,
        }
        record.update(roi_stats)
        if successes is not None:
            try:
                record["eval_success_rate"] = float(
                    np.mean(np.asarray(successes).astype(float))
                )
                record["eval_num_success"] = int(np.asarray(successes).sum())
            except Exception:
                pass
        if state_distances is not None:
            try:
                record["eval_mean_state_distance"] = float(
                    np.mean(np.asarray(state_distances))
                )
            except Exception:
                pass
        return record

    def _current_wm_roi_record(self):
        getter = getattr(self.wm, "get_last_roi_stats", None)
        if getter is None:
            return {}
        try:
            return getter() or {}
        except Exception:
            return {}

    def dump_logs(self, logs):
        logs_entry = json_safe({
            key: (
                value.item()
                if isinstance(value, (np.float32, np.int32, np.int64))
                else value
            )
            for key, value in logs.items()
        })
        if self.log_filename is not None:
            try:
                with open(self.log_filename, "a") as file:
                    file.write(json.dumps(logs_entry) + "\n")
            except Exception as exc:
                log.warning("Failed to write planner logs to %s: %s", self.log_filename, exc)
        if self.metrics_logger is not None:
            metrics_entry = {
                "event": "planner_eval",
                "planner_class": type(self).__name__,
            }
            metrics_entry.update(logs_entry)
            metrics_entry.update(self._current_eval_timing_metrics())
            metrics_entry.update(self._planner_budget_metrics())
            if self.last_candidate_stats is not None:
                metrics_entry["cem_candidate_stats"] = self.last_candidate_stats
            add_lightweight_defaults(metrics_entry)
            self.metrics_logger.append_jsonl("plan_metrics.jsonl", metrics_entry)

    @abstractmethod
    def plan(self):
        pass
