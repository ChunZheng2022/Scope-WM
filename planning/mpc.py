import torch
import hydra
import copy
import numpy as np
from einops import rearrange, repeat
from utils import slice_trajdict_with_t
from .base_planner import BasePlanner


class MPCPlanner(BasePlanner):
    """
    an online planner so feedback from env is allowed
    """

    def __init__(
        self,
        max_iter,
        n_taken_actions,
        sub_planner,
        wm,
        env,  # for online exec
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="mpc",
        log_filename="logs.json",
        metrics_logger=None,
        plan_active_only=False,
        force_full_mpc_iters=False,
        adaptive_cem_samples=None,
        frontloaded_cem_samples=None,
        **kwargs,
    ):
        super().__init__(
            wm,
            action_dim,
            objective_fn,
            preprocessor,
            evaluator,
            wandb_run,
            log_filename,
            metrics_logger=metrics_logger,
        )
        self.env = env
        self.max_iter = np.inf if max_iter is None else max_iter
        self.n_taken_actions = n_taken_actions
        self.logging_prefix = logging_prefix
        sub_planner["_target_"] = sub_planner["target"]
        self.sub_planner = hydra.utils.instantiate(
            sub_planner,
            wm=self.wm,
            action_dim=self.action_dim,
            objective_fn=self.objective_fn,
            preprocessor=self.preprocessor,
            evaluator=self.evaluator,  # evaluator is shared for mpc and sub_planner
            wandb_run=self.wandb_run,
            log_filename=None,
            metrics_logger=self.metrics_logger,
        )
        self.is_success = None
        self.action_len = None  # keep track of the step each traj reaches success
        self.iter = 0
        self.planned_actions = []
        self.plan_active_only = bool(plan_active_only)
        self.force_full_mpc_iters = bool(force_full_mpc_iters)
        self.adaptive_cem_samples = self._normalize_adaptive_cem_config(
            adaptive_cem_samples
        )
        self.frontloaded_cem_samples = self._normalize_frontloaded_cem_config(
            frontloaded_cem_samples
        )
        self._base_cem_num_samples = int(getattr(self.sub_planner, "num_samples", 0))
        self._base_cem_topk = int(getattr(self.sub_planner, "topk", 0))
        self._adaptive_no_new_success_iters = 0
        self._adaptive_boost_remaining = 0

    @staticmethod
    def _cfg_get(cfg, key, default=None):
        if cfg is None:
            return default
        if isinstance(cfg, dict):
            return cfg.get(key, default)
        return getattr(cfg, key, default)

    def _normalize_adaptive_cem_config(self, cfg):
        return {
            "enabled": bool(self._cfg_get(cfg, "enabled", False)),
            "base_num_samples": self._cfg_get(cfg, "base_num_samples", None),
            "base_topk": self._cfg_get(cfg, "base_topk", None),
            "boost_num_samples": self._cfg_get(cfg, "boost_num_samples", None),
            "boost_topk": self._cfg_get(cfg, "boost_topk", None),
            "failure_patience": int(self._cfg_get(cfg, "failure_patience", 2) or 0),
            "boost_duration": max(1, int(self._cfg_get(cfg, "boost_duration", 1) or 1)),
            "min_mpc_iter": int(self._cfg_get(cfg, "min_mpc_iter", 0) or 0),
        }

    def _normalize_frontloaded_cem_config(self, cfg):
        return {
            "enabled": bool(self._cfg_get(cfg, "enabled", False)),
            "frontload_iters": max(
                1,
                int(self._cfg_get(cfg, "frontload_iters", 1) or 1),
            ),
            "front_num_samples": self._cfg_get(cfg, "front_num_samples", None),
            "front_topk": self._cfg_get(cfg, "front_topk", None),
            "later_num_samples": self._cfg_get(cfg, "later_num_samples", None),
            "later_topk": self._cfg_get(cfg, "later_topk", None),
        }

    @staticmethod
    def _slice_batch(value, indices):
        if isinstance(value, torch.Tensor):
            return value[indices]
        return np.asarray(value)[indices]

    def _slice_trajdict_batch(self, dct, indices):
        return {key: self._slice_batch(value, indices) for key, value in dct.items()}

    @staticmethod
    def _slice_seed(seed_value, indices):
        idx = np.asarray(indices, dtype=int).tolist()
        if isinstance(seed_value, list):
            return [seed_value[i] for i in idx]
        if isinstance(seed_value, tuple):
            return [seed_value[i] for i in idx]
        arr = np.asarray(seed_value)
        if arr.ndim == 0:
            return [int(arr.item()) for _ in idx]
        return arr[idx].tolist()

    def _evaluator_snapshot(self):
        return {
            "obs_0": self.evaluator.obs_0,
            "obs_g": self.evaluator.obs_g,
            "state_0": self.evaluator.state_0,
            "state_g": self.evaluator.state_g,
            "seed": self.evaluator.seed,
        }

    def _restore_evaluator_snapshot(self, snapshot):
        self.evaluator.obs_0 = snapshot["obs_0"]
        self.evaluator.obs_g = snapshot["obs_g"]
        self.evaluator.state_0 = snapshot["state_0"]
        self.evaluator.state_g = snapshot["state_g"]
        self.evaluator.seed = snapshot["seed"]

    def _assign_evaluator_subset(self, obs_0, obs_g, state_0, state_g, seed):
        self.evaluator.obs_0 = obs_0
        self.evaluator.obs_g = obs_g
        self.evaluator.state_0 = state_0
        self.evaluator.state_g = state_g
        self.evaluator.seed = seed

    def _select_cem_budget(self):
        front_cfg = self.frontloaded_cem_samples
        if front_cfg["enabled"]:
            if self.iter < front_cfg["frontload_iters"]:
                num_samples = int(
                    front_cfg["front_num_samples"] or self._base_cem_num_samples
                )
                topk = int(front_cfg["front_topk"] or self._base_cem_topk)
            else:
                num_samples = int(
                    front_cfg["later_num_samples"] or self._base_cem_num_samples
                )
                topk = int(front_cfg["later_topk"] or self._base_cem_topk)
            return num_samples, min(topk, num_samples), False

        cfg = self.adaptive_cem_samples
        if not cfg["enabled"]:
            return self._base_cem_num_samples, self._base_cem_topk, False

        base_num_samples = int(cfg["base_num_samples"] or self._base_cem_num_samples)
        base_topk = int(cfg["base_topk"] or self._base_cem_topk)
        boost_num_samples = int(cfg["boost_num_samples"] or base_num_samples)
        boost_topk = int(cfg["boost_topk"] or max(1, boost_num_samples // 10))

        should_boost = False
        if self._adaptive_boost_remaining > 0:
            should_boost = True
            self._adaptive_boost_remaining -= 1
        elif (
            self.iter >= cfg["min_mpc_iter"]
            and self._adaptive_no_new_success_iters >= cfg["failure_patience"]
        ):
            should_boost = True
            self._adaptive_boost_remaining = cfg["boost_duration"] - 1

        if should_boost:
            return boost_num_samples, min(boost_topk, boost_num_samples), True
        return base_num_samples, min(base_topk, base_num_samples), False

    def _set_cem_budget(self, num_samples, topk):
        self.sub_planner.num_samples = int(num_samples)
        self.sub_planner.topk = int(max(1, min(topk, num_samples)))

    def _apply_success_mask(self, actions):
        device = actions.device
        mask = torch.tensor(self.is_success).bool()
        actions[mask] = 0
        masked_actions = rearrange(
            actions[mask], "... (f d) -> ... f d", f=self.evaluator.frameskip
        )
        masked_actions = self.preprocessor.normalize_actions(masked_actions.cpu())
        masked_actions = rearrange(masked_actions, "... f d -> ... (f d)")
        actions[mask] = masked_actions.to(device)
        return actions

    def plan(self, obs_0, obs_g, actions=None):
        """
        actions is NOT used
        Returns:
            actions: (B, T, action_dim) torch.Tensor
        """
        n_evals = obs_0["visual"].shape[0]
        self.reset_planner_timing()
        self.is_success = np.zeros(n_evals, dtype=bool)
        self.action_len = np.full(n_evals, np.inf)
        init_obs_0, init_state_0 = self.evaluator.get_init_cond()
        self.last_plan_iterations = 0
        self.last_termination_reason = "max_mpc_iter"
        self.last_wm_rollout_time_sec = 0.0
        self.last_candidate_stats = []
        self.reset_eval_timing()

        cur_obs_0 = obs_0
        memo_actions = None
        if self.force_full_mpc_iters and not np.isfinite(self.max_iter):
            raise ValueError(
                "planner.force_full_mpc_iters=true requires planner.max_iter to be finite."
            )

        while (self.force_full_mpc_iters or not np.all(self.is_success)) and self.iter < self.max_iter:
            mpc_iter_start = self._planner_time_start()
            self.sub_planner.logging_prefix = f"plan_{self.iter}"
            cem_num_samples, cem_topk, adaptive_boost_active = self._select_cem_budget()
            self._set_cem_budget(cem_num_samples, cem_topk)
            active_indices = np.where(~self.is_success)[0]
            use_active_subset = (
                self.plan_active_only
                and not self.force_full_mpc_iters
                and len(active_indices) < n_evals
            )

            plan_obs_0 = cur_obs_0
            plan_obs_g = obs_g
            plan_actions = memo_actions
            evaluator_snapshot = None
            if use_active_subset:
                plan_obs_0 = self._slice_trajdict_batch(cur_obs_0, active_indices)
                plan_obs_g = self._slice_trajdict_batch(obs_g, active_indices)
                plan_actions = (
                    memo_actions[active_indices] if memo_actions is not None else None
                )
                evaluator_snapshot = self._evaluator_snapshot()
                self._assign_evaluator_subset(
                    obs_0=plan_obs_0,
                    obs_g=plan_obs_g,
                    state_0=self._slice_batch(self.evaluator.state_0, active_indices),
                    state_g=self._slice_batch(self.evaluator.state_g, active_indices),
                    seed=self._slice_seed(self.evaluator.seed, active_indices),
                )

            start = self._planner_time_start()
            sub_planner_evaluator = getattr(self.sub_planner, "evaluator", None)
            set_active_eval_indices = getattr(
                self.sub_planner,
                "set_active_eval_indices",
                None,
            )
            if set_active_eval_indices is not None:
                if use_active_subset:
                    set_active_eval_indices(active_indices)
                else:
                    set_active_eval_indices(np.arange(n_evals))
            if use_active_subset:
                # CEM's intermediate evaluator calls are for logging/debugging only.
                # With active-only planning the batch is a subset of the vector env,
                # so evaluating here would mismatch action count and env worker ids.
                self.sub_planner.evaluator = None
            try:
                planned_subset_actions, _ = self.sub_planner.plan(
                    obs_0=plan_obs_0,
                    obs_g=plan_obs_g,
                    actions=plan_actions,
                )  # (b, t, act_dim)
            finally:
                self.sub_planner.evaluator = sub_planner_evaluator
                if evaluator_snapshot is not None:
                    self._restore_evaluator_snapshot(evaluator_snapshot)
            if use_active_subset:
                actions = torch.zeros(
                    n_evals,
                    planned_subset_actions.shape[1],
                    planned_subset_actions.shape[2],
                    dtype=planned_subset_actions.dtype,
                    device=planned_subset_actions.device,
                )
                actions[active_indices] = planned_subset_actions
            else:
                actions = planned_subset_actions
            self._record_planner_elapsed("sub_planner_plan", start)
            print(
                f"{type(self.sub_planner).__name__} inner iterations: "
                f"{getattr(self.sub_planner, 'last_plan_iterations', None)} "
                f"termination={getattr(self.sub_planner, 'last_termination_reason', None)}"
            )
            self._merge_timing_from_planner(self.sub_planner, prefix="sub_planner")
            self.last_wm_rollout_time_sec += (
                getattr(self.sub_planner, "last_wm_rollout_time_sec", 0.0) or 0.0
            )
            self.last_eval_world_model_rollout_time_sec += (
                getattr(
                    self.sub_planner,
                    "last_eval_world_model_rollout_time_sec",
                    0.0,
                )
                or 0.0
            )
            self.last_eval_environment_step_time_sec += (
                getattr(
                    self.sub_planner,
                    "last_eval_environment_step_time_sec",
                    0.0,
                )
                or 0.0
            )
            self.last_eval_decoder_time_sec += (
                getattr(self.sub_planner, "last_eval_decoder_time_sec", 0.0) or 0.0
            )
            self.last_eval_visualization_time_sec += (
                getattr(
                    self.sub_planner,
                    "last_eval_visualization_time_sec",
                    0.0,
                )
                or 0.0
            )
            self.last_eval_total_time_sec += (
                getattr(self.sub_planner, "last_eval_total_time_sec", 0.0) or 0.0
            )
            sub_stats = getattr(self.sub_planner, "last_candidate_stats", None)
            if sub_stats:
                self.last_candidate_stats.append(
                    {"mpc_iter": self.iter, "sub_planner_stats": sub_stats}
                )
            start = self._planner_time_start()
            taken_actions = actions.detach()[:, : self.n_taken_actions]
            self._apply_success_mask(taken_actions)
            memo_actions = actions.detach()[:, self.n_taken_actions :]
            self.planned_actions.append(taken_actions)
            self._record_planner_elapsed("action_bookkeeping", start)

            print(f"MPC iter {self.iter} Eval ------- ")
            action_so_far = torch.cat(self.planned_actions, dim=1)
            self.evaluator.assign_init_cond(
                obs_0=init_obs_0,
                state_0=init_state_0,
            )
            start = self._planner_time_start()
            logs, successes, e_obses, e_states = self.evaluator.eval_actions(
                action_so_far,
                self.action_len,
                filename=f"plan{self.iter}",
                save_video=True,
            )
            self._record_planner_elapsed("eval_actions", start)
            self.record_evaluator_timing()
            previous_successes = self.is_success.copy()
            new_successes = successes & ~previous_successes  # Identify new successes
            self.is_success = (
                previous_successes | successes
            )  # Update overall success status
            new_success_count = int(np.asarray(new_successes).sum())
            if new_success_count > 0:
                self._adaptive_no_new_success_iters = 0
            else:
                self._adaptive_no_new_success_iters += 1
            self.action_len[new_successes] = (
                (self.iter + 1) * self.n_taken_actions
            )  # Update only for the newly successful trajectories

            print("self.is_success: ", self.is_success)
            self._record_planner_elapsed("mpc_iteration_total", mpc_iter_start)
            sub_planner_timing = getattr(
                self.sub_planner, "last_planner_timing_breakdown", {}
            )
            sub_planner_timing_counts = getattr(
                self.sub_planner, "last_planner_timing_counts", {}
            )
            sub_wm_timing = getattr(self.sub_planner, "last_wm_timing_breakdown", {})
            sub_wm_timing_counts = getattr(self.sub_planner, "last_wm_timing_counts", {})
            self.append_metrics_record(
                "mpc_iterations.jsonl",
                {
                    "kind": "plan",
                    "event": "mpc_iteration",
                    "mpc_iter": self.iter,
                    "mpc_step": self.iter + 1,
                    "n_evals": n_evals,
                    "active_eval_count": int(len(active_indices)),
                    "active_eval_indices": active_indices,
                    "plan_active_only": bool(use_active_subset),
                    "force_full_mpc_iters": bool(self.force_full_mpc_iters),
                    "adaptive_cem_samples_enabled": bool(
                        self.adaptive_cem_samples["enabled"]
                    ),
                    "frontloaded_cem_samples_enabled": bool(
                        self.frontloaded_cem_samples["enabled"]
                    ),
                    "adaptive_cem_boost_active": bool(adaptive_boost_active),
                    "adaptive_cem_num_samples": int(cem_num_samples),
                    "adaptive_cem_topk": int(cem_topk),
                    "adaptive_no_new_success_iters": int(
                        self._adaptive_no_new_success_iters
                    ),
                    "max_iter": self.max_iter,
                    "n_taken_actions": self.n_taken_actions,
                    "action_horizon_so_far": action_so_far.shape[1],
                    "iteration_success_rate": float(
                        np.mean(np.asarray(successes).astype(float))
                    ),
                    "iteration_num_success": int(np.asarray(successes).sum()),
                    "cumulative_success_rate": float(
                        np.mean(np.asarray(self.is_success).astype(float))
                    ),
                    "cumulative_num_success": int(np.asarray(self.is_success).sum()),
                    "new_success_count": new_success_count,
                    "previous_successes": previous_successes,
                    "iteration_successes": successes,
                    "new_successes": new_successes,
                    "cumulative_successes": self.is_success,
                    "action_len": self.action_len,
                    "sub_planner_class": type(self.sub_planner).__name__,
                    "sub_planner_iterations": getattr(
                        self.sub_planner, "last_plan_iterations", None
                    ),
                    "sub_planner_termination_reason": getattr(
                        self.sub_planner, "last_termination_reason", None
                    ),
                    "sub_planner_candidate_stats": sub_stats,
                    "sub_planner_timing_breakdown": sub_planner_timing,
                    "sub_planner_timing_counts": sub_planner_timing_counts,
                    "sub_planner_wm_timing_breakdown": sub_wm_timing,
                    "sub_planner_wm_timing_counts": sub_wm_timing_counts,
                    "planner_world_model_rollout_time_sec": (
                        self.last_wm_rollout_time_sec
                    ),
                    "planner_eval_world_model_rollout_time_sec": (
                        self.last_eval_world_model_rollout_time_sec
                    ),
                    "planner_eval_environment_step_time_sec": (
                        self.last_eval_environment_step_time_sec
                    ),
                    "planner_eval_decoder_time_sec": self.last_eval_decoder_time_sec,
                    "planner_eval_visualization_time_sec": (
                        self.last_eval_visualization_time_sec
                    ),
                    "planner_eval_total_time_sec": self.last_eval_total_time_sec,
                    **self._timing_metrics(),
                    **self._last_eval_record(),
                },
            )
            logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
            logs.update({"step": self.iter + 1})
            self.wandb_run.log(logs)
            self.dump_logs(logs)

            # update evaluator's init conditions with new env feedback
            e_final_obs = slice_trajdict_with_t(e_obses, start_idx=-1)
            cur_obs_0 = e_final_obs
            e_final_state = e_states[:, -1]
            self.evaluator.assign_init_cond(
                obs_0=e_final_obs,
                state_0=e_final_state,
            )
            self.iter += 1
            self.last_plan_iterations = self.iter
            self.sub_planner.logging_prefix = f"plan_{self.iter}"

        if self.force_full_mpc_iters and self.iter >= self.max_iter:
            self.last_termination_reason = "max_mpc_iter_forced"
        elif np.all(self.is_success):
            self.last_termination_reason = "all_success"

        planned_actions = torch.cat(self.planned_actions, dim=1)
        self.evaluator.assign_init_cond(
            obs_0=init_obs_0,
            state_0=init_state_0,
        )

        return planned_actions, self.action_len
