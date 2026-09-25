import torch
import numpy as np
from einops import rearrange, repeat
from .base_planner import BasePlanner
from utils import move_to_device
from eval_logging import Timer


class CEMPlanner(BasePlanner):
    """CEM planner with optional EB-CEM proposal reuse.

    EB-CEM stores the top action sequences found during the initial MPC search
    and, in later MPC iterations, samples a local/global mixture around that
    persistent elite bank. In the reported Scope-WM configuration the bank is
    saved at MPC step 0, reused from step 1, updated in ``initial_only`` mode,
    and not shifted across action-sequence time steps.
    """

    def __init__(
        self,
        horizon,
        topk,
        num_samples,
        var_scale,
        opt_steps,
        eval_every,
        wm,
        action_dim,
        objective_fn,
        preprocessor,
        evaluator,
        wandb_run,
        logging_prefix="plan_0",
        log_filename="logs.json",
        metrics_logger=None,
        diagnose_candidate_env=False,
        diagnose_candidate_topk=10,
        diagnose_candidate_random=0,
        diagnose_candidate_eval_indices=None,
        diagnose_candidate_mpc_iters=None,
        diagnose_candidate_cem_iters=None,
        diagnose_candidate_filename="candidate_env_diagnostics.jsonl",
        eb_cem_enabled=None,
        eb_cem_save_mpc_iter=None,
        eb_cem_use_start_mpc_iter=None,
        eb_cem_topm=None,
        eb_cem_sample_fraction=None,
        eb_cem_noise_scale=None,
        eb_cem_shift_steps=None,
        eb_cem_update_mode=None,
        elite_bank_enabled=False,
        elite_bank_save_mpc_iter=0,
        elite_bank_use_start_mpc_iter=1,
        elite_bank_topm=30,
        elite_bank_sample_fraction=0.7,
        elite_bank_noise_scale=0.5,
        elite_bank_shift_steps=0,
        elite_bank_update_mode="initial_only",
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
        self.horizon = horizon
        self.topk = topk
        self.num_samples = num_samples
        self.var_scale = var_scale
        self.opt_steps = opt_steps
        self.eval_every = eval_every
        self.logging_prefix = logging_prefix
        self.diagnose_candidate_env = bool(diagnose_candidate_env)
        self.diagnose_candidate_topk = int(diagnose_candidate_topk or 0)
        self.diagnose_candidate_random = int(diagnose_candidate_random or 0)
        self.diagnose_candidate_eval_indices = self._optional_int_set(
            diagnose_candidate_eval_indices
        )
        self.diagnose_candidate_mpc_iters = self._optional_int_set(
            diagnose_candidate_mpc_iters
        )
        self.diagnose_candidate_cem_iters = self._optional_int_set(
            diagnose_candidate_cem_iters
        )
        self.diagnose_candidate_filename = diagnose_candidate_filename
        self._diagnostic_rng = np.random.default_rng(0)
        if eb_cem_enabled is not None:
            elite_bank_enabled = eb_cem_enabled
        if eb_cem_save_mpc_iter is not None:
            elite_bank_save_mpc_iter = eb_cem_save_mpc_iter
        if eb_cem_use_start_mpc_iter is not None:
            elite_bank_use_start_mpc_iter = eb_cem_use_start_mpc_iter
        if eb_cem_topm is not None:
            elite_bank_topm = eb_cem_topm
        if eb_cem_sample_fraction is not None:
            elite_bank_sample_fraction = eb_cem_sample_fraction
        if eb_cem_noise_scale is not None:
            elite_bank_noise_scale = eb_cem_noise_scale
        if eb_cem_shift_steps is not None:
            elite_bank_shift_steps = eb_cem_shift_steps
        if eb_cem_update_mode is not None:
            elite_bank_update_mode = eb_cem_update_mode
        self.elite_bank_enabled = bool(elite_bank_enabled)
        self.elite_bank_save_mpc_iter = int(elite_bank_save_mpc_iter)
        self.elite_bank_use_start_mpc_iter = int(elite_bank_use_start_mpc_iter)
        self.elite_bank_topm = max(1, int(elite_bank_topm))
        self.elite_bank_sample_fraction = float(elite_bank_sample_fraction)
        self.elite_bank_noise_scale = float(elite_bank_noise_scale)
        self.elite_bank_shift_steps = max(0, int(elite_bank_shift_steps))
        self.elite_bank_update_mode = str(elite_bank_update_mode)
        self._elite_bank_by_eval = {}
        self._active_eval_indices = None

    def set_active_eval_indices(self, indices):
        if indices is None:
            self._active_eval_indices = None
            return
        self._active_eval_indices = [int(idx) for idx in np.asarray(indices).tolist()]

    def _eval_id_for_traj(self, traj):
        if self._active_eval_indices is None:
            return int(traj)
        if traj >= len(self._active_eval_indices):
            return int(traj)
        return int(self._active_eval_indices[traj])

    @staticmethod
    def _optional_int_set(value):
        if value is None:
            return None
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
            value = [item.strip() for item in value.split(",") if item.strip()]
        if isinstance(value, int):
            value = [value]
        return {int(item) for item in value}

    def _loss_stats(self, prefix, values):
        values = values.detach().float()
        return {
            f"{prefix}_min": values.min().item(),
            f"{prefix}_mean": values.mean().item(),
            f"{prefix}_max": values.max().item(),
            f"{prefix}_std": values.std(unbiased=False).item(),
        }

    @staticmethod
    def _numeric_value(value):
        if value is None:
            return None
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                return None
            value = value.item()
        if isinstance(value, (bool, np.bool_)):
            return float(value)
        if isinstance(value, (int, float, np.integer, np.floating)):
            return float(value)
        return None

    def _aggregate_step_stats(self, step_stats):
        keys = sorted({key for stats in step_stats for key in stats.keys()})
        aggregated = {}
        for key in keys:
            values = []
            for stats in step_stats:
                if key not in stats:
                    continue
                numeric = self._numeric_value(stats[key])
                if numeric is not None and np.isfinite(numeric):
                    values.append(numeric)
            if not values:
                continue
            aggregated[key] = float(np.mean(values))
            if len(values) != len(step_stats):
                aggregated[f"{key}_num_records"] = int(len(values))
        return aggregated

    @staticmethod
    def _rank_1d(values):
        order = torch.argsort(values)
        ranks = torch.empty_like(order, dtype=torch.float32)
        ranks[order] = torch.arange(values.numel(), device=values.device, dtype=torch.float32)
        return ranks

    def _current_mpc_iter(self):
        if isinstance(self.logging_prefix, str) and self.logging_prefix.startswith(
            "plan_"
        ):
            try:
                return int(self.logging_prefix.rsplit("_", 1)[1])
            except ValueError:
                return None
        return None

    def _should_use_elite_bank(self):
        if not self.elite_bank_enabled:
            return False
        mpc_iter = self._current_mpc_iter()
        if mpc_iter is None:
            return False
        if mpc_iter < self.elite_bank_use_start_mpc_iter:
            return False
        return bool(self._elite_bank_by_eval)

    def _should_save_elite_bank(self):
        if not self.elite_bank_enabled:
            return False
        mpc_iter = self._current_mpc_iter()
        if mpc_iter is None:
            return False
        if self.elite_bank_update_mode == "initial_only":
            return mpc_iter == self.elite_bank_save_mpc_iter
        if self.elite_bank_update_mode == "always":
            return mpc_iter >= self.elite_bank_save_mpc_iter
        return False

    def _standard_action_samples(self, mu, sigma, num_samples):
        horizon = int(mu.shape[0])
        action = torch.randn(
            int(num_samples),
            horizon,
            self.action_dim,
            device=mu.device,
            dtype=mu.dtype,
        )
        return action * sigma.unsqueeze(0) + mu.unsqueeze(0)

    def _shift_elite_bank_actions(self, bank, mu, sigma):
        shift_steps = int(self.elite_bank_shift_steps)
        if shift_steps <= 0:
            return bank
        if shift_steps >= bank.shape[1]:
            return bank
        tail = self._standard_action_samples(
            mu[-shift_steps:].contiguous(),
            sigma[-shift_steps:].contiguous(),
            bank.shape[0],
        )
        return torch.cat([bank[:, shift_steps:], tail], dim=1)

    def _sample_actions_with_elite_bank(self, traj, mu, sigma, num_samples):
        if not self._should_use_elite_bank():
            return self._standard_action_samples(mu, sigma, num_samples), 0

        bank = self._elite_bank_by_eval.get(self._eval_id_for_traj(traj))
        if bank is None or bank.numel() == 0:
            return self._standard_action_samples(mu, sigma, num_samples), 0
        bank = bank.to(device=mu.device, dtype=mu.dtype)

        num_samples = int(num_samples)
        local_count = int(round(num_samples * self.elite_bank_sample_fraction))
        local_count = max(0, min(local_count, num_samples))
        global_count = num_samples - local_count

        chunks = []
        if local_count > 0:
            # EB-CEM: reuse elites collected from earlier MPC search as proposal centers.
            bank_idx = torch.randint(
                low=0,
                high=bank.shape[0],
                size=(local_count,),
                device=mu.device,
            )
            local_base = self._shift_elite_bank_actions(bank[bank_idx], mu, sigma)
            # Local perturbation around the persistent elite bank.
            local_noise = (
                torch.randn_like(local_base)
                * sigma.unsqueeze(0)
                * self.elite_bank_noise_scale
            )
            chunks.append(local_base + local_noise)
        if global_count > 0:
            chunks.append(self._standard_action_samples(mu, sigma, global_count))
        if not chunks:
            return self._standard_action_samples(mu, sigma, num_samples), 0
        return torch.cat(chunks, dim=0), local_count

    def _save_elite_bank(self, traj, action, loss_detached):
        if not self._should_save_elite_bank():
            return 0
        topm = min(self.elite_bank_topm, int(action.shape[0]))
        if topm <= 0:
            return 0
        bank_idx = torch.argsort(loss_detached)[:topm]
        # EB-CEM: store the lowest-cost sampled action sequences per evaluation seed.
        self._elite_bank_by_eval[self._eval_id_for_traj(traj)] = (
            action[bank_idx].detach()
        )
        return topm

    def _should_diagnose_candidate_env(self, traj, cem_iter):
        if not self.diagnose_candidate_env:
            return False
        mpc_iter = self._current_mpc_iter()
        if (
            self.diagnose_candidate_mpc_iters is not None
            and mpc_iter not in self.diagnose_candidate_mpc_iters
        ):
            return False
        if (
            self.diagnose_candidate_cem_iters is not None
            and cem_iter not in self.diagnose_candidate_cem_iters
        ):
            return False
        if (
            self.diagnose_candidate_eval_indices is not None
            and traj not in self.diagnose_candidate_eval_indices
        ):
            return False
        return True

    def _candidate_indices_for_diagnostics(self, loss_detached):
        num_candidates = int(loss_detached.numel())
        selected = []
        if self.diagnose_candidate_topk > 0:
            k = min(self.diagnose_candidate_topk, num_candidates)
            selected.extend(torch.argsort(loss_detached)[:k].detach().cpu().tolist())
        if self.diagnose_candidate_random > 0:
            k = min(self.diagnose_candidate_random, num_candidates)
            random_idx = self._diagnostic_rng.permutation(num_candidates)[:k]
            selected.extend(random_idx.tolist())
        # Preserve order while removing duplicates.
        return list(dict.fromkeys(int(idx) for idx in selected))

    def _rollout_candidate_in_env(self, traj, action):
        exec_action = rearrange(
            action.detach().cpu().unsqueeze(0),
            "b t (f d) -> b (t f) d",
            f=self.evaluator.frameskip,
        )
        exec_action = self.preprocessor.denormalize_actions(exec_action).numpy()
        seed_arr = np.asarray(self.evaluator.seed)
        cur_seed = np.asarray([seed_arr[traj]])
        cur_init_state = np.expand_dims(np.asarray(self.evaluator.state_0[traj]), axis=0)

        env = self.evaluator.env
        if hasattr(env, "workers"):
            _, states = env.rollout(
                cur_seed,
                cur_init_state,
                exec_action,
                id=[traj],
            )
        elif hasattr(env, "envs"):
            _, state = env.envs[traj].rollout(
                int(cur_seed[0]),
                cur_init_state[0],
                exec_action[0],
            )
            states = np.expand_dims(state, axis=0)
        else:
            _, states = env.rollout(cur_seed, cur_init_state, exec_action)
        return states[0, -1]

    def _diagnose_candidate_env(
        self,
        traj,
        cem_iter,
        action,
        loss_detached,
        topk_idx,
    ):
        if not self._should_diagnose_candidate_env(traj, cem_iter):
            return None
        candidate_indices = self._candidate_indices_for_diagnostics(loss_detached)
        if not candidate_indices:
            return None
        pred_ranks = self._rank_1d(loss_detached)
        goal_state = np.asarray(self.evaluator.state_g[traj])
        init_state = np.asarray(self.evaluator.state_0[traj])
        current_state_dist = float(np.linalg.norm(goal_state - init_state))
        pred_topk_set = set(topk_idx.detach().cpu().tolist())
        records = []
        for candidate_idx in candidate_indices:
            final_state = self._rollout_candidate_in_env(
                traj,
                action[candidate_idx],
            )
            state_dist = float(np.linalg.norm(goal_state - final_state))
            records.append(
                {
                    "candidate_idx": int(candidate_idx),
                    "pred_loss": float(loss_detached[candidate_idx].item()),
                    "pred_rank": float(pred_ranks[candidate_idx].item()),
                    "true_state_dist": state_dist,
                    "state_dist_improvement": current_state_dist - state_dist,
                    "is_pred_topk": bool(candidate_idx in pred_topk_set),
                }
            )
        best_pred_idx = int(torch.argsort(loss_detached)[0].item())
        best_true = min(records, key=lambda item: item["true_state_dist"])
        best_pred_record = next(
            (item for item in records if item["candidate_idx"] == best_pred_idx),
            None,
        )
        selected_true_dists = [
            item["true_state_dist"] for item in records if item["is_pred_topk"]
        ]
        seed_arr = np.asarray(self.evaluator.seed)
        result = {
            "kind": "diagnostic",
            "event": "candidate_env_diagnostic",
            "mpc_iter": self._current_mpc_iter(),
            "cem_iter": int(cem_iter),
            "cem_step": int(cem_iter + 1),
            "eval_index": int(traj),
            "eval_seed": int(seed_arr[traj]),
            "num_candidates_sampled": int(action.shape[0]),
            "num_candidates_evaluated": int(len(records)),
            "diagnosed_candidate_indices": candidate_indices,
            "current_state_dist": current_state_dist,
            "best_true_state_dist": float(best_true["true_state_dist"]),
            "best_true_candidate_idx": int(best_true["candidate_idx"]),
            "best_true_pred_rank": float(best_true["pred_rank"]),
            "best_true_pred_loss": float(best_true["pred_loss"]),
            "best_true_state_dist_improvement": float(
                best_true["state_dist_improvement"]
            ),
            "best_pred_candidate_idx": best_pred_idx,
            "best_pred_true_state_dist": (
                None
                if best_pred_record is None
                else float(best_pred_record["true_state_dist"])
            ),
            "best_pred_state_dist_improvement": (
                None
                if best_pred_record is None
                else float(best_pred_record["state_dist_improvement"])
            ),
            "pred_topk_min_true_state_dist": (
                None if not selected_true_dists else float(min(selected_true_dists))
            ),
            "pred_topk_mean_true_state_dist": (
                None if not selected_true_dists else float(np.mean(selected_true_dists))
            ),
            "candidate_records": records,
        }
        self.append_metrics_record(self.diagnose_candidate_filename, result)
        return result

    def init_mu_sigma(self, obs_0, actions=None):
        """
        actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        mu, sigma could depend on current obs, but obs_0 is only used for providing n_evals for now
        """
        n_evals = obs_0["visual"].shape[0]
        sigma = self.var_scale * torch.ones([n_evals, self.horizon, self.action_dim])
        if actions is None:
            mu = torch.zeros(n_evals, 0, self.action_dim)
        else:
            mu = actions
        device = mu.device
        t = mu.shape[1]
        remaining_t = self.horizon - t

        if remaining_t > 0:
            new_mu = torch.zeros(n_evals, remaining_t, self.action_dim)
            mu = torch.cat([mu, new_mu.to(device)], dim=1)
        return mu, sigma

    def plan(self, obs_0, obs_g, actions=None):
        """
        Args:
            actions: normalized
        Returns:
            actions: (B, T, action_dim) torch.Tensor, T <= self.horizon
        """
        self.reset_planner_timing()
        self._refresh_objective_context(obs_0, obs_g)
        start = self._planner_time_start()
        trans_obs_0 = move_to_device(
            self.preprocessor.transform_obs(obs_0), self.device
        )
        trans_obs_g = move_to_device(
            self.preprocessor.transform_obs(obs_g), self.device
        )
        self._record_planner_elapsed("preprocess_obs", start)

        reset_wm_timing = getattr(self.wm, "reset_timing_stats", None)
        if reset_wm_timing is not None:
            reset_wm_timing()
        start = self._planner_time_start()
        z_obs_g = self.wm.encode_obs(trans_obs_g)
        self._record_planner_elapsed("goal_encode", start)
        timing_getter = getattr(self.wm, "get_timing_stats", None)
        if timing_getter is not None:
            self._accumulate_wm_timing(timing_getter(), prefix="goal_encode")

        start = self._planner_time_start()
        mu, sigma = self.init_mu_sigma(obs_0, actions)
        mu, sigma = mu.to(self.device), sigma.to(self.device)
        self._record_planner_elapsed("init_distribution", start)
        n_evals = mu.shape[0]
        self.last_plan_iterations = 0
        self.last_termination_reason = "max_opt_steps"
        self.last_wm_rollout_time_sec = 0.0
        self.last_candidate_stats = []
        self.reset_eval_timing()

        for i in range(self.opt_steps):
            cem_iter_start = self._planner_time_start()
            # optimize individual instances
            losses = []
            step_stats = []
            aggregated_stats = None
            for traj in range(n_evals):
                if hasattr(self.objective_fn, "set_active_eval_index"):
                    self.objective_fn.set_active_eval_index(traj)
                start = self._planner_time_start()
                cur_trans_obs_0 = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in trans_obs_0.items()
                }
                cur_z_obs_g = {
                    key: repeat(
                        arr[traj].unsqueeze(0), "1 ... -> n ...", n=self.num_samples
                    )
                    for key, arr in z_obs_g.items()
                }
                action, elite_bank_local_count = self._sample_actions_with_elite_bank(
                    traj,
                    mu[traj],
                    sigma[traj],
                    self.num_samples,
                )
                action[0] = mu[traj]  # optional: make the first one mu itself
                self._record_planner_elapsed("candidate_sampling", start)

                with Timer() as rollout_timer:
                    with torch.no_grad():
                        i_z_obses, i_zs = self.wm.rollout(
                            obs_0=cur_trans_obs_0,
                            act=action,
                        )
                self.last_wm_rollout_time_sec += rollout_timer.elapsed
                self._add_planner_timing("wm_rollout_cuda_synced", rollout_timer.elapsed)
                if timing_getter is not None:
                    self._accumulate_wm_timing(timing_getter(), prefix="rollout")

                start = self._planner_time_start()
                loss = self.objective_fn(i_z_obses, cur_z_obs_g)
                loss_detached = loss.detach()
                self._record_planner_elapsed("objective_fn", start)

                start = self._planner_time_start()
                topk_idx = torch.argsort(loss)[: self.topk]
                topk_action = action[topk_idx]
                elite_loss = loss_detached[topk_idx]
                stats = self._loss_stats("candidate_loss", loss_detached)
                stats.update(self._loss_stats("elite_loss", elite_loss))
                saved_elite_bank_count = 0
                if i == self.opt_steps - 1:
                    saved_elite_bank_count = self._save_elite_bank(
                        traj,
                        action,
                        loss_detached,
                    )
                stats.update(
                    {
                        "elite_bank_enabled": bool(self.elite_bank_enabled),
                        "eb_cem_enabled": bool(self.elite_bank_enabled),
                        "elite_bank_size": int(
                            0
                            if self._elite_bank_by_eval.get(
                                self._eval_id_for_traj(traj)
                            )
                            is None
                            else self._elite_bank_by_eval[
                                self._eval_id_for_traj(traj)
                            ].shape[0]
                        ),
                        "eb_cem_size": int(
                            0
                            if self._elite_bank_by_eval.get(
                                self._eval_id_for_traj(traj)
                            )
                            is None
                            else self._elite_bank_by_eval[
                                self._eval_id_for_traj(traj)
                            ].shape[0]
                        ),
                        "elite_bank_local_count": int(elite_bank_local_count),
                        "eb_cem_local_count": int(elite_bank_local_count),
                        "elite_bank_global_count": int(
                            self.num_samples - elite_bank_local_count
                        ),
                        "eb_cem_global_count": int(
                            self.num_samples - elite_bank_local_count
                        ),
                        "elite_bank_saved_count": int(saved_elite_bank_count),
                        "eb_cem_saved_count": int(saved_elite_bank_count),
                        "elite_bank_sample_fraction": float(
                            self.elite_bank_sample_fraction
                        ),
                        "eb_cem_sample_fraction": float(
                            self.elite_bank_sample_fraction
                        ),
                        "elite_bank_noise_scale": float(self.elite_bank_noise_scale),
                        "eb_cem_noise_scale": float(self.elite_bank_noise_scale),
                        "elite_bank_shift_steps": int(self.elite_bank_shift_steps),
                        "eb_cem_shift_steps": int(self.elite_bank_shift_steps),
                    }
                )
                candidate_env_diag = self._diagnose_candidate_env(
                    traj=traj,
                    cem_iter=i,
                    action=action,
                    loss_detached=loss_detached,
                    topk_idx=topk_idx,
                )
                if candidate_env_diag is not None:
                    stats.update(
                        {
                            "candidate_env_best_true_state_dist": candidate_env_diag[
                                "best_true_state_dist"
                            ],
                            "candidate_env_best_true_pred_rank": candidate_env_diag[
                                "best_true_pred_rank"
                            ],
                            "candidate_env_best_true_improvement": candidate_env_diag[
                                "best_true_state_dist_improvement"
                            ],
                            "candidate_env_best_pred_true_state_dist": (
                                candidate_env_diag[
                                    "best_pred_true_state_dist"
                                ]
                            ),
                        }
                    )
                step_stats.append(stats)
                losses.append(loss[topk_idx[0]].item())
                mu[traj] = topk_action.mean(dim=0)
                sigma[traj] = topk_action.std(dim=0)
                self._record_planner_elapsed("elite_select_update", start)

            if step_stats:
                aggregated_stats = self._aggregate_step_stats(step_stats)
                self.last_candidate_stats.append(aggregated_stats)
            self.last_plan_iterations = i + 1
            mean_best_loss = float(np.mean(losses)) if losses else None
            mpc_iter = None
            if isinstance(self.logging_prefix, str) and self.logging_prefix.startswith(
                "plan_"
            ):
                try:
                    mpc_iter = int(self.logging_prefix.rsplit("_", 1)[1])
                except ValueError:
                    mpc_iter = None
            self.wandb_run.log(
                {f"{self.logging_prefix}/loss": mean_best_loss, "step": i + 1}
            )
            self._record_planner_elapsed("cem_iteration_total", cem_iter_start)
            self.append_metrics_record(
                "cem_iterations.jsonl",
                {
                    "kind": "plan",
                    "event": "cem_iteration",
                    "mpc_iter": mpc_iter,
                    "cem_iter": i,
                    "cem_step": i + 1,
                    "n_evals": n_evals,
                    "horizon": self.horizon,
                    "num_samples": self.num_samples,
                    "topk": self.topk,
                    "opt_steps": self.opt_steps,
                    "eval_every": self.eval_every,
                    "candidate_count": n_evals * self.num_samples,
                    "elite_count": n_evals * self.topk,
                    "mean_best_loss": mean_best_loss,
                    "per_eval_best_loss": losses,
                    "candidate_stats": aggregated_stats,
                    "elite_bank_enabled": bool(self.elite_bank_enabled),
                    "eb_cem_enabled": bool(self.elite_bank_enabled),
                    "elite_bank_num_entries": int(len(self._elite_bank_by_eval)),
                    "eb_cem_num_entries": int(len(self._elite_bank_by_eval)),
                    "elite_bank_topm": int(self.elite_bank_topm),
                    "eb_cem_topm": int(self.elite_bank_topm),
                    "elite_bank_sample_fraction": float(
                        self.elite_bank_sample_fraction
                    ),
                    "eb_cem_sample_fraction": float(
                        self.elite_bank_sample_fraction
                    ),
                    "elite_bank_noise_scale": float(self.elite_bank_noise_scale),
                    "eb_cem_noise_scale": float(self.elite_bank_noise_scale),
                    "elite_bank_shift_steps": int(self.elite_bank_shift_steps),
                    "eb_cem_shift_steps": int(self.elite_bank_shift_steps),
                    "planner_world_model_rollout_time_sec": (
                        self.last_wm_rollout_time_sec
                    ),
                    **self._timing_metrics(),
                    "termination_reason": self.last_termination_reason,
                    **self._current_wm_roi_record(),
                },
            )
            eval_every = int(self.eval_every or 0)
            if (
                self.evaluator is not None
                and eval_every > 0
                and (i + 1) % eval_every == 0
            ):
                logs, successes, _, _ = self.evaluator.eval_actions(
                    mu, filename=f"{self.logging_prefix}_output_{i+1}"
                )
                self.record_evaluator_timing()
                self.append_metrics_record(
                    "cem_evals.jsonl",
                    {
                        "kind": "plan",
                        "event": "cem_eval",
                        "mpc_iter": mpc_iter,
                        "cem_iter": i,
                        "cem_step": i + 1,
                        "n_evals": n_evals,
                        "horizon": self.horizon,
                        "num_samples": self.num_samples,
                        "topk": self.topk,
                        "mean_best_loss": mean_best_loss,
                        "candidate_stats": aggregated_stats,
                        **self._timing_metrics(),
                        **self._last_eval_record(),
                    },
                )
                logs = {f"{self.logging_prefix}/{k}": v for k, v in logs.items()}
                logs.update({"step": i + 1})
                self.wandb_run.log(logs)
                self.dump_logs(logs)
                if np.all(successes):
                    self.last_termination_reason = "all_success"
                    break  # terminate planning if all success

        return mu, np.full(n_evals, np.inf)  # all actions are valid
