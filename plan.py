import os
os.environ.setdefault("WANDB_MODE", "disabled")

import gym
import json
import math
import hydra
import random
import torch
import pickle
import wandb
import logging
import warnings
import numpy as np
import submitit
from itertools import product
from pathlib import Path
from einops import rearrange
from omegaconf import OmegaConf, open_dict

from env.venv import SubprocVectorEnv
from custom_resolvers import replace_slash
from models.roi import extract_drs_config_dict
from models.sparse_dynamics import extract_sparse_dynamics_config_dict
from preprocessor import Preprocessor
from planning.evaluator import PlanEvaluator
from utils import cfg_to_dict, seed
from eval_logging import (
    SafeMetricsLogger,
    add_lightweight_defaults,
    build_run_meta,
    count_parameters,
    cuda_peak_memory_mb,
    elapsed_since,
    json_safe,
    perf_counter,
    reset_cuda_peak_memory,
)

warnings.filterwarnings("ignore")
log = logging.getLogger(__name__)

ALL_MODEL_KEYS = [
    "encoder",
    "predictor",
    "decoder",
    "proprio_encoder",
    "action_encoder",
    "sparse_primary_dynamics",
]


def _parse_eval_seed_list(value):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [x.strip() for x in value.split(",") if x.strip()]
    else:
        parsed = value

    if isinstance(parsed, int):
        parsed = [parsed]
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(
            "eval_seed_list must be null, an integer, a list of integers, "
            "or a comma-separated string."
        )
    try:
        return [int(x) for x in parsed]
    except (TypeError, ValueError) as exc:
        raise ValueError("eval_seed_list contains a non-integer value.") from exc


def _parse_int_list(value, field_name):
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = [x.strip() for x in value.split(",") if x.strip()]
    else:
        parsed = value

    if isinstance(parsed, int):
        parsed = [parsed]
    if not isinstance(parsed, (list, tuple)):
        raise ValueError(
            f"{field_name} must be null, an integer, a list of integers, "
            "or a comma-separated string."
        )
    try:
        return [int(x) for x in parsed]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} contains a non-integer value.") from exc


def normalize_eval_seed_cfg(cfg_dict):
    eval_seed_list = _parse_eval_seed_list(cfg_dict.get("eval_seed_list"))
    if eval_seed_list is not None:
        cfg_dict["eval_seed_list"] = eval_seed_list
        cfg_dict["n_evals"] = len(eval_seed_list)
    dset_eval_indices = _parse_int_list(
        cfg_dict.get("dset_eval_indices"), "dset_eval_indices"
    )
    if dset_eval_indices is not None:
        cfg_dict["dset_eval_indices"] = dset_eval_indices
        cfg_dict["n_evals"] = len(dset_eval_indices)
        if eval_seed_list is not None and len(eval_seed_list) != len(dset_eval_indices):
            raise ValueError(
                "eval_seed_list and dset_eval_indices must have the same length "
                "when both are provided."
            )
    return cfg_dict


def planning_main_in_dir(working_dir, cfg_dict):
    os.chdir(working_dir)
    return planning_main(cfg_dict=cfg_dict)

def launch_plan_jobs(
    epoch,
    cfg_dicts,
    plan_output_dir,
):
    with submitit.helpers.clean_env():
        jobs = []
        for cfg_dict in cfg_dicts:
            subdir_name = f"{cfg_dict['planner']['name']}_goal_source={cfg_dict['goal_source']}_goal_H={cfg_dict['goal_H']}_alpha={cfg_dict['objective']['alpha']}"
            subdir_path = os.path.join(plan_output_dir, subdir_name)
            executor = submitit.AutoExecutor(
                folder=subdir_path, slurm_max_num_timeout=20
            )
            executor.update_parameters(
                **{
                    k: v
                    for k, v in cfg_dict["hydra"]["launcher"].items()
                    if k != "submitit_folder"
                }
            )
            cfg_dict["saved_folder"] = subdir_path
            cfg_dict["wandb_logging"] = False  # don't init wandb
            job = executor.submit(planning_main_in_dir, subdir_path, cfg_dict)
            jobs.append((epoch, subdir_name, job))
            print(
                f"Submitted evaluation job for checkpoint: {subdir_path}, job id: {job.job_id}"
            )
        return jobs


def build_plan_cfg_dicts(
    plan_cfg_path="",
    ckpt_base_path="",
    model_name="",
    model_epoch="final",
    planner=["gd", "cem"],
    goal_source=["dset"],
    goal_H=[1, 5, 10],
    alpha=[0, 0.1, 1],
):
    """
    Return a list of plan overrides, for model_path, add a key in the dict {"model_path": model_path}.
    """
    config_path = os.path.dirname(plan_cfg_path)
    overrides = [
        {
            "planner": p,
            "goal_source": g_source,
            "goal_H": g_H,
            "ckpt_base_path": ckpt_base_path,
            "model_name": model_name,
            "model_epoch": model_epoch,
            "objective": {"alpha": a},
        }
        for p, g_source, g_H, a in product(planner, goal_source, goal_H, alpha)
    ]
    cfg = OmegaConf.load(plan_cfg_path)
    cfg_dicts = []
    for override_args in overrides:
        planner = override_args["planner"]
        planner_cfg = OmegaConf.load(
            os.path.join(config_path, f"planner/{planner}.yaml")
        )
        cfg["planner"] = OmegaConf.merge(cfg.get("planner", {}), planner_cfg)
        override_args.pop("planner")
        cfg = OmegaConf.merge(cfg, OmegaConf.create(override_args))
        cfg_dict = OmegaConf.to_container(cfg)
        cfg_dict["planner"]["horizon"] = cfg_dict["goal_H"]  # assume planning horizon equals to goal horizon
        cfg_dicts.append(cfg_dict)
    return cfg_dicts


class PlanWorkspace:
    def __init__(
        self,
        cfg_dict: dict,
        wm: torch.nn.Module,
        dset,
        env: SubprocVectorEnv,
        env_name: str,
        frameskip: int,
        wandb_run: wandb.run,
        metrics_logger=None,
    ):
        self.cfg_dict = cfg_dict
        self.wm = wm
        self.dset = dset
        self.env = env
        self.env_name = env_name
        self.frameskip = frameskip
        self.wandb_run = wandb_run
        self.device = next(wm.parameters()).device
        self.metrics_logger = metrics_logger or SafeMetricsLogger(os.getcwd())

        if cfg_dict.get("eval_seed_list") is not None:
            self.eval_seed = [int(x) for x in cfg_dict["eval_seed_list"]]
            self.n_evals = len(self.eval_seed)
        else:
            self.eval_seed = [
                cfg_dict["seed"] * n + 1 for n in range(cfg_dict["n_evals"])
            ]
            self.n_evals = cfg_dict["n_evals"]
        print("eval_seed: ", self.eval_seed)
        self.goal_source = cfg_dict["goal_source"]
        self.goal_H = cfg_dict["goal_H"]
        self.dset_eval_offset = int(cfg_dict.get("dset_eval_offset", 0) or 0)
        self.dset_eval_indices = cfg_dict.get("dset_eval_indices", None)
        self.action_dim = self.dset.action_dim * self.frameskip
        self.debug_dset_init = cfg_dict["debug_dset_init"]

        self.data_preprocessor = Preprocessor(
            action_mean=self.dset.action_mean,
            action_std=self.dset.action_std,
            state_mean=self.dset.state_mean,
            state_std=self.dset.state_std,
            proprio_mean=self.dset.proprio_mean,
            proprio_std=self.dset.proprio_std,
            transform=self.dset.transform,
        )

        if self.cfg_dict["goal_source"] == "file":
            self.prepare_targets_from_file(cfg_dict["goal_file_path"])
        else:
            self.prepare_targets()

        self.evaluator = PlanEvaluator(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            state_0=self.state_0,
            state_g=self.state_g,
            env=self.env,
            wm=self.wm,
            frameskip=self.frameskip,
            seed=self.eval_seed,
            preprocessor=self.data_preprocessor,
            n_plot_samples=self.cfg_dict["n_plot_samples"],
        )

        objective_fn = hydra.utils.call(
            cfg_dict["objective"],
        )
        if hasattr(objective_fn, "set_context"):
            objective_fn.set_context(
                wm=self.wm,
                preprocessor=self.data_preprocessor,
                obs_0=self.obs_0,
                obs_g=self.obs_g,
                device=self.device,
            )

        if self.wandb_run is None or isinstance(
            self.wandb_run, wandb.sdk.lib.disabled.RunDisabled
        ):
            self.wandb_run = DummyWandbRun()

        self.log_filename = "logs.json"  # planner and final eval logs are dumped here
        self.planner = hydra.utils.instantiate(
            self.cfg_dict["planner"],
            wm=self.wm,
            env=self.env,  # only for mpc
            action_dim=self.action_dim,
            objective_fn=objective_fn,
            preprocessor=self.data_preprocessor,
            evaluator=self.evaluator,
            wandb_run=self.wandb_run,
            log_filename=self.log_filename,
            metrics_logger=self.metrics_logger,
        )

        # optional: assume planning horizon equals to goal horizon
        from planning.mpc import MPCPlanner
        if isinstance(self.planner, MPCPlanner):
            self.planner.sub_planner.horizon = cfg_dict["goal_H"]
            self.planner.n_taken_actions = cfg_dict["goal_H"]
        else:
            self.planner.horizon = cfg_dict["goal_H"]

        self.dump_targets()

    def prepare_targets(self):
        states = []
        actions = []
        observations = []
        
        if self.goal_source == "random_state":
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(traj_len=2, use_eval_offset=False)
            )
            self.env.update_env(env_info)

            # sample random states
            rand_init_state, rand_goal_state = self.env.sample_random_init_goal_states(
                self.eval_seed
            )
            if self.env_name == "deformable_env": # take rand init state from dset for deformable envs
                rand_init_state = np.array([x[0] for x in states])

            obs_0, state_0 = self.env.prepare(self.eval_seed, rand_init_state)
            obs_g, state_g = self.env.prepare(self.eval_seed, rand_goal_state)

            # add dim for t
            for k in obs_0.keys():
                obs_0[k] = np.expand_dims(obs_0[k], axis=1)
                obs_g[k] = np.expand_dims(obs_g[k], axis=1)

            self.obs_0 = obs_0
            self.obs_g = obs_g
            self.state_0 = rand_init_state  # (b, d)
            self.state_g = rand_goal_state
            self.gt_actions = None
        else:
            # update env config from val trajs
            observations, states, actions, env_info = (
                self.sample_traj_segment_from_dset(
                    traj_len=self.frameskip * self.goal_H + 1,
                    use_eval_offset=True,
                )
            )
            self.env.update_env(env_info)

            # get states from val trajs
            init_state = [x[0] for x in states]
            init_state = np.array(init_state)
            actions = torch.stack(actions)
            if self.goal_source == "random_action":
                actions = torch.randn_like(actions)
            wm_actions = rearrange(actions, "b (t f) d -> b t (f d)", f=self.frameskip)
            exec_actions = self.data_preprocessor.denormalize_actions(actions)
            # replay actions in env to get gt obses
            rollout_obses, rollout_states = self.env.rollout(
                self.eval_seed, init_state, exec_actions.numpy()
            )
            self.obs_0 = {
                key: np.expand_dims(arr[:, 0], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.obs_g = {
                key: np.expand_dims(arr[:, -1], axis=1)
                for key, arr in rollout_obses.items()
            }
            self.state_0 = init_state  # (b, d)
            self.state_g = rollout_states[:, -1]  # (b, d)
            self.gt_actions = wm_actions

    def _draw_dset_segment(self, traj_len):
        max_offset = -1
        while max_offset < 0:  # filter out traj that are not long enough
            traj_id = random.randint(0, len(self.dset) - 1)
            obs, act, state, e_info = self.dset[traj_id]
            max_offset = obs["visual"].shape[0] - traj_len
        offset = random.randint(0, max_offset)
        return traj_id, offset, obs, act, state, e_info

    def sample_traj_segment_from_dset(self, traj_len, use_eval_offset=True):
        states = []
        actions = []
        observations = []
        env_info = []
        segment_records = []

        # Check if any trajectory is long enough
        valid_traj = [
            self.dset[i][0]["visual"].shape[0]
            for i in range(len(self.dset))
            if self.dset[i][0]["visual"].shape[0] >= traj_len
        ]
        if len(valid_traj) == 0:
            raise ValueError("No trajectory in the dataset is long enough.")

        if not use_eval_offset:
            target_indices = list(range(self.n_evals))
        elif self.dset_eval_indices is not None:
            target_indices = [int(x) for x in self.dset_eval_indices]
        else:
            target_indices = list(
                range(self.dset_eval_offset, self.dset_eval_offset + self.n_evals)
            )
        if any(idx < 0 for idx in target_indices):
            raise ValueError(f"dset eval draw indices must be non-negative: {target_indices}")

        target_set = set(target_indices)
        max_draw_idx = max(target_indices) if target_indices else -1

        # Draw the same dset segment sequence as a single large plan run, then
        # collect only the requested draw indices. This makes chunked planning
        # comparable to one run with a larger n_evals.
        for draw_idx in range(max_draw_idx + 1):
            traj_id, offset, obs, act, state, e_info = self._draw_dset_segment(
                traj_len
            )
            if draw_idx not in target_set:
                continue
            state = state.numpy()
            obs = {
                key: arr[offset : offset + traj_len]
                for key, arr in obs.items()
            }
            state = state[offset : offset + traj_len]
            act = act[offset : offset + self.frameskip * self.goal_H]
            actions.append(act)
            states.append(state)
            observations.append(obs)
            env_info.append(e_info)
            segment_records.append(
                {
                    "draw_idx": int(draw_idx) if target_indices is not None else None,
                    "traj_id": int(traj_id),
                    "offset": int(offset),
                    "traj_len": int(traj_len),
                }
            )
        self.dset_segment_records = segment_records
        return observations, states, actions, env_info

    def prepare_targets_from_file(self, file_path):
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        self.obs_0 = data["obs_0"]
        self.obs_g = data["obs_g"]
        self.state_0 = data["state_0"]
        self.state_g = data["state_g"]
        self.gt_actions = data["gt_actions"]
        self.goal_H = data["goal_H"]

    def dump_targets(self):
        with open("plan_targets.pkl", "wb") as f:
            pickle.dump(
                {
                    "obs_0": self.obs_0,
                    "obs_g": self.obs_g,
                    "state_0": self.state_0,
                    "state_g": self.state_g,
                    "gt_actions": self.gt_actions,
                    "goal_H": self.goal_H,
                    "eval_seed": self.eval_seed,
                    "dset_eval_offset": self.dset_eval_offset,
                    "dset_eval_indices": self.dset_eval_indices,
                    "dset_segment_records": getattr(
                        self, "dset_segment_records", None
                    ),
                },
                f,
            )
        file_path = os.path.abspath("plan_targets.pkl")
        print(f"Dumped plan targets to {file_path}")

    def perform_planning(self):
        if self.debug_dset_init:
            actions_init = self.gt_actions
        else:
            actions_init = None
        reset_cuda_peak_memory(self.device)
        episode_start = perf_counter()
        planning_start = perf_counter()
        actions, action_len = self.planner.plan(
            obs_0=self.obs_0,
            obs_g=self.obs_g,
            actions=actions_init,
        )
        planning_time = elapsed_since(planning_start)
        final_eval_start = perf_counter()
        logs, successes, _, _ = self.evaluator.eval_actions(
            actions.detach(), action_len, save_video=True, filename="output_final"
        )
        final_eval_time = elapsed_since(final_eval_start)
        try:
            torch.save(
                {
                    "actions": actions.detach().cpu(),
                    "action_len": np.asarray(action_len),
                    "successes": np.asarray(successes),
                },
                "planned_actions.pt",
            )
        except Exception as exc:
            log.warning("Failed to save planned actions: %s", exc)
        total_episode_time = elapsed_since(episode_start)
        logs = {f"final_eval/{k}": v for k, v in logs.items()}
        self.wandb_run.log(logs)
        logs_entry = json_safe({
            key: (
                value.item()
                if isinstance(value, (np.float32, np.int32, np.int64))
                else value
            )
            for key, value in logs.items()
        })
        try:
            with open(self.log_filename, "a") as file:
                file.write(json.dumps(logs_entry) + "\n")
        except Exception as exc:
            log.warning("Failed to write final planner logs to %s: %s", self.log_filename, exc)
        try:
            self._write_plan_metrics(
                logs_entry=logs_entry,
                successes=successes,
                action_len=action_len,
                planning_time=planning_time,
                final_eval_time=final_eval_time,
                total_episode_time=total_episode_time,
            )
        except Exception as exc:
            log.warning("Failed to write planning metrics: %s", exc)
        return logs

    def _planner_iterations(self):
        return getattr(self.planner, "last_plan_iterations", None)

    def _mpc_iterations(self):
        if self.cfg_dict["planner"]["name"].startswith("mpc"):
            return getattr(self.planner, "last_plan_iterations", None)
        return None

    def _first_success_iter(self, action_length):
        if action_length == np.inf:
            return None
        n_taken_actions = getattr(self.planner, "n_taken_actions", None)
        if n_taken_actions is None:
            return None
        try:
            return int(math.ceil(float(action_length) / float(n_taken_actions)))
        except Exception:
            return None

    def _write_plan_metrics(
        self,
        logs_entry,
        successes,
        action_len,
        planning_time,
        final_eval_time,
        total_episode_time,
    ):
        eval_info = self.evaluator.last_eval_info or {}
        timings = eval_info.get("timings", {})
        planner_wm_time = getattr(self.planner, "last_wm_rollout_time_sec", None)
        planner_eval_wm_time = getattr(
            self.planner, "last_eval_world_model_rollout_time_sec", 0.0
        )
        planner_env_time = getattr(
            self.planner, "last_eval_environment_step_time_sec", 0.0
        )
        planner_decoder_time = getattr(self.planner, "last_eval_decoder_time_sec", 0.0)
        planner_visualization_time = getattr(
            self.planner, "last_eval_visualization_time_sec", 0.0
        )
        planner_eval_time = getattr(self.planner, "last_eval_total_time_sec", 0.0)
        planner_timing_breakdown = getattr(
            self.planner, "last_planner_timing_breakdown", {}
        )
        planner_timing_counts = getattr(self.planner, "last_planner_timing_counts", {})
        wm_timing_breakdown = getattr(self.planner, "last_wm_timing_breakdown", {})
        wm_timing_counts = getattr(self.planner, "last_wm_timing_counts", {})
        final_wm_timing_breakdown = timings.get("wm_timing_breakdown", {})
        final_wm_timing_counts = timings.get("wm_timing_counts", {})
        final_wm_time = timings.get("world_model_rollout_time_sec")
        final_env_time = timings.get("environment_step_time_sec")
        final_decoder_time = timings.get("decoder_time_sec")
        final_visualization_time = timings.get("visualization_time_sec")
        total_wm_time = None
        if planner_wm_time is not None and final_wm_time is not None:
            total_wm_time = planner_wm_time + planner_eval_wm_time + final_wm_time
        total_env_time = (
            planner_env_time + final_env_time
            if final_env_time is not None
            else None
        )
        total_decoder_time = (
            planner_decoder_time + final_decoder_time
            if final_decoder_time is not None
            else None
        )
        total_visualization_time = (
            planner_visualization_time + final_visualization_time
            if final_visualization_time is not None
            else None
        )

        cuda_peak_mb = cuda_peak_memory_mb(self.device)
        planner_iterations = self._planner_iterations()
        mpc_iterations = self._mpc_iterations()
        termination_reason = getattr(self.planner, "last_termination_reason", None)
        candidate_stats = getattr(self.planner, "last_candidate_stats", None)
        roi_stats = {}
        roi_getter = getattr(self.wm, "get_last_roi_stats", None)
        if roi_getter is not None:
            try:
                roi_stats = roi_getter()
            except Exception:
                roi_stats = {}

        record = {
            "kind": "plan",
            "event": "final_eval",
            "planner_name": self.cfg_dict["planner"]["name"],
            "goal_source": self.goal_source,
            "goal_H": self.goal_H,
            "n_evals": self.n_evals,
            "success_rate": float(np.mean(np.asarray(successes).astype(float))),
            "planner_iterations": planner_iterations,
            "mpc_iterations": mpc_iterations,
            "termination_reason": termination_reason,
            "total_episode_time_sec": total_episode_time,
            "total_planning_time_sec": planning_time,
            "final_eval_time_sec": final_eval_time,
            "planner_world_model_rollout_time_sec": planner_wm_time,
            "planner_eval_world_model_rollout_time_sec": planner_eval_wm_time,
            "planner_eval_time_sec": planner_eval_time,
            "planner_timing_breakdown": planner_timing_breakdown,
            "planner_timing_counts": planner_timing_counts,
            "wm_timing_breakdown": wm_timing_breakdown,
            "wm_timing_counts": wm_timing_counts,
            "final_eval_wm_timing_breakdown": final_wm_timing_breakdown,
            "final_eval_wm_timing_counts": final_wm_timing_counts,
            "final_eval_world_model_rollout_time_sec": final_wm_time,
            "total_world_model_rollout_time_sec": total_wm_time,
            "planner_decoder_time_sec": planner_decoder_time,
            "final_decoder_time_sec": final_decoder_time,
            "decoder_time_sec": total_decoder_time,
            "planner_visualization_time_sec": planner_visualization_time,
            "final_visualization_time_sec": final_visualization_time,
            "visualization_time_sec": total_visualization_time,
            "planner_environment_step_time_sec": planner_env_time,
            "final_environment_step_time_sec": final_env_time,
            "environment_step_time_sec": total_env_time,
            "cuda_peak_memory_mb": cuda_peak_mb,
            "cem_candidate_stats": candidate_stats,
            "final_metrics": logs_entry,
        }
        record.update(roi_stats)
        add_lightweight_defaults(record)
        self.metrics_logger.append_jsonl("plan_metrics.jsonl", record)

        per_eval = eval_info.get("per_eval", [])
        first_success_iters = []
        for result in per_eval:
            eval_id = result.get("eval_id")
            first_success_iter = (
                self._first_success_iter(action_len[eval_id])
                if eval_id is not None
                else None
            )
            if first_success_iter is not None:
                first_success_iters.append(first_success_iter)
            result.update(
                {
                    "planner_name": self.cfg_dict["planner"]["name"],
                    "goal_source": self.goal_source,
                    "goal_H": self.goal_H,
                    "planner_iterations": planner_iterations,
                    "mpc_iterations": mpc_iterations,
                    "first_success_iter": first_success_iter,
                    "termination_reason": termination_reason,
                    "total_planning_time_sec": planning_time,
                    "final_eval_time_sec": final_eval_time,
                    "planner_eval_time_sec": planner_eval_time,
                    "planner_environment_step_time_sec": planner_env_time,
                    "total_environment_step_time_sec": total_env_time,
                    "planner_decoder_time_sec": planner_decoder_time,
                    "total_decoder_time_sec": total_decoder_time,
                    "planner_visualization_time_sec": planner_visualization_time,
                    "total_visualization_time_sec": total_visualization_time,
                    "planner_world_model_rollout_time_sec": planner_wm_time,
                    "planner_eval_world_model_rollout_time_sec": planner_eval_wm_time,
                    "total_world_model_rollout_time_sec": total_wm_time,
                    "planner_timing_breakdown": planner_timing_breakdown,
                    "planner_timing_counts": planner_timing_counts,
                    "wm_timing_breakdown": wm_timing_breakdown,
                    "wm_timing_counts": wm_timing_counts,
                    "final_eval_wm_timing_breakdown": final_wm_timing_breakdown,
                    "final_eval_wm_timing_counts": final_wm_timing_counts,
                    "cuda_peak_memory_mb": cuda_peak_mb,
                }
            )
            if roi_stats:
                result.update(roi_stats)
            add_lightweight_defaults(result)
            self.metrics_logger.append_jsonl("per_eval_results.jsonl", result)

        state_distances = eval_info.get("state_distances")
        mean_final_state_distance = (
            float(np.mean(state_distances)) if state_distances is not None else None
        )
        summary = {
            "kind": "plan",
            "output_dir": os.path.abspath(os.getcwd()),
            "model_name": self.cfg_dict["model_name"],
            "model_epoch": self.cfg_dict["model_epoch"],
            "planner_name": self.cfg_dict["planner"]["name"],
            "goal_source": self.goal_source,
            "goal_H": self.goal_H,
            "n_evals": self.n_evals,
            "success_rate": record["success_rate"],
            "mean_final_state_distance": mean_final_state_distance,
            "planner_iterations": planner_iterations,
            "mpc_iterations": mpc_iterations,
            "first_success_iter": min(first_success_iters)
            if first_success_iters
            else None,
            "termination_reason": termination_reason,
            "total_episode_time_sec": total_episode_time,
            "total_planning_time_sec": planning_time,
            "final_eval_time_sec": final_eval_time,
            "planner_world_model_rollout_time_sec": planner_wm_time,
            "planner_eval_world_model_rollout_time_sec": planner_eval_wm_time,
            "planner_eval_time_sec": planner_eval_time,
            "planner_timing_breakdown": planner_timing_breakdown,
            "planner_timing_counts": planner_timing_counts,
            "wm_timing_breakdown": wm_timing_breakdown,
            "wm_timing_counts": wm_timing_counts,
            "final_eval_wm_timing_breakdown": final_wm_timing_breakdown,
            "final_eval_wm_timing_counts": final_wm_timing_counts,
            "final_eval_world_model_rollout_time_sec": final_wm_time,
            "total_world_model_rollout_time_sec": total_wm_time,
            "planner_decoder_time_sec": planner_decoder_time,
            "final_decoder_time_sec": final_decoder_time,
            "decoder_time_sec": total_decoder_time,
            "planner_visualization_time_sec": planner_visualization_time,
            "final_visualization_time_sec": final_visualization_time,
            "visualization_time_sec": total_visualization_time,
            "planner_environment_step_time_sec": planner_env_time,
            "final_environment_step_time_sec": final_env_time,
            "environment_step_time_sec": total_env_time,
            "cuda_peak_memory_mb": cuda_peak_mb,
            "cem_candidate_stats": candidate_stats,
            "final_metrics": logs_entry,
        }
        summary.update(roi_stats)
        add_lightweight_defaults(summary)
        self.metrics_logger.write_json("summary.json", summary)


def load_ckpt(snapshot_path, device, load_decoder=True):
    map_location = device if load_decoder else "cpu"
    with snapshot_path.open("rb") as f:
        payload = torch.load(f, map_location=map_location)
    loaded_keys = []
    result = {}
    for k, v in payload.items():
        if k == "decoder" and not load_decoder:
            continue
        if k in ALL_MODEL_KEYS:
            loaded_keys.append(k)
            result[k] = v.to(device)
    result["epoch"] = payload["epoch"]
    return result


def load_model(model_ckpt, train_cfg, num_action_repeat, device, roi_runtime_cfg=None):
    load_decoder = (
        roi_runtime_cfg.get("load_decoder", True)
        if roi_runtime_cfg is not None
        else True
    )
    result = {}
    if model_ckpt.exists():
        result = load_ckpt(model_ckpt, device, load_decoder=load_decoder)
        print(f"Resuming from epoch {result['epoch']}: {model_ckpt}")

    if not load_decoder:
        result["decoder"] = None

    if "encoder" not in result:
        result["encoder"] = hydra.utils.instantiate(
            train_cfg.encoder,
        )
    if "predictor" not in result:
        raise ValueError("Predictor not found in model checkpoint")

    if load_decoder and train_cfg.has_decoder and "decoder" not in result:
        base_path = os.path.dirname(os.path.abspath(__file__))
        if train_cfg.env.decoder_path is not None:
            decoder_path = os.path.join(base_path, train_cfg.env.decoder_path)
            ckpt = torch.load(decoder_path)
            if isinstance(ckpt, dict):
                result["decoder"] = ckpt["decoder"]
            else:
                result["decoder"] = torch.load(decoder_path)
        else:
            raise ValueError(
                "Decoder path not found in model checkpoint \
                                and is not provided in config"
            )
    elif not load_decoder or not train_cfg.has_decoder:
        result["decoder"] = None

    runtime_cfg = roi_runtime_cfg if roi_runtime_cfg is not None else train_cfg
    model = hydra.utils.instantiate(
        train_cfg.model,
        encoder=result["encoder"],
        proprio_encoder=result["proprio_encoder"],
        action_encoder=result["action_encoder"],
        predictor=result["predictor"],
        decoder=result["decoder"],
        proprio_dim=train_cfg.proprio_emb_dim,
        action_dim=train_cfg.action_emb_dim,
        concat_dim=train_cfg.concat_dim,
        num_action_repeat=num_action_repeat,
        num_proprio_repeat=train_cfg.num_proprio_repeat,
        use_roi=runtime_cfg.get(
            "use_drs",
            runtime_cfg.get("drs_enabled", runtime_cfg.get("use_roi", False)),
        ),
        roi_mode=runtime_cfg.get("drs_mode", runtime_cfg.get("roi_mode", "none")),
        roi_config=extract_drs_config_dict(
            runtime_cfg
        ),
        roi_head_checkpoint=runtime_cfg.get(
            "drs_head_checkpoint",
            runtime_cfg.get("roi_head_checkpoint", None),
        ),
        sparse_dynamics_config=extract_sparse_dynamics_config_dict(
            runtime_cfg
        ),
        profile_wm_timing=runtime_cfg.get("profile_wm_timing", True),
        profile_wm_timing_sync_cuda=runtime_cfg.get(
            "profile_wm_timing_sync_cuda", False
        ),
    )
    if "sparse_primary_dynamics" in result and getattr(
        model,
        "sparse_primary_dynamics",
        None,
    ) is not None:
        model.sparse_primary_dynamics = result["sparse_primary_dynamics"]
    model.to(device)
    return model


class DummyWandbRun:
    def __init__(self):
        self.mode = "disabled"

    def log(self, *args, **kwargs):
        pass

    def watch(self, *args, **kwargs):
        pass

    def config(self, *args, **kwargs):
        pass

    def finish(self):
        pass


def planning_main(cfg_dict):
    cfg_dict = normalize_eval_seed_cfg(cfg_dict)
    output_dir = cfg_dict["saved_folder"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    metrics_logger = SafeMetricsLogger(output_dir)
    if cfg_dict["wandb_logging"]:
        wandb_run = wandb.init(
            project=f"plan_{cfg_dict['planner']['name']}", config=cfg_dict
        )
        wandb.run.name = "{}".format(output_dir.split("plan_outputs/")[-1])
    else:
        wandb_run = None

    ckpt_base_path = cfg_dict["ckpt_base_path"]
    model_path = f"{ckpt_base_path}/outputs/{cfg_dict['model_name']}/"
    with open(os.path.join(model_path, "hydra.yaml"), "r") as f:
        model_cfg = OmegaConf.load(f)
    if cfg_dict.get("data_path", None) is not None:
        with open_dict(model_cfg):
            model_cfg.env.dataset.data_path = cfg_dict["data_path"]

    seed(cfg_dict["seed"])
    _, dset = hydra.utils.call(
        model_cfg.env.dataset,
        num_hist=model_cfg.num_hist,
        num_pred=model_cfg.num_pred,
        frameskip=model_cfg.frameskip,
    )
    dset = dset["valid"]

    num_action_repeat = model_cfg.num_action_repeat
    model_ckpt = (
        Path(model_path) / "checkpoints" / f"model_{cfg_dict['model_epoch']}.pth"
    )
    model_runtime_cfg = dict(cfg_dict)
    model_runtime_cfg.update(extract_drs_config_dict(cfg_dict))
    model_runtime_cfg["load_decoder"] = cfg_dict.get("load_decoder", True)
    model_runtime_cfg["profile_wm_timing"] = cfg_dict.get(
        "profile_wm_timing", True
    )
    model_runtime_cfg["profile_wm_timing_sync_cuda"] = cfg_dict.get(
        "profile_wm_timing_sync_cuda", False
    )
    model_runtime_cfg["sparse_dynamics"] = cfg_dict.get(
        "sparse_dynamics",
        model_cfg.get("sparse_dynamics", {}),
    )
    model = load_model(
        model_ckpt,
        model_cfg,
        num_action_repeat,
        device=device,
        roi_runtime_cfg=model_runtime_cfg,
    )
    model.eval()
    model_param_counts = {
        "model": count_parameters(model),
        "encoder": count_parameters(getattr(model, "encoder", None)),
        "predictor": count_parameters(getattr(model, "predictor", None)),
        "decoder": count_parameters(getattr(model, "decoder", None)),
        "proprio_encoder": count_parameters(getattr(model, "proprio_encoder", None)),
        "action_encoder": count_parameters(getattr(model, "action_encoder", None)),
        "drs_selector": count_parameters(getattr(model, "drs_selector", None)),
        "roi_selector": count_parameters(getattr(model, "roi_selector", None)),
        "sparse_dynamic_localizer": count_parameters(
            getattr(model, "sparse_dynamic_localizer", None)
        ),
        "sparse_primary_dynamics": count_parameters(
            getattr(model, "sparse_primary_dynamics", None)
        ),
    }

    # use dummy vector env for wall and deformable envs
    if model_cfg.env.name == "wall" or model_cfg.env.name == "deformable_env":
        from env.serial_vector_env import SerialVectorEnv
        env = SerialVectorEnv(
            [
                gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )
    else:
        env = SubprocVectorEnv(
            [
                lambda: gym.make(
                    model_cfg.env.name, *model_cfg.env.args, **model_cfg.env.kwargs
                )
                for _ in range(cfg_dict["n_evals"])
            ]
        )

    try:
        train_config = OmegaConf.to_container(model_cfg, resolve=True)
    except Exception:
        train_config = str(model_cfg)
    metrics_logger.write_json(
        "run_meta.json",
        build_run_meta(
            kind="plan",
            output_dir=output_dir,
            cfg=cfg_dict,
            extra={
                "device": str(device),
                "model_path": model_path,
                "model_checkpoint": str(model_ckpt),
                "model_params": model_param_counts,
                "train_config": train_config,
                "valid_traj_count": len(dset),
            },
        ),
    )

    plan_workspace = PlanWorkspace(
        cfg_dict=cfg_dict,
        wm=model,
        dset=dset,
        env=env,
        env_name=model_cfg.env.name,
        frameskip=model_cfg.frameskip,
        wandb_run=wandb_run,
        metrics_logger=metrics_logger,
    )

    logs = plan_workspace.perform_planning()
    return logs


@hydra.main(config_path="conf", config_name="plan")
def main(cfg: OmegaConf):
    with open_dict(cfg):
        cfg["saved_folder"] = os.getcwd()
        log.info(f"Planning result saved dir: {cfg['saved_folder']}")
    cfg_dict = cfg_to_dict(cfg)
    cfg_dict = normalize_eval_seed_cfg(cfg_dict)
    cfg_dict["wandb_logging"] = cfg_dict.get("wandb_logging", False)
    planning_main(cfg_dict)


if __name__ == "__main__":
    main()
