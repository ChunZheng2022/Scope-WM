import os
import torch
import imageio
import numpy as np
from einops import rearrange, repeat
from utils import (
    cfg_to_dict,
    seed,
    slice_trajdict_with_t,
    aggregate_dct,
    move_to_device,
    concat_trajdict,
)
from torchvision import utils
from eval_logging import add_lightweight_defaults, Timer, elapsed_since, perf_counter


class PlanEvaluator:  # evaluator for planning
    def __init__(
        self,
        obs_0,
        obs_g,
        state_0,
        state_g,
        env,
        wm,
        frameskip,
        seed,
        preprocessor,
        n_plot_samples,
    ):
        self.obs_0 = obs_0
        self.obs_g = obs_g
        self.state_0 = state_0
        self.state_g = state_g
        self.env = env
        self.wm = wm
        self.frameskip = frameskip
        self.seed = seed
        self.preprocessor = preprocessor
        self.n_plot_samples = n_plot_samples
        self.device = next(wm.parameters()).device

        self.plot_full = False  # plot all frames or frames after frameskip
        self.last_eval_info = None

    def assign_init_cond(self, obs_0, state_0):
        self.obs_0 = obs_0
        self.state_0 = state_0

    def assign_goal_cond(self, obs_g, state_g):
        self.obs_g = obs_g
        self.state_g = state_g

    def get_init_cond(self):
        return self.obs_0, self.state_0

    def _get_trajdict_last(self, dct, length):
        new_dct = {}
        for key, value in dct.items():
            new_dct[key] = self._get_traj_last(value, length)
        return new_dct

    def _get_traj_last(self, traj_data, length):
        last_index = np.where(length == np.inf, -1, length - 1)
        last_index = last_index.astype(int)
        if isinstance(traj_data, torch.Tensor):
            traj_data = traj_data[np.arange(traj_data.shape[0]), last_index].unsqueeze(
                1
            )
        else:
            traj_data = np.expand_dims(
                traj_data[np.arange(traj_data.shape[0]), last_index], axis=1
            )
        return traj_data

    def _mask_traj(self, data, length):
        """
        Zero out everything after specified indices for each trajectory in the tensor.
        data: tensor
        """
        result = data.clone()  # Clone to preserve the original tensor
        for i in range(data.shape[0]):
            if length[i] != np.inf:
                result[i, int(length[i]) :] = 0
        return result

    def eval_actions(
        self, actions, action_len=None, filename="output", save_video=False
    ):
        """
        actions: detached torch tensors on cuda
        Returns
            metrics, and feedback from env
        """
        total_start = perf_counter()
        n_evals = actions.shape[0]
        if action_len is None:
            action_len = np.full(n_evals, np.inf)
        decoder_time_sec = 0.0
        visualization_time_sec = 0.0
        preprocess_time_sec = 0.0
        metric_compute_time_sec = 0.0
        # rollout in wm
        with Timer() as preprocess_timer:
            trans_obs_0 = move_to_device(
                self.preprocessor.transform_obs(self.obs_0), self.device
            )
            trans_obs_g = move_to_device(
                self.preprocessor.transform_obs(self.obs_g), self.device
            )
        preprocess_time_sec = preprocess_timer.elapsed
        with Timer() as wm_rollout_timer:
            with torch.no_grad():
                i_z_obses, _ = self.wm.rollout(
                    obs_0=trans_obs_0,
                    act=actions,
                )
        wm_timing_stats = self._get_wm_timing_stats()
        roi_stats = self._get_wm_roi_stats()
        roi_stats = self._add_token_compute_stats(roi_stats, i_z_obses, actions)
        i_final_z_obs = self._get_trajdict_last(i_z_obses, action_len + 1)

        # rollout in env
        exec_actions = rearrange(
            actions.cpu(), "b t (f d) -> b (t f) d", f=self.frameskip
        )
        exec_actions = self.preprocessor.denormalize_actions(exec_actions).numpy()
        with Timer() as env_step_timer:
            e_obses, e_states = self.env.rollout(
                self.seed, self.state_0, exec_actions
            )
        e_visuals = e_obses["visual"]
        e_final_obs = self._get_trajdict_last(e_obses, action_len * self.frameskip + 1)
        e_final_state = self._get_traj_last(e_states, action_len * self.frameskip + 1)[
            :, 0
        ]  # reduce dim back

        # compute eval metrics
        with Timer() as metric_timer:
            logs, successes, eval_results, state_dists = self._compute_rollout_metrics(
                e_state=e_final_state,
                e_obs=e_final_obs,
                i_z_obs=i_final_z_obs,
            )
        metric_compute_time_sec = metric_timer.elapsed
        logs.update(roi_stats)

        # Plot only the requested subset. Official checkpoints may omit the
        # decoder, and decoding all eval rollouts is very memory-hungry.
        plot_count = min(int(self.n_plot_samples or 0), n_evals)
        decoder = getattr(self.wm, "decoder", None)
        can_plot_rollouts = decoder is not None and plot_count > 0
        logs["decoder_available"] = decoder is not None
        logs["num_plot_samples"] = plot_count

        if can_plot_rollouts:
            i_z_obses_plot = {
                key: value[:plot_count] for key, value in i_z_obses.items()
            }
            action_len_plot = action_len[:plot_count]
            with Timer() as decoder_timer:
                i_visuals = self.wm.decode_obs(i_z_obses_plot)[0]["visual"]
            decoder_time_sec = decoder_timer.elapsed
            i_visuals = self._mask_traj(
                i_visuals, action_len_plot + 1
            )  # we have action_len + 1 states
            e_visuals_plot = self.preprocessor.transform_obs_visual(
                e_visuals[:plot_count]
            )
            e_visuals_plot = self._mask_traj(
                e_visuals_plot, action_len_plot * self.frameskip + 1
            )
            with Timer() as visualization_timer:
                self._plot_rollout_compare(
                    e_visuals=e_visuals_plot,
                    i_visuals=i_visuals,
                    successes=successes[:plot_count],
                    save_video=save_video,
                    filename=filename,
                )
            visualization_time_sec = visualization_timer.elapsed

        timings = {
            "eval_total_time_sec": elapsed_since(total_start),
            "preprocess_time_sec": preprocess_time_sec,
            "world_model_rollout_time_sec": wm_rollout_timer.elapsed,
            "environment_step_time_sec": env_step_timer.elapsed,
            "metric_compute_time_sec": metric_compute_time_sec,
            "decoder_time_sec": decoder_time_sec,
            "visualization_time_sec": visualization_time_sec,
            "wm_timing_breakdown": wm_timing_stats.get("times_sec", {}),
            "wm_timing_counts": wm_timing_stats.get("counts", {}),
        }
        self.last_eval_info = {
            "filename": filename,
            "metrics": logs,
            "successes": successes,
            "eval_results": eval_results,
            "state_distances": state_dists,
            "action_len": action_len,
            "timings": timings,
            "roi_stats": roi_stats,
            "per_eval": self._build_per_eval_results(
                successes=successes,
                eval_results=eval_results,
                state_dists=state_dists,
                action_len=action_len,
                filename=filename,
                timings=timings,
                roi_stats=roi_stats,
            ),
        }
        return logs, successes, e_obses, e_states

    def _get_wm_roi_stats(self):
        getter = getattr(self.wm, "get_last_roi_stats", None)
        if getter is None:
            return {}
        try:
            return getter()
        except Exception:
            return {}

    def _get_wm_timing_stats(self):
        getter = getattr(self.wm, "get_timing_stats", None)
        if getter is None:
            return {"times_sec": {}, "counts": {}}
        try:
            return getter() or {"times_sec": {}, "counts": {}}
        except Exception:
            return {"times_sec": {}, "counts": {}}

    def _add_token_compute_stats(self, roi_stats, z_obses, actions):
        stats = dict(roi_stats or {})
        try:
            visual_token_count = int(z_obses["visual"].shape[2])
        except Exception:
            visual_token_count = None

        total_tokens = stats.get("drs_num_tokens") or stats.get("roi_num_tokens") or visual_token_count
        kept_tokens = (
            stats.get("drs_num_kept_tokens")
            or stats.get("roi_num_kept_tokens")
            or stats.get("effective_token_count")
            or visual_token_count
        )
        if total_tokens is not None and kept_tokens is not None and total_tokens > 0:
            keep_ratio = kept_tokens / total_tokens
            stats.setdefault("drs_keep_ratio", keep_ratio)
            stats.setdefault("roi_keep_ratio", keep_ratio)
            stats.setdefault("drs_num_tokens", total_tokens)
            stats.setdefault("drs_num_kept_tokens", kept_tokens)
            stats.setdefault("effective_token_count", kept_tokens)
            stats.setdefault("visual_token_count", total_tokens)
            stats.setdefault("kept_visual_token_count", kept_tokens)
            stats.setdefault("token_reduction_ratio", 1.0 - keep_ratio)
            stats.setdefault("attention_compute_ratio", keep_ratio ** 2)
            stats.setdefault("attention_compute_reduction", 1.0 - keep_ratio ** 2)

            try:
                wm_steps = int(actions.shape[0] * actions.shape[1] * kept_tokens)
                stats.setdefault("wm_rollout_token_steps", wm_steps)
            except Exception:
                pass
        elif visual_token_count is not None:
            stats.setdefault("visual_token_count", visual_token_count)
            stats.setdefault("effective_token_count", visual_token_count)
            stats.setdefault("kept_visual_token_count", visual_token_count)

        return stats

    def _batch_l2(self, a, b):
        try:
            diff = np.asarray(a) - np.asarray(b)
            diff = diff.reshape(diff.shape[0], -1)
            return np.linalg.norm(diff, axis=1)
        except Exception:
            return None

    def _build_per_eval_results(
        self,
        successes,
        eval_results,
        state_dists,
        action_len,
        filename,
        timings,
        roi_stats=None,
    ):
        per_eval = []
        n_evals = len(successes)
        for idx in range(n_evals):
            seed_value = self.seed[idx] if isinstance(self.seed, list) else self.seed
            result = {
                "eval_id": idx,
                "eval_seed": seed_value,
                "filename": filename,
                "success": bool(successes[idx]),
                "final_state_distance": (
                    state_dists[idx].item() if state_dists is not None else None
                ),
                "action_len": None if action_len[idx] == np.inf else action_len[idx],
                "total_episode_time_sec": timings["eval_total_time_sec"],
                "preprocess_time_sec": timings["preprocess_time_sec"],
                "world_model_rollout_time_sec": timings[
                    "world_model_rollout_time_sec"
                ],
                "metric_compute_time_sec": timings["metric_compute_time_sec"],
                "decoder_time_sec": timings["decoder_time_sec"],
                "visualization_time_sec": timings["visualization_time_sec"],
                "environment_step_time_sec": timings["environment_step_time_sec"],
            }
            for key, value in eval_results.items():
                if key == "success":
                    continue
                try:
                    result[f"task_{key}"] = value[idx]
                except Exception:
                    continue
            if roi_stats:
                result.update(roi_stats)
            add_lightweight_defaults(result)
            per_eval.append(result)
        return per_eval

    def _compute_rollout_metrics(self, e_state, e_obs, i_z_obs):
        """
        Args
            e_state
            e_obs
            i_z_obs
        Return
            logs
            successes
        """
        eval_results = self.env.eval_state(self.state_g, e_state)
        successes = eval_results['success']

        logs = {
            f"success_rate" if key == "success" else f"mean_{key}": np.mean(value) if key != "success" else np.mean(value.astype(float))
            for key, value in eval_results.items()
        }

        print("Success rate: ", logs['success_rate'])
        print(eval_results)

        state_dists = self._batch_l2(e_state, self.state_g)
        visual_dists = self._batch_l2(e_obs["visual"], self.obs_g["visual"])
        proprio_dists = self._batch_l2(e_obs["proprio"], self.obs_g["proprio"])
        mean_state_dist = np.mean(state_dists) if state_dists is not None else None
        mean_visual_dist = np.mean(visual_dists) if visual_dists is not None else None
        mean_proprio_dist = np.mean(proprio_dists) if proprio_dists is not None else None

        e_obs = move_to_device(self.preprocessor.transform_obs(e_obs), self.device)
        e_z_obs = self.wm.encode_obs(e_obs)
        div_visual_emb = torch.norm(e_z_obs["visual"] - i_z_obs["visual"]).item()
        div_proprio_emb = torch.norm(e_z_obs["proprio"] - i_z_obs["proprio"]).item()

        logs.update({
            "mean_state_dist": mean_state_dist,
            "mean_visual_dist": mean_visual_dist,
            "mean_proprio_dist": mean_proprio_dist,
            "mean_div_visual_emb": div_visual_emb,
            "mean_div_proprio_emb": div_proprio_emb,
        })

        return logs, successes, eval_results, state_dists

    def _plot_rollout_compare(
        self, e_visuals, i_visuals, successes, save_video=False, filename=""
    ):
        """
        i_visuals may have less frames than e_visuals due to frameskip, so pad accordingly
        e_visuals: (b, t, h, w, c)
        i_visuals: (b, t, h, w, c)
        goal: (b, h, w, c)
        """
        e_visuals = e_visuals[: self.n_plot_samples]
        i_visuals = i_visuals[: self.n_plot_samples]
        goal_visual = self.obs_g["visual"][: self.n_plot_samples]
        goal_visual = self.preprocessor.transform_obs_visual(goal_visual)

        i_visuals = i_visuals.unsqueeze(2)
        i_visuals = torch.cat(
            [i_visuals] + [i_visuals] * (self.frameskip - 1),
            dim=2,
        )  # pad i_visuals (due to frameskip)
        i_visuals = rearrange(i_visuals, "b t n c h w -> b (t n) c h w")
        i_visuals = i_visuals[:, : i_visuals.shape[1] - (self.frameskip - 1)]

        correction = 0.3  # to distinguish env visuals and imagined visuals

        if save_video:
            for idx in range(e_visuals.shape[0]):
                success_tag = "success" if successes[idx] else "failure"
                frames = []
                for i in range(e_visuals.shape[1]):
                    e_obs = e_visuals[idx, i, ...]
                    i_obs = i_visuals[idx, i, ...]
                    e_obs = torch.cat(
                        [e_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
                    )
                    i_obs = torch.cat(
                        [i_obs.cpu(), goal_visual[idx, 0] - correction], dim=2
                    )
                    frame = torch.cat([e_obs - correction, i_obs], dim=1)
                    frame = rearrange(frame, "c w1 w2 -> w1 w2 c")
                    frame = rearrange(frame, "w1 w2 c -> (w1) w2 c")
                    frame = frame.detach().cpu().numpy()
                    frames.append(frame)
                video_writer = imageio.get_writer(
                    f"{filename}_{idx}_{success_tag}.mp4", fps=12
                )

                for frame in frames:
                    frame = frame * 2 - 1 if frame.min() >= 0 else frame
                    video_writer.append_data(
                        (((np.clip(frame, -1, 1) + 1) / 2) * 255).astype(np.uint8)
                    )
                video_writer.close()

        # pad i_visuals or subsample e_visuals
        if not self.plot_full:
            e_visuals = e_visuals[:, :: self.frameskip]
            i_visuals = i_visuals[:, :: self.frameskip]

        n_columns = e_visuals.shape[1]
        assert (
            i_visuals.shape[1] == n_columns
        ), f"Rollout lengths do not match, {e_visuals.shape[1]} and {i_visuals.shape[1]}"

        # add a goal column
        e_visuals = torch.cat([e_visuals.cpu(), goal_visual - correction], dim=1)
        i_visuals = torch.cat([i_visuals.cpu(), goal_visual - correction], dim=1)
        rollout = torch.cat([e_visuals.cpu() - correction, i_visuals.cpu()], dim=1)
        n_columns += 1

        imgs_for_plotting = rearrange(rollout, "b h c w1 w2 -> (b h) c w1 w2")
        imgs_for_plotting = (
            imgs_for_plotting * 2 - 1
            if imgs_for_plotting.min() >= 0
            else imgs_for_plotting
        )
        utils.save_image(
            imgs_for_plotting,
            f"{filename}.png",
            nrow=n_columns,  # nrow is the number of columns
            normalize=True,
            value_range=(-1, 1),
        )
