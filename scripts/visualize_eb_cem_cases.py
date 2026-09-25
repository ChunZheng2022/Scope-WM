import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from einops import rearrange
from PIL import Image, ImageDraw, ImageFont

from datasets.pusht_dset import ACTION_MEAN, ACTION_STD
from env.pusht.pusht_wrapper import PushTWrapper


def _read_jsonl(path: Path) -> List[Dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_plan_dir(plan_dir: Path) -> Dict:
    targets_path = plan_dir / "plan_targets.pkl"
    actions_path = plan_dir / "planned_actions.pt"
    if not targets_path.exists():
        raise FileNotFoundError(f"Missing plan_targets.pkl in {plan_dir}")
    if not actions_path.exists():
        raise FileNotFoundError(f"Missing planned_actions.pt in {plan_dir}")
    with targets_path.open("rb") as f:
        targets = pickle.load(f)
    actions_payload = torch.load(actions_path, map_location="cpu")
    return {
        "dir": plan_dir,
        "targets": targets,
        "actions": actions_payload["actions"],
        "action_len": np.asarray(actions_payload.get("action_len", [])),
        "successes": np.asarray(actions_payload.get("successes", [])),
        "per_eval": _read_jsonl(plan_dir / "per_eval_results.jsonl"),
        "summary": _read_json_if_exists(plan_dir / "summary.json"),
    }


def _read_json_if_exists(path: Path) -> Dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _records_by_eval_id(rows: List[Dict]) -> Dict[int, Dict]:
    return {int(row.get("eval_id", idx)): row for idx, row in enumerate(rows)}


def _record_success(record: Optional[Dict], fallback_successes: np.ndarray, idx: int) -> bool:
    if record is not None and "success" in record:
        return bool(record["success"])
    if len(fallback_successes) > idx:
        return bool(fallback_successes[idx])
    return False


def _record_dist(record: Optional[Dict]) -> float:
    if record is None:
        return float("inf")
    for key in ["final_state_distance", "task_state_dist"]:
        value = record.get(key)
        if value is not None:
            return float(value)
    return float("inf")


def _select_cases(base: Dict, eb: Dict, num_cases: int, mode: str = "success_gap") -> List[int]:
    base_records = _records_by_eval_id(base["per_eval"])
    eb_records = _records_by_eval_id(eb["per_eval"])
    n = min(
        len(base["actions"]),
        len(eb["actions"]),
        len(base["targets"]["state_0"]),
        len(eb["targets"]["state_0"]),
    )
    scored = []
    for idx in range(n):
        b_rec = base_records.get(idx)
        e_rec = eb_records.get(idx)
        b_success = _record_success(b_rec, base["successes"], idx)
        e_success = _record_success(e_rec, eb["successes"], idx)
        b_dist = _record_dist(b_rec)
        e_dist = _record_dist(e_rec)
        if mode == "success_gap" and not (e_success and not b_success):
            continue
        if mode == "baseline_fail" and b_success:
            continue
        first_success_bonus = 0.0
        if e_success and not b_success:
            first_success_bonus = 10000.0
        elif e_success == b_success:
            first_success_bonus = 0.0
        else:
            first_success_bonus = -10000.0
        improvement = b_dist - e_dist
        scored.append((first_success_bonus + improvement, idx))
    scored.sort(reverse=True)
    return [idx for _, idx in scored[:num_cases]]


def _max_target_diff(base: Dict, eb: Dict) -> float:
    diffs = []
    for key in ["state_0", "state_g"]:
        if key in base["targets"] and key in eb["targets"]:
            a = np.asarray(base["targets"][key])
            b = np.asarray(eb["targets"][key])
            if a.shape == b.shape:
                diffs.append(float(np.max(np.abs(a - b))))
    return max(diffs) if diffs else float("inf")


def _denormalize_pusht_actions(actions: torch.Tensor, frameskip: int) -> np.ndarray:
    if actions.ndim == 2:
        actions = actions.unsqueeze(0)
    exec_actions = rearrange(actions.cpu(), "b t (f d) -> b (t f) d", f=frameskip)
    mean = ACTION_MEAN.view(1, 1, -1)
    std = ACTION_STD.view(1, 1, -1)
    return (exec_actions * std + mean).numpy()[0]


def _finite_action_len(action_len, max_steps: int) -> int:
    try:
        if np.isfinite(action_len):
            return max(0, min(max_steps, int(action_len)))
    except Exception:
        pass
    return max_steps


def _rollout_pusht(
    seed: int,
    init_state: np.ndarray,
    normalized_actions: torch.Tensor,
    action_len,
    frameskip: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    wm_steps = _finite_action_len(action_len, int(normalized_actions.shape[0]))
    exec_actions = _denormalize_pusht_actions(normalized_actions[:wm_steps], frameskip)
    env = PushTWrapper(with_velocity=True, with_target=True)
    obs0, state0 = env.prepare(seed, np.asarray(init_state, dtype=np.float32).copy())
    states = [state0]
    images = [obs0["visual"]]
    for action in exec_actions:
        obs, _, _, info = env.step(action)
        states.append(info["state"])
        images.append(obs["visual"])
    final_image = images[-1]
    env.close()
    return final_image, np.asarray(states), exec_actions


def _rollout_pusht_sequence(
    seed: int,
    init_state: np.ndarray,
    normalized_actions: torch.Tensor,
    wm_steps: int,
    frameskip: int,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    wm_steps = max(0, min(int(wm_steps), int(normalized_actions.shape[0])))
    exec_actions = _denormalize_pusht_actions(normalized_actions[:wm_steps], frameskip)
    env = PushTWrapper(with_velocity=True, with_target=True)
    obs0, state0 = env.prepare(seed, np.asarray(init_state, dtype=np.float32).copy())
    states = [state0]
    images = [obs0["visual"]]
    for action in exec_actions:
        obs, _, _, info = env.step(action)
        states.append(info["state"])
        images.append(obs["visual"])
    env.close()
    return images, np.asarray(states), exec_actions


def _render_state(seed: int, state: np.ndarray) -> np.ndarray:
    env = PushTWrapper(with_velocity=True, with_target=True)
    obs, _ = env.prepare(seed, np.asarray(state, dtype=np.float32).copy())
    image = obs["visual"]
    env.close()
    return image


def _draw_path(image: np.ndarray, states: np.ndarray, color: Tuple[int, int, int]) -> Image.Image:
    out = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(out, "RGBA")
    if states.size > 0:
        scale = out.width / 512.0
        pts = [(float(s[0]) * scale, float(s[1]) * scale) for s in states]
        if len(pts) > 1:
            draw.line(pts, fill=(*color, 220), width=3)
        for pt in pts[:: max(1, len(pts) // 10)]:
            r = 2.2
            draw.ellipse([pt[0] - r, pt[1] - r, pt[0] + r, pt[1] + r], fill=(*color, 210))
    return out


def _make_case_figure(
    case_idx: int,
    seed: int,
    init_image: np.ndarray,
    goal_image: np.ndarray,
    base_image: np.ndarray,
    eb_image: np.ndarray,
    base_states: np.ndarray,
    eb_states: np.ndarray,
    base_record: Optional[Dict],
    eb_record: Optional[Dict],
    output_path: Path,
):
    font = ImageFont.load_default()
    panels = [
        ("initial", Image.fromarray(init_image).convert("RGB")),
        ("goal", Image.fromarray(goal_image).convert("RGB")),
        ("baseline final", _draw_path(base_image, base_states, (235, 80, 60))),
        ("EB-CEM final", _draw_path(eb_image, eb_states, (45, 210, 95))),
    ]
    panel_w, panel_h = panels[0][1].size
    gap = 18
    label_h = 36
    canvas = Image.new(
        "RGB",
        (4 * panel_w + 5 * gap, panel_h + label_h + 28),
        (250, 250, 250),
    )
    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 10), f"eval_id={case_idx} seed={seed}", fill=(20, 20, 20), font=font)

    for col, (title, panel) in enumerate(panels):
        x = gap + col * (panel_w + gap)
        y = 28
        canvas.paste(panel, (x, y + label_h))
        draw.text((x, y), title, fill=(20, 20, 20), font=font)

    def _status(record: Optional[Dict]) -> str:
        if record is None:
            return "success=? dist=?"
        return f"success={bool(record.get('success'))} dist={float(record.get('final_state_distance', float('nan'))):.1f}"

    draw.text((gap + 2 * (panel_w + gap), 28 + 14), _status(base_record), fill=(180, 30, 20), font=font)
    draw.text((gap + 3 * (panel_w + gap), 28 + 14), _status(eb_record), fill=(0, 135, 45), font=font)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path)


def _state_xy_points(states: np.ndarray, width: int) -> List[Tuple[float, float]]:
    scale = width / 512.0
    return [(float(s[0]) * scale, float(s[1]) * scale) for s in states]


def _draw_state_path_on_frame(
    image: np.ndarray,
    states: np.ndarray,
    color: Tuple[int, int, int],
) -> Image.Image:
    out = Image.fromarray(image).convert("RGB")
    draw = ImageDraw.Draw(out, "RGBA")
    pts = _state_xy_points(states, out.width)
    if len(pts) > 1:
        draw.line(pts, fill=(*color, 210), width=3)
    if pts:
        r = 3
        draw.ellipse([pts[-1][0] - r, pts[-1][1] - r, pts[-1][0] + r, pts[-1][1] + r], fill=(*color, 230))
    return out


def _save_sequence_frames(
    output_dir: Path,
    prefix: str,
    images: List[np.ndarray],
    states: np.ndarray,
    frameskip: int,
    draw_path: bool,
    color: Tuple[int, int, int],
) -> List[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    max_wm_step = max(0, (len(images) - 1) // frameskip)
    for step in range(max_wm_step + 1):
        low_level_idx = min(step * frameskip, len(images) - 1)
        frame = images[low_level_idx]
        if draw_path:
            state_prefix = states[: low_level_idx + 1]
            out = _draw_state_path_on_frame(frame, state_prefix, color)
        else:
            out = Image.fromarray(frame).convert("RGB")
        path = output_dir / f"{prefix}_step{step:03d}.png"
        out.save(path)
        saved.append(str(path))
    return saved


def _save_reference_frames(
    output_dir: Path,
    seed: int,
    init_state: np.ndarray,
    goal_state: np.ndarray,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    init_path = output_dir / "initial.png"
    goal_path = output_dir / "goal.png"
    Image.fromarray(_render_state(seed, init_state)).convert("RGB").save(init_path)
    Image.fromarray(_render_state(seed, goal_state)).convert("RGB").save(goal_path)
    return {"initial": str(init_path), "goal": str(goal_path)}


def build_argparser():
    parser = argparse.ArgumentParser(
        description="Visualize hard PushT cases where EB-CEM improves over a baseline plan run."
    )
    parser.add_argument(
        "--baseline-dir",
        "--baseline-plan-dir",
        required=True,
        help="Plan output directory for the comparison planner run, e.g. original CEM with 100 initial samples.",
    )
    parser.add_argument(
        "--eb-dir",
        "--eb-plan-dir",
        required=True,
        help="Plan output directory for the EB-CEM/frontloaded run, e.g. 300 initial samples then 100.",
    )
    parser.add_argument("--output-dir", default="viz_eb_cem_cases")
    parser.add_argument("--num-cases", type=int, default=6)
    parser.add_argument(
        "--case-mode",
        choices=["success_gap", "baseline_fail", "improvement"],
        default="success_gap",
        help=(
            "success_gap: only EB succeeds and baseline fails; "
            "baseline_fail: baseline fails, sorted by EB improvement; "
            "improvement: fill by final distance improvement even if success labels match."
        ),
    )
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--allow-target-mismatch", action="store_true")
    parser.add_argument(
        "--export-sequences",
        action="store_true",
        help="Export per-step single images for the selected baseline-vs-EB trajectories.",
    )
    parser.add_argument(
        "--sequence-stop",
        choices=["eb_pre_success", "eb_success", "baseline_full", "full"],
        default="eb_pre_success",
        help=(
            "How far to roll out sequence panels. eb_pre_success saves both runs "
            "through the step before EB first succeeds; baseline_full saves the "
            "complete baseline failure trajectory and EB through first success."
        ),
    )
    parser.add_argument(
        "--draw-path",
        action="store_true",
        help="Overlay pusher trajectory path on exported sequence frames.",
    )
    return parser


def main():
    args = build_argparser().parse_args()
    base = _load_plan_dir(Path(args.baseline_dir))
    eb = _load_plan_dir(Path(args.eb_dir))
    target_diff = _max_target_diff(base, eb)
    if target_diff > 1e-5 and not args.allow_target_mismatch:
        raise ValueError(
            "baseline-dir and eb-dir do not share the same plan targets "
            f"(max abs target diff={target_diff:g}). Re-run with the same "
            "goal_file_path, or pass --allow-target-mismatch if this is intentional."
        )
    case_ids = _select_cases(base, eb, args.num_cases, mode=args.case_mode)
    base_records = _records_by_eval_id(base["per_eval"])
    eb_records = _records_by_eval_id(eb["per_eval"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = []
    for idx in case_ids:
        init_state = np.asarray(eb["targets"]["state_0"][idx]).reshape(-1)
        goal_state = np.asarray(eb["targets"]["state_g"][idx]).reshape(-1)
        eb_rec = eb_records.get(idx)
        base_rec = base_records.get(idx)
        seed = int((eb_rec or base_rec or {}).get("eval_seed", 1))
        init_image = _render_state(seed, init_state)
        goal_image = _render_state(seed, goal_state)

        base_final, base_states, _ = _rollout_pusht(
            seed,
            init_state,
            base["actions"][idx],
            base["action_len"][idx] if len(base["action_len"]) > idx else np.inf,
            frameskip=args.frameskip,
        )
        eb_final, eb_states, _ = _rollout_pusht(
            seed,
            init_state,
            eb["actions"][idx],
            eb["action_len"][idx] if len(eb["action_len"]) > idx else np.inf,
            frameskip=args.frameskip,
        )
        output_path = output_dir / f"ebcem_case_eval{idx:03d}.png"
        _make_case_figure(
            idx,
            seed,
            init_image,
            goal_image,
            base_final,
            eb_final,
            base_states,
            eb_states,
            base_rec,
            eb_rec,
            output_path,
        )
        sequence_summary = None
        if args.export_sequences:
            base_full_wm_steps = int(base["actions"][idx].shape[0])
            eb_full_wm_steps = int(eb["actions"][idx].shape[0])
            eb_wm_len = _finite_action_len(
                eb["action_len"][idx] if len(eb["action_len"]) > idx else np.inf,
                eb_full_wm_steps,
            )
            if args.sequence_stop == "eb_pre_success":
                base_sequence_wm_steps = max(0, eb_wm_len - 1)
                eb_sequence_wm_steps = max(0, eb_wm_len - 1)
            elif args.sequence_stop == "eb_success":
                base_sequence_wm_steps = eb_wm_len
                eb_sequence_wm_steps = eb_wm_len
            elif args.sequence_stop == "baseline_full":
                base_sequence_wm_steps = base_full_wm_steps
                eb_sequence_wm_steps = eb_wm_len
            else:
                sequence_wm_steps = min(base_full_wm_steps, eb_full_wm_steps)
                base_sequence_wm_steps = sequence_wm_steps
                eb_sequence_wm_steps = sequence_wm_steps
            sequence_dir = output_dir / f"case_eval{idx:03d}_seed{seed}_sequence"
            refs = _save_reference_frames(sequence_dir, seed, init_state, goal_state)
            base_images, base_seq_states, _ = _rollout_pusht_sequence(
                seed,
                init_state,
                base["actions"][idx],
                base_sequence_wm_steps,
                frameskip=args.frameskip,
            )
            eb_images, eb_seq_states, _ = _rollout_pusht_sequence(
                seed,
                init_state,
                eb["actions"][idx],
                eb_sequence_wm_steps,
                frameskip=args.frameskip,
            )
            base_frames = _save_sequence_frames(
                sequence_dir,
                "baseline100",
                base_images,
                base_seq_states,
                args.frameskip,
                args.draw_path,
                (235, 80, 60),
            )
            eb_frames = _save_sequence_frames(
                sequence_dir,
                "eb300",
                eb_images,
                eb_seq_states,
                args.frameskip,
                args.draw_path,
                (45, 210, 95),
            )
            sequence_summary = {
                "sequence_dir": str(sequence_dir),
                "sequence_stop": args.sequence_stop,
                "baseline100_sequence_wm_steps": int(base_sequence_wm_steps),
                "eb300_sequence_wm_steps": int(eb_sequence_wm_steps),
                "eb_first_success_wm_step": int(eb_wm_len),
                "reference_frames": refs,
                "baseline100_frames": base_frames,
                "eb300_frames": eb_frames,
            }
        summary.append(
            {
                "eval_id": idx,
                "eval_seed": seed,
                "baseline_success": None if base_rec is None else bool(base_rec.get("success")),
                "eb_success": None if eb_rec is None else bool(eb_rec.get("success")),
                "baseline_final_state_distance": None if base_rec is None else base_rec.get("final_state_distance"),
                "eb_final_state_distance": None if eb_rec is None else eb_rec.get("final_state_distance"),
                "image": str(output_path),
                "sequence": sequence_summary,
            }
        )
        print(f"Saved EB-CEM case visualization: {output_path}")
        if sequence_summary is not None:
            print(f"Saved sequence frames: {sequence_summary['sequence_dir']}")

    with (output_dir / "selected_cases.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved selected case summary: {output_dir / 'selected_cases.json'}")


if __name__ == "__main__":
    main()
