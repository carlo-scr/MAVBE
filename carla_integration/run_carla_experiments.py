#!/usr/bin/env python3
"""
End-to-end CARLA experiment pipeline for Simplex-Track validation.

Three-stage pipeline:
  Stage 1 — Record 100 CARLA scenarios (50 sfekf + 50 sfekf_simplex)
  Stage 2 — Run YOLOv9 + DeepSORT tracking on all recorded videos
  Stage 3 — Evaluate tracking vs GT, compute metrics, run validation

Each stage can be run independently via --stage flag, or all at once.

Usage:
    # Full pipeline (requires CARLA server)
    python -m carla_integration.run_carla_experiments

    # Record only (CARLA server required)
    python -m carla_integration.run_carla_experiments --stage record

    # Track only (no CARLA needed)
    python -m carla_integration.run_carla_experiments --stage track

    # Evaluate + validate only (no CARLA needed)
    python -m carla_integration.run_carla_experiments --stage evaluate

    # Quick test run (5+5 scenarios)
    python -m carla_integration.run_carla_experiments --quick
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

# ── Path setup ────────────────────────────────────────────────────────────────
FILE = Path(__file__).resolve()
REPO_ROOT = FILE.parents[1]
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = REPO_ROOT / "runs" / "carla_experiments"
WEIGHTS_PATH = REPO_ROOT / "perception" / "yolov9" / "weights" / "yolov9-c.pt"

# Camera intrinsics (must match run_scenarios.py: 1920×1080, FOV=90°)
IMG_W, IMG_H = 1920, 1080
FOV_DEG = 90.0
F_Y = IMG_H / (2.0 * math.tan(math.radians(FOV_DEG / 2.0)))  # ≈ 540 px
PEDESTRIAN_HEIGHT_M = 1.7  # average adult height


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 1: Record CARLA scenarios
# ═══════════════════════════════════════════════════════════════════════════════

def stage_record(output_dir: Path, n_per_tracker: int = 50,
                 n_ped: int = 5, seed: int = 42, v_ego: float = 10.0,
                 tau_safe: float = 2.0, disturbance: str = "fuzzed",
                 host: str = "127.0.0.1", port: int = 2000):
    """Record CARLA scenarios for both tracker configs."""
    from carla_integration.run_scenarios import (
        FUZZED_DISTURBANCE,
        NOMINAL_DISTURBANCE,
        generate_scenario_specs,
        run_single_scenario,
        _save_results,
    )
    import carla

    logger.info("=" * 60)
    logger.info("STAGE 1: Recording %d scenarios per tracker", n_per_tracker)
    logger.info("=" * 60)

    client = carla.Client(host, port)
    client.set_timeout(20.0)
    logger.info("Connected to CARLA %s at %s:%d",
                client.get_server_version(), host, port)

    profile = NOMINAL_DISTURBANCE if disturbance == "nominal" else FUZZED_DISTURBANCE

    for tracker in ["sfekf", "sfekf_simplex"]:
        tracker_dir = output_dir / tracker
        tracker_dir.mkdir(parents=True, exist_ok=True)

        # Check for existing results (resume support)
        results_path = tracker_dir / "scenario_results.json"
        existing = []
        start_idx = 0
        if results_path.exists():
            with open(results_path) as f:
                existing = json.load(f)
            start_idx = len(existing)
            logger.info("Resuming %s from scenario %d", tracker, start_idx)

        specs = generate_scenario_specs(
            n_scenarios=n_per_tracker,
            n_ped=n_ped,
            tracker=tracker,
            v_ego_kmh=v_ego,
            tau_safe=tau_safe,
            base_seed=seed,
            disturbance=profile,
        )

        # Save specs
        specs_path = tracker_dir / "scenario_specs.json"
        with open(specs_path, "w") as f:
            json.dump([asdict(s) for s in specs], f, indent=2)

        all_results = existing
        for i, spec in enumerate(specs):
            if i < start_idx:
                continue

            logger.info("═══ %s scenario %d/%d (seed=%d) ═══",
                        tracker, i + 1, n_per_tracker, spec.seed)

            result = run_single_scenario(
                spec=spec,
                client=client,
                output_dir=tracker_dir,
                save_video=True,
            )

            result_dict = {
                "scenario_id": result.scenario_id,
                "collision": result.collision,
                "collision_time": result.collision_time,
                "min_ttc": result.min_ttc if result.min_ttc != float("inf") else None,
                "rho_min": result.rho_min if result.rho_min != float("inf") else None,
                "ttc_trace": result.ttc_trace,
                "simplex_activations": result.simplex_activations,
                "duration_s": round(result.duration_s, 1),
                "n_ped_spawned": len(result.ped_traces),
                "spec": result.spec,
                "ego_trace": result.ego_trace,
                "ped_traces": result.ped_traces,
            }
            all_results.append(result_dict)

            # Save incrementally
            with open(results_path, "w") as f:
                json.dump(all_results, f, indent=2, default=str)

            status = "COLLISION" if result.collision else "Safe"
            logger.info("  %s (min_ttc=%.2f, simplex=%d)",
                        status,
                        result.min_ttc if result.min_ttc != float("inf") else -1,
                        result.simplex_activations)

        n_coll = sum(1 for r in all_results if r["collision"])
        logger.info("%s complete: %d/%d collisions (%.1f%%)",
                    tracker, n_coll, len(all_results),
                    100 * n_coll / max(len(all_results), 1))


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 2: Run tracking on all recorded videos
# ═══════════════════════════════════════════════════════════════════════════════

def stage_track(output_dir: Path, weights: str = None,
                conf_thres: float = 0.75):
    """Run YOLOv9 + DeepSORT on all CARLA videos."""
    from carla_integration.track_videos import track_video

    if weights is None:
        weights = str(WEIGHTS_PATH)

    logger.info("=" * 60)
    logger.info("STAGE 2: Tracking all recorded videos")
    logger.info("=" * 60)

    tracking_dir = output_dir / "tracking"
    tracking_dir.mkdir(parents=True, exist_ok=True)

    for tracker in ["sfekf", "sfekf_simplex"]:
        tracker_dir = output_dir / tracker
        if not tracker_dir.exists():
            logger.warning("No recordings for %s at %s", tracker, tracker_dir)
            continue

        # Find all scenario dirs with video.mp4
        scenario_dirs = sorted(
            d for d in tracker_dir.iterdir()
            if d.is_dir() and d.name.startswith("scenario_") and (d / "video.mp4").exists()
        )
        logger.info("Found %d videos for %s", len(scenario_dirs), tracker)

        tracker_track_dir = tracking_dir / tracker
        tracker_track_dir.mkdir(parents=True, exist_ok=True)

        for i, sd in enumerate(scenario_dirs):
            video_path = sd / "video.mp4"
            out_name = sd.name  # e.g. "scenario_000"

            # Skip if already tracked
            track_json = tracker_track_dir / out_name / f"{out_name}_tracks.json"
            if track_json.exists():
                logger.info("  [%d/%d] %s — already tracked, skipping",
                            i + 1, len(scenario_dirs), out_name)
                continue

            logger.info("  [%d/%d] Tracking %s/%s",
                        i + 1, len(scenario_dirs), tracker, out_name)
            try:
                # track_video uses the video stem as output subdir name.
                # All CARLA videos are "video.mp4", so create a symlink
                # with a unique name so outputs land in scenario_NNN/.
                scenario_out = tracker_track_dir / out_name
                scenario_out.mkdir(parents=True, exist_ok=True)
                link_path = scenario_out / f"{out_name}.mp4"
                if not link_path.exists():
                    link_path.symlink_to(video_path.resolve())

                result = track_video(
                    video_path=str(link_path),
                    output_dir=str(tracker_track_dir),
                    weights=weights,
                    tracker_type="sfekf",  # always use SF-EKF for perception
                    conf_thres=conf_thres,
                )
                logger.info("    %d tracks, %d detections",
                            result.n_tracks, result.n_detections)
            except Exception as e:
                logger.error("    Failed: %s", e)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage 3: Evaluate tracking + run validation
# ═══════════════════════════════════════════════════════════════════════════════

def _pixel_to_distance(bbox_height_px: float) -> float:
    """Estimate distance from bounding box height using pinhole camera model.

    d = f_y × H_real / h_pixel
    """
    if bbox_height_px <= 0:
        return float("inf")
    return F_Y * PEDESTRIAN_HEIGHT_M / bbox_height_px


def _match_tracks_to_gt(
    track_data: dict,
    ego_trace: List[dict],
    ped_traces: Dict[str, List[dict]],
    fps: float,
    dt_carla: float = 0.05,
) -> dict:
    """Match tracked bounding boxes to GT pedestrians via Hungarian matching.

    Returns per-pedestrian ADE/FDE plus aggregate metrics.
    """
    # Build per-frame track positions in image coords → approximate world coords
    # track_data["track_summaries"] has per-track trajectories in pixel coords
    summaries = track_data.get("track_summaries", [])
    if not summaries or not ego_trace:
        return {"ade": float("nan"), "fde": float("nan"), "n_matched": 0}

    # Convert tracking pixel trajectories to approximate world positions.
    # Strategy: use bbox bottom-center as foot position, estimate distance
    # from bbox height, then project to world coords using ego pose.
    #
    # We need per-frame bbox info — track_summaries only has (cx, cy).
    # For ADE/FDE we use pixel-space error as a proxy (normalized by image size),
    # then convert to meters using the average depth.

    # Build GT trajectories in image space for matching.
    # GT is in world coords — project to image using ego camera transform.
    # Since camera is fixed on ego, we use relative positions.

    # Simpler approach: match tracks to GT peds by temporal overlap and
    # spatial proximity in a common frame.

    n_gt_peds = len(ped_traces)
    n_tracks = len(summaries)

    if n_gt_peds == 0 or n_tracks == 0:
        return {"ade": float("nan"), "fde": float("nan"), "n_matched": 0}

    # For each GT ped, build a world-space trajectory
    gt_trajs = {}
    for pid_str, frames in ped_traces.items():
        pid = int(pid_str)
        gt_trajs[pid] = [(f["x"], f["y"]) for f in frames]

    # For each GT ped, compute an "image-like" trajectory relative to ego
    # by projecting world coords into ego-relative frame at each timestep
    # (forward, lateral) → approximate pixel position
    def _world_to_ego_relative(gx, gy, ego_f):
        """Project world point to ego-relative (forward, lateral) in metres."""
        ex, ey = ego_f["x"], ego_f["y"]
        evx, evy = ego_f.get("vx", 0), ego_f.get("vy", 0)
        heading = math.atan2(evy, evx) if (abs(evx) + abs(evy)) > 0.01 else 0
        dx, dy = gx - ex, gy - ey
        cos_h, sin_h = math.cos(-heading), math.sin(-heading)
        fwd = dx * cos_h - dy * sin_h
        lat = dx * sin_h + dy * cos_h
        return fwd, lat

    def _ego_relative_to_pixel(fwd, lat):
        """Approximate projection: (fwd, lat) metres → (px, py) pixels."""
        if fwd <= 0.5:
            return None
        # Pixel x: lateral offset → horizontal pixel
        px = IMG_W / 2.0 + (lat / fwd) * F_Y
        # Pixel y: vertical — assume pedestrian feet on ground plane
        # y_pixel ≈ IMG_H/2 + (camera_height / fwd) * F_Y  (rough)
        camera_h = 2.0  # camera is 2m above ground
        py = IMG_H / 2.0 + (camera_h / fwd) * F_Y
        return px, py

    # Build cost matrix for Hungarian matching (average pixel distance)
    cost = np.full((n_gt_peds, n_tracks), 1e6)
    gt_pids = sorted(gt_trajs.keys())

    for gi, pid in enumerate(gt_pids):
        gt_world = gt_trajs[pid]
        for ti, ts in enumerate(summaries):
            track_pixels = ts["trajectory"]  # list of (cx, cy)
            t_first = ts["first_frame"]
            t_last = ts["last_frame"]

            # Project GT to pixels at overlapping frames
            errors = []
            for fi, (tx, ty) in enumerate(track_pixels):
                frame_idx = t_first + fi
                # Map frame_idx to ego_trace index
                # Video FPS and CARLA dt may differ; ego_trace has one entry per CARLA tick
                # If video @ 20fps and CARLA @ 20Hz, they're 1:1
                ego_idx = min(frame_idx, len(ego_trace) - 1)
                gt_idx = min(frame_idx, len(gt_world) - 1)

                if ego_idx < 0 or gt_idx < 0:
                    continue

                gx, gy = gt_world[gt_idx]
                fwd, lat = _world_to_ego_relative(gx, gy, ego_trace[ego_idx])
                proj = _ego_relative_to_pixel(fwd, lat)
                if proj is None:
                    continue

                px_gt, py_gt = proj
                # Pixel error (normalized by image diagonal for stability)
                err = math.sqrt((tx - px_gt) ** 2 + (ty - py_gt) ** 2)
                errors.append(err)

            if errors:
                cost[gi, ti] = float(np.mean(errors))

    # Hungarian matching
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(cost)

    ade_list = []
    fde_list = []
    matched = []

    for gi, ti in zip(row_ind, col_ind):
        if cost[gi, ti] > 500:  # too far → not a real match
            continue

        pid = gt_pids[gi]
        ts = summaries[ti]
        track_pixels = ts["trajectory"]
        t_first = ts["first_frame"]
        gt_world = gt_trajs[pid]

        # Compute ADE/FDE in metres via depth estimation
        errors_m = []
        for fi, (tx, ty) in enumerate(track_pixels):
            frame_idx = t_first + fi
            ego_idx = min(frame_idx, len(ego_trace) - 1)
            gt_idx = min(frame_idx, len(gt_world) - 1)

            gx, gy = gt_world[gt_idx]
            fwd_gt, lat_gt = _world_to_ego_relative(gx, gy, ego_trace[ego_idx])
            proj = _ego_relative_to_pixel(fwd_gt, lat_gt)
            if proj is None:
                continue

            px_gt, py_gt = proj
            pixel_err = math.sqrt((tx - px_gt) ** 2 + (ty - py_gt) ** 2)

            # Convert pixel error to metres at the GT depth
            depth = max(fwd_gt, 1.0)
            err_m = pixel_err * depth / F_Y
            errors_m.append(err_m)

        if errors_m:
            ade_list.append(float(np.mean(errors_m)))
            fde_list.append(errors_m[-1])
            matched.append({"gt_pid": pid, "track_id": ts["track_id"],
                            "ade_m": round(float(np.mean(errors_m)), 3),
                            "fde_m": round(errors_m[-1], 3)})

    return {
        "ade": round(float(np.mean(ade_list)), 3) if ade_list else float("nan"),
        "fde": round(float(np.mean(fde_list)), 3) if fde_list else float("nan"),
        "n_matched": len(matched),
        "n_gt": n_gt_peds,
        "n_tracks": n_tracks,
        "matches": matched,
    }


def stage_evaluate(output_dir: Path):
    """Evaluate tracking vs GT, compute per-scenario metrics, run validation."""
    logger.info("=" * 60)
    logger.info("STAGE 3: Evaluate + Validate")
    logger.info("=" * 60)

    all_metrics = {}
    summary = {}

    for tracker in ["sfekf", "sfekf_simplex"]:
        # Load CARLA GT results
        gt_path = output_dir / tracker / "scenario_results.json"
        if not gt_path.exists():
            logger.warning("No GT results for %s", tracker)
            continue
        with open(gt_path) as f:
            gt_results = json.load(f)

        # Load tracking results
        tracking_base = output_dir / "tracking" / tracker
        if not tracking_base.exists():
            logger.warning("No tracking results for %s", tracker)
            continue

        per_scenario = []
        for gt in gt_results:
            sid = gt["scenario_id"]
            scenario_name = f"scenario_{sid:03d}"

            # Load tracking JSON — symlinked as scenario_NNN.mp4, so output
            # is in scenario_NNN/scenario_NNN_tracks.json
            track_json = tracking_base / scenario_name / f"{scenario_name}_tracks.json"
            if not track_json.exists():
                # Fallback: check "video" naming from earlier runs
                track_json = tracking_base / "video" / "video_tracks.json"
            if not track_json.exists():
                logger.warning("  No tracking for %s/%s", tracker, scenario_name)
                per_scenario.append({
                    "scenario_id": sid,
                    "collision": gt["collision"],
                    "min_ttc": gt["min_ttc"],
                    "rho_min": gt["rho_min"],
                    "simplex_activations": gt.get("simplex_activations", 0),
                    "ade": None, "fde": None,
                })
                continue

            with open(track_json) as f:
                track_data = json.load(f)

            # Match tracks to GT and compute ADE/FDE
            ego_trace = gt.get("ego_trace", [])
            ped_traces = gt.get("ped_traces", {})

            eval_result = _match_tracks_to_gt(
                track_data, ego_trace, ped_traces,
                fps=track_data.get("fps", 20.0),
            )

            per_scenario.append({
                "scenario_id": sid,
                "collision": gt["collision"],
                "collision_time": gt.get("collision_time", -1),
                "min_ttc": gt["min_ttc"],
                "rho_min": gt["rho_min"],
                "simplex_activations": gt.get("simplex_activations", 0),
                "ade": eval_result["ade"],
                "fde": eval_result["fde"],
                "n_matched": eval_result["n_matched"],
                "n_gt": eval_result.get("n_gt", 0),
                "n_tracks": eval_result.get("n_tracks", 0),
            })

        all_metrics[tracker] = per_scenario

        # Aggregate
        n = len(per_scenario)
        n_coll = sum(1 for s in per_scenario if s["collision"])
        ades = [s["ade"] for s in per_scenario if s["ade"] is not None and not math.isnan(s["ade"])]
        fdes = [s["fde"] for s in per_scenario if s["fde"] is not None and not math.isnan(s["fde"])]
        ttcs = [s["min_ttc"] for s in per_scenario if s["min_ttc"] is not None]
        simplex_acts = sum(s.get("simplex_activations", 0) for s in per_scenario)

        summary[tracker] = {
            "n_scenarios": n,
            "n_collisions": n_coll,
            "collision_rate": round(100 * n_coll / max(n, 1), 1),
            "ade_mean": round(float(np.mean(ades)), 3) if ades else None,
            "ade_std": round(float(np.std(ades)), 3) if ades else None,
            "fde_mean": round(float(np.mean(fdes)), 3) if fdes else None,
            "fde_std": round(float(np.std(fdes)), 3) if fdes else None,
            "min_ttc_mean": round(float(np.mean(ttcs)), 3) if ttcs else None,
            "simplex_activations": simplex_acts,
        }

        logger.info("  %s: %d scenarios, %d collisions (%.1f%%), "
                     "ADE=%.3f±%.3f, FDE=%.3f±%.3f",
                     tracker, n, n_coll,
                     100 * n_coll / max(n, 1),
                     summary[tracker]["ade_mean"] or 0,
                     summary[tracker]["ade_std"] or 0,
                     summary[tracker]["fde_mean"] or 0,
                     summary[tracker]["fde_std"] or 0)

    # ── Run validation methods on real collision data ─────────────────────
    validation = _run_validation_on_real_data(summary, all_metrics)

    # ── Save everything ──────────────────────────────────────────────────
    results = {
        "summary": summary,
        "validation": validation,
        "per_scenario": all_metrics,
    }

    out_path = output_dir / "carla_experiment_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=_json_default)
    logger.info("Results saved to %s", out_path)

    _print_summary(summary, validation)
    return results


def _json_default(obj):
    """Handle non-serializable types."""
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
    return str(obj)


def _run_validation_on_real_data(summary: dict, all_metrics: dict) -> dict:
    """Run MC, Bayesian, IS, and CE validation using real CARLA collision data.

    Direct MC and Bayesian work on the fixed sample of real outcomes.
    IS and CE use the lightweight model calibrated to the real collision rate.
    """
    validation = {}

    # ── Direct MC from real data (for sfekf — the baseline) ──────────────
    sfekf = all_metrics.get("sfekf", [])
    simplex = all_metrics.get("sfekf_simplex", [])

    for tracker_name, scenarios in [("sfekf", sfekf), ("sfekf_simplex", simplex)]:
        if not scenarios:
            continue

        n = len(scenarios)
        n_fail = sum(1 for s in scenarios if s["collision"])
        p_fail = n_fail / max(n, 1)
        se = math.sqrt(p_fail * (1 - p_fail) / max(n, 1))
        ci_lo = max(0.0, p_fail - 1.96 * se)
        ci_hi = min(1.0, p_fail + 1.96 * se)

        validation[f"mc_{tracker_name}"] = {
            "n_samples": n,
            "n_failures": n_fail,
            "p_fail": round(p_fail, 4),
            "se": round(se, 4),
            "ci_lo": round(ci_lo, 4),
            "ci_hi": round(ci_hi, 4),
        }

        logger.info("  MC %s: %d/%d failures, p=%.4f CI=[%.4f, %.4f]",
                     tracker_name, n_fail, n, p_fail, ci_lo, ci_hi)

        # ── Bayesian (Beta posterior) ────────────────────────────────────
        # Weakly informative prior: Beta(1, 1) = uniform
        alpha_prior, beta_prior = 1, 1
        alpha_post = alpha_prior + n_fail
        beta_post = beta_prior + (n - n_fail)

        map_est = (alpha_post - 1) / (alpha_post + beta_post - 2) if (alpha_post + beta_post) > 2 else p_fail
        bay_ci_lo, bay_ci_hi = stats.beta.ppf([0.025, 0.975], alpha_post, beta_post)

        validation[f"bayesian_{tracker_name}"] = {
            "alpha_prior": alpha_prior,
            "beta_prior": beta_prior,
            "alpha_post": alpha_post,
            "beta_post": beta_post,
            "map_estimate": round(map_est, 4),
            "mean": round(alpha_post / (alpha_post + beta_post), 4),
            "ci_lo": round(bay_ci_lo, 4),
            "ci_hi": round(bay_ci_hi, 4),
        }

        logger.info("  Bayesian %s: Beta(%d, %d), MAP=%.4f, CI=[%.4f, %.4f]",
                     tracker_name, alpha_post, beta_post, map_est,
                     bay_ci_lo, bay_ci_hi)

    # ── Adaptive methods using lightweight model ─────────────────────────
    # These need a callable oracle, so we use the lightweight simulation
    # calibrated to produce similar rates as the real data.
    try:
        from simplex_splat.validation import (
            run_cmaes,
            run_mcmc,
            run_importance_sampling,
            run_cross_entropy,
        )

        # CMA-ES: find worst-case scenario
        logger.info("  Running CMA-ES falsification...")
        cmaes = run_cmaes(n_ped=5, tracker="sfekf", max_evals=100)
        validation["cmaes"] = cmaes.to_dict()

        # MCMC: sample failure distribution
        logger.info("  Running MCMC...")
        init_point = (cmaes.worst_d, cmaes.worst_theta, cmaes.worst_v)
        mcmc = run_mcmc(n_steps=1000, burn_in=100, n_ped=5,
                        init_point=init_point)
        validation["mcmc"] = mcmc.to_dict()

        # IS: importance-weighted failure rate
        logger.info("  Running Importance Sampling...")
        if mcmc.accepted_samples:
            d_arr = np.array([s[0] for s in mcmc.accepted_samples])
            t_arr = np.array([s[1] for s in mcmc.accepted_samples])
            v_arr = np.array([s[2] for s in mcmc.accepted_samples])
            mu_q = np.array([np.mean(d_arr), np.mean(t_arr), np.mean(v_arr)])
            sigma_q = np.array([np.std(d_arr) + 0.5, np.std(t_arr) + 0.05,
                                np.std(v_arr) + 0.05])
        else:
            mu_q = np.array([10.0, math.radians(85.0), 1.5])
            sigma_q = np.array([4.0, math.radians(20.0), 0.3])

        sfekf_mc = validation.get("mc_sfekf", {})
        mc_se = sfekf_mc.get("se", 0.05)
        is_result = run_importance_sampling(
            n_samples=200, n_ped=5, mu_q=mu_q, sigma_q=sigma_q,
            mc_se=mc_se,
        )
        validation["is"] = is_result.to_dict()

        # CE: cross-entropy method
        logger.info("  Running Cross-Entropy...")
        ce = run_cross_entropy(n_per_iter=100, n_ped=5, mc_se=mc_se)
        validation["ce"] = ce.to_dict()

    except Exception as e:
        logger.warning("Adaptive validation methods failed: %s", e)

    return validation


def _print_summary(summary: dict, validation: dict):
    """Print a formatted summary table."""
    print("\n" + "=" * 72)
    print("CARLA EXPERIMENT RESULTS")
    print("=" * 72)

    for tracker in ["sfekf", "sfekf_simplex"]:
        s = summary.get(tracker, {})
        if not s:
            continue
        label = "SF-EKF" if tracker == "sfekf" else "SF-EKF + Simplex"
        print(f"\n  {label}:")
        print(f"    Scenarios:     {s['n_scenarios']}")
        print(f"    Collisions:    {s['n_collisions']} ({s['collision_rate']}%)")
        if s.get("ade_mean") is not None:
            print(f"    ADE:           {s['ade_mean']:.3f} ± {s['ade_std']:.3f} m")
            print(f"    FDE:           {s['fde_mean']:.3f} ± {s['fde_std']:.3f} m")
        if tracker == "sfekf_simplex":
            print(f"    Simplex acts:  {s.get('simplex_activations', 0)}")

    print("\n  Validation:")
    for key in ["mc_sfekf", "mc_sfekf_simplex"]:
        v = validation.get(key, {})
        if v:
            label = key.replace("mc_", "MC ")
            print(f"    {label}: p_fail={v['p_fail']:.4f} "
                  f"CI=[{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]")

    for key in ["bayesian_sfekf", "bayesian_sfekf_simplex"]:
        v = validation.get(key, {})
        if v:
            label = key.replace("bayesian_", "Bayes ")
            print(f"    {label}: Beta({v['alpha_post']},{v['beta_post']}) "
                  f"MAP={v['map_estimate']:.4f} "
                  f"CI=[{v['ci_lo']:.4f}, {v['ci_hi']:.4f}]")

    is_v = validation.get("is", {})
    if is_v:
        print(f"    IS: p_fail={is_v.get('p_fail_is', '?'):.4f} "
              f"VR={is_v.get('variance_reduction', '?'):.1f}x")

    ce_v = validation.get("ce", {})
    if ce_v:
        print(f"    CE: p_fail={ce_v.get('p_fail_ce', '?'):.4f} "
              f"VR={ce_v.get('variance_reduction', '?'):.1f}x")

    print("=" * 72)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="End-to-end CARLA experiment pipeline for Simplex-Track"
    )
    parser.add_argument("--stage", type=str, default="all",
                        choices=["all", "record", "track", "evaluate"],
                        help="Which stage to run (default: all)")
    parser.add_argument("--n-per-tracker", type=int, default=50,
                        help="Scenarios per tracker config (default: 50)")
    parser.add_argument("--n-ped", type=int, default=5,
                        help="Pedestrians per scenario (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--v-ego", type=float, default=10.0,
                        help="Ego speed in km/h (default: 10)")
    parser.add_argument("--tau-safe", type=float, default=2.0,
                        help="Simplex TTC threshold in seconds (default: 2.0)")
    parser.add_argument("--disturbance", type=str, default="fuzzed",
                        choices=["nominal", "fuzzed"],
                        help="Disturbance profile (default: fuzzed)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000,
                        help="CARLA server port")
    parser.add_argument("--weights", type=str, default=str(WEIGHTS_PATH),
                        help="YOLOv9 weights path")
    parser.add_argument("--conf-thres", type=float, default=0.75,
                        help="Detection confidence threshold (default: 0.75)")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT),
                        help="Output directory")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test: 5 scenarios per tracker")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    n_per_tracker = 5 if args.quick else args.n_per_tracker

    t0 = time.time()
    print("=" * 72)
    print("SIMPLEX-TRACK CARLA EXPERIMENT PIPELINE")
    print(f"  Scenarios: {n_per_tracker} × 2 trackers = {n_per_tracker * 2} total")
    print(f"  Disturbance: {args.disturbance}")
    print(f"  Output: {output_dir}")
    print("=" * 72)

    if args.stage in ("all", "record"):
        stage_record(
            output_dir=output_dir,
            n_per_tracker=n_per_tracker,
            n_ped=args.n_ped,
            seed=args.seed,
            v_ego=args.v_ego,
            tau_safe=args.tau_safe,
            disturbance=args.disturbance,
            host=args.host,
            port=args.port,
        )

    if args.stage in ("all", "track"):
        stage_track(
            output_dir=output_dir,
            weights=args.weights,
            conf_thres=args.conf_thres,
        )

    if args.stage in ("all", "evaluate"):
        stage_evaluate(output_dir=output_dir)

    elapsed = time.time() - t0
    print(f"\nPipeline complete ({elapsed:.0f}s)")


if __name__ == "__main__":
    main()
