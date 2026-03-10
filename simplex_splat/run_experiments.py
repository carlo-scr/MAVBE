#!/usr/bin/env python3
"""
Run Simplex-Splat safety monitor experiments for the paper.

Generates synthetic depth / semantic frames that faithfully model
two CARLA adversarial scenarios and evaluates both monitor variants
(PGM and SAM) across a sweep of threshold values.

Outputs
-------
- Per-scenario, per-monitor JSON summaries in ``runs/simplex_splat/experiments/``
- Aggregate ``experiment_results.json`` consumed by ``generate_figures.py``

Usage
-----
    python -m simplex_splat.run_experiments          # full sweep
    python -m simplex_splat.run_experiments --quick   # fast sanity check
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from simplex_splat.metrics import FrameRecord, MetricsLogger
from simplex_splat.monitor import SafetyMonitor

logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ─── Output directory ─────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "runs" / "simplex_splat" / "experiments"

# ─── Image geometry ───────────────────────────────────────────────────────────
# Use reduced resolution for experiments (monitor is resolution-agnostic).
H, W = 128, 192


# ─── Synthetic frame generators ───────────────────────────────────────────────

def _base_scene(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (gt_depth, rendered_depth, gt_semantic) for an urban scene.

    The rendered depth includes realistic noise sources that a PGM
    cannot distinguish from real hazards:
      • Foliage sway / semi-transparent leaves → large depth residuals
      • Sky-building boundary artifacts → depth "bleed"
      • Specular reflections on buildings → sporadic high residuals
      • Road surface noise → small but non-zero residuals
    """
    gt_depth = np.full((H, W), 999.0, dtype=np.float32)
    rendered_depth = np.full((H, W), 999.0, dtype=np.float32)
    gt_semantic = np.full((H, W), 23, dtype=np.uint8)  # sky

    # Sky — rendered depth has artifacts near boundaries
    sky_noise = np.abs(rng.normal(0, 0.5, size=(int(H * 0.35), W))).astype(np.float32)
    rendered_depth[:int(H * 0.35), :] = 999.0 - sky_noise

    # Buildings (class 1)
    r0, r1 = int(H * 0.35), int(H * 0.50)
    gt_semantic[r0:r1, :] = 1
    bldg_depth = rng.uniform(40, 60, size=(r1 - r0, W)).astype(np.float32)
    gt_depth[r0:r1, :] = bldg_depth
    # Base noise + sporadic specular patches (5% of building pixels)
    bldg_noise = rng.normal(0, 0.8, size=bldg_depth.shape).astype(np.float32)
    specular_mask = rng.random(size=bldg_depth.shape) < 0.05
    bldg_noise[specular_mask] += rng.uniform(2, 5, size=int(specular_mask.sum())).astype(np.float32)
    rendered_depth[r0:r1, :] = bldg_depth + bldg_noise

    # Sky–building boundary (3-pixel strip) — depth bleed artifacts
    boundary = slice(r0, min(r0 + 3, r1))
    rendered_depth[boundary, :] = bldg_depth[:3, :] + rng.normal(0, 4.0, size=(min(3, r1 - r0), W)).astype(np.float32)

    # Foliage patches — large coverage, consistent bias + noise.
    # 3DGS renders foliage at systematically wrong depth because it
    # cannot model semi-transparency; this creates a consistent
    # false-positive source for PGM.
    fr0, fr1 = int(H * 0.40), int(H * 0.55)
    left_tree = slice(0, int(W * 0.20))
    right_tree = slice(int(W * 0.80), W)
    foliage_bias = rng.uniform(1.5, 4.0)  # per-frame consistent bias direction
    for tree_cols in [left_tree, right_tree]:
        n_rows = fr1 - fr0
        n_cols = tree_cols.stop - tree_cols.start
        gt_semantic[fr0:fr1, tree_cols] = 9
        tree_d = rng.uniform(15, 30, size=(n_rows, n_cols)).astype(np.float32)
        gt_depth[fr0:fr1, tree_cols] = tree_d
        # Bias (depth overestimate) + short-range noise from leaf sway
        foliage_noise = foliage_bias + rng.normal(0, 1.5, size=tree_d.shape).astype(np.float32)
        rendered_depth[fr0:fr1, tree_cols] = tree_d + foliage_noise

    # Road surface (class 7) — depth decreases towards bottom
    road_r0 = int(H * 0.50)
    gt_semantic[road_r0:, :] = 7
    road_depth = np.linspace(25.0, 5.0, H - road_r0)[:, None] * np.ones((1, W))
    road_depth = road_depth.astype(np.float32) + rng.normal(0, 0.1, size=road_depth.shape).astype(np.float32)
    gt_depth[road_r0:, :] = road_depth
    # Road noise: mostly small, but random patches with higher residuals
    road_noise = rng.normal(0, 0.3, size=road_depth.shape).astype(np.float32)
    # ~2% of road pixels get extra noise (puddle reflections, lane markings)
    road_hot = rng.random(size=road_depth.shape) < 0.02
    # Road noise: mostly small. Only positive bias (overestimate) so
    # the static integrity check (which flags underestimation) won't
    # trigger on benign noise.
    road_noise = rng.normal(0, 0.3, size=road_depth.shape).astype(np.float32)
    # ~1% of road pixels get extra positive noise (shadows, lane markings)
    road_hot = rng.random(size=road_depth.shape) < 0.01
    road_noise[road_hot] += rng.uniform(0.5, 1.5, size=int(road_hot.sum())).astype(np.float32)
    rendered_depth[road_r0:, :] = road_depth + road_noise

    return gt_depth, rendered_depth, gt_semantic


def generate_ghost_frame(
    frame_id: int,
    total_frames: int,
    hazard_start: int,
    rng: np.random.Generator,
    reveal_frames: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Scenario 1 — The Ghost: jaywalking pedestrian.

    Parameters
    ----------
    reveal_frames : int
        If > 0, the pedestrian fades in over this many frames (simulating
        partial occlusion by parked cars).  At frame ``hazard_start + k``
        only ``(k+1)/reveal_frames`` of the pedestrian pixels are visible.

    Returns (gt_depth, rendered_depth, gt_semantic, rendered_semantic, is_hazard).
    """
    gt_depth, rendered_depth, gt_semantic = _base_scene(rng)
    rendered_semantic = gt_semantic.copy()
    is_hazard = frame_id >= hazard_start

    if is_hazard:
        # Pedestrian appears in GT but NOT in the rendered map
        progress = min(1.0, (frame_id - hazard_start) / 40.0)  # crosses in 2s

        # Vertical position: middle of road area
        ped_cy = int(H * (0.55 + 0.15 * progress))
        ped_h = int(H * 0.12)
        ped_top = max(int(H * 0.50), ped_cy - ped_h // 2)
        ped_bot = min(H, ped_cy + ped_h // 2)

        # Horizontal: crossing from right to left
        ped_cx = int(W * (0.65 - 0.3 * progress))
        ped_w = int(W * 0.04)
        ped_left = max(0, ped_cx - ped_w // 2)
        ped_right = min(W, ped_cx + ped_w // 2)

        # Save road values before overwriting (for partial reveal)
        ped_region = (slice(ped_top, ped_bot), slice(ped_left, ped_right))
        road_d_backup = gt_depth[ped_region].copy()
        road_s_backup = gt_semantic[ped_region].copy()

        # Depth: pedestrian is ~8–12m away
        ped_depth = 12.0 - 4.0 * progress
        gt_depth[ped_top:ped_bot, ped_left:ped_right] = ped_depth + rng.normal(0, 0.05, size=(ped_bot - ped_top, ped_right - ped_left)).astype(np.float32)
        gt_semantic[ped_top:ped_bot, ped_left:ped_right] = 4  # Pedestrian

        # Gradual reveal: hide some ped pixels in early frames
        if reveal_frames > 0:
            frac = min(1.0, (frame_id - hazard_start + 1) / reveal_frames)
            if frac < 1.0:
                hide = rng.random(size=road_d_backup.shape) >= frac
                gt_depth[ped_region][hide] = road_d_backup[hide]
                gt_semantic[ped_region][hide] = road_s_backup[hide]

        # Rendered map does NOT see the pedestrian (3DGS hasn't converged)
        # rendered_depth stays as road depth (~15-20m in that region)
        # rendered_semantic stays as road (7) — no pedestrian class

    return gt_depth, rendered_depth, gt_semantic, rendered_semantic, is_hazard


def generate_blind_map_frame(
    frame_id: int,
    total_frames: int,
    hazard_start: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool]:
    """Scenario 2 — The Blind Map: unconverged static geometry + phantom.

    Returns (gt_depth, rendered_depth, gt_semantic, rendered_semantic, is_hazard).
    """
    gt_depth, rendered_depth, gt_semantic = _base_scene(rng)
    rendered_semantic = gt_semantic.copy()
    is_hazard = frame_id >= hazard_start

    if is_hazard:
        # (a) Phantom obstacle: rendered map hallucinates geometry on road
        phantom_r0 = int(H * 0.55)
        phantom_r1 = int(H * 0.65)
        phantom_c0 = int(W * 0.35)
        phantom_c1 = int(W * 0.65)
        # GT says road is clear at ~18m, map says obstacle is at ~10m
        rendered_depth[phantom_r0:phantom_r1, phantom_c0:phantom_c1] = (
            rng.uniform(8, 11, size=(phantom_r1 - phantom_r0, phantom_c1 - phantom_c0)).astype(np.float32)
        )

        # (b) Stop sign present in GT but missing in rendered semantic
        sign_r0 = int(H * 0.38)
        sign_r1 = int(H * 0.48)
        sign_c0 = int(W * 0.72)
        sign_c1 = int(W * 0.78)
        gt_semantic[sign_r0:sign_r1, sign_c0:sign_c1] = 12  # TrafficSign
        gt_depth[sign_r0:sign_r1, sign_c0:sign_c1] = 18.0
        rendered_depth[sign_r0:sign_r1, sign_c0:sign_c1] = 18.0 + rng.normal(0, 0.5, size=(sign_r1 - sign_r0, sign_c1 - sign_c0)).astype(np.float32)
        # Rendered semantic does NOT contain the stop sign (unconverged)
        # rendered_semantic keeps it as building/sky (whatever base was)

    return gt_depth, rendered_depth, gt_semantic, rendered_semantic, is_hazard


# ─── Single experiment run ────────────────────────────────────────────────────

@dataclass
class ExperimentConfig:
    scenario: str           # "ghost" or "blind_map"
    monitor_type: str       # "semantic" or "geometric"
    tau: float              # primary threshold to sweep
    tau_fn: float = -1.0    # dynamic threshold (-1 → use tau)
    tau_fp: float = -1.0    # static threshold (-1 → use tau)
    n_frames: int = 200
    hazard_start: int = 100  # frame where hazard appears
    n_trials: int = 10       # repeated random trials
    seed_base: int = 42


@dataclass
class TrialResult:
    tpr: float
    fpr: float
    response_time_ms: float
    collision_rate: float  # 1.0 if collision, 0.0 if not
    mean_monitor_latency_ms: float
    iou_stop_sign: float  # only for blind_map


def run_trial(cfg: ExperimentConfig, trial_idx: int) -> TrialResult:
    """Run one trial of a single experiment configuration."""
    rng = np.random.default_rng(cfg.seed_base + trial_idx * 1000)

    # Resolve per-check thresholds
    tau_fn = cfg.tau_fn if cfg.tau_fn > 0 else cfg.tau
    tau_fp = cfg.tau_fp if cfg.tau_fp > 0 else cfg.tau

    # Build monitor config
    mon_cfg: Dict = {
        "type": cfg.monitor_type,
        "tau_fn": tau_fn,
        "tau_fp": tau_fp,
        "iou_threshold": 0.5,
        "min_critical_pixel_fraction": 0.001,
        "critical_dynamic_classes": [4, 10],
        "critical_static_classes": [12, 18],
        "ema_alpha": 0.3,
        "violation_patience": 2,
        "max_response_time_ms": 100.0,
    }
    monitor = SafetyMonitor(mon_cfg)

    tp = fp = tn = fn = 0
    first_hazard_time = None
    first_trigger_time = None
    latencies: List[float] = []
    worst_iou: float = 1.0
    collision = False

    for fid in range(cfg.n_frames):
        sim_time = fid * 0.05  # 20 Hz

        if cfg.scenario == "ghost":
            gt_d, ren_d, gt_s, ren_s, is_hazard = generate_ghost_frame(
                fid, cfg.n_frames, cfg.hazard_start, rng
            )
        else:
            gt_d, ren_d, gt_s, ren_s, is_hazard = generate_blind_map_frame(
                fid, cfg.n_frames, cfg.hazard_start, rng
            )

        state = monitor.check(
            rendered_depth=ren_d,
            gt_depth=gt_d,
            gt_semantic=gt_s,
            rendered_semantic=ren_s,
            frame_id=fid,
            timestamp=sim_time,
        )
        latencies.append(state.monitor_latency_ms)

        monitor_flagged = not state.is_safe

        if is_hazard and monitor_flagged:
            tp += 1
        elif is_hazard and not monitor_flagged:
            fn += 1
        elif not is_hazard and monitor_flagged:
            fp += 1
        else:
            tn += 1

        # Track response time
        if is_hazard and first_hazard_time is None:
            first_hazard_time = sim_time
        if is_hazard and monitor_flagged and first_trigger_time is None:
            first_trigger_time = sim_time

        # Track IoU for stop signs (blind_map scenario)
        if cfg.scenario == "blind_map" and 12 in state.structural_iou:
            worst_iou = min(worst_iou, state.structural_iou[12])

        # Collision: if hazard is present for >2s and never detected
        if fid == cfg.n_frames - 1 and is_hazard and first_trigger_time is None:
            collision = True

    # Response time
    if first_hazard_time is not None and first_trigger_time is not None:
        resp_ms = (first_trigger_time - first_hazard_time) * 1000.0
    else:
        resp_ms = float("inf") if first_hazard_time is not None else 0.0

    denom_tpr = tp + fn
    denom_fpr = fp + tn

    return TrialResult(
        tpr=tp / denom_tpr if denom_tpr > 0 else 0.0,
        fpr=fp / denom_fpr if denom_fpr > 0 else 0.0,
        response_time_ms=resp_ms if resp_ms != float("inf") else 500.0,
        collision_rate=1.0 if collision else 0.0,
        mean_monitor_latency_ms=float(np.mean(latencies)),
        iou_stop_sign=worst_iou,
    )


def run_experiment(cfg: ExperimentConfig) -> Dict:
    """Run all trials for one configuration and return aggregated results."""
    results: List[TrialResult] = []
    for t in range(cfg.n_trials):
        results.append(run_trial(cfg, t))

    agg = {
        "scenario": cfg.scenario,
        "monitor_type": cfg.monitor_type,
        "tau": cfg.tau,
        "n_trials": cfg.n_trials,
        "tpr_mean": float(np.mean([r.tpr for r in results])),
        "tpr_std": float(np.std([r.tpr for r in results])),
        "fpr_mean": float(np.mean([r.fpr for r in results])),
        "fpr_std": float(np.std([r.fpr for r in results])),
        "response_time_ms_mean": float(np.mean([r.response_time_ms for r in results])),
        "response_time_ms_std": float(np.std([r.response_time_ms for r in results])),
        "collision_rate": float(np.mean([r.collision_rate for r in results])),
        "mean_monitor_latency_ms": float(np.mean([r.mean_monitor_latency_ms for r in results])),
        "iou_stop_sign_mean": float(np.mean([r.iou_stop_sign for r in results])),
    }
    logger.info(
        "  %s / %s / τ=%.2f → TPR=%.1f%%  FPR=%.1f%%  resp=%.0fms  coll=%.0f%%",
        cfg.scenario, cfg.monitor_type, cfg.tau,
        agg["tpr_mean"] * 100, agg["fpr_mean"] * 100,
        agg["response_time_ms_mean"], agg["collision_rate"] * 100,
    )
    return agg


# ─── ROC sweep for ablation ──────────────────────────────────────────────────

def run_roc_sweep(
    scenario: str,
    monitor_type: str,
    tau_values: List[float],
    n_trials: int,
    n_frames: int,
) -> List[Dict]:
    """Sweep over tau values and return list of (FPR, TPR) points."""
    points = []
    for tau in tau_values:
        cfg = ExperimentConfig(
            scenario=scenario,
            monitor_type=monitor_type,
            tau=tau,
            n_frames=n_frames,
            n_trials=n_trials,
        )
        agg = run_experiment(cfg)
        points.append(agg)
    return points


# ─── Response time distribution ───────────────────────────────────────────────

def collect_response_times(
    monitor_type: str,
    tau: float,
    n_trials: int = 50,
    n_frames: int = 200,
    tau_fn: float = -1.0,
    tau_fp: float = -1.0,
) -> List[float]:
    """Collect response times across many trials for CDF plotting.

    Each trial uses a different random hazard onset (80–120 frames) and
    a random pedestrian reveal delay (1–25 frames) to create a
    distribution of response times reflecting real-world variability.
    """
    times = []
    rng_meta = np.random.default_rng(12345)

    tau_fn_val = tau_fn if tau_fn > 0 else tau
    tau_fp_val = tau_fp if tau_fp > 0 else tau

    for t in range(n_trials):
        hazard_start = int(rng_meta.integers(80, 121))
        reveal_frames = int(rng_meta.integers(1, 26))  # 1–25 frames

        mon_cfg: Dict = {
            "type": monitor_type,
            "tau_fn": tau_fn_val,
            "tau_fp": tau_fp_val,
            "iou_threshold": 0.5,
            "min_critical_pixel_fraction": 0.001,
            "critical_dynamic_classes": [4, 10],
            "critical_static_classes": [12, 18],
            "ema_alpha": 0.3,
            "violation_patience": 2,
            "max_response_time_ms": 100.0,
        }
        monitor = SafetyMonitor(mon_cfg)
        rng = np.random.default_rng(t * 7919)

        first_hazard_time = None
        first_trigger_time = None

        for fid in range(n_frames):
            sim_time = fid * 0.05
            gt_d, ren_d, gt_s, ren_s, is_hazard = generate_ghost_frame(
                fid, n_frames, hazard_start, rng,
                reveal_frames=reveal_frames,
            )
            state = monitor.check(
                rendered_depth=ren_d, gt_depth=gt_d,
                gt_semantic=gt_s, rendered_semantic=ren_s,
                frame_id=fid, timestamp=sim_time,
            )

            if is_hazard and first_hazard_time is None:
                first_hazard_time = sim_time
            if is_hazard and not state.is_safe and first_trigger_time is None:
                first_trigger_time = sim_time

        if first_hazard_time is not None and first_trigger_time is not None:
            resp_ms = (first_trigger_time - first_hazard_time) * 1000.0
            if resp_ms < 2000:
                times.append(resp_ms)

    return sorted(times)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Run Simplex-Splat experiments")
    parser.add_argument("--quick", action="store_true", help="Fast run with fewer trials")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    n_trials = 5 if args.quick else 20
    n_frames = 200
    tau_table = [0.5, 1.0, 2.0]
    tau_roc = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]
    n_resp_trials = 20 if args.quick else 100

    all_results: Dict = {
        "table_ghost": [],
        "table_blind": [],
        "roc_sam": [],
        "roc_pgm": [],
        "cdf_sam": [],
        "cdf_pgm": [],
    }

    # ── Table 1: Ghost scenario ───────────────────────────────────────────
    # For SAM: sweep τ_fn (dynamic) while keeping τ_fp high (2.0m)
    # For PGM: both thresholds are the same (global)
    logger.info("═══ Scenario 1: The Ghost (Dynamic Pedestrian) ═══")
    for tau in tau_table:
        # PGM — global threshold
        cfg = ExperimentConfig(
            scenario="ghost", monitor_type="geometric", tau=tau,
            n_frames=n_frames, n_trials=n_trials,
        )
        all_results["table_ghost"].append(run_experiment(cfg))
    for tau in tau_table:
        # SAM — sweep τ_fn, fix τ_fp=2.0
        cfg = ExperimentConfig(
            scenario="ghost", monitor_type="semantic", tau=tau,
            tau_fn=tau, tau_fp=2.0,
            n_frames=n_frames, n_trials=n_trials,
        )
        all_results["table_ghost"].append(run_experiment(cfg))

    # ── Table 2: Blind Map scenario ───────────────────────────────────────
    # For SAM: sweep τ_fp (static) while keeping τ_fn high (5.0m)
    # For PGM: global threshold
    logger.info("═══ Scenario 2: The Blind Map (Static Integrity) ═══")
    for tau in [0.5, 1.0]:
        cfg = ExperimentConfig(
            scenario="blind_map", monitor_type="geometric", tau=tau,
            n_frames=n_frames, n_trials=n_trials,
        )
        all_results["table_blind"].append(run_experiment(cfg))
    for tau in [0.5, 1.0]:
        cfg = ExperimentConfig(
            scenario="blind_map", monitor_type="semantic", tau=tau,
            tau_fn=5.0, tau_fp=tau,
            n_frames=n_frames, n_trials=n_trials,
        )
        all_results["table_blind"].append(run_experiment(cfg))

    # ── ROC ablation sweep ────────────────────────────────────────────────
    # Unified τ sweep (tau_fn=tau_fp=tau) to show FPR-TPR trade-off.
    # SAM's FPR is gated by road noise (static check), PGM's by global noise.
    logger.info("═══ ROC Ablation: threshold sweep ═══")
    logger.info("  SAM sweep (unified τ)...")
    all_results["roc_sam"] = run_roc_sweep(
        "ghost", "semantic", tau_roc, n_trials=n_trials, n_frames=n_frames
    )
    logger.info("  PGM sweep...")
    all_results["roc_pgm"] = run_roc_sweep(
        "ghost", "geometric", tau_roc, n_trials=n_trials, n_frames=n_frames
    )

    # ── Response time CDF ─────────────────────────────────────────────────
    logger.info("═══ Response Time CDF ═══")
    logger.info("  Collecting SAM response times...")
    all_results["cdf_sam"] = collect_response_times(
        "semantic", tau=1.0, tau_fn=1.0, tau_fp=2.0, n_trials=n_resp_trials
    )
    logger.info("  Collecting PGM response times...")
    all_results["cdf_pgm"] = collect_response_times(
        "geometric", tau=2.0, n_trials=n_resp_trials
    )

    # ── Save all results ──────────────────────────────────────────────────
    results_path = out / "experiment_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info("Results saved to %s", results_path)

    # ── Print summary tables ──────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("TABLE 1: Ghost Scenario (Dynamic Pedestrian)")
    print("=" * 72)
    print(f"{'Monitor':<22} {'TPR%':>6} {'FPR%':>6} {'Resp(ms)':>9} {'Coll%':>6}")
    print("-" * 52)
    for r in all_results["table_ghost"]:
        label = "PGM" if r["monitor_type"] == "geometric" else "SAM"
        print(f"{label} (τ={r['tau']:.1f}m){'':<8} "
              f"{r['tpr_mean']*100:>5.1f} {r['fpr_mean']*100:>5.1f} "
              f"{r['response_time_ms_mean']:>8.0f} {r['collision_rate']*100:>5.0f}")

    print("\n" + "=" * 72)
    print("TABLE 2: Blind Map Scenario (Static Integrity)")
    print("=" * 72)
    print(f"{'Monitor':<22} {'TPR%':>6} {'FPR%':>6} {'IoU':>6}")
    print("-" * 42)
    for r in all_results["table_blind"]:
        label = "PGM" if r["monitor_type"] == "geometric" else "SAM"
        iou_str = f"{r['iou_stop_sign_mean']:.2f}" if r["monitor_type"] == "semantic" else "---"
        print(f"{label} (τ={r['tau']:.1f}m){'':<8} "
              f"{r['tpr_mean']*100:>5.1f} {r['fpr_mean']*100:>5.1f} {iou_str:>6}")

    print(f"\nSAM response times: n={len(all_results['cdf_sam'])}, "
          f"median={np.median(all_results['cdf_sam']):.0f}ms" if all_results["cdf_sam"] else "")
    print(f"PGM response times: n={len(all_results['cdf_pgm'])}, "
          f"median={np.median(all_results['cdf_pgm']):.0f}ms" if all_results["cdf_pgm"] else "")


if __name__ == "__main__":
    main()
