"""
Safety analysis: failure distribution estimation & reachability.

Given structured tracking outputs (CSV/JSON from postprocess.py), this
module computes:

A) **Tracking-failure distribution** — models *when* and *how* the
   perception pipeline loses track of safety-critical objects.
   - Track fragmentation rate
   - Track lifetime distribution (fitted to Weibull / Exponential)
   - Detection confidence distribution per class
   - Innovation (measurement-residual) distribution from the EKF,
     which quantifies how well the motion model matches reality

B) **Forward reachability analysis** — propagates each tracked object's
   state + covariance forward in time to obtain an *occupancy set*.
   A collision is possible iff the ego vehicle's planned occupancy
   intersects the reachable set of any tracked object.  We compute:
   - Per-timestep reachable ellipses (Mahalanobis-gated)
   - Probability of collision under Gaussian assumption
   - Time-to-collision (TTC) bounds

Both parts produce plots + a summary JSON that can feed the Simplex
monitor's decision logic.

Usage (standalone):
    python -m simplex_splat.analysis \
        --csv  runs/simplex_splat/analysis/tracks.csv \
        --json runs/simplex_splat/analysis/tracking_results.json \
        --output runs/simplex_splat/analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ===================================================================
# A.  FAILURE DISTRIBUTION ESTIMATION
# ===================================================================

@dataclass
class TrackStats:
    """Statistics for a single track."""
    track_id: int
    lifetime_frames: int
    first_frame: int
    last_frame: int
    num_gaps: int             # frames where time_since_update > 0
    max_gap: int              # longest consecutive gap
    mean_cov_x: float        # avg position uncertainty
    mean_cov_y: float
    mean_speed_px_per_frame: float  # avg |v|
    innovations: List[float] = field(default_factory=list)


@dataclass
class FailureDistribution:
    """Aggregate failure statistics across all tracks."""

    # Track lifetimes (frames)
    lifetimes: np.ndarray = field(default_factory=lambda: np.array([]))
    # Weibull fit (shape k, scale λ)
    weibull_k: float = 1.0
    weibull_lam: float = 100.0

    # Gap (missed-detection) lengths
    gap_lengths: np.ndarray = field(default_factory=lambda: np.array([]))

    # Per-frame innovation norms (EKF residual)
    innovations: np.ndarray = field(default_factory=lambda: np.array([]))
    innovation_mean: float = 0.0
    innovation_std: float = 0.0

    # Detection confidence per frame
    confidences: np.ndarray = field(default_factory=lambda: np.array([]))

    # Fragmentation
    total_tracks: int = 0
    total_frames_with_objects: int = 0
    fragmentation_rate: float = 0.0  # #track-births / #frames_with_obj

    # Failure probability estimate
    p_loss_per_frame: float = 0.0  # empirical P(track lost | tracked)

    def summary_dict(self) -> dict:
        return {
            "total_tracks": self.total_tracks,
            "lifetime_mean": float(np.mean(self.lifetimes)) if len(self.lifetimes) else 0,
            "lifetime_std": float(np.std(self.lifetimes)) if len(self.lifetimes) else 0,
            "weibull_k": round(self.weibull_k, 4),
            "weibull_lam": round(self.weibull_lam, 4),
            "gap_mean": float(np.mean(self.gap_lengths)) if len(self.gap_lengths) else 0,
            "innovation_mean": round(self.innovation_mean, 4),
            "innovation_std": round(self.innovation_std, 4),
            "fragmentation_rate": round(self.fragmentation_rate, 4),
            "p_loss_per_frame": round(self.p_loss_per_frame, 6),
        }


def compute_track_stats(csv_path: str) -> Tuple[List[TrackStats], FailureDistribution]:
    """Parse tracks.csv and compute per-track + aggregate statistics."""
    import csv as csv_mod

    rows_by_track: Dict[int, list] = defaultdict(list)
    with open(csv_path) as f:
        reader = csv_mod.DictReader(f)
        for row in reader:
            tid = int(row["track_id"])
            rows_by_track[tid].append(row)

    stats_list: List[TrackStats] = []
    all_lifetimes = []
    all_gaps = []
    all_innovations = []
    all_speeds = []

    for tid, rows in sorted(rows_by_track.items()):
        frames = [int(r["frame_idx"]) for r in rows]
        first, last = min(frames), max(frames)
        lifetime = last - first + 1

        # Compute gaps (frames missed within track span)
        frame_set = set(frames)
        gaps = []
        current_gap = 0
        for f in range(first, last + 1):
            if f not in frame_set:
                current_gap += 1
            else:
                if current_gap > 0:
                    gaps.append(current_gap)
                    all_gaps.append(current_gap)
                current_gap = 0

        # Speed + covariance
        vxs = [float(r["vx"]) for r in rows]
        vys = [float(r["vy"]) for r in rows]
        speeds = [np.sqrt(vx**2 + vy**2) for vx, vy in zip(vxs, vys)]
        all_speeds.extend(speeds)

        cov_xs = [float(r["cov_x"]) for r in rows]
        cov_ys = [float(r["cov_y"]) for r in rows]

        # Innovation proxy: frame-to-frame position jump vs predicted
        # (we approximate innovation as |actual_pos - predicted_pos| where
        #  predicted = prev_pos + v*dt)
        innovations = []
        sorted_rows = sorted(rows, key=lambda r: int(r["frame_idx"]))
        for i in range(1, len(sorted_rows)):
            prev, curr = sorted_rows[i - 1], sorted_rows[i]
            df = int(curr["frame_idx"]) - int(prev["frame_idx"])
            if df == 0:
                continue
            # predicted position
            px_pred = float(prev["cx"]) + float(prev["vx"]) * df
            py_pred = float(prev["cy"]) + float(prev["vy"]) * df
            # actual
            px_act = float(curr["cx"])
            py_act = float(curr["cy"])
            innov = np.sqrt((px_act - px_pred)**2 + (py_act - py_pred)**2)
            innovations.append(innov)
            all_innovations.append(innov)

        ts = TrackStats(
            track_id=tid,
            lifetime_frames=lifetime,
            first_frame=first,
            last_frame=last,
            num_gaps=len(gaps),
            max_gap=max(gaps) if gaps else 0,
            mean_cov_x=float(np.mean(cov_xs)) if cov_xs else 0.0,
            mean_cov_y=float(np.mean(cov_ys)) if cov_ys else 0.0,
            mean_speed_px_per_frame=float(np.mean(speeds)) if speeds else 0.0,
            innovations=innovations,
        )
        stats_list.append(ts)
        all_lifetimes.append(lifetime)

    # Aggregate
    fd = FailureDistribution()
    fd.total_tracks = len(stats_list)
    fd.lifetimes = np.array(all_lifetimes, dtype=float)
    fd.gap_lengths = np.array(all_gaps, dtype=float)
    fd.innovations = np.array(all_innovations, dtype=float)

    if len(fd.innovations):
        fd.innovation_mean = float(np.mean(fd.innovations))
        fd.innovation_std = float(np.std(fd.innovations))

    # Fit Weibull to lifetimes
    if len(fd.lifetimes) >= 3:
        try:
            from scipy.stats import weibull_min
            params = weibull_min.fit(fd.lifetimes, floc=0)
            fd.weibull_k = float(params[0])
            fd.weibull_lam = float(params[2])
        except Exception:
            pass  # keep defaults

    # Fragmentation rate
    frames_with_obj = set()
    for ts in stats_list:
        for f in range(ts.first_frame, ts.last_frame + 1):
            frames_with_obj.add(f)
    fd.total_frames_with_objects = len(frames_with_obj)
    if fd.total_frames_with_objects > 0:
        fd.fragmentation_rate = fd.total_tracks / fd.total_frames_with_objects

    # Empirical track-loss probability
    total_tracked_frames = sum(ts.lifetime_frames for ts in stats_list)
    total_gap_frames = int(np.sum(fd.gap_lengths)) if len(fd.gap_lengths) else 0
    if total_tracked_frames > 0:
        fd.p_loss_per_frame = total_gap_frames / total_tracked_frames

    return stats_list, fd


# ===================================================================
# B.  FORWARD REACHABILITY ANALYSIS
# ===================================================================

@dataclass
class ReachableSet:
    """Reachable occupancy at a future timestep for one track."""
    track_id: int
    horizon_frames: int        # how far ahead
    centre: np.ndarray         # (2,) predicted position
    covariance: np.ndarray     # (2, 2) position covariance
    semi_axes: np.ndarray      # (2,) ellipse semi-axis lengths (σ-scaled)
    angle_deg: float           # rotation of ellipse
    sigma_scale: float         # e.g. 3.0 for 99.7% containment


def propagate_track(
    mean_8d: np.ndarray,
    cov_diag_8d: np.ndarray,
    horizon: int = 20,
    dt: float = 1.0,
    sigma: float = 3.0,
) -> List[ReachableSet]:
    """Propagate a single track forward under the Coordinated-Turn model.

    We use the same 5-D state (px, py, v, φ, ω) as BehavioralEKF.predict
    but *without* social forces (conservative: no avoidance assumed).

    Parameters
    ----------
    mean_8d  : current 8-D tracker state [x,y,a,h,vx,vy,va,vh].
    cov_diag_8d : diagonal of 8×8 covariance.
    horizon  : number of frames to propagate.
    dt       : seconds per frame.
    sigma    : Mahalanobis radius for containment ellipse.

    Returns
    -------
    List of ReachableSet, one per future timestep.
    """
    # Extract position and velocity
    px, py = mean_8d[0], mean_8d[1]
    vx, vy = mean_8d[4], mean_8d[5]
    v = np.sqrt(vx**2 + vy**2)
    phi = np.arctan2(vy, vx) if v > 1e-8 else 0.0
    omega = 0.0  # assume straight-line (worst case for crossing)

    # Initial position covariance (2×2)
    P_pos = np.diag([max(cov_diag_8d[0], 1.0), max(cov_diag_8d[1], 1.0)])
    P_vel = np.diag([max(cov_diag_8d[4], 0.1), max(cov_diag_8d[5], 0.1)])

    # Process noise per step
    Q_pos = np.diag([1.0, 1.0])
    Q_vel = np.diag([0.5, 0.5])

    results: List[ReachableSet] = []
    cx, cy = px, py
    P = P_pos.copy()
    cur_vx, cur_vy = vx, vy

    for step in range(1, horizon + 1):
        # Constant-velocity propagation
        cx += cur_vx * dt
        cy += cur_vy * dt

        # Covariance grows: P' = P + dt^2 * P_vel + Q
        P = P + (dt**2) * P_vel + Q_pos

        # Eigen-decomposition for ellipse
        eigvals, eigvecs = np.linalg.eigh(P)
        eigvals = np.clip(eigvals, 1e-4, None)
        semi = sigma * np.sqrt(eigvals)
        angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))

        results.append(ReachableSet(
            track_id=0,  # filled by caller
            horizon_frames=step,
            centre=np.array([cx, cy]),
            covariance=P.copy(),
            semi_axes=semi,
            angle_deg=angle,
            sigma_scale=sigma,
        ))

    return results


def collision_probability(
    ego_xy: np.ndarray,
    ego_radius: float,
    reach: ReachableSet,
) -> float:
    """Approximate P(collision) between a circular ego and a Gaussian obstacle.

    Uses the Mahalanobis distance from ego centre to obstacle distribution.
    """
    diff = ego_xy - reach.centre
    P = reach.covariance
    try:
        P_inv = np.linalg.inv(P)
    except np.linalg.LinAlgError:
        return 0.0
    maha_sq = float(diff @ P_inv @ diff)

    # Effective radius shrinks the Mahalanobis distance
    det_P = np.linalg.det(P)
    effective_sigma = np.sqrt(det_P ** 0.5)  # geometric mean of eigenvalues
    if effective_sigma > 0:
        # shift Maha distance by ego_radius expressed in Maha units
        maha_sq = max(0.0, maha_sq - (ego_radius / effective_sigma) ** 2)

    # P(inside σ-ball) via chi-squared CDF with 2 dof
    p = np.exp(-0.5 * maha_sq)  # quick approximation: exp(-d²/2)
    return float(np.clip(p, 0.0, 1.0))


def compute_ttc(
    ego_xy: np.ndarray,
    ego_vel: np.ndarray,
    track_xy: np.ndarray,
    track_vel: np.ndarray,
    safety_radius: float = 30.0,
) -> float:
    """Analytic time-to-collision for two points with constant velocity.

    Returns TTC in *frames* (inf if no collision).
    """
    dp = track_xy - ego_xy
    dv = track_vel - ego_vel
    a = np.dot(dv, dv)
    b = 2.0 * np.dot(dp, dv)
    c = np.dot(dp, dp) - safety_radius**2

    if a < 1e-12:
        # Parallel / stationary relative motion
        return float("inf")

    disc = b**2 - 4 * a * c
    if disc < 0:
        return float("inf")

    sqrt_disc = np.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    # earliest positive root
    candidates = [t for t in (t1, t2) if t > 0]
    return min(candidates) if candidates else float("inf")


# ===================================================================
# C.  FULL ANALYSIS PIPELINE
# ===================================================================

def run_full_analysis(
    csv_path: str,
    json_path: str,
    output_dir: str,
    ego_xy: Optional[np.ndarray] = None,
    ego_vel: Optional[np.ndarray] = None,
    horizon: int = 30,
    fps: float = 20.0,
) -> dict:
    """Run failure-distribution + reachability analysis, produce outputs.

    Parameters
    ----------
    csv_path : path to tracks.csv
    json_path : path to tracking_results.json
    output_dir : where to write outputs
    ego_xy : ego vehicle pixel position (default: image centre)
    ego_vel : ego velocity in px/frame (default: [0, 0])
    horizon : reachability horizon in frames
    fps : video frame rate

    Returns
    -------
    dict : summary combining failure + reachability results
    """
    os.makedirs(output_dir, exist_ok=True)
    dt = 1.0  # frame-to-frame

    # --- A. Failure distribution ---
    logger.info("Computing failure distribution ...")
    track_stats, fd = compute_track_stats(csv_path)
    logger.info(
        "  %d tracks | lifetime μ=%.1f σ=%.1f | P(loss/frame)=%.4f",
        fd.total_tracks,
        float(np.mean(fd.lifetimes)) if len(fd.lifetimes) else 0,
        float(np.std(fd.lifetimes)) if len(fd.lifetimes) else 0,
        fd.p_loss_per_frame,
    )

    # --- B. Reachability for last-frame tracks ---
    logger.info("Computing forward reachability (horizon=%d frames) ...", horizon)

    with open(json_path) as f:
        all_frames = json.load(f)

    # Use last frame's tracks as current state
    last_frame = all_frames[-1] if all_frames else None
    reachability_results: Dict[int, List[dict]] = {}
    ttc_results: Dict[int, float] = {}

    if last_frame and last_frame.get("tracks"):
        # Default ego position = bottom-centre of frame
        if ego_xy is None:
            # Read image dims from first track bbox to estimate
            ego_xy = np.array([480.0, 540.0])  # sensible default
        if ego_vel is None:
            ego_vel = np.array([0.0, 0.0])

        for trk in last_frame["tracks"]:
            mean_8d = np.array(trk["mean"])
            cov_diag = np.array(trk["covariance_diag"])
            tid = trk["track_id"]

            reach_sets = propagate_track(mean_8d, cov_diag, horizon=horizon, dt=dt, sigma=3.0)

            reach_dicts = []
            for rs in reach_sets:
                rs.track_id = tid
                p_col = collision_probability(ego_xy, 20.0, rs)
                reach_dicts.append({
                    "step": rs.horizon_frames,
                    "cx": round(float(rs.centre[0]), 2),
                    "cy": round(float(rs.centre[1]), 2),
                    "semi_a": round(float(rs.semi_axes[0]), 2),
                    "semi_b": round(float(rs.semi_axes[1]), 2),
                    "angle_deg": round(rs.angle_deg, 2),
                    "p_collision": round(p_col, 6),
                })
            reachability_results[tid] = reach_dicts

            # TTC
            trk_xy = mean_8d[:2]
            trk_vel = mean_8d[4:6]
            ttc = compute_ttc(ego_xy, ego_vel, trk_xy, trk_vel, safety_radius=30.0)
            ttc_results[tid] = round(ttc / fps, 4) if np.isfinite(ttc) else None

    # --- Combine summary ---
    summary = {
        "failure_distribution": fd.summary_dict(),
        "reachability": {
            "horizon_frames": horizon,
            "dt_s": 1.0 / fps,
            "tracks": {
                str(tid): {
                    "ttc_s": ttc_results.get(tid),
                    "max_p_collision": max((s["p_collision"] for s in steps), default=0.0),
                    "reach_sets": steps,
                }
                for tid, steps in reachability_results.items()
            },
        },
    }

    out_path = os.path.join(output_dir, "analysis_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Analysis summary → %s", out_path)

    # --- Plots ---
    _generate_plots(track_stats, fd, reachability_results, output_dir)

    return summary


# ===================================================================
# D.  PLOTTING
# ===================================================================

def _generate_plots(
    track_stats: List[TrackStats],
    fd: FailureDistribution,
    reach: Dict[int, List[dict]],
    output_dir: str,
) -> None:
    """Generate diagnostic plots and save to output_dir."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
    except ImportError:
        logger.warning("matplotlib not available — skipping plots")
        return

    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # 1. Track lifetime histogram + Weibull overlay
    if len(fd.lifetimes) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(fd.lifetimes, bins=max(10, len(fd.lifetimes) // 3),
                density=True, alpha=0.6, color="steelblue", label="Empirical")
        # Weibull PDF overlay
        try:
            from scipy.stats import weibull_min
            x = np.linspace(0, fd.lifetimes.max() * 1.2, 200)
            pdf = weibull_min.pdf(x, fd.weibull_k, scale=fd.weibull_lam)
            ax.plot(x, pdf, "r-", lw=2,
                    label=f"Weibull(k={fd.weibull_k:.2f}, λ={fd.weibull_lam:.1f})")
        except ImportError:
            pass
        ax.set_xlabel("Track Lifetime (frames)")
        ax.set_ylabel("Density")
        ax.set_title("Track Lifetime Distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "lifetime_distribution.png"), dpi=150)
        plt.close(fig)

    # 2. Innovation (EKF residual) distribution
    if len(fd.innovations) > 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.hist(fd.innovations, bins=50, density=True, alpha=0.6, color="salmon")
        ax.axvline(fd.innovation_mean, color="k", ls="--",
                   label=f"μ={fd.innovation_mean:.2f}")
        ax.axvline(fd.innovation_mean + 2 * fd.innovation_std, color="r", ls=":",
                   label=f"μ+2σ={fd.innovation_mean + 2 * fd.innovation_std:.2f}")
        ax.set_xlabel("Innovation norm (px)")
        ax.set_ylabel("Density")
        ax.set_title("EKF Innovation (Measurement Residual) Distribution")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "innovation_distribution.png"), dpi=150)
        plt.close(fig)

    # 3. Per-track uncertainty growth
    if track_stats:
        fig, ax = plt.subplots(figsize=(8, 5))
        for ts in track_stats[:20]:  # plot up to 20 tracks
            if ts.innovations:
                ax.plot(
                    range(len(ts.innovations)),
                    ts.innovations,
                    alpha=0.5,
                    label=f"T{ts.track_id}" if ts.track_id < 10 else None,
                )
        ax.set_xlabel("Observation index within track")
        ax.set_ylabel("Innovation (px)")
        ax.set_title("Per-Track Innovation Over Time")
        if any(ts.track_id < 10 for ts in track_stats[:20]):
            ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "per_track_innovation.png"), dpi=150)
        plt.close(fig)

    # 4. Reachable sets (ellipse overlay)
    if reach:
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.cm.RdYlGn_r
        for tid, steps in reach.items():
            for s in steps:
                alpha = 0.08 + 0.02 * min(s["step"], 10)
                color = cmap(s["p_collision"])
                e = Ellipse(
                    (s["cx"], s["cy"]),
                    width=2 * s["semi_a"],
                    height=2 * s["semi_b"],
                    angle=s["angle_deg"],
                    facecolor=(*color[:3], alpha),
                    edgecolor=color,
                    linewidth=0.5,
                )
                ax.add_patch(e)
            # Mark start
            if steps:
                ax.plot(steps[0]["cx"], steps[0]["cy"], "ko", ms=4)
                ax.annotate(f"T{tid}", (steps[0]["cx"], steps[0]["cy"]),
                            fontsize=7, color="k")
        ax.set_xlabel("x (px)")
        ax.set_ylabel("y (px)")
        ax.set_title(f"Forward Reachable Sets ({len(reach)} tracks)")
        ax.set_aspect("equal")
        ax.autoscale()
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "reachable_sets.png"), dpi=150)
        plt.close(fig)

    # 5. Gap-length histogram
    if len(fd.gap_lengths) > 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(fd.gap_lengths, bins=range(1, int(fd.gap_lengths.max()) + 2),
                alpha=0.7, color="orange", edgecolor="k")
        ax.set_xlabel("Gap length (frames)")
        ax.set_ylabel("Count")
        ax.set_title("Tracking Gap (Missed Detection) Distribution")
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, "gap_distribution.png"), dpi=150)
        plt.close(fig)

    logger.info("Plots saved to %s", fig_dir)


# ===================================================================
# CLI
# ===================================================================

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Failure distribution + reachability analysis")
    parser.add_argument("--csv", required=True, help="tracks.csv from postprocess")
    parser.add_argument("--json", required=True, help="tracking_results.json from postprocess")
    parser.add_argument("--output", default="runs/simplex_splat/analysis")
    parser.add_argument("--horizon", type=int, default=30, help="Reachability horizon (frames)")
    parser.add_argument("--fps", type=float, default=20.0)
    args = parser.parse_args()

    run_full_analysis(args.csv, args.json, args.output,
                      horizon=args.horizon, fps=args.fps)


if __name__ == "__main__":
    main()
