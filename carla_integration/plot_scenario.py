#!/usr/bin/env python3
"""
Plot ground-truth + tracker-estimated paths for a CARLA scenario.

Reads gt_traces.json and tracker_data.json from a scenario directory and
produces a 2-D bird's-eye plot with:
  - Ego vehicle path (solid blue line with directional arrow)
  - Ego predicted trajectory (dotted blue) every 0.5 s
  - GT pedestrian paths (solid coloured lines, labelled)
  - Tracker-estimated paths (dashed lines, same colour)
  - 3-second predicted trajectories (dotted lines, same colour) every 0.5 s
  - 95 % covariance ellipses every 0.5 s
  - Red X at collision point on the PEDESTRIAN that was hit

Usage:
    python -m carla_integration.plot_scenario runs/carla_scenarios/scenario_000
    python -m carla_integration.plot_scenario runs/carla_scenarios/scenario_000 --save
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ELLIPSE_PROB = 0.95
ELLIPSE_INTERVAL_S = 0.5
PRED_HORIZON = 3.0
PRED_DT = 0.1
PED_COLOURS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
]


# ─── Covariance ellipse (from user-provided logic) ───────────────────────────

def _sqrtm_2x2(Sigma: np.ndarray) -> np.ndarray:
    """Matrix square root of a 2x2 PSD matrix via eigendecomposition."""
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    eigvals = np.maximum(eigvals, 0.0)
    return eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def get_ellipse_points(
    mu: np.ndarray,
    Sigma: np.ndarray,
    P: float = 0.95,
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """(x, y) coordinates for the P-probability covariance ellipse."""
    c = -2.0 * np.log(1.0 - P)
    r = np.sqrt(c)
    theta = np.linspace(0, 2 * np.pi, n_points)
    circle = r * np.vstack((np.cos(theta), np.sin(theta)))
    S_sqrt = _sqrtm_2x2(Sigma)
    pts = mu.reshape(2, 1) + S_sqrt @ circle
    return pts[0, :], pts[1, :]


# ─── Trajectory prediction helpers ───────────────────────────────────────────

def _predict_cv(x: float, y: float, vx: float, vy: float) -> list[tuple[float, float]]:
    """Constant-velocity 3-s forward prediction."""
    n = int(PRED_HORIZON / PRED_DT)
    traj = [(x, y)]
    px, py = x, y
    for _ in range(n):
        px += vx * PRED_DT
        py += vy * PRED_DT
        traj.append((px, py))
    return traj


def _predict_ct(
    x: float, y: float, v: float, phi_rad: float, omega: float,
) -> list[tuple[float, float]]:
    """Coordinated-turn 3-s forward prediction."""
    n = int(PRED_HORIZON / PRED_DT)
    traj = [(x, y)]
    px, py, sv, sp, sw = x, y, v, phi_rad, omega
    for _ in range(n):
        dt = PRED_DT
        if abs(sw) > 1e-6:
            px += sv / sw * (math.sin(sp + sw * dt) - math.sin(sp))
            py += sv / sw * (math.cos(sp) - math.cos(sp + sw * dt))
        else:
            px += sv * math.cos(sp) * dt
            py += sv * math.sin(sp) * dt
        sp += sw * dt
        traj.append((px, py))
    return traj


def predict_from_state(state: dict, tracker_type: str) -> list[tuple[float, float]]:
    """Pick the right prediction model based on tracker type."""
    if "vx" in state:
        return _predict_cv(state["x"], state["y"], state["vx"], state["vy"])
    elif "v" in state:
        phi = math.radians(state.get("phi_deg", 0.0))
        omega = state.get("omega", 0.0)
        return _predict_ct(state["x"], state["y"], state["v"], phi, omega)
    return [(state["x"], state["y"])]


def _predict_ego_cv(x: float, y: float, vx: float, vy: float) -> list[tuple[float, float]]:
    """Constant-velocity 3-s forward prediction for the ego vehicle."""
    n = int(PRED_HORIZON / PRED_DT)
    traj = [(x, y)]
    px, py = x, y
    for _ in range(n):
        px += vx * PRED_DT
        py += vy * PRED_DT
        traj.append((px, py))
    return traj


# ─── Main plotting ───────────────────────────────────────────────────────────

def load_scenario(sc_dir: Path) -> tuple[dict, dict]:
    gt_path = sc_dir / "gt_traces.json"
    tk_path = sc_dir / "tracker_data.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"Missing {gt_path}")
    if not tk_path.exists():
        raise FileNotFoundError(f"Missing {tk_path}")
    with open(gt_path) as f:
        gt = json.load(f)
    with open(tk_path) as f:
        tk = json.load(f)
    return gt, tk


def plot_scenario(sc_dir: Path, save: bool = False, out_path: Path | None = None):
    gt, tk = load_scenario(sc_dir)
    tracker_type = tk.get("tracker", "cv_kf")

    fig, ax = plt.subplots(figsize=(16, 10))

    # ── Ego path ──────────────────────────────────────────────────────────
    ego = gt["ego_trace"]
    ex = [p["x"] for p in ego]
    ey = [p["y"] for p in ego]
    ax.plot(ex, ey, color="royalblue", linewidth=2.0, label="Ego (GT)", zorder=5)
    if len(ex) >= 2:
        ax.annotate("", xy=(ex[-1], ey[-1]),
                     xytext=(ex[-2], ey[-2]),
                     arrowprops=dict(arrowstyle="->", color="royalblue", lw=2))
    ax.plot(ex[0], ey[0], "s", color="royalblue", markersize=10, zorder=6)
    ax.annotate("Ego start", (ex[0], ey[0]), fontsize=8,
                xytext=(8, 8), textcoords="offset points")

    # ── Ego predicted trajectories every ELLIPSE_INTERVAL_S ───────────────
    ego_pred_legend_added = False
    ego_next_t = ego[0]["t"] + ELLIPSE_INTERVAL_S if ego else 0.0
    for pt in ego:
        if pt["t"] < ego_next_t:
            continue
        pred = _predict_ego_cv(pt["x"], pt["y"], pt["vx"], pt["vy"])
        px = [p[0] for p in pred]
        py = [p[1] for p in pred]
        lbl = "Ego pred (3 s)" if not ego_pred_legend_added else None
        ax.plot(px, py, color="royalblue", linewidth=0.8, linestyle=":",
                alpha=0.4, label=lbl, zorder=3)
        ego_pred_legend_added = True
        ego_next_t = pt["t"] + ELLIPSE_INTERVAL_S

    # ── Build tracker history: {ped_id: [{t, x, y, P_pos, ...full state}, ...]}
    ped_traces = gt["ped_traces"]
    ped_ids = sorted(ped_traces.keys(), key=lambda k: int(k))

    tracker_history: dict[int, list[dict]] = {}
    for step in tk.get("state_history", []):
        t = step["t"]
        for pid_str, state in step["states"].items():
            pid = int(pid_str)
            if pid not in tracker_history:
                tracker_history[pid] = []
            entry = dict(state)
            entry["t"] = t
            tracker_history[pid].append(entry)

    # ── Find collision point (on the PEDESTRIAN's path) ───────────────────
    collision_point = None
    collision_ped_id = gt.get("collision_ped_id", -1)
    collision_time = gt.get("collision_time", -1.0)

    if gt.get("collision", False) and collision_ped_id >= 0:
        pid_str = str(collision_ped_id)
        if pid_str in ped_traces:
            for p in ped_traces[pid_str]:
                if abs(p["t"] - collision_time) < 0.06:
                    collision_point = (p["x"], p["y"], collision_time, collision_ped_id)
                    break

    # Fallback: try scenario_results.json (for older runs without gt collision fields)
    if collision_point is None:
        results_path = sc_dir.parent / "scenario_results.json"
        if results_path.exists():
            with open(results_path) as f:
                all_results = json.load(f)
            sc_id = int(sc_dir.name.replace("scenario_", ""))
            for r in all_results:
                if r["scenario_id"] == sc_id and r["collision"]:
                    coll_t = r["collision_time"]
                    cpid = r.get("collision_ped_id", -1)
                    if cpid >= 0 and str(cpid) in ped_traces:
                        for p in ped_traces[str(cpid)]:
                            if abs(p["t"] - coll_t) < 0.06:
                                collision_point = (p["x"], p["y"], coll_t, cpid)
                                break
                    if collision_point is None:
                        for p in ego:
                            if abs(p["t"] - coll_t) < 0.06:
                                collision_point = (p["x"], p["y"], coll_t, cpid)
                                break
                    break

    # ── Per-pedestrian: GT + estimated + predictions + ellipses ───────────
    pred_legend_added = False
    ellipse_legend_added = False

    for i, pid_str in enumerate(ped_ids):
        pid = int(pid_str)
        colour = PED_COLOURS[i % len(PED_COLOURS)]
        trace = ped_traces[pid_str]

        # GT path (solid)
        gx = [p["x"] for p in trace]
        gy = [p["y"] for p in trace]
        ax.plot(gx, gy, color=colour, linewidth=1.5, alpha=0.85,
                label=f"Ped {pid} GT")
        ax.plot(gx[0], gy[0], "o", color=colour, markersize=7, zorder=6)
        ax.annotate(f"P{pid}", (gx[0], gy[0]), fontsize=7, fontweight="bold",
                    color=colour, xytext=(5, 5), textcoords="offset points")

        # Tracker estimated path (dashed)
        if pid not in tracker_history:
            continue

        th = tracker_history[pid]
        tx_arr = [p["x"] for p in th]
        ty_arr = [p["y"] for p in th]
        ax.plot(tx_arr, ty_arr, color=colour, linewidth=1.5, linestyle="--",
                alpha=0.7, label=f"Ped {pid} est")

        # Predictions + ellipses every ELLIPSE_INTERVAL_S
        t_start = th[0]["t"] if th else 0.0
        next_t = t_start + ELLIPSE_INTERVAL_S
        for entry in th:
            if entry["t"] < next_t:
                continue

            # ── Predicted trajectory (dotted) ─────────────────────────
            pred = predict_from_state(entry, tracker_type)
            px = [p[0] for p in pred]
            py = [p[1] for p in pred]
            lbl = "Ped pred (3 s)" if not pred_legend_added else None
            ax.plot(px, py, color=colour, linewidth=1.0, linestyle=":",
                    alpha=0.5, label=lbl, zorder=3)
            ax.plot(px[0], py[0], "^", color=colour, markersize=4,
                    alpha=0.6, zorder=4)
            pred_legend_added = True

            # ── Covariance ellipse ────────────────────────────────────
            pp = entry.get("P_pos")
            if pp is not None and len(pp) == 4:
                mu = np.array([entry["x"], entry["y"]])
                Sigma = np.array([[pp[0], pp[1]], [pp[2], pp[3]]])
                eigvals = np.linalg.eigvalsh(Sigma)
                if np.all(eigvals > 1e-8):
                    ex_e, ey_e = get_ellipse_points(mu, Sigma, ELLIPSE_PROB)
                    lbl_e = f"{int(ELLIPSE_PROB*100)}% cov" if not ellipse_legend_added else None
                    ax.plot(ex_e, ey_e, color=colour, linewidth=0.9, alpha=0.55)
                    ax.fill(ex_e, ey_e, color=colour, alpha=0.07, label=lbl_e)
                    ellipse_legend_added = True

            next_t = entry["t"] + ELLIPSE_INTERVAL_S

    # ── Collision marker (on the pedestrian path) ─────────────────────────
    if collision_point is not None:
        cx, cy, ct = collision_point[0], collision_point[1], collision_point[2]
        cpid = collision_point[3] if len(collision_point) > 3 else -1
        ax.plot(cx, cy, "x", color="red", markersize=16, markeredgewidth=3,
                zorder=10)
        ped_label = f" Ped {cpid}" if cpid >= 0 else ""
        ax.plot([], [], "x", color="red", markersize=8, markeredgewidth=2,
                label=f"Collision{ped_label} (t={ct:.1f}s)")

    # ── Labels and legend (outside plot) ──────────────────────────────────
    tracker_name = tk.get("tracker", "?")
    ade = tk.get("ade", 0)
    fde = tk.get("fde", 0)
    n_fp = tk.get("n_fp_brakes", 0)

    ax.set_xlabel("World X (m)", fontsize=11)
    ax.set_ylabel("World Y (m)", fontsize=11)
    ax.set_title(
        f"{sc_dir.name}  |  tracker: {tracker_name}  |  "
        f"ADE={ade:.3f}  FDE={fde:.3f}  FP_brakes={n_fp}",
        fontsize=12,
    )
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.3)

    ax.legend(
        fontsize=8, loc="upper left",
        bbox_to_anchor=(1.02, 1.0), borderaxespad=0,
        ncol=1, framealpha=0.9,
    )

    fig.subplots_adjust(right=0.78)

    if save:
        if out_path is None:
            out_path = sc_dir / "scenario_plot.png"
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to {out_path}")
    else:
        plt.show()

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot GT + tracker paths for a CARLA scenario"
    )
    parser.add_argument("scenario_dir", type=str,
                        help="Path to scenario directory "
                             "(e.g. runs/carla_scenarios/scenario_000)")
    parser.add_argument("--save", action="store_true",
                        help="Save to scenario_plot.png instead of showing")
    parser.add_argument("--output", type=str, default=None,
                        help="Custom output path for saved plot")
    args = parser.parse_args()

    sc_dir = Path(args.scenario_dir)
    if not sc_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {sc_dir}")

    out = Path(args.output) if args.output else None
    plot_scenario(sc_dir, save=args.save, out_path=out)


if __name__ == "__main__":
    main()
