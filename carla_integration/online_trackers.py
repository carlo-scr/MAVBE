"""
Online pedestrian trackers for CARLA validation.

Tracker variants
================
  cv_kf              – Constant-velocity Kalman Filter
  sf_ct_ekf          – Social-Force CT-EKF (Coordinated Turn + Helbing–Molnár)
  sf_ct_ekf_simplex  – SF-CT-EKF + TTC-based Simplex safety controller
  sf_cv_ekf          – Social-Force CV-EKF (Cartesian state + Helbing–Molnár EKF)
  sf_cv_ekf_simplex  – SF-CV-EKF + TTC-based Simplex safety controller

All trackers share the same workflow every simulation tick:
  1. Accept noisy (x, y) detections in **world** coordinates
  2. Maintain per-pedestrian track states via predict/update
  3. Predict every pedestrian trajectory 3 s forward
  4. Predict the ego trajectory via constant-velocity Euler integration
  5. Brake if any predicted pedestrian path intersects the ego path

The ``sf_ct_ekf_simplex`` and ``sf_cv_ekf_simplex`` variants additionally brake whenever the
*tracked* TTC falls below τ_safe, providing a safety-layer backup.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
# TTC (circle-circle constant-velocity model)
# ═════════════════════════════════════════════════════════════════════════════

def compute_ttc(
    ego_pos: np.ndarray,
    ego_vel: np.ndarray,
    ped_pos: np.ndarray,
    ped_vel: np.ndarray,
    ego_radius: float = 1.2,
    ped_radius: float = 0.4,
) -> float:
    dp = ped_pos - ego_pos
    dv = ped_vel - ego_vel
    r = ego_radius + ped_radius
    a = float(dv @ dv)
    b = 2.0 * float(dp @ dv)
    c = float(dp @ dp) - r * r
    if c < 0:
        return 0.0
    disc = b * b - 4.0 * a * c
    if disc < 0 or a < 1e-12:
        return float("inf")
    sq = math.sqrt(disc)
    t1 = (-b - sq) / (2.0 * a)
    t2 = (-b + sq) / (2.0 * a)
    if t1 > 0:
        return t1
    if t2 > 0:
        return t2
    return float("inf")


# ═════════════════════════════════════════════════════════════════════════════
# Data classes
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class BrakeEvent:
    t: float
    reason: str            # "prediction_intersect" | "simplex_ttc"
    ped_id: int
    min_pred_dist: float
    ttc: float = float("inf")


@dataclass
class TrackerStepResult:
    brake: bool
    brake_reason: str      # "" | "prediction" | "simplex" | "both"
    events: List[BrakeEvent] = field(default_factory=list)
    track_states: Dict[int, dict] = field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════════
# Internal per-pedestrian track
# ═════════════════════════════════════════════════════════════════════════════

class _PedTrack:
    __slots__ = ("ped_id", "x", "P", "age", "missed")

    def __init__(self, ped_id: int, x: np.ndarray, P: np.ndarray):
        self.ped_id = ped_id
        self.x = x.copy()
        self.P = P.copy()
        self.age = 1
        self.missed = 0


# ═════════════════════════════════════════════════════════════════════════════
# Constant-Velocity Kalman Filter
# ═════════════════════════════════════════════════════════════════════════════

class CVKFTracker:
    """Linear constant-velocity Kalman Filter.

    State  x = [px, py, vx, vy]   (4-D)
    Meas   z = [px, py]           (2-D)
    """

    def __init__(
        self,
        dt: float,
        pred_horizon: float = 3.0,
        pred_dt: float = 0.1,
        collision_radius: float = 1.6,
        q_std: float = 1.0,
        r_std: float = 0.3,
        max_missed: int = 10,
    ):
        self.dt = dt
        self.pred_horizon = pred_horizon
        self.pred_dt = pred_dt
        self.collision_radius = collision_radius
        self.max_missed = max_missed

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float64)

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        # Piece-wise constant white-noise acceleration model
        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        q = q_std**2
        self.Q = q * np.array([
            [dt4/4, 0,     dt3/2, 0    ],
            [0,     dt4/4, 0,     dt3/2],
            [dt3/2, 0,     dt2,   0    ],
            [0,     dt3/2, 0,     dt2  ],
        ], dtype=np.float64)

        self.R = np.diag([r_std**2, r_std**2])
        self.P0 = np.diag([1.0, 1.0, 4.0, 4.0])
        self.I4 = np.eye(4)
        self.tracks: Dict[int, _PedTrack] = {}

    # ── one simulation tick ───────────────────────────────────────────────

    def step(
        self,
        detections: List[dict],
        ego_pos: np.ndarray,
        ego_vel: np.ndarray,
        sim_time: float,
    ) -> TrackerStepResult:
        # Predict all existing tracks
        for trk in self.tracks.values():
            trk.x = self.F @ trk.x
            trk.P = self.F @ trk.P @ self.F.T + self.Q
            trk.P = 0.5 * (trk.P + trk.P.T)
            trk.missed += 1

        # Update with detections (ID-based association)
        for det in detections:
            pid = det["ped_id"]
            z = np.array([det["x_world"], det["y_world"]])
            if pid in self.tracks:
                trk = self.tracks[pid]
                y = z - self.H @ trk.x
                S = self.H @ trk.P @ self.H.T + self.R
                K = trk.P @ self.H.T @ np.linalg.inv(S)
                trk.x = trk.x + K @ y
                trk.P = (self.I4 - K @ self.H) @ trk.P
                trk.P = 0.5 * (trk.P + trk.P.T)
                trk.age += 1
                trk.missed = 0
            else:
                x0 = np.array([z[0], z[1], 0.0, 0.0])
                self.tracks[pid] = _PedTrack(pid, x0, self.P0.copy())

        # Prune lost tracks
        lost = [pid for pid, trk in self.tracks.items()
                if trk.missed > self.max_missed]
        for pid in lost:
            del self.tracks[pid]

        # Predict trajectories → check intersection with ego
        brake = False
        events: List[BrakeEvent] = []
        n_pred = int(self.pred_horizon / self.pred_dt)
        ego_traj = _predict_const_vel(ego_pos, ego_vel, self.pred_dt, n_pred)

        for trk in self.tracks.values():
            if trk.missed > 3:
                continue
            ped_traj = _predict_const_vel(
                trk.x[:2], trk.x[2:4], self.pred_dt, n_pred,
            )
            min_d = _min_traj_dist(ped_traj, ego_traj)
            if min_d < self.collision_radius:
                brake = True
                events.append(BrakeEvent(
                    t=sim_time, reason="prediction_intersect",
                    ped_id=trk.ped_id, min_pred_dist=min_d,
                ))

        states = {
            trk.ped_id: {
                "x": float(trk.x[0]), "y": float(trk.x[1]),
                "vx": float(trk.x[2]), "vy": float(trk.x[3]),
                "age": trk.age, "missed": trk.missed,
                "P_pos": [float(trk.P[0, 0]), float(trk.P[0, 1]),
                          float(trk.P[1, 0]), float(trk.P[1, 1])],
            }
            for trk in self.tracks.values()
        }

        return TrackerStepResult(
            brake=brake,
            brake_reason="prediction" if brake else "",
            events=events,
            track_states=states,
        )


# ═════════════════════════════════════════════════════════════════════════════
# Social-Force Extended Kalman Filter (Coordinated Turn + Helbing–Molnár)
# ═════════════════════════════════════════════════════════════════════════════

class SFEKFTracker:
    """Coordinated-Turn EKF with social-force repulsion.

    State  x = [px, py, v, φ, ω]   (5-D)
    Meas   z = [px, py]             (2-D)

    Matches the process model in ``behavioral_ekf.py`` but operates in
    world coordinates for online CARLA integration.
    """

    A_SOC = 2.1
    B_SOC = 0.3
    PED_RADIUS = 0.5
    SF_CLAMP = 3.0

    def __init__(
        self,
        dt: float,
        pred_horizon: float = 3.0,
        pred_dt: float = 0.1,
        collision_radius: float = 1.6,
        q_diag: Tuple[float, ...] = (0.5, 0.5, 0.5, 0.5, 0.5),
        r_std: float = 0.3,
        max_missed: int = 10,
    ):
        self.dt = dt
        self.pred_horizon = pred_horizon
        self.pred_dt = pred_dt
        self.collision_radius = collision_radius
        self.max_missed = max_missed

        self.Q = np.diag([q**2 for q in q_diag]).astype(np.float64)
        self.R = np.diag([r_std**2, r_std**2])
        self.H = np.zeros((2, 5), dtype=np.float64)
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.P0 = np.diag([1.0, 1.0, 2.0, 1.0, 1.0])
        self.I5 = np.eye(5)
        self.tracks: Dict[int, _PedTrack] = {}
        self._pending_init: Dict[int, np.ndarray] = {}

    # ── social force (2D vector, matching GT Helbing-Molnar model) ───────

    def _social_force_2d(self, x: np.ndarray, exclude_pid: int) -> np.ndarray:
        """2D Helbing-Molnar repulsive force vector (matches GT model)."""
        px, py = x[0], x[1]
        force = np.zeros(2)
        for trk in self.tracks.values():
            if trk.ped_id == exclude_pid:
                continue
            dx = px - trk.x[0]
            dy = py - trk.x[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1e-3 or dist > 5.0:
                continue
            n_ij = np.array([dx / dist, dy / dist])
            magnitude = self.A_SOC * math.exp(
                (2.0 * self.PED_RADIUS - dist) / self.B_SOC
            )
            force += magnitude * n_ij
        f_mag = np.linalg.norm(force)
        if f_mag > self.SF_CLAMP:
            force *= self.SF_CLAMP / f_mag
        return force

    def _decompose_force(
        self, force: np.ndarray, x: np.ndarray,
    ) -> Tuple[float, float]:
        """Decompose 2D force into along-heading (a_v) and perpendicular (a_omega)."""
        phi = x[3]
        heading = np.array([math.cos(phi), math.sin(phi)])
        perp = np.array([-math.sin(phi), math.cos(phi)])
        a_v = float(force @ heading)
        f_perp = float(force @ perp)
        a_omega = f_perp / max(abs(x[2]), 0.3)
        return a_v, a_omega

    # ── coordinated-turn process model ────────────────────────────────────

    def _f(self, x: np.ndarray, dt: float,
           a_v: float = 0.0, a_omega: float = 0.0) -> np.ndarray:
        px, py, v, phi, omega = x
        if abs(omega) > 1e-6:
            px_n = px + v / omega * (math.sin(phi + omega * dt) - math.sin(phi))
            py_n = py + v / omega * (math.cos(phi) - math.cos(phi + omega * dt))
        else:
            px_n = px + v * math.cos(phi) * dt
            py_n = py + v * math.sin(phi) * dt
        v_n = max(v + a_v * dt, 0.0)
        phi_n = phi + omega * dt
        omega_n = omega + a_omega * dt
        return np.array([px_n, py_n, v_n, phi_n, omega_n])

    def _jacobian(self, x: np.ndarray, dt: float) -> np.ndarray:
        _, _, v, phi, omega = x
        F = np.eye(5, dtype=np.float64)
        if abs(omega) > 1e-6:
            so = math.sin(phi + omega * dt)
            co = math.cos(phi + omega * dt)
            sp = math.sin(phi)
            cp = math.cos(phi)
            inv_w = 1.0 / omega
            inv_w2 = inv_w * inv_w
            F[0, 2] = inv_w * (so - sp)
            F[0, 3] = v * inv_w * (co - cp)
            F[0, 4] = v * inv_w2 * (omega * dt * co - so + sp)
            F[1, 2] = inv_w * (cp - co)
            F[1, 3] = v * inv_w * (so - sp)
            F[1, 4] = v * inv_w2 * (omega * dt * so + cp - co)
        else:
            cp, sp = math.cos(phi), math.sin(phi)
            F[0, 2] = cp * dt
            F[0, 3] = -v * sp * dt
            F[0, 4] = -0.5 * v * sp * dt * dt
            F[1, 2] = sp * dt
            F[1, 3] = v * cp * dt
            F[1, 4] = 0.5 * v * cp * dt * dt
        return F

    # ── one simulation tick ───────────────────────────────────────────────

    def step(
        self,
        detections: List[dict],
        ego_pos: np.ndarray,
        ego_vel: np.ndarray,
        sim_time: float,
    ) -> TrackerStepResult:
        # Predict (CT + 2D social force decomposed into along-heading + angular)
        for trk in self.tracks.values():
            sf = self._social_force_2d(trk.x, exclude_pid=trk.ped_id)
            a_v, a_omega = self._decompose_force(sf, trk.x)
            Fj = self._jacobian(trk.x, self.dt)
            trk.x = self._f(trk.x, self.dt, a_v, a_omega)
            trk.P = Fj @ trk.P @ Fj.T + self.Q
            trk.P = 0.5 * (trk.P + trk.P.T)
            trk.missed += 1

        # Update
        for det in detections:
            pid = det["ped_id"]
            z = np.array([det["x_world"], det["y_world"]])
            if pid in self.tracks:
                trk = self.tracks[pid]
                y = z - self.H @ trk.x
                S = self.H @ trk.P @ self.H.T + self.R
                K = trk.P @ self.H.T @ np.linalg.inv(S)
                trk.x = trk.x + K @ y
                trk.P = (self.I5 - K @ self.H) @ trk.P
                trk.P = 0.5 * (trk.P + trk.P.T)
                trk.age += 1
                trk.missed = 0
            else:
                if pid in self._pending_init:
                    prev_z = self._pending_init.pop(pid)
                    dz = z - prev_z
                    d = np.linalg.norm(dz)
                    if d > 0.05:
                        phi0 = math.atan2(dz[1], dz[0])
                        v0 = d / self.dt
                    else:
                        phi0 = 0.0
                        v0 = 0.5
                    x0 = np.array([z[0], z[1], v0, phi0, 0.0])
                    self.tracks[pid] = _PedTrack(pid, x0, self.P0.copy())
                else:
                    self._pending_init[pid] = z.copy()

        # Prune
        lost = [pid for pid, trk in self.tracks.items()
                if trk.missed > self.max_missed]
        for pid in lost:
            del self.tracks[pid]

        # Predict trajectories → check intersection
        brake = False
        events: List[BrakeEvent] = []
        n_pred = int(self.pred_horizon / self.pred_dt)
        ego_traj = _predict_const_vel(ego_pos, ego_vel, self.pred_dt, n_pred)

        for trk in self.tracks.values():
            if trk.missed > 3:
                continue
            ped_traj = self._predict_ct(trk.x, n_pred, ped_id=trk.ped_id)
            min_d = _min_traj_dist(ped_traj, ego_traj)
            if min_d < self.collision_radius:
                brake = True
                events.append(BrakeEvent(
                    t=sim_time, reason="prediction_intersect",
                    ped_id=trk.ped_id, min_pred_dist=min_d,
                ))

        states = {
            trk.ped_id: {
                "x": float(trk.x[0]), "y": float(trk.x[1]),
                "v": float(trk.x[2]),
                "phi_deg": round(math.degrees(trk.x[3]), 1),
                "omega": round(float(trk.x[4]), 4),
                "age": trk.age, "missed": trk.missed,
                "P_pos": [float(trk.P[0, 0]), float(trk.P[0, 1]),
                          float(trk.P[1, 0]), float(trk.P[1, 1])],
            }
            for trk in self.tracks.values()
        }

        return TrackerStepResult(
            brake=brake,
            brake_reason="prediction" if brake else "",
            events=events,
            track_states=states,
        )

    def _predict_ct(self, x: np.ndarray, n_steps: int,
                    ped_id: int = -1) -> List[np.ndarray]:
        """CT extrapolation with social-force repulsion from tracked states."""
        traj = [x[:2].copy()]
        state = x.copy()
        for _ in range(n_steps):
            sf = self._social_force_2d(state, exclude_pid=ped_id)
            a_v, a_omega = self._decompose_force(sf, state)
            state = self._f(state, self.pred_dt, a_v, a_omega)
            traj.append(state[:2].copy())
        return traj


# ═════════════════════════════════════════════════════════════════════════════
# SF-EKF + Simplex Safety Controller
# ═════════════════════════════════════════════════════════════════════════════

class SFEKFSimplexTracker(SFEKFTracker):
    """SF-EKF with an additional TTC-based Simplex safety layer.

    Inherits all SF-EKF tracking and prediction-based braking.
    Additionally triggers full braking whenever the *tracked* TTC
    (computed from the tracker's state estimates, NOT ground truth)
    drops below ``tau_safe``.
    """

    def __init__(self, dt: float, tau_safe: float = 2.0, **kwargs):
        super().__init__(dt, **kwargs)
        self.tau_safe = tau_safe

    def step(
        self,
        detections: List[dict],
        ego_pos: np.ndarray,
        ego_vel: np.ndarray,
        sim_time: float,
    ) -> TrackerStepResult:
        result = super().step(detections, ego_pos, ego_vel, sim_time)

        # Simplex TTC check on tracked states
        for trk in self.tracks.values():
            if trk.missed > 3:
                continue
            ped_pos = trk.x[:2]
            v, phi = trk.x[2], trk.x[3]
            ped_vel = np.array([v * math.cos(phi), v * math.sin(phi)])
            ttc = compute_ttc(ego_pos, ego_vel, ped_pos, ped_vel)
            if ttc < self.tau_safe:
                if not result.brake:
                    result.brake = True
                    result.brake_reason = "simplex"
                elif result.brake_reason == "prediction":
                    result.brake_reason = "both"
                result.events.append(BrakeEvent(
                    t=sim_time, reason="simplex_ttc",
                    ped_id=trk.ped_id, min_pred_dist=0.0, ttc=ttc,
                ))

        return result


# ═════════════════════════════════════════════════════════════════════════════
# Social-Force CV-EKF (Cartesian state + Helbing–Molnár EKF)
# ═════════════════════════════════════════════════════════════════════════════

class SFCVEKFTracker:
    """Social-Force Constant-Velocity Extended Kalman Filter.

    State  x = [px, py, vx, vy]   (4-D)
    Meas   z = [px, py]           (2-D)

    Nonlinear transition:
        f(x) = F_cv @ x + [0, 0, sf_x(px,py)*dt, sf_y(px,py)*dt]

    where sf(px,py) is the 2D Helbing-Molnar social-force repulsion.
    Because the social force depends on position, the transition is
    nonlinear and covariance is propagated via the EKF Jacobian:
        J = F_cv + d[0,0,sf_x*dt,sf_y*dt]/d[px,py,vx,vy]

    This keeps the well-observed Cartesian [px,py,vx,vy] state from
    the CV-KF while properly accounting for social-force uncertainty
    in the covariance propagation.
    """

    A_SOC = 2.1
    B_SOC = 0.3
    PED_RADIUS = 0.5
    SF_CLAMP = 3.0

    def __init__(
        self,
        dt: float,
        pred_horizon: float = 3.0,
        pred_dt: float = 0.1,
        collision_radius: float = 1.6,
        q_std: float = 1.0,
        r_std: float = 0.3,
        max_missed: int = 10,
    ):
        self.dt = dt
        self.pred_horizon = pred_horizon
        self.pred_dt = pred_dt
        self.collision_radius = collision_radius
        self.max_missed = max_missed

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=np.float64)

        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=np.float64)

        dt2, dt3, dt4 = dt**2, dt**3, dt**4
        q = q_std**2
        self.Q = q * np.array([
            [dt4/4, 0,     dt3/2, 0    ],
            [0,     dt4/4, 0,     dt3/2],
            [dt3/2, 0,     dt2,   0    ],
            [0,     dt3/2, 0,     dt2  ],
        ], dtype=np.float64)

        self.R = np.diag([r_std**2, r_std**2])
        self.P0 = np.diag([1.0, 1.0, 4.0, 4.0])
        self.I4 = np.eye(4)
        self.tracks: Dict[int, _PedTrack] = {}

    # ── social force + analytical Jacobian ─────────────────────────────────

    def _social_force_2d(self, x: np.ndarray, exclude_pid: int) -> np.ndarray:
        """2D Helbing-Molnar repulsive force vector."""
        px, py = x[0], x[1]
        force = np.zeros(2)
        for trk in self.tracks.values():
            if trk.ped_id == exclude_pid:
                continue
            dx = px - trk.x[0]
            dy = py - trk.x[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1e-3 or dist > 5.0:
                continue
            n_ij = np.array([dx / dist, dy / dist])
            magnitude = self.A_SOC * math.exp(
                (2.0 * self.PED_RADIUS - dist) / self.B_SOC
            )
            force += magnitude * n_ij
        f_mag = np.linalg.norm(force)
        if f_mag > self.SF_CLAMP:
            force *= self.SF_CLAMP / f_mag
        return force

    def _social_force_jacobian(self, x: np.ndarray,
                               exclude_pid: int) -> np.ndarray:
        """Analytical 2x2 Jacobian d(sf)/d(px,py) of the Helbing-Molnar force.

        For each neighbour j, the force on i is:
            f_ij = A * exp((2R - d) / B) * n_ij
        where d = ||p_i - p_j||, n_ij = (p_i - p_j) / d.

        Differentiating w.r.t. p_i:
            df_ij/dp_i = magnitude * [ (I - n n^T) / d  -  n n^T / B ]
        where magnitude = A * exp((2R - d) / B).
        """
        px, py = x[0], x[1]
        J_sf = np.zeros((2, 2))
        I2 = np.eye(2)
        for trk in self.tracks.values():
            if trk.ped_id == exclude_pid:
                continue
            dx = px - trk.x[0]
            dy = py - trk.x[1]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 1e-3 or dist > 5.0:
                continue
            n = np.array([dx / dist, dy / dist])
            nnT = np.outer(n, n)
            mag = self.A_SOC * math.exp(
                (2.0 * self.PED_RADIUS - dist) / self.B_SOC
            )
            J_sf += mag * ((I2 - nnT) / dist - nnT / self.B_SOC)
        return J_sf

    def _jacobian(self, x: np.ndarray, exclude_pid: int) -> np.ndarray:
        """4x4 EKF Jacobian of the full transition f(x).

        J = F_cv + [[0,0],[0,0],[dG_sf/dp]] where dG_sf/dp is the
        social force gradient scaled by dt.
        """
        J = self.F.copy()
        J_sf = self._social_force_jacobian(x, exclude_pid)
        J[2, 0] += J_sf[0, 0] * self.dt
        J[2, 1] += J_sf[0, 1] * self.dt
        J[3, 0] += J_sf[1, 0] * self.dt
        J[3, 1] += J_sf[1, 1] * self.dt
        return J

    # ── one simulation tick ────────────────────────────────────────────────

    def step(
        self,
        detections: List[dict],
        ego_pos: np.ndarray,
        ego_vel: np.ndarray,
        sim_time: float,
    ) -> TrackerStepResult:
        # EKF Predict: CV kinematics + social force (nonlinear in position)
        for trk in self.tracks.values():
            sf = self._social_force_2d(trk.x, exclude_pid=trk.ped_id)
            Jk = self._jacobian(trk.x, exclude_pid=trk.ped_id)
            trk.x = self.F @ trk.x
            trk.x[2] += sf[0] * self.dt
            trk.x[3] += sf[1] * self.dt
            trk.P = Jk @ trk.P @ Jk.T + self.Q
            trk.P = 0.5 * (trk.P + trk.P.T)
            trk.missed += 1

        # Update with detections (ID-based association)
        for det in detections:
            pid = det["ped_id"]
            z = np.array([det["x_world"], det["y_world"]])
            if pid in self.tracks:
                trk = self.tracks[pid]
                y = z - self.H @ trk.x
                S = self.H @ trk.P @ self.H.T + self.R
                K = trk.P @ self.H.T @ np.linalg.inv(S)
                trk.x = trk.x + K @ y
                trk.P = (self.I4 - K @ self.H) @ trk.P
                trk.P = 0.5 * (trk.P + trk.P.T)
                trk.age += 1
                trk.missed = 0
            else:
                x0 = np.array([z[0], z[1], 0.0, 0.0])
                self.tracks[pid] = _PedTrack(pid, x0, self.P0.copy())

        # Prune lost tracks
        lost = [pid for pid, trk in self.tracks.items()
                if trk.missed > self.max_missed]
        for pid in lost:
            del self.tracks[pid]

        # Predict trajectories with social force → check intersection
        brake = False
        events: List[BrakeEvent] = []
        n_pred = int(self.pred_horizon / self.pred_dt)
        ego_traj = _predict_const_vel(ego_pos, ego_vel, self.pred_dt, n_pred)

        for trk in self.tracks.values():
            if trk.missed > 3:
                continue
            ped_traj = self._predict_sf_cv(trk.x, n_pred, trk.ped_id)
            min_d = _min_traj_dist(ped_traj, ego_traj)
            if min_d < self.collision_radius:
                brake = True
                events.append(BrakeEvent(
                    t=sim_time, reason="prediction_intersect",
                    ped_id=trk.ped_id, min_pred_dist=min_d,
                ))

        states = {
            trk.ped_id: {
                "x": float(trk.x[0]), "y": float(trk.x[1]),
                "vx": float(trk.x[2]), "vy": float(trk.x[3]),
                "age": trk.age, "missed": trk.missed,
                "P_pos": [float(trk.P[0, 0]), float(trk.P[0, 1]),
                          float(trk.P[1, 0]), float(trk.P[1, 1])],
            }
            for trk in self.tracks.values()
        }

        return TrackerStepResult(
            brake=brake,
            brake_reason="prediction" if brake else "",
            events=events,
            track_states=states,
        )

    def _predict_sf_cv(self, x: np.ndarray, n_steps: int,
                       ped_id: int) -> List[np.ndarray]:
        """CV extrapolation with social-force repulsion."""
        traj = [x[:2].copy()]
        state = x.copy()
        for _ in range(n_steps):
            sf = self._social_force_2d(state, exclude_pid=ped_id)
            state[:2] += state[2:4] * self.pred_dt
            state[2] += sf[0] * self.pred_dt
            state[3] += sf[1] * self.pred_dt
            traj.append(state[:2].copy())
        return traj


class SFCVEKFSimplexTracker(SFCVEKFTracker):
    """SF-CV-EKF with TTC-based Simplex safety layer."""

    def __init__(self, dt: float, tau_safe: float = 2.0, **kwargs):
        super().__init__(dt, **kwargs)
        self.tau_safe = tau_safe

    def step(
        self,
        detections: List[dict],
        ego_pos: np.ndarray,
        ego_vel: np.ndarray,
        sim_time: float,
    ) -> TrackerStepResult:
        result = super().step(detections, ego_pos, ego_vel, sim_time)

        for trk in self.tracks.values():
            if trk.missed > 3:
                continue
            ped_pos = trk.x[:2]
            ped_vel = trk.x[2:4]
            ttc = compute_ttc(ego_pos, ego_vel, ped_pos, ped_vel)
            if ttc < self.tau_safe:
                if not result.brake:
                    result.brake = True
                    result.brake_reason = "simplex"
                elif result.brake_reason == "prediction":
                    result.brake_reason = "both"
                result.events.append(BrakeEvent(
                    t=sim_time, reason="simplex_ttc",
                    ped_id=trk.ped_id, min_pred_dist=0.0, ttc=ttc,
                ))

        return result


# ═════════════════════════════════════════════════════════════════════════════
# Helpers
# ═════════════════════════════════════════════════════════════════════════════

def _predict_const_vel(
    pos: np.ndarray, vel: np.ndarray, dt: float, n_steps: int,
) -> List[np.ndarray]:
    """Constant-velocity Euler extrapolation."""
    traj = [pos.copy()]
    p = pos.copy()
    for _ in range(n_steps):
        p = p + vel * dt
        traj.append(p.copy())
    return traj


def _min_traj_dist(
    traj_a: List[np.ndarray], traj_b: List[np.ndarray],
) -> float:
    """Minimum point-wise distance between two same-length trajectories."""
    n = min(len(traj_a), len(traj_b))
    min_d = float("inf")
    for i in range(n):
        d = np.linalg.norm(traj_a[i] - traj_b[i])
        if d < min_d:
            min_d = d
    return min_d


# ═════════════════════════════════════════════════════════════════════════════
# Factory
# ═════════════════════════════════════════════════════════════════════════════

def create_tracker(
    tracker_type: str,
    dt: float,
    tau_safe: float = 2.0,
    **kwargs,
):
    """Create the appropriate tracker based on the scenario spec."""
    if tracker_type == "cv_kf":
        return CVKFTracker(dt, **kwargs)
    if tracker_type == "sf_ct_ekf":
        return SFEKFTracker(dt, **kwargs)
    if tracker_type == "sf_ct_ekf_simplex":
        return SFEKFSimplexTracker(dt, tau_safe=tau_safe, **kwargs)
    if tracker_type == "sf_cv_ekf":
        return SFCVEKFTracker(dt, **kwargs)
    if tracker_type == "sf_cv_ekf_simplex":
        return SFCVEKFSimplexTracker(dt, tau_safe=tau_safe, **kwargs)
    raise ValueError(f"Unknown tracker type: {tracker_type!r}")
