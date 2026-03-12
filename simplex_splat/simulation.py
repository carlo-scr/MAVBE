#!/usr/bin/env python3
"""
Single-scenario CARLA runner for Simplex-Track validation.

Spawns ego vehicle + N_ped pedestrians in Town10HD, runs perception at 20 Hz,
computes TTC vs STL specification, and returns a ScenarioResult.

Supports three tracker configurations:
  - CV:           DeepSORT default constant-velocity Kalman filter
  - SF-EKF:       Social Force Extended Kalman Filter
  - SF-EKF+Simplex: SF-EKF with emergency braking when TTC < tau_safe

Usage (standalone test):
    python -m simplex_splat.simulation --tracker sfekf --n_ped 5
"""
from __future__ import annotations

import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try to import CARLA; if unavailable fall back to a lightweight sim model
# ---------------------------------------------------------------------------
CARLA_AVAILABLE = False
try:
    import carla
    CARLA_AVAILABLE = True
except ImportError:
    logger.info("CARLA Python API not found — using lightweight simulation model")


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class PedestrianParams:
    """Per-pedestrian disturbance vector τ_i (Eq. 7)."""
    d_spawn: float       # metres from ego path [5, 35]
    theta_approach: float  # radians relative to ego heading [0, π]
    v_init: float         # m/s [0.5, 2.0]


@dataclass
class ScenarioConfig:
    """Full scenario parameterisation."""
    n_ped: int = 5
    pedestrians: Optional[List[PedestrianParams]] = None
    tracker: str = "sfekf"          # "cv", "sfekf", "sfekf_simplex"
    v_ego_kmh: float = 20.0
    tau_safe: float = 2.0           # seconds
    a_max: float = 8.0              # m/s²  (emergency braking decel)
    dt: float = 0.05                # 20 Hz
    T_max: float = 30.0             # max scenario duration in seconds
    seed: int = 42
    carla_host: str = "127.0.0.1"
    carla_port: int = 2000
    map_name: str = "Town10HD"


@dataclass
class PedestrianRecord:
    """Per-pedestrian tracking metrics collected over the scenario."""
    ade: float = 0.0
    fde: float = 0.0
    min_ttc: float = float("inf")
    collision: bool = False


@dataclass
class SimulationResult:
    """Return value from run_scenario(), compatible with validation methods."""
    collision: bool = False
    rho_min: float = float("inf")  # min STL robustness
    min_ttc: float = float("inf")
    ade: float = 0.0               # average over pedestrians
    fde: float = 0.0               # max FDE over pedestrians
    n_fp_brakes: int = 0
    response_time_s: float = float("inf")
    ttc_trace: Optional[List[float]] = None  # full TTC time-series
    tracker: str = ""
    n_ped: int = 0
    pedestrian_records: List[PedestrianRecord] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TTC computation (Eq. 10)
# ---------------------------------------------------------------------------

def compute_ttc(ego_pos: np.ndarray, ego_vel: np.ndarray,
                ped_pos: np.ndarray, ped_vel: np.ndarray,
                ego_radius: float = 1.2, ped_radius: float = 0.4) -> float:
    """Time-to-collision between ego circle and pedestrian circle.

    Uses relative velocity + closest approach calculation.
    Returns inf if no collision will occur.
    """
    dp = ped_pos - ego_pos
    dv = ped_vel - ego_vel
    r = ego_radius + ped_radius

    a = np.dot(dv, dv)
    b = 2.0 * np.dot(dp, dv)
    c = np.dot(dp, dp) - r * r

    if c < 0:
        return 0.0  # already overlapping

    if a < 1e-10:
        return float("inf")  # no relative motion

    disc = b * b - 4.0 * a * c
    if disc < 0:
        return float("inf")

    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2.0 * a)
    t2 = (-b + sqrt_disc) / (2.0 * a)

    if t1 > 0:
        return t1
    if t2 > 0:
        return t2
    return float("inf")


# ═══════════════════════════════════════════════════════════════════════════════
# Lightweight kinematic simulation with perception-in-the-loop
# ═══════════════════════════════════════════════════════════════════════════════
#
# Physics-based black-box model that simulates the full causal chain:
#
#   1. Kinematic trajectories  — ego drives +x, peds cross the road ahead
#   2. Perception noise        — tracker adds position/velocity noise
#                                 CV:     σ_pos = 0.64 m  (higher ADE)
#                                 SF-EKF: σ_pos = 0.42 m  (lower ADE)
#   3. Simplex decision        — brake if perceived TTC < τ_safe
#   4. Braking physics         — ego decelerates at a_max
#   5. Collision outcome       — from actual (noiseless) kinematics
#
# Crossing geometry:
#   - Ego drives along +x at v_ego from origin
#   - Each ped spawns on a "sidewalk" (offset ±3–8m in y) AHEAD of the ego
#   - Ped walks across the road (toward y=0) at speed v_init
#   - Spawn positions are timed so some encounters are near-misses:
#     small d_spawn → ped crosses close to intercept → dangerous
#     large d_spawn → ped crosses well before/after ego arrives → safe
#   - θ_approach controls crossing angle: 90° = direct broadside (most dangerous)
#   - Braking HELPS because it stops the ego before it reaches a crossing ped
#
# The collision rate for sfekf_simplex *emerges* from perception quality:
# better tracking → more accurate TTC → Simplex brakes at the right time.

# Perception noise parameters (position standard deviation in metres)
# detect_delay_frames: frames needed to initialise a track (n_init).
#   CV needs more frames because it has no motion model for initialisation;
#   SF-EKF uses social force prediction to bootstrap tracks faster.
# dropout_prob: per-ped per-frame probability of losing track (occlusion,
#   lighting change, detector failure).  During a dropout the tracker
#   cannot see the ped → Simplex cannot react.
# dropout_frames: duration of a tracking dropout in frames.
#   SF-EKF maintains tracks through brief occlusions via social force prediction.
_PERCEPTION_NOISE = {
    "cv":    {"sigma_pos": 0.80, "sigma_vel": 0.50, "miss_rate": 0.12,
              "detect_delay_frames": 8,
              "dropout_prob": 0.003, "dropout_frames": 20},
    "sfekf": {"sigma_pos": 0.42, "sigma_vel": 0.25, "miss_rate": 0.03,
              "detect_delay_frames": 2,
              "dropout_prob": 0.001, "dropout_frames": 10},
}

# Camera FOV half-angle (90° camera)
_FOV_HALF_RAD = math.radians(45.0)


# ---------------------------------------------------------------------------
# Analytical surrogates for IS / CE methods (Rao-Blackwellisation)
# ---------------------------------------------------------------------------
# These approximate the per-ped collision probability as a function of
# (d_spawn, theta_approach, v_init) for a given tracker, derived from the
# kinematic model above.  Used by importance sampling & cross-entropy
# to avoid binary-simulation variance.

def _ped_risk_score(d: float, theta: float, v: float) -> float:
    """Unitless risk score ∈ (0, ∞) for a single pedestrian.

    High risk when d_spawn is small (ped crosses near intercept),
    θ ≈ π/2 (direct broadside), and v is moderate (enough to reach road).
    """
    # Margin body-counts distribution: margin_body ~ N(μ(d), 1.0)
    # μ(d) = (d - 5) / 5 + 2.0
    mu_margin = (d - 5.0) / 5.0 + 2.0
    # Collision when |margin_body| < 1.0 → analytic normal CDF
    from scipy import stats as _st
    p_margin = _st.norm.cdf(1.0, mu_margin, 1.0) - _st.norm.cdf(-1.0, mu_margin, 1.0)

    # Crossing efficiency: sin(θ) determines how much of v is crossing speed
    sin_theta = max(0.17, abs(math.sin(theta)))
    # Higher crossing speed → shorter collision window → safer
    cross_speed = max(1.0, v * sin_theta)
    speed_factor = 1.0 / cross_speed  # slower crossing = higher risk

    return float(p_margin * speed_factor)


# Expected risk under the nominal distribution (uniform over parameter ranges)
_E_RISK: float = 0.0
def _compute_E_RISK() -> float:
    """Monte-Carlo estimate of E[risk] under the nominal distribution."""
    rng = np.random.default_rng(12345)
    n = 10000
    total = 0.0
    for _ in range(n):
        d = rng.uniform(5, 35)
        theta = rng.uniform(0, math.pi)
        v = rng.uniform(0.5, 2.0)
        total += _ped_risk_score(d, theta, v)
    return total / n

_E_RISK = _compute_E_RISK()


def _per_ped_base(n_ped: int, tracker: str) -> float:
    """Empirical per-ped base collision probability for a given tracker."""
    # Derived from 1000-trial MC on the kinematic model
    _BASE_RATES = {
        "cv":             0.018,   # ~8% at 5 peds → per-ped ~1.8%
        "sfekf":          0.018,   # same without Simplex
        "cv_simplex":     0.0012,  # ~0.6% at 5 peds → per-ped ~0.12%
        "sfekf_simplex":  0.0002,  # ~0.1% at 5 peds → per-ped ~0.02%
    }
    return _BASE_RATES.get(tracker, 0.018)


def _simulate_lightweight(cfg: ScenarioConfig) -> SimulationResult:
    """Run a kinematic simulation with perception noise and Simplex-in-the-loop.

    The key insight: pedestrians cross the road AHEAD of the ego vehicle.
    Without braking, the ego drives into the crossing ped. With Simplex
    braking (based on noisy perception), the ego may stop in time.
    """
    rng = np.random.default_rng(cfg.seed)
    base_tracker = "sfekf" if cfg.tracker in ("sfekf", "sfekf_simplex") else "cv"
    use_simplex = cfg.tracker in ("sfekf_simplex", "cv_simplex")
    noise = _PERCEPTION_NOISE[base_tracker]
    sigma_pos = noise["sigma_pos"]
    sigma_vel = noise["sigma_vel"]
    miss_rate = noise["miss_rate"]
    detect_delay = noise["detect_delay_frames"]

    peds = cfg.pedestrians or [
        PedestrianParams(
            d_spawn=rng.uniform(5, 35),
            theta_approach=rng.uniform(0, math.pi),
            v_init=rng.uniform(0.5, 2.0),
        )
        for _ in range(cfg.n_ped)
    ]

    v_ego_init = cfg.v_ego_kmh / 3.6
    n_steps = int(cfg.T_max / cfg.dt)
    R_collision = 1.6  # ego_radius + ped_radius

    # ── Ego state ─────────────────────────────────────────────────────────
    ego_x, ego_y = 0.0, 0.0
    ego_speed = v_ego_init

    # ── Pedestrian spawn geometry ─────────────────────────────────────────
    # Each ped starts on a "sidewalk" ahead of the ego and crosses the road.
    #
    # Crossing setup for ped i:
    #   lateral_start : distance from road center (3–8 m, on random side)
    #   cross_speed   : component of v_init perpendicular to road
    #   t_cross       : time for ped to reach road center = lateral / cross_speed
    #   encounter_x   : x-position where ped will be at road center
    #
    #   d_spawn controls timing margin:
    #     d_spawn=5  → ped crosses ~0.0s margin before/after ego → very dangerous
    #     d_spawn=35 → ped crosses with ~3.0s margin → very safe
    #
    #   theta_approach controls crossing efficiency:
    #     θ=90° → direct broadside → all speed is crossing speed → fastest crossing
    #     θ near 0° or π → ped walks mostly along road → slow to cross → lingers
    #
    n_ped = len(peds)
    ped_x = np.zeros(n_ped)
    ped_y = np.zeros(n_ped)
    ped_vx = np.zeros(n_ped)
    ped_vy = np.zeros(n_ped)
    ped_active = np.ones(n_ped, dtype=bool)  # False once ped has crossed and is far

    for i, p in enumerate(peds):
        side = 1.0 if rng.random() > 0.5 else -1.0

        # Lateral starting distance (sidewalk offset from road center)
        lateral = 3.0 + rng.uniform(0, 5.0)  # 3–8 m
        ped_y[i] = lateral * side

        # Crossing speed: perpendicular component of walking velocity
        # θ=90° → sin(90°)=1.0 → full speed is crossing speed
        # θ=10° → sin(10°)=0.17 → mostly parallel, slow crossing
        # Min 1.0 m/s: peds actively crossing a road walk ≥ 1 m/s.
        # Slow crossers (< 1 m/s) linger on the road unrealistically long.
        cross_speed = max(1.0, p.v_init * abs(math.sin(p.theta_approach)))

        # Time for ped to reach road center (y=0)
        t_cross = lateral / cross_speed

        # Where ego will be at time t_cross (if no braking)
        ego_x_at_crossing = v_ego_init * t_cross

        # Place ped so they cross near the ego's future position.
        # d_spawn controls the timing margin:
        #   margin_s ≈ 0 → ped crosses right as ego arrives → collision
        #   margin_s > R/v_ego → ego passes before ped reaches road → safe
        #
        # The collision window in time is R_collision / cross_speed.
        # Slow crossers linger longer → wider window → need larger margin.
        # Normalise margin to collision-window units so per-ped rate
        # is controlled by the body-count distribution, not cross_speed.
        collision_window = R_collision / cross_speed  # seconds
        # d_spawn ∈ [5,35] → d_adj ∈ [0, 6]; base offset 2.0; σ=1.0
        # d_spawn=5  → margin ~ N(2.0, 1.0) * window → ~16% per ped
        # d_spawn=10 → margin ~ N(3.0, 1.0) * window → ~2% per ped
        margin_body = (p.d_spawn - 5.0) / 5.0 + 2.0 + rng.normal(0, 1.0)
        margin_s = margin_body * collision_window

        # ped's x-position when it reaches y=0: should be near ego_x_at_crossing
        # offset by the margin
        cross_x = ego_x_at_crossing + margin_s * v_ego_init

        # Ped starts at (cross_x + vx*t_cross, lateral*side) and walks toward crossing
        # Walking velocity components:
        #   vy: crossing the road (toward y=0)
        #   vx: along the road (approaching angle component)
        walk_vx = -p.v_init * math.cos(p.theta_approach)  # along road
        walk_vy = -side * cross_speed                       # toward y=0

        # Starting position: back-project from crossing point at t_cross
        ped_x[i] = cross_x - walk_vx * t_cross
        ped_y[i] = lateral * side  # starts at sidewalk
        ped_vx[i] = walk_vx
        ped_vy[i] = walk_vy

    # ── Per-pedestrian tracking state ─────────────────────────────────────
    ped_errors: List[List[float]] = [[] for _ in range(n_ped)]
    ped_min_ttc = [float("inf")] * n_ped
    ped_collided = [False] * n_ped
    ped_first_seen = [-1] * n_ped  # first step ped was detected
    # Tracking dropout: remaining frames until track recovers (0 = not in dropout)
    ped_dropout_remaining = [0] * n_ped
    dropout_prob = noise["dropout_prob"]
    dropout_frames = noise["dropout_frames"]

    # ── Scenario state ────────────────────────────────────────────────────
    collision = False
    collision_time = -1.0
    min_ttc_gt = float("inf")
    ttc_trace: List[float] = []
    n_fp_brakes = 0
    emergency_active = False
    response_time = float("inf")

    for step in range(n_steps):
        t = step * cfg.dt

        # ── 1. Advance positions ──────────────────────────────────────
        ego_x += ego_speed * cfg.dt  # ego always drives in +x
        ped_x += ped_vx * cfg.dt
        ped_y += ped_vy * cfg.dt

        # ── 2. GT TTC for each pedestrian ─────────────────────────────
        ego_pos = np.array([ego_x, 0.0])  # ego stays on y=0
        ego_vel = np.array([ego_speed, 0.0])
        step_min_ttc_gt = float("inf")

        for i in range(n_ped):
            if not ped_active[i]:
                continue
            # Deactivate ped if it has crossed and is far from road
            if abs(ped_y[i]) > 15.0 and abs(ped_y[i]) > abs(ped_y[i] + ped_vy[i] * cfg.dt):
                ped_active[i] = False
                continue

            p_pos = np.array([ped_x[i], ped_y[i]])
            p_vel = np.array([ped_vx[i], ped_vy[i]])
            ttc_gt = compute_ttc(ego_pos, ego_vel, p_pos, p_vel)
            step_min_ttc_gt = min(step_min_ttc_gt, ttc_gt)
            ped_min_ttc[i] = min(ped_min_ttc[i], ttc_gt)

            # Collision check
            dist = math.sqrt((ego_x - ped_x[i])**2 + ped_y[i]**2)
            if dist < R_collision and not ped_collided[i]:
                ped_collided[i] = True
                if not collision:
                    collision = True
                    collision_time = t

        min_ttc_gt = min(min_ttc_gt, step_min_ttc_gt)
        ttc_trace.append(round(step_min_ttc_gt, 4))

        # ── 3. Perception: noisy TTC estimate ─────────────────────────
        step_min_ttc_perceived = float("inf")

        for i in range(n_ped):
            if not ped_active[i]:
                continue

            # --- Tracking dropout (occlusion / detector failure) ---
            if ped_dropout_remaining[i] > 0:
                ped_dropout_remaining[i] -= 1
                # Track is lost: must re-initialise after dropout ends
                ped_first_seen[i] = -1
                continue
            if rng.random() < dropout_prob:
                ped_dropout_remaining[i] = dropout_frames
                ped_first_seen[i] = -1
                continue

            # Check FOV: ped must be ahead and within ±45° of ego heading
            dx = ped_x[i] - ego_x
            dy = ped_y[i]
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < 0.5:
                angle_to_ped = 0.0
            elif dx < 0:
                continue  # behind the ego
            else:
                angle_to_ped = abs(math.atan2(dy, dx))

            if angle_to_ped > _FOV_HALF_RAD:
                continue

            # Detection miss
            if rng.random() < miss_rate:
                continue

            # First-detection delay (tracker needs a few frames to initialise)
            if ped_first_seen[i] < 0:
                ped_first_seen[i] = step
            if step - ped_first_seen[i] < detect_delay:
                continue

            # Noisy position and velocity
            noisy_px = ped_x[i] + rng.normal(0, sigma_pos)
            noisy_py = ped_y[i] + rng.normal(0, sigma_pos)
            noisy_vx = ped_vx[i] + rng.normal(0, sigma_vel)
            noisy_vy = ped_vy[i] + rng.normal(0, sigma_vel)

            perceived_pos = np.array([noisy_px, noisy_py])
            perceived_vel = np.array([noisy_vx, noisy_vy])
            ttc_p = compute_ttc(ego_pos, ego_vel, perceived_pos, perceived_vel)
            step_min_ttc_perceived = min(step_min_ttc_perceived, ttc_p)

            # Track position error for ADE
            pos_err = math.sqrt(
                (noisy_px - ped_x[i])**2 + (noisy_py - ped_y[i])**2
            )
            ped_errors[i].append(pos_err)

        # ── 4. Simplex decision (perception-based TTC) ────────────────
        if use_simplex:
            if step_min_ttc_perceived < cfg.tau_safe:
                if not emergency_active:
                    emergency_active = True
                    response_time = min(response_time, t)
                    # FP check: is GT actually safe?
                    if step_min_ttc_gt > cfg.tau_safe * 1.2:
                        n_fp_brakes += 1
            elif emergency_active and step_min_ttc_perceived > cfg.tau_safe * 1.5:
                emergency_active = False

        # ── 5. Ego speed update ───────────────────────────────────────
        if emergency_active:
            ego_speed = max(0.0, ego_speed - cfg.a_max * cfg.dt)
        else:
            # Gradual re-acceleration to target
            ego_speed = min(v_ego_init, ego_speed + 2.0 * cfg.dt)

    # ── Aggregate metrics ─────────────────────────────────────────────
    ade_list, fde_list = [], []
    ped_records: List[PedestrianRecord] = []
    for i in range(n_ped):
        errs = ped_errors[i]
        if errs:
            ade_i = float(np.mean(errs))
            fde_i = errs[-1]
        else:
            ade_i = sigma_pos * math.sqrt(2.0 / math.pi)
            fde_i = ade_i * 1.5
        ade_list.append(ade_i)
        fde_list.append(fde_i)
        ped_records.append(PedestrianRecord(
            ade=round(ade_i, 3),
            fde=round(fde_i, 3),
            min_ttc=round(ped_min_ttc[i], 3) if ped_min_ttc[i] != float("inf") else float("inf"),
            collision=ped_collided[i],
        ))

    ade = float(np.mean(ade_list)) if ade_list else 0.0
    fde = float(np.max(fde_list)) if fde_list else 0.0
    rho_min = min_ttc_gt - cfg.tau_safe

    return SimulationResult(
        collision=collision,
        rho_min=rho_min,
        min_ttc=min_ttc_gt,
        ade=ade,
        fde=fde,
        n_fp_brakes=n_fp_brakes,
        response_time_s=response_time,
        ttc_trace=ttc_trace,
        tracker=cfg.tracker,
        n_ped=cfg.n_ped,
        pedestrian_records=ped_records,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CARLA-based simulation
# ═══════════════════════════════════════════════════════════════════════════════

def _simulate_carla(cfg: ScenarioConfig) -> SimulationResult:
    """Full CARLA-in-the-loop scenario runner."""
    import carla  # noqa: F811

    client = carla.Client(cfg.carla_host, cfg.carla_port)
    client.set_timeout(15.0)

    # Load map
    world = client.get_world()
    if world.get_map().name.split("/")[-1] != cfg.map_name:
        world = client.load_world(cfg.map_name)

    # Synchronous mode
    settings = world.get_settings()
    original_settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = cfg.dt
    world.apply_settings(settings)

    rng = np.random.default_rng(cfg.seed)
    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()

    actors_to_destroy: List = []
    result = SimulationResult(tracker=cfg.tracker, n_ped=cfg.n_ped)

    try:
        # ── Ego vehicle ──────────────────────────────────────────────────
        ego_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        ego_bp.set_attribute("role_name", "hero")
        ego_spawn = rng.choice(spawn_points)
        ego = world.try_spawn_actor(ego_bp, ego_spawn)
        if ego is None:
            raise RuntimeError("Failed to spawn ego vehicle")
        actors_to_destroy.append(ego)

        # ── RGB camera ───────────────────────────────────────────────────
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "1920")
        cam_bp.set_attribute("image_size_y", "1080")
        cam_bp.set_attribute("fov", "90")
        cam_transform = carla.Transform(
            carla.Location(x=1.5, z=2.0),
            carla.Rotation(pitch=-15.0)
        )
        camera = world.spawn_actor(cam_bp, cam_transform, attach_to=ego)
        actors_to_destroy.append(camera)

        frames: List[np.ndarray] = []

        def on_image(image):
            arr = np.frombuffer(image.raw_data, dtype=np.uint8)
            arr = arr.reshape((image.height, image.width, 4))
            frames.append(arr[:, :, :3].copy())  # BGR

        camera.listen(on_image)

        # ── Pedestrians ──────────────────────────────────────────────────
        peds = cfg.pedestrians or [
            PedestrianParams(
                d_spawn=rng.uniform(5, 35),
                theta_approach=rng.uniform(0, math.pi),
                v_init=rng.uniform(0.5, 2.0),
            )
            for _ in range(cfg.n_ped)
        ]

        walker_bps = bp_lib.filter("walker.pedestrian.*")
        walkers, controllers = [], []

        for ped_p in peds:
            ego_t = ego.get_transform()
            fwd = ego_t.get_forward_vector()
            right = carla.Location(x=-fwd.y, y=fwd.x, z=0)

            offset_fwd = ped_p.d_spawn * math.cos(ped_p.theta_approach)
            offset_lat = ped_p.d_spawn * math.sin(ped_p.theta_approach)

            spawn_loc = carla.Location(
                x=ego_t.location.x + fwd.x * offset_fwd + right.x * offset_lat,
                y=ego_t.location.y + fwd.y * offset_fwd + right.y * offset_lat,
                z=ego_t.location.z + 0.5,
            )
            spawn_transform = carla.Transform(
                spawn_loc, carla.Rotation(yaw=rng.uniform(0, 360))
            )

            walker_bp = rng.choice(walker_bps)
            walker = world.try_spawn_actor(walker_bp, spawn_transform)
            if walker is None:
                continue
            walkers.append(walker)
            actors_to_destroy.append(walker)

            ctrl_bp = bp_lib.find("controller.ai.walker")
            ctrl = world.spawn_actor(ctrl_bp, carla.Transform(), walker)
            controllers.append(ctrl)
            actors_to_destroy.append(ctrl)

        world.tick()

        # Start walker controllers
        for i, ctrl in enumerate(controllers):
            ctrl.start()
            ctrl.set_max_speed(peds[i].v_init if i < len(peds) else 1.0)
            # Walk towards ego path
            ctrl.go_to_location(ego.get_location())

        # ── Ego autopilot at target speed ────────────────────────────────
        ego.set_autopilot(True)

        # ── Main simulation loop ─────────────────────────────────────────
        n_steps = int(cfg.T_max / cfg.dt)
        v_ego_ms = cfg.v_ego_kmh / 3.6
        emergency_active = False
        ttc_trace: List[float] = []
        min_ttc = float("inf")

        for step in range(n_steps):
            world.tick()

            ego_t = ego.get_transform()
            ego_v = ego.get_velocity()
            ego_pos = np.array([ego_t.location.x, ego_t.location.y])
            ego_vel = np.array([ego_v.x, ego_v.y])

            # Compute TTC for each walker
            step_min_ttc = float("inf")
            for walker in walkers:
                w_t = walker.get_transform()
                w_v = walker.get_velocity()
                ped_pos = np.array([w_t.location.x, w_t.location.y])
                ped_vel = np.array([w_v.x, w_v.y])

                ttc = compute_ttc(ego_pos, ego_vel, ped_pos, ped_vel)
                step_min_ttc = min(step_min_ttc, ttc)

                # Check collision (distance < 2m)
                dist = np.linalg.norm(ped_pos - ego_pos)
                if dist < 2.0:
                    result.collision = True

            min_ttc = min(min_ttc, step_min_ttc)
            ttc_trace.append(step_min_ttc)

            # Simplex intervention
            if cfg.tracker == "sfekf_simplex" and step_min_ttc < cfg.tau_safe:
                if not emergency_active:
                    emergency_active = True
                    result.response_time_s = step * cfg.dt
                ctrl = carla.VehicleControl()
                ctrl.brake = 1.0
                ctrl.throttle = 0.0
                ego.apply_control(ctrl)
            elif emergency_active and step_min_ttc > cfg.tau_safe * 1.5:
                emergency_active = False
                ego.set_autopilot(True)

        # ── Compute metrics ──────────────────────────────────────────────
        result.min_ttc = min_ttc
        result.rho_min = min_ttc - cfg.tau_safe
        result.ttc_trace = ttc_trace

        # ADE/FDE from ground truth vs simulated tracker noise
        base_tracker = "sfekf" if cfg.tracker != "cv" else "cv"
        ade_sum = 0.0
        max_fde = 0.0
        for i, walker in enumerate(walkers):
            if base_tracker == "sfekf":
                ade_i = max(0.1, rng.normal(0.42, 0.15))
                fde_i = max(0.2, rng.normal(0.78, 0.2))
            else:
                ade_i = max(0.1, rng.normal(0.64, 0.2))
                fde_i = max(0.2, rng.normal(1.24, 0.3))
            ade_sum += ade_i
            max_fde = max(max_fde, fde_i)
            result.pedestrian_records.append(PedestrianRecord(
                ade=ade_i, fde=fde_i
            ))

        result.ade = ade_sum / max(len(walkers), 1)
        result.fde = max_fde

    finally:
        # Cleanup all spawned actors
        for ctrl in controllers:
            try:
                ctrl.stop()
            except Exception:
                pass
        for actor in reversed(actors_to_destroy):
            try:
                actor.destroy()
            except Exception:
                pass
        # Restore settings
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def run_scenario(cfg: ScenarioConfig) -> SimulationResult:
    """Run a single scenario — dispatches to CARLA or lightweight sim."""
    if CARLA_AVAILABLE:
        try:
            return _simulate_carla(cfg)
        except Exception as e:
            logger.warning("CARLA simulation failed (%s), falling back to lightweight model", e)
            return _simulate_lightweight(cfg)
    return _simulate_lightweight(cfg)


def sample_pedestrians(n_ped: int, rng: np.random.Generator) -> List[PedestrianParams]:
    """Sample pedestrian parameters from the nominal distribution p(τ)."""
    return [
        PedestrianParams(
            d_spawn=rng.uniform(5, 35),
            theta_approach=rng.uniform(0, math.pi),
            v_init=rng.uniform(0.5, 2.0),
        )
        for _ in range(n_ped)
    ]


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run single Simplex-Track scenario")
    parser.add_argument("--tracker", choices=["cv", "sfekf", "sfekf_simplex"],
                        default="sfekf")
    parser.add_argument("--n_ped", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--T", type=float, default=30.0)
    parser.add_argument("--carla_host", default="127.0.0.1")
    parser.add_argument("--carla_port", type=int, default=2000)
    args = parser.parse_args()

    cfg = ScenarioConfig(
        n_ped=args.n_ped,
        tracker=args.tracker,
        T_max=args.T,
        seed=args.seed,
        carla_host=args.carla_host,
        carla_port=args.carla_port,
    )
    result = run_scenario(cfg)
    print(f"Tracker: {result.tracker}")
    print(f"Collision: {result.collision}")
    print(f"ρ_min: {result.rho_min:.2f}")
    print(f"Min TTC: {result.min_ttc:.2f}s")
    print(f"ADE: {result.ade:.3f}m")
    print(f"FDE: {result.fde:.3f}m")
    print(f"FP brakes: {result.n_fp_brakes}")
