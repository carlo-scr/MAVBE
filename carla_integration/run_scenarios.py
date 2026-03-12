#!/usr/bin/env python3
"""
CARLA Scenario Generator for Simplex-Track Validation.

Runs N parameterised scenarios in CARLA, records per-scenario:
  - Video frames (for offline YOLOv9 + DeepSORT perception)
  - Ground-truth pedestrian positions (for ADE/FDE computation)
  - Ego vehicle state (position, velocity)
  - Collision flag (from CARLA collision sensor)
  - TTC trace (computed from GT positions)

Each scenario follows the paper's disturbance model:
  - Ego drives at v_ego = 20 km/h in Town10HD
  - N_ped pedestrians spawn with (d_spawn, theta_approach, v_init)
  - Duration: 30 s at 20 Hz

60 % of pedestrians use CARLA's AI controller (nav-mesh walking).
40 % are "disturbed" — prescribed velocity via WalkerControl each tick.
The --disturbance flag selects which distribution the 40 % sample from:
  NOMINAL  – gentle crossings, rare failures
             θ ∈ [10°, 40°], v ∈ [0.5, 1.2] m/s
  FUZZED   – broadside crossings, frequent failures
             θ ∈ [55°, 110°], v ∈ [1.5, 2.0] m/s
All traffic lights are forced green so the ego is never stopped by signals.

Usage:
    python -m carla_integration.run_scenarios --n-scenarios 10
    python -m carla_integration.run_scenarios --n-scenarios 10 --disturbance nominal
    python -m carla_integration.run_scenarios --n-scenarios 10 --disturbance fuzzed --disturbed-frac 0.4
    python -m carla_integration.run_scenarios --n-scenarios 10 --n-ped 7 --tracker sfekf_simplex
    python -m carla_integration.run_scenarios --n-scenarios 10 --save-video

Requires:
    - CARLA server running (default localhost:2000)
    - carla Python package on PYTHONPATH
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Try CARLA import
# ---------------------------------------------------------------------------
try:
    import carla
except ImportError:
    logger.error(
        "carla Python package not found.\n"
        "Install it via:  pip install carla==0.9.11\n"
        "Or add the CARLA egg to PYTHONPATH."
    )
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# Data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class DisturbanceProfile:
    """Tunable disturbance parameters for the 40 % non-AI (prescribed) pedestrians.

    Both profiles keep disturbed_frac = 0.4 (40 % prescribed, 60 % AI-controlled).
    Only the *crossing behaviour* of the disturbed peds differs:

    NOMINAL  – shallow crossing angles, moderate speed → failures are rare.
               θ^approach ∈ [10°, 40°], v^init ∈ [0.5, 1.2] m/s.

    FUZZED   – broadside crossings at high speed (paper's critical failure region)
               → failures are frequent.
               θ^approach ∈ [55°, 110°], v^init ∈ [1.5, 2.0] m/s.

    All angles are in **degrees** in this config; converted to radians at sampling time.
    """
    theta_min_deg: float = 55.0     # min approach angle (degrees)
    theta_max_deg: float = 110.0    # max approach angle (degrees)
    v_min: float = 1.5              # min walk speed (m/s)
    v_max: float = 2.0              # max walk speed (m/s)
    disturbed_frac: float = 0.4     # fraction of peds that are disturbed (40 % by default)
    d_spawn_min: float = 8.0        # min spawn distance (m)
    d_spawn_max: float = 20.0       # max spawn distance (m)


NOMINAL_DISTURBANCE = DisturbanceProfile(
    theta_min_deg=10.0, theta_max_deg=40.0,
    v_min=0.5, v_max=1.2,
    disturbed_frac=0.4,
    d_spawn_min=10.0, d_spawn_max=20.0,
)

FUZZED_DISTURBANCE = DisturbanceProfile(
    theta_min_deg=55.0, theta_max_deg=110.0,
    v_min=1.5, v_max=2.0,
    disturbed_frac=0.4,
    d_spawn_min=8.0, d_spawn_max=20.0,
)


@dataclass
class PedestrianConfig:
    """Per-pedestrian disturbance vector τ_i."""
    ped_id: int
    d_spawn: float          # metres from ego path (used only when no explicit spawn_xyz)
    theta_approach: float   # radians relative to ego heading
    v_init: float           # m/s
    disturbed: bool = False # True → prescribed broadside crossing via WalkerControl
    spawn_x: Optional[float] = None  # absolute world x (if provided)
    spawn_y: Optional[float] = None  # absolute world y
    spawn_z: Optional[float] = None  # absolute world z


@dataclass
class ScenarioSpec:
    """Full specification for one CARLA scenario."""
    scenario_id: int
    n_ped: int
    tracker: str            # "cv", "sfekf", "sfekf_simplex"
    v_ego_kmh: float
    tau_safe: float
    seed: int
    pedestrians: List[PedestrianConfig] = field(default_factory=list)


@dataclass
class PedestrianGT:
    """Ground-truth record for one pedestrian at one timestep."""
    ped_id: int
    t: float
    x: float
    y: float
    vx: float
    vy: float


@dataclass
class EgoGT:
    """Ground-truth ego state at one timestep."""
    t: float
    x: float
    y: float
    vx: float
    vy: float
    speed_kmh: float


@dataclass
class ScenarioResult:
    """Everything recorded from one CARLA scenario run."""
    scenario_id: int
    spec: dict
    collision: bool = False
    collision_time: float = -1.0
    min_ttc: float = float("inf")
    rho_min: float = float("inf")
    ttc_trace: List[float] = field(default_factory=list)
    ego_trace: List[dict] = field(default_factory=list)
    ped_traces: Dict[int, List[dict]] = field(default_factory=dict)
    simplex_activations: int = 0
    duration_s: float = 0.0
    frames_dir: str = ""


# ═══════════════════════════════════════════════════════════════════════════════
# TTC computation (circle-circle model)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_ttc(ego_pos: np.ndarray, ego_vel: np.ndarray,
                ped_pos: np.ndarray, ped_vel: np.ndarray,
                ego_radius: float = 1.2, ped_radius: float = 0.4) -> float:
    """Time-to-collision between ego and pedestrian circles."""
    dp = ped_pos - ego_pos
    dv = ped_vel - ego_vel
    r = ego_radius + ped_radius

    a = np.dot(dv, dv)
    b = 2.0 * np.dot(dp, dv)
    c = np.dot(dp, dp) - r * r

    if c < 0:
        return 0.0

    if a < 1e-10:
        return float("inf")

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
# Scenario generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_scenario_specs(
    n_scenarios: int = 10,
    n_ped: int = 5,
    tracker: str = "sfekf",
    v_ego_kmh: float = 10.0,
    tau_safe: float = 2.0,
    base_seed: int = 42,
    disturbance: Optional[DisturbanceProfile] = None,
    spawn_locations: Optional[List[Tuple[float, float, float]]] = None,
    spawn_jitter: float = 1.0,
) -> List[ScenarioSpec]:
    """Generate N scenario specifications.

    Args:
        disturbance: Disturbance profile for non-AI (prescribed) pedestrians.
                     Defaults to FUZZED_DISTURBANCE.
        spawn_locations: List of (x, y, z) base positions for pedestrians.
                         Each scenario Monte-Carlos the position ± spawn_jitter metres.
                         n_ped is clamped to len(spawn_locations) when provided.
        spawn_jitter: Uniform jitter radius in metres applied to each base
                      position per scenario (default ±1.0 m in x and y).

    Normal (AI-controlled) pedestrians always use:
        θ^approach ∈ [-45°, 45°] (within camera FOV), v_init ∈ [0.5, 2.0] m/s.

    Disturbed (WalkerControl) pedestrians sample from the given profile.
    """
    if disturbance is None:
        disturbance = FUZZED_DISTURBANCE

    if spawn_locations is not None:
        n_ped = min(n_ped, len(spawn_locations))

    rng = np.random.default_rng(base_seed)
    specs = []

    for i in range(n_scenarios):
        seed = int(rng.integers(0, 2**31))
        ped_rng = np.random.default_rng(seed)

        peds = []
        n_disturbed = max(1, round(n_ped * disturbance.disturbed_frac))
        disturbed_ids = set(ped_rng.choice(n_ped, size=n_disturbed, replace=False))

        for j in range(n_ped):
            # Spawn position: explicit base + jitter, or ego-relative
            if spawn_locations is not None:
                bx, by, bz = spawn_locations[j]
                sx = float(bx + ped_rng.uniform(-spawn_jitter, spawn_jitter))
                sy = float(by + ped_rng.uniform(-spawn_jitter, spawn_jitter))
                sz = float(bz)
            else:
                sx, sy, sz = None, None, None

            if j in disturbed_ids:
                d_spawn = float(ped_rng.uniform(disturbance.d_spawn_min, disturbance.d_spawn_max))
                theta_approach = math.radians(float(
                    ped_rng.uniform(disturbance.theta_min_deg, disturbance.theta_max_deg)
                ))
                v_init = float(ped_rng.uniform(disturbance.v_min, disturbance.v_max))
                peds.append(PedestrianConfig(
                    ped_id=j, d_spawn=d_spawn, theta_approach=theta_approach,
                    v_init=v_init, disturbed=True,
                    spawn_x=sx, spawn_y=sy, spawn_z=sz,
                ))
            else:
                d_spawn = float(ped_rng.uniform(8, 20))
                theta_approach = float(ped_rng.uniform(-math.pi / 4, math.pi / 4))
                peds.append(PedestrianConfig(
                    ped_id=j, d_spawn=d_spawn, theta_approach=theta_approach,
                    v_init=float(ped_rng.uniform(0.5, 2.0)), disturbed=False,
                    spawn_x=sx, spawn_y=sy, spawn_z=sz,
                ))

        specs.append(ScenarioSpec(
            scenario_id=i,
            n_ped=n_ped,
            tracker=tracker,
            v_ego_kmh=v_ego_kmh,
            tau_safe=tau_safe,
            seed=seed,
            pedestrians=peds,
        ))

    return specs


# ═══════════════════════════════════════════════════════════════════════════════
# Single scenario runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_scenario(
    spec: ScenarioSpec,
    client: "carla.Client",
    output_dir: Path,
    save_video: bool = False,
    dt: float = 0.05,
    t_max: float = 30.0,
) -> ScenarioResult:
    """Run one scenario in CARLA and collect all data."""

    world = client.get_world()
    map_name = world.get_map().name.split("/")[-1]
    if map_name != "Town10HD":
        logger.info("Loading Town10HD...")
        world = client.load_world("Town10HD")
        # Wait for map to load
        time.sleep(2.0)
        world = client.get_world()

    # ── Save / restore settings ──────────────────────────────────────────
    original_settings = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = dt
    world.apply_settings(settings)

    # ── Force all traffic lights to green and freeze them ─────────────
    for tl in world.get_actors().filter("traffic.traffic_light*"):
        tl.set_state(carla.TrafficLightState.Green)
        tl.set_green_time(9999.0)
        tl.freeze(True)
    logger.info("  All traffic lights forced to green")

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    rng = np.random.default_rng(spec.seed)

    actors_to_destroy = []
    result = ScenarioResult(
        scenario_id=spec.scenario_id,
        spec=asdict(spec),
    )

    # Video output
    scenario_dir = output_dir / f"scenario_{spec.scenario_id:03d}"
    scenario_dir.mkdir(parents=True, exist_ok=True)
    video_path = scenario_dir / "video.mp4"
    vid_writer = None
    result.frames_dir = str(scenario_dir)

    try:
        # ── Ego vehicle ──────────────────────────────────────────────────
        ego_bp = bp_lib.filter("vehicle.tesla.model3")[0]
        ego_bp.set_attribute("role_name", "hero")

        # Pick a spawn point deterministically from seed
        sp_idx = int(rng.integers(0, len(spawn_points)))
        ego_spawn = spawn_points[sp_idx]
        ego = world.try_spawn_actor(ego_bp, ego_spawn)
        if ego is None:
            # Try a few more spawn points
            for offset in range(1, min(10, len(spawn_points))):
                ego = world.try_spawn_actor(
                    ego_bp, spawn_points[(sp_idx + offset) % len(spawn_points)]
                )
                if ego is not None:
                    break
        if ego is None:
            raise RuntimeError("Failed to spawn ego vehicle after multiple attempts")
        actors_to_destroy.append(ego)
        logger.info("  Ego spawned at (%.1f, %.1f)",
                     ego.get_location().x, ego.get_location().y)

        # ── Collision sensor ─────────────────────────────────────────────
        collision_bp = bp_lib.find("sensor.other.collision")
        collision_sensor = world.spawn_actor(
            collision_bp, carla.Transform(), attach_to=ego
        )
        actors_to_destroy.append(collision_sensor)

        collision_events = []

        def on_collision(event):
            other = event.other_actor
            collision_events.append({
                "time": event.timestamp,
                "other_id": other.id,
                "other_type": other.type_id,
            })

        collision_sensor.listen(on_collision)

        # ── RGB camera (for perception pipeline) ─────────────────────────
        cam_bp = bp_lib.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", "1920")
        cam_bp.set_attribute("image_size_y", "1080")
        cam_bp.set_attribute("fov", "90")
        cam_transform = carla.Transform(
            carla.Location(x=1.5, z=2.0),
            carla.Rotation(pitch=-15.0),
        )
        camera = world.spawn_actor(cam_bp, cam_transform, attach_to=ego)
        actors_to_destroy.append(camera)

        # Video writer (initialised on first frame)
        import cv2

        def on_image(image):
            nonlocal vid_writer
            if not save_video:
                return
            arr = np.frombuffer(image.raw_data, dtype=np.uint8)
            arr = arr.reshape((image.height, image.width, 4))[:, :, :3]  # BGRA→BGR
            if vid_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vid_writer = cv2.VideoWriter(
                    str(video_path), fourcc, 1.0 / dt,
                    (image.width, image.height),
                )
            vid_writer.write(arr)

        camera.listen(on_image)

        # ── Prepare pedestrian spawning: 2 pre-spawned, rest staggered ───
        # Disturbed peds (~20%): prescribed velocity via WalkerControl (broadside crossing)
        # Normal peds (~80%): AI controller follows nav mesh
        walker_bps = bp_lib.filter("walker.pedestrian.*")
        walkers = []
        walker_configs: List[Tuple["carla.Actor", PedestrianConfig]] = []  # disturbed peds only
        ai_controllers = []  # (controller, ped_cfg) for normal peds
        ped_id_map = {}  # CARLA actor ID -> our ped_id

        # Schedule: spawn one ped every spawn_interval seconds
        spawn_interval = max(2.0, (t_max * 0.6) / max(spec.n_ped, 1))
        spawn_queue = list(enumerate(spec.pedestrians))  # (idx, ped_cfg)
        next_spawn_idx = 0

        world.tick()  # tick so ego transform is settled

        def _is_in_front(ego_transform, location: carla.Location, min_forward: float = 5.0) -> bool:
            """True if location is ahead of ego (positive along vehicle forward, at least min_forward m)."""
            fwd = ego_transform.get_forward_vector()
            dx = location.x - ego_transform.location.x
            dy = location.y - ego_transform.location.y
            return dx * fwd.x + dy * fwd.y >= min_forward

        def _spawn_one_ped(ped_cfg):
            """Spawn a pedestrian at explicit (x,y,z) or in front of ego via d_spawn/theta."""
            ego_t = ego.get_transform()
            fwd = ego_t.get_forward_vector()
            right_x, right_y = -fwd.y, fwd.x

            # ── Determine desired location ───────────────────────────────
            if ped_cfg.spawn_x is not None:
                # Explicit world coordinates (already jittered in generate_scenario_specs)
                desired_loc = carla.Location(
                    x=ped_cfg.spawn_x, y=ped_cfg.spawn_y, z=ped_cfg.spawn_z,
                )
            else:
                # Ego-relative from d_spawn / theta_approach
                offset_fwd = ped_cfg.d_spawn * math.cos(ped_cfg.theta_approach)
                offset_lat = ped_cfg.d_spawn * math.sin(ped_cfg.theta_approach)
                desired_loc = carla.Location(
                    x=ego_t.location.x + fwd.x * offset_fwd + right_x * offset_lat,
                    y=ego_t.location.y + fwd.y * offset_fwd + right_y * offset_lat,
                    z=ego_t.location.z,
                )

            spawn_loc = None
            min_forward = 5.0

            # Prefer sidewalk/shoulder waypoint only if it stays in front
            waypoint = world.get_map().get_waypoint(
                desired_loc, project_to_road=True,
                lane_type=carla.LaneType.Sidewalk | carla.LaneType.Shoulder,
            )
            if waypoint is not None:
                wp_loc = waypoint.transform.location
                if _is_in_front(ego_t, wp_loc, min_forward):
                    dx = wp_loc.x - desired_loc.x
                    dy = wp_loc.y - desired_loc.y
                    if math.sqrt(dx * dx + dy * dy) < 15.0:
                        spawn_loc = wp_loc + carla.Location(z=0.5)

            # For explicit coords: use desired_loc directly if sidewalk snap failed
            if spawn_loc is None and ped_cfg.spawn_x is not None:
                any_wp = world.get_map().get_waypoint(desired_loc, project_to_road=True)
                z = any_wp.transform.location.z + 0.5 if any_wp else desired_loc.z + 0.5
                spawn_loc = carla.Location(x=desired_loc.x, y=desired_loc.y, z=z)

            # For ego-relative: search along forward direction for sidewalk/shoulder
            if spawn_loc is None:
                for dist in (8, 10, 12, 15, 18, 20, 25):
                    for lat in (0.0, -3.0, 3.0, -6.0, 6.0):
                        cand = carla.Location(
                            x=ego_t.location.x + fwd.x * dist + right_x * lat,
                            y=ego_t.location.y + fwd.y * dist + right_y * lat,
                            z=ego_t.location.z,
                        )
                        wp = world.get_map().get_waypoint(
                            cand, project_to_road=True,
                            lane_type=carla.LaneType.Sidewalk | carla.LaneType.Shoulder,
                        )
                        if wp is not None and _is_in_front(ego_t, wp.transform.location, min_forward):
                            spawn_loc = wp.transform.location + carla.Location(z=0.5)
                            break
                    if spawn_loc is not None:
                        break

            # Fallback: use desired_loc with road z if it is in front
            if spawn_loc is None:
                any_wp = world.get_map().get_waypoint(desired_loc, project_to_road=True)
                if any_wp is not None and _is_in_front(ego_t, desired_loc, min_forward):
                    spawn_loc = carla.Location(
                        x=desired_loc.x, y=desired_loc.y,
                        z=any_wp.transform.location.z + 0.5,
                    )
                elif _is_in_front(ego_t, desired_loc, min_forward):
                    spawn_loc = carla.Location(
                        x=desired_loc.x, y=desired_loc.y,
                        z=ego_t.location.z + 0.5,
                    )

            if spawn_loc is None:
                logger.warning("  No valid spawn found for pedestrian %d", ped_cfg.ped_id)
                return

            spawn_transform = carla.Transform(
                spawn_loc, carla.Rotation(yaw=float(rng.uniform(0, 360)))
            )

            walker_bp = walker_bps[int(rng.integers(0, len(walker_bps)))]
            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")

            walker = world.try_spawn_actor(walker_bp, spawn_transform)
            if walker is None:
                fallback_loc = world.get_random_location_from_navigation()
                if fallback_loc is not None:
                    walker = world.try_spawn_actor(walker_bp, carla.Transform(
                        fallback_loc + carla.Location(z=0.5),
                        spawn_transform.rotation,
                    ))
            if walker is None:
                logger.warning("  Failed to spawn pedestrian %d", ped_cfg.ped_id)
                return

            walkers.append(walker)
            actors_to_destroy.append(walker)
            ped_id_map[walker.id] = ped_cfg.ped_id

            # Print spawn info (use spawn_transform — actor hasn't ticked yet)
            mode = "DISTURBED" if ped_cfg.disturbed else "AI"
            logger.info("  [%s] ped %d  pos=(%.1f, %.1f, %.1f)  yaw=%.1f°  "
                        "θ_approach=%.1f°  v_init=%.2f m/s  (%d/%d)",
                        mode, ped_cfg.ped_id,
                        spawn_transform.location.x,
                        spawn_transform.location.y,
                        spawn_transform.location.z,
                        spawn_transform.rotation.yaw,
                        math.degrees(ped_cfg.theta_approach), ped_cfg.v_init,
                        len(walkers), spec.n_ped)

            if ped_cfg.disturbed:
                walker_configs.append((walker, ped_cfg))
            else:
                ctrl_bp = bp_lib.find("controller.ai.walker")
                ctrl = world.spawn_actor(ctrl_bp, carla.Transform(), walker)
                ai_controllers.append((ctrl, ped_cfg))
                actors_to_destroy.append(ctrl)
                world.tick()
                ctrl.start()
                ctrl.set_max_speed(ped_cfg.v_init)
                ctrl.go_to_location(ego.get_location())

        # Spawn at least two pedestrians before the scenario begins
        sim_time = 0.0
        n_pre_spawn = min(2, len(spawn_queue))
        for i in range(n_pre_spawn):
            _, ped_cfg = spawn_queue[i]
            _spawn_one_ped(ped_cfg)
        next_spawn_idx = n_pre_spawn
        # Brief settling so pre-spawned peds are active when scenario starts
        settle_ticks = max(1, int(1.0 / dt))
        for _ in range(settle_ticks):
            world.tick()

        # ── Ego control: smooth PID speed controller ─────────────────────
        target_speed_ms = spec.v_ego_kmh / 3.6
        # PID gains for throttle/brake
        _kp, _ki, _kd = 0.8, 0.05, 0.1
        _speed_integral = 0.0
        _speed_prev_err = 0.0

        # ── Main simulation loop ─────────────────────────────────────────
        n_steps = int(t_max / dt)
        emergency_active = False

        t0_wall = time.time()
        for step in range(n_steps):
            world.tick()
            sim_time = (step + 1) * dt

            # ── Staggered pedestrian spawning ─────────────────────────
            if next_spawn_idx < len(spawn_queue):
                next_spawn_time = next_spawn_idx * spawn_interval
                if sim_time >= next_spawn_time:
                    _, ped_cfg = spawn_queue[next_spawn_idx]
                    _spawn_one_ped(ped_cfg)
                    next_spawn_idx += 1

            # ── PID speed control + traffic-law compliance ─────────────────
            # Lane-following steering (always applied)
            ego_ctrl = carla.VehicleControl()
            ego_wp = world.get_map().get_waypoint(
                ego.get_transform().location, project_to_road=True)
            if ego_wp is not None:
                wp_yaw = math.radians(ego_wp.transform.rotation.yaw)
                ego_yaw = math.radians(ego.get_transform().rotation.yaw)
                yaw_err = wp_yaw - ego_yaw
                yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi
                ego_ctrl.steer = max(-1.0, min(1.0, yaw_err * 2.0))

            if emergency_active:
                ego_ctrl.throttle = 0.0
                ego_ctrl.brake = 1.0
            else:
                # PID for target speed (we stay under speed limit via --v-ego)
                cur_v = ego.get_velocity()
                cur_speed = math.sqrt(cur_v.x**2 + cur_v.y**2 + cur_v.z**2)
                err = target_speed_ms - cur_speed
                _speed_integral += err * dt
                _speed_integral = max(-2.0, min(2.0, _speed_integral))
                derr = (err - _speed_prev_err) / dt
                _speed_prev_err = err
                pid_out = _kp * err + _ki * _speed_integral + _kd * derr
                if pid_out >= 0:
                    ego_ctrl.throttle = min(1.0, pid_out)
                    ego_ctrl.brake = 0.0
                else:
                    ego_ctrl.throttle = 0.0
                    ego_ctrl.brake = min(1.0, -pid_out)
            ego.apply_control(ego_ctrl)

            # Prescribed velocity for disturbed peds only (broadside crossing from τ)
            ego_t = ego.get_transform()
            fwd = ego_t.get_forward_vector()
            for walker, ped_cfg in walker_configs:
                # World-frame direction = ego forward rotated by θ^approach (radians)
                c, s = math.cos(ped_cfg.theta_approach), math.sin(ped_cfg.theta_approach)
                dx = fwd.x * c - fwd.y * s
                dy = fwd.x * s + fwd.y * c
                n = math.sqrt(dx * dx + dy * dy) or 1.0
                dx, dy = dx / n, dy / n
                wc = carla.WalkerControl()
                wc.direction = carla.Vector3D(x=dx, y=dy, z=0.0)
                wc.speed = ped_cfg.v_init
                wc.jump = False
                walker.apply_control(wc)

            # Ego state
            ego_v = ego.get_velocity()
            ego_pos = np.array([ego_t.location.x, ego_t.location.y])
            ego_vel = np.array([ego_v.x, ego_v.y])
            ego_speed = math.sqrt(ego_v.x**2 + ego_v.y**2 + ego_v.z**2) * 3.6

            result.ego_trace.append({
                "t": round(sim_time, 3),
                "x": round(ego_t.location.x, 3),
                "y": round(ego_t.location.y, 3),
                "vx": round(ego_v.x, 3),
                "vy": round(ego_v.y, 3),
                "speed_kmh": round(ego_speed, 1),
            })

            # Per-pedestrian ground truth + TTC
            step_min_ttc = float("inf")
            for walker in walkers:
                w_t = walker.get_transform()
                w_v = walker.get_velocity()
                ped_pos = np.array([w_t.location.x, w_t.location.y])
                ped_vel = np.array([w_v.x, w_v.y])

                pid = ped_id_map.get(walker.id, -1)
                if pid not in result.ped_traces:
                    result.ped_traces[pid] = []
                result.ped_traces[pid].append({
                    "t": round(sim_time, 3),
                    "x": round(w_t.location.x, 3),
                    "y": round(w_t.location.y, 3),
                    "vx": round(w_v.x, 3),
                    "vy": round(w_v.y, 3),
                })

                # TTC
                ttc = compute_ttc(ego_pos, ego_vel, ped_pos, ped_vel)
                step_min_ttc = min(step_min_ttc, ttc)

                # Physical collision check (distance < 2 m)
                dist = np.linalg.norm(ped_pos - ego_pos)
                if dist < 2.0 and not result.collision:
                    result.collision = True
                    result.collision_time = sim_time

            result.min_ttc = min(result.min_ttc, step_min_ttc)
            result.ttc_trace.append(round(step_min_ttc, 4))

            # ── Simplex intervention ─────────────────────────────────────
            if spec.tracker == "sfekf_simplex" and step_min_ttc < spec.tau_safe:
                if not emergency_active:
                    emergency_active = True
                    result.simplex_activations += 1
                    _speed_integral = 0.0  # reset PID state
                ctrl = carla.VehicleControl()
                ctrl.brake = 1.0
                ctrl.throttle = 0.0
                ego.apply_control(ctrl)
            elif emergency_active and step_min_ttc > spec.tau_safe * 1.5:
                emergency_active = False
                _speed_integral = 0.0
                _speed_prev_err = 0.0

            # Progress logging every 5 seconds
            if (step + 1) % int(5.0 / dt) == 0:
                logger.info("    t=%.0fs  min_ttc=%.2f  ego_speed=%.1f km/h",
                            sim_time, step_min_ttc, ego_speed)

        result.duration_s = time.time() - t0_wall
        result.rho_min = result.min_ttc - spec.tau_safe

        # Check CARLA collision sensor events
        if collision_events:
            # Filter for pedestrian collisions
            ped_collisions = [
                e for e in collision_events
                if "walker" in e.get("other_type", "")
            ]
            if ped_collisions and not result.collision:
                result.collision = True

    except Exception as e:
        logger.error("  Scenario %d failed: %s", spec.scenario_id, e)
        raise

    finally:
        # Release video writer
        if vid_writer is not None:
            vid_writer.release()
            logger.info("  Video saved to %s", video_path)

        # Stop AI controllers first
        for ctrl, _ in ai_controllers:
            try:
                ctrl.stop()
            except Exception:
                pass

        # Destroy all actors (reverse order)
        for actor in reversed(actors_to_destroy):
            try:
                actor.destroy()
            except Exception:
                pass

        # Restore original settings
        try:
            world.apply_settings(original_settings)
        except Exception:
            pass

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Run CARLA scenarios for Simplex-Track validation"
    )
    parser.add_argument("--n-scenarios", type=int, default=10,
                        help="Number of scenarios to run (default: 10)")
    parser.add_argument("--n-ped", type=int, default=5,
                        help="Number of pedestrians per scenario (default: 5)")
    parser.add_argument("--tracker", type=str, default="sfekf",
                        choices=["cv", "sfekf", "sfekf_simplex"],
                        help="Tracker configuration (default: sfekf)")
    parser.add_argument("--v-ego", type=float, default=10.0,
                        help="Ego speed in km/h (default: 10)")
    parser.add_argument("--tau-safe", type=float, default=2.0,
                        help="Safety threshold in seconds (default: 2.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--host", type=str, default="127.0.0.1",
                        help="CARLA server host")
    parser.add_argument("--port", type=int, default=2000,
                        help="CARLA server port")
    parser.add_argument("--save-video", action="store_true",
                        help="Save per-frame PNGs for perception pipeline")
    parser.add_argument("--disturbance", type=str, default="fuzzed",
                        choices=["nominal", "fuzzed"],
                        help="Disturbance profile for prescribed pedestrians "
                             "(nominal: gentle crossings, rare failures; "
                             "fuzzed: broadside crossings, frequent failures)")
    parser.add_argument("--disturbed-frac", type=float, default=None,
                        help="Override fraction of disturbed peds (0.0-1.0)")
    parser.add_argument("--theta-min", type=float, default=None,
                        help="Override min approach angle for disturbed peds (degrees)")
    parser.add_argument("--theta-max", type=float, default=None,
                        help="Override max approach angle for disturbed peds (degrees)")
    parser.add_argument("--v-ped-min", type=float, default=None,
                        help="Override min speed for disturbed peds (m/s)")
    parser.add_argument("--v-ped-max", type=float, default=None,
                        help="Override max speed for disturbed peds (m/s)")
    parser.add_argument("--spawn-locations", type=str, default=None,
                        help="JSON file with pedestrian base positions: "
                             '[[x1,y1,z1], [x2,y2,z2], ...]. '
                             "n_ped is clamped to the number of entries.")
    parser.add_argument("--spawn-jitter", type=float, default=1.0,
                        help="Uniform ± jitter in metres applied to each base "
                             "spawn position per scenario (default: 1.0)")
    parser.add_argument("--output-dir", type=str, default="runs/carla_scenarios",
                        help="Output directory for results")
    args = parser.parse_args()

    # ── Connect to CARLA ─────────────────────────────────────────────────
    logger.info("Connecting to CARLA at %s:%d ...", args.host, args.port)
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    server_version = client.get_server_version()
    logger.info("Connected. Server version: %s", server_version)

    # ── Build disturbance profile ────────────────────────────────────────
    profile = NOMINAL_DISTURBANCE if args.disturbance == "nominal" else FUZZED_DISTURBANCE
    # Apply any per-field overrides
    if args.disturbed_frac is not None:
        profile = DisturbanceProfile(**{**asdict(profile), "disturbed_frac": args.disturbed_frac})
    if args.theta_min is not None:
        profile = DisturbanceProfile(**{**asdict(profile), "theta_min_deg": args.theta_min})
    if args.theta_max is not None:
        profile = DisturbanceProfile(**{**asdict(profile), "theta_max_deg": args.theta_max})
    if args.v_ped_min is not None:
        profile = DisturbanceProfile(**{**asdict(profile), "v_min": args.v_ped_min})
    if args.v_ped_max is not None:
        profile = DisturbanceProfile(**{**asdict(profile), "v_max": args.v_ped_max})

    logger.info("Disturbance profile (%s): θ=[%.0f°,%.0f°]  v=[%.1f,%.1f] m/s  "
                "frac=%.0f%%  d=[%.0f,%.0f] m",
                args.disturbance, profile.theta_min_deg, profile.theta_max_deg,
                profile.v_min, profile.v_max, profile.disturbed_frac * 100,
                profile.d_spawn_min, profile.d_spawn_max)

    # ── Load spawn locations (if provided) ──────────────────────────────
    spawn_locs = None
    if args.spawn_locations is not None:
        with open(args.spawn_locations) as f:
            spawn_locs = [tuple(p) for p in json.load(f)]
        logger.info("Loaded %d spawn locations from %s (jitter=±%.1f m)",
                     len(spawn_locs), args.spawn_locations, args.spawn_jitter)

    # ── Generate scenarios ───────────────────────────────────────────────
    specs = generate_scenario_specs(
        n_scenarios=args.n_scenarios,
        n_ped=args.n_ped,
        tracker=args.tracker,
        v_ego_kmh=args.v_ego,
        tau_safe=args.tau_safe,
        base_seed=args.seed,
        disturbance=profile,
        spawn_locations=spawn_locs,
        spawn_jitter=args.spawn_jitter,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save specs
    specs_path = output_dir / "scenario_specs.json"
    with open(specs_path, "w") as f:
        json.dump([asdict(s) for s in specs], f, indent=2)
    logger.info("Saved %d scenario specs to %s", len(specs), specs_path)

    # ── Run scenarios ────────────────────────────────────────────────────
    all_results = []
    n_collisions = 0
    t0 = time.time()

    for i, spec in enumerate(specs):
        logger.info("═══ Scenario %d/%d (seed=%d, n_ped=%d, tracker=%s) ═══",
                     i + 1, len(specs), spec.seed, spec.n_ped, spec.tracker)

        result = run_single_scenario(
            spec=spec,
            client=client,
            output_dir=output_dir,
            save_video=args.save_video,
        )

        if result.collision:
            n_collisions += 1
            logger.info("  ⚠ COLLISION at t=%.1fs (min_ttc=%.2f, ρ=%.2f)",
                         result.collision_time, result.min_ttc, result.rho_min)
        else:
            logger.info("  ✓ Safe (min_ttc=%.2f, ρ=%.2f)", result.min_ttc, result.rho_min)

        all_results.append(result)

        # Save incrementally
        results_path = output_dir / "scenario_results.json"
        _save_results(all_results, results_path)

    elapsed = time.time() - t0

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("COMPLETE: %d scenarios in %.0fs (%.1f s/scenario)",
                len(specs), elapsed, elapsed / max(len(specs), 1))
    logger.info("Collisions: %d / %d (%.1f%%)",
                n_collisions, len(specs), 100 * n_collisions / max(len(specs), 1))
    logger.info("Min ρ_min: %.3f",
                min(r.rho_min for r in all_results))
    logger.info("Results saved to %s", output_dir / "scenario_results.json")

    if args.save_video:
        logger.info("Videos saved to %s/scenario_*/video.mp4", output_dir)
        logger.info(
            "\nNext step: run perception on saved video:\n"
            "  python perception/detect_dual_tracking.py \\\n"
            "    --source runs/carla_scenarios/scenario_000/video.mp4 \\\n"
            "    --weights perception/yolov9/weights/yolov9-c.pt \\\n"
            "    --save_plot_name scenario_000"
        )


def _save_results(results: List[ScenarioResult], path: Path):
    """Save results to JSON (convert non-serializable types)."""
    data = []
    for r in results:
        d = {
            "scenario_id": r.scenario_id,
            "collision": r.collision,
            "collision_time": r.collision_time,
            "min_ttc": r.min_ttc if r.min_ttc != float("inf") else None,
            "rho_min": r.rho_min if r.rho_min != float("inf") else None,
            "ttc_trace": r.ttc_trace,
            "simplex_activations": r.simplex_activations,
            "duration_s": round(r.duration_s, 1),
            "n_ped_spawned": len(r.ped_traces),
            "spec": r.spec,
        }
        # Omit large traces to keep file manageable; save separately if needed
        d["ego_trace_length"] = len(r.ego_trace)
        d["ttc_trace_length"] = len(r.ttc_trace)
        data.append(d)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
