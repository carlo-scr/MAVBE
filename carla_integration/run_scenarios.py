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
  - Ego drives at v_ego km/h in Town10HD
  - N_ped pedestrians spawn with (d_spawn, θ_spawn, θ_approach, v_init)
  - Duration: 30 s at 20 Hz

Pedestrians are controlled via WalkerControl with Helbing-Molnar social-force
repulsion between nearby walkers, producing realistic curved avoidance paths.
τ = (d, θ_spawn, θ_approach, v):
  NOMINAL  – wide fan, any walk direction; failures rare
             d ∈ [5, 35] m,  θ_spawn ∈ [-75°, 75°],
             θ_approach ∈ [0°, 360°],  v ∈ [0.5, 2.0] m/s
  FUZZED   – one-sided spawn, broadside crossings; failures frequent
             d ∈ [5, 14] m,  θ_spawn ∈ [-75°, -20°],
             θ_approach ∈ [55°, 110°],  v ∈ [1.5, 2.0] m/s
All pedestrians spawn before the scenario begins.
All traffic lights are forced green.

Usage:
    python -m carla_integration.run_scenarios --n-scenarios 10
    python -m carla_integration.run_scenarios --n-scenarios 10 --disturbance nominal
    python -m carla_integration.run_scenarios --n-scenarios 10 --disturbance fuzzed
    python -m carla_integration.run_scenarios --n-scenarios 10 --n-ped 7 --tracker sf_ct_ekf_simplex
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

from carla_integration.online_trackers import create_tracker

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
    """Disturbance distribution τ = (N_ped, {d_spawn, θ_spawn, θ_approach, v_init}).

    Pedestrians are driven by WalkerControl with Helbing-Molnar social-force
    repulsion between nearby pedestrians, producing realistic curved paths.

    Parameters (all angles in **degrees**; converted to radians at sampling time):
        d_spawn        – distance from ego to spawn point
        θ_spawn        – angular offset from ego heading for the spawn position
                         (positive = right of ego, negative = left)
        θ_approach     – walk direction relative to ego heading
                         0° = same direction as car, 90° = left-to-right crossing,
                         180° = head-on toward car, 270° = right-to-left crossing
        v_init         – initial walking speed

    NOMINAL  – wide fan, any walk direction; failures rare.
               d ∈ [5, 35] m, θ_spawn ∈ [-75°, 75°],
               θ_approach ∈ [0°, 360°], v ∈ [0.5, 2.0] m/s.

    FUZZED   – peds spawn to one side, walk broadside across the road; failures frequent.
               d ∈ [5, 14] m, θ_spawn ∈ [-75°, -20°],
               θ_approach ∈ [55°, 110°], v ∈ [1.5, 2.0] m/s.
    """
    d_spawn_min: float = 5.0
    d_spawn_max: float = 35.0
    theta_spawn_min_deg: float = -75.0
    theta_spawn_max_deg: float = 75.0
    theta_approach_min_deg: float = 0.0
    theta_approach_max_deg: float = 360.0
    v_min: float = 0.5
    v_max: float = 2.0


NOMINAL_DISTURBANCE = DisturbanceProfile(
    d_spawn_min=5.0, d_spawn_max=35.0,
    theta_spawn_min_deg=-75.0, theta_spawn_max_deg=75.0,
    theta_approach_min_deg=0.0, theta_approach_max_deg=360.0,
    v_min=0.5, v_max=2.0,
)

FUZZED_DISTURBANCE = DisturbanceProfile(
    d_spawn_min=5.0, d_spawn_max=14.0,
    theta_spawn_min_deg=-75.0, theta_spawn_max_deg=-20.0,
    theta_approach_min_deg=55.0, theta_approach_max_deg=110.0,
    v_min=1.5, v_max=2.0,
)


@dataclass
class PedestrianConfig:
    """Per-pedestrian disturbance vector τ_i.  WalkerControl + social-force repulsion."""
    ped_id: int
    d_spawn: float          # metres from ego to spawn point
    theta_spawn: float      # radians – angular offset from ego heading for spawn position
    theta_approach: float   # radians – walk direction relative to ego heading
    v_init: float           # m/s
    spawn_x: Optional[float] = None
    spawn_y: Optional[float] = None
    spawn_z: Optional[float] = None


@dataclass
class ScenarioSpec:
    """Full specification for one CARLA scenario."""
    scenario_id: int
    n_ped: int
    tracker: str            # "cv_kf", "sf_ct_ekf", "sf_ct_ekf_simplex", etc.
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
    collision_ped_id: int = -1
    min_ttc: float = float("inf")
    rho_min: float = float("inf")
    ttc_trace: List[float] = field(default_factory=list)
    ego_trace: List[dict] = field(default_factory=list)
    ped_traces: Dict[int, List[dict]] = field(default_factory=dict)
    noisy_detections: List[dict] = field(default_factory=list)
    brake_trace: List[bool] = field(default_factory=list)
    tracker_brake_events: List[dict] = field(default_factory=list)
    n_brake_steps: int = 0
    brake_first_time: float = -1.0
    collision_despite_brake: bool = False
    simplex_activations: int = 0
    n_fp_brakes: int = 0
    ade: float = 0.0
    fde: float = 0.0
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
# Smart noise sensor (FOV + occlusion + distance-dependent noise)
# ═══════════════════════════════════════════════════════════════════════════════

class SmartNoiseSensor:
    """Simulates a realistic perception pipeline on top of CARLA ground truth.

    Pipeline per tick:
        1. Range gate     – ignore peds beyond max_range
        2. FOV culling    – ignore peds outside the camera cone
        3. Raycast        – drop fully occluded peds (simulates YOLO miss)
        4. Noise          – add distance-dependent Gaussian noise to (x, y)

    Returns measurements in **ego-relative** coordinates so they can be
    fed directly into a tracker (EKF / IMM).
    """

    def __init__(
        self,
        world: "carla.World",
        ego: "carla.Vehicle",
        ped_id_map: Dict[int, int],
        fov_degrees: float = 90.0,
        max_range: float = 50.0,
        base_noise: float = 0.2,
        alpha: float = 0.01,
    ):
        self.world = world
        self.ego = ego
        self.ped_id_map = ped_id_map
        self.half_fov = fov_degrees / 2.0
        self.max_range = max_range
        self.base_noise = base_noise
        self.alpha = alpha

        # Pre-build set of labels that count as real occluders
        self._occluder_labels = set()
        for name in ("Buildings", "Walls", "Fences", "Poles",
                      "Vegetation", "Other", "Static", "Dynamic",
                      "Bridge", "RailTrack", "GuardRail", "Water"):
            lbl = getattr(carla.CityObjectLabel, name, None)
            if lbl is not None:
                self._occluder_labels.add(lbl)

    # ── public API ────────────────────────────────────────────────────────

    def get_detections(self, sim_time: float) -> List[dict]:
        """Return noisy ego-relative detections for all visible pedestrians."""
        ego_tf = self.ego.get_transform()
        ego_loc = ego_tf.location
        ego_fwd = ego_tf.get_forward_vector()
        ego_yaw = math.radians(ego_tf.rotation.yaw)

        detections = []
        for ped in self.world.get_actors().filter("walker.pedestrian.*"):
            ped_loc = ped.get_transform().location
            pid = self.ped_id_map.get(ped.id, -1)

            dx = ped_loc.x - ego_loc.x
            dy = ped_loc.y - ego_loc.y
            distance = math.sqrt(dx * dx + dy * dy)

            if distance > self.max_range or distance < 1e-3:
                continue

            # FOV culling
            inv_d = 1.0 / distance
            dot = ego_fwd.x * dx * inv_d + ego_fwd.y * dy * inv_d
            dot = max(-1.0, min(1.0, dot))
            angle_deg = math.degrees(math.acos(dot))
            if angle_deg > self.half_fov:
                continue

            # Raycast occlusion (camera height → ped chest)
            # Only solid objects (buildings, walls, etc.) count as occluders.
            # Road, sidewalk, terrain, sky, and the ped itself are transparent.
            start = ego_loc + carla.Location(z=1.5)
            end = ped_loc + carla.Location(z=1.0)
            occluded = False
            for hit in self.world.cast_ray(start, end):
                if hit.label in self._occluder_labels:
                    occluded = True
                    break
            if occluded:
                continue

            # Distance-dependent noise: σ² = base² · (1 + α · d²)
            var = (self.base_noise ** 2) * (1.0 + self.alpha * distance * distance)
            sigma = math.sqrt(var)
            nx = np.random.normal(0.0, sigma)
            ny = np.random.normal(0.0, sigma)

            # World → ego-relative frame
            rel_x, rel_y = self._world_to_ego(
                ped_loc.x + nx, ped_loc.y + ny, ego_loc, ego_yaw,
            )

            detections.append({
                "t": round(sim_time, 3),
                "ped_id": pid,
                "x_ego": round(rel_x, 4),
                "y_ego": round(rel_y, 4),
                "distance": round(distance, 3),
                "noise_sigma": round(sigma, 4),
            })

        return detections

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _world_to_ego(
        wx: float, wy: float,
        ego_loc: "carla.Location", ego_yaw: float,
    ) -> Tuple[float, float]:
        """Transform a world-frame point into the ego vehicle's local frame."""
        dx = wx - ego_loc.x
        dy = wy - ego_loc.y
        cos_y = math.cos(-ego_yaw)
        sin_y = math.sin(-ego_yaw)
        return dx * cos_y - dy * sin_y, dx * sin_y + dy * cos_y


# ═══════════════════════════════════════════════════════════════════════════════
# Scenario generation
# ═══════════════════════════════════════════════════════════════════════════════

def generate_scenario_specs(
    n_scenarios: int = 10,
    n_ped: int = 5,
    tracker: str = "sf_ct_ekf",
    v_ego_kmh: float = 10.0,
    tau_safe: float = 2.0,
    base_seed: int = 42,
    disturbance: Optional[DisturbanceProfile] = None,
    spawn_locations: Optional[List[Tuple[float, float, float]]] = None,
    spawn_jitter: float = 1.0,
) -> List[ScenarioSpec]:
    """Generate N scenario specifications.

    Pedestrians are controlled via WalkerControl + social-force repulsion.
    τ sampled from the given disturbance profile.

    Args:
        disturbance: Distribution to sample (d_spawn, θ_spawn, θ_approach, v_init) from.
                     NOMINAL = paper's full range; FUZZED = critical failure region.
        spawn_locations: Optional list of (x, y, z) base positions.
                         Each scenario jitters ± spawn_jitter m in x,y.
                         n_ped is clamped to len(spawn_locations).
        spawn_jitter: Uniform ± jitter in metres (default 1.0 m).
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
        for j in range(n_ped):
            # Spawn position: explicit base + jitter, or ego-relative
            if spawn_locations is not None:
                bx, by, bz = spawn_locations[j]
                sx = float(bx + ped_rng.uniform(-spawn_jitter, spawn_jitter))
                sy = float(by + ped_rng.uniform(-spawn_jitter, spawn_jitter))
                sz = float(bz)
            else:
                sx, sy, sz = None, None, None

            d_spawn = float(ped_rng.uniform(disturbance.d_spawn_min, disturbance.d_spawn_max))
            theta_spawn = math.radians(float(
                ped_rng.uniform(disturbance.theta_spawn_min_deg, disturbance.theta_spawn_max_deg)
            ))
            theta_approach = math.radians(float(
                ped_rng.uniform(disturbance.theta_approach_min_deg, disturbance.theta_approach_max_deg)
            ))
            v_init = float(ped_rng.uniform(disturbance.v_min, disturbance.v_max))
            peds.append(PedestrianConfig(
                ped_id=j, d_spawn=d_spawn, theta_spawn=theta_spawn,
                theta_approach=theta_approach,
                v_init=v_init, spawn_x=sx, spawn_y=sy, spawn_z=sz,
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

        # Always use spawn point 0 so the ego starts at the same place
        # across all scenarios (deterministic comparison).
        ego_spawn = spawn_points[0]
        ego = world.try_spawn_actor(ego_bp, ego_spawn)
        if ego is None:
            # Try a few more spawn points
            for offset in range(1, min(10, len(spawn_points))):
                ego = world.try_spawn_actor(
                    ego_bp, spawn_points[offset % len(spawn_points)]
                )
                if ego is not None:
                    break
        if ego is None:
            raise RuntimeError("Failed to spawn ego vehicle after multiple attempts")
        actors_to_destroy.append(ego)

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

        # ── 4 RGB cameras (front / right / rear / left) for composite video ──
        import cv2
        import threading

        CAM_W, CAM_H = 960, 540  # each sub-frame (composite = 1920x1080)
        _cam_buffers: Dict[str, np.ndarray] = {}
        _cam_lock = threading.Lock()

        cam_configs = {
            "front": carla.Transform(
                carla.Location(x=1.5, z=2.0), carla.Rotation(pitch=-15.0, yaw=0.0)),
            "right": carla.Transform(
                carla.Location(x=0.0, y=0.5, z=2.0), carla.Rotation(pitch=-10.0, yaw=90.0)),
            "rear":  carla.Transform(
                carla.Location(x=-1.5, z=2.0), carla.Rotation(pitch=-10.0, yaw=180.0)),
            "left":  carla.Transform(
                carla.Location(x=0.0, y=-0.5, z=2.0), carla.Rotation(pitch=-10.0, yaw=-90.0)),
        }

        cameras = []
        for cam_name, cam_tf in cam_configs.items():
            cbp = bp_lib.find("sensor.camera.rgb")
            cbp.set_attribute("image_size_x", str(CAM_W))
            cbp.set_attribute("image_size_y", str(CAM_H))
            cbp.set_attribute("fov", "90")
            cam_actor = world.spawn_actor(cbp, cam_tf, attach_to=ego)
            actors_to_destroy.append(cam_actor)
            cameras.append((cam_name, cam_actor))

            def _make_cb(name):
                def _on_image(image):
                    arr = np.frombuffer(image.raw_data, dtype=np.uint8)
                    arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
                    with _cam_lock:
                        _cam_buffers[name] = arr
                return _on_image

            cam_actor.listen(_make_cb(cam_name))

        _composite_frame_count = 0

        def _write_composite_frame():
            """Stitch 4 camera feeds into a 2x2 grid and write to video."""
            nonlocal vid_writer, _composite_frame_count
            if not save_video:
                return
            with _cam_lock:
                if len(_cam_buffers) < 4:
                    return
                front = _cam_buffers["front"].copy()
                right = _cam_buffers["right"].copy()
                rear  = _cam_buffers["rear"].copy()
                left  = _cam_buffers["left"].copy()

            label_h = 30
            font = cv2.FONT_HERSHEY_SIMPLEX
            labels = {"front": "FRONT", "right": "RIGHT", "rear": "REAR", "left": "LEFT"}
            frames = {"front": front, "right": right, "rear": rear, "left": left}
            labelled = {}
            for key, frame in frames.items():
                f = frame.copy()
                cv2.putText(f, labels[key], (10, label_h), font, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                labelled[key] = f

            top_row = np.hstack([labelled["front"], labelled["right"]])
            bot_row = np.hstack([labelled["rear"],  labelled["left"]])
            composite = np.vstack([top_row, bot_row])

            if vid_writer is None:
                h, w = composite.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vid_writer = cv2.VideoWriter(str(video_path), fourcc, 1.0 / dt, (w, h))
            vid_writer.write(composite)
            _composite_frame_count += 1

        # ── Prepare pedestrian spawning (all at once, before scenario) ────
        # All peds use WalkerControl with social-force repulsion
        walker_bps = bp_lib.filter("walker.pedestrian.*")
        walkers = []
        walker_configs: List[Tuple["carla.Actor", PedestrianConfig]] = []
        ped_id_map = {}  # CARLA actor ID -> our ped_id

        world.tick()  # tick so ego transform is settled
        ego_loc = ego.get_transform().location
        logger.info("  Ego spawned at (%.1f, %.1f, %.1f)  yaw=%.1f°",
                     ego_loc.x, ego_loc.y, ego_loc.z,
                     ego.get_transform().rotation.yaw)

        def _spawn_one_ped(ped_cfg):
            """Spawn a pedestrian directly from the disturbance model.

            Position = ego_pos + d_spawn at angle theta_spawn (no sidewalk snapping).
            Walk direction is controlled per-tick via WalkerControl + social force.
            """
            ego_t = ego.get_transform()
            fwd = ego_t.get_forward_vector()
            right_x, right_y = -fwd.y, fwd.x

            if ped_cfg.spawn_x is not None:
                desired_x = ped_cfg.spawn_x
                desired_y = ped_cfg.spawn_y
            else:
                # Spawn at d_spawn metres from ego, offset by theta_spawn from forward
                fwd_comp = ped_cfg.d_spawn * math.cos(ped_cfg.theta_spawn)
                lat_comp = ped_cfg.d_spawn * math.sin(ped_cfg.theta_spawn)
                desired_x = ego_t.location.x + fwd.x * fwd_comp + right_x * lat_comp
                desired_y = ego_t.location.y + fwd.y * fwd_comp + right_y * lat_comp

            # Get ground z from the map
            probe = carla.Location(x=desired_x, y=desired_y, z=ego_t.location.z + 5.0)
            wp = world.get_map().get_waypoint(probe, project_to_road=True)
            z = wp.transform.location.z + 0.5 if wp is not None else ego_t.location.z + 0.5

            spawn_loc = carla.Location(x=desired_x, y=desired_y, z=z)
            spawn_transform = carla.Transform(
                spawn_loc, carla.Rotation(yaw=float(rng.uniform(0, 360)))
            )

            walker_bp = walker_bps[int(rng.integers(0, len(walker_bps)))]
            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")

            walker = world.try_spawn_actor(walker_bp, spawn_transform)
            if walker is None:
                # Nudge laterally if spawn overlaps another actor
                for nudge in (2.0, -2.0, 4.0, -4.0):
                    nudged_loc = carla.Location(
                        x=desired_x + right_x * nudge,
                        y=desired_y + right_y * nudge,
                        z=z,
                    )
                    walker = world.try_spawn_actor(walker_bp, carla.Transform(
                        nudged_loc, spawn_transform.rotation,
                    ))
                    if walker is not None:
                        spawn_loc = nudged_loc
                        break

            if walker is None:
                logger.warning("  Failed to spawn pedestrian %d", ped_cfg.ped_id)
                return

            walkers.append(walker)
            actors_to_destroy.append(walker)
            ped_id_map[walker.id] = ped_cfg.ped_id

            logger.info("  ped %d  spawn=(%.1f, %.1f, %.1f)  "
                        "θ_spawn=%.1f°  θ_approach=%.1f°  v=%.2f m/s  d=%.1f m  (%d/%d)",
                        ped_cfg.ped_id,
                        spawn_loc.x, spawn_loc.y, spawn_loc.z,
                        math.degrees(ped_cfg.theta_spawn),
                        math.degrees(ped_cfg.theta_approach), ped_cfg.v_init,
                        ped_cfg.d_spawn, len(walkers), spec.n_ped)

            walker_configs.append((walker, ped_cfg))

        # Spawn all pedestrians before the scenario begins
        sim_time = 0.0
        for ped_cfg in spec.pedestrians:
            _spawn_one_ped(ped_cfg)

        # Brief settling so all walkers are active when scenario starts
        settle_ticks = max(1, int(1.0 / dt))
        for _ in range(settle_ticks):
            world.tick()

        # Log actual settled positions (after physics has placed them)
        for walker, ped_cfg in walker_configs:
            wt = walker.get_transform()
            logger.info("  ped %d settled at (%.1f, %.1f, %.1f)",
                        ped_cfg.ped_id, wt.location.x, wt.location.y, wt.location.z)

        # ── Smart noise sensor (perception model) ──────────────────────
        noise_sensor = SmartNoiseSensor(
            world=world, ego=ego, ped_id_map=ped_id_map,
            fov_degrees=90.0, max_range=50.0,
            base_noise=0.2, alpha=0.01,
        )

        # ── Online tracker (fed with noisy detections) ────────────────────
        tracker = create_tracker(spec.tracker, dt, tau_safe=spec.tau_safe)
        braking_active = False
        brake_hold_counter = 0
        BRAKE_HOLD_STEPS = 40  # hold brake ≥ 2.0 s after last tracker brake command

        # ── ADE/FDE accumulators ──────────────────────────────────────────
        _ade_errors: List[float] = []
        _fde_last_errors: Dict[int, float] = {}
        _tracker_history: List[dict] = []  # per-timestep tracker states + cov
        FP_TTC_THRESHOLD = 5.0  # GT TTC above this → braking is a false positive

        # ── Ego control: smooth PID speed controller ─────────────────────
        target_speed_ms = spec.v_ego_kmh / 3.6
        _kp, _ki, _kd = 0.8, 0.05, 0.1
        _speed_integral = 0.0
        _speed_prev_err = 0.0

        # ── Main simulation loop ─────────────────────────────────────────
        n_steps = int(t_max / dt)
        POST_COLLISION_S = 5.0
        max_steps = n_steps + int(POST_COLLISION_S / dt)

        t0_wall = time.time()
        for step in range(max_steps):
            world.tick()
            _write_composite_frame()
            sim_time = (step + 1) * dt

            # ── Read ego state ────────────────────────────────────────────
            ego_t = ego.get_transform()
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

            # ── WalkerControl with social-force repulsion ─────────────────
            # Helbing-Molnar parameters
            SF_A = 2.1    # repulsion strength (N)
            SF_B = 0.3    # range scale (m)
            SF_R = 0.5    # pedestrian radius (m)
            SF_CLAMP = 3.0  # max force magnitude to avoid instability
            EGO_RADIUS = 2.0  # effective ego vehicle radius for repulsion

            fwd = ego_t.get_forward_vector()
            ped_positions = {}
            for walker, _ in walker_configs:
                wt = walker.get_transform().location
                ped_positions[walker.id] = np.array([wt.x, wt.y])

            for walker, ped_cfg in walker_configs:
                # Base walk direction from theta_approach
                c, s = math.cos(ped_cfg.theta_approach), math.sin(ped_cfg.theta_approach)
                dx = fwd.x * c - fwd.y * s
                dy = fwd.x * s + fwd.y * c
                nmag = math.sqrt(dx * dx + dy * dy) or 1.0
                base_dir = np.array([dx / nmag, dy / nmag])

                # Accumulate social-force repulsion from other pedestrians
                pi = ped_positions[walker.id]
                sf = np.zeros(2)
                for other_id, pj in ped_positions.items():
                    if other_id == walker.id:
                        continue
                    diff = pi - pj
                    dist = np.linalg.norm(diff)
                    if dist < 1e-3 or dist > 5.0:
                        continue
                    n_ij = diff / dist
                    magnitude = SF_A * math.exp((2.0 * SF_R - dist) / SF_B)
                    sf += magnitude * n_ij

                # Repulsion from the ego vehicle (larger radius)
                diff_ego = pi - ego_pos
                dist_ego = np.linalg.norm(diff_ego)
                if dist_ego > 1e-3 and dist_ego < 8.0:
                    n_ego = diff_ego / dist_ego
                    overlap_ego = SF_R + EGO_RADIUS - dist_ego
                    mag_ego = SF_A * math.exp(overlap_ego / SF_B)
                    sf += mag_ego * n_ego

                sf_mag = np.linalg.norm(sf)
                if sf_mag > SF_CLAMP:
                    sf = sf * (SF_CLAMP / sf_mag)

                # Blend: base direction * speed + social force, then re-normalise
                desired = base_dir * ped_cfg.v_init + sf
                speed = float(np.linalg.norm(desired))
                if speed < 1e-3:
                    desired = base_dir
                    speed = ped_cfg.v_init
                else:
                    desired = desired / speed
                    speed = min(speed, ped_cfg.v_init * 1.5)

                wc = carla.WalkerControl()
                wc.direction = carla.Vector3D(x=float(desired[0]), y=float(desired[1]), z=0.0)
                wc.speed = speed
                wc.jump = False
                walker.apply_control(wc)

            # ── Per-pedestrian ground truth + TTC (for evaluation) ───────
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

                ttc = compute_ttc(ego_pos, ego_vel, ped_pos, ped_vel)
                step_min_ttc = min(step_min_ttc, ttc)

                dist = np.linalg.norm(ped_pos - ego_pos)
                if dist < 2.0 and not result.collision:
                    result.collision = True
                    result.collision_time = sim_time
                    result.collision_ped_id = pid
                    if braking_active:
                        result.collision_despite_brake = True
                        logger.warning("    t=%.2fs  COLLISION with ped %d  "
                                       "dist=%.2f m  DESPITE BRAKING  "
                                       "ego_speed=%.1f km/h",
                                       sim_time, pid, dist, ego_speed)
                    else:
                        logger.warning("    t=%.2fs  COLLISION with ped %d  "
                                       "dist=%.2f m  (no brake active)  "
                                       "ego_speed=%.1f km/h",
                                       sim_time, pid, dist, ego_speed)

            # Also check CARLA physics collision sensor (fires on actual contact)
            if not result.collision and collision_events:
                ped_hits = [e for e in collision_events
                            if "walker" in e.get("other_type", "")]
                if ped_hits:
                    hit = ped_hits[0]
                    cpid = ped_id_map.get(hit["other_id"], -1)
                    result.collision = True
                    result.collision_time = sim_time
                    result.collision_ped_id = cpid
                    result.collision_despite_brake = braking_active
                    logger.warning("    t=%.2fs  PHYSICS COLLISION with ped %d  "
                                   "(CARLA sensor)  brake=%s  ego_speed=%.1f km/h",
                                   sim_time, cpid,
                                   "ACTIVE" if braking_active else "off",
                                   ego_speed)

            result.min_ttc = min(result.min_ttc, step_min_ttc)
            result.ttc_trace.append(round(step_min_ttc, 4))

            # ── Noisy detections → world frame → tracker ─────────────────
            raw_dets = noise_sensor.get_detections(sim_time)
            result.noisy_detections.extend(raw_dets)

            ego_yaw_rad = math.radians(ego_t.rotation.yaw)
            cos_y, sin_y = math.cos(ego_yaw_rad), math.sin(ego_yaw_rad)
            for det in raw_dets:
                xe, ye = det["x_ego"], det["y_ego"]
                det["x_world"] = ego_t.location.x + xe * cos_y - ye * sin_y
                det["y_world"] = ego_t.location.y + xe * sin_y + ye * cos_y

            tracker_out = tracker.step(raw_dets, ego_pos, ego_vel, sim_time)

            result.brake_trace.append(tracker_out.brake)
            for evt in tracker_out.events:
                result.tracker_brake_events.append({
                    "t": evt.t, "reason": evt.reason,
                    "ped_id": evt.ped_id,
                    "min_pred_dist": round(evt.min_pred_dist, 3),
                    "ttc": round(evt.ttc, 3) if evt.ttc < 1e6 else None,
                })

            # ── ADE accumulation + tracker state history ─────────────────
            step_tracker_states = {}
            for trk_pid, trk_state in tracker_out.track_states.items():
                step_tracker_states[trk_pid] = trk_state
                if trk_pid in result.ped_traces and result.ped_traces[trk_pid]:
                    gt = result.ped_traces[trk_pid][-1]
                    err = math.sqrt(
                        (trk_state["x"] - gt["x"])**2
                        + (trk_state["y"] - gt["y"])**2
                    )
                    _ade_errors.append(err)
                    _fde_last_errors[trk_pid] = err

            _tracker_history.append({
                "t": round(sim_time, 3),
                "states": step_tracker_states,
            })

            # ── False-positive brake detection ────────────────────────────
            if tracker_out.brake and step_min_ttc > FP_TTC_THRESHOLD:
                if not braking_active:
                    result.n_fp_brakes += 1

            # ── Ego control: tracker brake → PID ─────────────────────────
            ego_ctrl = carla.VehicleControl()

            # Lane-following steering (always active)
            ego_wp = world.get_map().get_waypoint(
                ego_t.location, project_to_road=True)
            if ego_wp is not None:
                wp_yaw = math.radians(ego_wp.transform.rotation.yaw)
                ego_yaw_ctrl = math.radians(ego_t.rotation.yaw)
                yaw_err = wp_yaw - ego_yaw_ctrl
                yaw_err = (yaw_err + math.pi) % (2 * math.pi) - math.pi
                ego_ctrl.steer = max(-1.0, min(1.0, yaw_err * 2.0))

            # Braking decision from tracker with hold to avoid chatter.
            # Release requires BOTH: hold timer expired AND GT TTC safe.
            if tracker_out.brake:
                if not braking_active:
                    result.simplex_activations += 1
                    ped_ids = [e.ped_id for e in tracker_out.events]
                    min_dists = [e.min_pred_dist for e in tracker_out.events
                                 if e.reason == "prediction_intersect"]
                    ttcs = [e.ttc for e in tracker_out.events
                            if e.reason == "simplex_ttc" and e.ttc < 1e6]
                    parts = [f"reason={tracker_out.brake_reason}",
                             f"peds={ped_ids}"]
                    if min_dists:
                        parts.append(f"min_pred_dist={min(min_dists):.2f}m")
                    if ttcs:
                        parts.append(f"ttc={min(ttcs):.2f}s")
                    logger.info("    t=%.2fs  BRAKE ACTIVATED  %s  "
                                "ego_speed=%.1f km/h → applying brake=1.0, throttle=0.0",
                                sim_time, "  ".join(parts), ego_speed)
                braking_active = True
                brake_hold_counter = BRAKE_HOLD_STEPS
            elif braking_active:
                brake_hold_counter -= 1
                if brake_hold_counter <= 0:
                    braking_active = False
                    _speed_integral = 0.0
                    _speed_prev_err = 0.0
                    logger.info("    t=%.2fs  BRAKE RELEASED  "
                                "ego_speed=%.1f km/h → resuming PID throttle",
                                sim_time, ego_speed)

            if braking_active:
                ego_ctrl.throttle = 0.0
                ego_ctrl.brake = 1.0
                result.n_brake_steps += 1
                if result.brake_first_time < 0:
                    result.brake_first_time = sim_time
            else:
                cur_speed = math.sqrt(ego_v.x**2 + ego_v.y**2 + ego_v.z**2)
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

            # Progress logging every 5 seconds
            if (step + 1) % int(5.0 / dt) == 0:
                n_tracks = len([t for t in (tracker.tracks if hasattr(tracker, 'tracks') else {}).values()
                                if t.missed <= 3])
                logger.info("    t=%.0fs  gt_ttc=%.2f  ego=%.1f km/h  "
                            "brake=%s (ctrl: throttle=%.2f brake=%.2f)  "
                            "tracks=%d",
                            sim_time, step_min_ttc, ego_speed,
                            "ACTIVE" if braking_active else "off",
                            ego_ctrl.throttle, ego_ctrl.brake,
                            n_tracks)

            # ── Early termination: 5 s after collision, or t_max if no collision
            if result.collision and (sim_time - result.collision_time) >= POST_COLLISION_S:
                logger.info("    Ending scenario — %.1fs after collision at t=%.2fs",
                            POST_COLLISION_S, result.collision_time)
                break
            if not result.collision and step >= n_steps - 1:
                break

        # ── Compute ADE / FDE ─────────────────────────────────────────────
        result.ade = float(np.mean(_ade_errors)) if _ade_errors else 0.0
        result.fde = float(np.mean(list(_fde_last_errors.values()))) if _fde_last_errors else 0.0
        result._tracker_history = _tracker_history  # type: ignore[attr-defined]

        result.duration_s = time.time() - t0_wall
        result.rho_min = result.min_ttc - spec.tau_safe

        # End-of-scenario summary
        logger.info("  ── Scenario %d summary (%s tracker) ──", spec.scenario_id, spec.tracker)
        logger.info("    collision=%s  min_ttc=%.2f  rho_min=%.2f",
                     result.collision, result.min_ttc if result.min_ttc < 1e6 else float('inf'),
                     result.rho_min if result.rho_min < 1e6 else float('inf'))
        logger.info("    ADE=%.3f  FDE=%.3f  FP_brakes=%d",
                     result.ade, result.fde, result.n_fp_brakes)
        logger.info("    brake_activations=%d  brake_steps=%d (%.1fs)  "
                     "first_brake_t=%s  collision_despite_brake=%s",
                     result.simplex_activations, result.n_brake_steps,
                     result.n_brake_steps * dt,
                     f"{result.brake_first_time:.2f}s" if result.brake_first_time >= 0 else "none",
                     result.collision_despite_brake)

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

        # Destroy all actors (reverse order)
        for actor in reversed(actors_to_destroy):
            try:
                actor.destroy()
            except Exception:
                pass

        # Always disable synchronous mode so server doesn't freeze if we crash
        try:
            s = world.get_settings()
            s.synchronous_mode = False
            s.fixed_delta_seconds = None
            world.apply_settings(s)
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
    parser.add_argument("--tracker", type=str, default="sf_ct_ekf",
                        choices=["cv_kf", "sf_ct_ekf", "sf_ct_ekf_simplex",
                                 "sf_cv_ekf", "sf_cv_ekf_simplex"],
                        help="Tracker configuration (default: sf_ct_ekf)")
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
                        help="Disturbance profile "
                             "(nominal: wide fan, any direction, rare failures; "
                             "fuzzed: broadside crossings, frequent failures)")
    parser.add_argument("--theta-spawn-min", type=float, default=None,
                        help="Override min spawn angle (degrees)")
    parser.add_argument("--theta-spawn-max", type=float, default=None,
                        help="Override max spawn angle (degrees)")
    parser.add_argument("--theta-approach-min", type=float, default=None,
                        help="Override min approach/walk angle (degrees)")
    parser.add_argument("--theta-approach-max", type=float, default=None,
                        help="Override max approach/walk angle (degrees)")
    parser.add_argument("--v-ped-min", type=float, default=None,
                        help="Override min ped speed (m/s)")
    parser.add_argument("--v-ped-max", type=float, default=None,
                        help="Override max ped speed (m/s)")
    parser.add_argument("--load-specs", type=str, default=None,
                        help="Load scenario specs from a JSON file instead of "
                             "generating new ones.  Guarantees identical runs.")
    parser.add_argument("--spawn-locations", type=str, default=None,
                        help="JSON file with pedestrian base positions: "
                             '[[x1,y1,z1], [x2,y2,z2], ...]. '
                             "n_ped is clamped to the number of entries.")
    parser.add_argument("--spawn-jitter", type=float, default=1.0,
                        help="Uniform ± jitter in metres applied to each base "
                             "spawn position per scenario (default: 1.0)")
    parser.add_argument("--output-dir", type=str, default="runs/carla_scenarios",
                        help="Output directory for results")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-step logging (only show summary)")
    args = parser.parse_args()

    if args.quiet:
        logging.getLogger(__name__).setLevel(logging.WARNING)
        logging.getLogger("carla_integration.online_trackers").setLevel(logging.WARNING)

    # ── Connect to CARLA ─────────────────────────────────────────────────
    logger.info("Connecting to CARLA at %s:%d ...", args.host, args.port)
    client = carla.Client(args.host, args.port)
    client.set_timeout(60.0)

    # Always try to disable synchronous mode first, in case a previous crash
    # left the server stuck waiting for tick() calls.
    try:
        _w = client.get_world()
        _s = _w.get_settings()
        if _s.synchronous_mode:
            logger.warning("CARLA was stuck in synchronous mode — disabling ...")
            _s.synchronous_mode = False
            _s.fixed_delta_seconds = None
            _w.apply_settings(_s)
            time.sleep(1.0)
    except Exception:
        pass

    server_version = client.get_server_version()
    logger.info("Connected. Server version: %s", server_version)

    # ── Build disturbance profile ────────────────────────────────────────
    profile = NOMINAL_DISTURBANCE if args.disturbance == "nominal" else FUZZED_DISTURBANCE
    # Apply any per-field overrides
    overrides = {}
    if args.theta_spawn_min is not None:
        overrides["theta_spawn_min_deg"] = args.theta_spawn_min
    if args.theta_spawn_max is not None:
        overrides["theta_spawn_max_deg"] = args.theta_spawn_max
    if args.theta_approach_min is not None:
        overrides["theta_approach_min_deg"] = args.theta_approach_min
    if args.theta_approach_max is not None:
        overrides["theta_approach_max_deg"] = args.theta_approach_max
    if args.v_ped_min is not None:
        overrides["v_min"] = args.v_ped_min
    if args.v_ped_max is not None:
        overrides["v_max"] = args.v_ped_max
    if overrides:
        profile = DisturbanceProfile(**{**asdict(profile), **overrides})

    logger.info("Disturbance profile (%s): d=[%.0f,%.0f] m  "
                "θ_spawn=[%.0f°,%.0f°]  θ_approach=[%.0f°,%.0f°]  "
                "v=[%.1f,%.1f] m/s  (WalkerControl + social force)",
                args.disturbance,
                profile.d_spawn_min, profile.d_spawn_max,
                profile.theta_spawn_min_deg, profile.theta_spawn_max_deg,
                profile.theta_approach_min_deg, profile.theta_approach_max_deg,
                profile.v_min, profile.v_max)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.load_specs is not None:
        # ── Deterministic replay: load specs from JSON ────────────────
        with open(args.load_specs) as f:
            raw = json.load(f)
        specs = []
        for s in raw:
            peds = [PedestrianConfig(**p) for p in s["pedestrians"]]
            specs.append(ScenarioSpec(
                scenario_id=s["scenario_id"],
                n_ped=s["n_ped"],
                tracker=s["tracker"],
                v_ego_kmh=s["v_ego_kmh"],
                tau_safe=s["tau_safe"],
                seed=s["seed"],
                pedestrians=peds,
            ))
        logger.info("Loaded %d scenario specs from %s", len(specs), args.load_specs)
    else:
        # ── Generate new specs ────────────────────────────────────────
        spawn_locs = None
        if args.spawn_locations is not None:
            with open(args.spawn_locations) as f:
                spawn_locs = [tuple(p) for p in json.load(f)]
            logger.info("Loaded %d spawn locations from %s (jitter=±%.1f m)",
                         len(spawn_locs), args.spawn_locations, args.spawn_jitter)

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

        # Save specs for future deterministic replay
        specs_path = output_dir / "scenario_specs.json"
        with open(specs_path, "w") as f:
            json.dump([asdict(s) for s in specs], f, indent=2)
        logger.info("Saved %d scenario specs to %s", len(specs), specs_path)

    # ── Run scenarios ────────────────────────────────────────────────────
    all_results = []
    n_collisions = 0
    t0 = time.time()

    def _ensure_async_mode():
        """Safety net: always restore CARLA to async mode on exit."""
        try:
            w = client.get_world()
            s = w.get_settings()
            if s.synchronous_mode:
                s.synchronous_mode = False
                s.fixed_delta_seconds = None
                w.apply_settings(s)
                logger.info("Restored CARLA to asynchronous mode.")
        except Exception:
            pass

    import atexit
    atexit.register(_ensure_async_mode)

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
            extra = " (despite braking)" if result.collision_despite_brake else ""
            logger.info("  COLLISION at t=%.1fs%s (min_ttc=%.2f, rho=%.2f)",
                         result.collision_time, extra, result.min_ttc, result.rho_min)
        else:
            logger.info("  Safe (min_ttc=%.2f, rho=%.2f)", result.min_ttc, result.rho_min)
        logger.info("  Tracker: %s | ADE=%.3f | FDE=%.3f | FP_brakes=%d | "
                     "brake_steps=%d | activations=%d",
                     spec.tracker, result.ade, result.fde, result.n_fp_brakes,
                     result.n_brake_steps, result.simplex_activations)

        all_results.append(result)

        # Save per-scenario traces
        sc_dir = output_dir / f"scenario_{spec.scenario_id:03d}"
        sc_dir.mkdir(parents=True, exist_ok=True)

        gt_path = sc_dir / "gt_traces.json"
        with open(gt_path, "w") as f:
            json.dump({
                "ego_trace": result.ego_trace,
                "ped_traces": {str(k): v for k, v in result.ped_traces.items()},
                "collision": result.collision,
                "collision_time": result.collision_time,
                "collision_ped_id": result.collision_ped_id,
            }, f, indent=2)

        noisy_path = sc_dir / "noisy_detections.json"
        with open(noisy_path, "w") as f:
            json.dump(result.noisy_detections, f, indent=2)

        tracker_path = sc_dir / "tracker_data.json"
        with open(tracker_path, "w") as f:
            json.dump({
                "tracker": spec.tracker,
                "ade": round(result.ade, 4),
                "fde": round(result.fde, 4),
                "n_fp_brakes": result.n_fp_brakes,
                "brake_trace": result.brake_trace,
                "brake_events": result.tracker_brake_events,
                "n_brake_steps": result.n_brake_steps,
                "brake_first_time": result.brake_first_time,
                "collision_despite_brake": result.collision_despite_brake,
                "brake_activations": result.simplex_activations,
                "state_history": result._tracker_history,
            }, f, indent=2)

        logger.info("  Saved GT → %s", gt_path)
        logger.info("  Saved tracker data → %s", tracker_path)

        # Save incrementally
        results_path = output_dir / "scenario_results.json"
        _save_results(all_results, results_path)

    elapsed = time.time() - t0

    # ── Summary ──────────────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("COMPLETE: %d scenarios in %.0fs (%.1f s/scenario)",
                len(specs), elapsed, elapsed / max(len(specs), 1))
    logger.info("Tracker: %s  |  Disturbance: %s", args.tracker, args.disturbance)
    coll_rate = 100 * n_collisions / max(len(specs), 1)
    mean_ade = float(np.mean([r.ade for r in all_results])) if all_results else 0.0
    mean_fde = float(np.mean([r.fde for r in all_results])) if all_results else 0.0
    mean_fp = float(np.mean([r.n_fp_brakes for r in all_results])) if all_results else 0.0
    logger.info("Collisions: %d / %d (%.1f%%)", n_collisions, len(specs), coll_rate)
    logger.info("Mean ADE: %.3f  |  Mean FDE: %.3f  |  Mean FP brakes: %.1f",
                mean_ade, mean_fde, mean_fp)
    logger.info("Min rho_min: %.3f", min(r.rho_min for r in all_results))

    # Save aggregate summary
    agg_path = output_dir / "aggregate_summary.json"
    agg = {
        "tracker": args.tracker,
        "disturbance": args.disturbance,
        "n_scenarios": len(specs),
        "n_ped": args.n_ped,
        "collision_rate_pct": round(coll_rate, 1),
        "n_collisions": n_collisions,
        "mean_ade": round(mean_ade, 4),
        "mean_fde": round(mean_fde, 4),
        "mean_fp_brakes": round(mean_fp, 2),
        "mean_brake_steps": round(float(np.mean([r.n_brake_steps for r in all_results])), 1),
        "min_rho_min": round(min(r.rho_min for r in all_results), 3),
        "elapsed_s": round(elapsed, 1),
    }
    with open(agg_path, "w") as f:
        json.dump(agg, f, indent=2)

    logger.info("Results saved to %s", output_dir / "scenario_results.json")
    logger.info("Aggregate summary saved to %s", agg_path)

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
            "collision_ped_id": r.collision_ped_id,
            "min_ttc": r.min_ttc if r.min_ttc != float("inf") else None,
            "rho_min": r.rho_min if r.rho_min != float("inf") else None,
            "ade": round(r.ade, 4),
            "fde": round(r.fde, 4),
            "n_fp_brakes": r.n_fp_brakes,
            "brake_activations": r.simplex_activations,
            "n_brake_steps": r.n_brake_steps,
            "brake_first_time": r.brake_first_time,
            "collision_despite_brake": r.collision_despite_brake,
            "ttc_trace": r.ttc_trace,
            "duration_s": round(r.duration_s, 1),
            "n_ped_spawned": len(r.ped_traces),
            "ego_trace_length": len(r.ego_trace),
            "ttc_trace_length": len(r.ttc_trace),
            "noisy_detections_count": len(r.noisy_detections),
            "spec": r.spec,
        }
        data.append(d)

    with open(path, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    main()
