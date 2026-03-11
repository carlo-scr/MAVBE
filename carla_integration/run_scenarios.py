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

Usage:
    python -m carla_integration.run_scenarios --n-scenarios 10
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
class PedestrianConfig:
    """Per-pedestrian disturbance vector τ_i."""
    ped_id: int
    d_spawn: float          # metres from ego path [5, 35]
    theta_approach: float   # radians relative to ego heading [0, π]
    v_init: float           # m/s [0.5, 2.0]


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
    v_ego_kmh: float = 20.0,
    tau_safe: float = 2.0,
    base_seed: int = 42,
) -> List[ScenarioSpec]:
    """Generate N scenario specifications from the paper's disturbance model."""
    rng = np.random.default_rng(base_seed)
    specs = []

    for i in range(n_scenarios):
        seed = int(rng.integers(0, 2**31))
        ped_rng = np.random.default_rng(seed)

        peds = []
        for j in range(n_ped):
            peds.append(PedestrianConfig(
                ped_id=j,
                d_spawn=float(ped_rng.uniform(5, 35)),
                theta_approach=float(ped_rng.uniform(0, math.pi)),
                v_init=float(ped_rng.uniform(0.5, 2.0)),
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

    bp_lib = world.get_blueprint_library()
    spawn_points = world.get_map().get_spawn_points()
    rng = np.random.default_rng(spec.seed)

    actors_to_destroy = []
    result = ScenarioResult(
        scenario_id=spec.scenario_id,
        spec=asdict(spec),
    )

    # Frame output directory
    scenario_dir = output_dir / f"scenario_{spec.scenario_id:03d}"
    frames_dir = scenario_dir / "frames"
    if save_video:
        frames_dir.mkdir(parents=True, exist_ok=True)
    result.frames_dir = str(frames_dir)

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

        frame_counter = [0]

        def on_image(image):
            if save_video:
                # Save as PNG for later perception processing
                arr = np.frombuffer(image.raw_data, dtype=np.uint8)
                arr = arr.reshape((image.height, image.width, 4))[:, :, :3]
                frame_path = frames_dir / f"frame_{frame_counter[0]:06d}.png"
                try:
                    import cv2
                    cv2.imwrite(str(frame_path), arr)
                except ImportError:
                    # Fallback: save as raw numpy
                    np.save(str(frame_path.with_suffix(".npy")), arr)
                frame_counter[0] += 1

        camera.listen(on_image)

        # ── Spawn pedestrians ────────────────────────────────────────────
        walker_bps = bp_lib.filter("walker.pedestrian.*")
        walkers = []
        controllers = []
        ped_id_map = {}  # CARLA actor ID -> our ped_id

        world.tick()  # tick so ego transform is settled

        for ped_cfg in spec.pedestrians:
            ego_t = ego.get_transform()
            fwd = ego_t.get_forward_vector()
            # Right vector (perpendicular to forward in XY plane)
            right_x, right_y = -fwd.y, fwd.x

            offset_fwd = ped_cfg.d_spawn * math.cos(ped_cfg.theta_approach)
            offset_lat = ped_cfg.d_spawn * math.sin(ped_cfg.theta_approach)

            spawn_loc = carla.Location(
                x=ego_t.location.x + fwd.x * offset_fwd + right_x * offset_lat,
                y=ego_t.location.y + fwd.y * offset_fwd + right_y * offset_lat,
                z=ego_t.location.z + 1.0,
            )
            spawn_transform = carla.Transform(
                spawn_loc, carla.Rotation(yaw=float(rng.uniform(0, 360)))
            )

            walker_bp = walker_bps[int(rng.integers(0, len(walker_bps)))]
            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")

            walker = world.try_spawn_actor(walker_bp, spawn_transform)
            if walker is None:
                # Try slightly different location
                spawn_loc.z += 0.5
                walker = world.try_spawn_actor(walker_bp, carla.Transform(
                    spawn_loc, spawn_transform.rotation
                ))
            if walker is None:
                logger.warning("  Failed to spawn pedestrian %d", ped_cfg.ped_id)
                continue

            walkers.append(walker)
            actors_to_destroy.append(walker)
            ped_id_map[walker.id] = ped_cfg.ped_id

            # AI controller
            ctrl_bp = bp_lib.find("controller.ai.walker")
            ctrl = world.spawn_actor(ctrl_bp, carla.Transform(), walker)
            controllers.append((ctrl, ped_cfg))
            actors_to_destroy.append(ctrl)

        logger.info("  Spawned %d / %d pedestrians", len(walkers), spec.n_ped)

        # Tick to register controllers, then start them
        world.tick()
        for ctrl, ped_cfg in controllers:
            ctrl.start()
            ctrl.set_max_speed(ped_cfg.v_init)
            # Walk towards the ego's path
            ctrl.go_to_location(ego.get_location())

        # ── Ego control: autopilot at target speed ───────────────────────
        ego.set_autopilot(True)
        # Note: CARLA autopilot doesn't guarantee exact speed; it follows
        # traffic rules. For more precise control, use a PID controller
        # on vehicle.apply_control(). For now, autopilot is sufficient.

        # ── Main simulation loop ─────────────────────────────────────────
        n_steps = int(t_max / dt)
        emergency_active = False

        t0_wall = time.time()
        for step in range(n_steps):
            world.tick()
            sim_time = (step + 1) * dt

            # Ego state
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
                ctrl = carla.VehicleControl()
                ctrl.brake = 1.0
                ctrl.throttle = 0.0
                ego.apply_control(ctrl)
            elif emergency_active and step_min_ttc > spec.tau_safe * 1.5:
                emergency_active = False
                ego.set_autopilot(True)

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
        # Stop controllers first
        for ctrl, _ in controllers:
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
    parser.add_argument("--v-ego", type=float, default=20.0,
                        help="Ego speed in km/h (default: 20)")
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
    parser.add_argument("--output-dir", type=str, default="runs/carla_scenarios",
                        help="Output directory for results")
    args = parser.parse_args()

    # ── Connect to CARLA ─────────────────────────────────────────────────
    logger.info("Connecting to CARLA at %s:%d ...", args.host, args.port)
    client = carla.Client(args.host, args.port)
    client.set_timeout(20.0)
    server_version = client.get_server_version()
    logger.info("Connected. Server version: %s", server_version)

    # ── Generate scenarios ───────────────────────────────────────────────
    specs = generate_scenario_specs(
        n_scenarios=args.n_scenarios,
        n_ped=args.n_ped,
        tracker=args.tracker,
        v_ego_kmh=args.v_ego,
        tau_safe=args.tau_safe,
        base_seed=args.seed,
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
        logger.info("Frames saved to %s/scenario_*/frames/", output_dir)
        logger.info(
            "\nNext step: run perception on saved frames:\n"
            "  python perception/detect_dual_tracking.py \\\n"
            "    --source runs/carla_scenarios/scenario_000/frames/ \\\n"
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
