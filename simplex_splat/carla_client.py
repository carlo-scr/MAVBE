"""
Synchronous CARLA client with ground-truth sensor suite.

Provides RGB, depth (metres), and semantic segmentation at every
simulation tick using CARLA's synchronous mode.  Also exposes the
vehicle's ground-truth transform for pose-oracle experiments.
"""

from __future__ import annotations

import logging
import math
import queue
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import carla
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CARLA semantic-segmentation tag → human-readable label
# (subset relevant to safety monitoring)
# ---------------------------------------------------------------------------
CARLA_SEM_TAGS = {
    0: "Unlabeled",
    1: "Building",
    2: "Fence",
    3: "Other",
    4: "Pedestrian",
    5: "Pole",
    6: "RoadLine",
    7: "Road",
    8: "SideWalk",
    9: "Vegetation",
    10: "Vehicle",
    11: "Wall",
    12: "TrafficSign",
    13: "Sky",
    14: "Ground",
    15: "Bridge",
    16: "RailTrack",
    17: "GuardRail",
    18: "TrafficLight",
    19: "Static",
    20: "Dynamic",
    21: "Water",
    22: "Terrain",
}


@dataclass
class SensorFrame:
    """Container for one synchronous sensor snapshot."""

    frame_id: int
    timestamp: float  # simulation seconds

    # Ground-truth pose (4×4 homogeneous, world frame)
    pose: np.ndarray  # (4, 4) float64

    # Sensor data
    rgb: np.ndarray  # (H, W, 3) uint8  BGR
    depth: np.ndarray  # (H, W) float32  metres
    semantic: np.ndarray  # (H, W) uint8  class ids

    # Camera intrinsics
    intrinsics: np.ndarray  # (3, 3) float64

    # Vehicle state
    velocity_mps: float = 0.0


class SyncCARLAClient:
    """Deterministic, synchronous CARLA interface.

    All sensors share the same tick so frames are pixel-aligned.
    """

    def __init__(self, cfg: dict):
        carla_cfg = cfg["carla"]
        self._host = carla_cfg["host"]
        self._port = carla_cfg["port"]
        self._timeout = carla_cfg.get("timeout", 20.0)
        self._delta = carla_cfg["delta_seconds"]
        self._map_name = carla_cfg.get("map", "Town03")

        self._sensor_cfg = cfg["sensors"]
        self._vehicle_cfg = cfg["vehicle"]

        self._client: Optional[carla.Client] = None
        self._world: Optional[carla.World] = None
        self._vehicle: Optional[carla.Actor] = None
        self._sensors: Dict[str, carla.Sensor] = {}
        self._queues: Dict[str, queue.Queue] = {}
        self._spawned_actors: List[carla.Actor] = []
        self._intrinsics: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Connect to CARLA and configure synchronous mode."""
        self._client = carla.Client(self._host, self._port)
        self._client.set_timeout(self._timeout)

        # Load map if different
        current_map = self._client.get_world().get_map().name
        if not current_map.endswith(self._map_name):
            logger.info("Loading map %s ...", self._map_name)
            self._client.load_world(self._map_name)
            time.sleep(2.0)

        self._world = self._client.get_world()
        settings = self._world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = self._delta
        self._world.apply_settings(settings)

        # Traffic manager
        tm = self._client.get_trafficmanager()
        tm.set_synchronous_mode(True)

        logger.info(
            "Connected to CARLA %s  map=%s  dt=%.3f",
            self._client.get_server_version(),
            self._map_name,
            self._delta,
        )

    def spawn_vehicle(self) -> carla.Actor:
        """Spawn the ego vehicle and attach sensors."""
        bp_lib = self._world.get_blueprint_library()
        vehicle_bp = bp_lib.find(self._vehicle_cfg["blueprint"])

        spawn_points = self._world.get_map().get_spawn_points()
        idx = self._vehicle_cfg.get("spawn_index")
        if idx is not None and 0 <= idx < len(spawn_points):
            transform = spawn_points[idx]
        else:
            import random
            transform = random.choice(spawn_points)

        self._vehicle = self._world.spawn_actor(vehicle_bp, transform)
        self._spawned_actors.append(self._vehicle)
        logger.info("Ego vehicle spawned at %s", transform.location)

        # Attach sensors
        self._attach_sensor("rgb", "sensor.camera.rgb")
        self._attach_sensor("depth", "sensor.camera.depth")
        self._attach_sensor("semantic", "sensor.camera.semantic_segmentation")

        # Compute intrinsics from RGB config (same for all co-located cameras)
        self._intrinsics = self._compute_intrinsics(self._sensor_cfg["rgb"])

        # Let physics settle
        for _ in range(10):
            self._world.tick()

        return self._vehicle

    def tick(self) -> SensorFrame:
        """Advance simulation by one step and return aligned sensor data."""
        self._world.tick()

        # Collect sensor data with timeout
        data: Dict[str, carla.SensorData] = {}
        for name, q in self._queues.items():
            try:
                data[name] = q.get(timeout=5.0)
            except queue.Empty:
                raise RuntimeError(f"Sensor '{name}' did not deliver data")

        frame_id = data["rgb"].frame
        timestamp = data["rgb"].timestamp

        rgb = self._parse_rgb(data["rgb"])
        depth = self._parse_depth(data["depth"])
        semantic = self._parse_semantic(data["semantic"])
        pose = self._get_vehicle_pose()
        velocity = self._get_vehicle_speed()

        return SensorFrame(
            frame_id=frame_id,
            timestamp=timestamp,
            pose=pose,
            rgb=rgb,
            depth=depth,
            semantic=semantic,
            intrinsics=self._intrinsics,
            velocity_mps=velocity,
        )

    def apply_control(self, control: carla.VehicleControl) -> None:
        """Send a control command to the ego vehicle."""
        self._vehicle.apply_control(control)

    def destroy(self) -> None:
        """Clean up all spawned actors and restore async mode."""
        for actor in reversed(self._spawned_actors):
            if actor is not None and actor.is_alive:
                actor.destroy()
        self._spawned_actors.clear()
        self._sensors.clear()
        self._queues.clear()

        if self._world is not None:
            settings = self._world.get_settings()
            settings.synchronous_mode = False
            settings.fixed_delta_seconds = None
            self._world.apply_settings(settings)
        logger.info("CARLA client cleaned up.")

    # ------------------------------------------------------------------
    # Actor spawning helpers (for scenarios)
    # ------------------------------------------------------------------

    def spawn_pedestrian(
        self, location: carla.Location, destination: Optional[carla.Location] = None,
        speed: float = 1.4,
    ) -> Tuple[carla.Actor, Optional[carla.Actor]]:
        """Spawn a pedestrian and optionally a walker AI controller."""
        bp_lib = self._world.get_blueprint_library()
        walker_bps = bp_lib.filter("walker.pedestrian.*")
        import random
        walker_bp = random.choice(walker_bps)
        if walker_bp.has_attribute("is_invincible"):
            walker_bp.set_attribute("is_invincible", "false")

        transform = carla.Transform(location)
        walker = self._world.spawn_actor(walker_bp, transform)
        self._spawned_actors.append(walker)

        controller = None
        if destination is not None:
            ctrl_bp = bp_lib.find("controller.ai.walker")
            controller = self._world.spawn_actor(ctrl_bp, carla.Transform(), walker)
            self._spawned_actors.append(controller)
            self._world.tick()  # ensure controller is attached
            controller.start()
            controller.go_to_location(destination)
            controller.set_max_speed(speed)

        logger.info("Pedestrian spawned at %s", location)
        return walker, controller

    def spawn_static_obstacle(
        self, blueprint_name: str, transform: carla.Transform
    ) -> carla.Actor:
        """Spawn a static prop (barrier, cone, etc.)."""
        bp_lib = self._world.get_blueprint_library()
        bp = bp_lib.find(blueprint_name)
        actor = self._world.spawn_actor(bp, transform)
        self._spawned_actors.append(actor)
        logger.info("Static obstacle '%s' spawned at %s", blueprint_name, transform.location)
        return actor

    def get_vehicle(self) -> carla.Actor:
        return self._vehicle

    def get_world(self) -> carla.World:
        return self._world

    @property
    def intrinsics(self) -> np.ndarray:
        return self._intrinsics

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _attach_sensor(self, name: str, blueprint_id: str) -> None:
        bp_lib = self._world.get_blueprint_library()
        bp = bp_lib.find(blueprint_id)

        scfg = self._sensor_cfg[name]
        bp.set_attribute("image_size_x", str(scfg["width"]))
        bp.set_attribute("image_size_y", str(scfg["height"]))
        bp.set_attribute("fov", str(scfg["fov"]))

        pos = scfg["position"]
        rot = scfg["rotation"]
        transform = carla.Transform(
            carla.Location(x=pos[0], y=pos[1], z=pos[2]),
            carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2]),
        )

        sensor = self._world.spawn_actor(bp, transform, attach_to=self._vehicle)
        self._spawned_actors.append(sensor)

        q: queue.Queue = queue.Queue(maxsize=1)
        sensor.listen(q.put)
        self._sensors[name] = sensor
        self._queues[name] = q

    @staticmethod
    def _parse_rgb(image: carla.Image) -> np.ndarray:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))[:, :, :3]  # drop alpha
        return array.copy()  # BGR

    @staticmethod
    def _parse_depth(image: carla.Image) -> np.ndarray:
        """Convert CARLA encoded depth to metres (float32)."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4)).astype(np.float32)
        # CARLA depth encoding: depth = (R + G*256 + B*65536) / (256^3 - 1) * 1000
        depth = (
            array[:, :, 2] + array[:, :, 1] * 256.0 + array[:, :, 0] * 65536.0
        ) / (256.0**3 - 1.0) * 1000.0
        return depth

    @staticmethod
    def _parse_semantic(image: carla.Image) -> np.ndarray:
        """Extract class-id channel from CARLA semantic image."""
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        return array[:, :, 2].copy()  # red channel = class tag

    def _get_vehicle_pose(self) -> np.ndarray:
        """Return 4×4 world-frame pose of the ego vehicle."""
        t = self._vehicle.get_transform()
        loc = t.location
        rot = t.rotation  # degrees

        # Convert to radians
        pitch = math.radians(rot.pitch)
        yaw = math.radians(rot.yaw)
        roll = math.radians(rot.roll)

        # Rotation matrix (CARLA: left-handed UE4 → right-handed)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cr, sr = math.cos(roll), math.sin(roll)

        R = np.array(
            [
                [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                [-sp, cp * sr, cp * cr],
            ]
        )

        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = [loc.x, loc.y, loc.z]
        return pose

    def _get_vehicle_speed(self) -> float:
        v = self._vehicle.get_velocity()
        return math.sqrt(v.x**2 + v.y**2 + v.z**2)

    @staticmethod
    def _compute_intrinsics(cam_cfg: dict) -> np.ndarray:
        """Pinhole intrinsics from CARLA camera parameters."""
        W = cam_cfg["width"]
        H = cam_cfg["height"]
        fov = cam_cfg["fov"]
        fx = W / (2.0 * math.tan(math.radians(fov / 2.0)))
        fy = fx  # square pixels
        cx = W / 2.0
        cy = H / 2.0
        return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)
