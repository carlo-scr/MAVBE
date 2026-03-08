"""
Deterministic Emergency Braking Controller.

The safety controller in the Simplex Architecture: when the monitor
detects a violation, this controller applies maximum braking until the
vehicle is stopped, then holds the brake for a configurable duration.

This is intentionally simple and deterministic — no neural networks,
no learned components.  Correctness > performance.
"""

from __future__ import annotations

import logging
import time
from enum import Enum, auto

logger = logging.getLogger(__name__)


class BrakeState(Enum):
    IDLE = auto()
    BRAKING = auto()
    HOLDING = auto()
    RELEASED = auto()


class EmergencyController:
    """Deterministic emergency-stop controller.

    Parameters
    ----------
    cfg : dict
        The ``emergency`` section of the master config.
    delta_t : float
        Simulation timestep in seconds.
    """

    def __init__(self, cfg: dict, delta_t: float = 0.05):
        self.decel = cfg.get("deceleration_mps2", 8.0)
        self.max_steer = cfg.get("max_steer_lock", 0.0)
        self.hold_s = cfg.get("hold_duration_s", 3.0)
        self.dt = delta_t

        self.state = BrakeState.IDLE
        self._trigger_time: float = 0.0
        self._stop_time: float = 0.0
        self._trigger_speed: float = 0.0

    def trigger(self, current_speed_mps: float, sim_time: float) -> None:
        """Initiate emergency braking."""
        if self.state != BrakeState.IDLE:
            return  # already braking
        self.state = BrakeState.BRAKING
        self._trigger_time = sim_time
        self._trigger_speed = current_speed_mps
        logger.warning(
            "EMERGENCY BRAKE triggered at t=%.2fs  speed=%.1f m/s",
            sim_time,
            current_speed_mps,
        )

    def get_control(self, current_speed_mps: float, sim_time: float):
        """Return a CARLA-style VehicleControl dict.

        We return a plain dict so this module doesn't import carla directly;
        the orchestrator converts it.

        Returns
        -------
        dict with keys: throttle, brake, steer, hand_brake, reverse
        """
        if self.state == BrakeState.IDLE:
            return None  # no override

        if self.state == BrakeState.BRAKING:
            if current_speed_mps < 0.05:
                # Vehicle has stopped
                self.state = BrakeState.HOLDING
                self._stop_time = sim_time
                logger.info(
                    "Vehicle stopped. Holding brake for %.1fs", self.hold_s
                )
            return {
                "throttle": 0.0,
                "brake": 1.0,
                "steer": self.max_steer,
                "hand_brake": True,
                "reverse": False,
            }

        if self.state == BrakeState.HOLDING:
            elapsed = sim_time - self._stop_time
            if elapsed >= self.hold_s:
                self.state = BrakeState.RELEASED
                logger.info("Brake hold complete. Controller released.")
                return None
            return {
                "throttle": 0.0,
                "brake": 1.0,
                "steer": 0.0,
                "hand_brake": True,
                "reverse": False,
            }

        # RELEASED
        return None

    @property
    def is_active(self) -> bool:
        return self.state in (BrakeState.BRAKING, BrakeState.HOLDING)

    @property
    def stopping_distance_m(self) -> float:
        """Theoretical stopping distance from trigger speed."""
        # d = v^2 / (2a)
        if self.decel <= 0:
            return float("inf")
        return self._trigger_speed**2 / (2.0 * self.decel)

    @property
    def trigger_time(self) -> float:
        return self._trigger_time

    def reset(self) -> None:
        """Reset to idle (call between scenarios)."""
        self.state = BrakeState.IDLE
        self._trigger_time = 0.0
        self._stop_time = 0.0
        self._trigger_speed = 0.0
