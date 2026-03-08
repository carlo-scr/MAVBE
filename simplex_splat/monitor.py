"""
Simplex Architecture Safety Monitor.

Implements both *Semantically-Aware* and *Pure Geometric* monitors that
cross-validate a 3DGS-rendered view against ground-truth sensor streams.

Safety specification  φ_safe
-----------------------------
1. **Dynamic Safety** (false-negative bound):
   Every safety-critical dynamic object (pedestrian, vehicle) visible in
   the ground-truth semantic stream must have a corresponding non-empty
   region in the neural depth map: |D_pred − D_obs| < τ_fn.

2. **Static Integrity** (false-positive bound):
   The map's predicted depth in *drivable* regions must not under-
   estimate observed free space beyond τ_fp → prevents phantom braking.

3. **Structural Consistency**:
   Static safety features (stop signs, traffic lights) must satisfy
   IoU ≥ 0.5 between rendered and GT semantic masks.

When any sub-specification is violated (after debounce), the monitor
emits a ``SafetyViolation`` that the orchestrator hands to the
emergency controller.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ── Violation types ──────────────────────────────────────────────────────────

class ViolationType(Enum):
    DYNAMIC_FALSE_NEGATIVE = auto()   # unmodelled dynamic object
    STATIC_FALSE_POSITIVE = auto()    # phantom obstacle in drivable area
    STRUCTURAL_INCONSISTENCY = auto() # static feature (stop sign) mismatch


@dataclass
class SafetyViolation:
    """Describes one safety violation detected by the monitor."""

    violation_type: ViolationType
    severity: float             # 0..1 (normalised residual magnitude)
    affected_pixels: int        # number of pixels in violation region
    mean_residual_m: float      # mean |D_pred − D_obs| in region
    max_residual_m: float
    timestamp: float            # simulation time
    frame_id: int
    region_mask: Optional[np.ndarray] = None  # (H, W) bool
    details: str = ""


@dataclass
class MonitorState:
    """Per-frame diagnostic state exposed for logging / visualisation."""

    frame_id: int = 0
    timestamp: float = 0.0
    monitor_latency_ms: float = 0.0

    # Residual statistics
    global_mean_residual: float = 0.0
    global_max_residual: float = 0.0
    dynamic_mean_residual: float = 0.0
    static_mean_residual: float = 0.0

    # Per-class IoU for structural checks
    structural_iou: Dict[int, float] = field(default_factory=dict)

    # Violation info
    is_safe: bool = True
    violations: List[SafetyViolation] = field(default_factory=list)

    # EMA-smoothed safety score (0 = unsafe, 1 = fully safe)
    safety_score: float = 1.0


class SafetyMonitor:
    """Cross-validates a rendered 3DGS view against ground-truth sensors.

    Parameters
    ----------
    cfg : dict
        The ``monitor`` section of the master config.
    """

    def __init__(self, cfg: dict):
        self.mode = cfg.get("type", "semantic")  # "semantic" | "geometric"

        # Thresholds
        self.tau_fn = cfg.get("tau_fn", 1.5)
        self.tau_fp = cfg.get("tau_fp", 2.0)
        self.iou_thr = cfg.get("iou_threshold", 0.5)
        self.min_pix_frac = cfg.get("min_critical_pixel_fraction", 0.001)

        # Critical semantic classes
        self.critical_dynamic: Set[int] = set(cfg.get("critical_dynamic_classes", [4, 10]))
        self.critical_static: Set[int] = set(cfg.get("critical_static_classes", [12, 18]))

        # EMA smoothing
        self.ema_alpha = cfg.get("ema_alpha", 0.3)
        self._ema_score = 1.0

        # Debounce
        self.patience = cfg.get("violation_patience", 2)
        self._consecutive_violations = 0

        # Timing budget
        self.max_response_ms = cfg.get("max_response_time_ms", 100.0)

    # ------------------------------------------------------------------
    # Main entry
    # ------------------------------------------------------------------

    def check(
        self,
        rendered_depth: np.ndarray,
        gt_depth: np.ndarray,
        gt_semantic: np.ndarray,
        rendered_semantic: Optional[np.ndarray] = None,
        frame_id: int = 0,
        timestamp: float = 0.0,
    ) -> MonitorState:
        """Run all safety checks for one frame.

        Parameters
        ----------
        rendered_depth : (H, W) float32 — depth from 3DGS map.
        gt_depth       : (H, W) float32 — ground-truth depth.
        gt_semantic    : (H, W) uint8   — ground-truth semantic labels.
        rendered_semantic : (H, W) uint8 — if available from map.
        frame_id, timestamp : bookkeeping.

        Returns
        -------
        MonitorState with all diagnostics + violation list.
        """
        t0 = time.perf_counter()

        state = MonitorState(frame_id=frame_id, timestamp=timestamp)
        violations: List[SafetyViolation] = []

        # ---------- 0. Global depth residual ----------------------------------
        residual = np.abs(rendered_depth.astype(np.float64) - gt_depth.astype(np.float64))
        valid = (gt_depth > 0.1) & (rendered_depth > 0.1)
        if valid.any():
            state.global_mean_residual = float(np.mean(residual[valid]))
            state.global_max_residual = float(np.max(residual[valid]))

        total_pixels = gt_depth.shape[0] * gt_depth.shape[1]

        if self.mode == "semantic":
            # ---------- 1. Dynamic Safety (false-negative check) --------------
            dyn_violations = self._check_dynamic_safety(
                residual, valid, gt_semantic, frame_id, timestamp, total_pixels
            )
            violations.extend(dyn_violations)

            # ---------- 2. Static Integrity (false-positive check) ------------
            stat_violations = self._check_static_integrity(
                rendered_depth, gt_depth, valid, gt_semantic, frame_id, timestamp,
                total_pixels,
            )
            violations.extend(stat_violations)

            # ---------- 3. Structural Consistency (IoU for static feats) ------
            struct_violations, iou_map = self._check_structural_consistency(
                gt_semantic, rendered_semantic, frame_id, timestamp, total_pixels
            )
            violations.extend(struct_violations)
            state.structural_iou = iou_map

        else:
            # Pure-geometric baseline: global threshold only
            geo_violations = self._check_geometric_only(
                residual, valid, frame_id, timestamp, total_pixels
            )
            violations.extend(geo_violations)

        # ---------- 4. Aggregate decision ---------------------------------
        state.violations = violations
        latency_ms = (time.perf_counter() - t0) * 1000.0
        state.monitor_latency_ms = latency_ms

        if latency_ms > self.max_response_ms:
            logger.warning(
                "Monitor latency %.1f ms exceeds budget %.1f ms",
                latency_ms,
                self.max_response_ms,
            )

        # Update EMA safety score
        if violations:
            max_sev = max(v.severity for v in violations)
            raw_score = 1.0 - max_sev
        else:
            raw_score = 1.0
        self._ema_score = self.ema_alpha * raw_score + (1 - self.ema_alpha) * self._ema_score
        state.safety_score = self._ema_score

        # Debounce: require consecutive violations
        if violations:
            self._consecutive_violations += 1
        else:
            self._consecutive_violations = 0

        state.is_safe = self._consecutive_violations < self.patience
        return state

    # ------------------------------------------------------------------
    # Sub-checks
    # ------------------------------------------------------------------

    def _check_dynamic_safety(
        self,
        residual: np.ndarray,
        valid: np.ndarray,
        gt_semantic: np.ndarray,
        frame_id: int,
        timestamp: float,
        total_pixels: int,
    ) -> List[SafetyViolation]:
        """Ensure every dynamic actor in GT has non-empty map coverage."""
        violations = []
        for cls in self.critical_dynamic:
            mask = (gt_semantic == cls) & valid
            n_pixels = int(np.sum(mask))
            if n_pixels < self.min_pix_frac * total_pixels:
                continue  # not enough pixels to matter

            region_res = residual[mask]
            mean_res = float(np.mean(region_res))
            max_res = float(np.max(region_res))

            # A large residual in the dynamic region means the map has
            # NO geometry where the real object is → false negative
            if mean_res > self.tau_fn:
                severity = min(1.0, mean_res / (self.tau_fn * 3.0))
                violations.append(
                    SafetyViolation(
                        violation_type=ViolationType.DYNAMIC_FALSE_NEGATIVE,
                        severity=severity,
                        affected_pixels=n_pixels,
                        mean_residual_m=mean_res,
                        max_residual_m=max_res,
                        timestamp=timestamp,
                        frame_id=frame_id,
                        region_mask=mask,
                        details=f"Dynamic class {cls}: "
                        f"mean_residual={mean_res:.2f}m > τ_fn={self.tau_fn:.1f}m "
                        f"({n_pixels} px)",
                    )
                )
        return violations

    def _check_static_integrity(
        self,
        rendered_depth: np.ndarray,
        gt_depth: np.ndarray,
        valid: np.ndarray,
        gt_semantic: np.ndarray,
        frame_id: int,
        timestamp: float,
        total_pixels: int,
    ) -> List[SafetyViolation]:
        """Detect phantom obstacles in drivable road surface."""
        violations = []
        # CARLA class 7 = Road
        road_mask = (gt_semantic == 7) & valid
        n_pixels = int(np.sum(road_mask))
        if n_pixels < self.min_pix_frac * total_pixels:
            return violations

        # Phantom obstacle: rendered depth << GT depth (map thinks obstacle
        # is closer than reality)
        underestimate = gt_depth[road_mask] - rendered_depth[road_mask]
        phantom_mask_local = underestimate > self.tau_fp
        n_phantom = int(np.sum(phantom_mask_local))

        if n_phantom > self.min_pix_frac * total_pixels:
            mean_under = float(np.mean(underestimate[phantom_mask_local]))
            max_under = float(np.max(underestimate[phantom_mask_local]))
            severity = min(1.0, mean_under / (self.tau_fp * 3.0))

            # Reconstruct full-image mask
            full_phantom = np.zeros_like(gt_depth, dtype=bool)
            road_indices = np.where(road_mask)
            phantom_indices = (
                road_indices[0][phantom_mask_local],
                road_indices[1][phantom_mask_local],
            )
            full_phantom[phantom_indices] = True

            violations.append(
                SafetyViolation(
                    violation_type=ViolationType.STATIC_FALSE_POSITIVE,
                    severity=severity,
                    affected_pixels=n_phantom,
                    mean_residual_m=mean_under,
                    max_residual_m=max_under,
                    timestamp=timestamp,
                    frame_id=frame_id,
                    region_mask=full_phantom,
                    details=f"Phantom obstacle on road: {n_phantom} px, "
                    f"mean_gap={mean_under:.2f}m > τ_fp={self.tau_fp:.1f}m",
                )
            )
        return violations

    def _check_structural_consistency(
        self,
        gt_semantic: np.ndarray,
        rendered_semantic: Optional[np.ndarray],
        frame_id: int,
        timestamp: float,
        total_pixels: int,
    ) -> Tuple[List[SafetyViolation], Dict[int, float]]:
        """Check IoU of static safety-critical features."""
        violations = []
        iou_map: Dict[int, float] = {}

        if rendered_semantic is None:
            return violations, iou_map

        for cls in self.critical_static:
            gt_mask = gt_semantic == cls
            pred_mask = rendered_semantic == cls

            intersection = int(np.sum(gt_mask & pred_mask))
            union = int(np.sum(gt_mask | pred_mask))
            if union == 0:
                continue  # class absent in both

            n_gt = int(np.sum(gt_mask))
            if n_gt < self.min_pix_frac * total_pixels:
                continue

            iou = intersection / union
            iou_map[cls] = iou

            if iou < self.iou_thr:
                severity = min(1.0, (self.iou_thr - iou) / self.iou_thr)
                violations.append(
                    SafetyViolation(
                        violation_type=ViolationType.STRUCTURAL_INCONSISTENCY,
                        severity=severity,
                        affected_pixels=n_gt,
                        mean_residual_m=0.0,
                        max_residual_m=0.0,
                        timestamp=timestamp,
                        frame_id=frame_id,
                        region_mask=gt_mask & ~pred_mask,
                        details=f"Static class {cls}: IoU={iou:.3f} < threshold={self.iou_thr}",
                    )
                )
        return violations, iou_map

    def _check_geometric_only(
        self,
        residual: np.ndarray,
        valid: np.ndarray,
        frame_id: int,
        timestamp: float,
        total_pixels: int,
    ) -> List[SafetyViolation]:
        """Pure geometric baseline: single global residual threshold."""
        violations = []
        if not valid.any():
            return violations

        res_valid = residual[valid]
        mean_res = float(np.mean(res_valid))
        max_res = float(np.max(res_valid))

        # Use average of τ_fn and τ_fp as global threshold
        tau_global = (self.tau_fn + self.tau_fp) / 2.0

        if mean_res > tau_global:
            severity = min(1.0, mean_res / (tau_global * 3.0))
            violations.append(
                SafetyViolation(
                    violation_type=ViolationType.DYNAMIC_FALSE_NEGATIVE,
                    severity=severity,
                    affected_pixels=int(np.sum(valid)),
                    mean_residual_m=mean_res,
                    max_residual_m=max_res,
                    timestamp=timestamp,
                    frame_id=frame_id,
                    details=f"Global geometric: mean={mean_res:.2f}m > τ={tau_global:.1f}m",
                )
            )

        # Also flag large local regions (top-5% residual)
        thr_95 = float(np.percentile(res_valid, 95))
        if thr_95 > tau_global:
            hotspot = residual > thr_95
            n_hot = int(np.sum(hotspot & valid))
            if n_hot > self.min_pix_frac * total_pixels:
                severity = min(1.0, thr_95 / (tau_global * 3.0))
                violations.append(
                    SafetyViolation(
                        violation_type=ViolationType.STATIC_FALSE_POSITIVE,
                        severity=severity,
                        affected_pixels=n_hot,
                        mean_residual_m=float(np.mean(residual[hotspot & valid])),
                        max_residual_m=max_res,
                        timestamp=timestamp,
                        frame_id=frame_id,
                        region_mask=hotspot & valid,
                        details=f"Global geometric hotspot: "
                        f"P95={thr_95:.2f}m, {n_hot} px",
                    )
                )
        return violations

    def reset(self) -> None:
        """Reset internal state (call between scenarios)."""
        self._ema_score = 1.0
        self._consecutive_violations = 0
