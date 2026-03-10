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


class ViolationType(Enum):
    DYNAMIC_FALSE_NEGATIVE = auto()
    STATIC_FALSE_POSITIVE = auto()
    STRUCTURAL_INCONSISTENCY = auto()


@dataclass
class SafetyViolation:
    """Describes one safety violation detected by the monitor."""
    violation_type: ViolationType
    severity: float
    affected_pixels: int
    mean_residual_m: float
    max_residual_m: float
    timestamp: float
    frame_id: int
    region_mask: Optional[np.ndarray] = None
    details: str = ""


@dataclass
class MonitorState:
    """Per-frame diagnostic state exposed for logging / visualisation."""
    frame_id: int = 0
    timestamp: float = 0.0
    monitor_latency_ms: float = 0.0
    global_mean_residual: float = 0.0
    global_max_residual: float = 0.0
    dynamic_mean_residual: float = 0.0
    static_mean_residual: float = 0.0
    structural_iou: Dict[int, float] = field(default_factory=dict)
    is_safe: bool = True
    violations: List[SafetyViolation] = field(default_factory=list)
    safety_score: float = 1.0


class SafetyMonitor:
    """Cross-validates a rendered 3DGS view against ground-truth sensors.

    Parameters
    ----------
    cfg : dict
        The ``monitor`` section of the master config.
    """

    def __init__(self, cfg: dict):
        self.mode = cfg.get("type", "semantic")
        self.tau_fn = cfg.get("tau_fn", 1.0)
        self.tau_fp = cfg.get("tau_fp", 2.0)
        self.iou_thr = cfg.get("iou_threshold", 0.5)
        self.min_pix_frac = cfg.get("min_critical_pixel_fraction", 0.001)
        self.critical_dynamic = set(cfg.get("critical_dynamic_classes", [4, 10]))
        self.critical_static = set(cfg.get("critical_static_classes", [12, 18]))
        self.ema_alpha = cfg.get("ema_alpha", 0.3)
        self._ema_score = 1.0
        self.patience = cfg.get("violation_patience", 2)
        self._consecutive_violations = 0
        self.max_response_ms = cfg.get("max_response_time_ms", 100.0)

    def check(
        self,
        rendered_depth: np.ndarray,
        gt_depth: np.ndarray,
        gt_semantic: np.ndarray,
        rendered_semantic: np.ndarray,
        frame_id: int,
        timestamp: float,
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

        # Compute residual
        residual = np.abs(rendered_depth.astype(np.float64) - gt_depth.astype(np.float64))
        valid = np.any(gt_depth > 0)
        total_pixels = gt_depth.shape[0] * gt_depth.shape[1]

        # Global statistics
        state.global_mean_residual = float(np.mean(residual))
        state.global_max_residual = float(np.max(residual))

        if self.mode == "semantic":
            # Semantic-aware checks
            dyn_violations = self._check_dynamic_safety(
                residual, valid, gt_semantic, frame_id, timestamp, total_pixels
            )
            violations.extend(dyn_violations)

            stat_violations = self._check_static_integrity(
                rendered_depth, gt_depth, valid, gt_semantic, frame_id, timestamp, total_pixels
            )
            violations.extend(stat_violations)

            struct_violations, iou_map = self._check_structural_consistency(
                gt_semantic, rendered_semantic, frame_id, timestamp, total_pixels
            )
            violations.extend(struct_violations)
            state.structural_iou = iou_map
        else:
            # Pure geometric baseline
            geo_violations = self._check_geometric_only(
                residual, valid, frame_id, timestamp, total_pixels
            )
            violations.extend(geo_violations)

        state.violations = violations

        # Latency
        latency_ms = (time.perf_counter() - t0) * 1000.0
        state.monitor_latency_ms = latency_ms
        if latency_ms > self.max_response_ms:
            logger.warning("Monitor latency %.1f ms exceeds budget %.1f ms", latency_ms, self.max_response_ms)

        # EMA safety score
        max_sev = max((v.severity for v in violations), default=0.0)
        raw_score = max(0.0, 1.0 - max_sev)
        self._ema_score = self.ema_alpha * raw_score + (1.0 - self.ema_alpha) * self._ema_score
        state.safety_score = self._ema_score

        # Debounce via patience
        if violations:
            self._consecutive_violations += 1
        else:
            self._consecutive_violations = 0

        state.is_safe = self._consecutive_violations < self.patience

        return state

    def _check_dynamic_safety(
        self, residual, valid, gt_semantic, frame_id, timestamp, total_pixels
    ) -> List[SafetyViolation]:
        """Ensure every dynamic actor in GT has non-empty map coverage."""
        violations = []
        for cls in self.critical_dynamic:
            mask = gt_semantic == int(cls)
            n_pixels = int(np.sum(mask))
            if n_pixels < self.min_pix_frac * total_pixels:
                continue
            region_res = residual[mask]
            mean_res = float(np.mean(region_res))
            max_res = float(np.max(region_res))
            if mean_res > self.tau_fn:
                severity = min(1.0, mean_res / self.tau_fn - 1.0)
                violations.append(SafetyViolation(
                    violation_type=ViolationType.DYNAMIC_FALSE_NEGATIVE,
                    severity=severity,
                    affected_pixels=n_pixels,
                    mean_residual_m=mean_res,
                    max_residual_m=max_res,
                    timestamp=timestamp,
                    frame_id=frame_id,
                    details=f"Dynamic class {cls}: mean_residual={mean_res:.2f}m > τ_fn={self.tau_fn:.1f}m ({n_pixels} px)",
                ))
        return violations

    def _check_static_integrity(
        self, rendered_depth, gt_depth, valid, gt_semantic, frame_id, timestamp, total_pixels
    ) -> List[SafetyViolation]:
        """Detect phantom obstacles in drivable road surface."""
        violations = []
        road_mask = gt_semantic == int(7)
        n_pixels = int(np.sum(road_mask))
        if n_pixels < self.min_pix_frac * total_pixels:
            return violations

        # Phantom = rendered depth significantly LESS than GT (obstacle where none exists)
        underestimate = gt_depth[road_mask] - rendered_depth[road_mask]
        phantom_mask_local = underestimate > self.tau_fp
        n_phantom = int(np.sum(phantom_mask_local))

        if n_phantom < self.min_pix_frac * total_pixels:
            return violations

        mean_under = float(np.mean(underestimate[phantom_mask_local]))
        max_under = float(np.max(underestimate[phantom_mask_local]))
        severity = min(1.0, mean_under / self.tau_fp - 1.0)

        full_phantom = np.zeros_like(gt_semantic, dtype=np.bool_)
        road_indices = np.where(road_mask)
        phantom_indices = np.where(phantom_mask_local)

        violations.append(SafetyViolation(
            violation_type=ViolationType.STATIC_FALSE_POSITIVE,
            severity=severity,
            affected_pixels=n_phantom,
            mean_residual_m=mean_under,
            max_residual_m=max_under,
            timestamp=timestamp,
            frame_id=frame_id,
            details=f"Phantom obstacle on road: {n_phantom} px, mean_gap={mean_under:.2f}m > τ_fp={self.tau_fp:.1f}m",
        ))
        return violations

    def _check_structural_consistency(
        self, gt_semantic, rendered_semantic, frame_id, timestamp, total_pixels
    ) -> Tuple[List[SafetyViolation], Dict[int, float]]:
        """Check IoU of static safety-critical features."""
        violations = []
        iou_map: Dict[int, float] = {}

        for cls in self.critical_static:
            gt_mask = gt_semantic == int(cls)
            pred_mask = rendered_semantic == int(cls)
            intersection = int(np.sum(gt_mask & pred_mask))
            union = int(np.sum(gt_mask | pred_mask))
            n_gt = int(np.sum(gt_mask))

            if n_gt < self.min_pix_frac * total_pixels:
                continue

            iou = intersection / union if union > 0 else 0.0
            iou_map[cls] = iou

            if iou < self.iou_thr:
                severity = min(1.0, 1.0 - iou)
                violations.append(SafetyViolation(
                    violation_type=ViolationType.STRUCTURAL_INCONSISTENCY,
                    severity=severity,
                    affected_pixels=n_gt,
                    mean_residual_m=0.0,
                    max_residual_m=0.0,
                    timestamp=timestamp,
                    frame_id=frame_id,
                    details=f"Static class {cls}: IoU={iou:.3f} < threshold={self.iou_thr}",
                ))

        return violations, iou_map

    def _check_geometric_only(
        self, residual, valid, frame_id, timestamp, total_pixels
    ) -> List[SafetyViolation]:
        """Pure geometric baseline: single global residual threshold."""
        violations = []
        if not np.any(residual > 0):
            return violations

        res_valid = residual
        mean_res = float(np.mean(res_valid))
        max_res = float(np.max(res_valid))
        tau_global = min(self.tau_fn, self.tau_fp)

        # Global mean check
        if mean_res > tau_global:
            severity = min(1.0, mean_res / tau_global - 1.0)
            violations.append(SafetyViolation(
                violation_type=ViolationType.DYNAMIC_FALSE_NEGATIVE,
                severity=severity,
                affected_pixels=total_pixels,
                mean_residual_m=mean_res,
                max_residual_m=max_res,
                timestamp=timestamp,
                frame_id=frame_id,
                details=f"Global geometric: mean={mean_res:.2f}m > τ={tau_global:.1f}m",
            ))

        # Hotspot check (95th percentile)
        thr_95 = float(np.percentile(res_valid, 95))
        hotspot = res_valid > thr_95
        n_hot = int(np.sum(hotspot))
        if thr_95 > tau_global and n_hot > self.min_pix_frac * total_pixels:
            severity = min(1.0, thr_95 / tau_global - 1.0)
            violations.append(SafetyViolation(
                violation_type=ViolationType.STATIC_FALSE_POSITIVE,
                severity=severity,
                affected_pixels=n_hot,
                mean_residual_m=thr_95,
                max_residual_m=max_res,
                timestamp=timestamp,
                frame_id=frame_id,
                details=f"Global geometric hotspot: P95={thr_95:.2f}m, {n_hot} px",
            ))

        return violations

    def reset(self):
        """Reset internal state (call between scenarios)."""
        self._ema_score = 1.0
        self._consecutive_violations = 0
