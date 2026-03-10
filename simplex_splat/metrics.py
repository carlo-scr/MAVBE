"""
Evaluation metrics and structured logging for Simplex-Splat experiments.

Records per-frame diagnostics and computes aggregate metrics:
  - True/False Positive/Negative rates
  - Response time (violation → brake trigger)
  - Safety score time-series
  - Per-scenario summaries
"""
from __future__ import annotations

import csv
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FrameRecord:
    """One row in the per-frame CSV log."""
    frame_id: int
    sim_time: float
    monitor_latency_ms: float
    safety_score: float
    is_safe: bool
    global_mean_residual: float
    global_max_residual: float
    dynamic_mean_residual: float
    static_mean_residual: float
    num_violations: int
    violation_types: str
    vehicle_speed_mps: float
    num_gaussians: int
    emergency_active: bool


@dataclass
class ScenarioResult:
    """Aggregate metrics for one scenario run."""
    scenario_name: str
    monitor_type: str
    total_frames: int = 0
    total_violations: int = 0
    true_positives: int = 0
    false_positives: int = 0
    true_negatives: int = 0
    false_negatives: int = 0
    response_time_ms: float = 0.0
    mean_monitor_latency_ms: float = 0.0
    min_safety_score: float = 0.0
    mean_safety_score: float = 0.0
    stopping_distance_m: float = 0.0
    vehicle_speed_at_trigger_mps: float = 0.0

    @property
    def tpr(self) -> float:
        """True Positive Rate (sensitivity)."""
        denom = self.true_positives + self.false_negatives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def fpr(self) -> float:
        """False Positive Rate."""
        denom = self.false_positives + self.true_negatives
        return self.false_positives / denom if denom > 0 else 0.0

    @property
    def precision(self) -> float:
        denom = self.true_positives + self.false_positives
        return self.true_positives / denom if denom > 0 else 0.0

    @property
    def f1(self) -> float:
        p = self.precision
        r = self.tpr
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


class MetricsLogger:
    """Writes per-frame CSV and scenario-level JSON summaries."""

    def __init__(self, output_dir: str, scenario_name: str, monitor_type: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.scenario_name = scenario_name
        self.monitor_type = monitor_type
        self._frame_records: List[FrameRecord] = []
        self._result = ScenarioResult(scenario_name=scenario_name, monitor_type=monitor_type)

        csv_path = self.output_dir / f"{scenario_name}_frames.csv"
        fieldnames = list(FrameRecord.__dataclass_fields__.keys())
        self._csv_file = open(csv_path, "w", newline="")
        self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
        self._csv_writer.writeheader()

    def log_frame(self, record: FrameRecord):
        """Append a frame record."""
        self._frame_records.append(record)
        self._csv_writer.writerow(asdict(record))
        self._csv_file.flush()

    def mark_ground_truth_hazard(self, frame_id: int, is_hazard: bool):
        """Label whether a ground-truth hazard is present at this frame.

        Called externally by the scenario runner which knows the GT events.
        Compare with monitor's ``is_safe`` to compute TP/FP/FN/TN.
        """
        rec = self._frame_records[len(self._frame_records) - 1]
        monitor_flagged = not rec.is_safe

        if is_hazard and monitor_flagged:
            self._result.true_positives += 1
        elif is_hazard and not monitor_flagged:
            self._result.false_negatives += 1
        elif not is_hazard and monitor_flagged:
            self._result.false_positives += 1
        else:
            self._result.true_negatives += 1

    def set_response_time(self, response_ms: float):
        self._result.response_time_ms = response_ms

    def set_stopping_info(self, distance_m: float, speed_mps: float):
        self._result.stopping_distance_m = distance_m
        self._result.vehicle_speed_at_trigger_mps = speed_mps

    def finalise(self):
        """Compute aggregate metrics and write JSON summary."""
        r = self._result
        r.total_frames = len(self._frame_records)

        scores = [f.safety_score for f in self._frame_records]
        latencies = [f.monitor_latency_ms for f in self._frame_records]

        r.min_safety_score = float(min(scores))
        r.mean_safety_score = float(np.mean(scores))
        r.mean_monitor_latency_ms = float(np.mean(latencies))
        r.total_violations = sum(f.num_violations for f in self._frame_records)

        summary_path = self.output_dir / f"{self.scenario_name}_{self.monitor_type}_summary.json"
        summary = {
            "scenario": r.scenario_name,
            "monitor": r.monitor_type,
            "total_frames": r.total_frames,
            "total_violations": r.total_violations,
            "true_positives": r.true_positives,
            "false_positives": r.false_positives,
            "true_negatives": r.true_negatives,
            "false_negatives": r.false_negatives,
            "TPR": round(r.tpr, 4),
            "FPR": round(r.fpr, 4),
            "precision": round(r.precision, 4),
            "response_time_ms": r.response_time_ms,
            "mean_monitor_latency_ms": r.mean_monitor_latency_ms,
            "min_safety_score": r.min_safety_score,
            "mean_safety_score": r.mean_safety_score,
            "stopping_distance_m": r.stopping_distance_m,
            "vehicle_speed_at_trigger_mps": r.vehicle_speed_at_trigger_mps,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)

        logger.info(
            "Scenario '%s' [%s] — TPR=%.2f  FPR=%.2f  F1=%.2f  response=%.1fms",
            r.scenario_name, r.monitor_type, r.tpr, r.fpr, r.f1, r.response_time_ms,
        )

        self._csv_file.close()


def save_image(img, path):
    """Save an image array (BGR or grayscale) to disk."""
    import cv2
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, img)


def depth_to_colormap(depth, max_depth=100.0):
    """Convert depth map to a JET colormap visualisation (BGR uint8)."""
    import cv2
    norm = np.clip(depth / max_depth, 0, 1)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return colored


def residual_to_colormap(residual, max_val=5.0):
    """Colourmap for depth residuals: green=low, red=high."""
    import cv2
    norm = np.clip(residual / max_val, 0, 1)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    return colored
