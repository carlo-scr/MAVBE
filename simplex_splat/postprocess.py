"""
Postprocessing pipeline for CARLA MP4 videos.

Runs YOLO + ReID + Deep SORT (Behavioral EKF) tracking, then exports
structured per-frame / per-track data as JSON + CSV for downstream
safety analysis (failure distribution, reachability).

Usage:
    python -m simplex_splat.postprocess \
        --source carla_pedestrian_60s.mp4 \
        --weights perception/yolov9/weights/yolov9-c.pt \
        --output runs/simplex_splat/analysis
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T

# ---------------------------------------------------------------------------
# Path setup — mirrors detect_dual_tracking.py
# ---------------------------------------------------------------------------
_THIS = Path(__file__).resolve()
_PROJECT = _THIS.parents[1]  # MAVBE root
_PERC = _PROJECT / "perception"
_YOLO = _PERC / "yolov9"
_DS = _PERC / "deep_sort"

for p in [_DS, _YOLO, _PERC]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages
from utils.general import check_img_size, non_max_suppression, scale_boxes
from utils.torch_utils import select_device

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class DetectionRecord:
    """One YOLO detection (before tracking)."""
    frame_idx: int
    bbox_tlwh: List[float]   # [x, y, w, h]
    confidence: float
    class_id: int
    class_name: str


@dataclass
class TrackRecord:
    """Per-frame state for one confirmed track."""
    frame_idx: int
    track_id: int
    bbox_xyxy: List[float]
    centre_xy: List[float]
    class_id: int
    class_name: str

    # EKF state (8-D)
    mean: List[float]          # [x, y, a, h, vx, vy, va, vh]
    covariance_diag: List[float]  # diagonal of P
    time_since_update: int
    age: int


@dataclass
class FrameResult:
    """Aggregate per-frame output."""
    frame_idx: int
    timestamp_s: float
    num_detections: int
    num_tracks: int
    detections: List[DetectionRecord] = field(default_factory=list)
    tracks: List[TrackRecord] = field(default_factory=list)
    inference_ms: float = 0.0


# COCO class names subset relevant to driving
_COCO = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]

SAFETY_CLASSES = {0: "person", 2: "car", 3: "motorbike", 5: "bus", 7: "truck"}

# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class TrackingPipeline:
    """Wraps YOLO + ReID + Deep SORT into a callable postprocessor."""

    def __init__(
        self,
        weights: str,
        device: str = "",
        conf_thres: float = 0.5,
        iou_thres: float = 0.55,
        imgsz: int = 640,
        max_cosine_dist: float = 0.4,
        nn_budget: int = 100,
    ):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.imgsz = (imgsz, imgsz)

        # YOLO
        self.device = select_device(device)
        self.model = DetectMultiBackend(weights, device=self.device, fp16=False)
        self.stride = self.model.stride
        self.imgsz = check_img_size(self.imgsz, s=self.stride)

        # ReID
        import torchreid
        self.reid = torchreid.models.build_model(
            name="osnet_x1_0", num_classes=1000, pretrained=True
        )
        self.reid.eval().to(self.device)
        self.reid_tf = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Tracker
        metric = nn_matching.NearestNeighborDistanceMetric(
            metric="cosine",
            matching_threshold=max_cosine_dist,
            budget=nn_budget,
        )
        self.tracker = Tracker(metric)
        self.model.warmup(imgsz=(1, 3, *self.imgsz))

    def process_video(
        self,
        source: str,
        vid_stride: int = 1,
        draw: bool = True,
    ) -> Tuple[List[FrameResult], Optional[str]]:
        """Run detection + tracking on a video file.

        Returns
        -------
        results : list[FrameResult]
        annotated_path : str | None  (path to annotated video if draw=True)
        """
        dataset = LoadImages(source, img_size=self.imgsz, stride=self.stride, auto=self.model.pt, vid_stride=vid_stride)
        results: List[FrameResult] = []
        trail_buf: Dict[int, deque] = {}

        # video writer
        out_path = None
        vid_writer = None

        for frame_idx, (path, im, im0s, vid_cap, _s) in enumerate(dataset):
            t0 = time.perf_counter()
            fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30.0
            timestamp = frame_idx / fps

            im_t = torch.from_numpy(im).to(self.device).float() / 255.0
            if im_t.ndim == 3:
                im_t = im_t.unsqueeze(0)

            # Detection
            pred = self.model(im_t, augment=False)[0][1]
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, max_det=1000)

            frame_dets: List[DetectionRecord] = []
            ds_dets: List[Detection] = []

            for det in pred:
                if len(det) == 0:
                    continue
                det[:, :4] = scale_boxes(im_t.shape[2:], det[:, :4], im0s.shape).round()

                for *xyxy, conf, cls_id in reversed(det):
                    x1, y1, x2, y2 = (int(v) for v in xyxy)
                    w, h = x2 - x1, y2 - y1
                    cls_int = int(cls_id)
                    cls_name = _COCO[cls_int] if cls_int < len(_COCO) else str(cls_int)

                    frame_dets.append(DetectionRecord(
                        frame_idx=frame_idx,
                        bbox_tlwh=[x1, y1, w, h],
                        confidence=float(conf),
                        class_id=cls_int,
                        class_name=cls_name,
                    ))

                    # ReID feature
                    crop = im0s[max(0, y1):y2, max(0, x1):x2]
                    if crop.size == 0:
                        continue
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    inp = self.reid_tf(crop_rgb).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        feat = self.reid(inp).cpu().numpy().flatten()
                    feat = feat / (np.linalg.norm(feat) + 1e-8)

                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    ds_dets.append(Detection([x1, y1, w, h], float(conf), feat))

            # Track
            self.tracker.predict()
            self.tracker.update(ds_dets)

            frame_tracks: List[TrackRecord] = []
            for trk in self.tracker.tracks:
                if not trk.is_confirmed() or trk.time_since_update > 1:
                    continue
                tlwh = trk.to_tlwh()
                x1t, y1t = tlwh[0], tlwh[1]
                x2t, y2t = x1t + tlwh[2], y1t + tlwh[3]
                cx, cy = (x1t + x2t) / 2, (y1t + y2t) / 2
                frame_tracks.append(TrackRecord(
                    frame_idx=frame_idx,
                    track_id=trk.track_id,
                    bbox_xyxy=[float(x1t), float(y1t), float(x2t), float(y2t)],
                    centre_xy=[float(cx), float(cy)],
                    class_id=0,  # class bookkeeping not in base tracker
                    class_name="object",
                    mean=trk.mean.tolist(),
                    covariance_diag=np.diag(trk.covariance).tolist(),
                    time_since_update=trk.time_since_update,
                    age=trk.age,
                ))

                if trk.track_id not in trail_buf:
                    trail_buf[trk.track_id] = deque(maxlen=64)
                trail_buf[trk.track_id].appendleft((int(cx), int(cy)))

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            results.append(FrameResult(
                frame_idx=frame_idx,
                timestamp_s=round(timestamp, 4),
                num_detections=len(frame_dets),
                num_tracks=len(frame_tracks),
                detections=frame_dets,
                tracks=frame_tracks,
                inference_ms=round(elapsed_ms, 2),
            ))

            # Draw annotated frame
            if draw:
                vis = im0s.copy()
                for tr in frame_tracks:
                    x1v, y1v, x2v, y2v = (int(v) for v in tr.bbox_xyxy)
                    color = (85, 45, 255) if tr.class_id == 0 else (222, 82, 175)
                    cv2.rectangle(vis, (x1v, y1v), (x2v, y2v), color, 2)
                    label = f"T{tr.track_id}"
                    cv2.putText(vis, label, (x1v, y1v - 4), 0, 0.5,
                                (255, 255, 255), 1, cv2.LINE_AA)
                    tid = tr.track_id
                    if tid in trail_buf:
                        pts = list(trail_buf[tid])
                        for j in range(1, len(pts)):
                            if pts[j - 1] is None or pts[j] is None:
                                continue
                            thick = int(np.sqrt(64 / float(j + 1)) * 1.5)
                            cv2.line(vis, pts[j - 1], pts[j], color, thick)

                if vid_writer is None:
                    src_p = Path(source)
                    out_path = str(src_p.parent / f"{src_p.stem}_tracked.mp4")
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    h_v, w_v = vis.shape[:2]
                    vid_writer = cv2.VideoWriter(out_path, fourcc, fps, (w_v, h_v))
                vid_writer.write(vis)

            if frame_idx % 50 == 0:
                logger.info(
                    "frame %d | dets=%d tracks=%d  (%.1f ms)",
                    frame_idx, len(frame_dets), len(frame_tracks), elapsed_ms,
                )

        if vid_writer is not None:
            vid_writer.release()
        return results, out_path


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_results(results: List[FrameResult], out_dir: str) -> Dict[str, str]:
    """Persist results to JSON + per-track CSV.

    Returns dict of output file paths.
    """
    os.makedirs(out_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    # ---- Full JSON (all frames) ----
    json_path = os.path.join(out_dir, "tracking_results.json")
    serialisable = []
    for fr in results:
        d = {
            "frame_idx": fr.frame_idx,
            "timestamp_s": fr.timestamp_s,
            "num_detections": fr.num_detections,
            "num_tracks": fr.num_tracks,
            "inference_ms": fr.inference_ms,
            "tracks": [asdict(t) for t in fr.tracks],
        }
        serialisable.append(d)
    with open(json_path, "w") as f:
        json.dump(serialisable, f, indent=1)
    paths["json"] = json_path

    # ---- Per-track trajectory CSV (for reachability) ----
    track_data: Dict[int, List[TrackRecord]] = defaultdict(list)
    for fr in results:
        for t in fr.tracks:
            track_data[t.track_id].append(t)

    csv_path = os.path.join(out_dir, "tracks.csv")
    with open(csv_path, "w") as f:
        header = "track_id,frame_idx,cx,cy,vx,vy,cov_x,cov_y,cov_vx,cov_vy,bbox_w,bbox_h,age,time_since_update\n"
        f.write(header)
        for tid, recs in sorted(track_data.items()):
            for r in recs:
                m = r.mean
                cd = r.covariance_diag
                bw = r.bbox_xyxy[2] - r.bbox_xyxy[0]
                bh = r.bbox_xyxy[3] - r.bbox_xyxy[1]
                f.write(
                    f"{tid},{r.frame_idx},{r.centre_xy[0]:.1f},{r.centre_xy[1]:.1f},"
                    f"{m[4]:.4f},{m[5]:.4f},"
                    f"{cd[0]:.6f},{cd[1]:.6f},{cd[4]:.6f},{cd[5]:.6f},"
                    f"{bw:.1f},{bh:.1f},{r.age},{r.time_since_update}\n"
                )
    paths["csv"] = csv_path

    logger.info("Results saved to %s", out_dir)
    return paths


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="Postprocess CARLA video → tracking + analysis data")
    parser.add_argument("--source", required=True, help="Path to MP4 video")
    parser.add_argument("--weights", default=str(_PERC / "yolov9" / "weights" / "yolov9-c.pt"))
    parser.add_argument("--output", default="runs/simplex_splat/analysis")
    parser.add_argument("--conf-thres", type=float, default=0.5)
    parser.add_argument("--iou-thres", type=float, default=0.55)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    parser.add_argument("--vid-stride", type=int, default=1)
    parser.add_argument("--no-draw", action="store_true")
    args = parser.parse_args()

    pipe = TrackingPipeline(
        weights=args.weights,
        device=args.device,
        conf_thres=args.conf_thres,
        iou_thres=args.iou_thres,
        imgsz=args.imgsz,
    )

    results, vid_path = pipe.process_video(
        args.source, vid_stride=args.vid_stride, draw=not args.no_draw,
    )
    paths = save_results(results, args.output)

    print(f"\n{'='*60}")
    print(f"Tracking complete: {len(results)} frames processed")
    print(f"  JSON : {paths['json']}")
    print(f"  CSV  : {paths['csv']}")
    if vid_path:
        print(f"  Video: {vid_path}")
    print(f"{'='*60}")

    # Automatically run analysis
    from simplex_splat.analysis import run_full_analysis
    run_full_analysis(paths["csv"], paths["json"], args.output)


if __name__ == "__main__":
    main()
