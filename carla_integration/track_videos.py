#!/usr/bin/env python3
"""
Batch video tracking pipeline for Simplex-Track validation.

Discovers .mp4 videos in an input directory, runs YOLOv9 + DeepSORT
(Behavioral EKF + OSNet ReID) tracking on each, and exports structured
per-video results (annotated video, trajectory plot, per-frame JSON).

Produces an aggregate summary JSON ready for the validation framework.

Usage:
    # Track all videos in carla_integration/
    python -m carla_integration.track_videos

    # Track videos in a specific folder
    python -m carla_integration.track_videos --input-dir runs/carla_scenarios/scenario_000

    # Use constant-velocity Kalman filter instead of behavioral EKF
    python -m carla_integration.track_videos --tracker cv

    # Custom confidence threshold + view results live
    python -m carla_integration.track_videos --conf-thres 0.5 --view-img
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torchvision.transforms as T

# ── Path setup ────────────────────────────────────────────────────────────────
FILE = Path(__file__).resolve()
REPO_ROOT = FILE.parents[1]  # MAVBE/
PERCEPTION_ROOT = REPO_ROOT / "perception"
YOLO_ROOT = PERCEPTION_ROOT / "yolov9"
DEEP_SORT_ROOT = PERCEPTION_ROOT / "deep_sort"

for p in [DEEP_SORT_ROOT, YOLO_ROOT, PERCEPTION_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ── Lazy imports (heavy) ──────────────────────────────────────────────────────
_models_loaded = False
_yolo_model = None
_reid_model = None
_reid_transform = None
_device = None


def _load_models(weights: str, device_str: str = "", half: bool = False):
    """Load YOLOv9 + OSNet ReID models once."""
    global _models_loaded, _yolo_model, _reid_model, _reid_transform, _device
    if _models_loaded:
        return

    from models.common import DetectMultiBackend
    from utils.torch_utils import select_device
    import torchreid

    _device = select_device(device_str)
    logger.info("Using device: %s", _device)

    # YOLOv9
    _yolo_model = DetectMultiBackend(weights, device=_device, fp16=half)
    _yolo_model.warmup(imgsz=(1, 3, 640, 640))
    logger.info("YOLOv9 loaded from %s", weights)

    # OSNet ReID
    _reid_model = torchreid.models.build_model(
        name="osnet_x1_0", num_classes=1000, pretrained=True
    )
    _reid_model.eval()
    _reid_model.to(_device)

    _reid_transform = T.Compose([
        T.ToPILImage(),
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    _models_loaded = True
    logger.info("OSNet ReID loaded")


# ── Result data classes ───────────────────────────────────────────────────────

@dataclass
class TrackFrame:
    """One tracked object in one frame."""
    frame_id: int
    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    cx: float
    cy: float


@dataclass
class TrackSummary:
    """Summary for one track across the video."""
    track_id: int
    n_frames: int
    first_frame: int
    last_frame: int
    trajectory: List[Tuple[float, float]]  # (cx, cy) per frame


@dataclass
class VideoResult:
    """Complete tracking result for one video."""
    video_path: str
    video_name: str
    total_frames: int = 0
    fps: float = 0.0
    duration_s: float = 0.0
    n_tracks: int = 0
    n_detections: int = 0
    track_summaries: List[dict] = field(default_factory=list)
    frames: List[List[dict]] = field(default_factory=list)
    output_video: str = ""
    output_trajectory_plot: str = ""
    processing_time_s: float = 0.0


# ── Tracking core ─────────────────────────────────────────────────────────────

COCO_NAMES = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant",
    "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush",
]


def _color_for_class(cls_id: int) -> Tuple[int, int, int]:
    if cls_id == 0:
        return (85, 45, 255)
    elif cls_id == 2:
        return (222, 82, 175)
    elif cls_id == 3:
        return (0, 204, 255)
    elif cls_id == 5:
        return (0, 149, 255)
    return (200, 100, 0)


def _extract_reid_feature(crop_bgr: np.ndarray) -> np.ndarray:
    """Extract normalised ReID feature from a BGR crop."""
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    tensor = _reid_transform(crop_rgb).unsqueeze(0).to(_device)
    with torch.no_grad():
        feat = _reid_model(tensor)
    feat = feat.cpu().numpy().flatten()
    norm = np.linalg.norm(feat)
    if norm > 0:
        feat /= norm
    return feat


def track_video(
    video_path: str,
    output_dir: str,
    weights: str,
    *,
    tracker_type: str = "sfekf",
    conf_thres: float = 0.75,
    iou_thres: float = 0.45,
    imgsz: int = 640,
    device: str = "",
    half: bool = False,
    view_img: bool = False,
    draw_trails: bool = True,
    vid_stride: int = 1,
    classes: Optional[List[int]] = None,
) -> VideoResult:
    """Run YOLOv9 + DeepSORT tracking on a single video.

    Args:
        video_path: Path to input video.
        output_dir: Directory for outputs (annotated video, trajectory plot, JSON).
        weights: Path to YOLOv9 weights.
        tracker_type: "sfekf" (behavioral EKF) or "cv" (constant-velocity KF).
        conf_thres: Detection confidence threshold.
        iou_thres: NMS IoU threshold.
        classes: Filter to these COCO class IDs (default: [0] = person only).

    Returns:
        VideoResult with per-frame tracking data and aggregate stats.
    """
    from deep_sort import nn_matching
    from deep_sort.detection import Detection
    from deep_sort.tracker import Tracker
    from utils.general import non_max_suppression, scale_boxes, check_img_size
    from utils.dataloaders import LoadImages

    if classes is None:
        classes = [0]  # person only by default

    _load_models(weights, device, half)

    video_path = str(video_path)
    video_name = Path(video_path).stem
    out_dir = Path(output_dir) / video_name
    out_dir.mkdir(parents=True, exist_ok=True)

    result = VideoResult(
        video_path=video_path,
        video_name=video_name,
    )

    t0 = time.time()
    logger.info("Tracking %s → %s", video_path, out_dir)

    # ── Tracker init ──────────────────────────────────────────────────────
    metric = nn_matching.NearestNeighborDistanceMetric(
        metric="cosine", matching_threshold=0.4, budget=100,
    )
    tracker = Tracker(metric)

    if tracker_type == "cv":
        # Swap the behavioral EKF for the plain Kalman filter
        from deep_sort import kalman_filter
        tracker.kf = kalman_filter.KalmanFilter()
        logger.info("Using constant-velocity Kalman filter")
    else:
        logger.info("Using behavioral EKF (SF-EKF)")

    # ── Dataloader ────────────────────────────────────────────────────────
    stride = _yolo_model.stride
    sz = check_img_size(imgsz, s=stride)
    dataset = LoadImages(video_path, img_size=sz, stride=stride,
                         auto=_yolo_model.pt, vid_stride=vid_stride)

    # ── Video writer ──────────────────────────────────────────────────────
    out_video_path = out_dir / f"{video_name}_tracked.mp4"
    vid_writer = None

    # ── Trail buffer ──────────────────────────────────────────────────────
    trails: Dict[int, deque] = {}

    # Per-frame storage
    all_frame_tracks: List[List[dict]] = []
    all_track_frames: Dict[int, List[TrackFrame]] = {}
    frame_idx = 0
    total_detections = 0

    for path, im, im0s, vid_cap, s in dataset:
        # Preprocess
        im_t = torch.from_numpy(im).to(_device)
        im_t = im_t.half() if _yolo_model.fp16 else im_t.float()
        im_t /= 255.0
        if im_t.ndim == 3:
            im_t = im_t.unsqueeze(0)

        # Inference + NMS
        pred = _yolo_model(im_t, augment=False)[0][1]
        pred = non_max_suppression(pred, conf_thres, iou_thres, max_det=1000)

        for det in pred:
            im0 = im0s.copy()
            vis = im0.copy()

            # Init video writer on first frame
            if vid_writer is None:
                fps = vid_cap.get(cv2.CAP_PROP_FPS) if vid_cap else 30.0
                h, w = im0.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                vid_writer = cv2.VideoWriter(str(out_video_path), fourcc, fps, (w, h))
                result.fps = fps
                result.total_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT)) if vid_cap else 0

            frame_tracks = []

            if len(det):
                det[:, :4] = scale_boxes(im_t.shape[2:], det[:, :4], im0.shape).round()

                # Filter classes
                if classes:
                    mask = torch.zeros(len(det), dtype=torch.bool)
                    for c in classes:
                        mask |= (det[:, 5] == c)
                    det = det[mask]

                # Build DeepSORT detections with ReID features
                detections = []
                for *xyxy, conf, cls in det:
                    x1, y1, x2, y2 = map(int, xyxy)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(im0.shape[1], x2), min(im0.shape[0], y2)
                    w_box, h_box = x2 - x1, y2 - y1
                    if w_box <= 0 or h_box <= 0:
                        continue

                    crop = im0[y1:y2, x1:x2]
                    if crop.size == 0:
                        continue

                    feature = _extract_reid_feature(crop)
                    detections.append(Detection([x1, y1, w_box, h_box], float(conf), feature))

                total_detections += len(detections)

                # Update tracker
                tracker.predict()
                tracker.update(detections)

                # Collect confirmed tracks
                for track in tracker.tracks:
                    if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                    tx1, ty1, tw, th = track.to_tlwh()
                    tx2, ty2 = tx1 + tw, ty1 + th
                    tid = track.track_id
                    cx, cy = (tx1 + tx2) / 2, (ty1 + ty2) / 2

                    tf = TrackFrame(
                        frame_id=frame_idx, track_id=tid,
                        x1=tx1, y1=ty1, x2=tx2, y2=ty2,
                        cx=cx, cy=cy,
                    )
                    frame_tracks.append(asdict(tf))

                    if tid not in all_track_frames:
                        all_track_frames[tid] = []
                    all_track_frames[tid].append(tf)

                    # Trail
                    if tid not in trails:
                        trails[tid] = deque(maxlen=64)
                    trails[tid].appendleft((int(cx), int(cy)))

                    # Draw on vis
                    color = _color_for_class(0)
                    cv2.rectangle(vis, (int(tx1), int(ty1)), (int(tx2), int(ty2)), color, 2)
                    label = f"{tid}:person"
                    (tw_t, th_t), _ = cv2.getTextSize(label, 0, 0.5, 2)
                    cv2.rectangle(vis, (int(tx1), int(ty1)),
                                  (int(tx1) + tw_t, int(ty1) - th_t - 3), color, -1)
                    cv2.putText(vis, label, (int(tx1), int(ty1) - 2),
                                0, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

                    # Draw trail
                    if draw_trails and tid in trails:
                        pts = trails[tid]
                        for j in range(1, len(pts)):
                            if pts[j - 1] is None or pts[j] is None:
                                continue
                            thickness = int(np.sqrt(64 / float(j + 1)) * 1.5)
                            cv2.line(vis, pts[j - 1], pts[j], color, thickness)
            else:
                tracker.predict()

            all_frame_tracks.append(frame_tracks)

            # Write frame
            if vid_writer is not None:
                vid_writer.write(vis)

            if view_img:
                cv2.imshow(video_name, vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            frame_idx += 1

    # ── Finalize ──────────────────────────────────────────────────────────
    if vid_writer is not None:
        vid_writer.release()
    if view_img:
        cv2.destroyAllWindows()

    result.output_video = str(out_video_path)
    result.processing_time_s = time.time() - t0
    result.n_detections = total_detections
    result.n_tracks = len(all_track_frames)
    result.duration_s = frame_idx / result.fps if result.fps > 0 else 0
    result.frames = all_frame_tracks

    # Build track summaries
    for tid, frames in all_track_frames.items():
        traj = [(f.cx, f.cy) for f in frames]
        result.track_summaries.append(asdict(TrackSummary(
            track_id=tid,
            n_frames=len(frames),
            first_frame=frames[0].frame_id,
            last_frame=frames[-1].frame_id,
            trajectory=traj,
        )))

    # ── Save trajectory plot ──────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(12, 8))
        for tid, frames in all_track_frames.items():
            pts = np.array([(f.cx, f.cy) for f in frames])
            ax.plot(pts[:, 0], pts[:, 1], marker="o", markersize=2, label=f"ID {tid}")
        ax.invert_yaxis()
        ax.set_xlabel("X pixels")
        ax.set_ylabel("Y pixels")
        ax.set_title(f"Trajectories — {video_name}")
        if len(all_track_frames) <= 20:
            ax.legend(fontsize=7)
        traj_path = out_dir / f"{video_name}_trajectories.png"
        fig.savefig(traj_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        result.output_trajectory_plot = str(traj_path)
        logger.info("Trajectory plot → %s", traj_path)
    except Exception as e:
        logger.warning("Could not save trajectory plot: %s", e)

    # ── Save per-frame JSON ───────────────────────────────────────────────
    json_path = out_dir / f"{video_name}_tracks.json"
    with open(json_path, "w") as f:
        json.dump({
            "video": video_name,
            "fps": result.fps,
            "total_frames": frame_idx,
            "n_tracks": result.n_tracks,
            "n_detections": result.n_detections,
            "track_summaries": result.track_summaries,
        }, f, indent=2, default=str)
    logger.info("Track data → %s", json_path)

    logger.info("Done: %d frames, %d tracks, %d detections in %.1fs",
                frame_idx, result.n_tracks, total_detections, result.processing_time_s)

    return result


# ── Batch pipeline ────────────────────────────────────────────────────────────

VID_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def discover_videos(input_dir: str, recursive: bool = False) -> List[Path]:
    """Find all video files in the given directory."""
    root = Path(input_dir)
    if not root.exists():
        logger.error("Input directory does not exist: %s", root)
        return []
    pattern = "**/*" if recursive else "*"
    videos = sorted(
        p for p in root.glob(pattern)
        if p.suffix.lower() in VID_EXTS and not p.name.startswith(".")
    )
    logger.info("Found %d video(s) in %s", len(videos), root)
    return videos


def run_pipeline(args: argparse.Namespace) -> dict:
    """Run the full batch tracking pipeline."""
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = discover_videos(str(input_dir), recursive=args.recursive)
    if not videos:
        logger.warning("No videos found in %s", input_dir)
        return {"videos": [], "summary": {}}

    results: List[VideoResult] = []
    t0 = time.time()

    for i, vpath in enumerate(videos):
        logger.info("═══ [%d/%d] %s ═══", i + 1, len(videos), vpath.name)
        try:
            vr = track_video(
                video_path=str(vpath),
                output_dir=str(output_dir),
                weights=args.weights,
                tracker_type=args.tracker,
                conf_thres=args.conf_thres,
                iou_thres=args.iou_thres,
                imgsz=args.imgsz,
                device=args.device,
                half=args.half,
                view_img=args.view_img,
                draw_trails=args.draw_trails,
                vid_stride=args.vid_stride,
                classes=args.classes,
            )
            results.append(vr)
        except Exception as e:
            logger.error("Failed on %s: %s", vpath.name, e, exc_info=True)

    total_time = time.time() - t0

    # ── Aggregate summary ─────────────────────────────────────────────────
    summary = {
        "n_videos": len(results),
        "total_processing_time_s": round(total_time, 1),
        "total_frames": sum(r.total_frames for r in results),
        "total_tracks": sum(r.n_tracks for r in results),
        "total_detections": sum(r.n_detections for r in results),
        "tracker_type": args.tracker,
        "conf_thres": args.conf_thres,
        "iou_thres": args.iou_thres,
        "per_video": [],
    }

    for vr in results:
        summary["per_video"].append({
            "video": vr.video_name,
            "frames": vr.total_frames,
            "duration_s": round(vr.duration_s, 1),
            "fps": round(vr.fps, 1),
            "n_tracks": vr.n_tracks,
            "n_detections": vr.n_detections,
            "processing_time_s": round(vr.processing_time_s, 1),
            "output_video": vr.output_video,
            "output_trajectory_plot": vr.output_trajectory_plot,
        })

    # Save aggregate summary
    summary_path = output_dir / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("Pipeline complete: %d videos, %d total tracks in %.0fs",
                len(results), summary["total_tracks"], total_time)
    logger.info("Summary → %s", summary_path)

    return summary


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch YOLOv9 + DeepSORT tracking pipeline for CARLA videos",
    )
    parser.add_argument("--input-dir", type=str,
                        default=str(REPO_ROOT / "carla_integration"),
                        help="Directory containing input videos")
    parser.add_argument("--output-dir", type=str,
                        default=str(REPO_ROOT / "runs" / "tracking"),
                        help="Output directory for tracked videos + data")
    parser.add_argument("--weights", type=str,
                        default=str(PERCEPTION_ROOT / "yolov9" / "weights" / "yolov9-c.pt"),
                        help="YOLOv9 weights path")
    parser.add_argument("--tracker", type=str, default="sfekf",
                        choices=["sfekf", "cv"],
                        help="Tracker type: sfekf (behavioral EKF) or cv (constant velocity)")
    parser.add_argument("--conf-thres", type=float, default=0.5,
                        help="Detection confidence threshold (default: 0.5)")
    parser.add_argument("--iou-thres", type=float, default=0.45,
                        help="NMS IoU threshold (default: 0.45)")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Inference image size (default: 640)")
    parser.add_argument("--device", type=str, default="",
                        help="Device: '' (auto), 'cpu', '0', 'mps'")
    parser.add_argument("--half", action="store_true",
                        help="Use FP16 inference")
    parser.add_argument("--view-img", action="store_true",
                        help="Show tracking results live")
    parser.add_argument("--draw-trails", action="store_true", default=True,
                        help="Draw trajectory trails on output video")
    parser.add_argument("--vid-stride", type=int, default=1,
                        help="Video frame stride (default: 1)")
    parser.add_argument("--recursive", action="store_true",
                        help="Search for videos recursively in subdirs")
    parser.add_argument("--classes", type=int, nargs="+", default=[0],
                        help="COCO class IDs to track (default: 0=person)")

    args = parser.parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
