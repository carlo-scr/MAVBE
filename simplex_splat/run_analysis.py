"""
simplex_splat.run_analysis — unified CLI entry point.

End-to-end: takes a CARLA MP4, runs detection + tracking, then
computes failure distributions and reachability analysis.

    python -m simplex_splat.run_analysis \
        --source carla_pedestrian_60s.mp4 \
        --weights perception/yolov9/weights/yolov9-c.pt \
        --output runs/simplex_splat/analysis
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    _proj = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser(
        description="Simplex-Splat: CARLA video → tracking → safety analysis"
    )
    parser.add_argument("--source", required=True, help="CARLA MP4 video path")
    parser.add_argument(
        "--weights",
        default=str(_proj / "perception" / "yolov9" / "weights" / "yolov9-c.pt"),
        help="YOLOv9 weights",
    )
    parser.add_argument("--output", default="runs/simplex_splat/analysis")
    parser.add_argument("--conf-thres", type=float, default=0.5)
    parser.add_argument("--iou-thres", type=float, default=0.55)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="")
    parser.add_argument("--vid-stride", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=30,
                        help="Reachability horizon in frames")
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--no-draw", action="store_true")
    parser.add_argument(
        "--skip-tracking", action="store_true",
        help="Skip tracking (use existing CSV/JSON in output dir)",
    )
    args = parser.parse_args()

    csv_path = str(Path(args.output) / "tracks.csv")
    json_path = str(Path(args.output) / "tracking_results.json")

    if not args.skip_tracking:
        logger.info("=" * 60)
        logger.info("Step 1/2: Detection + Tracking")
        logger.info("=" * 60)
        from simplex_splat.postprocess import TrackingPipeline, save_results

        pipe = TrackingPipeline(
            weights=args.weights,
            device=args.device,
            conf_thres=args.conf_thres,
            iou_thres=args.iou_thres,
            imgsz=args.imgsz,
        )
        results, vid_path = pipe.process_video(
            args.source, vid_stride=args.vid_stride, draw=not args.no_draw
        )
        paths = save_results(results, args.output)
        csv_path = paths["csv"]
        json_path = paths["json"]
        logger.info("Tracking done: %d frames → %s", len(results), args.output)
        if vid_path:
            logger.info("Annotated video: %s", vid_path)
    else:
        logger.info("Skipping tracking — using existing outputs in %s", args.output)

    logger.info("=" * 60)
    logger.info("Step 2/2: Failure Distribution + Reachability Analysis")
    logger.info("=" * 60)
    from simplex_splat.analysis import run_full_analysis

    summary = run_full_analysis(
        csv_path, json_path, args.output,
        horizon=args.horizon, fps=args.fps,
    )

    # Print key results
    fd = summary["failure_distribution"]
    print(f"\n{'='*60}")
    print("  SIMPLEX-SPLAT ANALYSIS RESULTS")
    print(f"{'='*60}")
    print(f"  Tracks analysed       : {fd['total_tracks']}")
    print(f"  Track lifetime (mean) : {fd['lifetime_mean']:.1f} frames")
    print(f"  Weibull fit           : k={fd['weibull_k']:.2f}  λ={fd['weibull_lam']:.1f}")
    print(f"  P(loss per frame)     : {fd['p_loss_per_frame']:.4f}")
    print(f"  Innovation μ±σ        : {fd['innovation_mean']:.2f} ± {fd['innovation_std']:.2f} px")
    print(f"  Fragmentation rate    : {fd['fragmentation_rate']:.4f}")

    reach = summary.get("reachability", {}).get("tracks", {})
    if reach:
        print(f"\n  Reachability ({args.horizon} frames ahead):")
        for tid, info in reach.items():
            ttc = info.get("ttc_s")
            pmax = info.get("max_p_collision", 0)
            ttc_str = f"{ttc:.2f}s" if ttc is not None else "∞"
            print(f"    Track {tid}: TTC={ttc_str}  max P(col)={pmax:.4f}")

    print(f"\n  Full results → {args.output}/analysis_summary.json")
    print(f"  Plots        → {args.output}/figures/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
