#!/usr/bin/env python3
"""
Batch runner for the CARLA Simplex-Track validation experiments.

Runs all combinations of {tracker} × {disturbance} with a single command.
Each combination gets its own output directory under runs/carla_experiments/.

Example:
    python -m carla_integration.run_experiment_batch
    python -m carla_integration.run_experiment_batch --n-scenarios 5 --n-ped 7
    python -m carla_integration.run_experiment_batch --quick
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

TRACKERS = ["cv", "sfekf", "sfekf_simplex"]
DISTURBANCES = ["nominal", "fuzzed"]


def main():
    parser = argparse.ArgumentParser(
        description="Batch runner: all tracker × disturbance combinations"
    )
    parser.add_argument("--n-scenarios", type=int, default=2,
                        help="Scenarios per combination (default: 2)")
    parser.add_argument("--n-ped", type=int, default=5,
                        help="Pedestrians per scenario (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed (default: 42)")
    parser.add_argument("--v-ego", type=float, default=10.0,
                        help="Ego speed km/h (default: 10)")
    parser.add_argument("--tau-safe", type=float, default=2.0,
                        help="Safety threshold for simplex (default: 2.0)")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--output-root", type=str,
                        default="runs/carla_experiments",
                        help="Root output directory")
    parser.add_argument("--quick", action="store_true",
                        help="Alias for --n-scenarios 2")
    parser.add_argument("--trackers", type=str, nargs="+",
                        default=TRACKERS, choices=TRACKERS,
                        help="Tracker(s) to run (default: all)")
    parser.add_argument("--disturbances", type=str, nargs="+",
                        default=DISTURBANCES, choices=DISTURBANCES,
                        help="Disturbance(s) to run (default: all)")
    args = parser.parse_args()

    if args.quick:
        args.n_scenarios = 2

    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    combos = [(t, d) for t in args.trackers for d in args.disturbances]
    total = len(combos)
    summaries = {}

    logger.info("=" * 60)
    logger.info("CARLA EXPERIMENT BATCH: %d combinations", total)
    logger.info("  trackers:     %s", args.trackers)
    logger.info("  disturbances: %s", args.disturbances)
    logger.info("  n_scenarios:  %d  |  n_ped: %d  |  seed: %d",
                args.n_scenarios, args.n_ped, args.seed)
    logger.info("=" * 60)

    t0 = time.time()

    for idx, (tracker, disturbance) in enumerate(combos, 1):
        run_dir = root / f"{tracker}_{disturbance}"
        logger.info("\n[%d/%d]  tracker=%s  disturbance=%s  → %s",
                     idx, total, tracker, disturbance, run_dir)

        cmd = [
            sys.executable, "-m", "carla_integration.run_scenarios",
            "--tracker", tracker,
            "--disturbance", disturbance,
            "--n-scenarios", str(args.n_scenarios),
            "--n-ped", str(args.n_ped),
            "--seed", str(args.seed),
            "--v-ego", str(args.v_ego),
            "--tau-safe", str(args.tau_safe),
            "--host", args.host,
            "--port", str(args.port),
            "--output-dir", str(run_dir),
        ]
        if args.save_video:
            cmd.append("--save-video")

        logger.info("  CMD: %s", " ".join(cmd))
        result = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parent.parent))

        if result.returncode != 0:
            logger.error("  FAILED (exit code %d)", result.returncode)
        else:
            logger.info("  DONE")

        # Read aggregate summary if it was created
        agg_path = run_dir / "aggregate_summary.json"
        if agg_path.exists():
            with open(agg_path) as f:
                summaries[f"{tracker}_{disturbance}"] = json.load(f)

    elapsed = time.time() - t0

    # Print comparison table
    logger.info("\n" + "=" * 60)
    logger.info("BATCH COMPLETE: %d combinations in %.0fs", total, elapsed)
    logger.info("=" * 60)
    logger.info("%-20s  %6s  %6s  %6s  %6s  %6s",
                "Config", "Coll%", "ADE", "FDE", "FP_brk", "rho_min")
    logger.info("-" * 60)
    for key, s in summaries.items():
        logger.info("%-20s  %5.1f%%  %6.3f  %6.3f  %6.1f  %7.3f",
                     key,
                     s.get("collision_rate_pct", 0),
                     s.get("mean_ade", 0),
                     s.get("mean_fde", 0),
                     s.get("mean_fp_brakes", 0),
                     s.get("min_rho_min", 0))

    # Save combined results
    combined_path = root / "batch_summary.json"
    with open(combined_path, "w") as f:
        json.dump(summaries, f, indent=2)
    logger.info("\nBatch summary saved to %s", combined_path)


if __name__ == "__main__":
    main()
