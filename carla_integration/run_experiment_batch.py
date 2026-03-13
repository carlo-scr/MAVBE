#!/usr/bin/env python3
"""
End-to-end CARLA Simplex-Track validation pipeline.

Runs all requested (tracker × disturbance) combinations, generates per-scenario
plots, and produces a unified summary with metrics.

Output structure (timestamped to avoid overwrites):

    runs/carla_scenarios/output_YYYYMMDD_HHMMSS/
    ├── nominal/
    │   ├── cv_kf/
    │   │   ├── scenario_000/
    │   │   │   ├── gt_traces.json
    │   │   │   ├── noisy_detections.json
    │   │   │   ├── tracker_data.json
    │   │   │   ├── scenario_plot.png
    │   │   │   └── video.mp4          (if --save-video)
    │   │   ├── scenario_001/
    │   │   ├── scenario_specs.json
    │   │   ├── scenario_results.json
    │   │   └── aggregate_summary.json
    │   └── sf_cv_ekf/
    │       └── ...
    ├── fuzzed/
    │   └── ...
    ├── batch_summary.json
    └── batch_metrics.txt

Example:
    python -m carla_integration.run_experiment_batch
    python -m carla_integration.run_experiment_batch --n-runs 5 --save-video
    python -m carla_integration.run_experiment_batch --verbose
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_TRACKERS = ["cv_kf", "sf_cv_ekf", "sf_cv_ekf_simplex"]
ALL_TRACKERS = ["cv_kf", "sf_ct_ekf", "sf_ct_ekf_simplex",
                "sf_cv_ekf", "sf_cv_ekf_simplex"]
DISTURBANCES = ["nominal", "fuzzed"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _generate_plots(combo_dir: Path) -> int:
    """Run plot_scenario.py --save for every scenario_* sub-directory."""
    count = 0
    for sc_dir in sorted(combo_dir.glob("scenario_*")):
        if not sc_dir.is_dir():
            continue
        gt = sc_dir / "gt_traces.json"
        tk = sc_dir / "tracker_data.json"
        if not gt.exists() or not tk.exists():
            continue
        cmd = [
            sys.executable, "-m", "carla_integration.plot_scenario",
            str(sc_dir), "--save",
        ]
        subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        count += 1
    return count


def _write_metrics_txt(summaries: dict, out_path: Path):
    """Write a human-readable comparison table."""
    lines = []
    lines.append(f"{'Config':<35s}  {'Coll%':>6s}  {'ADE':>6s}  {'FDE':>6s}  "
                 f"{'FP_brk':>6s}  {'rho_min':>7s}  {'n':>4s}")
    lines.append("-" * 82)
    for key, s in summaries.items():
        lines.append(
            f"{key:<35s}  {s.get('collision_rate_pct', 0):5.1f}%  "
            f"{s.get('mean_ade', 0):6.3f}  {s.get('mean_fde', 0):6.3f}  "
            f"{s.get('mean_fp_brakes', 0):6.1f}  "
            f"{s.get('min_rho_min', 0):7.3f}  "
            f"{s.get('n_scenarios', 0):4d}"
        )
    lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end CARLA Simplex-Track validation pipeline"
    )
    parser.add_argument("--n-runs", type=int, default=2,
                        help="Scenarios per (tracker, disturbance) combo (default: 2)")
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
    parser.add_argument("--save-video", action="store_true",
                        help="Record 4-camera composite video per scenario")
    parser.add_argument("--output-root", type=str,
                        default="runs/carla_scenarios",
                        help="Root output directory (timestamp appended)")
    parser.add_argument("--trackers", type=str, nargs="+",
                        default=DEFAULT_TRACKERS, choices=ALL_TRACKERS,
                        help="Tracker(s) to run (default: cv_kf sf_cv_ekf sf_cv_ekf_simplex)")
    parser.add_argument("--disturbances", type=str, nargs="+",
                        default=DISTURBANCES, choices=DISTURBANCES,
                        help="Disturbance(s) to run (default: nominal fuzzed)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show full per-step CARLA logging (quiet by default)")
    parser.add_argument("--skip-plots", action="store_true",
                        help="Skip automatic plot generation")
    args = parser.parse_args()

    # ── Pre-flight: verify CARLA is reachable and not stuck ─────────────
    logger.info("Checking CARLA connection at %s:%d ...", args.host, args.port)
    preflight_cmd = [
        sys.executable, "-c",
        "import carla, sys; "
        f"c = carla.Client('{args.host}', {args.port}); "
        "c.set_timeout(30.0); "
        "w = c.get_world(); "
        "s = w.get_settings(); "
        "s.synchronous_mode = False; s.fixed_delta_seconds = None; "
        "w.apply_settings(s); "
        "print('CARLA OK:', c.get_server_version())"
    ]
    pf = subprocess.run(preflight_cmd, cwd=str(PROJECT_ROOT),
                        capture_output=True, text=True, timeout=45)
    if pf.returncode != 0:
        logger.error("Cannot reach CARLA server. Is it running?")
        logger.error("stderr: %s", pf.stderr.strip())
        sys.exit(1)
    logger.info(pf.stdout.strip())

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_root = Path(args.output_root) / f"output_{timestamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    combos = [(t, d) for d in args.disturbances for t in args.trackers]
    total = len(combos)
    summaries = {}

    logger.info("=" * 70)
    logger.info("CARLA VALIDATION PIPELINE  |  %s", timestamp)
    logger.info("  output:       %s", run_root)
    logger.info("  trackers:     %s", args.trackers)
    logger.info("  disturbances: %s", args.disturbances)
    logger.info("  n_runs: %d  |  n_ped: %d  |  seed: %d  |  v_ego: %.0f km/h",
                args.n_runs, args.n_ped, args.seed, args.v_ego)
    logger.info("  combinations: %d  |  total scenarios: %d",
                total, total * args.n_runs)
    logger.info("=" * 70)

    t0 = time.time()
    failed_combos = []

    for idx, (tracker, disturbance) in enumerate(combos, 1):
        combo_dir = run_root / disturbance / tracker
        combo_key = f"{disturbance}/{tracker}"

        logger.info("[%d/%d]  %s  →  %s", idx, total, combo_key, combo_dir)

        cmd = [
            sys.executable, "-m", "carla_integration.run_scenarios",
            "--tracker", tracker,
            "--disturbance", disturbance,
            "--n-scenarios", str(args.n_runs),
            "--n-ped", str(args.n_ped),
            "--seed", str(args.seed),
            "--v-ego", str(args.v_ego),
            "--tau-safe", str(args.tau_safe),
            "--host", args.host,
            "--port", str(args.port),
            "--output-dir", str(combo_dir),
        ]
        if args.save_video:
            cmd.append("--save-video")
        if not args.verbose:
            cmd.append("--quiet")

        stdout_pipe = None if args.verbose else subprocess.DEVNULL
        result = subprocess.run(cmd, cwd=str(PROJECT_ROOT),
                                stdout=stdout_pipe,
                                stderr=None,
                                text=True)

        if result.returncode != 0:
            logger.error("        FAILED (exit code %d)", result.returncode)
            failed_combos.append(combo_key)
            continue

        # Read aggregate summary
        agg_path = combo_dir / "aggregate_summary.json"
        if agg_path.exists():
            with open(agg_path) as f:
                agg = json.load(f)
            summaries[combo_key] = agg
            logger.info("        Coll: %5.1f%%  ADE: %.3f  FDE: %.3f  FP: %.1f",
                        agg.get("collision_rate_pct", 0),
                        agg.get("mean_ade", 0),
                        agg.get("mean_fde", 0),
                        agg.get("mean_fp_brakes", 0))
        else:
            logger.warning("        No aggregate_summary.json found")

        # Generate plots
        if not args.skip_plots:
            n_plots = _generate_plots(combo_dir)
            logger.info("        Generated %d scenario plot(s)", n_plots)

    elapsed = time.time() - t0

    # ── Summary ───────────────────────────────────────────────────────────
    logger.info("")
    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE: %d/%d combos in %.0fs (%.1f s/combo)",
                total - len(failed_combos), total, elapsed,
                elapsed / max(total, 1))
    logger.info("=" * 70)

    if summaries:
        logger.info("")
        logger.info("%-35s  %6s  %6s  %6s  %6s  %7s",
                     "Config", "Coll%", "ADE", "FDE", "FP_brk", "rho_min")
        logger.info("-" * 82)
        for key, s in summaries.items():
            logger.info("%-35s  %5.1f%%  %6.3f  %6.3f  %6.1f  %7.3f",
                        key,
                        s.get("collision_rate_pct", 0),
                        s.get("mean_ade", 0),
                        s.get("mean_fde", 0),
                        s.get("mean_fp_brakes", 0),
                        s.get("min_rho_min", 0))

    if failed_combos:
        logger.error("\nFailed combinations: %s", failed_combos)

    # Save machine-readable summary
    batch_path = run_root / "batch_summary.json"
    with open(batch_path, "w") as f:
        json.dump({
            "timestamp": timestamp,
            "config": {
                "n_runs": args.n_runs,
                "n_ped": args.n_ped,
                "seed": args.seed,
                "v_ego_kmh": args.v_ego,
                "tau_safe": args.tau_safe,
                "trackers": args.trackers,
                "disturbances": args.disturbances,
            },
            "elapsed_s": round(elapsed, 1),
            "failed": failed_combos,
            "results": summaries,
        }, f, indent=2)

    # Save human-readable metrics
    metrics_path = run_root / "batch_metrics.txt"
    _write_metrics_txt(summaries, metrics_path)

    logger.info("")
    logger.info("Output:  %s", run_root)
    logger.info("Summary: %s", batch_path)
    logger.info("Metrics: %s", metrics_path)

    if failed_combos:
        sys.exit(1)


if __name__ == "__main__":
    main()
