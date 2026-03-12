#!/usr/bin/env python3
"""
Full experiment sweep for the Simplex-Track paper.

Orchestrates all validation methods and saves results as JSON.

Experiments:
  4a. Density Sweep        — Table I, Fig. failure_rate_density
  4b. Validation Pipeline  — Tables II-III, IS/Bayesian/failure Figs
  4c. Threshold Sweep      — Fig. roc_threshold
  4d. STL Robustness Trace — Fig. stl_robustness

Usage:
    python -m simplex_splat.run_full_experiments [--quick] [--resume]
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

from simplex_splat.simulation import (
    PedestrianParams,
    ScenarioConfig,
    SimulationResult,
    run_scenario,
    sample_pedestrians,
)
from simplex_splat.validation import (
    run_monte_carlo,
    run_cmaes,
    run_mcmc,
    run_importance_sampling,
    run_cross_entropy,
    run_bayesian,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "runs" / "simplex_splat" / "experiments"


def save_results(results: dict, path: Path):
    """Atomic-ish save of results dict to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f, indent=2, default=str)
    tmp.rename(path)


def load_results(path: Path) -> dict:
    """Load existing results for resume support."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# 4a. Density Sweep
# ═══════════════════════════════════════════════════════════════════════════════

def run_density_sweep(n_trials: int = 100, quick: bool = False) -> dict:
    """Sweep collision rates across trackers × densities.

    Uses the SAME scenarios (pedestrian params + simulation seeds) for all
    trackers at each density, so any difference in collision rate is purely
    from tracker behaviour, not scenario variance.
    """
    logger.info("═══ 4a: Density Sweep ═══")
    densities = [3, 5, 7, 10]
    trackers = ["cv", "sfekf", "cv_simplex", "sfekf_simplex"]
    n = n_trials if not quick else 20
    results = {}

    for n_ped in densities:
        # Pre-generate scenarios shared across all trackers
        scenario_rng = np.random.default_rng(n_ped * 1000)
        scenarios = [sample_pedestrians(n_ped, scenario_rng) for _ in range(n)]
        seeds = [int(scenario_rng.integers(0, 2**31)) for _ in range(n)]

        for tracker in trackers:
            key = f"{tracker}_{n_ped}"
            collisions = []
            ades = []
            fdes = []
            fp_brakes = []

            t0 = time.time()
            for i in range(n):
                cfg = ScenarioConfig(
                    n_ped=n_ped,
                    pedestrians=scenarios[i],
                    tracker=tracker,
                    seed=seeds[i],
                )
                result = run_scenario(cfg)
                collisions.append(result.collision)
                ades.append(result.ade)
                fdes.append(result.fde)
                fp_brakes.append(result.n_fp_brakes)

            elapsed = time.time() - t0
            coll_rate = 100.0 * np.mean(collisions)
            coll_std = 100.0 * np.std(collisions)

            results[key] = {
                "collision_rate": round(coll_rate, 1),
                "collision_std": round(coll_std, 1),
                "ade": round(float(np.mean(ades)), 2),
                "ade_std": round(float(np.std(ades)), 2),
                "fde": round(float(np.mean(fdes)), 2),
                "fde_std": round(float(np.std(fdes)), 2),
                "fp_brake_rate": round(float(np.mean(fp_brakes)), 1),
                "n_trials": n,
            }

            logger.info("  %s n_ped=%d: coll=%.1f±%.1f%%, ADE=%.2f, FDE=%.2f (%.1fs)",
                        tracker, n_ped, coll_rate, coll_std,
                        np.mean(ades), np.mean(fdes), elapsed)

    return results

# ═══════════════════════════════════════════════════════════════════════════════
# 4b. Validation Pipeline at N_ped=5
# ═══════════════════════════════════════════════════════════════════════════════

def run_validation_pipeline(n_ped: int = 5, quick: bool = False) -> dict:
    """Run all six validation methods sequentially."""
    logger.info("═══ 4b: Validation Pipeline (N_ped=%d) ═══", n_ped)
    results = {}

    # 1. Direct MC
    mc_n = 100 if quick else 500
    mc = run_monte_carlo(n_samples=mc_n, n_ped=n_ped, tracker="sfekf")
    results["mc"] = mc.to_dict()
    results["mc"]["results"] = mc.results  # keep for convergence analysis

    # CV baseline
    mc_cv = run_monte_carlo(n_samples=mc_n, n_ped=n_ped, tracker="cv", seed=3024)
    results["cv_mc"] = mc_cv.to_dict()

    # 2. CMA-ES
    cmaes_n = 50 if quick else 100
    cmaes = run_cmaes(n_ped=n_ped, tracker="sfekf", max_evals=cmaes_n)
    results["cmaes"] = cmaes.to_dict()

    # 3. MCMC (seeded from CMA-ES worst case)
    mcmc_n = 500 if quick else 2000
    mcmc_burn = 50 if quick else 200
    init_point = (cmaes.worst_d, cmaes.worst_theta, cmaes.worst_v)
    mcmc = run_mcmc(n_steps=mcmc_n, burn_in=mcmc_burn, n_ped=n_ped,
                    init_point=init_point)
    results["mcmc"] = mcmc.to_dict()
    # Save accepted samples for plotting
    results["mcmc_samples"] = [
        {"d": s[0], "theta": math.degrees(s[1]), "v": s[2]}
        for s in mcmc.accepted_samples
    ]

    # 4. IS (proposal from MCMC failure region)
    is_n = 100 if quick else 500
    if mcmc.accepted_samples:
        d_arr = np.array([s[0] for s in mcmc.accepted_samples])
        t_arr = np.array([s[1] for s in mcmc.accepted_samples])
        v_arr = np.array([s[2] for s in mcmc.accepted_samples])
        mu_q = np.array([np.mean(d_arr), np.mean(t_arr), np.mean(v_arr)])
        sigma_q = np.array([np.std(d_arr) + 0.5, np.std(t_arr) + 0.05,
                            np.std(v_arr) + 0.05])
    else:
        mu_q = np.array([10.0, math.radians(85.0), 1.5])
        sigma_q = np.array([4.0, math.radians(20.0), 0.3])

    is_result = run_importance_sampling(
        n_samples=is_n, n_ped=n_ped, mu_q=mu_q, sigma_q=sigma_q,
        mc_se=mc.se,
    )
    results["is"] = is_result.to_dict()

    # 5. Cross-Entropy
    ce_per = 50 if quick else 200
    ce = run_cross_entropy(n_per_iter=ce_per, n_ped=n_ped, mc_se=mc.se)
    results["ce"] = ce.to_dict()
    results["ce"]["iteration_history"] = ce.iteration_history

    # 6. Bayesian
    bayes = run_bayesian(mc.n_failures, mc.n_samples)
    results["bayesian"] = bayes.to_dict()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4c. Threshold Sweep
# ═══════════════════════════════════════════════════════════════════════════════

def run_threshold_sweep(n_trials: int = 100, quick: bool = False) -> dict:
    """Sweep τ_safe for collision rate and FP brake rate."""
    logger.info("═══ 4c: Threshold Sweep ═══")
    tau_values = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    trackers = ["cv", "sfekf"]
    n = n_trials if not quick else 20
    results = {}

    for base_tracker in trackers:
        tracker_results = []
        for tau_safe in tau_values:
            rng = np.random.default_rng(int(tau_safe * 1000) + hash(base_tracker) % 10000)
            collisions = []
            fp_brakes = []

            # Run with simplex at each threshold
            simplex_tracker = f"{base_tracker}_simplex"

            for _ in range(n):
                cfg = ScenarioConfig(
                    n_ped=5,
                    pedestrians=sample_pedestrians(5, rng),
                    tracker=simplex_tracker,
                    tau_safe=tau_safe,
                    seed=int(rng.integers(0, 2**31)),
                )
                result = run_scenario(cfg)
                collisions.append(result.collision)
                fp_brakes.append(result.n_fp_brakes > 0)

            tracker_results.append({
                "tau_safe": tau_safe,
                "collision_rate": round(100.0 * np.mean(collisions), 1),
                "fp_brake_rate": round(100.0 * np.mean(fp_brakes), 1),
            })

            logger.info("  %s τ_safe=%.1f: coll=%.1f%%, FP=%.1f%%",
                        base_tracker, tau_safe,
                        100.0 * np.mean(collisions),
                        100.0 * np.mean(fp_brakes))

        results[base_tracker] = tracker_results

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4d. STL Robustness Trace
# ═══════════════════════════════════════════════════════════════════════════════

def run_stl_trace(worst_case_params: dict = None, quick: bool = False) -> dict:
    """Run a representative scenario and record full TTC time-series."""
    logger.info("═══ 4d: STL Robustness Trace ═══")

    # Use CMA-ES worst case if available and dangerous enough
    if worst_case_params and worst_case_params.get("d_spawn", 99) < 15:
        d = worst_case_params.get("d_spawn", 6.0)
        theta = math.radians(worst_case_params.get("theta_approach", 85))
        v = worst_case_params.get("v_init", 1.2)
    else:
        # Fallback: known-dangerous scenario (small d_spawn, broadside crossing)
        d, theta, v = 6.0, math.radians(85.0), 1.2

    primary_ped = PedestrianParams(d_spawn=d, theta_approach=theta, v_init=v)
    rng = np.random.default_rng(9999)
    other_peds = sample_pedestrians(4, rng)
    peds = [primary_ped] + other_peds

    # Find a seed that produces a near-collision (min_ttc < tau_safe * 2)
    # so the TTC trace is interesting for the paper figure.
    best_seed, best_ttc = 42, float("inf")
    for s in range(200):
        cfg = ScenarioConfig(
            n_ped=5, pedestrians=peds, tracker="cv", T_max=20.0, seed=s,
        )
        r = run_scenario(cfg)
        if r.min_ttc < best_ttc:
            best_seed = s
            best_ttc = r.min_ttc
        if r.min_ttc < 2.0:
            break
    logger.info("  STL: using seed=%d (min_ttc=%.2f)", best_seed, best_ttc)

    results = {}

    for tracker in ["cv", "cv_simplex", "sfekf_simplex"]:
        cfg = ScenarioConfig(
            n_ped=5,
            pedestrians=peds,
            tracker=tracker,
            T_max=20.0,
            seed=best_seed,
        )
        result = run_scenario(cfg)
        results[tracker] = {
            "ttc_trace": result.ttc_trace,
            "collision": result.collision,
            "rho_min": result.rho_min,
            "min_ttc": result.min_ttc,
        }
        logger.info("  %s: collision=%s, ρ_min=%.2f, min_ttc=%.2f",
                    tracker, result.collision, result.rho_min, result.min_ttc)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4e. Ablation Study
# ═══════════════════════════════════════════════════════════════════════════════

def run_ablation(n_trials: int = 200, quick: bool = False) -> dict:
    """Ablation: full system, CV+Simplex, no-Simplex, CV-only."""
    logger.info("═══ Ablation Study ═══")
    n = n_trials if not quick else 30
    configs = {
        "full": "sfekf_simplex",
        "no_social": "cv_simplex",
        "no_simplex": "sfekf",
        "cv_only": "cv",
    }

    results = {}
    for name, tracker in configs.items():
        rng = np.random.default_rng(hash(name) % 2**31)
        collisions, ades, fdes, fp_brakes = [], [], [], []

        for _ in range(n):
            cfg = ScenarioConfig(
                n_ped=5,
                pedestrians=sample_pedestrians(5, rng),
                tracker=tracker,
                seed=int(rng.integers(0, 2**31)),
            )
            result = run_scenario(cfg)
            collisions.append(result.collision)
            ades.append(result.ade)
            fdes.append(result.fde)
            fp_brakes.append(result.n_fp_brakes)

        # FP brake rate: fraction of scenarios with at least one FP brake
        fp_rate = 0.0
        if tracker.endswith("_simplex"):
            fp_rate = round(100.0 * np.mean([1 for f in fp_brakes if f > 0]) / n, 1)

        results[name] = {
            "collision_rate": round(100.0 * np.mean(collisions), 1),
            "ade": round(float(np.mean(ades)), 2),
            "fde": round(float(np.mean(fdes)), 2),
            "fp_brake_rate": round(max(0, fp_rate), 1),
        }
        logger.info("  %s: coll=%.1f%%, ADE=%.2f, FDE=%.2f, FP=%.1f%%",
                    name, results[name]["collision_rate"],
                    results[name]["ade"], results[name]["fde"],
                    results[name]["fp_brake_rate"])

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="Run full Simplex-Track experiments")
    parser.add_argument("--quick", action="store_true",
                        help="Reduced sample sizes for quick testing")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing results")
    parser.add_argument("--output", type=str, default=str(OUTPUT_DIR / "full_results.json"))
    args = parser.parse_args()

    out_path = Path(args.output)
    results = load_results(out_path) if args.resume else {}

    t0 = time.time()
    print("=" * 72)
    print("SIMPLEX-TRACK FULL EXPERIMENT SWEEP")
    print("=" * 72)

    # 4a. Density sweep
    if "density" not in results:
        n_density = 100 if not args.quick else 20
        results["density"] = run_density_sweep(n_trials=n_density, quick=args.quick)
        save_results(results, out_path)

    # 4b. Validation pipeline
    if "validation" not in results:
        results["validation"] = run_validation_pipeline(n_ped=5, quick=args.quick)
        save_results(results, out_path)

    # 4c. Threshold sweep
    if "threshold" not in results:
        n_thresh = 100 if not args.quick else 20
        results["threshold"] = run_threshold_sweep(n_trials=n_thresh, quick=args.quick)
        save_results(results, out_path)

    # 4d. STL trace
    if "stl_trace" not in results:
        worst = results.get("validation", {}).get("cmaes", None)
        results["stl_trace"] = run_stl_trace(worst_case_params=worst, quick=args.quick)
        save_results(results, out_path)

    # Ablation
    if "ablation" not in results:
        n_abl = 200 if not args.quick else 30
        results["ablation"] = run_ablation(n_trials=n_abl, quick=args.quick)
        save_results(results, out_path)

    elapsed = time.time() - t0

    # Print summary
    print("\n" + "=" * 72)
    print(f"ALL EXPERIMENTS COMPLETE ({elapsed:.0f}s)")
    print("=" * 72)

    # Summary stats
    val = results.get("validation", {})
    mc = val.get("mc", {})
    print(f"\nMC: p_fail={mc.get('p_fail', '?')}, "
          f"CI=[{mc.get('ci_lo', '?')}, {mc.get('ci_hi', '?')}]")

    bayes = val.get("bayesian", {})
    print(f"Bayesian: Beta({bayes.get('alpha_post', '?')}, {bayes.get('beta_post', '?')}), "
          f"MAP={bayes.get('map_estimate', '?')}")

    is_r = val.get("is", {})
    print(f"IS: p_fail={is_r.get('p_fail_is', '?')}, VR={is_r.get('variance_reduction', '?')}x")

    dens = results.get("density", {})
    for t in ["cv", "sfekf", "cv_simplex", "sfekf_simplex"]:
        rates = [dens.get(f"{t}_{n}", {}).get("collision_rate", "?") for n in [3, 5, 7, 10]]
        print(f"Density {t}: {rates}")

    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
