#!/usr/bin/env python3
"""
Validation methods for Simplex-Track.

Implements the six validation methods from the paper:
  3a. Direct Monte Carlo           (Section IV-A)
  3b. CMA-ES Falsification         (Section IV-B)
  3c. MCMC Failure Distribution     (Section IV-C)
  3d. Importance Sampling           (Section V-B)
  3e. Cross-Entropy Method          (Section V-C)
  3f. Bayesian Estimation           (Section V-A)

All methods call the scenario runner in simulation.py as the black-box.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from simplex_splat.simulation import (
    PedestrianParams,
    ScenarioConfig,
    SimulationResult,
    run_scenario,
    sample_pedestrians,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# Result data classes
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class MCResult:
    n_samples: int = 0
    n_failures: int = 0
    p_fail: float = 0.0
    se: float = 0.0
    ci_lo: float = 0.0
    ci_hi: float = 0.0
    results: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "n_failures": self.n_failures,
            "p_fail": round(self.p_fail, 4),
            "se": round(self.se, 4),
            "ci_lo": round(self.ci_lo, 4),
            "ci_hi": round(self.ci_hi, 4),
        }


@dataclass
class CMAESResult:
    n_evals: int = 0
    rho_min: float = 0.0
    worst_d: float = 0.0
    worst_theta: float = 0.0
    worst_v: float = 0.0
    history: List[float] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_evals": self.n_evals,
            "rho_min": round(self.rho_min, 2),
            "d_spawn": round(self.worst_d, 1),
            "theta_approach": round(math.degrees(self.worst_theta)),
            "v_init": round(self.worst_v, 2),
        }


@dataclass
class MCMCResult:
    n_steps: int = 0
    burn_in: int = 0
    n_accepted: int = 0
    accepted_samples: List[Tuple[float, float, float]] = field(default_factory=list)
    critical_region_pct: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n_steps": self.n_steps,
            "burn_in": self.burn_in,
            "n_accepted": self.n_accepted,
            "critical_region_pct": round(self.critical_region_pct),
        }


@dataclass
class ISResult:
    n_samples: int = 0
    p_fail_is: float = 0.0
    se_is: float = 0.0
    ess: float = 0.0
    variance_reduction: float = 0.0

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "p_fail_is": round(self.p_fail_is, 4),
            "se_is": round(self.se_is, 4),
            "ess": round(self.ess),
            "variance_reduction": round(self.variance_reduction, 1),
        }


@dataclass
class CEResult:
    n_per_iter: int = 0
    rho_quantile: float = 0.0
    n_iters: int = 0
    p_fail_ce: float = 0.0
    se_ce: float = 0.0
    variance_reduction: float = 0.0
    iteration_history: List[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_per_iter": self.n_per_iter,
            "rho_quantile": self.rho_quantile,
            "n_iters": self.n_iters,
            "p_fail_ce": round(self.p_fail_ce, 4),
            "se_ce": round(self.se_ce, 4),
            "variance_reduction": round(self.variance_reduction, 1),
        }


@dataclass
class BayesianResult:
    alpha_prior: int = 5
    beta_prior: int = 60
    alpha_post: int = 0
    beta_post: int = 0
    map_estimate: float = 0.0
    ci_lo: float = 0.0
    ci_hi: float = 0.0

    def to_dict(self) -> dict:
        return {
            "alpha_prior": self.alpha_prior,
            "beta_prior": self.beta_prior,
            "alpha_post": self.alpha_post,
            "beta_post": self.beta_post,
            "map_estimate": round(self.map_estimate, 4),
            "ci_lo": round(self.ci_lo, 4),
            "ci_hi": round(self.ci_hi, 4),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: run a batch of scenarios with progress
# ═══════════════════════════════════════════════════════════════════════════════

def _run_batch(n: int, n_ped: int, tracker: str, rng: np.random.Generator,
               tau_safe: float = 2.0,
               pedestrians_list: Optional[List[List[PedestrianParams]]] = None,
               label: str = "") -> List[SimulationResult]:
    """Run n scenarios and return results."""
    results = []
    t0 = time.time()

    for i in range(n):
        peds = pedestrians_list[i] if pedestrians_list else sample_pedestrians(n_ped, rng)
        cfg = ScenarioConfig(
            n_ped=n_ped,
            pedestrians=peds,
            tracker=tracker,
            tau_safe=tau_safe,
            seed=int(rng.integers(0, 2**31)),
        )
        result = run_scenario(cfg)
        results.append(result)

        if (i + 1) % max(1, n // 10) == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            logger.info("  %s [%d/%d] (%.1f/s, ETA %.0fs)", label, i + 1, n, rate, eta)

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 3a. Direct Monte Carlo (Section IV-A)
# ═══════════════════════════════════════════════════════════════════════════════

def run_monte_carlo(n_samples: int = 500, n_ped: int = 5,
                    tracker: str = "sfekf", seed: int = 2024,
                    tau_safe: float = 2.0) -> MCResult:
    """Direct Monte Carlo failure rate estimation."""
    logger.info("Running Direct MC: N=%d, tracker=%s", n_samples, tracker)
    rng = np.random.default_rng(seed)

    results = _run_batch(n_samples, n_ped, tracker, rng,
                         tau_safe=tau_safe, label="MC")

    n_fail = sum(1 for r in results if r.collision)
    p_fail = n_fail / n_samples
    se = math.sqrt(p_fail * (1 - p_fail) / n_samples) if n_samples > 0 else 0
    ci_lo = max(0.0, p_fail - 1.96 * se)
    ci_hi = min(1.0, p_fail + 1.96 * se)

    mc = MCResult(
        n_samples=n_samples,
        n_failures=n_fail,
        p_fail=p_fail,
        se=se,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
        results=[{"collision": r.collision, "rho_min": r.rho_min,
                  "ade": r.ade, "fde": r.fde} for r in results],
    )

    logger.info("MC result: %d/%d failures, p_fail=%.4f, 95%% CI [%.4f, %.4f]",
                n_fail, n_samples, p_fail, ci_lo, ci_hi)
    return mc


# ═══════════════════════════════════════════════════════════════════════════════
# 3b. CMA-ES Falsification (Section IV-B)
# ═══════════════════════════════════════════════════════════════════════════════

def run_cmaes(n_ped: int = 5, tracker: str = "sfekf",
              max_evals: int = 100, seed: int = 4024,
              tau_safe: float = 2.0) -> CMAESResult:
    """CMA-ES optimisation to find worst-case scenario (minimise ρ_min)."""
    logger.info("Running CMA-ES falsification: max_evals=%d", max_evals)
    rng = np.random.default_rng(seed)

    try:
        import cma
        CMA_AVAILABLE = True
    except ImportError:
        CMA_AVAILABLE = False
        logger.warning("cma package not available, using random search fallback")

    best_rho = float("inf")
    best_params = (20.0, math.pi / 2, 1.25)
    history = []
    n_evals = 0

    def objective(x):
        """Evaluate one point: x = [d_spawn, theta, v_init]."""
        nonlocal best_rho, best_params, n_evals
        d = np.clip(x[0], 5, 35)
        theta = np.clip(x[1], 0, math.pi)
        v = np.clip(x[2], 0.5, 2.0)

        peds = [PedestrianParams(d_spawn=d, theta_approach=theta, v_init=v)]
        # Fill remaining pedestrians randomly
        for _ in range(n_ped - 1):
            peds.append(PedestrianParams(
                d_spawn=rng.uniform(5, 35),
                theta_approach=rng.uniform(0, math.pi),
                v_init=rng.uniform(0.5, 2.0),
            ))

        cfg = ScenarioConfig(
            n_ped=n_ped, pedestrians=peds, tracker=tracker,
            tau_safe=tau_safe, seed=int(rng.integers(0, 2**31)),
        )
        result = run_scenario(cfg)
        n_evals += 1

        rho = result.rho_min
        history.append(rho)

        if rho < best_rho:
            best_rho = rho
            best_params = (d, theta, v)

        return rho

    if CMA_AVAILABLE:
        x0 = [20.0, math.pi / 2, 1.25]
        sigma0 = 5.0
        bounds = [[5, 0, 0.5], [35, math.pi, 2.0]]
        opts = {
            "maxfevals": max_evals,
            "bounds": bounds,
            "seed": seed,
            "verbose": -1,
        }
        es = cma.CMAEvolutionStrategy(x0, sigma0, opts)
        while not es.stop() and n_evals < max_evals:
            solutions = es.ask()
            es.tell(solutions, [objective(s) for s in solutions])
        es.result_pretty()
    else:
        # Random search fallback
        pop_size = 20
        for gen in range(max_evals // pop_size + 1):
            if n_evals >= max_evals:
                break
            for _ in range(pop_size):
                if n_evals >= max_evals:
                    break
                d = rng.uniform(5, 35)
                theta = rng.uniform(0, math.pi)
                v = rng.uniform(0.5, 2.0)
                objective([d, theta, v])

        # Local refinement around best
        for _ in range(min(30, max_evals - n_evals)):
            d = np.clip(rng.normal(best_params[0], 2.0), 5, 35)
            theta = np.clip(rng.normal(best_params[1], 0.2), 0, math.pi)
            v = np.clip(rng.normal(best_params[2], 0.2), 0.5, 2.0)
            objective([d, theta, v])

    result = CMAESResult(
        n_evals=n_evals,
        rho_min=best_rho,
        worst_d=best_params[0],
        worst_theta=best_params[1],
        worst_v=best_params[2],
        history=history,
    )

    logger.info("CMA-ES: ρ_min=%.2f after %d evals (d=%.1f, θ=%.0f°, v=%.2f)",
                best_rho, n_evals, best_params[0],
                math.degrees(best_params[1]), best_params[2])
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# 3c. MCMC Failure Distribution (Section IV-C)
# ═══════════════════════════════════════════════════════════════════════════════

def run_mcmc(n_steps: int = 2000, burn_in: int = 200,
             n_ped: int = 5, tracker: str = "sfekf",
             init_point: Optional[Tuple[float, float, float]] = None,
             seed: int = 5024, tau_safe: float = 2.0) -> MCMCResult:
    """Metropolis-Hastings targeting p_fail(τ) ∝ p(τ) · F(τ)."""
    logger.info("Running MCMC: %d steps, burn-in=%d", n_steps, burn_in)
    rng = np.random.default_rng(seed)

    # Proposal std (Σ_q = diag(2², 10², 0.2²)) — theta in degrees for proposal
    sigma_d, sigma_theta, sigma_v = 2.0, math.radians(10.0), 0.2

    # Initialise from CMA-ES worst-case or default
    if init_point:
        d, theta, v = init_point
    else:
        d, theta, v = 10.0, math.radians(85.0), 1.5

    accepted: List[Tuple[float, float, float]] = []
    total_accepted = 0
    t0 = time.time()

    for step in range(n_steps):
        # Propose
        d_p = np.clip(d + rng.normal(0, sigma_d), 5, 35)
        theta_p = np.clip(theta + rng.normal(0, sigma_theta), 0, math.pi)
        v_p = np.clip(v + rng.normal(0, sigma_v), 0.5, 2.0)

        # Evaluate: run scenario, check failure
        peds = [PedestrianParams(d_spawn=d_p, theta_approach=theta_p, v_init=v_p)]
        for _ in range(n_ped - 1):
            peds.append(PedestrianParams(
                d_spawn=rng.uniform(5, 35),
                theta_approach=rng.uniform(0, math.pi),
                v_init=rng.uniform(0.5, 2.0),
            ))

        cfg = ScenarioConfig(
            n_ped=n_ped, pedestrians=peds, tracker=tracker,
            tau_safe=tau_safe, seed=int(rng.integers(0, 2**31)),
        )
        result = run_scenario(cfg)

        # Accept if failure (target distribution = failure indicator × prior)
        if result.collision:
            d, theta, v = d_p, theta_p, v_p
            total_accepted += 1
            if step >= burn_in:
                accepted.append((d, theta, v))
        else:
            # Accept with small probability for exploration near boundary
            if result.rho_min < 0.5 and rng.random() < 0.1:
                d, theta, v = d_p, theta_p, v_p
                total_accepted += 1
                if step >= burn_in:
                    accepted.append((d, theta, v))

        if (step + 1) % max(1, n_steps // 10) == 0:
            elapsed = time.time() - t0
            logger.info("  MCMC [%d/%d] accepted=%d (%.1f/s)",
                        step + 1, n_steps, len(accepted), (step + 1) / elapsed)

    # Analyse failure distribution
    critical_pct = 0.0
    if accepted:
        critical = sum(1 for a in accepted
                       if a[0] < 14 and math.radians(55) < a[1] < math.radians(110))
        critical_pct = 100.0 * critical / len(accepted)

    result_obj = MCMCResult(
        n_steps=n_steps,
        burn_in=burn_in,
        n_accepted=len(accepted),
        accepted_samples=accepted,
        critical_region_pct=critical_pct,
    )

    logger.info("MCMC: %d accepted samples, %.0f%% in critical region",
                len(accepted), critical_pct)
    return result_obj


# ═══════════════════════════════════════════════════════════════════════════════
# 3d. Importance Sampling (Section V-B)
# ═══════════════════════════════════════════════════════════════════════════════

def run_importance_sampling(n_samples: int = 500, n_ped: int = 5,
                            tracker: str = "sfekf",
                            mu_q: Optional[np.ndarray] = None,
                            sigma_q: Optional[np.ndarray] = None,
                            mc_se: Optional[float] = None,
                            seed: int = 7024,
                            tau_safe: float = 2.0) -> ISResult:
    """IS with Gaussian proposal, using analytical compound collision probability.

    Instead of running binary simulations, we compute the analytical compound
    collision probability for each sample (Rao-Blackwellisation).  Background
    pedestrians are marginalised analytically, so the IS operates in 3-D and
    the only variance source is the IS weights.
    """
    from simplex_splat.simulation import _per_ped_base, _ped_risk_score, _E_RISK

    logger.info("Running Importance Sampling: N=%d", n_samples)
    rng = np.random.default_rng(seed)

    if mu_q is None:
        mu_q = np.array([10.0, math.radians(85.0), 1.5])
    if sigma_q is None:
        sigma_q = np.array([4.0, math.radians(20.0), 0.3])

    # Nominal distribution: uniform over parameter ranges
    p_nom = 1.0 / (30.0 * math.pi * 1.5)

    # Per-ped collision base rate and background survival
    p_base = _per_ped_base(n_ped, tracker)
    bg_surv = (1.0 - p_base) ** max(n_ped - 1, 0)

    weights = []
    failures_weighted = []
    t0 = time.time()

    for i in range(n_samples):
        d = float(np.clip(rng.normal(mu_q[0], sigma_q[0]), 5, 35))
        theta = float(np.clip(rng.normal(mu_q[1], sigma_q[1]), 0, math.pi))
        v = float(np.clip(rng.normal(mu_q[2], sigma_q[2]), 0.5, 2.0))

        # Analytical compound collision probability (marginalised background)
        risk = _ped_risk_score(d, theta, v)
        p_0 = float(np.clip(p_base * risk / _E_RISK, 0, 0.95))
        p_scenario = 1.0 - (1.0 - p_0) * bg_surv

        # Likelihood ratio for the primary pedestrian
        q_prop = (stats.norm.pdf(d, mu_q[0], sigma_q[0]) *
                  stats.norm.pdf(theta, mu_q[1], sigma_q[1]) *
                  stats.norm.pdf(v, mu_q[2], sigma_q[2]))
        w = p_nom / max(q_prop, 1e-30)

        weights.append(w)
        failures_weighted.append(w * p_scenario)

        if (i + 1) % max(1, n_samples // 10) == 0:
            elapsed = time.time() - t0
            logger.info("  IS [%d/%d] (%.1f/s)", i + 1, n_samples, (i + 1) / elapsed)

    w_arr = np.array(weights)
    fw_arr = np.array(failures_weighted)

    p_fail_is = float(np.sum(fw_arr) / np.sum(w_arr))
    ess = float(np.sum(w_arr) ** 2 / np.sum(w_arr ** 2))
    se_is = float(np.std(fw_arr / np.mean(w_arr)) / math.sqrt(n_samples))

    vr = (mc_se / se_is) ** 2 if (mc_se and se_is > 1e-10) else 1.0

    is_result = ISResult(
        n_samples=n_samples,
        p_fail_is=p_fail_is,
        se_is=se_is,
        ess=ess,
        variance_reduction=vr,
    )

    logger.info("IS: p_fail=%.4f, SE=%.4f, ESS=%.0f, VR=%.1fx",
                p_fail_is, se_is, ess, vr)
    return is_result


# ═══════════════════════════════════════════════════════════════════════════════
# 3e. Cross-Entropy Method (Section V-C)
# ═══════════════════════════════════════════════════════════════════════════════

def run_cross_entropy(n_per_iter: int = 200, rho_quantile: float = 0.1,
                      max_iters: int = 15, n_ped: int = 5,
                      tracker: str = "sfekf", seed: int = 8024,
                      mc_se: Optional[float] = None,
                      tau_safe: float = 2.0) -> CEResult:
    """Cross-entropy method for failure probability estimation.

    Uses the analytical compound collision probability for elite selection
    (no Bernoulli noise), then applies IS with the converged proposal.
    """
    from simplex_splat.simulation import _per_ped_base, _ped_risk_score, _E_RISK

    logger.info("Running Cross-Entropy: %d/iter, ρ=%.2f, max %d iters",
                n_per_iter, rho_quantile, max_iters)
    rng = np.random.default_rng(seed)

    p_base = _per_ped_base(n_ped, tracker)
    bg_surv = (1.0 - p_base) ** max(n_ped - 1, 0)

    # Start from nominal distribution (broad)
    mu = np.array([20.0, math.pi / 2, 1.25])
    sigma = np.array([8.0, math.pi / 4, 0.5])

    iter_history = []

    for iteration in range(max_iters):
        samples = []
        p_scenarios = []

        for _ in range(n_per_iter):
            d = float(np.clip(rng.normal(mu[0], sigma[0]), 5, 35))
            theta = float(np.clip(rng.normal(mu[1], sigma[1]), 0, math.pi))
            v = float(np.clip(rng.normal(mu[2], sigma[2]), 0.5, 2.0))

            risk = _ped_risk_score(d, theta, v)
            p_0 = float(np.clip(p_base * risk / _E_RISK, 0, 0.95))
            p_scenario = 1.0 - (1.0 - p_0) * bg_surv

            samples.append([d, theta, v])
            p_scenarios.append(p_scenario)

        samples_arr = np.array(samples)
        p_arr = np.array(p_scenarios)

        # Select elite: highest collision probability
        threshold = np.percentile(p_arr, (1.0 - rho_quantile) * 100)
        elite_mask = p_arr >= threshold
        elite = samples_arr[elite_mask]

        n_dangerous = int(np.sum(p_arr > 0.5))
        iter_history.append({
            "iteration": iteration + 1,
            "mu": mu.tolist(),
            "sigma": sigma.tolist(),
            "threshold": float(threshold),
            "n_failures": n_dangerous,
        })

        logger.info("  CE iter %d: p_threshold=%.3f, %d elite, %d dangerous",
                     iteration + 1, threshold, len(elite), n_dangerous)

        if len(elite) < 3:
            continue

        mu_new = np.mean(elite, axis=0)
        sigma_new = np.maximum(np.std(elite, axis=0) + 1.0,
                               np.array([6.0, math.pi / 5, 0.30]))

        # Convergence check (require at least 3 iterations)
        if iteration >= 2 and np.all(np.abs(mu_new - mu) < 0.1):
            mu = mu_new
            sigma = sigma_new
            break

        mu = mu_new
        sigma = sigma_new

    # Final IS estimate using converged proposal (analytical model)
    n_final = 500
    p_nom = 1.0 / (30.0 * math.pi * 1.5)
    w_list: List[float] = []
    fw_list: List[float] = []

    for _ in range(n_final):
        d = float(np.clip(rng.normal(mu[0], sigma[0]), 5, 35))
        theta = float(np.clip(rng.normal(mu[1], sigma[1]), 0, math.pi))
        v = float(np.clip(rng.normal(mu[2], sigma[2]), 0.5, 2.0))

        risk = _ped_risk_score(d, theta, v)
        p_0 = float(np.clip(p_base * risk / _E_RISK, 0, 0.95))
        p_scenario = 1.0 - (1.0 - p_0) * bg_surv

        q_prop = (stats.norm.pdf(d, mu[0], sigma[0]) *
                  stats.norm.pdf(theta, mu[1], sigma[1]) *
                  stats.norm.pdf(v, mu[2], sigma[2]))
        w = p_nom / max(q_prop, 1e-30)
        w_list.append(w)
        fw_list.append(w * p_scenario)

    w_arr = np.array(w_list)
    fw_arr = np.array(fw_list)
    p_fail = float(np.sum(fw_arr) / np.sum(w_arr))
    se_ce = float(np.std(fw_arr / np.mean(w_arr)) / math.sqrt(n_final))
    vr = (mc_se / se_ce) ** 2 if (mc_se and se_ce > 1e-10) else 1.0

    ce_result = CEResult(
        n_per_iter=n_per_iter,
        rho_quantile=rho_quantile,
        n_iters=len(iter_history),
        p_fail_ce=p_fail,
        se_ce=se_ce,
        variance_reduction=vr,
        iteration_history=iter_history,
    )

    logger.info("CE: p_fail=%.4f, SE=%.4f, %d iters, VR=%.1fx",
                p_fail, se_ce, len(iter_history), vr)
    return ce_result


# ═══════════════════════════════════════════════════════════════════════════════
# 3f. Bayesian Estimation (Section V-A)
# ═══════════════════════════════════════════════════════════════════════════════

def run_bayesian(n_failures: int, n_trials: int,
                 alpha_prior: int = 5, beta_prior: int = 60) -> BayesianResult:
    """Beta-Binomial Bayesian posterior update."""
    logger.info("Running Bayesian: Beta(%d,%d) + %d/%d",
                alpha_prior, beta_prior, n_failures, n_trials)

    alpha_post = alpha_prior + n_failures
    beta_post = beta_prior + (n_trials - n_failures)

    dist = stats.beta(alpha_post, beta_post)
    ci_lo, ci_hi = dist.ppf(0.025), dist.ppf(0.975)
    mode = (alpha_post - 1) / (alpha_post + beta_post - 2) if alpha_post > 1 else 0.0

    result = BayesianResult(
        alpha_prior=alpha_prior,
        beta_prior=beta_prior,
        alpha_post=alpha_post,
        beta_post=beta_post,
        map_estimate=mode,
        ci_lo=ci_lo,
        ci_hi=ci_hi,
    )

    logger.info("Bayesian: Beta(%d, %d), MAP=%.4f, 95%% CI [%.4f, %.4f]",
                alpha_post, beta_post, mode, ci_lo, ci_hi)
    return result
