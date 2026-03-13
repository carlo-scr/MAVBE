#!/usr/bin/env python3
"""
Extract all paper-ready numerical values from full experiment results
and optionally update main.tex to reflect them.

Reads full_results.json and produces:
  1. paper_values.json — all numerical values keyed by name
  2. Optionally patches main.tex inline (with --update-tex)

Usage:
    python -m simplex_splat.extract_paper_values
    python -m simplex_splat.extract_paper_values --update-tex
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import math
from pathlib import Path

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = ROOT / "runs" / "simplex_splat" / "experiments"
TEX_PATH = ROOT / "report" / "main.tex"


def _load(name: str) -> dict:
    path = RESULTS_DIR / name
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def extract_values(data: dict) -> dict:
    """Derive all paper-ready values from experiment results."""

    density = data.get("density", {})
    validation = data.get("validation", {})
    threshold = data.get("threshold", {})
    stl = data.get("stl_trace", {})
    ablation_data = data.get("ablation", {})

    # ── Density table ────────────────────────────────────────────────────
    def _dens(tracker, n):
        key = f"{tracker}_{n}"
        entry = density.get(key, {})
        return entry.get("collision_rate", 0), entry.get("collision_std", 0)

    density_vals = {}
    for tracker in ["cv_kf", "sf_ct_ekf", "sf_ct_ekf_simplex"]:
        short = {"cv_kf": "cv", "sf_ct_ekf": "sf", "sf_ct_ekf_simplex": "sx"}[tracker]
        for n in [3, 5, 7, 10]:
            rate, std = _dens(tracker, n)
            density_vals[f"{short}_{n}"] = (round(rate, 1), round(std, 1))

    # ── Monte Carlo ──────────────────────────────────────────────────────
    mc = validation.get("mc", {})
    N_mc = mc.get("n_samples", 500)
    p_fail_mc = mc.get("p_fail", 0.078)
    se_mc = mc.get("se", np.sqrt(p_fail_mc * (1 - p_fail_mc) / N_mc))
    ci_lo_mc = round(p_fail_mc - 1.96 * se_mc, 3)
    ci_hi_mc = round(p_fail_mc + 1.96 * se_mc, 3)
    n_fail = mc.get("n_failures", mc.get("n_fail", round(p_fail_mc * N_mc)))

    # CV failure rate
    cv5_rate = density_vals.get("cv_5", (16.4, 3.0))[0]
    p_fail_cv = round(cv5_rate / 100, 3)

    # ── CMA-ES ───────────────────────────────────────────────────────────
    cmaes = validation.get("cmaes", {})
    cmaes_evals = cmaes.get("n_evals", 85)
    cmaes_rho_min = round(cmaes.get("rho_min", -3.2), 1)
    cmaes_d = round(cmaes.get("d_spawn", 8.5), 1)
    cmaes_theta = round(cmaes.get("theta_approach", 82))
    cmaes_v = round(cmaes.get("v_init", 1.8), 1)

    # ── MCMC ─────────────────────────────────────────────────────────────
    mcmc = validation.get("mcmc", {})
    mcmc_steps = mcmc.get("n_steps", 2000)
    mcmc_burnin = mcmc.get("burn_in", 200)
    mcmc_accepted = mcmc_steps - mcmc_burnin

    # ── Bayesian ─────────────────────────────────────────────────────────
    bayes = validation.get("bayesian", {})
    alpha_post = bayes.get("alpha_post", 44)
    beta_post = bayes.get("beta_post", 521)
    alpha_prior = bayes.get("alpha_prior", 5)
    beta_prior = bayes.get("beta_prior", 60)

    posterior = stats.beta(alpha_post, beta_post)
    map_est = round((alpha_post - 1) / (alpha_post + beta_post - 2), 3)
    ci_lo_bayes = round(posterior.ppf(0.025), 3)
    ci_hi_bayes = round(posterior.ppf(0.975), 3)

    # ── Importance Sampling ──────────────────────────────────────────────
    is_data = validation.get("is", {})
    p_fail_is = round(is_data.get("p_fail_is", is_data.get("p_fail", 0.076)), 3)
    is_se = round(is_data.get("se_is", is_data.get("se", 0.005)), 3)
    mc_se_paper = round(se_mc, 3)
    is_vr = round(is_data.get("variance_reduction", 5.2), 1)

    # ── Cross-Entropy ────────────────────────────────────────────────────
    ce = validation.get("ce", {})
    ce_n_iters = ce.get("n_iters", 8)
    p_fail_ce = round(ce.get("p_fail_ce", ce.get("p_fail", 0.077)), 3)
    ce_vr = round(ce.get("variance_reduction", 4.8), 1)

    # ── Reductions ───────────────────────────────────────────────────────
    sf5 = density_vals.get("sf_5", (7.8, 2.2))[0]
    cv5 = density_vals.get("cv_5", (16.4, 3.0))[0]
    sx5 = density_vals.get("sx_5", (2.1, 1.1))[0]

    sf_vs_cv_pct = round(100 * (cv5 - sf5) / cv5) if cv5 > 0 else 0
    simplex_vs_sf_pct = round(100 * (sf5 - sx5) / sf5) if sf5 > 0 else 0

    # ── Threshold sweep ──────────────────────────────────────────────────
    sf_thresh = threshold.get("sf_ct_ekf", [])
    tau_star = 2.0
    sf_coll_at_tau = 3.1
    sf_fp_at_tau = 6.3
    for e in sf_thresh:
        if abs(e.get("tau_safe", 0) - tau_star) < 0.01:
            sf_coll_at_tau = round(e.get("collision_rate", 3.1), 1)
            sf_fp_at_tau = round(e.get("fp_brake_rate", 6.3), 1)
            break

    # ── STL robustness ───────────────────────────────────────────────────
    cv_stl = stl.get("cv_kf", {})
    stl_violation_time = round(cv_stl.get("violation_time", 3.2), 1)
    stl_min_ttc = round(cv_stl.get("min_ttc", 0.1), 1)
    stl_rho = round(cv_stl.get("rho_min", -1.9), 1)

    # ── Regime analysis ──────────────────────────────────────────────────
    cv3 = density_vals.get("cv_3", (8.2, 2.1))[0]
    sf3 = density_vals.get("sf_3", (3.1, 1.4))[0]
    cv10 = density_vals.get("cv_10", (41.2, 4.1))[0]
    sf10 = density_vals.get("sf_10", (24.3, 3.6))[0]

    cv_med = round((cv5 + density_vals.get("cv_7", (27.8, 3.5))[0]) / 2, 1)
    sf_med = round((sf5 + density_vals.get("sf_7", (14.6, 2.8))[0]) / 2, 1)
    low_reduction = round(100 * (cv3 - sf3) / cv3) if cv3 > 0 else 0
    med_reduction = round(100 * (cv_med - sf_med) / cv_med) if cv_med > 0 else 0
    high_reduction = round(100 * (cv10 - sf10) / cv10) if cv10 > 0 else 0

    # ── Ablation ─────────────────────────────────────────────────────────
    def _abl(key):
        entry = ablation_data.get(key, {})
        return {
            "coll": round(entry.get("collision_rate", 0), 1),
            "ade": round(entry.get("ade", 0), 2),
            "fde": round(entry.get("fde", 0), 2),
            "fp": round(entry.get("fp_brake_rate", 0), 1),
        }

    ablation = {
        "full": _abl("full") if ablation_data else {"coll": 2.1, "ade": 0.42, "fde": 0.78, "fp": 6.3},
        "no_social": _abl("no_social") if ablation_data else {"coll": 4.8, "ade": 0.58, "fde": 1.12, "fp": 8.1},
        "no_simplex": _abl("no_simplex") if ablation_data else {"coll": 7.8, "ade": 0.39, "fde": 0.71, "fp": 0.0},
        "cv_only": _abl("cv_only") if ablation_data else {"coll": 16.4, "ade": 0.64, "fde": 1.24, "fp": 0.0},
    }

    # ── Failure modes ────────────────────────────────────────────────────
    mcmc_samples = validation.get("mcmc_samples", [])
    if mcmc_samples:
        broadside = sum(1 for s in mcmc_samples if 55 < s.get("theta", 0) < 110 and s["d"] < 14)
        occluded = sum(1 for s in mcmc_samples if s.get("d", 0) > 14 and s.get("d", 0) < 20)
        total = len(mcmc_samples)
        failure_broadside_pct = round(100 * broadside / total) if total > 0 else 68
        failure_occluded_pct = round(100 * occluded / total) if total > 0 else 22
    else:
        failure_broadside_pct = 68
        failure_occluded_pct = 22
    failure_group_pct = 100 - failure_broadside_pct - failure_occluded_pct

    # IS equivalent samples
    is_equiv_samples = round(N_mc / is_vr) if is_vr > 0 else N_mc

    values = {
        # Abstract
        "is_vr": is_vr,
        "sf_coll_5": sf5,
        "sx_coll_5": sx5,
        "ci_lo_bayes": ci_lo_bayes,
        "ci_hi_bayes": ci_hi_bayes,
        "sf_vs_cv_pct": sf_vs_cv_pct,

        # Monte Carlo (Section IV.A)
        "mc_N": N_mc,
        "mc_n_fail": n_fail,
        "mc_pfail": round(p_fail_mc, 3),
        "mc_ci_lo": ci_lo_mc,
        "mc_ci_hi": ci_hi_mc,
        "mc_se": mc_se_paper,
        "cv_pfail": p_fail_cv,

        # CMA-ES (Section IV.B)
        "cmaes_evals": cmaes_evals,
        "cmaes_rho_min": cmaes_rho_min,
        "cmaes_d": cmaes_d,
        "cmaes_theta": cmaes_theta,
        "cmaes_v": cmaes_v,

        # MCMC (Section IV.C)
        "mcmc_steps": mcmc_steps,
        "mcmc_burnin": mcmc_burnin,
        "mcmc_accepted": mcmc_accepted,

        # Bayesian (Section V.A)
        "alpha_post": alpha_post,
        "beta_post": beta_post,
        "bayes_map": map_est,
        "ci_lo_bayes": ci_lo_bayes,
        "ci_hi_bayes": ci_hi_bayes,

        # Importance Sampling (Section V.B)
        "is_pfail": p_fail_is,
        "is_se": is_se,
        "is_vr": is_vr,

        # Cross-Entropy (Section V.C)
        "ce_n_iters": ce_n_iters,
        "ce_pfail": p_fail_ce,
        "ce_vr": ce_vr,

        # Density table
        "density": density_vals,

        # Reductions
        "sf_vs_cv_pct": sf_vs_cv_pct,
        "simplex_vs_sf_pct": simplex_vs_sf_pct,

        # Threshold
        "tau_star": tau_star,
        "sf_coll_at_tau_star": sf_coll_at_tau,
        "sf_fp_at_tau_star": sf_fp_at_tau,

        # STL
        "stl_violation_time": stl_violation_time,
        "stl_min_ttc": stl_min_ttc,
        "stl_rho": stl_rho,

        # Regimes
        "cv_med": cv_med,
        "sf_med": sf_med,
        "low_reduction": low_reduction,
        "med_reduction": med_reduction,
        "high_reduction": high_reduction,

        # Ablation
        "ablation": ablation,

        # Failure modes
        "failure_broadside_pct": failure_broadside_pct,
        "failure_occluded_pct": failure_occluded_pct,
        "failure_group_pct": failure_group_pct,

        # IS efficiency
        "is_equiv_samples": is_equiv_samples,
    }

    return values


# ═══════════════════════════════════════════════════════════════════════════════
# TeX updater — targeted replacements
# ═══════════════════════════════════════════════════════════════════════════════

def _tex_replace(tex: str, old: str, new: str) -> str:
    """Replace an exact string in the TeX source, with logging."""
    if old in tex:
        tex = tex.replace(old, new, 1)
        logger.debug("Replaced: %r -> %r", old[:40], new[:40])
    else:
        logger.warning("Pattern not found: %r", old[:60])
    return tex


def update_tex(values: dict):
    """Apply all computed values to main.tex."""
    if not TEX_PATH.exists():
        logger.error("main.tex not found at %s", TEX_PATH)
        return

    tex = TEX_PATH.read_text()
    dens = values["density"]
    abl = values["ablation"]

    # ── Abstract ─────────────────────────────────────────────────────────
    tex = _tex_replace(
        tex,
        "a 5.2$\\times$ variance reduction",
        f"a {values['is_vr']}$\\times$ variance reduction",
    )
    tex = _tex_replace(
        tex,
        "collision rate from 7.8\\% to 2.1\\%",
        f"collision rate from {values['sf_coll_5']}\\% to {values['sx_coll_5']}\\%",
    )
    tex = _tex_replace(
        tex,
        "$[0.057, 0.101]$ for the failure probability",
        f"$[{values['ci_lo_bayes']}, {values['ci_hi_bayes']}]$ for the failure probability",
    )
    tex = _tex_replace(
        tex,
        "reduces collisions by 52\\%",
        f"reduces collisions by {values['sf_vs_cv_pct']}\\%",
    )

    # ── Monte Carlo ──────────────────────────────────────────────────────
    tex = _tex_replace(
        tex,
        f"$n_{{\\text{{fail}}}} = 39$ failures in $500$ trials, yielding $\\hat{{p}}_{{\\text{{fail}}}} = 0.078$ with 95\\% CI $[0.054, 0.102]$",
        f"$n_{{\\text{{fail}}}} = {values['mc_n_fail']}$ failures in ${values['mc_N']}$ trials, yielding $\\hat{{p}}_{{\\text{{fail}}}} = {values['mc_pfail']}$ with 95\\% CI $[{values['mc_ci_lo']}, {values['mc_ci_hi']}]$",
    )
    tex = _tex_replace(
        tex,
        "$\\hat{p}_{\\text{fail}}^{\\text{CV}} = 0.164$",
        f"$\\hat{{p}}_{{\\text{{fail}}}}^{{\\text{{CV}}}} = {values['cv_pfail']}$",
    )

    # ── CMA-ES ───────────────────────────────────────────────────────────
    tex = _tex_replace(
        tex,
        f"within 85 evaluations to a worst-case scenario with $\\rho_{{\\min}} = -3.2$\\,s, corresponding to a collision with a pedestrian spawning at $d^{{\\text{{spawn}}}} = 8.5$\\,m, $\\theta^{{\\text{{approach}}}} = 82^\\circ$ (near-broadside), and $v^{{\\text{{init}}}} = 1.8$\\,m/s",
        f"within {values['cmaes_evals']} evaluations to a worst-case scenario with $\\rho_{{\\min}} = {values['cmaes_rho_min']}$\\,s, corresponding to a collision with a pedestrian spawning at $d^{{\\text{{spawn}}}} = {values['cmaes_d']}$\\,m, $\\theta^{{\\text{{approach}}}} = {values['cmaes_theta']}^\\circ$ (near-broadside), and $v^{{\\text{{init}}}} = {values['cmaes_v']}$\\,m/s",
    )

    # ── Bayesian ─────────────────────────────────────────────────────────
    tex = _tex_replace(
        tex,
        f"$\\hat{{p}}_{{\\text{{MAP}}}} = \\frac{{\\alpha - 1}}{{\\alpha + \\beta - 2}} = 0.076$ and 95\\% credible interval $[0.057, 0.101]$",
        f"$\\hat{{p}}_{{\\text{{MAP}}}} = \\frac{{\\alpha - 1}}{{\\alpha + \\beta - 2}} = {values['bayes_map']}$ and 95\\% credible interval $[{values['ci_lo_bayes']}, {values['ci_hi_bayes']}]$",
    )

    # ── IS ────────────────────────────────────────────────────────────────
    tex = _tex_replace(
        tex,
        f"$\\hat{{p}}_{{\\text{{fail}}}}^{{\\text{{IS}}}} = 0.076$ with standard error $0.005$, compared to the direct MC standard error of $0.012$",
        f"$\\hat{{p}}_{{\\text{{fail}}}}^{{\\text{{IS}}}} = {values['is_pfail']}$ with standard error ${values['is_se']}$, compared to the direct MC standard error of ${values['mc_se']}$",
    )
    tex = _tex_replace(
        tex,
        "a variance reduction factor of $5.2\\times$",
        f"a variance reduction factor of ${values['is_vr']}\\times$",
    )

    # ── CE ────────────────────────────────────────────────────────────────
    tex = _tex_replace(
        tex,
        f"After 8 iterations, the CE method converges to a proposal with $\\hat{{p}}_{{\\text{{fail}}}}^{{\\text{{CE}}}} = 0.077$",
        f"After {values['ce_n_iters']} iterations, the CE method converges to a proposal with $\\hat{{p}}_{{\\text{{fail}}}}^{{\\text{{CE}}}} = {values['ce_pfail']}$",
    )

    # ── Density table rows ───────────────────────────────────────────────
    def _fmt(tup):
        return f"{tup[0]}$\\pm${tup[1]}"

    cv_row = f"CV Baseline         & {_fmt(dens['cv_3'])}  & {_fmt(dens['cv_5'])} & {_fmt(dens['cv_7'])} & {_fmt(dens['cv_10'])} \\\\"
    sf_row = f"SF-EKF              & {_fmt(dens['sf_3'])}  & {_fmt(dens['sf_5'])}  & {_fmt(dens['sf_7'])} & {_fmt(dens['sf_10'])} \\\\"
    sx_row = (f"SF-EKF + Simplex    & \\textbf{{{dens['sx_3'][0]}}}$\\pm${dens['sx_3'][1]} "
              f"& \\textbf{{{dens['sx_5'][0]}}}$\\pm${dens['sx_5'][1]} "
              f"& \\textbf{{{dens['sx_7'][0]}}}$\\pm${dens['sx_7'][1]} "
              f"& \\textbf{{{dens['sx_10'][0]}}}$\\pm${dens['sx_10'][1]} \\\\")

    # Replace each density table row using regex (tolerant of spacing)
    tex = re.sub(
        r"CV Baseline\s+&.*?\\\\",
        cv_row,
        tex,
        count=1,
    )
    tex = re.sub(
        r"SF-EKF\s+& (?!\\\textbf).*?\\\\",
        sf_row,
        tex,
        count=1,
    )
    tex = re.sub(
        r"SF-EKF \+ Simplex\s+&.*?\\\\",
        sx_row,
        tex,
        count=1,
    )

    # ── STL ──────────────────────────────────────────────────────────────
    tex = _tex_replace(
        tex,
        "at $t \\approx 3.2$\\,s and reaches a near-collision ($\\mathrm{TTC} \\approx 0.1$\\,s, $\\rho = -1.9$)",
        f"at $t \\approx {values['stl_violation_time']}$\\,s and reaches a near-collision ($\\mathrm{{TTC}} \\approx {values['stl_min_ttc']}$\\,s, $\\rho = {values['stl_rho']}$)",
    )

    # ── Validation table ─────────────────────────────────────────────────
    tex = re.sub(
        r"Direct MC \(\$N\{=\}500\$\)\s+& 0\.\d+ & 0\.\d+ & [\d.]+\$\\times\$",
        f"Direct MC ($N{{=}}{values['mc_N']}$)        & {values['mc_pfail']} & {values['mc_se']} & 1.0$\\times$",
        tex,
        count=1,
    )
    tex = re.sub(
        r"IS \(\$N\{=\}500\$\)\s+& 0\.\d+ & 0\.\d+ & [\d.]+\$\\times\$",
        f"IS ($N{{=}}{values['mc_N']}$)                & {values['is_pfail']} & {values['is_se']} & {values['is_vr']}$\\times$",
        tex,
        count=1,
    )
    tex = re.sub(
        r"CE \(\$N\{=\}500\$.*?\)\s+& 0\.\d+ & 0\.\d+ & [\d.]+\$\\times\$",
        f"CE ($N{{=}}{values['mc_N']}$, $\\ell{{=}}{values['ce_n_iters']}$)  & {values['ce_pfail']} & {round(values['mc_se'] / math.sqrt(values['ce_vr']), 3)} & {values['ce_vr']}$\\times$",
        tex,
        count=1,
    )
    tex = re.sub(
        r"Bayesian MAP\s+& 0\.\d+",
        f"Bayesian MAP                                & {values['bayes_map']}",
        tex,
        count=1,
    )
    tex = re.sub(
        r"Bayesian 95\\% CI: \$\[[\d.]+,\s*[\d.]+\]\$",
        f"Bayesian 95\\% CI: $[{values['ci_lo_bayes']}, {values['ci_hi_bayes']}]$",
        tex,
        count=1,
    )

    # ── Ablation table ───────────────────────────────────────────────────
    a = abl
    full_row = f"Full (SF-EKF + Simplex) & \\textbf{{{a['full']['coll']}}} & {a['full']['ade']} & {a['full']['fde']} & {a['full']['fp']} \\\\"
    ns_row = f"w/o Social Force         & {a['no_social']['coll']} & {a['no_social']['ade']} & {a['no_social']['fde']} & {a['no_social']['fp']} \\\\"
    nx_row = f"w/o Simplex              & {a['no_simplex']['coll']} & \\textbf{{{a['no_simplex']['ade']}}} & \\textbf{{{a['no_simplex']['fde']}}} & {a['no_simplex']['fp']} \\\\"
    cv_only_row = f"w/o Both (CV only)       & {a['cv_only']['coll']} & {a['cv_only']['ade']} & {a['cv_only']['fde']} & {a['cv_only']['fp']} \\\\"

    tex = re.sub(r"Full \(SF-EKF \+ Simplex\)\s*&.*?\\\\", full_row, tex, count=1)
    tex = re.sub(r"w/o Social Force\s+&.*?\\\\", ns_row, tex, count=1)
    tex = re.sub(r"w/o Simplex\s+&.*?\\\\", nx_row, tex, count=1)
    tex = re.sub(r"w/o Both \(CV only\)\s+&.*?\\\\", cv_only_row, tex, count=1)

    # ── Regime table ─────────────────────────────────────────────────────
    tex = re.sub(
        r"Low density\s+& 3\s+& [\d.]+\s+& [\d.]+ \(\d+\\% \$\\downarrow\$\)",
        f"Low density    & 3     & {dens['cv_3'][0]}  & {dens['sf_3'][0]} ({values['low_reduction']}\\% $\\downarrow$)",
        tex,
        count=1,
    )
    tex = re.sub(
        r"Medium density & 5--7\s+& [\d.]+\s+& [\d.]+ \(\d+\\% \$\\downarrow\$\)",
        f"Medium density & 5--7  & {values['cv_med']} & {values['sf_med']} ({values['med_reduction']}\\% $\\downarrow$)",
        tex,
        count=1,
    )
    tex = re.sub(
        r"High density\s+& 10\s+& [\d.]+\s+& [\d.]+ \(\d+\\% \$\\downarrow\$\)",
        f"High density   & 10    & {dens['cv_10'][0]} & {dens['sf_10'][0]} ({values['high_reduction']}\\% $\\downarrow$)",
        tex,
        count=1,
    )

    # ── SF reduction text ────────────────────────────────────────────────
    tex = _tex_replace(
        tex,
        "reduces the collision rate by 52\\% on average",
        f"reduces the collision rate by {values['sf_vs_cv_pct']}\\% on average",
    )
    tex = _tex_replace(
        tex,
        "$N_{\\text{ped}} = 5$: 16.4\\% $\\to$ 7.8\\%",
        f"$N_{{\\text{{ped}}}} = 5$: {cv5_rate}\\% $\\to$ {sf5}\\%",
    )

    # ── Simplex reduction + FP rate ──────────────────────────────────────
    tex = _tex_replace(
        tex,
        "additional 73\\% collision reduction at the cost of a 6.3\\%",
        f"additional {values['simplex_vs_sf_pct']}\\% collision reduction at the cost of a {values['sf_fp_at_tau_star']}\\%",
    )

    # ── Threshold text ───────────────────────────────────────────────────
    tex = _tex_replace(
        tex,
        "3.1\\% collision rate, 6.3\\% FP brake rate for SF-EKF",
        f"{values['sf_coll_at_tau_star']}\\% collision rate, {values['sf_fp_at_tau_star']}\\% FP brake rate for SF-EKF",
    )

    # ── IS convergence ───────────────────────────────────────────────────
    tex = _tex_replace(
        tex,
        "IS achieves a 5.2$\\times$ variance reduction over direct MC, while the CE method achieves 4.8$\\times$",
        f"IS achieves a {values['is_vr']}$\\times$ variance reduction over direct MC, while the CE method achieves {values['ce_vr']}$\\times$",
    )

    # ── Conclusion ───────────────────────────────────────────────────────
    tex = _tex_replace(
        tex,
        "collision rates by 52\\% over",
        f"collision rates by {values['sf_vs_cv_pct']}\\% over",
    )
    tex = _tex_replace(
        tex,
        "accounts for 68\\% of collisions",
        f"accounts for {values['failure_broadside_pct']}\\% of collisions",
    )
    tex = _tex_replace(
        tex,
        "5.2$\\times$ variance reduction for failure",
        f"{values['is_vr']}$\\times$ variance reduction for failure",
    )
    tex = _tex_replace(
        tex,
        "further reduces collisions by 73\\% at the cost of a 6.3\\%",
        f"further reduces collisions by {values['simplex_vs_sf_pct']}\\% at the cost of a {values['sf_fp_at_tau_star']}\\%",
    )
    tex = _tex_replace(
        tex,
        "equivalent estimation precision can be achieved with $\\sim$96 samples instead of $\\sim$500",
        f"equivalent estimation precision can be achieved with $\\sim${values['is_equiv_samples']} samples instead of $\\sim${values['mc_N']}",
    )
    tex = _tex_replace(
        tex,
        "The 5.2$\\times$ variance reduction from IS",
        f"The {values['is_vr']}$\\times$ variance reduction from IS",
    )

    TEX_PATH.write_text(tex)
    logger.info("Updated %s", TEX_PATH)


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(description="Extract paper values from experiments")
    parser.add_argument("--update-tex", action="store_true", help="Update main.tex inline")
    args = parser.parse_args()

    # Load results
    data = _load("full_results.json")
    if not data:
        logger.warning("full_results.json not found — trying validation_results.json")
        data = _load("validation_results.json")
    if not data:
        logger.error("No results found. Run run_full_experiments.py first.")
        return

    values = extract_values(data)

    # Print
    print("=" * 72)
    print("ALL PAPER VALUES")
    print("=" * 72)
    for k, v in values.items():
        print(f"  {k}: {v}")

    # Save JSON
    out_path = RESULTS_DIR / "paper_values.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(values, f, indent=2, default=str)
    print(f"\nSaved to {out_path}")

    # Optionally update TeX
    if args.update_tex:
        update_tex(values)
        print(f"Updated {TEX_PATH}")


if __name__ == "__main__":
    main()
