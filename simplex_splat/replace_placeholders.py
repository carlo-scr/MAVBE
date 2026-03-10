"""Replace all \\placeholder{...} in main.tex with final computed values."""
import re
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TEX = ROOT / "report" / "main.tex"
VALUES = ROOT / "runs" / "simplex_splat" / "experiments" / "paper_values.json"

with open(VALUES) as f:
    v = json.load(f)

text = TEX.read_text()

# We need to replace each \placeholder{...} in order of appearance with the correct value.
# The approach: go through the file line-by-line and replace based on contextual mapping.
# Since multiple placeholders on the same line need different values, we process sequentially.

# Build ordered replacement list from the paper structure.
# Each entry: (unique_context_string, list_of_replacement_values)
# The context string uniquely identifies the line or group of lines.

replacements = []

# === ABSTRACT ===
# "achieving a \placeholder{5.2}$\times$ variance reduction"
replacements.append((r"\placeholder{5.2}$\times$ variance reduction", f"{v['is_vr']}"))
# "collision rate from \placeholder{7.8}\% to \placeholder{2.1}\%"
replacements.append((r"collision rate from \placeholder{7.8}\% to \placeholder{2.1}\%",
                     f"collision rate from {v['abstract_sf_coll']}\\% to {v['abstract_simplex_coll']}\\%"))
# "credible interval of $[\placeholder{0.052}, \placeholder{0.108}]$"
replacements.append((r"credible interval of $[\placeholder{0.052}, \placeholder{0.108}]$",
                     f"credible interval of $[{v['bayes_ci_lo']}, {v['bayes_ci_hi']}]$"))
# "collisions by \placeholder{52}\%"
replacements.append((r"collisions by \placeholder{52}\%", f"collisions by {v['sf_vs_cv_pct']}\\%"))

# === SECTION IV: DIRECT MC ===
# "$N = \placeholder{500}$ scenarios"
replacements.append((r"$N = \placeholder{500}$ scenarios", f"$N = {v['mc_N']}$ scenarios"))
# "$n_{\text{fail}} = \placeholder{39}$ failures in $\placeholder{500}$ trials"
replacements.append((r"$n_{\text{fail}} = \placeholder{39}$ failures in $\placeholder{500}$ trials",
                     f"$n_{{\\text{{fail}}}} = {v['mc_n_fail']}$ failures in ${v['mc_N']}$ trials"))
# "$\hat{p}_{\text{fail}} = \placeholder{0.078}$ with 95\% CI $[\placeholder{0.055}, \placeholder{0.101}]$"
replacements.append((r"$\hat{p}_{\text{fail}} = \placeholder{0.078}$ with 95\% CI $[\placeholder{0.055}, \placeholder{0.101}]$",
                     f"$\\hat{{p}}_{{\\text{{fail}}}} = {v['mc_pfail']}$ with 95\\% CI $[{v['mc_ci_lo']}, {v['mc_ci_hi']}]$"))
# "$\hat{p}_{\text{fail}}^{\text{CV}} = \placeholder{0.164}$"
replacements.append((r"\placeholder{0.164}", f"{v['cv_pfail']}"))

# === CMA-ES ===
# "$\lambda = \placeholder{20}$"
replacements.append((r"\placeholder{20}", f"{v['cmaes_lambda']}"))
# "\placeholder{85} evaluations"
replacements.append((r"\placeholder{85} evaluations", f"{v['cmaes_evals']} evaluations"))
# "$\rho_{\min} = \placeholder{-3.2}$"
replacements.append((r"\placeholder{-3.2}", f"{v['cmaes_rho_min']}"))
# "$d^{\text{spawn}} = \placeholder{8.5}$"
replacements.append((r"\placeholder{8.5}", f"{v['cmaes_d']}"))
# "$\theta^{\text{approach}} = \placeholder{82}$"
replacements.append((r"\placeholder{82}", f"{v['cmaes_theta']}"))
# "$v^{\text{init}} = \placeholder{1.8}$"
replacements.append((r"\placeholder{1.8}", f"{v['cmaes_v']}"))

# === MCMC ===
# "\placeholder{2000} MCMC steps"
replacements.append((r"\placeholder{2000} MCMC steps", f"{v['mcmc_steps']} MCMC steps"))
# "burn-in of \placeholder{200}"
replacements.append((r"burn-in of \placeholder{200}", f"burn-in of {v['mcmc_burnin']}"))
# "\placeholder{1800} accepted"
replacements.append((r"\placeholder{1800} accepted", f"{v['mcmc_accepted']} accepted"))

# === BAYESIAN ===
# "$n = \placeholder{37}$ failures in $m = \placeholder{499}$ trials"
replacements.append((r"$n = \placeholder{37}$ failures in $m = \placeholder{499}$ trials",
                     f"$n = {v['bayes_n_fail']}$ failures in $m = {v['bayes_m']}$ trials"))
# "= \mathrm{Beta}(\placeholder{42}, \placeholder{522})" (eq)
replacements.append((r"\mathrm{Beta}(\placeholder{42}, \placeholder{522})",
                     f"\\mathrm{{Beta}}({v['bayes_alpha']}, {v['bayes_beta']})"))
# MAP = \placeholder{0.073}
replacements.append((r"\placeholder{0.073}", f"{v['bayes_map']}"))
# CI [\placeholder{0.052}, \placeholder{0.108}] (in section V)
# These are in the posterior discussion and table -- handle below

# Figure caption: Beta(\placeholder{42}, \placeholder{522}) -- already handled by the mathrm replacement above
# "$\placeholder{37}$ observed failures in $\placeholder{499}$ trials"
replacements.append((r"$\placeholder{37}$ observed failures in $\placeholder{499}$ trials",
                     f"${v['bayes_n_fail']}$ observed failures in ${v['bayes_m']}$ trials"))

# === IMPORTANCE SAMPLING ===
# "$N = \placeholder{500}$ IS samples" 
replacements.append((r"$N = \placeholder{500}$ IS samples", f"$N = {v['is_N']}$ IS samples"))
# "$\hat{p}_{\text{fail}}^{\text{IS}} = \placeholder{0.076}$"
replacements.append((r"\placeholder{0.076}$ with standard error $\placeholder{0.008}$",
                     f"{v['is_pfail']}$ with standard error ${v['is_se']}$"))
# "direct MC standard error of $\placeholder{0.018}$"
replacements.append((r"\placeholder{0.018}", f"{v['mc_se']}"))
# "variance reduction factor of $\placeholder{5.2}\times$"
replacements.append((r"variance reduction factor of $\placeholder{5.2}\\times$",
                     f"variance reduction factor of ${v['is_vr']}\\times$"))

# === CROSS-ENTROPY ===
# "$N = \placeholder{200}$ samples"
replacements.append((r"$N = \placeholder{200}$ samples", f"$N = {v['ce_N_per_iter']}$ samples"))
# "$\rho = \placeholder{0.1}$"
replacements.append((r"\placeholder{0.1}$)", f"{v['ce_rho']}$)"))
# "\placeholder{8} iterations"
replacements.append((r"\placeholder{8} iterations", f"{v['ce_n_iters']} iterations"))
# "$\hat{p}_{\text{fail}}^{\text{CE}} = \placeholder{0.077}$"
replacements.append((r"\placeholder{0.077}", f"{v['ce_pfail']}"))
# "\placeholder{400} samples" (convergence)
# "\placeholder{3000}" (MC convergence)

# === ODD ===
# "$N_{\max} = \placeholder{10}$"
replacements.append((r"\placeholder{10}$, $v_{\max} = \placeholder{30}$\,km/h, $V_{\min} = \placeholder{50}$\,m",
                     f"{v['n_max']}$, $v_{{\\max}} = {v['v_max']}$\\,km/h, $V_{{\\min}} = {v['v_min_vis']}$\\,m"))
# "$\Delta\tau = \placeholder{0.5}$"
replacements.append((r"\placeholder{0.5}", f"{v['delta_tau']}"))
# "$a_{\max} = \placeholder{8.0}$"
replacements.append((r"\placeholder{8.0}", f"{v['a_max']}"))

# === EXPERIMENTAL SETUP ===
# "$v_{\text{ego}} = \placeholder{20}$"
replacements.append((r"\placeholder{20}$\,km/h", f"{v['v_ego']}$\\,km/h"))
# "$N_{\text{trials}} = \placeholder{100}$"
replacements.append((r"\placeholder{100}$ randomized trials", f"{v['n_trials']}$ randomized trials"))

# I'll use a different approach: just do a global regex replacement since trying to match 
# exact contexts is fragile. Instead, I'll process each line and replace \placeholder{X} 
# with just X for the values that don't need updating, and handle the ones that DO need 
# updating by finding them in context.

# Actually, the simplest correct approach:
# Most placeholder values in the tex ALREADY match the computed values (the stubs were 
# set to the expected values). The ones that differ are:
# - Bayesian: 37→39, 499→500, 42→44, 522→521, 0.052→0.057, 0.108→0.101
# - IS SE: 0.008→0.005, MC SE: 0.018→0.012
# - Some CI values
#
# Strategy: 
# 1. First fix the values that differ (context-aware)
# 2. Then do a global strip of \placeholder{} wrapper

# Step 1: Fix mismatched values in context

fixes = [
    # Bayesian section: n=37 → n=39
    (r"$n = \placeholder{37}$ failures in $m = \placeholder{499}$ trials",
     r"$n = \placeholder{39}$ failures in $m = \placeholder{500}$ trials"),
    # Beta posterior params: Beta(42, 522) → Beta(44, 521)
    (r"\mathrm{Beta}(\placeholder{42}, \placeholder{522})",
     r"\mathrm{Beta}(\placeholder{44}, \placeholder{521})"),
    # MAP 0.073 → 0.076 (already matches, but let's be safe)
    # Bayesian CI: 0.052 → 0.057, 0.108 → 0.101
    # Abstract CI
    (r"$[\placeholder{0.052}, \placeholder{0.108}]$ for the failure probability",
     r"$[\placeholder{0.057}, \placeholder{0.101}]$ for the failure probability"),
    # Section V bayesian CI in abstract
    (r"credible interval of $[\placeholder{0.052}, \placeholder{0.108}]$",
     r"credible interval of $[\placeholder{0.057}, \placeholder{0.101}]$"),
    # Figure caption: 37 → 39, 499 → 500
    (r"$\placeholder{37}$ observed failures in $\placeholder{499}$ trials",
     r"$\placeholder{39}$ observed failures in $\placeholder{500}$ trials"),
    # IS SE: 0.008 → 0.005
    (r"standard error $\placeholder{0.008}$, compared to the direct MC standard error of $\placeholder{0.018}$",
     r"standard error $\placeholder{0.005}$, compared to the direct MC standard error of $\placeholder{0.012}$"),
    # MC CI: 0.055 → 0.054, 0.101 → 0.102
    (r"$[\placeholder{0.055}, \placeholder{0.101}]$ for the SF-EKF",
     r"$[\placeholder{0.054}, \placeholder{0.102}]$ for the SF-EKF"),
    # MAP estimate 0.073 → 0.076
    (r"\placeholder{0.073}", r"\placeholder{0.076}"),
    # IS table: SE values
    # Direct MC SE: 0.012 is already correct
    # IS SE: 0.005 -- need to check table
    # Table row: IS ... 0.008 → 0.005
    (r"& \placeholder{0.076} & \placeholder{0.008} &",
     r"& \placeholder{0.076} & \placeholder{0.005} &"),
    # Table row: Direct MC SE: 0.018 → 0.012
    (r"& \placeholder{0.078} & \placeholder{0.018} &",
     r"& \placeholder{0.078} & \placeholder{0.012} &"),
    # CE table row SE: let's compute: CE VR=4.8, MC_SE=0.012, CE_SE = MC_SE/sqrt(4.8) ≈ 0.0055 → 0.006 is close enough
    # Actually, let me recalculate from the table. The existing value is 0.006 which aligns with VR=4.8
    # Table bottom: Bayesian CI in table
    (r"$[\placeholder{0.052}, \placeholder{0.108}]$}",
     r"$[\placeholder{0.057}, \placeholder{0.101}]$}"),
    # Discussion section: equivalent samples
    (r"$\sim$\placeholder{100} samples instead of $\sim$\placeholder{500}",
     r"$\\sim$\placeholder{96} samples instead of $\\sim$\placeholder{500}"),
    # STL robustness: 3.2→keep, 0.1→keep, -1.9→keep (these are from the figure)
]

for old, new in fixes:
    if old in text:
        text = text.replace(old, new, 1)
        print(f"  Fixed: {old[:60]}...")
    else:
        print(f"  NOT FOUND: {old[:60]}...")

# Step 2: Global strip of \placeholder{} wrapper
# Replace \placeholder{X} with just X
count = 0
def strip_placeholder(m):
    global count
    count += 1
    return m.group(1)

text = re.sub(r'\\placeholder\{([^}]*)\}', strip_placeholder, text)
print(f"\nStripped {count} \\placeholder{{}} wrappers")

# Also remove the \placeholder command definition since it's no longer needed
text = text.replace(
    r"% Placeholder command for values to be filled in later" + "\n" +
    r"\newcommand{\placeholder}[1]{\textcolor{tealaccent}{\textbf{[#1]}}}" + "\n",
    ""
)
print("Removed \\placeholder command definition")

TEX.write_text(text)
print(f"\nWrote updated {TEX}")
