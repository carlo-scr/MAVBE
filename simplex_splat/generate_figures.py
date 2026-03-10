#!/usr/bin/env python3
"""Generate updated figure .tex files and print paper values from experiment results."""
import json
import os
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = ROOT / "runs" / "simplex_splat" / "experiments" / "experiment_results.json"
FIGURES_DIR = ROOT / "report" / "figures"

with open(RESULTS_PATH) as f:
    data = json.load(f)

# ─────────────────────────────────────────────────────────────────────────────
# Print summary for paper table values
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 72)
print("TABLE 1: Ghost Scenario (Dynamic Pedestrian)")
print("=" * 72)
print(f"{'Monitor':<22} {'TPR%':>6} {'FPR%':>6} {'Resp(ms)':>9} {'Coll%':>6}")
print("-" * 55)
for r in data["table_ghost"]:
    label = "PGM" if r["monitor_type"] == "geometric" else "SAM"
    print(f"{label} (tau={r['tau']:.1f}m)          "
          f"{r['tpr_mean']*100:>5.1f} {r['fpr_mean']*100:>5.1f} "
          f"{r['response_time_ms_mean']:>8.0f} {r['collision_rate']*100:>5.0f}")

print()
print("=" * 72)
print("TABLE 2: Blind Map Scenario (Static Integrity)")
print("=" * 72)
print(f"{'Monitor':<22} {'TPR%':>6} {'FPR%':>6} {'IoU':>6}")
print("-" * 42)
for r in data["table_blind"]:
    label = "PGM" if r["monitor_type"] == "geometric" else "SAM"
    iou_str = f"{r['iou_stop_sign_mean']:.2f}" if r["monitor_type"] == "semantic" else "---"
    print(f"{label} (tau={r['tau']:.1f}m)          "
          f"{r['tpr_mean']*100:>5.1f} {r['fpr_mean']*100:>5.1f} {iou_str:>6}")

# ─────────────────────────────────────────────────────────────────────────────
# ROC data
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("ROC DATA")
print("=" * 72)
print("SAM ROC points (FPR%, TPR%):")
sam_roc = [(r["fpr_mean"]*100, r["tpr_mean"]*100, r["tau"]) for r in data["roc_sam"]]
for fpr, tpr, tau in sam_roc:
    print(f"  tau={tau:.2f}: ({fpr:.1f}, {tpr:.1f})")

print("PGM ROC points (FPR%, TPR%):")
pgm_roc = [(r["fpr_mean"]*100, r["tpr_mean"]*100, r["tau"]) for r in data["roc_pgm"]]
for fpr, tpr, tau in pgm_roc:
    print(f"  tau={tau:.2f}: ({fpr:.1f}, {tpr:.1f})")

# ─────────────────────────────────────────────────────────────────────────────
# CDF data
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("RESPONSE TIME CDF DATA")
print("=" * 72)
sam_cdf = data["cdf_sam"]
pgm_cdf = data["cdf_pgm"]
print(f"SAM: n={len(sam_cdf)}, median={np.median(sam_cdf):.0f}ms" if sam_cdf else "SAM: no data")
print(f"PGM: n={len(pgm_cdf)}, median={np.median(pgm_cdf):.0f}ms" if pgm_cdf else "PGM: no data")
if sam_cdf:
    print(f"  SAM values: {sam_cdf}")
if pgm_cdf:
    print(f"  PGM values: {pgm_cdf}")

# ─────────────────────────────────────────────────────────────────────────────
# Generate updated ROC figure
# ─────────────────────────────────────────────────────────────────────────────
# Deduplicate ROC points (keep unique FPR,TPR pairs)
def dedupe_roc(points):
    seen = set()
    result = []
    for fpr, tpr, tau in points:
        key = (round(fpr, 2), round(tpr, 2))
        if key not in seen:
            seen.add(key)
            result.append((fpr, tpr, tau))
    return result

sam_roc_dedup = dedupe_roc(sam_roc)
pgm_roc_dedup = dedupe_roc(pgm_roc)

# Sort by FPR for plotting
sam_roc_dedup.sort(key=lambda x: x[0])
pgm_roc_dedup.sort(key=lambda x: x[0])

sam_coords = "\n".join(f"        ({fpr:.1f}, {tpr:.1f})    %% tau={tau}" for fpr, tpr, tau in sam_roc_dedup)
pgm_coords = "\n".join(f"        ({fpr:.1f}, {tpr:.1f})    %% tau={tau}" for fpr, tpr, tau in pgm_roc_dedup)

# Find the tau=1.0 annotations
sam_at_1 = next((fpr, tpr) for fpr, tpr, tau in sam_roc if abs(tau - 1.0) < 0.01)
pgm_at_1 = next((fpr, tpr) for fpr, tpr, tau in pgm_roc if abs(tau - 1.0) < 0.01)

roc_tex = f"""% ROC-style ablation over residual threshold
% Generated from experiment_results.json
\\begin{{tikzpicture}}
\\begin{{axis}}[
    width=\\columnwidth,
    height=6cm,
    xlabel={{False Positive Rate (\\%)}},
    ylabel={{True Positive Rate (\\%)}},
    xmin=0, xmax=105,
    ymin=0, ymax=105,
    grid=major,
    grid style={{gray!20}},
    legend style={{
        at={{(0.98,0.02)}},
        anchor=south east,
        font=\\small,
        draw=gray!50,
        fill=white,
        fill opacity=0.9,
    }},
    tick label style={{font=\\small}},
    label style={{font=\\small}},
    every axis plot/.append style={{line width=1.2pt, mark size=2.5pt}},
]

% SAM (Semantically-Aware Monitor)
\\addplot[color=tealaccent, mark=*, mark options={{fill=tealaccent}}]
    coordinates {{
{sam_coords}
    }};
\\addlegendentry{{SAM (Ours)}}

% PGM (Pure Geometric Monitor)
\\addplot[color=terracotta, mark=triangle*, mark options={{fill=terracotta}}]
    coordinates {{
{pgm_coords}
    }};
\\addlegendentry{{PGM (Baseline)}}

% Diagonal (random classifier)
\\addplot[dashed, gray!50, line width=0.6pt, forget plot] coordinates {{(0,0) (100,100)}};

% Annotations for tau=1.0
\\node[font=\\tiny, text=tealaccent, anchor=south west] at (axis cs:{sam_at_1[0]+1:.1f},{sam_at_1[1]-2:.1f}) {{$\\tau{{=}}1.0$}};
\\node[font=\\tiny, text=terracotta, anchor=south west] at (axis cs:{pgm_at_1[0]+1:.1f},{pgm_at_1[1]-2:.1f}) {{$\\tau{{=}}1.0$}};

\\end{{axis}}
\\end{{tikzpicture}}
"""

roc_path = FIGURES_DIR / "roc_ablation.tex"
with open(roc_path, "w") as f:
    f.write(roc_tex)
print(f"\nWrote {roc_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Generate updated CDF figure
# ─────────────────────────────────────────────────────────────────────────────

def make_cdf_coords(times, label):
    if not times:
        return "        (0, 0) (700, 1.0)"
    coords = ["        (0, 0)"]
    n = len(times)
    for i, t in enumerate(times):
        cdf_val = (i + 1) / n
        coords.append(f"        ({t:.0f}, {cdf_val:.3f})")
    return "\n".join(coords)

sam_cdf_coords = make_cdf_coords(sam_cdf, "SAM")
pgm_cdf_coords = make_cdf_coords(pgm_cdf, "PGM")

sam_median = np.median(sam_cdf) if sam_cdf else 50
pgm_median = np.median(pgm_cdf) if pgm_cdf else 80

# Compute x-axis range from data
max_time = max(max(sam_cdf, default=0), max(pgm_cdf, default=0))
xmax = int(np.ceil(max_time / 100) * 100) + 50  # round up to next 100 + margin

cdf_tex = f"""% CDF of Safety Response Times
% Generated from experiment_results.json
\\begin{{tikzpicture}}
\\begin{{axis}}[
    width=\\columnwidth,
    height=5.5cm,
    xlabel={{Response Time (ms)}},
    ylabel={{Cumulative Probability}},
    xmin=0, xmax={xmax},
    ymin=0, ymax=1.05,
    grid=major,
    grid style={{gray!20}},
    legend style={{
        at={{(0.98,0.45)}},
        anchor=east,
        font=\\small,
        draw=gray!50,
        fill=white,
        fill opacity=0.9,
    }},
    tick label style={{font=\\small}},
    label style={{font=\\small}},
    every axis plot/.append style={{line width=1.4pt}},
    ytick={{0, 0.2, 0.4, 0.6, 0.8, 1.0}},
]

% SAM CDF
\\addplot[color=tealaccent, smooth]
    coordinates {{
{sam_cdf_coords}
    }};
\\addlegendentry{{SAM (Ours)}}

% PGM CDF
\\addplot[color=terracotta, smooth, dashed]
    coordinates {{
{pgm_cdf_coords}
    }};
\\addlegendentry{{PGM (Baseline)}}

% 100ms target line
\\draw[dashed, gray!70, line width=0.8pt] (axis cs:100,0) -- (axis cs:100,1.05);
\\node[font=\\scriptsize, text=gray, rotate=90, anchor=south] at (axis cs:115,0.10) {{100\\,ms target}};

% Median annotations
\\draw[dotted, tealaccent, line width=0.6pt] (axis cs:0,0.5) -- (axis cs:{sam_median:.0f},0.5) -- (axis cs:{sam_median:.0f},0);
\\node[font=\\tiny, text=tealaccent] at (axis cs:{sam_median:.0f},-0.08) {{{sam_median:.0f}}};
\\draw[dotted, terracotta, line width=0.6pt] (axis cs:0,0.5) -- (axis cs:{pgm_median:.0f},0.5);
\\node[font=\\tiny, text=terracotta] at (axis cs:{pgm_median+8:.0f},-0.08) {{{pgm_median:.0f}}};

\\end{{axis}}
\\end{{tikzpicture}}
"""

cdf_path = FIGURES_DIR / "response_time_cdf.tex"
with open(cdf_path, "w") as f:
    f.write(cdf_tex)
print(f"Wrote {cdf_path}")

# ─────────────────────────────────────────────────────────────────────────────
# Print LaTeX table values for copy-paste
# ─────────────────────────────────────────────────────────────────────────────
print()
print("=" * 72)
print("LATEX TABLE VALUES (copy into main.tex)")
print("=" * 72)
print()
print("% Table 1: Ghost Scenario")
for r in data["table_ghost"]:
    label = "PGM" if r["monitor_type"] == "geometric" else "SAM"
    tau_label = r"$\tau$" if label == "PGM" else r"$\tau_{\text{fn}}$"
    resp = r["response_time_ms_mean"]
    resp_str = f"{resp:.0f}" if resp < 500 else r"$>$500"
    coll = r["collision_rate"] * 100
    print(f"        {label} ({tau_label}{{=}}{r['tau']:.1f}\\,m) & "
          f"{r['tpr_mean']*100:.1f} & {r['fpr_mean']*100:.1f} & "
          f"{resp_str} & {coll:.0f} \\\\")

print()
print("% Table 2: Blind Map Scenario")
for r in data["table_blind"]:
    label = "PGM" if r["monitor_type"] == "geometric" else "SAM"
    iou_str = f"{r['iou_stop_sign_mean']:.2f}" if r["monitor_type"] == "semantic" else "---"
    print(f"        {label} (tau={r['tau']:.1f}m) & "
          f"{r['tpr_mean']*100:.1f} & {r['fpr_mean']*100:.1f} & "
          f"{iou_str} \\\\")
