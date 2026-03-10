# Simplex-Track

**Validating Behavior-Aware Pedestrian Tracking for Safe Autonomous Navigation**

> Final project for CS238V — Algorithms for Validation (Stanford, Winter 2025)
>
> Rahul Ayanampudi · Carlo Schreiber

---

## Overview

Simplex-Track is a comprehensive safety-validation framework for a **Social Force Extended Kalman Filter (SF-EKF)** pedestrian tracker deployed in the CARLA simulator with YOLOv9 detection and DeepSORT association.

Following the validation pipeline of [Kochenderfer et al. (2025)](https://algorithmsbook.com/validation/), the framework:

1. Formalizes the tracking system as a stochastic process and specifies safety properties in **Signal Temporal Logic (STL)**.
2. Applies **direct Monte Carlo** simulation and **CMA-ES** optimization-based falsification to discover failure modes.
3. Characterizes the failure distribution via **Metropolis–Hastings MCMC**.
4. Estimates failure probabilities using **importance sampling** and the **cross-entropy method** (5.2× variance reduction over direct MC).
5. Deploys a **Simplex runtime monitor** with TTC-based switching that reduces the collision rate from 7.8% to 2.1% at medium pedestrian density.
6. Provides **Bayesian estimation** with a Beta posterior yielding a 95% credible interval of [0.057, 0.101].

---

## Repository Structure

```
MAVBE/
├── simplex_splat/              # Core validation framework
│   ├── simulation.py           # Scenario runner (CARLA or lightweight fallback)
│   ├── validation.py           # 6 validation methods (MC, CMA-ES, MCMC, IS, CE, Bayesian)
│   ├── run_full_experiments.py # Full experiment sweep orchestrator
│   ├── generate_all_figures.py # Regenerate all PGFplots figures from results
│   ├── extract_paper_values.py # Extract numbers from results → update main.tex
│   ├── monitor.py              # Safety monitor (SAM / PGM modes)
│   ├── metrics.py              # Per-frame and per-scenario evaluation metrics
│   ├── run_experiments.py      # SAM/PGM ghost & blind-map experiments
│   ├── run_validation.py       # Standalone synthetic validation pipeline
│   ├── generate_figures.py     # ROC ablation + CDF figure generation
│   ├── compute_paper_values.py # Hardcoded paper values (pre-pipeline)
│   └── replace_placeholders.py # Strip \placeholder{} wrappers from TeX
│
├── perception/
│   ├── detect_dual_tracking.py # YOLOv9 + DeepSORT end-to-end tracking
│   ├── deep_sort/              # DeepSORT tracker with Behavioral EKF
│   │   ├── deep_sort/
│   │   │   ├── behavioral_ekf.py   # 5D CT + social force EKF
│   │   │   ├── kalman_filter.py     # 8D constant-velocity baseline
│   │   │   ├── tracker.py           # Multi-target tracker
│   │   │   ├── track.py             # Single track state machine
│   │   │   ├── nn_matching.py       # Cosine/Euclidean ReID matching
│   │   │   └── ...
│   │   ├── deep_sort_app.py         # MOTChallenge runner
│   │   └── evaluate_motchallenge.py # Batch evaluation
│   └── yolov9/                 # YOLOv9 detection (inference, training, export)
│
├── carla_integration/
│   ├── spawn_pedestrian_video.py   # CARLA pedestrian scenario video capture
│   └── trajectory_planning.py      # Ego vehicle control + local planner
│
├── report/
│   ├── main.tex                # IEEE conference paper
│   ├── references.bib
│   └── figures/                # PGFplots .tex figures (auto-generated)
│
├── runs/simplex_splat/experiments/ # Experiment outputs (JSON)
├── configs/deep_sort.yaml      # DeepSORT hyperparameters
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install cma  # for CMA-ES falsification
```

### 2. Run the full validation pipeline

```bash
# Quick smoke test (~1 second, small samples)
python -m simplex_splat.run_full_experiments --quick

# Full run (larger samples, more precise estimates)
python -m simplex_splat.run_full_experiments
```

This executes:
- **Density sweep** — 3 trackers × 4 pedestrian densities × N trials
- **Validation pipeline** — MC → CMA-ES → MCMC → IS → CE → Bayesian (chained)
- **Threshold sweep** — τ_safe from 0.5s to 4.0s
- **STL robustness trace** — worst-case TTC time series
- **Ablation study** — full / no social force / no simplex / CV only

Results are saved to `runs/simplex_splat/experiments/full_results.json`.

### 3. Generate figures

```bash
python -m simplex_splat.generate_all_figures
```

Writes 6 PGFplots `.tex` files to `report/figures/`:
`failure_rate_density`, `failure_distribution`, `importance_sampling`,
`roc_threshold`, `stl_robustness`, `bayesian_posterior`.

### 4. Update paper values

```bash
# Print all derived values
python -m simplex_splat.extract_paper_values

# Update main.tex inline with values from the latest run
python -m simplex_splat.extract_paper_values --update-tex
```

---

## Validation Methods

| Method | Module | Description |
|--------|--------|-------------|
| **Direct Monte Carlo** | `validation.run_monte_carlo` | Uniform sampling over disturbance space, standard CI |
| **CMA-ES Falsification** | `validation.run_cmaes` | Optimization-based worst-case search (minimizes ρ_min) |
| **MH-MCMC** | `validation.run_mcmc` | Metropolis–Hastings sampling of the failure distribution |
| **Importance Sampling** | `validation.run_importance_sampling` | Gaussian proposal centered on MCMC failure region |
| **Cross-Entropy** | `validation.run_cross_entropy` | Iterative elite selection with adaptive proposal |
| **Bayesian Estimation** | `validation.run_bayesian` | Beta-Binomial posterior on p_fail with credible interval |

---

## Perception Stack

The tracker fuses three components:

- **YOLOv9** — real-time object detection (`yolov9-c.pt`)
- **DeepSORT** — appearance + motion association with OSNet ReID features
- **Behavioral EKF** — 5D coordinated-turn model with Helbing–Molnár social force repulsion

The **BehavioralEKFFilter** wraps the 5D EKF as a drop-in replacement for the standard 8D constant-velocity Kalman filter, converting between the `[x, y, a, h, vx, vy, va, vh]` DeepSORT state and the internal `[px, py, v, φ, ω]` EKF state.

---

## Simplex Runtime Monitor

The Simplex architecture provides a safety layer on top of the tracker:

1. **Advanced controller** — SF-EKF tracker provides behavior-aware predictions
2. **Decision module** — computes TTC for each tracked pedestrian; if TTC < τ_safe, switches to the baseline controller
3. **Baseline controller** — conservative constant-velocity tracker (guaranteed safe under its assumptions)

The monitor operates in two modes:
- **SAM** (Semantic-Aware Monitor) — checks dynamic, static, and structural integrity
- **PGM** (Pure Geometric Monitor) — global depth-residual thresholding baseline

---

## CARLA Integration

When CARLA 0.9.11+ is available, the pipeline runs full closed-loop simulation:

- Town10HD map, synchronous mode at 20 Hz
- Tesla Model 3 ego vehicle with RGB camera
- AI walker controllers for pedestrian spawning
- Ground-truth TTC computation from world positions

Without CARLA, a lightweight analytical simulation model is used automatically as a fallback — matching the collision probability characteristics of the full simulator.

```bash
# Start CARLA server first, then:
python -m simplex_splat.run_full_experiments --carla
```

---

## Existing Component Runners

### Detection + Tracking (video / webcam)

```bash
python detect_dual_tracking.py --source path/to/video.mp4 --weights perception/yolov9/yolov9-c.pt
```

### Behavioral EKF on MOTChallenge

```bash
cd perception/deep_sort
python deep_sort_app.py \
    --sequence_dir=./MOT16/train/MOT16-02 \
    --detection_file=./resources/detections/MOT16-02.npy \
    --min_confidence=0.3 --display=True
```

### CARLA Pedestrian Video

```bash
# Requires a running CARLA server
python carla_integration/spawn_pedestrian_video.py
```

See `perception/deep_sort/README.md` and `perception/yolov9/README.md` for further details.

---

## Key Results

| Metric | Value |
|--------|-------|
| SF-EKF collision reduction vs CV baseline | 52% |
| Simplex collision reduction vs SF-EKF alone | 73% |
| IS variance reduction over direct MC | 5.2× |
| Bayesian 95% CI for p_fail | [0.057, 0.101] |
| CMA-ES worst-case ρ_min | −3.2 s |
| Dominant failure mode | close-range broadside (68%) |

---

## Citation

```bibtex
@article{ayanampudi2025simplextrack,
  title={Simplex-Track: Validating Behavior-Aware Pedestrian Tracking
         for Safe Autonomous Navigation},
  author={Ayanampudi, Rahul and Schreiber, Carlo},
  journal={CS238V Final Project, Stanford University},
  year={2025}
}
```

---

## License

This project was developed for academic purposes as part of Stanford CS238V.
YOLOv9 and DeepSORT components retain their original licenses (see `perception/yolov9/LICENSE.md` and `perception/deep_sort/LICENSE`).
