"""
Simplex-Splat: Runtime Safety Monitor for 3D Gaussian Splatting Maps
====================================================================

A Simplex Architecture-based framework that validates the geometric and
semantic integrity of an online 3DGS map for autonomous navigation by
cross-validating it against raw sensor streams.

Modules
-------
postprocess   : Video → detection + tracking pipeline
analysis      : Failure distribution estimation + forward reachability
carla_client  : Synchronous CARLA client with ground truth sensor suite
splatam       : Online 3DGS mapping wrapper (SplaTAM integration)
monitor       : Safety monitor (hybrid residual computation)
emergency     : Deterministic emergency braking controller
metrics       : Evaluation metrics and logging
run_analysis  : Unified CLI entry point
"""

__version__ = "0.1.0"
