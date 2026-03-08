"""
SplaTAM wrapper for online 3D Gaussian Splatting mapping.

This module interfaces with the SplaTAM pipeline to maintain an online
3DGS map within a sliding spatial window.  It provides:
  - Incremental map construction from RGBD + pose
  - On-demand rendering of depth and colour from arbitrary poses
  - Sliding-window pruning for O(1) memory usage

When SplaTAM is unavailable (e.g. no GPU rasteriser), the module falls
back to a simple depth-buffer proxy so the safety monitor can still be
exercised.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class GaussianMapStats:
    """Bookkeeping for the current Gaussian map."""

    num_gaussians: int = 0
    window_centre: Optional[np.ndarray] = None  # (3,)
    last_densify_frame: int = 0
    total_frames_ingested: int = 0
    avg_ingest_ms: float = 0.0
    avg_render_ms: float = 0.0


class SplaTAMMapper:
    """Online 3DGS mapper wrapping the SplaTAM pipeline.

    Parameters
    ----------
    cfg : dict
        The ``splatam`` section of the master config.
    intrinsics : np.ndarray
        Camera intrinsic matrix (3×3).
    img_hw : tuple[int, int]
        (height, width) of sensor images.
    device : str
        Torch device string, e.g. ``"cuda:0"``.
    """

    def __init__(
        self,
        cfg: dict,
        intrinsics: np.ndarray,
        img_hw: Tuple[int, int],
        device: str = "cuda:0",
    ):
        self.cfg = cfg
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.intrinsics = torch.tensor(intrinsics, dtype=torch.float32, device=self.device)
        self.H, self.W = img_hw

        self.sliding_window_m = cfg.get("sliding_window_m", 50.0)
        self.max_gaussians = cfg.get("max_gaussians", 500_000)
        self.densify_interval = cfg.get("densification_interval", 5)
        self.prune_opacity_thr = cfg.get("pruning_opacity_threshold", 0.01)
        self.warmup_frames = cfg.get("warmup_frames", 10)

        self.stats = GaussianMapStats()

        # Gaussian parameters (all on device)
        self._means3D: Optional[torch.Tensor] = None        # (N, 3)
        self._opacities: Optional[torch.Tensor] = None      # (N, 1)
        self._scales: Optional[torch.Tensor] = None         # (N, 3)
        self._rotations: Optional[torch.Tensor] = None      # (N, 4)  quaternion
        self._colors: Optional[torch.Tensor] = None         # (N, 3)

        self._splatam_available = self._try_import_splatam()
        if not self._splatam_available:
            logger.warning(
                "SplaTAM / diff-gaussian-rasterization not found. "
                "Using depth-buffer fallback."
            )

        # Frame buffer used by fallback renderer
        self._last_depth: Optional[np.ndarray] = None
        self._last_rgb: Optional[np.ndarray] = None
        self._last_semantic: Optional[np.ndarray] = None
        self._frame_poses: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_frame(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        pose: np.ndarray,
        semantic: Optional[np.ndarray] = None,
        frame_id: int = 0,
    ) -> None:
        """Add one RGBD frame to the map.

        Parameters
        ----------
        rgb : (H, W, 3) uint8
        depth : (H, W) float32 metres
        pose : (4, 4) float64
        semantic : (H, W) uint8, optional
        frame_id : int
        """
        t0 = time.perf_counter()

        if self._splatam_available:
            self._splatam_ingest(rgb, depth, pose, frame_id)
        else:
            self._fallback_ingest(rgb, depth, pose, semantic)

        self.stats.total_frames_ingested += 1
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        alpha = 0.1
        self.stats.avg_ingest_ms = (
            alpha * elapsed_ms + (1 - alpha) * self.stats.avg_ingest_ms
        )
        self.stats.window_centre = pose[:3, 3].copy()

    def render(
        self, pose: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Render depth and colour from the current map at *pose*.

        Returns
        -------
        rendered_depth : (H, W) float32 metres
        rendered_rgb   : (H, W, 3) uint8
        """
        t0 = time.perf_counter()

        if self._splatam_available and self._means3D is not None:
            rendered_depth, rendered_rgb = self._splatam_render(pose)
        else:
            rendered_depth, rendered_rgb = self._fallback_render(pose)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        alpha = 0.1
        self.stats.avg_render_ms = (
            alpha * elapsed_ms + (1 - alpha) * self.stats.avg_render_ms
        )
        return rendered_depth, rendered_rgb

    @property
    def is_warmed_up(self) -> bool:
        return self.stats.total_frames_ingested >= self.warmup_frames

    # ------------------------------------------------------------------
    # SplaTAM integration
    # ------------------------------------------------------------------

    @staticmethod
    def _try_import_splatam() -> bool:
        try:
            import diff_gaussian_rasterization  # noqa: F401
            return True
        except ImportError:
            return False

    def _splatam_ingest(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        pose: np.ndarray,
        frame_id: int,
    ) -> None:
        """Full 3DGS ingest: unproject RGBD → initialise Gaussians → optimise."""
        import diff_gaussian_rasterization as dgr  # noqa: F811

        rgb_t = (
            torch.tensor(rgb, dtype=torch.float32, device=self.device).permute(2, 0, 1)
            / 255.0
        )
        depth_t = torch.tensor(depth, dtype=torch.float32, device=self.device)
        pose_t = torch.tensor(pose, dtype=torch.float32, device=self.device)

        # Unproject valid depth pixels to 3D
        valid_mask = (depth_t > 0.1) & (depth_t < 80.0)
        v_coords, u_coords = torch.where(valid_mask)
        z = depth_t[v_coords, u_coords]
        fx, fy = self.intrinsics[0, 0], self.intrinsics[1, 1]
        cx, cy = self.intrinsics[0, 2], self.intrinsics[1, 2]
        x = (u_coords.float() - cx) * z / fx
        y = (v_coords.float() - cy) * z / fy
        pts_cam = torch.stack([x, y, z], dim=-1)  # (M, 3)

        # Transform to world frame
        R = pose_t[:3, :3]
        t = pose_t[:3, 3]
        pts_world = (R @ pts_cam.T).T + t  # (M, 3)

        # Subsample to control density
        stride = max(1, len(pts_world) // 10000)
        pts_world = pts_world[::stride]
        colors_new = rgb_t[:, v_coords[::stride], u_coords[::stride]].T  # (M', 3)

        if self._means3D is None:
            # First frame — initialise
            self._means3D = pts_world.detach().requires_grad_(True)
            self._colors = colors_new.detach().requires_grad_(True)
            self._opacities = torch.full(
                (len(pts_world), 1), 0.5, device=self.device, requires_grad=True
            )
            self._scales = torch.full(
                (len(pts_world), 3), -3.0, device=self.device, requires_grad=True
            )  # log-scale
            self._rotations = torch.zeros(
                (len(pts_world), 4), device=self.device, requires_grad=True
            )
            self._rotations.data[:, 0] = 1.0  # identity quaternion
        else:
            # Append new Gaussians
            self._means3D = torch.nn.Parameter(
                torch.cat([self._means3D.data, pts_world.detach()], dim=0)
            )
            self._colors = torch.nn.Parameter(
                torch.cat([self._colors.data, colors_new.detach()], dim=0)
            )
            self._opacities = torch.nn.Parameter(
                torch.cat(
                    [
                        self._opacities.data,
                        torch.full(
                            (len(pts_world), 1), 0.5, device=self.device
                        ),
                    ],
                    dim=0,
                )
            )
            self._scales = torch.nn.Parameter(
                torch.cat(
                    [
                        self._scales.data,
                        torch.full(
                            (len(pts_world), 3), -3.0, device=self.device
                        ),
                    ],
                    dim=0,
                )
            )
            self._rotations = torch.nn.Parameter(
                torch.cat(
                    [
                        self._rotations.data,
                        torch.cat(
                            [
                                torch.ones(len(pts_world), 1, device=self.device),
                                torch.zeros(len(pts_world), 3, device=self.device),
                            ],
                            dim=-1,
                        ),
                    ],
                    dim=0,
                )
            )

        # Sliding-window prune: remove Gaussians far from current position
        centre = pose_t[:3, 3]
        dists = torch.norm(self._means3D.data - centre.unsqueeze(0), dim=-1)
        keep = dists < self.sliding_window_m
        # Also prune low-opacity
        keep = keep & (self._opacities.data.squeeze(-1) > self.prune_opacity_thr)
        if keep.sum() < self._means3D.shape[0]:
            self._means3D = torch.nn.Parameter(self._means3D.data[keep])
            self._colors = torch.nn.Parameter(self._colors.data[keep])
            self._opacities = torch.nn.Parameter(self._opacities.data[keep])
            self._scales = torch.nn.Parameter(self._scales.data[keep])
            self._rotations = torch.nn.Parameter(self._rotations.data[keep])

        # Cap total Gaussians
        if self._means3D.shape[0] > self.max_gaussians:
            idx = torch.randperm(self._means3D.shape[0], device=self.device)[
                : self.max_gaussians
            ]
            self._means3D = torch.nn.Parameter(self._means3D.data[idx])
            self._colors = torch.nn.Parameter(self._colors.data[idx])
            self._opacities = torch.nn.Parameter(self._opacities.data[idx])
            self._scales = torch.nn.Parameter(self._scales.data[idx])
            self._rotations = torch.nn.Parameter(self._rotations.data[idx])

        self.stats.num_gaussians = self._means3D.shape[0]

        # Quick optimisation step (2 iterations)
        self._optimise_step(rgb_t, depth_t, pose_t, n_iters=2)

    def _optimise_step(
        self,
        rgb_target: torch.Tensor,
        depth_target: torch.Tensor,
        pose: torch.Tensor,
        n_iters: int = 2,
    ) -> None:
        """Run a few gradient steps to refine Gaussian parameters."""
        lr = self.cfg.get("learning_rate", {})
        params = [
            {"params": [self._means3D], "lr": lr.get("position", 0.00016)},
            {"params": [self._opacities], "lr": lr.get("opacity", 0.05)},
            {"params": [self._scales], "lr": lr.get("scaling", 0.005)},
            {"params": [self._rotations], "lr": lr.get("rotation", 0.001)},
            {"params": [self._colors], "lr": lr.get("color", 0.0025)},
        ]
        optimizer = torch.optim.Adam(params)

        for _ in range(n_iters):
            optimizer.zero_grad()
            rendered_rgb, rendered_depth = self._rasterize(pose)

            # Photometric + depth loss
            loss_rgb = torch.nn.functional.l1_loss(rendered_rgb, rgb_target)
            valid = depth_target > 0.1
            if valid.any():
                loss_depth = torch.nn.functional.l1_loss(
                    rendered_depth[valid], depth_target[valid]
                )
            else:
                loss_depth = torch.tensor(0.0, device=self.device)

            loss = loss_rgb + 0.5 * loss_depth
            loss.backward()
            optimizer.step()

    def _rasterize(
        self, pose: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Rasterize current Gaussians from given pose using diff-gaussian-rasterization."""
        from diff_gaussian_rasterization import (
            GaussianRasterizationSettings,
            GaussianRasterizer,
        )

        # Camera extrinsics
        R = pose[:3, :3]
        T = pose[:3, 3]
        # World-to-camera
        R_inv = R.T
        T_inv = -R_inv @ T

        fx = self.intrinsics[0, 0].item()
        fy = self.intrinsics[1, 1].item()

        tanfovx = self.W / (2.0 * fx)
        tanfovy = self.H / (2.0 * fy)

        # Build view and projection matrices
        viewmatrix = torch.eye(4, device=self.device)
        viewmatrix[:3, :3] = R_inv
        viewmatrix[:3, 3] = T_inv

        projmatrix = viewmatrix.clone()  # perspective handled by rasteriser

        cam_pos = pose[:3, 3]

        raster_settings = GaussianRasterizationSettings(
            image_height=self.H,
            image_width=self.W,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.zeros(3, device=self.device),
            scale_modifier=1.0,
            viewmatrix=viewmatrix.T.contiguous(),
            projmatrix=projmatrix.T.contiguous(),
            sh_degree=0,
            campos=cam_pos,
            prefiltered=False,
            debug=False,
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # Activate scales / opacities
        scales = torch.exp(self._scales)
        opacities = torch.sigmoid(self._opacities)

        rendered_image, radii = rasterizer(
            means3D=self._means3D,
            means2D=torch.zeros_like(self._means3D[:, :2]),
            shs=None,
            colors_precomp=self._colors,
            opacities=opacities,
            scales=scales,
            rotations=self._rotations,
            cov3D_precomp=None,
        )

        # rendered_image: (3, H, W)
        # Depth: use alpha-weighted z-buffer from means (approximation)
        with torch.no_grad():
            pts_cam = (R_inv @ self._means3D.T).T + T_inv  # (N, 3)
            z_vals = pts_cam[:, 2].clamp(min=0.01)

        # Simple depth compositing: render depth using same rasteriser
        depth_colors = z_vals.unsqueeze(-1).repeat(1, 3)
        rendered_depth_img, _ = rasterizer(
            means3D=self._means3D,
            means2D=torch.zeros_like(self._means3D[:, :2]),
            shs=None,
            colors_precomp=depth_colors,
            opacities=opacities,
            scales=scales,
            rotations=self._rotations,
            cov3D_precomp=None,
        )
        rendered_depth = rendered_depth_img[0]  # (H, W)

        return rendered_image, rendered_depth

    def _splatam_render(
        self, pose: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Render from current map at given pose."""
        pose_t = torch.tensor(pose, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            rgb_t, depth_t = self._rasterize(pose_t)

        rendered_rgb = (
            (rgb_t.clamp(0, 1) * 255).byte().permute(1, 2, 0).cpu().numpy()
        )
        rendered_depth = depth_t.cpu().numpy()
        return rendered_depth, rendered_rgb

    # ------------------------------------------------------------------
    # Fallback (no SplaTAM / no GPU)
    # ------------------------------------------------------------------

    def _fallback_ingest(
        self,
        rgb: np.ndarray,
        depth: np.ndarray,
        pose: np.ndarray,
        semantic: Optional[np.ndarray] = None,
    ) -> None:
        """Store the latest frame as a trivial 'map'."""
        self._last_rgb = rgb.copy()
        self._last_depth = depth.copy()
        self._last_semantic = semantic.copy() if semantic is not None else None
        self._frame_poses.append(pose.copy())
        self.stats.num_gaussians = 0
        self.stats.total_frames_ingested += 1

    def _fallback_render(
        self, pose: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return last ingested frame as the 'rendered' map view."""
        if self._last_depth is not None:
            return self._last_depth.copy(), self._last_rgb.copy()
        # Empty map
        return (
            np.zeros((self.H, self.W), dtype=np.float32),
            np.zeros((self.H, self.W, 3), dtype=np.uint8),
        )
