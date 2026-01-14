from typing import NamedTuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.augmentation as K
import kornia.geometry.transform as G
import kornia.color
import numpy as np

__all__ = [
    "AugmentationPipe",
    "AugmentationResult",
    "generate_random_homography",
    "generate_random_tps",
]


class AugmentationResult(NamedTuple):
    """Structured result from AugmentationPipe.forward()."""

    img1: torch.Tensor  # Source images [B, C, H, W]
    img2: torch.Tensor  # Warped images [B, C, H, W]
    H_mat: torch.Tensor  # Homography matrix [B, 3, 3]
    mask: torch.Tensor  # Valid pixel mask [B, H, W]
    coords_map: torch.Tensor  # Dense correspondence map [B, 2, H, W]


def generate_random_homography(
    shape: tuple[int, int], difficulty: float = 0.3
) -> np.ndarray:
    """
    Generates a random homography matrix with controlled difficulty.
    Logic adapted from XFeat (CVPR 2024).

    Args:
        shape (tuple): Input image shape (H, W).
        difficulty (float): Difficulty multiplier (approx 0.1 to 0.5).

    Returns:
        np.array: 3x3 Homography matrix.
    """
    h, w = shape

    # Random In-plane Rotation
    theta = np.radians(np.random.uniform(-30, 30))
    c, s = np.cos(theta), np.sin(theta)

    #  Random Scale
    scale_x, scale_y = np.random.uniform(0.35, 1.2, 2)

    # Random Translation
    # Translation to center
    tx, ty = -w / 2.0, -h / 2.0
    # Random offset
    txn, tyn = np.random.normal(0, 120.0 * difficulty, 2)

    # Affine Shear
    sx, sy = np.random.normal(0, 0.6 * difficulty, 2)

    # Projective
    p1, p2 = np.random.normal(0, 0.006 * difficulty, 2)

    # Compose Matrices
    H_t = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])  # Center
    H_r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])  # Rotation
    H_a = np.array([[1, sy, 0], [sx, 1, 0], [0, 0, 1]])  # Affine
    H_p = np.array([[1, 0, 0], [0, 1, 0], [p1, p2, 1]])  # Projective
    H_s = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])  # Scale
    H_b = np.array([[1, 0, -tx + txn], [0, 1, -ty + tyn], [0, 0, 1]])  # Back + Offset

    # Order: H = H_b * H_s * H_p * H_a * H_r * H_t
    H = H_b @ H_s @ H_p @ H_a @ H_r @ H_t

    return H


def generate_random_tps(
    shape: tuple[int, int],
    grid: tuple[int, int] = (8, 6),
    difficulty: float = 0.3,
    prob: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates parameters for Thin Plate Spline (TPS) warping.
    """
    h, w = shape

    # Calculate grid point counts (N+1 points for N cells)
    grid_h = grid[0] + 1
    grid_w = grid[1] + 1

    # Use linspace for stable grid generation (Avoids arange float errors)
    ys = torch.linspace(0, h, steps=grid_h)
    xs = torch.linspace(0, w, steps=grid_w)

    src = torch.dstack(torch.meshgrid(ys, xs, indexing="ij"))

    # Generate random offsets
    offsets = torch.rand(grid_h, grid_w, 2) - 0.5

    # Scale offsets relative to cell size
    sh, sw = h / grid[0], w / grid[1]
    offsets *= torch.tensor([sh / 2, sw / 2]).view(1, 1, 2) * min(
        0.97, 2.0 * difficulty
    )

    # Apply offsets with probability
    if np.random.uniform() < prob:
        dst = src + offsets
    else:
        dst = src

    # Normalize to [-1, 1] for Kornia
    src = src.view(1, -1, 2)
    dst = dst.view(1, -1, 2)

    src = (src / torch.tensor([h, w]).view(1, 1, 2)) * 2 - 1.0
    dst = (dst / torch.tensor([h, w]).view(1, 1, 2)) * 2 - 1.0

    # Compute TPS weights
    weights, A = G.get_tps_transform(dst, src)

    return src, weights, A


class AugmentationPipe(nn.Module):
    """
    Augmentation Pipeline adapting XFeat strategies for MicroFeatEX.
    Handles:
    - Homography Warping
    - Thin Plate Spline (TPS) Deformation
    - Smart Border Handling (Texture Filling)
    - Photometric Augmentation
    """

    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.config = config

        # Configuration
        aug_cfg = config.get("augmentation", {})
        self.warp_res = tuple(aug_cfg.get("warp_resolution", [1200, 900]))  # (W, H)
        self.out_res = tuple(aug_cfg.get("out_resolution", [640, 480]))  # (W, H)
        self.sides_crop = aug_cfg.get("sides_crop", 0.2)

        self.enable_photo = aug_cfg.get("photometric", True)
        self.enable_geom = aug_cfg.get("geometric", True)
        self.use_tps = aug_cfg.get("use_tps", False)
        self.tps_prob = aug_cfg.get("tps_prob", 0.5)
        self.difficulty = aug_cfg.get("difficulty", 0.3)

        # Precompute Cropping Indices
        W, H = self.warp_res
        self.low_h = int(H * self.sides_crop)
        self.low_w = int(W * self.sides_crop)
        self.high_h = int(H * (1.0 - self.sides_crop))
        self.high_w = int(W * (1.0 - self.sides_crop))

        # Scaling factor from Crop -> Output
        crop_h = self.high_h - self.low_h
        crop_w = self.high_w - self.low_w
        self.scale_x = self.out_res[0] / crop_w
        self.scale_y = self.out_res[1] / crop_h

        # Photometric Pipeline
        self.photo_aug = nn.Identity()
        if self.enable_photo:
            self.photo_aug = K.ImageSequential(
                K.ColorJitter(0.15, 0.15, 0.15, 0.15, p=1.0),
                K.RandomEqualize(p=0.4),
                K.RandomGaussianBlur((7, 7), (2.0, 2.0), p=0.3),
            )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> AugmentationResult:
        """
        Args:
            x: [B, C, H, W] Input images (normalized 0-1)
        Returns:
            img1: Source crop [B, C, H_out, W_out]
            img2: Warped crop [B, C, H_out, W_out]
            H: Homography matrix [B, H_out, W_out]
            mask: Valid pixel mask [B, H_out, W_out]
            coords_map: Dense correspondence map [B, 2, H_out, W_out]
                        Contains coordinates of img1 for every pixel in img2.
        """
        x = x.to(self.device)
        B, C, H, W = x.shape
        difficulty = self.difficulty if self.enable_geom else 0.0

        # Initialize Coordinate Grid & Mask
        # Grid contains (x, y) coordinates of the original image
        grid_y, grid_x = torch.meshgrid(
            torch.arange(H, device=self.device),
            torch.arange(W, device=self.device),
            indexing="ij",
        )
        # [B, 2, H, W] -> Channels are X, Y
        grid = (
            torch.stack([grid_x, grid_y], dim=0).float().unsqueeze(0).repeat(B, 1, 1, 1)
        )

        # Ones mask to track valid pixels safely
        ones_mask = torch.ones((B, 1, H, W), device=self.device, dtype=torch.float32)

        # --- Geometric: Homography ---
        H_list = [
            torch.from_numpy(generate_random_homography((H, W), difficulty)).float()
            for _ in range(B)
        ]
        H_mat = torch.stack(H_list).to(self.device)

        # Warp entire image first
        img2_full = G.warp_perspective(x, H_mat, dsize=(H, W), padding_mode="zeros")
        grid = G.warp_perspective(grid, H_mat, dsize=(H, W), padding_mode="zeros")
        ones_mask = G.warp_perspective(
            ones_mask, H_mat, dsize=(H, W), padding_mode="zeros"
        )

        # --- Crop to Remove Borders ---
        # "Zoom in" to central region to avoid sampling pure black borders
        img1 = x[..., self.low_h : self.high_h, self.low_w : self.high_w]
        img2 = img2_full[..., self.low_h : self.high_h, self.low_w : self.high_w]
        grid = grid[..., self.low_h : self.high_h, self.low_w : self.high_w]
        ones_mask = ones_mask[..., self.low_h : self.high_h, self.low_w : self.high_w]

        # --- Geometric: TPS ---
        if self.use_tps and self.enable_geom:
            src_list, w_list, A_list = [], [], []
            curr_h, curr_w = img2.shape[2], img2.shape[3]

            for _ in range(B):
                s, w_tps, a = generate_random_tps(
                    (curr_h, curr_w), difficulty=difficulty, prob=self.tps_prob
                )
                src_list.append(s.to(self.device))
                w_list.append(w_tps.to(self.device))
                A_list.append(a.to(self.device))

            src_tps = torch.cat(src_list)
            weights_tps = torch.cat(w_list)
            A_tps = torch.cat(A_list)

            # Apply TPS to image, grid, and mask
            img2 = G.warp_image_tps(img2, src_tps, weights_tps, A_tps)
            grid = G.warp_image_tps(grid, src_tps, weights_tps, A_tps)
            ones_mask = G.warp_image_tps(ones_mask, src_tps, weights_tps, A_tps)

        # --- Resize to Output Resolution ---
        img1 = F.interpolate(
            img1, size=self.out_res[::-1], mode="bilinear", align_corners=False
        )
        img2 = F.interpolate(
            img2, size=self.out_res[::-1], mode="bilinear", align_corners=False
        )
        grid = F.interpolate(
            grid, size=self.out_res[::-1], mode="bilinear", align_corners=False
        )
        ones_mask = F.interpolate(
            ones_mask, size=self.out_res[::-1], mode="bilinear", align_corners=False
        )

        # Compute final valid mask
        mask = ones_mask.squeeze(1) > 0.9  # [B, H, W]

        # Handle Background filling
        # Expand mask for image channels
        mask_expanded = mask.unsqueeze(1).expand(-1, C, -1, -1)

        roll_idx = 2 if self.use_tps else 1
        background = torch.roll(img1, roll_idx, dims=0)
        img2 = torch.where(mask_expanded, img2, background)

        # Currently 'grid' holds coordinates in the ORIGINAL image system (0..W, 0..H).
        # We want to know where these points are in 'img1' (which is cropped & resized).
        # Mapping: P1_coord = (Orig_coord - Crop_Offset) * Scale
        # Channel 0 is X, Channel 1 is Y
        grid[:, 0, ...] = (grid[:, 0, ...] - self.low_w) * self.scale_x
        grid[:, 1, ...] = (grid[:, 1, ...] - self.low_h) * self.scale_y

        # This grid now tells us: For a pixel in img2, what is its coordinate in img1?
        coords_map = grid

        # --- 8. Photometric Augmentation ---
        if self.enable_photo:
            is_gray = C == 1
            if is_gray:
                img1 = img1.repeat(1, 3, 1, 1)
                img2 = img2.repeat(1, 3, 1, 1)

            img1 = self.photo_aug(img1)
            img2 = self.photo_aug(img2)

            # (Gaussian noise / shadow logic omitted for brevity, keep original if desired)
            if np.random.uniform() > 0.5:
                noise = torch.randn_like(img2) * (10 / 255.0)
                img2 = torch.clamp(img2 + noise, 0, 1)

            if is_gray:
                img1 = kornia.color.rgb_to_grayscale(img1)
                img2 = kornia.color.rgb_to_grayscale(img2)

        return AugmentationResult(img1, img2, H_mat, mask, coords_map)

    def warp_points(self, H: torch.Tensor, pts: torch.Tensor) -> torch.Tensor:
        """
        Maps keypoints from the Source image frame to the Target image frame.
        Accounts for: Scale -> Crop -> Homography -> Crop -> Scale.

        Args:
            H: [3, 3] Homography matrix (Original scale)
            pts: [N, 2] Keypoints in Source Output (Final) coordinates

        Returns:
            [N, 2] Keypoints in Target Output (Final) coordinates
        """
        # Ensure H and pts are on the same device
        H = H.to(pts.device)

        # Project Source Points back to Crop Coordinates
        # Scale factor: Output -> Crop
        crop_h = self.high_h - self.low_h
        crop_w = self.high_w - self.low_w
        scale_x = crop_w / self.out_res[0]
        scale_y = crop_h / self.out_res[1]

        scale = torch.tensor([scale_x, scale_y], device=pts.device).view(1, 2)
        offset = torch.tensor([self.low_w, self.low_h], device=pts.device).view(1, 2)

        # Scale up and add offset (Output -> Original Large Coords)
        pts_orig = (pts * scale) + offset

        # Apply Homography
        pts_homo = torch.cat(
            [pts_orig, torch.ones(pts.shape[0], 1, device=pts.device)], dim=1
        )
        pts_warped = (H @ pts_homo.T).T
        pts_warped = pts_warped[:, :2] / (pts_warped[:, 2:3] + 1e-8)

        # Project Target Points to Output Coordinates
        pts_final = (pts_warped - offset) / scale

        return pts_final
