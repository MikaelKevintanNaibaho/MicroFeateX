import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import kornia.augmentation as K
from kornia.geometry.transform import (
    warp_perspective,
    get_tps_transform,
    warp_image_tps,
    warp_points_tps,
)
import numpy as np


def generate_random_homography(shape, difficulty=0.3):
    """
    Generate random homography matrix with controlled difficulty.
    Based on XFeat implementation.

    Args:
        shape: (H, W) tuple
        difficulty: float controlling transformation strength (0.1-0.5)
    """
    h, w = shape

    # Random in-plane rotation
    theta = np.radians(np.random.uniform(-30, 30))

    # Random scale in both x and y
    scale_x, scale_y = np.random.uniform(0.35, 1.2, 2)

    # Translation to origin and random offset
    tx, ty = -w / 2.0, -h / 2.0
    txn, tyn = np.random.normal(0, 120.0 * difficulty, 2)

    c, s = np.cos(theta), np.sin(theta)

    # Affine shear coefficients
    sx, sy = np.random.normal(0, 0.6 * difficulty, 2)

    # Projective coefficients
    p1, p2 = np.random.normal(0, 0.006 * difficulty, 2)

    # Build homography from parameterizations
    H_t = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])  # translate to origin
    H_r = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])  # rotation
    H_a = np.array([[1, sy, 0], [sx, 1, 0], [0, 0, 1]])  # affine shear
    H_p = np.array([[1, 0, 0], [0, 1, 0], [p1, p2, 1]])  # projective
    H_s = np.array([[scale_x, 0, 0], [0, scale_y, 0], [0, 0, 1]])  # scale
    H_b = np.array([[1, 0, -tx + txn], [0, 1, -ty + tyn], [0, 0, 1]])  # translate back

    # Compose: H = H_b * H_s * H_p * H_a * H_r * H_t
    H = H_b @ H_s @ H_p @ H_a @ H_r @ H_t

    return H


def generate_random_tps(shape, grid=(8, 6), difficulty=0.3, prob=0.5):
    """
    Generate random TPS (Thin Plate Spline) transformation for non-rigid warps.

    Args:
        shape: (H, W) tuple
        grid: grid resolution for control points
        difficulty: transformation strength
        prob: probability of applying deformation

    Returns:
        src: source control points (normalized to [-1, 1])
        weights: TPS weights
        A: TPS affine matrix
    """
    h, w = shape
    sh, sw = h / grid[0], w / grid[1]

    # Create regular grid of control points
    src = torch.dstack(
        torch.meshgrid(
            torch.arange(0, h + sh, sh), torch.arange(0, w + sw, sw), indexing="ij"
        )
    )

    # Generate random offsets
    offsets = torch.rand(grid[0] + 1, grid[1] + 1, 2) - 0.5
    offsets *= torch.tensor([sh / 2, sw / 2]).view(1, 1, 2) * min(
        0.97, 2.0 * difficulty
    )

    # Apply offsets with probability
    dst = src + offsets if np.random.uniform() < prob else src

    # Reshape and normalize to [-1, 1]
    src = src.view(1, -1, 2)
    dst = dst.view(1, -1, 2)
    src = (src / torch.tensor([h, w]).view(1, 1, 2)) * 2 - 1.0
    dst = (dst / torch.tensor([h, w]).view(1, 1, 2)) * 2 - 1.0

    # Compute TPS transformation
    weights, A = get_tps_transform(dst, src)

    return src, weights, A


class AugmentationPipe(nn.Module):
    """
    XFeat-style augmentation pipeline with:
    - Custom homography generation
    - Optional TPS warps
    - Side cropping to reduce invalid pixels
    - Intelligent handling of black borders
    - Photometric augmentations
    """

    def __init__(self, config, device):
        super().__init__()
        self.device = device
        self.config = config

        # Get parameters from config
        aug_config = config["augmentation"]
        self.warp_resolution = tuple(aug_config["warp_resolution"])  # (W, H)
        self.out_resolution = tuple(
            aug_config.get("out_resolution", (640, 480))
        )  # (W, H)
        self.sides_crop = aug_config.get("sides_crop", 0.2)
        self.photometric = aug_config.get("photometric", True)
        self.geometric = aug_config.get("geometric", True)
        self.use_tps = aug_config.get("use_tps", False)
        self.tps_prob = aug_config.get("tps_prob", 0.5)
        self.difficulty = aug_config.get("difficulty", 0.3)

        # Photometric augmentation sequence
        photo_list = [
            K.ColorJitter(0.15, 0.15, 0.15, 0.15, p=1.0),
            K.RandomEqualize(p=0.4),
            K.RandomGaussianBlur((7, 7), (2.0, 2.0), p=0.3),
        ]

        if not self.photometric:
            photo_list = []

        self.photo_aug = K.ImageSequential(*photo_list)

        # Precompute cropping dimensions
        h, w = self.warp_resolution[1], self.warp_resolution[0]
        self.low_h = int(h * self.sides_crop)
        self.low_w = int(w * self.sides_crop)
        self.high_h = int(h * (1.0 - self.sides_crop))
        self.high_w = int(w * (1.0 - self.sides_crop))

    @torch.no_grad()
    def forward(self, x):
        """
        Apply augmentation pipeline to create synthetic image pairs.

        Args:
            x: [B, C, H, W] input images (normalized 0-1)

        Returns:
            img1: [B, C, H_out, W_out] original image (cropped & resized)
            img2: [B, C, H_out, W_out] warped image
            H_mat: [B, 3, 3] homography matrices
            mask: [B, H_out, W_out] valid pixel mask
        """
        x = x.to(self.device)
        B, C, H, W = x.shape

        # Override difficulty if geometric augmentation is disabled
        difficulty = self.difficulty if self.geometric else 0.0

        # ===== 1. Generate Random Homographies =====
        H_list = []
        for _ in range(B):
            H_mat_np = generate_random_homography((H, W), difficulty)
            H_list.append(torch.from_numpy(H_mat_np).float())
        H_mat = torch.stack(H_list).to(self.device)

        # ===== 2. Apply Homography Warp =====
        img2 = warp_perspective(x, H_mat, dsize=(H, W), padding_mode="zeros")

        # ===== 3. Crop Sides to Reduce Invalid Pixels =====
        img1_crop = x[..., self.low_h : self.high_h, self.low_w : self.high_w]
        img2_crop = img2[..., self.low_h : self.high_h, self.low_w : self.high_w]

        # ===== 4. Optional TPS Warping (Non-rigid Deformation) =====
        if self.use_tps:
            src_list, weights_list, A_list = [], [], []

            for _ in range(B):
                src, weights, A = generate_random_tps(
                    (img2_crop.shape[2], img2_crop.shape[3]),
                    grid=(8, 6),
                    difficulty=difficulty,
                    prob=self.tps_prob,
                )
                src_list.append(src.to(self.device))
                weights_list.append(weights.to(self.device))
                A_list.append(A.to(self.device))

            src_batch = torch.cat(src_list, dim=0)
            weights_batch = torch.cat(weights_list, dim=0)
            A_batch = torch.cat(A_list, dim=0)

            img2_crop = warp_image_tps(img2_crop, src_batch, weights_batch, A_batch)

        # ===== 5. Resize to Output Resolution =====
        img1_out = F.interpolate(
            img1_crop,
            size=self.out_resolution[::-1],
            mode="bilinear",
            align_corners=False,
        )
        img2_out = F.interpolate(
            img2_crop,
            size=self.out_resolution[::-1],
            mode="bilinear",
            align_corners=False,
        )

        # ===== 6. Generate Valid Pixel Mask =====
        # Mask is True where all channels are non-zero (valid pixels)
        mask = ~torch.all(img2_out == 0, dim=1, keepdim=True)
        mask_expanded = mask.expand(-1, C, -1, -1)

        # ===== 7. Fill Invalid Regions with Texture from Batch =====
        # Use rolled version of img1 as background texture
        roll_amount = 1 if not self.use_tps else 2
        img1_shifted = torch.roll(img1_out, roll_amount, dims=0)
        img2_out = torch.where(mask_expanded, img2_out, img1_shifted)
        mask = mask.squeeze(1)  # [B, H, W]

        # ===== 8. Apply Photometric Augmentations =====
        if self.photometric:
            # Handle grayscale images for photometric augmentation
            is_gray = C == 1

            if is_gray:
                img1_rgb = img1_out.repeat(1, 3, 1, 1)
                img2_rgb = img2_out.repeat(1, 3, 1, 1)
            else:
                img1_rgb = img1_out
                img2_rgb = img2_out

            # Apply color augmentations
            img1_aug = self.photo_aug(img1_rgb)
            img2_aug = self.photo_aug(img2_rgb)

            # Convert back to grayscale if needed
            if is_gray:
                img1_out = kornia.color.rgb_to_grayscale(img1_aug)
                img2_out = kornia.color.rgb_to_grayscale(img2_aug)
            else:
                img1_out = img1_aug
                img2_out = img2_aug

            # Correlated Gaussian noise (50% probability)
            if np.random.uniform() > 0.5:
                h_out, w_out = img2_out.shape[2], img2_out.shape[3]
                noise = F.interpolate(
                    torch.randn_like(img2_out) * (10 / 255),
                    size=(h_out // 2, w_out // 2),
                )
                noise = F.interpolate(
                    noise, size=(h_out, w_out), mode="bicubic", align_corners=False
                )
                img2_out = torch.clamp(img2_out + noise, 0.0, 1.0)

            # Random shadows (40% probability)
            if np.random.uniform() > 0.6:
                h_out, w_out = img2_out.shape[2], img2_out.shape[3]
                shadow = (
                    torch.rand((B, 1, h_out // 64, w_out // 64), device=self.device)
                    * 1.3
                )
                shadow = torch.clamp(shadow, 0.25, 1.0)
                shadow = F.interpolate(
                    shadow, size=(h_out, w_out), mode="bicubic", align_corners=False
                )
                shadow = shadow.expand(-1, C, -1, -1)
                img2_out = torch.clamp(img2_out * shadow, 0.0, 1.0)

        return img1_out, img2_out, H_mat, mask

    def warp_points(self, H, pts):
        """
        Warp keypoints using homography matrix.

        Args:
            H: [3, 3] homography matrix
            pts: [N, 2] keypoint coordinates

        Returns:
            [N, 2] warped coordinates
        """
        # Adjust for cropping offset
        offset = torch.tensor([self.low_w, self.low_h], device=pts.device).float()

        # Scale factor from cropped to output resolution
        crop_h = self.high_h - self.low_h
        crop_w = self.high_w - self.low_w
        scale_h = self.out_resolution[1] / crop_h
        scale_w = self.out_resolution[0] / crop_w
        scale = torch.tensor([scale_w, scale_h], device=pts.device).float()

        # Convert to homogeneous coordinates
        pts_homo = torch.cat(
            [pts, torch.ones(pts.shape[0], 1, device=pts.device)], dim=1
        )

        # Apply homography
        pts_warped = (H @ pts_homo.T).T
        pts_warped = pts_warped[:, :2] / pts_warped[:, 2:3]

        # Apply cropping and scaling transformations
        pts_warped = (pts_warped - offset) * scale

        return pts_warped
