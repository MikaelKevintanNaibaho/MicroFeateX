"""
Lightweight evaluation module for MicroFeatEX.

Provides functions for running MegaDepth-1500 evaluation during training.
"""

from __future__ import annotations

import os
import json
import copy

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from microfeatex.utils.logger import get_logger

logger = get_logger(__name__)

# Try to import poselib
try:
    import poselib

    HAS_POSELIB = True
except ImportError:
    HAS_POSELIB = False


class MegaDepthSubset(Dataset):
    """Subset of MegaDepth-1500 for fast evaluation during training."""

    def __init__(self, json_file: str, root_dir: str, max_pairs: int = 100):
        """Initialize dataset.

        Args:
            json_file: Path to megadepth_1500.json.
            root_dir: Path to megadepth_test_1500 folder.
            max_pairs: Maximum number of pairs to evaluate.
        """
        with open(json_file, "r") as f:
            data = json.load(f)

        # Take evenly spaced subset
        step = max(1, len(data) // max_pairs)
        self.data = data[::step][:max_pairs]
        self.root_dir = root_dir

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        data = copy.deepcopy(self.data[idx])

        h1, w1 = data["size0_hw"]
        h2, w2 = data["size1_hw"]

        img0_path = os.path.join(self.root_dir, data["pair_names"][0])
        img1_path = os.path.join(self.root_dir, data["pair_names"][1])

        image0 = cv2.resize(cv2.imread(img0_path), (w1, h1))
        image1 = cv2.resize(cv2.imread(img1_path), (w2, h2))

        data["image0"] = torch.from_numpy(
            cv2.cvtColor(image0, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        data["image1"] = torch.from_numpy(
            cv2.cvtColor(image1, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ).permute(2, 0, 1)

        for k, v in data.items():
            if k not in (
                "dataset_name",
                "scene_id",
                "pair_id",
                "pair_names",
                "size0_hw",
                "size1_hw",
                "image0",
                "image1",
            ):
                data[k] = torch.tensor(np.array(v, dtype=np.float32))

        return data


def extract_and_match(
    model: torch.nn.Module,
    img0: torch.Tensor,
    img1: torch.Tensor,
    device: str,
    top_k: int = 2000,
    thresh: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract features and match two images.

    Args:
        model: MicroFeatEX model.
        img0: Image tensor [1, 3, H, W].
        img1: Image tensor [1, 3, H, W].
        device: Device string.
        top_k: Maximum keypoints.
        thresh: Detection threshold.

    Returns:
        Tuple of (mkpts0, mkpts1) matched keypoint arrays.
    """
    with torch.no_grad():
        out0 = model(img0.to(device))
        out1 = model(img1.to(device))

    # Extract heatmaps and descriptors
    h0 = out0["heatmap"].squeeze().cpu().numpy()
    h1 = out1["heatmap"].squeeze().cpu().numpy()
    d0 = out0["descriptors"].squeeze()
    d1 = out1["descriptors"].squeeze()

    # Simple NMS keypoint detection
    def detect_kpts(heatmap: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(heatmap, kernel)
        kps_mask = (heatmap == dilated) & (heatmap > thresh)
        y_idxs, x_idxs = np.where(kps_mask)
        scores = heatmap[y_idxs, x_idxs]
        ids = np.argsort(scores)[::-1][:top_k]
        return x_idxs[ids], y_idxs[ids], scores[ids]

    x0, y0, _ = detect_kpts(h0)
    x1, y1, _ = detect_kpts(h1)

    if len(x0) < 10 or len(x1) < 10:
        return np.zeros((0, 2)), np.zeros((0, 2))

    # Sample descriptors
    def sample_desc(desc: torch.Tensor, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        D, H_feat, W_feat = desc.shape
        desc_list = []
        for xi, yi in zip(x, y):
            gx = int(np.clip(xi / 8.0, 0, W_feat - 1))
            gy = int(np.clip(yi / 8.0, 0, H_feat - 1))
            desc_list.append(desc[:, gy, gx].cpu().numpy())
        return np.stack(desc_list) if desc_list else np.zeros((0, D))

    desc0 = sample_desc(d0, x0, y0)
    desc1 = sample_desc(d1, x1, y1)

    # Match
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc0.astype(np.float32), desc1.astype(np.float32))

    if len(matches) == 0:
        return np.zeros((0, 2)), np.zeros((0, 2))

    mkpts0 = np.array([[x0[m.queryIdx], y0[m.queryIdx]] for m in matches])
    mkpts1 = np.array([[x1[m.trainIdx], y1[m.trainIdx]] for m in matches])

    return mkpts0, mkpts1


def compute_pose_error(
    pts0: np.ndarray,
    pts1: np.ndarray,
    K0: np.ndarray,
    K1: np.ndarray,
    T_0to1: np.ndarray,
    ransac_thr: float = 2.5,
) -> float:
    """Compute pose error for matched points.

    Returns:
        Maximum of translation and rotation error in degrees.
    """
    if len(pts0) < 8:
        return np.inf

    try:
        if HAS_POSELIB:
            px, py = K0[0, 2], K0[1, 2]
            fx, fy = K0[0, 0], K0[1, 1]
            cam0 = {
                "model": "PINHOLE",
                "width": int(2 * px),
                "height": int(2 * py),
                "params": [fx, fy, px, py],
            }
            px, py = K1[0, 2], K1[1, 2]
            fx, fy = K1[0, 0], K1[1, 1]
            cam1 = {
                "model": "PINHOLE",
                "width": int(2 * px),
                "height": int(2 * py),
                "params": [fx, fy, px, py],
            }

            M, _ = poselib.estimate_relative_pose(
                pts0,
                pts1,
                cam0,
                cam1,
                {"max_epipolar_error": ransac_thr, "min_iterations": 20},
            )
            R, t = M.R, M.t
        else:
            E, mask = cv2.findEssentialMat(
                pts0, pts1, K0, method=cv2.RANSAC, threshold=ransac_thr
            )
            if E is None:
                return np.inf
            _, R, t, _ = cv2.recoverPose(E, pts0, pts1, K0, mask=mask)
            t = t.squeeze()

        # Compute errors
        t_gt = T_0to1[:3, 3]
        n = np.linalg.norm(t) * np.linalg.norm(t_gt)
        if n < 1e-8:
            t_err = 0
        else:
            t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
            t_err = min(t_err, 180 - t_err)

        R_gt = T_0to1[:3, :3]
        cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
        R_err = np.rad2deg(np.abs(np.arccos(np.clip(cos, -1.0, 1.0))))

        return max(t_err, R_err)

    except Exception:
        return np.inf


def error_auc(errors: list[float], thresholds: list[int] = [5, 10, 20]) -> dict:
    """Compute AUC for pose errors."""
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = {}
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index - 1]]
        x = errors[:last_index] + [thr]
        aucs[f"AUC@{thr}"] = np.trapz(y, x) / thr

    return aucs


@torch.no_grad()
def run_evaluation(
    model: torch.nn.Module,
    image_dir: str,
    json_path: str = "assets/megadepth_1500.json",
    device: str = "cuda",
    num_pairs: int = 100,
    top_k: int = 2000,
) -> dict[str, float]:
    """Run MegaDepth evaluation on model.

    Args:
        model: MicroFeatEX model (already on device).
        image_dir: Path to MegaDepth images (containing Undistorted_SfM/).
        json_path: Path to megadepth_1500.json metadata.
        device: Device string.
        num_pairs: Number of pairs to evaluate.
        top_k: Maximum keypoints per image.

    Returns:
        Dictionary with AUC@5, AUC@10, AUC@20 values.
    """
    if not os.path.exists(json_path):
        logger.warning(f"MegaDepth JSON not found at {json_path}")
        return {"AUC@5": 0.0, "AUC@10": 0.0, "AUC@20": 0.0}

    if not os.path.exists(image_dir):
        logger.warning(f"MegaDepth images not found at {image_dir}")
        return {"AUC@5": 0.0, "AUC@10": 0.0, "AUC@20": 0.0}

    model.eval()
    dataset = MegaDepthSubset(json_path, image_dir, max_pairs=num_pairs)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    errors = []
    for data in loader:
        img0 = data["image0"]
        img1 = data["image1"]

        mkpts0, mkpts1 = extract_and_match(model, img0, img1, device, top_k=top_k)

        # Rescale keypoints
        mkpts0 = mkpts0 * data["scale0"].numpy()
        mkpts1 = mkpts1 * data["scale1"].numpy()

        K0 = data["K0"].numpy()[0]
        K1 = data["K1"].numpy()[0]
        T_0to1 = data["T_0to1"].numpy()[0]

        err = compute_pose_error(mkpts0, mkpts1, K0, K1, T_0to1)
        errors.append(err)

    aucs = error_auc(errors)
    model.train()

    return aucs
