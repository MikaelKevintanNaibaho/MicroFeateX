"""
MicroFeatEX Pose Estimation Benchmark on MegaDepth-1500

Adapted from XFeat's benchmark:
https://github.com/verlab/accelerated_features

Usage:
    python3 scripts/eval_megadepth.py \
        --dataset-dir data/megadepth \
        --checkpoint checkpoints/lite_hadamard/last.pth \
        --config config/lite_hadamard.yaml
"""

import argparse
import os
import sys
import copy
import json

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from microfeatex.models.student import EfficientFeatureExtractor
from microfeatex.utils.config import load_config

# Try to import poselib for robust pose estimation
try:
    import poselib

    HAS_POSELIB = True
except ImportError:
    HAS_POSELIB = False
    print("Warning: poselib not installed. Using OpenCV RANSAC (slower, less stable).")
    print("Install with: pip install poselib")


class MegaDepth1500(Dataset):
    """
    MegaDepth-1500 benchmark dataset.

    Expects:
        - json_file: Path to megadepth_1500.json with pair metadata
        - root_dir: Path to megadepth_test_1500 folder with images
    """

    def __init__(self, json_file, root_dir):
        with open(json_file, "r") as f:
            self.data = json.load(f)

        self.root_dir = root_dir

        if not os.path.exists(self.root_dir):
            raise RuntimeError(
                f"Dataset {self.root_dir} does not exist!\n"
                "Download from: https://github.com/verlab/accelerated_features"
            )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = copy.deepcopy(self.data[idx])

        h1, w1 = data["size0_hw"]
        h2, w2 = data["size1_hw"]

        # Load and resize images
        img0_path = os.path.join(self.root_dir, data["pair_names"][0])
        img1_path = os.path.join(self.root_dir, data["pair_names"][1])

        image0 = cv2.resize(cv2.imread(img0_path), (w1, h1))
        image1 = cv2.resize(cv2.imread(img1_path), (w2, h2))

        # Convert to tensor [3, H, W] RGB
        data["image0"] = torch.from_numpy(
            cv2.cvtColor(image0, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ).permute(2, 0, 1)
        data["image1"] = torch.from_numpy(
            cv2.cvtColor(image1, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        ).permute(2, 0, 1)

        # Convert calibration data to tensors
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


class MicroFeatEXMatcher:
    """Wrapper for MicroFeatEX matching."""

    def __init__(
        self, checkpoint_path, config_path, device="cuda", top_k=2000, thresh=0.01
    ):
        self.device = device
        self.top_k = top_k
        self.thresh = thresh

        # Load config
        config = load_config(config_path)
        model_conf = config.get("model", {})

        # Build model
        self.model = EfficientFeatureExtractor(
            descriptor_dim=model_conf.get("descriptor_dim", 64),
            width_mult=model_conf.get("width_mult", 1.0),
            use_depthwise=model_conf.get("use_depthwise", True),
            use_hadamard=model_conf.get("use_hadamard", True),
        )

        # Load weights
        ckpt = torch.load(checkpoint_path, map_location=device)
        if "model_state" in ckpt:
            self.model.load_state_dict(ckpt["model_state"])
        else:
            self.model.load_state_dict(ckpt)

        self.model.to(device)
        self.model.eval()
        print(f"Loaded model from {checkpoint_path}")

    @torch.no_grad()
    def extract(self, img_tensor):
        """Extract keypoints and descriptors from image tensor [1, 3, H, W]."""
        out = self.model(img_tensor.to(self.device))

        heatmap = out["heatmap"].squeeze().cpu().numpy()  # [H, W]
        descriptors = out["descriptors"].squeeze()  # [D, H/8, W/8]
        reliability = out["reliability"].squeeze().cpu().numpy()  # [1, H/8, W/8]

        return heatmap, descriptors, reliability

    def detect_keypoints(self, heatmap, reliability=None):
        """Detect keypoints from heatmap using NMS."""
        # Simple NMS
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(heatmap, kernel)
        kps_mask = (heatmap == dilated) & (heatmap > self.thresh)

        y_idxs, x_idxs = np.where(kps_mask)
        scores = heatmap[y_idxs, x_idxs]

        # Sort by score and take top_k
        ids = np.argsort(scores)[::-1][: self.top_k]

        return x_idxs[ids], y_idxs[ids], scores[ids]

    def sample_descriptors(self, descriptors, x_coords, y_coords):
        """Sample descriptors at keypoint locations."""
        D, H_feat, W_feat = descriptors.shape
        desc_list = []

        for x, y in zip(x_coords, y_coords):
            gx = int(np.clip(x / 8.0, 0, W_feat - 1))
            gy = int(np.clip(y / 8.0, 0, H_feat - 1))
            desc_list.append(descriptors[:, gy, gx].cpu().numpy())

        if len(desc_list) == 0:
            return np.zeros((0, D))

        return np.stack(desc_list)

    def match(self, img0_bgr, img1_bgr):
        """
        Match two BGR images.

        Returns:
            mkpts0: np.array (N, 2) matched keypoints in image 0
            mkpts1: np.array (N, 2) matched keypoints in image 1
        """
        # Convert to RGB tensors
        img0 = (
            torch.from_numpy(
                cv2.cvtColor(img0_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        img1 = (
            torch.from_numpy(
                cv2.cvtColor(img1_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
        )

        # Extract features
        h0, d0, r0 = self.extract(img0)
        h1, d1, r1 = self.extract(img1)

        # Detect keypoints
        x0, y0, s0 = self.detect_keypoints(h0, r0)
        x1, y1, s1 = self.detect_keypoints(h1, r1)

        if len(x0) < 10 or len(x1) < 10:
            return np.zeros((0, 2)), np.zeros((0, 2))

        # Sample descriptors
        desc0 = self.sample_descriptors(d0, x0, y0)
        desc1 = self.sample_descriptors(d1, x1, y1)

        # Match using BFMatcher
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = bf.match(desc0.astype(np.float32), desc1.astype(np.float32))

        if len(matches) == 0:
            return np.zeros((0, 2)), np.zeros((0, 2))

        # Extract matched coordinates
        mkpts0 = np.array([[x0[m.queryIdx], y0[m.queryIdx]] for m in matches])
        mkpts1 = np.array([[x1[m.trainIdx], y1[m.trainIdx]] for m in matches])

        return mkpts0, mkpts1


# ==================== Pose Estimation Metrics ====================


def intrinsics_to_camera(K):
    """Convert 3x3 intrinsic matrix to poselib camera dict."""
    px, py = K[0, 2], K[1, 2]
    fx, fy = K[0, 0], K[1, 1]
    return {
        "model": "PINHOLE",
        "width": int(2 * px),
        "height": int(2 * py),
        "params": [fx, fy, px, py],
    }


def estimate_pose_poselib(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    """Estimate relative pose using poselib."""
    M, info = poselib.estimate_relative_pose(
        kpts0,
        kpts1,
        intrinsics_to_camera(K0),
        intrinsics_to_camera(K1),
        {
            "max_epipolar_error": thresh,
            "success_prob": conf,
            "min_iterations": 20,
            "max_iterations": 1000,
        },
    )
    return M.R, M.t, np.array(info["inliers"])


def estimate_pose_opencv(kpts0, kpts1, K0, K1, thresh, conf=0.99999):
    """Fallback: Estimate relative pose using OpenCV."""
    E, mask = cv2.findEssentialMat(
        kpts0, kpts1, K0, method=cv2.RANSAC, prob=conf, threshold=thresh
    )
    if E is None:
        return None, None, np.array([])

    _, R, t, mask = cv2.recoverPose(E, kpts0, kpts1, K0, mask=mask)
    return R, t.squeeze(), mask.ravel().astype(bool)


def relative_pose_error(T_0to1, R, t, ignore_gt_t_thr=0.0):
    """Compute rotation and translation errors."""
    t_gt = T_0to1[:3, 3]
    n = np.linalg.norm(t) * np.linalg.norm(t_gt)

    if n < 1e-8:
        t_err = 0
    else:
        t_err = np.rad2deg(np.arccos(np.clip(np.dot(t, t_gt) / n, -1.0, 1.0)))
        t_err = np.minimum(t_err, 180 - t_err)

    if np.linalg.norm(t_gt) < ignore_gt_t_thr:
        t_err = 0

    R_gt = T_0to1[:3, :3]
    cos = (np.trace(np.dot(R.T, R_gt)) - 1) / 2
    cos = np.clip(cos, -1.0, 1.0)
    R_err = np.rad2deg(np.abs(np.arccos(cos)))

    return t_err, R_err


def compute_pose_error(pair, use_poselib=True):
    """Compute pose error for a matched pair."""
    pixel_thr = pair.get("ransac_thr", 1.0)

    pts0 = pair["pts0"]
    pts1 = pair["pts1"]
    K0 = pair["K0"].cpu().numpy()[0]
    K1 = pair["K1"].cpu().numpy()[0]
    T_0to1 = pair["T_0to1"].cpu().numpy()[0]

    if len(pts0) < 8:
        return np.inf, np.inf, 0

    try:
        if use_poselib and HAS_POSELIB:
            R, t, inliers = estimate_pose_poselib(pts0, pts1, K0, K1, pixel_thr)
        else:
            R, t, inliers = estimate_pose_opencv(pts0, pts1, K0, K1, pixel_thr)

        if R is None:
            return np.inf, np.inf, 0

        t_err, R_err = relative_pose_error(T_0to1, R, t)
        return t_err, R_err, inliers.sum()

    except Exception:
        return np.inf, np.inf, 0


def error_auc(errors, thresholds=[5, 10, 20]):
    """Compute AUC for pose errors."""
    errors = [0] + sorted(list(errors))
    recall = list(np.linspace(0, 1, len(errors)))

    aucs = {}
    for thr in thresholds:
        last_index = np.searchsorted(errors, thr)
        y = recall[:last_index] + [recall[last_index - 1]]
        x = errors[:last_index] + [thr]
        aucs[f"AUC@{thr}°"] = np.trapz(y, x) / thr

    return aucs


def tensor2bgr(t):
    """Convert tensor [1, 3, H, W] to BGR numpy array."""
    return (t[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)[:, :, ::-1]


@torch.no_grad()
def run_benchmark(matcher, loader, ransac_thr=2.5):
    """Run the full MegaDepth-1500 pose benchmark."""
    errors = []
    num_matches_list = []

    for data in tqdm.tqdm(loader, desc="Evaluating"):
        # Match images
        mkpts0, mkpts1 = matcher.match(
            tensor2bgr(data["image0"]), tensor2bgr(data["image1"])
        )

        # Rescale keypoints
        mkpts0 = mkpts0 * data["scale0"].numpy()
        mkpts1 = mkpts1 * data["scale1"].numpy()

        # Compute pose error
        data["pts0"] = mkpts0
        data["pts1"] = mkpts1
        data["ransac_thr"] = ransac_thr

        t_err, R_err, n_inliers = compute_pose_error(data)
        errors.append(max(t_err, R_err))
        num_matches_list.append(len(mkpts0))

    # Compute metrics
    errors = np.array(errors)
    aucs = error_auc(errors)

    print("\n" + "=" * 50)
    print("MegaDepth-1500 Pose Estimation Results")
    print("=" * 50)

    for k, v in aucs.items():
        print(f"  {k}: {v * 100:.1f}%")

    for thr in [5, 10, 20]:
        acc = (errors <= thr).sum() / len(errors)
        print(f"  mAcc@{thr}°: {acc * 100:.1f}%")

    print(f"\n  Avg. Matches: {np.mean(num_matches_list):.1f}")
    print(f"  Median Error: {np.median(errors):.1f}°")
    print("=" * 50)

    return aucs, errors


def parse_args():
    parser = argparse.ArgumentParser(description="MicroFeatEX MegaDepth-1500 Benchmark")
    parser.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Path to MegaDepth dataset root (containing megadepth_test_1500)",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to model config YAML"
    )
    parser.add_argument(
        "--ransac-thr",
        type=float,
        default=2.5,
        help="RANSAC inlier threshold in pixels (default: 2.5)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=2000,
        help="Max keypoints to detect (default: 2000)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run on (default: cuda)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Load dataset
    json_file = os.path.join(args.dataset_dir, "megadepth_1500.json")
    image_dir = os.path.join(args.dataset_dir, "megadepth_test_1500")

    if not os.path.exists(json_file):
        # Try alternate location
        json_file = "assets/megadepth_1500.json"

    print(f"Loading dataset from {image_dir}...")
    dataset = MegaDepth1500(json_file, image_dir)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Found {len(dataset)} image pairs")

    # Load matcher
    print("\nLoading MicroFeatEX matcher...")
    matcher = MicroFeatEXMatcher(
        checkpoint_path=args.checkpoint,
        config_path=args.config,
        device=args.device,
        top_k=args.top_k,
    )

    # Run benchmark
    print(f"\nRunning benchmark with RANSAC threshold={args.ransac_thr}px...")
    aucs, errors = run_benchmark(matcher, loader, ransac_thr=args.ransac_thr)
