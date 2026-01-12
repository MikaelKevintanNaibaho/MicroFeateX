import torch
import cv2
import numpy as np
import sys
import argparse
from pathlib import Path

# Fix imports if running as script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from microfeatex.models.student import EfficientFeatureExtractor
from microfeatex.utils.config import load_config
from microfeatex.utils import geometry


def parse_args():
    parser = argparse.ArgumentParser(description="Debug Correspondences")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to model config"
    )
    parser.add_argument(
        "--ckpt_path", type=str, required=True, help="Path to model checkpoint (.pth)"
    )
    parser.add_argument(
        "--img1", type=str, default="assets/ref.png", help="Path to first image"
    )
    parser.add_argument(
        "--img2", type=str, default="assets/tgt.png", help="Path to second image"
    )
    parser.add_argument(
        "--output", type=str, default="debug_matches.png", help="Output image path"
    )
    parser.add_argument(
        "--conf", type=float, default=0.015, help="Keypoint confidence threshold"
    )
    return parser.parse_args()


def draw_matches(img1, img2, kpts1, kpts2, color=(0, 255, 0)):
    """
    Draws lines between matching keypoints.
    """
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]

    # Create a canvas
    out = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
    out[:h1, :w1] = img1
    out[:h2, w1:] = img2

    for (x1, y1), (x2, y2) in zip(kpts1, kpts2):
        pt1 = (int(x1), int(y1))
        pt2 = (int(x2) + w1, int(y2))
        cv2.line(out, pt1, pt2, color, 1)
        cv2.circle(out, pt1, 2, color, -1)
        cv2.circle(out, pt2, 2, color, -1)

    return out


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. Load Config & Model
    print(f"Loading Config: {args.config}")
    config = load_config(args.config)

    model_conf = config.get("model", {})

    print("Building Model...")
    model = EfficientFeatureExtractor(
        descriptor_dim=model_conf.get("descriptor_dim", 64),
        width_mult=model_conf.get("width_mult", 1.0),
        use_depthwise=model_conf.get("use_depthwise", False),
        use_hadamard=model_conf.get(
            "use_hadamard", False
        ),  # CRITICAL UPDATE for your new models
    ).to(device)

    # Load Checkpoint
    print(f"Loading checkpoint from {args.ckpt_path}...")
    try:
        ckpt = torch.load(args.ckpt_path, map_location=device)
        # Handle both full training state (with 'model_state') and raw weights
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return

    model.eval()

    # 2. Load Images
    print(f"Processing: {args.img1} <-> {args.img2}")
    img1_raw = cv2.imread(args.img1)
    img2_raw = cv2.imread(args.img2)

    if img1_raw is None or img2_raw is None:
        print(f"Error: Could not load images. Check paths.")
        return

    # Resize for consistency
    H, W = 480, 640
    img1 = cv2.resize(img1_raw, (W, H))
    img2 = cv2.resize(img2_raw, (W, H))

    # Convert to Tensor [1, 1, H, W]
    def preprocess(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        t = torch.from_numpy(gray).float() / 255.0
        return t.unsqueeze(0).unsqueeze(0).to(device)

    t1 = preprocess(img1)
    t2 = preprocess(img2)

    # 3. Inference
    with torch.no_grad():
        out1 = model(t1)
        out2 = model(t2)

    # 4. Extract Features
    def extract_features(out, conf_thresh=0.015):
        # Heatmap
        heatmap = out["heatmap"]  # [1, 1, H, W]
        # Simple NMS (Max Pooling)
        nms_map = geometry.simple_nms(heatmap, nms_radius=4)

        # Get coordinates where score > thresh
        ys, xs = torch.where(nms_map.squeeze() > conf_thresh)

        if len(xs) == 0:
            return None, None, None

        scores = heatmap[0, 0, ys, xs]
        kpts = torch.stack([xs, ys], dim=1).float()  # [N, 2]

        # Sample Descriptors
        desc_map = out["descriptors"]

        # Normalize kpts: (x / W) * 2 - 1
        norm_kpts = torch.zeros_like(kpts)
        norm_kpts[:, 0] = (kpts[:, 0] / (W - 1)) * 2 - 1
        norm_kpts[:, 1] = (kpts[:, 1] / (H - 1)) * 2 - 1

        # grid_sample expects [1, 1, N, 2]
        grid = norm_kpts.view(1, 1, -1, 2)

        # Sample
        desc = torch.nn.functional.grid_sample(desc_map, grid, align_corners=True)
        # [N, D]
        desc = desc.squeeze().t()

        # Normalize descriptors
        desc = torch.nn.functional.normalize(desc, p=2, dim=1)

        return kpts, desc, scores

    kpts1, desc1, scores1 = extract_features(out1, args.conf)
    kpts2, desc2, scores2 = extract_features(out2, args.conf)

    if kpts1 is None or kpts2 is None:
        print("No keypoints found in one of the images. Try lowering --conf.")
        return

    print(f"Extracted: Img1 ({len(kpts1)}), Img2 ({len(kpts2)})")

    # 5. Matching (Mutual Nearest Neighbor)
    sim = torch.matmul(desc1, desc2.t())

    max_val_1, max_idx_1 = torch.max(sim, dim=1)
    max_val_2, max_idx_2 = torch.max(sim, dim=0)

    matches = []

    for i in range(len(kpts1)):
        j = max_idx_1[i]
        # Mutual Check
        if max_idx_2[j] == i and max_val_1[i] > 0.75:
            matches.append((kpts1[i], kpts2[j]))

    print(f"Found {len(matches)} mutual matches.")

    # 6. Visualize
    mkpts1 = [m[0].cpu().numpy() for m in matches]
    mkpts2 = [m[1].cpu().numpy() for m in matches]

    vis = draw_matches(img1, img2, mkpts1, mkpts2)

    cv2.imwrite(args.output, vis)
    print(f"Saved visualization to {args.output}")


if __name__ == "__main__":
    main()
