import torch
import cv2
import numpy as np
import argparse
import sys
import os
import glob

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from microfeatex.models.student import EfficientFeatureExtractor


def load_image(path, resize=(640, 480)):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {path}")

    # Resize for consistency
    img = cv2.resize(img, resize)
    return img


def to_tensor(img, device):
    tensor = torch.from_numpy(img).float() / 255.0
    return tensor.unsqueeze(0).unsqueeze(0).to(device)


def run_inference(model, img_tensor):
    with torch.no_grad():
        heatmap, binary_desc, _ = model(img_tensor)

    # 1. Process Heatmap
    heatmap = heatmap.squeeze().cpu().numpy()

    # 2. Process Descriptors
    # In eval mode, DeepBit returns {-1, 1}. We keep them as floats for L2 matching.
    descriptors = binary_desc.squeeze()

    return heatmap, descriptors


# --- FIX 1: Lower threshold to 0.001 to catch "Green" features ---
def extract_points(heatmap, descriptors, top_k=1000, thresh=0.015):
    """
    Extracts (x, y) keypoints and their corresponding descriptors.
    """
    # Simple NMS
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(heatmap, kernel)
    kps_mask = (heatmap == dilated) & (heatmap > thresh)

    y_idxs, x_idxs = np.where(kps_mask)
    scores = heatmap[y_idxs, x_idxs]

    # Sort by score
    ids = np.argsort(scores)[::-1][:top_k]
    y_idxs = y_idxs[ids]
    x_idxs = x_idxs[ids]

    keypoints = np.stack([x_idxs, y_idxs], axis=1).astype(np.float32)

    # Sample descriptors at keypoints
    D, H_feat, W_feat = descriptors.shape
    desc_list = []

    for x, y in zip(x_idxs, y_idxs):
        # Nearest neighbor sample
        gx = int(np.clip(x / 8.0, 0, W_feat - 1))
        gy = int(np.clip(y / 8.0, 0, H_feat - 1))
        desc_list.append(descriptors[:, gy, gx].cpu().numpy())

    if len(desc_list) == 0:
        return np.zeros((0, 2)), np.zeros((0, D))

    return keypoints, np.stack(desc_list)


def match_and_draw(img1, kps1, desc1, img2, kps2, desc2):
    # Match using BFMatcher with L2 Norm
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(desc1, desc2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Extract matched points
    pts1 = np.float32([kps1[m.queryIdx] for m in matches])
    pts2 = np.float32([kps2[m.trainIdx] for m in matches])

    if len(pts1) < 8:
        print("Not enough matches to compute geometry!")
        return 0.0, None

    # Geometric Verification (RANSAC)
    F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, 3.0, 0.99)

    inliers = mask.ravel().sum() if mask is not None else 0
    ratio = inliers / len(matches) if len(matches) > 0 else 0

    print(f"Matches: {len(matches)}")
    print(f"Inliers (RANSAC): {inliers}")
    print(f"Inlier Ratio: {ratio:.2%}")

    # Draw matches
    vis_img = cv2.drawMatches(
        img1,
        [cv2.KeyPoint(x=p[0], y=p[1], size=3) for p in kps1],
        img2,
        [cv2.KeyPoint(x=p[0], y=p[1], size=3) for p in kps2],
        matches,
        None,
        matchesMask=mask.ravel().tolist(),
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )

    return ratio, vis_img


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=str,
        default="data/megadepth_test_1500/Undistorted_SfM/0015/images",
    )
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    # 1. Load Images
    search_path = os.path.join(args.dir, "*.jpg")
    image_files = sorted(glob.glob(search_path))

    if len(image_files) < 2:
        print(f"Error: Need at least 2 images in {args.dir}")
        print("Found:", image_files)
        return

    path1 = image_files[0]
    path2 = image_files[1]

    print(f"Testing on:\n Image 1: {path1}\n Image 2: {path2}")

    img1 = load_image(path1)
    img2 = load_image(path2)

    t1 = to_tensor(img1, args.device)
    t2 = to_tensor(img2, args.device)

    # 2. Load Model
    print(f"Loading model from {args.checkpoint}...")
    model = EfficientFeatureExtractor(descriptor_dim=64, binary_bits=256)

    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    model.to(args.device)
    model.eval()

    # 3. Extract Features
    print("Extracting features...")
    h1, d1 = run_inference(model, t1)
    h2, d2 = run_inference(model, t2)

    # Pass the lower threshold explicitly if needed, but defaults are updated above
    kps1, desc1 = extract_points(h1, d1)
    kps2, desc2 = extract_points(h2, d2)

    print(f"Image 1: {len(kps1)} keypoints")
    print(f"Image 2: {len(kps2)} keypoints")

    # 4. Match & Verify
    ratio, result_img = match_and_draw(img1, kps1, desc1, img2, kps2, desc2)

    # --- FIX 2: Only save if result_img exists ---
    if result_img is not None:
        output_filename = "megadepth_result.jpg"
        cv2.imwrite(output_filename, result_img)
        print(f"Result saved to {output_filename}")
    else:
        print("Skipping image save due to lack of matches.")

    # Quick verdict
    if ratio > 0.4:
        print("\n✅ SUCCESS: Model is ready for VSLAM!")
    elif ratio > 0.2:
        print("\n⚠️ OKAY: Model works but might drift.")
    else:
        print("\n❌ FAILURE: Model needs more training.")


if __name__ == "__main__":
    main()
