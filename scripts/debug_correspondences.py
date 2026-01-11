import torch
import matplotlib.pyplot as plt
import numpy as np
from microfeatex.data.augmentation import AugmentationPipe
from microfeatex.data.dataset import Dataset
from microfeatex.utils.config import load_config


def visualize_correspondences(p1, p2, coords_map, num_samples=50):
    """
    Visualize correspondences using the Dense Coordinate Map.
    We sample points in Image 2 (Target) and look up their position in Image 1 (Source).
    """
    B, C, H, W = p2.shape
    device = p1.device

    # 1. Generate random integer coordinates in Image 2
    # We pick random pixels in the warped image
    pts2_x = torch.randint(0, W, (num_samples,), device=device)
    pts2_y = torch.randint(0, H, (num_samples,), device=device)

    # 2. Look up where these pixels came from using coords_map
    # coords_map shape: [B, 2, H, W]
    # Channel 0 is X, Channel 1 is Y
    map_b = coords_map[0]  # First batch item

    pts1_x = map_b[0, pts2_y, pts2_x]
    pts1_y = map_b[1, pts2_y, pts2_x]

    pts1 = torch.stack([pts1_x, pts1_y], dim=1)  # [N, 2]
    pts2 = torch.stack([pts2_x, pts2_y], dim=1).float()  # [N, 2]

    # 3. Filter points that land outside Image 1 (due to padding/cropping)
    valid_mask = (
        (pts1[:, 0] >= 0) & (pts1[:, 0] < W) & (pts1[:, 1] >= 0) & (pts1[:, 1] < H)
    )

    print(f"Valid correspondences found: {valid_mask.sum().item()}/{num_samples}")

    # Filter for visualization
    pts1 = pts1[valid_mask]
    pts2 = pts2[valid_mask]

    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Image 1 (Source)
    # Move channel to last for plotting
    img1_np = p1[0].permute(1, 2, 0).cpu().numpy()
    if img1_np.shape[2] == 1:
        img1_np = img1_np.squeeze(2)

    axes[0].imshow(img1_np, cmap="gray")
    # Plot using a colormap so we can see which point matches which
    colors = np.linspace(0, 1, len(pts1))
    axes[0].scatter(
        pts1[:, 0].cpu(),
        pts1[:, 1].cpu(),
        c=colors,
        cmap="hsv",
        s=50,
        edgecolors="white",
    )
    axes[0].set_title("Image 1 (Source Positions)")

    # Image 2 (Target/Warped)
    img2_np = p2[0].permute(1, 2, 0).cpu().numpy()
    if img2_np.shape[2] == 1:
        img2_np = img2_np.squeeze(2)

    axes[1].imshow(img2_np, cmap="gray")
    axes[1].scatter(
        pts2[:, 0].cpu(),
        pts2[:, 1].cpu(),
        c=colors,
        cmap="hsv",
        s=50,
        edgecolors="white",
    )
    axes[1].set_title("Image 2 (Sampled Grid)")

    plt.tight_layout()
    plt.savefig("debug_correspondences.png", dpi=150)
    print("Saved visualization to debug_correspondences.png")

    return pts1, pts2


def test_descriptor_matching(desc1, desc2, pts1, pts2):
    """
    Test if descriptor matching works correctly
    """
    # Sample descriptors using nearest neighbor
    # Clamp to ensure we don't crash on edge pixels
    pts1_y = pts1[:, 1].long().clamp(0, desc1.shape[2] - 1)
    pts1_x = pts1[:, 0].long().clamp(0, desc1.shape[3] - 1)
    pts2_y = pts2[:, 1].long().clamp(0, desc2.shape[2] - 1)
    pts2_x = pts2[:, 0].long().clamp(0, desc2.shape[3] - 1)

    d1 = desc1[0, :, pts1_y, pts1_x].t()  # [N, C]
    d2 = desc2[0, :, pts2_y, pts2_x].t()  # [N, C]

    # Check normalization
    norms1 = torch.norm(d1, dim=1)
    norms2 = torch.norm(d2, dim=1)
    print(
        f"Descriptor norms (should be ~1.0) - d1: {norms1.mean():.4f}, d2: {norms2.mean():.4f}"
    )

    # Compute similarity
    # Since descriptors are normalized, dot product is cosine similarity
    similarity = (d1 * d2).sum(dim=1)
    print(f"Mean descriptor similarity (Correct Matches): {similarity.mean():.4f}")
    print(f"Min: {similarity.min():.4f}, Max: {similarity.max():.4f}")

    # Check if matching works (Self-Matching)
    # We check if d1[i] is closest to d2[i] among all d2
    dist_mat = d1 @ d2.t()  # [N, N]
    nn_indices = dist_mat.argmax(dim=1)

    correct = (nn_indices == torch.arange(len(d1), device=d1.device)).float()
    accuracy = correct.mean()

    print(f"Self-matching accuracy (Batch of {len(d1)} points): {accuracy:.2%}")
    return accuracy


def main():
    # Load config
    # Ensure this points to your actual config file
    config = load_config("config/coco_train.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create augmentor
    augmentor = AugmentationPipe(config, device)

    dataset = Dataset(config["paths"]["coco_root"], config)

    # Get the first image (dataset returns dictionary)
    data_item = dataset[0]
    # data_item is already the Tensor [C, H, W]
    img = data_item.unsqueeze(0).to(device)

    print("Running Augmentation on Real Image...")
    p1, p2, H, mask, coords_map = augmentor(img)

    print("Testing correspondence mapping...")
    # Pass coords_map instead of H
    pts1, pts2 = visualize_correspondences(p1, p2, coords_map, num_samples=100)

    if len(pts1) < 5:
        print("Not enough valid points to test descriptors. Try running again.")
        return

    # Test with actual model
    from microfeatex.models.student import EfficientFeatureExtractor

    model = EfficientFeatureExtractor().to(device)
    ckpt_path = "checkpoints/last.pth"  # Or microfeatex_11999.pth
    print(f"Loading weights from {ckpt_path}...")
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.eval()

    print("\nRunning Model...")
    with torch.no_grad():
        out1 = model(p1)
        out2 = model(p2)

    print("\nTesting descriptor matching...")
    # Scale points to descriptor resolution (e.g. 480x640 -> 60x80 means divide by 8)
    pts1_scaled = pts1 / 8.0
    pts2_scaled = pts2 / 8.0

    acc = test_descriptor_matching(
        out1["descriptors"], out2["descriptors"], pts1_scaled, pts2_scaled
    )

    if acc < 0.1:  # Threshold is low because untrained model is random
        print("\n⚠️  Accuracy is low (Expected for untrained model).")
        print("If this is a trained checkpoint, this indicates a problem.")
    else:
        print(f"\n✅ Matching worked! ({acc:.1%} correct)")


if __name__ == "__main__":
    main()
