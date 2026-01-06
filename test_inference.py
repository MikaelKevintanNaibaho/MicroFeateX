import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Import YOUR model class (adjust path if needed)
from src.model import EfficientFeatureExtractor

# 1. Device & Model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create instance with the same parameters you used during training
model = EfficientFeatureExtractor(
    descriptor_dim=64,  # must match your training
    binary_bits=256,  # must match your training
).to(device)

# Load your trained weights (use the epoch you want – 19 or 20 are likely best)
checkpoint_path = "checkpoints/hybrid_slam_epoch_19.pth"  # ← change to 20 if you prefer
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()

print(f"Model loaded from {checkpoint_path} on {device}")

# 2. Load two test images (e.g., from your dataset or webcam)
img1 = cv2.imread(
    "assets/test1.jpg", cv2.IMREAD_GRAYSCALE
)  # Replace with actual image paths
img2 = cv2.imread("assets/test2.jpg", cv2.IMREAD_GRAYSCALE)
img1 = cv2.resize(img1, (640, 480))
img2 = cv2.resize(img2, (640, 480))

# Preprocess for Grayscale
# 1. Convert to Tensor (H, W)
tens1 = torch.from_numpy(img1).float()
tens2 = torch.from_numpy(img2).float()

# 2. Add Channel and Batch Dims: (H, W) -> (1, 1, H, W)
tens1 = tens1.unsqueeze(0).unsqueeze(0).to(device) / 255.0
tens2 = tens2.unsqueeze(0).unsqueeze(0).to(device) / 255.0

print("Input tensor shapes:", tens1.shape)  # Should be [1, 1, 480, 640]

# 3. Inference
with torch.no_grad():
    heatmap1, binary_desc1, _ = model(tens1)  # ignore teacher adapter
    heatmap2, binary_desc2, _ = model(tens2)


# 4. Extract sparse keypoints + descriptors from dense outputs
def extract_sparse(heatmap, desc_map, top_k=2000, threshold=0.005):
    """
    Simple non-max-like extraction: threshold + top-k by score
    heatmap: [1,1,H/8,W/8]
    desc_map: [1, bits, H/8, W/8]
    """
    h = heatmap.squeeze(0).squeeze(0).cpu()  # [H/8, W/8]
    mask = h > threshold
    if mask.sum() == 0:
        print("Warning: No keypoints above threshold!")
        return np.empty((0, 2)), np.empty((0, desc_map.shape[1]))

    # Coordinates (y, x) in feature map space
    coords = torch.argwhere(mask).float()  # [N, 2]
    scores = h[mask]

    # Sort descending by score, take top_k
    idx = torch.argsort(scores, descending=True)[:top_k]
    kpts_feat = coords[idx]  # [top_k, 2] y,x in feat space
    kpts = kpts_feat * 8.0  # upsample ×8 to original image scale

    # Sample descriptors at those locations
    y = kpts_feat[:, 0].long()
    x = kpts_feat[:, 1].long()
    desc = desc_map[0, :, y, x].t().cpu().numpy()  # [top_k, bits]

    return kpts.numpy(), desc


# Extract features
kpts1, desc1 = extract_sparse(heatmap1, binary_desc1)
kpts2, desc2 = extract_sparse(heatmap2, binary_desc2)

print(f"Extracted {len(kpts1)} keypoints from image 1")
print(f"Extracted {len(kpts2)} keypoints from image 2")

if len(kpts1) == 0 or len(kpts2) == 0:
    raise ValueError("No keypoints detected – try lowering threshold to 0.001")

# 5. Prepare for OpenCV matching (THE FIX)
# Standard "Hard" Binarization: Valid bit if value > 0
# We convert 256 bits -> 32 bytes using packbits.
desc1_binary = np.packbits(desc1 > 0, axis=1)
desc2_binary = np.packbits(desc2 > 0, axis=1)

# Create OpenCV KeyPoints
# Note: kpts is (y, x) from your function, we need (x, y) for OpenCV
cv_kpts1 = [cv2.KeyPoint(float(x), float(y), 1) for y, x in kpts1]
cv_kpts2 = [cv2.KeyPoint(float(x), float(y), 1) for y, x in kpts2]

# Hamming matcher (Strict cross-check can be too harsh for early models)
# Let's use standard matching first to see if structure exists
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(desc1_binary, desc2_binary)

print(f"Found {len(matches)} matches")

# Filter by distance (lower Hamming distance is better)
matches = sorted(matches, key=lambda x: x.distance)

# 6. Draw top matches
img_matches = cv2.drawMatches(
    img1,
    cv_kpts1,
    img2,
    cv_kpts2,
    matches[:50],  # Show top 50
    None,
    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
)

plt.figure(figsize=(15, 8))
plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
plt.title(f"Top 50 Matches (Epoch 19) - Found {len(matches)}")
plt.axis("off")
plt.show()
