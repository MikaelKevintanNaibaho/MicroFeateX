import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicLayer(nn.Module):
    """
    Simple convolutional block used throughout the backbone
    """

    def __init__(self, in_c, out_c, stride=1, kernel_size=3):
        super().__init__()
        # Standard padding to keep size if stride=1
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            in_c, out_c, kernel_size, stride=stride, padding=padding, bias=False
        )
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class LightweightBackbone(nn.Module):
    """
    Custom Lightweight Backbone
    """

    def __init__(self):
        super().__init__()

        self.block1 = nn.Sequential(
            BasicLayer(1, 4, stride=2, kernel_size=3),  # H/2
            BasicLayer(4, 4, kernel_size=3),
        )

        self.block2 = nn.Sequential(
            BasicLayer(4, 8, stride=2, kernel_size=3),  # H/4
            BasicLayer(8, 8, kernel_size=3),
        )

        self.block3 = nn.Sequential(
            BasicLayer(8, 24, stride=2, kernel_size=3),  # H/8
            BasicLayer(24, 24, kernel_size=3),
        )

        self.block4 = nn.Sequential(
            BasicLayer(24, 64, stride=2, kernel_size=3),  # H/16
            BasicLayer(64, 64, kernel_size=3),
        )

        self.block5 = BasicLayer(64, 64, kernel_size=3)
        self.block6 = BasicLayer(64, 128, stride=2, kernel_size=3)  # H/32

        # Fusion layer to combine multi-scale features
        self.fusion = nn.Sequential(
            BasicLayer(24 + 64 + 128, 64, kernel_size=3),
            BasicLayer(64, 64, kernel_size=3),
        )

    def forward(self, x):
        x1 = self.block1(x)  # H/2
        x2 = self.block2(x1)  # H/4
        f8 = self.block3(x2)  # H/8
        x4 = self.block4(f8)  # H/16
        x5 = self.block5(x4)  # H/16
        f32 = self.block6(x5)  # H/32

        # Multi-scale feature fusion
        f16 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        f32_up = F.interpolate(
            f32, scale_factor=4, mode="bilinear", align_corners=False
        )

        # Concatenate features at H/8 resolution
        # Note: Ensure sizes match exactly
        f16 = F.interpolate(
            f16, size=f8.shape[2:], mode="bilinear", align_corners=False
        )
        f32_up = F.interpolate(
            f32_up, size=f8.shape[2:], mode="bilinear", align_corners=False
        )

        fused = torch.cat([f8, f16, f32_up], dim=1)

        return self.fusion(fused)  # Output at H/8 resolution


class EfficientFeatureExtractor(nn.Module):
    def __init__(self, descriptor_dim=64, binary_bits=256):
        super().__init__()

        self.backbone = LightweightBackbone()

        # 1. Detector Head (Heatmap)
        # Note: Must output 65 channels (64 cells + 1 dustbin) for PixelShuffle
        self.keypoint_head = nn.Sequential(
            BasicLayer(64, 32, kernel_size=3), nn.Conv2d(32, 65, kernel_size=1)
        )

        # 2. Descriptor Head
        self.descriptor_head = nn.Sequential(
            BasicLayer(64, 128, kernel_size=3),
            nn.Conv2d(128, descriptor_dim, kernel_size=1),
            nn.BatchNorm2d(descriptor_dim),  # Normalize before output
        )

        # 3. Reliability Head (Required by trainer)
        self.reliability_head = nn.Sequential(
            BasicLayer(64, 32, kernel_size=3), nn.Conv2d(32, 1, kernel_size=1)
        )

        # 4. Offset Head (CRITICAL: Added this back so loss_fine works)
        self.offset_head = nn.Sequential(
            BasicLayer(64, 32, kernel_size=3), nn.Conv2d(32, 2, kernel_size=1)
        )

        # Optional: DeepBit hashing if you want binary codes
        # self.hashing = DeepBitHash(descriptor_dim, binary_bits)

    def forward(self, x):
        features = self.backbone(x)

        # --- Keypoints (Heatmap) ---
        kpts_logits = self.keypoint_head(features)

        # Process Logits into Heatmap (Student Logic)
        probs = F.softmax(kpts_logits, dim=1)
        corners = probs[:, :-1, :, :]  # Remove dustbin
        heatmap = F.pixel_shuffle(corners, 8)  # [B, 1, H, W]

        # --- Descriptors ---
        desc_raw = self.descriptor_head(features)
        descriptor = F.normalize(desc_raw, p=2, dim=1)

        # --- Reliability ---
        reliability = torch.sigmoid(self.reliability_head(features))

        # --- Offset (New) ---
        offsets = torch.tanh(self.offset_head(features))

        return {
            "heatmap": heatmap,  # RENAMED from "keypoint_logits"
            "descriptors": descriptor,
            "reliability": reliability,
            "offset": offsets,  # ADDED back
            # "keypoint_logits": kpts_logits # Optional: keep if you want raw logits
        }
