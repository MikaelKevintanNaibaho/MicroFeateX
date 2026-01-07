import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class DepthwiseSeparableConv(nn.Module):
    """
    Splits spatial and channel mixing for efficiency.
    """

    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_c, in_c, 3, stride=stride, padding=1, groups=in_c, bias=False
        )
        self.pointwise = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class DeepBitHash(nn.Module):
    """
    DeepBit layer with Straight-Through Estimator (STE) for training.
    """

    def __init__(self, in_dim, out_bits=256):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_bits, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        # x shape: [Batch, Channels, Height, Width]
        x = self.proj(x)
        x = self.tanh(x)  # Range [-1, 1]

        if self.training:
            return x  # Return float for backprop
        else:
            return torch.sign(x)  # Return binary {-1, 1} for inference


class EfficientFeatureExtractor(nn.Module):
    def __init__(self, descriptor_dim=64, binary_bits=256):
        super().__init__()

        # 1. Load Pretrained Weights
        mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

        # 2. Modify First Layer for Grayscale (Efficiency Hack)
        # Sum weights of RGB filters to create a 1-channel filter
        original_first_layer = mobilenet.features[0][0]
        new_first_layer = nn.Conv2d(
            1, 16, kernel_size=3, stride=2, padding=1, bias=False
        )
        with torch.no_grad():
            new_first_layer.weight[:] = original_first_layer.weight.sum(
                dim=1, keepdim=True
            )

        # 3. Backbone (Stride 2 -> Stride 4 -> Stride 8)
        self.backbone = nn.Sequential(
            new_first_layer,  # /2 resolution
            mobilenet.features[0][1],  # BN
            mobilenet.features[0][2],  # HardSwish
            mobilenet.features[1],  # /2 Depthwise (Total /4)
            # Custom layers (XFeat style)
            DepthwiseSeparableConv(16, 32, stride=2),  # /8 resolution
            DepthwiseSeparableConv(32, 64, stride=1),
        )

        # 4. Detector Head (Pixel-wise Keypoint Heatmap)
        self.detector_head = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid(),
        )

        # 5. Descriptor Head (Dense Local Descriptors)
        self.descriptor_head = nn.Sequential(
            DepthwiseSeparableConv(64, descriptor_dim, stride=1),
        )

        # 6. Hashing Head
        self.hashing = DeepBitHash(descriptor_dim, binary_bits)

        # Distillation Adapter
        # Projects student dim (64) to Teacher dim (256) for loss calculation
        self.adapter = nn.Conv2d(descriptor_dim, 256, kernel_size=1)

    def forward(self, x):
        # x is grayscale [B, 1, H, W]
        features = self.backbone(x)

        # 1. Heatmap
        heatmap = self.detector_head(features)

        # 2. Dense Descriptors
        desc_raw = self.descriptor_head(features)

        # L2 Normalize before hashing (crucial for stability)
        desc_raw = F.normalize(desc_raw, p=2, dim=1)

        # 3. Binary Descriptors (DeepBit)
        binary_desc = self.hashing(desc_raw)

        teacher_aligned_desc = None
        if self.training:
            teacher_aligned_desc = self.adapter(desc_raw)
            # Normalize again after projection
            teacher_aligned_desc = F.normalize(teacher_aligned_desc, p=2, dim=1)

        return heatmap, binary_desc, teacher_aligned_desc
