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
    DeepBit layer with Straight-Through Estimator (STE).
    """

    def __init__(self, in_dim, out_bits=256):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_bits, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.proj(x)
        x = self.tanh(x)
        if self.training:
            return x
        else:
            return torch.sign(x)


class EfficientFeatureExtractor(nn.Module):
    def __init__(self, descriptor_dim=64, binary_bits=256):
        super().__init__()

        # 1. Load Pretrained Weights
        mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

        # 2. Modify First Layer for Grayscale
        original_first_layer = mobilenet.features[0][0]
        new_first_layer = nn.Conv2d(
            1, 16, kernel_size=3, stride=2, padding=1, bias=False
        )
        with torch.no_grad():
            new_first_layer.weight[:] = original_first_layer.weight.sum(
                dim=1, keepdim=True
            )

        # 3. Backbone
        self.backbone = nn.Sequential(
            new_first_layer,  # /2
            mobilenet.features[0][1],  # BN
            mobilenet.features[0][2],  # HardSwish
            mobilenet.features[1],  # /2 (Total /4)
            DepthwiseSeparableConv(16, 32, stride=2),  # /8 resolution
            DepthwiseSeparableConv(32, 64, stride=1),
        )

        # --- UPGRADE: High-Res PixelShuffle Head ---
        # Instead of 1 channel, we predict 65 channels (8x8 grid + 1 dustbin)
        self.detector_head = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 65, kernel_size=1),  # 64 corners + 1 dustbin
        )

        # 5. Descriptor Head
        self.descriptor_head = nn.Sequential(
            DepthwiseSeparableConv(64, descriptor_dim, stride=1),
        )

        # 6. Hashing Head
        self.hashing = DeepBitHash(descriptor_dim, binary_bits)

        # Distillation Adapter
        self.adapter = nn.Conv2d(descriptor_dim, 256, kernel_size=1)

    def forward(self, x):
        features = self.backbone(x)

        # --- UPGRADE: High-Res Heatmap Logic ---
        # 1. Get Logits [B, 65, H/8, W/8]
        logits = self.detector_head(features)

        # 2. Softmax to get probabilities
        probs = F.softmax(logits, dim=1)

        # 3. Drop dustbin channel (last one) -> [B, 64, H/8, W/8]
        corners = probs[:, :-1, :, :]

        # 4. Pixel Shuffle to full resolution -> [B, 1, H, W]
        heatmap = F.pixel_shuffle(corners, 8)

        # ---------------------------------------

        # Descriptors
        desc_raw = self.descriptor_head(features)
        desc_raw = F.normalize(desc_raw, p=2, dim=1)
        binary_desc = self.hashing(desc_raw)

        teacher_aligned_desc = self.adapter(desc_raw)
        teacher_aligned_desc = F.normalize(teacher_aligned_desc, p=2, dim=1)

        return heatmap, binary_desc, teacher_aligned_desc
