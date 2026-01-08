import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class DepthwiseSeparableConv(nn.Module):
    """
    Splits spatial and channel mixing for efficiency.
    Standard in MobileNet and XFeat architectures.
    """

    def __init__(self, in_c, out_c, stride=1, kernel_size=3):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_c,
            in_c,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=in_c,
            bias=False,
        )
        self.pointwise = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))


class DeepBitHash(nn.Module):
    """
    DeepBit layer.
    During training: Outputs tanh(x) to approximate sign function (differentiable).
    During eval: Outputs sign(x) for binary codes.
    """

    def __init__(self, in_dim, out_bits=256):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_bits, kernel_size=1)
        self.tanh = nn.Tanh()
        self.t = 1.0

    def forward(self, x):
        x = self.proj(x)
        if self.training:
            return self.tanh(x * self.t)
        else:
            # Binarize to {-1, 1}
            return torch.sign(x)


class EfficientFeatureExtractor(nn.Module):
    def __init__(self, descriptor_dim=64, binary_bits=256):
        super().__init__()

        # --- 1. Improved Backbone (MobileNetV3-Small) ---
        # We use weights=DEFAULT for pretrained ImageNet features.
        mobilenet = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)

        # Modify first layer to accept 1-channel Grayscale input
        original_first_layer = mobilenet.features[0][0]
        new_first_layer = nn.Conv2d(
            1, 16, kernel_size=3, stride=2, padding=1, bias=False
        )
        with torch.no_grad():
            # Sum RGB weights to initialize Grayscale weights
            new_first_layer.weight[:] = original_first_layer.weight.sum(
                dim=1, keepdim=True
            )

        # We construct a deeper backbone than the original code.
        # MobileNetV3-Small blocks:
        # 0: Conv (s2) -> /2 resolution
        # 1: InvertedResidual (s2) -> /4 resolution
        # 2: InvertedResidual (s2) -> /8 resolution (Target for SuperPoint alignment)
        # 3: InvertedResidual (s1) -> /8 resolution (Refinement)

        # Taking layers 0 through 3 ensures we get to /8 resolution with sufficient context.
        # Note: The indices below rely on the standard torchvision MobileNetV3 structure.
        self.backbone = nn.Sequential(
            new_first_layer,  # Layer 0 (Modified)
            mobilenet.features[0][1],  # BN
            mobilenet.features[0][2],  # HardSwish
            mobilenet.features[1],  # Layer 1 (Stride 2)
            mobilenet.features[2],  # Layer 2 (Stride 2)
            mobilenet.features[3],  # Layer 3 (Stride 1, deeper features)
        )

        # SHARED Mixer (The "Neck")
        self.neck = nn.Sequential(
            nn.Conv2d(24, 64, kernel_size=1),  # CHANGED: 64 -> 24
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # 1. Detector Head (Needs spatial precision)
        self.detector_head = nn.Sequential(
            nn.Conv2d(64, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 65, 1),  # PixelShuffle input
        )

        # 2. Descriptor Head (Needs semantic context)
        # Making this deeper helps matching accuracy
        self.descriptor_head = nn.Sequential(
            DepthwiseSeparableConv(64, 128, stride=1),
            DepthwiseSeparableConv(128, descriptor_dim, stride=1),
        )
        # --- 3. Hashing Head (DeepBit) ---
        self.hashing = DeepBitHash(descriptor_dim, binary_bits)

        # --- 4. Distillation Adapter ---
        # Maps student descriptors to teacher dimension (256) for loss calculation
        self.adapter = nn.Conv2d(descriptor_dim, 256, kernel_size=1)

    def forward(self, x):
        # 1. Extract Backbone Features
        # Output resolution: H/8, W/8
        features = self.backbone(x)
        features = self.neck(features)

        # 2. Compute Heatmap (Detector)
        logits = self.detector_head(features)

        # Softmax over the 65 channels
        probs = F.softmax(logits, dim=1)

        # Remove the "dustbin" channel (last one) -> [B, 64, H/8, W/8]
        corners = probs[:, :-1, :, :]

        # Pixel Shuffle to recover full resolution -> [B, 1, H, W]
        heatmap = F.pixel_shuffle(corners, 8)

        # 3. Compute Descriptors
        desc_raw = self.descriptor_head(features)

        # L2 Normalize floating point descriptors (for training stability)
        desc_raw = F.normalize(desc_raw, p=2, dim=1)

        # Generate Binary Code (tanh in train, sign in eval)
        binary_desc = self.hashing(desc_raw)

        # 4. Adapt for Teacher Distillation
        # We output this specifically for the loss function
        teacher_aligned_desc = self.adapter(desc_raw)
        teacher_aligned_desc = F.normalize(teacher_aligned_desc, p=2, dim=1)

        return heatmap, binary_desc, teacher_aligned_desc
