import torch
import torch.nn as nn
import torch.nn.functional as F

from microfeatex.models.utils import estimate_flops, count_parameters, print_model_stats


def _make_divisible(v, divisor=8, min_value=None):
    """
    Ensures all layer channels are divisible by 8 (optimal for GPU/NPU hardware).
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ConvBlock(nn.Module):
    """
    Switchable Block: Can be Standard Conv or Depthwise Separable
    """

    def __init__(self, in_c, out_c, stride=1, kernel_size=3, use_depthwise=False):
        super().__init__()
        self.use_depthwise = use_depthwise
        padding = kernel_size // 2

        if use_depthwise:
            # ~8-9x fewer FLOPs
            self.net = nn.Sequential(
                # Depthwise
                nn.Conv2d(
                    in_c, in_c, kernel_size, stride, padding, groups=in_c, bias=False
                ),
                nn.BatchNorm2d(in_c),
                nn.ReLU6(inplace=True),
                # Pointwise
                nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU6(inplace=True),
            )
        else:
            # Standard Conv (Your original BasicLayer)
            self.net = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(
                    inplace=True
                ),  # Using ReLU (standard) or ReLU6 (better for quantization)
            )

    def forward(self, x):
        return self.net(x)


class LightweightBackbone(nn.Module):
    def __init__(self, width_mult=1.0, use_depthwise=False):
        super().__init__()

        # Helper to scale channels
        def c(v):
            return _make_divisible(v * width_mult)

        self.use_depthwise = use_depthwise

        # Channel definitions (Base config: 4, 8, 24, 64, 128)
        self.ch_sizes = [
            c(4),  # c1
            c(8),  # c2
            c(24),  # c3
            c(64),  # c4
            c(128),  # c5
        ]

        # Define Block Type
        Block = ConvBlock

        # --- Layers ---
        # Stem (Keep standard conv for first layer usually)
        self.block1 = nn.Sequential(
            Block(1, self.ch_sizes[0], stride=2, use_depthwise=False),  # H/2
            Block(self.ch_sizes[0], self.ch_sizes[0], use_depthwise=use_depthwise),
        )

        self.block2 = nn.Sequential(
            Block(
                self.ch_sizes[0],
                self.ch_sizes[1],
                stride=2,
                use_depthwise=use_depthwise,
            ),  # H/4
            Block(self.ch_sizes[1], self.ch_sizes[1], use_depthwise=use_depthwise),
        )

        self.block3 = nn.Sequential(
            Block(
                self.ch_sizes[1],
                self.ch_sizes[2],
                stride=2,
                use_depthwise=use_depthwise,
            ),  # H/8
            Block(self.ch_sizes[2], self.ch_sizes[2], use_depthwise=use_depthwise),
        )

        self.block4 = nn.Sequential(
            Block(
                self.ch_sizes[2],
                self.ch_sizes[3],
                stride=2,
                use_depthwise=use_depthwise,
            ),  # H/16
            Block(self.ch_sizes[3], self.ch_sizes[3], use_depthwise=use_depthwise),
        )

        self.block5 = Block(
            self.ch_sizes[3], self.ch_sizes[3], use_depthwise=use_depthwise
        )
        self.block6 = Block(
            self.ch_sizes[3], self.ch_sizes[4], stride=2, use_depthwise=use_depthwise
        )  # H/32

        # --- Fusion (Optimized) ---
        # Calculate total channels entering fusion
        fusion_in_channels = self.ch_sizes[2] + self.ch_sizes[3] + self.ch_sizes[4]
        self.fusion_out_channels = c(64)

        self.fusion = nn.Sequential(
            # OPTIMIZATION: 1x1 Conv to squeeze channels first
            nn.Conv2d(
                fusion_in_channels, self.fusion_out_channels, kernel_size=1, bias=False
            ),
            nn.BatchNorm2d(self.fusion_out_channels),
            nn.ReLU(inplace=True),
            # Then spatial processing
            Block(
                self.fusion_out_channels,
                self.fusion_out_channels,
                kernel_size=3,
                use_depthwise=use_depthwise,
            ),
        )

    def forward(self, x):
        x1 = self.block1(x)  # H/2
        x2 = self.block2(x1)  # H/4
        f8 = self.block3(x2)  # H/8
        x4 = self.block4(f8)  # H/16
        x5 = self.block5(x4)  # H/16
        f32 = self.block6(x5)  # H/32

        # Up-sampling
        f16 = F.interpolate(x5, scale_factor=2, mode="bilinear", align_corners=False)
        f32_up = F.interpolate(
            f32, scale_factor=4, mode="bilinear", align_corners=False
        )

        # Ensure sizes match f8
        if f16.shape[-2:] != f8.shape[-2:]:
            f16 = F.interpolate(
                f16, size=f8.shape[2:], mode="bilinear", align_corners=False
            )
        if f32_up.shape[-2:] != f8.shape[-2:]:
            f32_up = F.interpolate(
                f32_up, size=f8.shape[2:], mode="bilinear", align_corners=False
            )

        fused = torch.cat([f8, f16, f32_up], dim=1)
        return self.fusion(fused)


class EfficientFeatureExtractor(nn.Module):
    def __init__(self, descriptor_dim=64, width_mult=1.0, use_depthwise=False):
        super().__init__()

        # Initialize Backbone
        self.backbone = LightweightBackbone(
            width_mult=width_mult, use_depthwise=use_depthwise
        )

        # Get output channels from backbone to size heads correctly
        backbone_out = self.backbone.fusion_out_channels

        # Define Block Type for Heads
        Block = ConvBlock

        # 1. Detector Head
        self.keypoint_head = nn.Sequential(
            Block(backbone_out, 32, use_depthwise=use_depthwise),
            nn.Conv2d(32, 65, kernel_size=1),
        )

        # 2. Descriptor Head
        self.descriptor_head = nn.Sequential(
            Block(backbone_out, 128, use_depthwise=use_depthwise),
            nn.Conv2d(128, descriptor_dim, kernel_size=1),
            nn.BatchNorm2d(descriptor_dim),
        )

        # 3. Reliability Head
        self.reliability_head = nn.Sequential(
            Block(backbone_out, 32, use_depthwise=use_depthwise),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        # 4. Offset Head
        self.offset_head = nn.Sequential(
            Block(backbone_out, 32, use_depthwise=use_depthwise),
            nn.Conv2d(32, 2, kernel_size=1),
        )

    def forward(self, x):
        features = self.backbone(x)

        # Keypoints
        kpts_logits = self.keypoint_head(features)
        probs = F.softmax(kpts_logits, dim=1)
        corners = probs[:, :-1, :, :]
        heatmap = F.pixel_shuffle(corners, 8)

        # Descriptors
        desc_raw = self.descriptor_head(features)
        descriptor = F.normalize(desc_raw, p=2, dim=1)

        # Reliability & Offset
        reliability = torch.sigmoid(self.reliability_head(features))
        offsets = torch.tanh(self.offset_head(features)) * 0.5

        return {
            "heatmap": heatmap,
            "descriptors": descriptor,
            "reliability": reliability,
            "offset": offsets,
            "keypoint_logits": kpts_logits,
        }

    def profile(self, input_size=(1, 1, 480, 640)):
        """
        Self-profiling method.
        Usage: model.profile()
        """
        print_model_stats(self, name=self.__class__.__name__, input_size=input_size)


if __name__ == "__main__":
    print("=" * 60)
    print("MicroFeatEX Model Benchmarking")
    print("=" * 60)

    # Define the experiments you want to run
    configs = [
        ("Baseline (Accuracy)", 1.0, False),  # Width 1.0, Standard Conv
        ("Lite (Balanced)", 1.0, True),  # Width 1.0, Depthwise
        ("Nano (Speed)", 0.5, True),  # Width 0.5, Depthwise
    ]

    input_res = (1, 1, 480, 640)

    for name, width, dw in configs:
        print(f"\n--- Testing Configuration: {name} ---")
        model = EfficientFeatureExtractor(
            descriptor_dim=64, width_mult=width, use_depthwise=dw
        )

        # Use the imported utility directly on the instance
        model.profile(input_size=input_res)
