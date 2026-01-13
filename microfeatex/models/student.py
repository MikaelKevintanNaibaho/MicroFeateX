import torch
import torch.nn as nn
import torch.nn.functional as F

from microfeatex.models.utils import estimate_flops, count_parameters, print_model_stats

from microfeatex.models.walsh_hadamard import HadamardMixing, GhostHadamardMixing


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
    Switchable Block: Can be Standard Conv, Depthwise Separable, or Hadamard Mixing.
    """

    def __init__(
        self,
        in_c,
        out_c,
        stride=1,
        kernel_size=3,
        use_depthwise=False,
        use_hadamard=False,
    ):
        super().__init__()
        self.use_depthwise = use_depthwise
        padding = kernel_size // 2

        if use_depthwise:
            # Select Pointwise Layer Type
            if use_hadamard:
                pointwise = HadamardMixing(in_c, out_c)
            else:
                # Standard Learnable 1x1 Conv
                pointwise = nn.Conv2d(in_c, out_c, 1, 1, 0, bias=False)

            self.net = nn.Sequential(
                # Depthwise (Spatial)
                nn.Conv2d(
                    in_c, in_c, kernel_size, stride, padding, groups=in_c, bias=False
                ),
                nn.BatchNorm2d(in_c),
                nn.ReLU6(inplace=True),
                # Pointwise (Channel Mixing)
                pointwise,
                nn.BatchNorm2d(out_c),
                nn.ReLU6(inplace=True),
            )
        else:
            # Standard Conv (BasicLayer) - Hadamard is usually not applied here directly
            # as it replaces the 1x1 mixing part of a separable block.
            self.net = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

    def forward(self, x):
        return self.net(x)


class LightweightBackbone(nn.Module):
    def __init__(self, width_mult=1.0, use_depthwise=False, use_hadamard=False):
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

        # Define Block Type wrapper to pass params easily
        def Block(in_c, out_c, stride=1, kernel_size=3, use_depthwise=use_depthwise):
            return ConvBlock(
                in_c, out_c, stride, kernel_size, use_depthwise, use_hadamard
            )

        # --- Layers ---
        # Stem (Keep standard conv for first layer usually)
        self.block1 = nn.Sequential(
            # Force standard conv for the very first layer (H/2) to retain info
            ConvBlock(1, self.ch_sizes[0], stride=2, use_depthwise=False),
            Block(self.ch_sizes[0], self.ch_sizes[0]),
        )

        self.block2 = nn.Sequential(
            Block(self.ch_sizes[0], self.ch_sizes[1], stride=2),  # H/4
            Block(self.ch_sizes[1], self.ch_sizes[1]),
        )

        self.block3 = nn.Sequential(
            Block(self.ch_sizes[1], self.ch_sizes[2], stride=2),  # H/8
            Block(self.ch_sizes[2], self.ch_sizes[2]),
        )

        self.block4 = nn.Sequential(
            Block(self.ch_sizes[2], self.ch_sizes[3], stride=2),  # H/16
            Block(self.ch_sizes[3], self.ch_sizes[3]),
        )

        self.block5 = Block(self.ch_sizes[3], self.ch_sizes[3])
        self.block6 = Block(self.ch_sizes[3], self.ch_sizes[4], stride=2)  # H/32

        # --- Fusion (Optimized) ---
        fusion_in_channels = self.ch_sizes[2] + self.ch_sizes[3] + self.ch_sizes[4]
        self.fusion_out_channels = c(64)

        # Determine Fusion 1x1 Layer
        if use_hadamard and use_depthwise:
            fusion_reducer = HadamardMixing(
                fusion_in_channels, self.fusion_out_channels
            )
        else:
            fusion_reducer = nn.Conv2d(
                fusion_in_channels, self.fusion_out_channels, kernel_size=1, bias=False
            )

        self.fusion = nn.Sequential(
            # 1x1 Conv or Hadamard to squeeze channels
            fusion_reducer,
            nn.BatchNorm2d(self.fusion_out_channels),
            nn.ReLU(inplace=True),
            # Spatial processing
            Block(
                self.fusion_out_channels,
                self.fusion_out_channels,
                kernel_size=3,
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
        return self.fusion(fused), f8


class HadamardGatedFusion(nn.Module):
    """
    Adaptive Fusion Block: Fuses Local (Low-Level) and Global (High-Level) features
    using a learnable channel-wise gating mechanism.
    
    Formula: Out = Gate * Global + (1 - Gate) * Proj(Local)
    """

    def __init__(self, in_local, in_global, out_ch):
        super().__init__()
        
        # 1. Project Local (e.g. 24 -> 64) to match Global
        self.local_proj = nn.Conv2d(in_local, in_global, 1, bias=False)
        self.norm_local = nn.BatchNorm2d(in_global)
        
        # 2. Gating Weight (Learnable per-channel mix)
        # Initialize to 0.5 (equal mix)
        self.gate = nn.Parameter(torch.ones(1, in_global, 1, 1) * 0.5)
        
        # 3. Output Compression (if needed, usually in_global == out_ch)
        if in_global != out_ch:
             self.out_conv = nn.Conv2d(in_global, out_ch, 1, bias=False)
        else:
             self.out_conv = nn.Identity()

    def forward(self, x_local, x_global):
        # Resize Global to match Local spatial resolution if needed
        if x_global.shape[-2:] != x_local.shape[-2:]:
            x_global = F.interpolate(x_global, size=x_local.shape[-2:], mode="bilinear", align_corners=False)
            
        local_feat = self.norm_local(self.local_proj(x_local))
        
        # Learnable Soft-Gating
        # 0.0 -> Use Local Only
        # 1.0 -> Use Global Only
        w = torch.sigmoid(self.gate)
        
        fused = w * x_global + (1 - w) * local_feat
        return self.out_conv(fused)

class EfficientFeatureExtractor(nn.Module):
    def __init__(
        self, descriptor_dim=64, width_mult=1.0, use_depthwise=False, use_hadamard=False
    ):
        super().__init__()

        # Initialize Backbone with Hadamard option
        self.backbone = LightweightBackbone(
            width_mult=width_mult,
            use_depthwise=use_depthwise,
            use_hadamard=use_hadamard,
        )

        backbone_out = self.backbone.fusion_out_channels
        fine_feats_ch = self.backbone.ch_sizes[2]  # typically 24
        Block = ConvBlock

        # Note: Heads generally require learnable parameters to map features to
        # specific outputs (heatmaps/descriptors), so we generally keep standard convs here.
        # However, the intermediate blocks in the heads can respect the use_hadamard flag.

        # 1. Detector Head - Uses FINE features (f8)
        self.keypoint_head = nn.Sequential(
            Block(
                fine_feats_ch,
                32,
                use_depthwise=use_depthwise,
                use_hadamard=use_hadamard,
            ),
            nn.Conv2d(32, 65, kernel_size=1),
        )

        # 2. Descriptor Head - Uses FUSED features
        self.descriptor_head = nn.Sequential(
            Block(
                backbone_out,
                128,
                use_depthwise=use_depthwise,
                use_hadamard=use_hadamard,
            ),
            nn.Conv2d(128, descriptor_dim, kernel_size=1),
            nn.BatchNorm2d(descriptor_dim),
        )

        # 3. Reliability Head - Adaptive Gated Fusion
        # Fuses fine_feats (24) and backbone_out (64) -> 32 -> Output
        self.reliability_gate = HadamardGatedFusion(fine_feats_ch, backbone_out, backbone_out)
        
        self.reliability_head = nn.Sequential(
            Block(
                backbone_out,
                32,
                use_depthwise=use_depthwise,
                use_hadamard=use_hadamard,
            ),
            nn.Conv2d(32, 1, kernel_size=1),
        )

        # 4. Offset Head - Uses FINE features (f8)
        self.offset_head = nn.Sequential(
            Block(
                fine_feats_ch,
                32,
                use_depthwise=use_depthwise,
                use_hadamard=use_hadamard,
            ),
            nn.Conv2d(32, 2, kernel_size=1),
        )

    def forward(self, x):
        features_fused, features_f8 = self.backbone(x)

        # Keypoints
        kpts_logits = self.keypoint_head(features_f8)
        probs = F.softmax(kpts_logits, dim=1)
        corners = probs[:, :-1, :, :]
        heatmap = F.pixel_shuffle(corners, 8)

        # Descriptors
        desc_raw = self.descriptor_head(features_fused)
        descriptor = F.normalize(desc_raw, p=2, dim=1)

        # Reliability (Using Adaptive Gate)
        # Note: "features_fused" is usually conceptually "Global" here relative to f8
        rel_feat = self.reliability_gate(features_f8, features_fused)
        reliability = torch.sigmoid(self.reliability_head(rel_feat))
        
        offsets = torch.tanh(self.offset_head(features_f8)) * 0.5

        return {
            "heatmap": heatmap,
            "descriptors": descriptor,
            "reliability": reliability,
            "offset": offsets,
            "keypoint_logits": kpts_logits,
        }

    def profile(self, input_size=(1, 1, 480, 640)):
        print_model_stats(self, name=self.__class__.__name__, input_size=input_size)


if __name__ == "__main__":
    print("=" * 60)
    print("MicroFeatEX Model Benchmarking")
    print("=" * 60)

    # Added "Hadamard" config to experiments
    configs = [
        ("Baseline (Accuracy)", 1.0, False, False),
        ("Lite (Balanced)", 1.0, True, False),
        ("Lite-Hadamard (Ablation)", 1.0, True, True),
        ("Nano (Speed)", 0.5, True, False),
        ("Nano-Hadamard (Fixed Mixing)", 0.5, True, True),
    ]

    input_res = (1, 1, 480, 640)

    for name, width, dw, hadamard_flag in configs:
        print(f"\n--- Testing Configuration: {name} ---")
        try:
            model = EfficientFeatureExtractor(
                descriptor_dim=64,
                width_mult=width,
                use_depthwise=dw,
                use_hadamard=hadamard_flag,
            )
            model.profile(input_size=input_res)
        except Exception as e:
            print(f"Error: {e}")
