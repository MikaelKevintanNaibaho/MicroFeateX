"""Unit tests for MicroFeatEX model architecture.

These tests validate:
- Output shapes and types
- Component functionality
- Configuration options
"""

import torch

from microfeatex.models.student import (
    EfficientFeatureExtractor,
    LightweightBackbone,
    HadamardGatedFusion,
)
from microfeatex.models.walsh_hadamard import HadamardMixing, UnweightedHadamardMixing


class TestHadamardMixing:
    """Tests for HadamardMixing module."""

    def test_output_shape(self, device: str):
        """Test that output shape matches expected dimensions."""
        in_ch, out_ch = 24, 64
        batch_size = 2
        h, w = 60, 80

        layer = HadamardMixing(in_ch, out_ch, learnable=True).to(device)
        x = torch.randn(batch_size, in_ch, h, w, device=device)

        out = layer(x)

        assert out.shape == (
            batch_size,
            out_ch,
            h,
            w,
        ), f"Expected shape {(batch_size, out_ch, h, w)}, got {out.shape}"

    def test_learnable_weights(self, device: str):
        """Test that learnable version has trainable parameters."""
        layer = HadamardMixing(16, 32, learnable=True).to(device)

        # Should have exactly one learnable parameter (scale)
        params = list(layer.parameters())
        assert len(params) == 1, f"Expected 1 learnable param, got {len(params)}"
        assert params[0].shape == (32, 1, 1, 1), "Scale shape should be (32, 1, 1, 1)"

    def test_non_learnable_no_params(self, device: str):
        """Test that non-learnable version has no trainable parameters."""
        layer = HadamardMixing(16, 32, learnable=False).to(device)

        params = list(layer.parameters())
        assert len(params) == 0, "Non-learnable version should have no trainable params"

    def test_gradient_flow(self, device: str):
        """Test that gradients flow through the layer."""
        layer = HadamardMixing(16, 32, learnable=True).to(device)
        x = torch.randn(1, 16, 10, 10, device=device, requires_grad=True)

        out = layer(x)
        loss = out.sum()
        loss.backward()

        assert x.grad is not None, "Gradients should flow to input"
        assert layer.scale.grad is not None, "Gradients should flow to scale parameter"


class TestUnweightedHadamardMixing:
    """Tests for UnweightedHadamardMixing module."""

    def test_no_learnable_params(self, device: str):
        """Test that unweighted version has no parameters."""
        layer = UnweightedHadamardMixing(16, 32).to(device)

        params = list(layer.parameters())
        assert len(params) == 0, "Unweighted version should have no parameters"

    def test_output_shape(self, device: str):
        """Test output shape."""
        layer = UnweightedHadamardMixing(24, 48).to(device)
        x = torch.randn(2, 24, 30, 40, device=device)

        out = layer(x)
        assert out.shape == (2, 48, 30, 40)


class TestLightweightBackbone:
    """Tests for LightweightBackbone."""

    def test_output_shapes(self, device: str):
        """Test backbone output shapes at different scales."""
        backbone = LightweightBackbone(width_mult=1.0, use_depthwise=False).to(device)
        x = torch.randn(1, 3, 480, 640, device=device)

        fused, f8 = backbone(x)

        # Fused output should be at H/8 scale
        assert fused.shape[2:] == (
            60,
            80,
        ), f"Fused spatial size should be (60, 80), got {fused.shape[2:]}"
        assert f8.shape[2:] == (
            60,
            80,
        ), f"F8 spatial size should be (60, 80), got {f8.shape[2:]}"

    def test_depthwise_variant(self, device: str):
        """Test depthwise separable variant."""
        backbone = LightweightBackbone(width_mult=1.0, use_depthwise=True).to(device)
        x = torch.randn(1, 3, 240, 320, device=device)

        fused, f8 = backbone(x)

        assert fused.dim() == 4, "Output should be 4D tensor"
        assert f8.dim() == 4, "F8 should be 4D tensor"

    def test_hadamard_variant(self, device: str):
        """Test Hadamard mixing variant."""
        backbone = LightweightBackbone(
            width_mult=1.0, use_depthwise=True, use_hadamard=True
        ).to(device)
        x = torch.randn(1, 3, 240, 320, device=device)

        fused, f8 = backbone(x)

        assert fused.dim() == 4, "Output should be 4D tensor"


class TestHadamardGatedFusion:
    """Tests for HadamardGatedFusion module."""

    def test_output_shape(self, device: str):
        """Test fusion output shape."""
        fusion = HadamardGatedFusion(in_local=24, in_global=64, out_ch=64).to(device)

        local = torch.randn(2, 24, 60, 80, device=device)
        global_feat = torch.randn(2, 64, 60, 80, device=device)

        out = fusion(local, global_feat)

        assert out.shape == (2, 64, 60, 80), f"Output shape mismatch: {out.shape}"

    def test_spatial_upsampling(self, device: str):
        """Test that global features are upsampled to match local."""
        fusion = HadamardGatedFusion(in_local=24, in_global=64, out_ch=64).to(device)

        local = torch.randn(2, 24, 60, 80, device=device)
        global_feat = torch.randn(2, 64, 30, 40, device=device)  # Half resolution

        out = fusion(local, global_feat)

        # Output should match local spatial size
        assert (
            out.shape[2:] == local.shape[2:]
        ), "Output should match local spatial size"

    def test_gate_initialization(self, device: str):
        """Test that gate is initialized to 0.5."""
        fusion = HadamardGatedFusion(in_local=24, in_global=64, out_ch=64).to(device)

        assert torch.allclose(
            fusion.gate, torch.ones_like(fusion.gate) * 0.5
        ), "Gate should be initialized to 0.5"


class TestEfficientFeatureExtractor:
    """Tests for the complete EfficientFeatureExtractor model."""

    def test_output_keys(self, device: str):
        """Test that forward returns all expected keys."""
        model = EfficientFeatureExtractor(descriptor_dim=64).to(device)
        x = torch.randn(1, 3, 480, 640, device=device)

        with torch.no_grad():
            out = model(x)

        expected_keys = {
            "heatmap",
            "descriptors",
            "reliability",
            "offset",
            "keypoint_logits",
        }
        assert (
            set(out.keys()) == expected_keys
        ), f"Missing keys: {expected_keys - set(out.keys())}"

    def test_output_shapes(self, device: str):
        """Test output tensor shapes."""
        model = EfficientFeatureExtractor(descriptor_dim=64).to(device)
        x = torch.randn(2, 3, 480, 640, device=device)

        with torch.no_grad():
            out = model(x)

        # Heatmap: [B, 1, H, W]
        assert out["heatmap"].shape == (
            2,
            1,
            480,
            640,
        ), f"Heatmap shape: {out['heatmap'].shape}"

        # Descriptors: [B, D, H/8, W/8]
        assert out["descriptors"].shape == (
            2,
            64,
            60,
            80,
        ), f"Descriptors shape: {out['descriptors'].shape}"

        # Reliability: [B, 1, H/8, W/8]
        assert out["reliability"].shape == (
            2,
            1,
            60,
            80,
        ), f"Reliability shape: {out['reliability'].shape}"

        # Keypoint logits: [B, 65, H/8, W/8]
        assert out["keypoint_logits"].shape == (
            2,
            65,
            60,
            80,
        ), f"Logits shape: {out['keypoint_logits'].shape}"

    def test_descriptor_normalization(self, device: str):
        """Test that descriptors are L2 normalized."""
        model = EfficientFeatureExtractor(descriptor_dim=64).to(device)
        x = torch.randn(1, 3, 240, 320, device=device)

        with torch.no_grad():
            out = model(x)

        # Check L2 norm along channel dimension
        norms = torch.norm(out["descriptors"], p=2, dim=1)
        assert torch.allclose(
            norms, torch.ones_like(norms), atol=1e-5
        ), "Descriptors should be L2 normalized"

    def test_heatmap_range(self, device: str):
        """Test that heatmap values are in valid probability range."""
        model = EfficientFeatureExtractor(descriptor_dim=64).to(device)
        x = torch.randn(1, 3, 240, 320, device=device)

        with torch.no_grad():
            out = model(x)

        assert out["heatmap"].min() >= 0, "Heatmap should be >= 0"
        assert out["heatmap"].max() <= 1, "Heatmap should be <= 1"

    def test_reliability_range(self, device: str):
        """Test that reliability is in [0, 1] (sigmoid output)."""
        model = EfficientFeatureExtractor(descriptor_dim=64).to(device)
        x = torch.randn(1, 3, 240, 320, device=device)

        with torch.no_grad():
            out = model(x)

        assert out["reliability"].min() >= 0, "Reliability should be >= 0"
        assert out["reliability"].max() <= 1, "Reliability should be <= 1"

    def test_all_variants(self, device: str):
        """Test all model configuration variants."""
        configs = [
            {"use_depthwise": False, "use_hadamard": False},  # Baseline
            {"use_depthwise": True, "use_hadamard": False},  # Depthwise
            {"use_depthwise": True, "use_hadamard": True},  # Hadamard
        ]

        x = torch.randn(1, 3, 240, 320, device=device)

        for cfg in configs:
            model = EfficientFeatureExtractor(descriptor_dim=64, **cfg).to(device)
            with torch.no_grad():
                out = model(x)

            assert "heatmap" in out, f"Config {cfg} failed to produce heatmap"

    def test_width_multiplier(self, device: str):
        """Test that width multiplier scales channels correctly."""
        model_full = EfficientFeatureExtractor(width_mult=1.0).to(device)
        model_half = EfficientFeatureExtractor(width_mult=0.5).to(device)

        # Half-width model should have fewer parameters
        params_full = sum(p.numel() for p in model_full.parameters())
        params_half = sum(p.numel() for p in model_half.parameters())

        assert params_half < params_full, "Half-width model should have fewer params"
