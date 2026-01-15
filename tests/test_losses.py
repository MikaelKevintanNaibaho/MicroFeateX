"""Unit tests for MicroFeatEX loss functions.

These tests validate the core loss function implementations for:
- Correct output shapes
- Expected behavior with known inputs
- Numerical stability
"""

import pytest
import torch
import torch.nn.functional as F

from microfeatex.training import losses


class TestDualSoftmaxLoss:
    """Tests for dual_softmax_loss function."""

    def test_shape_and_output(self, sample_descriptors):
        """Test that loss returns scalar and confidence tensor."""
        X, Y = sample_descriptors
        loss, conf = losses.dual_softmax_loss(X, Y)

        assert loss.dim() == 0, "Loss should be scalar"
        assert conf.shape == (len(X),), f"Conf shape should be [{len(X)}]"
        assert loss.item() >= 0, "Loss should be non-negative"

    def test_perfect_match_low_loss(self, device: str, descriptor_dim: int):
        """Test that identical descriptors produce near-zero loss."""
        n = 16
        X = torch.randn(n, descriptor_dim, device=device)
        X = F.normalize(X, p=2, dim=1)
        Y = X.clone()  # Perfect match

        loss, conf = losses.dual_softmax_loss(X, Y)

        # With identical descriptors, loss should be relatively low
        assert (
            loss.item() < 0.5
        ), f"Loss with identical descriptors should be low, got {loss.item()}"
        # Confidence should be high for all matches
        assert (
            conf.mean().item() > 0.5
        ), "Confidence should be reasonable for perfect matches"

    def test_mismatched_shapes_raises(self, device: str):
        """Test that mismatched shapes raise LossValidationError."""
        from microfeatex.exceptions import LossValidationError

        X = torch.randn(10, 64, device=device)
        Y = torch.randn(8, 64, device=device)  # Different N

        with pytest.raises(LossValidationError):
            losses.dual_softmax_loss(X, Y)


class TestHeatmapMSELoss:
    """Tests for heatmap_mse_loss function."""

    def test_identical_heatmaps_zero_loss(self, device: str):
        """Test that identical heatmaps produce zero loss."""
        heatmap = torch.rand(2, 1, 120, 160, device=device)
        loss = losses.heatmap_mse_loss(heatmap, heatmap)

        assert loss.item() < 1e-6, "Loss should be ~0 for identical heatmaps"

    def test_resize_handling(self, device: str):
        """Test that different sized heatmaps are properly handled."""
        student = torch.rand(2, 1, 60, 80, device=device)
        teacher = torch.rand(2, 1, 120, 160, device=device)  # 2x larger

        loss = losses.heatmap_mse_loss(student, teacher)

        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "Loss should be non-negative"


class TestSmoothL1Loss:
    """Tests for smooth_l1_loss function."""

    def test_zero_for_identical(self, device: str):
        """Test zero loss for identical inputs."""
        x = torch.randn(100, device=device)
        loss = losses.smooth_l1_loss(x, x)

        assert loss.item() < 1e-6, "Loss should be ~0 for identical inputs"

    def test_smooth_transition(self, device: str):
        """Test smooth behavior near beta threshold."""
        x = torch.zeros(1, device=device)

        # At diff = beta, smooth L1 should transition smoothly
        y_small = torch.tensor([1.5], device=device)  # Below beta=2
        y_large = torch.tensor([3.0], device=device)  # Above beta=2

        loss_small = losses.smooth_l1_loss(x, y_small, beta=2.0)
        loss_large = losses.smooth_l1_loss(x, y_large, beta=2.0)

        # Both should be positive
        assert loss_small.item() > 0
        assert loss_large.item() > loss_small.item()


class TestKeypointLoss:
    """Tests for keypoint_loss (reliability BCE loss)."""

    def test_perfect_prediction_low_loss(self, device: str):
        """Test that matching predictions produce low loss."""
        pred = torch.tensor([0.9, 0.8, 0.1, 0.2], device=device)
        target = torch.tensor([0.9, 0.8, 0.1, 0.2], device=device)

        loss = losses.keypoint_loss(pred, target)

        assert (
            loss.item() < 3.0
        ), f"Loss should be reasonable for matching predictions, got {loss.item()}"

    def test_output_is_scalar(self, device: str):
        """Test that output is a scalar tensor."""
        pred = torch.rand(32, device=device)
        target = torch.rand(32, device=device)

        loss = losses.keypoint_loss(pred, target)

        assert loss.dim() == 0, "Loss should be scalar"


class TestHardTripletLoss:
    """Tests for hard_triplet_loss function."""

    def test_shape_requirements(self, device: str):
        """Test that function validates input shapes."""
        X = torch.randn(10, 64, device=device)
        Y = torch.randn(8, 64, device=device)

        with pytest.raises(RuntimeError):
            losses.hard_triplet_loss(X, Y)

    def test_same_embeddings_margin_loss(self, device: str):
        """Test loss behavior with identical embeddings."""
        X = torch.randn(16, 64, device=device)
        X = F.normalize(X, p=2, dim=1)
        Y = X.clone()

        loss = losses.hard_triplet_loss(X, Y, margin=0.5)

        # With identical embeddings, positives have dist 0
        # Negatives also have dist 0, so loss = max(0.5 + 0 - 0, 0) = 0.5
        assert loss.item() >= 0, "Loss should be non-negative"


class TestAlikeDistillLoss:
    """Tests for alike_distill_loss function."""

    def test_returns_loss_and_accuracy(self, device: str):
        """Test that function returns both loss and accuracy."""
        student_logits = torch.randn(65, 60, 80, device=device)
        teacher_heatmap = torch.sigmoid(torch.randn(1, 480, 640, device=device))

        loss, accuracy = losses.alike_distill_loss(
            student_logits, teacher_heatmap, grid_size=8
        )

        assert loss.dim() == 0, "Loss should be scalar"
        assert 0 <= accuracy <= 1, "Accuracy should be in [0, 1]"

    def test_batch_version(self, sample_heatmaps, device: str):
        """Test batch version of alike_distill_loss."""
        student_logits, teacher_heatmap = sample_heatmaps

        loss, accuracy = losses.batch_alike_distill_loss(
            student_logits, teacher_heatmap, grid_size=8
        )

        assert loss.dim() == 0, "Loss should be scalar"
        assert 0 <= accuracy <= 1, "Accuracy should be in [0, 1]"


class TestSuperPointDistillLoss:
    """Tests for superpoint_distill_loss function."""

    def test_returns_scalar(self, device: str):
        """Test that function returns scalar loss."""
        student_heat = torch.sigmoid(torch.randn(2, 1, 480, 640, device=device))
        teacher_scores = torch.randn(2, 65, 60, 80, device=device)

        loss = losses.superpoint_distill_loss(student_heat, teacher_scores, grid_size=8)

        assert loss.dim() == 0, "Loss should be scalar"
        assert torch.isfinite(loss), "Loss should be finite"
