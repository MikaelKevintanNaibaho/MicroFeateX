"""Property-based tests for MicroFeatEX using Hypothesis.

These tests verify mathematical properties that should hold for all valid inputs.
"""

import torch
import torch.nn.functional as F
from hypothesis import given, settings, strategies as st

from microfeatex.training import losses


# Custom strategies for generating tensors
@st.composite
def descriptor_tensors(draw, min_n: int = 4, max_n: int = 64, dim: int = 64):
    """Generate pairs of normalized descriptor tensors."""
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    X = torch.randn(n, dim)
    Y = torch.randn(n, dim)
    X = F.normalize(X, p=2, dim=1)
    Y = F.normalize(Y, p=2, dim=1)
    return X, Y


@st.composite
def reliability_tensors(draw, min_n: int = 8, max_n: int = 128):
    """Generate reliability prediction and target tensors."""
    n = draw(st.integers(min_value=min_n, max_value=max_n))
    pred = torch.sigmoid(torch.randn(n))  # 0-1 range
    target = torch.sigmoid(torch.randn(n))  # 0-1 range
    return pred, target


class TestDualSoftmaxProperties:
    """Property-based tests for dual_softmax_loss."""

    @given(descriptors=descriptor_tensors())
    @settings(max_examples=20, deadline=None)
    def test_loss_is_non_negative(self, descriptors):
        """Loss should always be non-negative."""
        X, Y = descriptors
        loss, conf = losses.dual_softmax_loss(X, Y)

        assert loss.item() >= 0, f"Loss should be >= 0, got {loss.item()}"

    @given(descriptors=descriptor_tensors())
    @settings(max_examples=20, deadline=None)
    def test_confidence_in_valid_range(self, descriptors):
        """Confidence should be in [0, 1]."""
        X, Y = descriptors
        _, conf = losses.dual_softmax_loss(X, Y)

        assert conf.min() >= 0, "Confidence should be >= 0"
        assert conf.max() <= 1, "Confidence should be <= 1"

    @given(n=st.integers(min_value=4, max_value=32))
    @settings(max_examples=10, deadline=None)
    def test_identical_inputs_low_loss(self, n: int):
        """Identical inputs should produce relatively low loss."""
        X = F.normalize(torch.randn(n, 64), p=2, dim=1)
        Y = X.clone()

        loss, conf = losses.dual_softmax_loss(X, Y)

        # With identical descriptors, loss should be near zero
        assert (
            loss.item() < 1.0
        ), f"Loss with identical inputs should be low, got {loss.item()}"


class TestKeypointLossProperties:
    """Property-based tests for keypoint_loss."""

    @given(tensors=reliability_tensors())
    @settings(max_examples=20, deadline=None)
    def test_loss_is_non_negative(self, tensors):
        """Keypoint loss should always be non-negative."""
        pred, target = tensors
        loss = losses.keypoint_loss(pred, target)

        assert loss.item() >= 0, f"Loss should be >= 0, got {loss.item()}"

    @given(n=st.integers(min_value=8, max_value=64))
    @settings(max_examples=10, deadline=None)
    def test_identical_inputs_low_loss(self, n: int):
        """Identical predictions should produce low loss."""
        pred = torch.sigmoid(torch.randn(n))
        target = pred.clone()

        loss = losses.keypoint_loss(pred, target)

        # BCE with identical inputs should be relatively low
        assert (
            loss.item() < 5.0
        ), f"Loss with identical inputs should be low, got {loss.item()}"


class TestSmoothL1Properties:
    """Property-based tests for smooth_l1_loss."""

    @given(n=st.integers(min_value=1, max_value=100))
    @settings(max_examples=20, deadline=None)
    def test_identical_inputs_zero_loss(self, n: int):
        """Identical inputs should produce zero loss."""
        x = torch.randn(n)
        loss = losses.smooth_l1_loss(x, x)

        assert loss.item() < 1e-6, f"Loss should be ~0, got {loss.item()}"

    @given(n=st.integers(min_value=1, max_value=100))
    @settings(max_examples=20, deadline=None)
    def test_loss_is_non_negative(self, n: int):
        """Smooth L1 loss should always be non-negative."""
        x = torch.randn(n)
        y = torch.randn(n)
        loss = losses.smooth_l1_loss(x, y)

        assert loss.item() >= 0, f"Loss should be >= 0, got {loss.item()}"

    @given(
        n=st.integers(min_value=1, max_value=50),
        beta=st.floats(min_value=0.1, max_value=10.0),
    )
    @settings(max_examples=15, deadline=None)
    def test_symmetry(self, n: int, beta: float):
        """Loss should be symmetric: L(x, y) == L(y, x)."""
        x = torch.randn(n)
        y = torch.randn(n)

        loss_xy = losses.smooth_l1_loss(x, y, beta=beta)
        loss_yx = losses.smooth_l1_loss(y, x, beta=beta)

        assert torch.isclose(loss_xy, loss_yx, atol=1e-6), "Loss should be symmetric"


class TestHardTripletProperties:
    """Property-based tests for hard_triplet_loss."""

    @given(descriptors=descriptor_tensors(min_n=8, max_n=32))
    @settings(max_examples=15, deadline=None)
    def test_loss_is_non_negative(self, descriptors):
        """Triplet loss should always be non-negative."""
        X, Y = descriptors
        loss = losses.hard_triplet_loss(X, Y)

        assert loss.item() >= 0, f"Loss should be >= 0, got {loss.item()}"
