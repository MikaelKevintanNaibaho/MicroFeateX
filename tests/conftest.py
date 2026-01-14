"""Pytest fixtures for MicroFeatEX tests."""

import pytest
import torch


@pytest.fixture
def device() -> str:
    """Return device for testing (cuda if available, else cpu)."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def batch_size() -> int:
    """Default batch size for tests."""
    return 2


@pytest.fixture
def descriptor_dim() -> int:
    """Default descriptor dimension."""
    return 64


@pytest.fixture
def sample_descriptors(batch_size: int, descriptor_dim: int, device: str):
    """Generate sample normalized descriptors for testing.
    
    Returns:
        Tuple of (X, Y) matching descriptor pairs of shape [N, D].
    """
    n_points = 32
    X = torch.randn(n_points, descriptor_dim, device=device)
    X = torch.nn.functional.normalize(X, p=2, dim=1)

    # Y is same as X (perfect match) with small noise
    Y = X + torch.randn_like(X) * 0.01
    Y = torch.nn.functional.normalize(Y, p=2, dim=1)

    return X, Y


@pytest.fixture
def sample_heatmaps(batch_size: int, device: str):
    """Generate sample heatmaps for testing.
    
    Returns:
        Tuple of (student_logits, teacher_heatmap).
    """
    h, w = 60, 80  # H/8, W/8 for 480x640 input

    # Student logits: [B, 65, H/8, W/8]
    student_logits = torch.randn(batch_size, 65, h, w, device=device)

    # Teacher heatmap: [B, 1, H, W] (full resolution)
    teacher_heatmap = torch.sigmoid(torch.randn(batch_size, 1, h * 8, w * 8, device=device))

    return student_logits, teacher_heatmap
