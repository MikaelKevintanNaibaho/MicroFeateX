import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import hadamard


class GhostHadamardMixing(nn.Module):
    """
    Ghost Module style mixing using Hadamard Transform.
    
    Splits the output channels into two parts:
    1. Primary: Standard 1x1 Conv (Learnable, Heavy) -> Captures complex features
    2. Secondary: Weighted Hadamard (Learnable, Cheap) -> Generates redundant/frequency features
    
    This provides a middle ground between full Conv1x1 and pure WHT.
    """
    def __init__(self, in_channels, out_channels, ratio=0.5):
        super().__init__()
        self.out_channels = out_channels

        # Calculate split
        self.primary_out = int(out_channels * ratio)
        self.secondary_out = out_channels - self.primary_out

        # 1. Primary Path (Standard Conv)
        self.primary = nn.Conv2d(in_channels, self.primary_out, kernel_size=1, bias=False)

        # 2. Secondary Path (WHT)
        # Note: We use the existing HadamardMixing class
        if self.secondary_out > 0:
            self.secondary = HadamardMixing(in_channels, self.secondary_out, learnable=True)
        else:
            self.secondary = None

    def forward(self, x):
        out1 = self.primary(x)

        if self.secondary is not None:
            out2 = self.secondary(x)
            return torch.cat([out1, out2], dim=1)

        return out1


class HadamardMixing(nn.Module):
    """
    Weighted Walsh-Hadamard Transform for channel mixing.
    
    Combines the computational efficiency of fixed Hadamard transforms with
    learnable per-channel scaling weights. This allows the network to learn
    which spectral components (Hadamard basis vectors) are important.
    
    Formula: x_out = (diag(scale) @ H @ x_in) / sqrt(in_channels)
    
    where:
        - H is the fixed Hadamard matrix (orthogonal, values ±1)
        - scale is a learnable per-output-channel weight vector
        
    Benefits over fixed WHT:
        - Learnable importance weighting for each output channel
        - Retains orthogonality structure of Hadamard basis
        - Minimal parameter overhead (only out_channels params)
        - Much better gradient flow vs completely fixed transform
    
    Benefits over 1x1 Conv:
        - Structured transform reduces overfitting risk
        - Fewer parameters (out_channels vs in_channels * out_channels)
        - Hadamard basis provides multi-scale frequency mixing
    """

    def __init__(self, in_channels, out_channels, learnable=True, init_scale=1.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.learnable = learnable

        # Calculate the required Hadamard dimension (must be power of 2)
        max_ch = max(in_channels, out_channels)
        base_dim = 2 ** (max_ch - 1).bit_length()

        h_mat = hadamard(base_dim)

        # Slice to select subset of orthogonal basis vectors
        # Take first 'out_channels' rows and 'in_channels' columns
        weights_np = h_mat[:out_channels, :in_channels]
        weights = torch.from_numpy(weights_np).float()

        # Scale by 1/sqrt(in_channels) to maintain variance (Xavier/He style)
        weights = weights / (in_channels**0.5)

        # Register fixed Hadamard basis as buffer (not trained)
        self.register_buffer("h_basis", weights.view(out_channels, in_channels, 1, 1))

        # Learnable per-channel scaling weights
        if learnable:
            # Initialize close to 1.0 so initial behavior ≈ unweighted WHT
            self.scale = nn.Parameter(torch.ones(out_channels, 1, 1, 1) * init_scale)
        else:
            self.register_buffer("scale", torch.ones(out_channels, 1, 1, 1))

    def forward(self, x):
        # Apply scaled Hadamard transform
        # h_kernel = scale * h_basis (broadcasted multiplication)
        h_kernel = self.scale * self.h_basis
        return F.conv2d(x, h_kernel)

    def extra_repr(self):
        return (
            f"in_channels={self.in_channels}, out_channels={self.out_channels}, "
            f"learnable={self.learnable}"
        )


class UnweightedHadamardMixing(nn.Module):
    """
    Original fixed (non-trainable) Walsh-Hadamard Transform.
    Kept for comparison/ablation studies.
    
    x_out = (H @ x_in) / sqrt(in_channels)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        max_ch = max(in_channels, out_channels)
        base_dim = 2 ** (max_ch - 1).bit_length()

        h_mat = hadamard(base_dim)
        weights_np = h_mat[:out_channels, :in_channels]
        weights = torch.from_numpy(weights_np).float()
        weights = weights / (in_channels**0.5)

        self.register_buffer("h_kernel", weights.view(out_channels, in_channels, 1, 1))

    def forward(self, x):
        return F.conv2d(x, self.h_kernel)
