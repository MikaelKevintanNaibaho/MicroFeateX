import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import hadamard


class HadamardMixing(nn.Module):
    """
    Replaces 1x1 Conv with a fixed, non-trainable Walsh-Hadamard Transform.
    x_out = (H @ x_in) / sqrt(in_channels)
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # Calculate the required Hadamard dimension
        # the dimension must be a power of 2 >= in_channels
        max_ch = max(in_channels, out_channels)
        base_dim = 2 ** (max_ch - 1).bit_length()

        h_mat = hadamard(base_dim)

        # Slice and convert to tensor
        # take the first 'out_channels' rows and 'in_channels' columns.
        # selects a subset of othogonal codes.
        weights_np = h_mat[:out_channels, :in_channels]
        weights = torch.from_numpy(weights_np).float()

        # Scale by 1/sqrt(in_channels) to maintain variance (Xavier/He style)
        weights = weights / (in_channels**0.5)

        # Reshape to (Out, In, H, W) -> (out, in, 1, 1)
        self.register_buffer("h_kernel", weights.view(out_channels, in_channels, 1, 1))

    def forward(self, x):
        return F.conv2d(x, self.h_kernel)
