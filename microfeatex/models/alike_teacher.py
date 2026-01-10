import torch
import torch.nn as nn
from third_party.alike_wrapper import get_alike_model


class AlikeTeacher(nn.Module):
    def __init__(self, model_name="alike-t", device="cuda"):
        super().__init__()
        self.net = get_alike_model(model_name, device=device)

        # Freeze
        for param in self.net.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Input: [B, 1, H, W] normalized 0-1.
        """
        # 1. Adapt Input: Grayscale -> RGB
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # 2. Forward Pass using extract_dense_map
        # This handles batching and padding automatically.
        with torch.no_grad():
            # returns: descriptor_map, scores_map
            # Note: extract_dense_map returns (desc, scores) NOT a dict by default
            desc_map, score_map = self.net.extract_dense_map(x)

        return {
            "heatmap": score_map,  # [B, 1, H, W]
            "descriptors": desc_map,  # [B, 64, H, W]
        }
