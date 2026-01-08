import torch
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter


class Visualizer:
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)

    def close(self):
        self.writer.close()

    def log_scalars(self, scalar_dict, step, prefix="Loss"):
        """Log a dictionary of scalars."""
        for k, v in scalar_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.writer.add_scalar(f"{prefix}/{k}", v, step)

    def apply_colormap(self, heatmap, colormap="jet"):
        """Apply colormap to grayscale heatmap."""
        B, C, H, W = heatmap.shape
        heatmap = heatmap.squeeze(1)  # [B, H, W]

        if colormap == "jet":
            colored = torch.zeros(B, 3, H, W, device=heatmap.device)
            colored[:, 0] = torch.clamp(1.5 * heatmap - 0.25, 0, 1)  # Red
            colored[:, 1] = torch.where(
                heatmap < 0.5, 2 * heatmap, -2 * heatmap + 2
            )  # Green
            colored[:, 2] = torch.clamp(-1.5 * heatmap + 1.25, 0, 1)  # Blue
        else:
            colored = heatmap.unsqueeze(1).repeat(1, 3, 1, 1)

        return colored

    def create_overlay(self, image, heatmap, alpha=0.4):
        """Overlay colored heatmap on image."""
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)

        if image.shape[2:] != heatmap.shape[2:]:
            heatmap = F.interpolate(
                heatmap, size=image.shape[2:], mode="bilinear", align_corners=False
            )

        overlay = alpha * heatmap + (1 - alpha) * image
        return torch.clamp(overlay, 0, 1)

    def process_heatmap_for_vis(self, heatmap):
        """Normalize and format heatmap for visualization."""
        # Handle SuperPoint 65-channel raw output
        if heatmap.shape[1] == 65:
            dense_map = heatmap[:, :-1, :, :]
            return F.pixel_shuffle(dense_map, 8)

        # Handle RGB or other multi-channel
        if heatmap.shape[1] == 3:
            return (
                0.299 * heatmap[:, 0:1]
                + 0.587 * heatmap[:, 1:2]
                + 0.114 * heatmap[:, 2:3]
            )

        return heatmap

    def log_training_images(self, step, img1, img2, mask, student_raw, teacher_raw):
        """Log comprehensive grid of images."""
        N = min(img1.shape[0], 4)

        # Preprocessing
        s_heat = self.process_heatmap_for_vis(student_raw[:N])
        t_heat = self.process_heatmap_for_vis(teacher_raw[:N])

        # Normalize 0-1
        s_heat = (s_heat - s_heat.min()) / (s_heat.max() - s_heat.min() + 1e-8)
        t_heat = (t_heat - t_heat.min()) / (t_heat.max() - t_heat.min() + 1e-8)

        # Colorize
        s_colored = self.apply_colormap(s_heat)
        t_colored = self.apply_colormap(t_heat)

        # Overlay
        s_overlay = self.create_overlay(img1[:N], s_colored)
        t_overlay = self.create_overlay(img1[:N], t_colored)

        # Create Grids
        pad = 2
        grids = {
            "1_Images/View1_vs_View2": [img1[:N], img2[:N]],
            "2_Heatmaps/Student_vs_Teacher": [s_colored, t_colored],
            "3_Overlays/Student_vs_Teacher": [s_overlay, t_overlay],
        }

        for name, tensors in grids.items():
            grid = torchvision.utils.make_grid(
                torch.cat(tensors), nrow=N, padding=pad, normalize=False
            )
            self.writer.add_image(name, grid, step)
