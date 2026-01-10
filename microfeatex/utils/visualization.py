import torch
import torch.nn.functional as F
import torchvision
import cv2
import numpy as np
from torch.utils.tensorboard import SummaryWriter


# Define available colormaps
COLORMAPS = {
    "jet": cv2.COLORMAP_JET,
    "inferno": cv2.COLORMAP_INFERNO,
    "turbo": cv2.COLORMAP_TURBO,
    "hot": cv2.COLORMAP_HOT,
    "bone": cv2.COLORMAP_BONE,
    "ocean": cv2.COLORMAP_OCEAN,
    "viridis": cv2.COLORMAP_VIRIDIS,
    "plasma": cv2.COLORMAP_PLASMA,
    "magma": cv2.COLORMAP_MAGMA,
    "cividis": cv2.COLORMAP_CIVIDIS,
}


class Visualizer:
    def __init__(self, log_dir, colormap_name="jet"):
        self.writer = SummaryWriter(log_dir=log_dir)

        self.colormap_name = colormap_name.lower()
        if self.colormap_name not in COLORMAPS:
            print(
                f"Warming: Colormap '{colormap_name}' not, found. Defaulting to 'jet,'"
            )
            self.colormap_name = "jet"

        self.cv2_colormap = COLORMAPS[self.colormap_name]

    def close(self):
        self.writer.close()

    def log_scalars(self, scalar_dict, step, prefix="Loss"):
        for k, v in scalar_dict.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            self.writer.add_scalar(f"{prefix}/{k}", v, step)

    def apply_colormap(self, heatmap):
        """Apply colormap to grayscale heatmap [B, 1, H, W] -> [B, 3, H, W]"""
        B, C, H, W = heatmap.shape
        heatmap_np = heatmap.detach().cpu().numpy()

        colored_batch = []
        for i in range(B):
            # Normalize to 0-255
            h_img = heatmap_np[i, 0]
            # h_img = (h_img - h_img.min()) / (h_img.max() - h_img.min() + 1e-8)
            h_img = np.clip(h_img, 0, 1)
            h_img = (h_img * 255).astype(np.uint8)

            # Apply OpenCV colormap
            c_img = cv2.applyColorMap(h_img, self.cv2_colormap)
            c_img = cv2.cvtColor(c_img, cv2.COLOR_BGR2RGB)  # OpenCV is BGR

            # To Tensor [3, H, W]
            c_tensor = torch.from_numpy(c_img).permute(2, 0, 1).float() / 255.0
            colored_batch.append(c_tensor)

        return torch.stack(colored_batch).to(heatmap.device)

    def create_overlay(self, image, heatmap, alpha=0.5):
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)

        # Ensure heatmap is colored [B, 3, H, W]
        if heatmap.shape[1] == 1:
            heatmap = self.apply_colormap(heatmap)

        if image.shape[2:] != heatmap.shape[2:]:
            heatmap = F.interpolate(
                heatmap, size=image.shape[2:], mode="bilinear", align_corners=False
            )

        overlay = alpha * heatmap + (1 - alpha) * image
        return torch.clamp(overlay, 0, 1)

    def process_heatmap_for_vis(self, heatmap):
        # Handle SuperPoint 65-channel raw output
        if heatmap.shape[1] == 65:
            dense_map = heatmap[:, :-1, :, :]
            return F.pixel_shuffle(dense_map, 8)
        return heatmap

    def draw_matches(self, img1, img2, kpts1, kpts2):
        """
        Draws side-by-side matches.
        Input: img1 [1, H, W], kpts1 [N, 2] (x, y)
        """
        # Convert to numpy/cv2
        h, w = img1.shape[-2], img1.shape[-1]
        i1 = (img1.squeeze().cpu().numpy() * 255).astype(np.uint8)
        i2 = (img2.squeeze().cpu().numpy() * 255).astype(np.uint8)

        # Create canvas
        canvas = np.zeros((h, w * 2), dtype=np.uint8)
        canvas[:, :w] = i1
        canvas[:, w:] = i2
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)

        # Draw lines
        # Limit to 50 random matches to avoid clutter
        num_matches = len(kpts1)
        indices = np.random.choice(num_matches, min(num_matches, 50), replace=False)

        for idx in indices:
            pt1 = (int(kpts1[idx, 0]), int(kpts1[idx, 1]))
            pt2 = (int(kpts2[idx, 0]) + w, int(kpts2[idx, 1]))  # Offset x by width

            color = tuple(np.random.randint(0, 255, 3).tolist())
            cv2.line(canvas, pt1, pt2, color, 1, cv2.LINE_AA)
            cv2.circle(canvas, pt1, 2, color, -1)
            cv2.circle(canvas, pt2, 2, color, -1)

        return torch.from_numpy(canvas).permute(2, 0, 1).float() / 255.0

    @torch.no_grad()
    def log_advanced_visuals(
        self, step, img1, img2, s_heat, t_heat, desc1, desc2, s_rel
    ):
        """
        Logs:
        1. Heatmap comparison
        2. Reliability Map
        3. Feature Matches (calculated on the fly)
        """
        N = min(img1.shape[0], 3)  # Only log first 3 images to save space

        # Process Heatmaps
        s_heat = self.process_heatmap_for_vis(s_heat[:N])
        t_heat = self.process_heatmap_for_vis(t_heat[:N])

        # Normalize
        s_heat = (s_heat - s_heat.min()) / (s_heat.max() - s_heat.min() + 1e-8)
        t_heat = (t_heat - t_heat.min()) / (t_heat.max() - t_heat.min() + 1e-8)

        # Reliability is [B, 1, H/8, W/8], interpolate up
        rel_map = F.interpolate(s_rel[:N], size=img1.shape[2:], mode="bilinear")

        # 3. Create Overlays
        s_overlay = self.create_overlay(img1[:N], s_heat)
        t_overlay = self.create_overlay(img1[:N], t_heat)
        rel_overlay = self.create_overlay(img1[:N], rel_map, alpha=0.6)

        # Compute Matches for the first image pair (On-the-fly Matcher)
        # Simple grid sampling for visualization
        B, C, Hc, Wc = desc1.shape
        match_img = torch.zeros_like(s_overlay[0])

        # Flatten descriptors
        d1 = F.normalize(desc1[0].reshape(C, -1).t(), p=2, dim=1)  # [N, C]
        d2 = F.normalize(desc2[0].reshape(C, -1).t(), p=2, dim=1)  # [N, C]

        # Match (Brute Force)
        sim = d1 @ d2.t()
        vals, idxs = sim.max(dim=1)

        # Filter weak matches
        mask = vals > 0.85

        if mask.sum() > 10:
            # Convert grid indices back to (x, y)
            grid_y, grid_x = torch.meshgrid(
                torch.arange(Hc), torch.arange(Wc), indexing="ij"
            )
            grid_pts = (
                torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2).to(desc1.device)
            )

            src_pts = grid_pts[mask] * 8 + 4  # Scale up to original res
            dst_pts = grid_pts[idxs[mask]] * 8 + 4

            match_img = self.draw_matches(img1[0:1], img2[0:1], src_pts, dst_pts)

        # Log to TensorBoard
        self.writer.add_image(
            "1_Heatmaps/Student_vs_Teacher",
            torchvision.utils.make_grid(torch.cat([s_overlay, t_overlay]), nrow=N),
            step,
        )

        self.writer.add_image(
            "2_Reliability/Student_Confidence",
            torchvision.utils.make_grid(rel_overlay, nrow=N),
            step,
        )

        self.writer.add_image("3_Matches/Batch_Img0_vs_Img1", match_img, step)
