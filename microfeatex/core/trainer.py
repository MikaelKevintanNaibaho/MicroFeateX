import torch
import torch.optim as optim
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import torchvision
import torch.nn.functional as F

from microfeatex.utils.losses import compute_total_loss


class Trainer:
    def __init__(self, student, teacher, train_loader, val_loader, config, augmenter):
        self.student = student
        self.teacher = teacher
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.augmenter = augmenter
        self.device = config["system"]["device"]

        self.optimizer = optim.Adam(
            self.student.parameters(), lr=config["training"]["lr"]
        )
        self.writer = SummaryWriter(log_dir=config["paths"]["log_dir"])

        self.start_epoch = 0
        self.best_loss = float("inf")

        # Directory setup
        os.makedirs(config["paths"]["checkpoint_dir"], exist_ok=True)

    def load_checkpoint(self, path):
        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        self.student.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {self.start_epoch}")

    def save_checkpoint(self, epoch, loss, is_best=False):
        state = {
            "epoch": epoch,
            "model_state": self.student.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss": loss,
        }
        # Save latest
        last_path = os.path.join(self.config["paths"]["checkpoint_dir"], "last.pth")
        torch.save(state, last_path)

        # Save best
        if is_best:
            best_path = os.path.join(
                self.config["paths"]["checkpoint_dir"], "best_model.pth"
            )
            torch.save(state, best_path)

        # Periodic save
        if epoch % self.config["training"]["save_interval"] == 0:
            p = os.path.join(
                self.config["paths"]["checkpoint_dir"], f"epoch_{epoch}.pth"
            )
            torch.save(state, p)

    def apply_colormap(self, heatmap, colormap="jet"):
        """
        Apply colormap to grayscale heatmap for better visualization.

        Args:
            heatmap: [B, 1, H, W] normalized tensor (0-1)
            colormap: 'jet', 'hot', 'viridis', 'turbo'

        Returns:
            [B, 3, H, W] colored heatmap
        """
        B, C, H, W = heatmap.shape
        assert C == 1, "Input must be single channel"

        # Squeeze channel dimension
        heatmap = heatmap.squeeze(1)  # [B, H, W]

        if colormap == "jet":
            # Custom Jet colormap (Blue -> Cyan -> Green -> Yellow -> Red)
            colored = torch.zeros(B, 3, H, W, device=heatmap.device)

            # Red channel
            colored[:, 0] = torch.clamp(1.5 * heatmap - 0.25, 0, 1)

            # Green channel
            colored[:, 1] = torch.where(heatmap < 0.5, 2 * heatmap, -2 * heatmap + 2)

            # Blue channel
            colored[:, 2] = torch.clamp(-1.5 * heatmap + 1.25, 0, 1)

        elif colormap == "hot":
            # Hot colormap (Black -> Red -> Yellow -> White)
            colored = torch.zeros(B, 3, H, W, device=heatmap.device)
            colored[:, 0] = torch.clamp(3 * heatmap, 0, 1)
            colored[:, 1] = torch.clamp(3 * heatmap - 1, 0, 1)
            colored[:, 2] = torch.clamp(3 * heatmap - 2, 0, 1)

        else:
            # Default: grayscale
            colored = heatmap.unsqueeze(1).repeat(1, 3, 1, 1)

        return colored

    def create_overlay(self, image, heatmap, alpha=0.4, brightness=1.2):
        """
        Overlay colored heatmap on image with brightness adjustment.

        Args:
            image: [B, C, H, W] original image
            heatmap: [B, 3, H, W] colored heatmap
            alpha: transparency of heatmap (0-1)
            brightness: brightness multiplier for final image

        Returns:
            [B, 3, H, W] overlaid image
        """
        # Convert grayscale to RGB if needed
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)

        # Ensure same size
        if image.shape[2:] != heatmap.shape[2:]:
            heatmap = F.interpolate(
                heatmap, size=image.shape[2:], mode="bilinear", align_corners=False
            )

        # Blend with brightness adjustment
        overlay = alpha * heatmap + (1 - alpha) * image * brightness
        return torch.clamp(overlay, 0, 1)

    def threshold_heatmap(self, heatmap, threshold=0.015, use_adaptive=True):
        """
        Apply thresholding to remove noise and weak activations.

        Args:
            heatmap: [B, 1, H, W] tensor
            threshold: fixed threshold value (0-1)
            use_adaptive: if True, use adaptive threshold based on statistics

        Returns:
            [B, 1, H, W] thresholded heatmap
        """
        if use_adaptive:
            # Adaptive thresholding: use mean + std
            B = heatmap.shape[0]
            thresholded = torch.zeros_like(heatmap)

            for b in range(B):
                vals = heatmap[b]
                mean = vals.mean()
                std = vals.std()
                # Keep only activations above mean + 0.5*std
                adaptive_thresh = mean + 0.5 * std
                thresholded[b] = torch.where(
                    vals > adaptive_thresh, vals, torch.zeros_like(vals)
                )
            return thresholded
        else:
            # Fixed thresholding
            return torch.where(heatmap > threshold, heatmap, torch.zeros_like(heatmap))

    def extract_keypoints(
        self, heatmap, top_k=1000, nms_radius=4, threshold=0.015, border_dist=8
    ):
        """
        Extract keypoints using NMS, ignoring image borders.
        """
        B, C, H, W = heatmap.shape

        # 1. Apply Max Pooling (NMS)
        max_pooled = F.max_pool2d(
            heatmap, kernel_size=2 * nms_radius + 1, stride=1, padding=nms_radius
        )

        # 2. Find local maxima above threshold
        is_peak = (heatmap == max_pooled) & (heatmap > threshold)

        # 3. --- NEW: Remove Border Artifacts ---
        # Set borders to False so we don't pick up padding noise
        is_peak[:, :, :border_dist, :] = False  # Top
        is_peak[:, :, -border_dist:, :] = False  # Bottom
        is_peak[:, :, :, :border_dist] = False  # Left
        is_peak[:, :, :, -border_dist:] = False  # Right

        keypoint_map = torch.zeros_like(heatmap)

        for b in range(B):
            # Get coordinates
            peaks = is_peak[b, 0].nonzero(as_tuple=False)

            if len(peaks) > 0:
                scores = heatmap[b, 0, peaks[:, 0], peaks[:, 1]]

                # Keep top-k
                if len(scores) > top_k:
                    _, top_indices = torch.topk(scores, top_k)
                    peaks = peaks[top_indices]
                    scores = scores[top_indices]

                keypoint_map[b, 0, peaks[:, 0], peaks[:, 1]] = scores

        return keypoint_map

    def visualize_keypoints(self, image, keypoint_map, color=[1, 0, 0], radius=2):
        """
        Visualize keypoints as colored circles on image.

        Args:
            image: [B, C, H, W] image tensor
            keypoint_map: [B, 1, H, W] keypoint locations and scores
            color: [R, G, B] color for keypoints
            radius: radius of keypoint markers

        Returns:
            [B, 3, H, W] image with keypoints drawn
        """
        B, C, H, W = keypoint_map.shape

        # Convert to RGB if grayscale
        if image.shape[1] == 1:
            vis_image = image.repeat(1, 3, 1, 1).clone()
        else:
            vis_image = image.clone()

        # Create circular kernel
        y, x = torch.meshgrid(
            torch.arange(-radius, radius + 1, device=image.device),
            torch.arange(-radius, radius + 1, device=image.device),
            indexing="ij",
        )
        circle_mask = (x**2 + y**2 <= radius**2).float()

        for b in range(B):
            # Get keypoint coordinates
            kps = keypoint_map[b, 0].nonzero(as_tuple=False)

            for kp in kps:
                y_center, x_center = kp[0].item(), kp[1].item()

                # Define bounds in image space
                y_start = max(0, y_center - radius)
                y_end = min(H, y_center + radius + 1)
                x_start = max(0, x_center - radius)
                x_end = min(W, x_center + radius + 1)

                # Calculate corresponding mask indices
                mask_y_start = radius - (y_center - y_start)
                mask_y_end = mask_y_start + (y_end - y_start)
                mask_x_start = radius - (x_center - x_start)
                mask_x_end = mask_x_start + (x_end - x_start)

                if y_end > y_start and x_end > x_start:
                    mask_crop = circle_mask[
                        mask_y_start:mask_y_end, mask_x_start:mask_x_end
                    ]

                    # Apply color to all channels
                    for c in range(3):
                        vis_image[b, c, y_start:y_end, x_start:x_end] = torch.where(
                            mask_crop > 0,
                            torch.tensor(color[c], device=image.device),
                            vis_image[b, c, y_start:y_end, x_start:x_end],
                        )

        return vis_image

    def create_difference_map(self, student_heat, teacher_heat):
        """
        Create a difference map showing where student and teacher disagree.

        Args:
            student_heat: [B, 1, H, W] student heatmap
            teacher_heat: [B, 1, H, W] teacher heatmap

        Returns:
            [B, 3, H, W] colored difference map (green=match, red=teacher>student, blue=student>teacher)
        """
        B, C, H, W = student_heat.shape

        # Compute difference
        diff = teacher_heat - student_heat

        # Create RGB channels
        diff_colored = torch.zeros(B, 3, H, W, device=student_heat.device)

        # Red: where teacher is stronger
        diff_colored[:, 0] = torch.clamp(diff.squeeze(1), 0, 1)

        # Green: where they agree (inverse of absolute difference)
        agreement = 1 - torch.abs(diff).squeeze(1)
        diff_colored[:, 1] = torch.clamp(agreement, 0, 1)

        # Blue: where student is stronger
        diff_colored[:, 2] = torch.clamp(-diff.squeeze(1), 0, 1)

        return diff_colored

    def enhance_contrast(self, heatmap, percentile=1):
        """
        Enhance contrast using percentile normalization.

        Args:
            heatmap: [B, 1, H, W] tensor
            percentile: percentage for min/max clipping

        Returns:
            [B, 1, H, W] contrast-enhanced tensor
        """
        B, C, H, W = heatmap.shape
        enhanced = torch.zeros_like(heatmap)

        for b in range(B):
            vals = heatmap[b].flatten()
            if vals.max() > 0:
                vmin = torch.quantile(vals, percentile / 100.0)
                vmax = torch.quantile(vals, 1.0 - percentile / 100.0)
                enhanced[b] = torch.clamp(
                    (heatmap[b] - vmin) / (vmax - vmin + 1e-8), 0, 1
                )

        return enhanced

    def process_heatmap_for_vis(self, heatmap):
        """
        Convert multi-channel heatmap to single-channel for visualization.
        """
        if heatmap.shape[1] == 1:
            return heatmap

        if heatmap.shape[1] == 3:
            gray = (
                0.299 * heatmap[:, 0:1]
                + 0.587 * heatmap[:, 1:2]
                + 0.114 * heatmap[:, 2:3]
            )
            return gray
        # --- FIX: Handle SuperPoint 65-channel output ---
        # If we see 65 channels, it is the raw SuperPoint output
        if heatmap.shape[1] == 65:
            # 1. Drop the 65th "dustbin" channel (garbage class)
            dense_map = heatmap[:, :-1, :, :]

            # 2. Pixel Shuffle: Reshape (B, 64, H/8, W/8) -> (B, 1, H, W)
            # This recovers the full HD resolution!
            return F.pixel_shuffle(dense_map, 8)

        # For multi-channel: take max activation
        heatmap_max, _ = torch.max(heatmap, dim=1, keepdim=True)
        return heatmap_max

    def log_visualizations(
        self, step, img1, img2, mask, student_heatmap, teacher_heatmap
    ):
        """
        Logs comprehensive grid of training visualizations to TensorBoard.
        """
        # Take the first N images from the batch
        N = min(img1.shape[0], 4)

        # Get target image size
        target_h, target_w = img1.shape[2], img1.shape[3]

        # Process heatmaps to single-channel
        student_heat = self.process_heatmap_for_vis(student_heatmap[:N])
        teacher_heat = self.process_heatmap_for_vis(teacher_heatmap[:N])

        # Resize heatmaps to match image resolution if needed
        if student_heat.shape[2:] != (target_h, target_w):
            student_heat = F.interpolate(
                student_heat,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )
        if teacher_heat.shape[2:] != (target_h, target_w):
            teacher_heat = F.interpolate(
                teacher_heat,
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            )

        # Normalize to 0-1 range
        student_heat = (student_heat - student_heat.min()) / (
            student_heat.max() - student_heat.min() + 1e-8
        )
        teacher_heat = (teacher_heat - teacher_heat.min()) / (
            teacher_heat.max() - teacher_heat.min() + 1e-8
        )

        # Apply thresholding to reduce noise
        student_thresh = self.threshold_heatmap(student_heat, use_adaptive=True)
        teacher_thresh = self.threshold_heatmap(teacher_heat, use_adaptive=True)

        # Enhance contrast
        student_enhanced = self.enhance_contrast(student_thresh, percentile=1)
        teacher_enhanced = self.enhance_contrast(teacher_thresh, percentile=1)

        # You can add a 'vis_top_k' parameter to your yaml under 'training' or 'model'
        vis_top_k = self.config["training"].get("vis_top_k", 500)

        # Extract keypoints (now at correct resolution)
        student_kps = self.extract_keypoints(
            student_heat, top_k=vis_top_k, nms_radius=4, threshold=0.015, border_dist=8
        )
        teacher_kps = self.extract_keypoints(
            teacher_heat, top_k=vis_top_k, nms_radius=4, threshold=0.015, border_dist=8
        )

        # Apply colormaps to thresholded heatmaps
        student_colored = self.apply_colormap(student_enhanced, colormap="jet")
        teacher_colored = self.apply_colormap(teacher_enhanced, colormap="jet")

        # Create overlays with better visibility
        student_overlay = self.create_overlay(
            img1[:N], student_colored, alpha=0.4, brightness=1.3
        )
        teacher_overlay = self.create_overlay(
            img1[:N], teacher_colored, alpha=0.4, brightness=1.3
        )

        # Visualize keypoints with larger radius for visibility
        student_kp_vis = self.visualize_keypoints(
            img1[:N], student_kps, color=[1, 0, 0], radius=3
        )
        teacher_kp_vis = self.visualize_keypoints(
            img1[:N], teacher_kps, color=[0, 1, 0], radius=3
        )

        # Create difference map
        diff_map = self.create_difference_map(student_enhanced, teacher_enhanced)

        # Ensure mask has channel dimension
        if len(mask.shape) == 3:
            mask = mask.unsqueeze(1)

        # Create grids with padding
        pad = 4
        grid_img1 = torchvision.utils.make_grid(
            img1[:N], normalize=False, nrow=2, padding=pad, pad_value=1
        )
        grid_img2 = torchvision.utils.make_grid(
            img2[:N], normalize=False, nrow=2, padding=pad, pad_value=1
        )
        grid_mask = torchvision.utils.make_grid(
            mask[:N].float(), normalize=False, nrow=2, padding=pad, pad_value=1
        )

        # Heatmap grids (colored)
        grid_student_heat = torchvision.utils.make_grid(
            student_colored, normalize=False, nrow=2, padding=pad, pad_value=1
        )
        grid_teacher_heat = torchvision.utils.make_grid(
            teacher_colored, normalize=False, nrow=2, padding=pad, pad_value=1
        )

        # Overlay grids
        grid_student_overlay = torchvision.utils.make_grid(
            student_overlay, normalize=False, nrow=2, padding=pad, pad_value=1
        )
        grid_teacher_overlay = torchvision.utils.make_grid(
            teacher_overlay, normalize=False, nrow=2, padding=pad, pad_value=1
        )

        # Keypoint grids
        grid_student_kps = torchvision.utils.make_grid(
            student_kp_vis, normalize=False, nrow=2, padding=pad, pad_value=1
        )
        grid_teacher_kps = torchvision.utils.make_grid(
            teacher_kp_vis, normalize=False, nrow=2, padding=pad, pad_value=1
        )

        # Difference grid
        grid_diff = torchvision.utils.make_grid(
            diff_map, normalize=False, nrow=2, padding=pad, pad_value=1
        )

        # Write to TensorBoard
        # 1. Original images
        self.writer.add_image("1_Images/A_View1_Original", grid_img1, step)
        self.writer.add_image("1_Images/B_View2_Warped", grid_img2, step)
        self.writer.add_image("1_Images/C_Valid_Mask", grid_mask, step)

        # 2. Clean heatmaps (thresholded and colored)
        self.writer.add_image("2_Heatmaps/A_Student_Clean", grid_student_heat, step)
        self.writer.add_image("2_Heatmaps/B_Teacher_Clean", grid_teacher_heat, step)

        # 3. Overlays (heatmaps on images)
        self.writer.add_image(
            "3_Overlays/A_Student_Overlay", grid_student_overlay, step
        )
        self.writer.add_image(
            "3_Overlays/B_Teacher_Overlay", grid_teacher_overlay, step
        )

        # 4. Keypoint detection
        self.writer.add_image("4_Keypoints/A_Student_Keypoints", grid_student_kps, step)
        self.writer.add_image("4_Keypoints/B_Teacher_Keypoints", grid_teacher_kps, step)

        # 5. Comparison
        self.writer.add_image("5_Comparison/Difference_Map", grid_diff, step)

        # Log statistics
        student_kp_count = (student_kps > 0).sum().item() / N
        teacher_kp_count = (teacher_kps > 0).sum().item() / N
        self.writer.add_scalar(
            "Stats/Student_Keypoints_PerImage", student_kp_count, step
        )
        self.writer.add_scalar(
            "Stats/Teacher_Keypoints_PerImage", teacher_kp_count, step
        )

    def train_loop(self):
        print("Starting training loop...")
        self.student.train()
        self.teacher.eval()

        # Visualization logging frequency (update images every N steps)
        image_log_interval = self.config["training"].get("image_log_interval", 100)
        print(f"Images will be logged every {image_log_interval} steps")

        try:
            for epoch in range(self.start_epoch, self.config["training"]["epochs"]):
                epoch_loss = 0.0
                progress = tqdm(self.train_loader, desc=f"Epoch {epoch}")

                for i, batch_img1 in enumerate(progress):
                    step = epoch * len(self.train_loader) + i

                    # 1. Move to GPU
                    batch_img1 = batch_img1.to(self.device)

                    # 2. Augmentation Pipeline (GPU)
                    img1, img2, H_mat, mask = self.augmenter(batch_img1)

                    # 3. Student Forward
                    outs1 = self.student(img1)
                    outs2 = self.student(img2)

                    # 4. Teacher Forward (Only on View 1)
                    with torch.no_grad():
                        teacher_out = self.teacher(img1)

                    # 5. Loss Calculation
                    loss_dict = compute_total_loss(
                        outs1, outs2, teacher_out, H_mat, mask, self.config
                    )

                    # 6. Optimization
                    self.optimizer.zero_grad()
                    loss_dict["total"].backward()
                    self.optimizer.step()

                    # 7. Logging Scalars
                    epoch_loss += loss_dict["total"].item()
                    progress.set_postfix({"loss": loss_dict["total"].item()})

                    if i % self.config["training"]["log_interval"] == 0:
                        self.writer.add_scalar(
                            "Loss/Total", loss_dict["total"].item(), step
                        )
                        self.writer.add_scalar(
                            "Loss/Siamese", loss_dict["siamese"].item(), step
                        )
                        self.writer.add_scalar(
                            "Loss/Distill", loss_dict["distill"].item(), step
                        )
                        self.writer.add_scalar(
                            "Loss/Heatmap", loss_dict["heatmap"].item(), step
                        )

                    # 8. Visualization Logging
                    if step % image_log_interval == 0:
                        # Extract student heatmap
                        student_heatmap = outs1[0]

                        # Extract teacher heatmap
                        if isinstance(teacher_out, dict):
                            if "scores" in teacher_out:
                                teacher_heatmap = teacher_out["scores"]
                            elif "semi" in teacher_out:
                                teacher_heatmap = teacher_out["semi"]
                            elif "heatmap" in teacher_out:
                                teacher_heatmap = teacher_out["heatmap"]
                            else:
                                teacher_heatmap = next(iter(teacher_out.values()))
                        elif isinstance(teacher_out, (tuple, list)):
                            teacher_heatmap = teacher_out[0]
                        else:
                            teacher_heatmap = teacher_out

                        # Ensure teacher heatmap has batch dimension
                        if len(teacher_heatmap.shape) == 3:
                            teacher_heatmap = teacher_heatmap.unsqueeze(1)

                        self.log_visualizations(
                            step,
                            img1,
                            img2,
                            mask,
                            student_heatmap,
                            teacher_heatmap,
                        )

                avg_train_loss = epoch_loss / len(self.train_loader)
                print(f"Epoch {epoch} complete. Avg Loss: {avg_train_loss:.4f}")

                avg_val_loss = self.validate()
                print(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f}")

                # Log to TensorBoard
                self.writer.add_scalar("Loss/Train_Epoch", avg_train_loss, epoch)
                self.writer.add_scalar("Loss/Val_Epoch", avg_val_loss, epoch)

                # Checkpointing (Use VAL loss for best model)
                is_best = avg_val_loss < self.best_loss
                if is_best:
                    print(f"New Best Model! (Val Loss: {avg_val_loss:.4f})")
                    self.best_loss = avg_val_loss

                self.save_checkpoint(epoch, avg_val_loss, is_best)

        except KeyboardInterrupt:
            print("\nTraining interrupted! Saving emergency checkpoint...")
            self.save_checkpoint(epoch, 0.0, is_best=False)
            print("Saved.")

        self.writer.close()

    def validate(self):
        """
        Runs validation loop on unseen data.
        """
        print("Running Validation...")
        self.student.eval()
        self.teacher.eval()
        self.augmenter.eval()  # Disable photometric noise for consistency

        total_val_loss = 0.0

        with torch.no_grad():
            for batch_img1 in tqdm(self.val_loader, desc="Validating"):
                batch_img1 = batch_img1.to(self.device)

                # Use same augmentations to generate pairs, but without random noise
                img1, img2, H_mat, mask = self.augmenter(batch_img1)

                # Forward Pass
                outs1 = self.student(img1)
                outs2 = self.student(img2)
                teacher_out = self.teacher(img1)

                # Calculate Loss
                loss_dict = compute_total_loss(
                    outs1, outs2, teacher_out, H_mat, mask, self.config
                )

                total_val_loss += loss_dict["total"].item()

        avg_val_loss = total_val_loss / len(self.val_loader)

        # Switch back to training mode
        self.student.train()
        self.augmenter.train()

        return avg_val_loss
