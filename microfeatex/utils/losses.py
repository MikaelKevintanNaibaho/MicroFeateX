import torch
import torch.nn as nn
import torch.nn.functional as F
from microfeatex.utils.geometry import (
    scale_homography,
    warp_features,
    create_valid_mask,
)


class MicroFeatEXLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.weights = config["training"].get("loss_weights", {})

        # Cache weights
        self.w_heat = self.weights.get("heatmap", 1000.0)
        self.w_distill = self.weights.get("distill", 1.0)
        self.w_siamese = self.weights.get("siamese", 2.0)
        self.w_quant = self.weights.get("quant", 0.1)
        self.distill_mode = config["training"].get("distill_mode", "cosine")

    def hard_triplet_loss(self, student_desc, teacher_desc, margin=1.0):
        """
        Computes Batch-Hard Triplet Loss for Dense Maps.

        Args:
            student_desc: [B, C, H, W]
            teacher_desc: [B, C, H, W]
        """
        # 1. Normalize
        student_desc = F.normalize(student_desc, p=2, dim=1)
        teacher_desc = F.normalize(teacher_desc, p=2, dim=1)

        B, C, H, W = student_desc.shape

        # 2. Reshape to (Pixel_Count, Batch, Channels)
        # We process each spatial location (H, W) as its own batch of examples
        s_flat = student_desc.permute(2, 3, 0, 1).reshape(H * W, B, C)
        t_flat = teacher_desc.permute(2, 3, 0, 1).reshape(H * W, B, C)

        # 3. Compute Similarity Matrix per pixel location
        # Shape: (H*W, B, C) x (H*W, C, B) -> (H*W, B, B)
        # This gives us the sim between Student(img_i) and Teacher(img_j) at every pixel
        sim_mat = torch.matmul(s_flat, t_flat.transpose(1, 2))

        # 4. Mask Diagonal (Self = Positive)
        # We want to find the hardest negative (best match in WRONG image)
        eye = torch.eye(B, device=student_desc.device).unsqueeze(0)  # [1, B, B]
        # Subtract huge number from diagonal so it's never selected as "max" (hardest negative)
        sim_mat = sim_mat - (eye * 10.0)

        # 5. Hardest Negative
        # Max over the Teacher-Batch dimension -> [H*W, B]
        hard_neg_sim, _ = sim_mat.max(dim=2)

        # 6. Positive Similarity (Diagonal elements)
        # Cosine sim between S[b] and T[b] at every pixel -> [H*W, B]
        pos_sim = F.cosine_similarity(s_flat, t_flat, dim=2)

        # 7. Loss Calculation
        # Loss = ReLU( (1 - pos_sim) - (1 - neg_sim) + margin )
        #      = ReLU( neg_sim - pos_sim + margin )
        loss = F.relu(hard_neg_sim - pos_sim + margin)

        return loss.mean()

    def compute_heatmap_loss(self, student_heat, teacher_out):
        # Extract teacher heatmap based on output format
        if isinstance(teacher_out, dict):
            t_heat = teacher_out.get("scores", teacher_out.get("semi"))
        else:
            t_heat = teacher_out

        # Handle SuperPoint format (65 channels)
        if t_heat.shape[1] == 65:
            t_probs = F.softmax(t_heat, dim=1)
            dense = t_probs[:, :-1, :, :]  # Drop dustbin
            t_heat = F.pixel_shuffle(dense, 8)

        # Align sizes
        if t_heat.shape[2:] != student_heat.shape[2:]:
            t_heat = F.interpolate(
                t_heat,
                size=student_heat.shape[2:],
                mode="bilinear",
                align_corners=False,
            )

        pred = torch.clamp(student_heat, min=1e-6, max=1.0 - 1e-6)

        # 2. Focal Weights
        alpha = 2.0
        beta = 4.0

        # 'pos_inds' are where teacher says there IS a corner
        pos_inds = t_heat.gt(0.01).float()
        neg_inds = 1.0 - pos_inds

        # Loss Calculation
        # If teacher says YES: Penalize if prediction is low (1 - pred)^alpha
        pos_loss = pos_inds * torch.pow(1 - pred, alpha) * torch.log(pred + 1e-12)

        # If teacher says NO: Penalize if pred is high, but penalize LESS if near a corner (1-t_heat)^beta
        neg_loss = (
            neg_inds
            * torch.pow(1 - t_heat, beta)
            * torch.pow(pred, alpha)
            * torch.log(1 - pred + 1e-12)
        )

        num_pos = pos_inds.sum()
        loss = -(pos_loss + neg_loss).sum()

        if num_pos == 0:
            return loss
        else:
            return loss / num_pos

    def forward(self, student_out1, student_out2, teacher_out, H, mask):
        heat1, bin1_raw, proj1 = student_out1
        _, bin2_raw, _ = student_out2
        teacher_desc = teacher_out["descriptors"]

        # 1. Heatmap Loss
        loss_heatmap = self.compute_heatmap_loss(heat1, teacher_out)

        # 2. Distillation Loss
        if self.distill_mode == "triplet":
            loss_distill = self.hard_triplet_loss(proj1, teacher_desc)
        else:
            loss_distill = 1 - F.cosine_similarity(proj1, teacher_desc, dim=1).mean()

        # 3. Siamese Loss
        H_feat = scale_homography(H, stride=8.0)
        h_map, w_map = bin1_raw.shape[2], bin1_raw.shape[3]

        desc1_warped = warp_features(bin1_raw, H_feat, dsize=(h_map, w_map))
        mask_warped = create_valid_mask(h_map, w_map, H_feat, bin1_raw.device)

        diff = (desc1_warped - bin2_raw) ** 2
        loss_siamese = (diff * mask_warped).sum() / (mask_warped.sum() + 1e-6)

        # 4. Quantization Loss
        loss_quant = (bin1_raw.abs() - 1).pow(2).mean()

        total = (
            self.w_heat * loss_heatmap
            + self.w_distill * loss_distill
            + self.w_siamese * loss_siamese
            + self.w_quant * loss_quant
        )

        return {
            "total": total,
            "heatmap": loss_heatmap,
            "distill": loss_distill,
            "siamese": loss_siamese,
            "quant": loss_quant,
        }
