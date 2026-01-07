import torch
import torch.nn.functional as F

# Corrected imports
from microfeatex.utils.geometry import (
    scale_homography,
    warp_features,
    create_valid_mask,
)


def compute_total_loss(student_out1, student_out2, teacher_out, H, mask, config):
    """
    Computes weighted sum of:
    1. Distillation Loss (Student Proj <-> Teacher Desc)
    2. Siamese Loss (Student View1 Warped <-> Student View2)
    3. Quantization Loss (DeepBit)
    """

    # Unpack
    # Student: heatmap, binary_raw (tanh), projected_desc
    _, bin1_raw, proj1 = student_out1
    _, bin2_raw, _ = student_out2

    # Teacher
    teacher_desc = teacher_out["descriptors"]

    # --- 1. Distillation Loss ---
    # Cosine distance between student projection and teacher
    loss_distill = 1 - F.cosine_similarity(proj1, teacher_desc, dim=1).mean()

    # --- 2. Siamese Stability Loss ---
    # Warp view1 descriptors to view2 using H

    # 1. Scale Homography for feature map dimensions (uses helper)
    H_feat = scale_homography(H, stride=8.0)

    # 2. Warp View 1 descriptors to align with View 2 (uses helper)
    h_map, w_map = bin1_raw.shape[2], bin1_raw.shape[3]
    desc1_warped = warp_features(bin1_raw, H_feat, dsize=(h_map, w_map))

    # 3. Create Valid Mask (uses helper)
    # Generates a mask of valid pixels (1s) where the warp is valid, 0s elsewhere
    mask_warped = create_valid_mask(h_map, w_map, H_feat, bin1_raw.device)

    # MSE between warped view1 and view2 (L2 on Tanh space)
    # We multiply by mask_warped to ignore regions that fell outside the image boundary
    diff = (desc1_warped - bin2_raw) ** 2
    loss_siamese = (diff * mask_warped).sum() / (mask_warped.sum() + 1e-6)

    # --- 3. DeepBit Quantization Losses ---
    loss_quant = (bin1_raw.abs() - 1).pow(2).mean()

    # Weighted Sum (Weights can be moved to config)
    w_distill = 1.0
    w_siamese = 2.0
    w_quant = 0.1

    total = (
        (w_distill * loss_distill) + (w_siamese * loss_siamese) + (w_quant * loss_quant)
    )

    return {
        "total": total,
        "distill": loss_distill,
        "siamese": loss_siamese,
        "quant": loss_quant,
    }
