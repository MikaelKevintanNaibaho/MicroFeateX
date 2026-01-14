"""
microfeatex/training/losses.py - Complete Loss Functions
"""

import torch
import torch.nn.functional as F
from . import utils

from microfeatex.utils.logger import get_logger

logger = get_logger(__name__)

# Optional imports depending on environment
try:
    from third_party.alike_wrapper import extract_alike_kpts
except ImportError:
    extract_alike_kpts = None


def dual_softmax_loss(X, Y, temp=0.2):
    """
    Dual Softmax loss for descriptor matching.
    """
    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError("Error: X and Y shapes must match and be 2D matrices")

    dist_mat = (X @ Y.t()) / temp
    conf_matrix12 = F.log_softmax(dist_mat, dim=1)
    conf_matrix21 = F.log_softmax(dist_mat.t(), dim=1)

    with torch.no_grad():
        conf12 = torch.exp(conf_matrix12).max(dim=-1)[0]
        conf21 = torch.exp(conf_matrix21).max(dim=-1)[0]
        conf = conf12 * conf21

    target = torch.arange(len(X), device=X.device)

    loss = F.nll_loss(conf_matrix12, target) + F.nll_loss(conf_matrix21, target)

    return loss, conf


def smooth_l1_loss(input, target, beta=2.0, size_average=True):
    """Smooth L1 loss."""
    diff = torch.abs(input - target)
    loss = torch.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    return loss.mean() if size_average else loss.sum()


def fine_loss(f1, f2, pts1, pts2, fine_module, ws=7):
    """
    Compute Fine features and spatial loss.
    """
    C, H, W = f1.shape
    N = len(pts1)

    # Sort random offsets
    with torch.no_grad():
        a = -(ws // 2)
        b = ws // 2
        offset_gt = (a - b) * torch.rand(N, 2, device=f1.device) + b
        pts2_random = pts2 + offset_gt

    patches1 = (
        utils.crop_patches(f1.unsqueeze(0), (pts1 + 0.5).long(), size=ws)
        .view(C, N, ws * ws)
        .permute(1, 2, 0)
    )
    patches2 = (
        utils.crop_patches(f2.unsqueeze(0), (pts2_random + 0.5).long(), size=ws)
        .view(C, N, ws * ws)
        .permute(1, 2, 0)
    )

    # Apply transformer/refiner
    patches1, patches2 = fine_module(patches1, patches2)

    features = patches1.view(N, ws, ws, C)[:, ws // 2, ws // 2, :].view(N, 1, 1, C)
    patches2 = patches2.view(N, ws, ws, C)

    # Dot Product
    heatmap_match = (features * patches2).sum(-1)
    offset_coords = utils.subpix_softmax2d(heatmap_match)

    # Invert offset because center crop inverts it
    offset_gt = -offset_gt

    # MSE
    error = ((offset_coords - offset_gt) ** 2).sum(-1).mean()
    return error


def heatmap_mse_loss(student_heat, teacher_heat):
    """
    Computes MSE loss between the student's predicted heatmap and the teacher's target heatmap.

    Args:
        student_heat (torch.Tensor): [B, 1, H, W], range [0, 1]
        teacher_heat (torch.Tensor): [B, 1, H, W], range [0, 1]
    """
    if student_heat.shape != teacher_heat.shape:
        teacher_heat = F.interpolate(
            teacher_heat,
            size=student_heat.shape[2:],
            mode="bilinear",
            align_corners=False,
        )

    return F.mse_loss(student_heat, teacher_heat)


def superpoint_distill_loss(student_heat, teacher_scores, grid_size=8):
    """
    Distills the heatmap from the SuperPoint teacher to the student using Focal Loss.

    Args:
        student_heat (torch.Tensor): [B, 1, H, W] Student predicted heatmap (probabilities).
        teacher_scores (torch.Tensor): [B, 65, H/8, W/8] Teacher raw logits.
        grid_size (int): Pixel shuffle factor (default 8).

    Returns:
        torch.Tensor: Scalar loss value.
    """
    # 1. Process Teacher Output to Heatmap
    with torch.no_grad():
        # Softmax over 65 channels
        t_probs = F.softmax(teacher_scores, dim=1)
        # Remove "dustbin" channel (index 64)
        t_dense = t_probs[:, :-1, :, :]
        # Pixel Shuffle to full resolution [B, 1, H, W]
        t_heat = F.pixel_shuffle(t_dense, grid_size)

    # 2. Compute Focal Loss
    # Clamp for numerical stability
    pred = torch.clamp(student_heat, min=1e-6, max=1.0 - 1e-6)

    # Focal Loss Weights
    alpha = 2.0
    beta = 4.0

    # Define positive/negative regions based on teacher
    pos_inds = t_heat.gt(0.01).float()
    neg_inds = 1.0 - pos_inds

    # Loss Calculation
    pos_loss = pos_inds * torch.pow(1 - pred, alpha) * torch.log(pred)
    neg_loss = (
        neg_inds
        * torch.pow(1 - t_heat, beta)
        * torch.pow(pred, alpha)
        * torch.log(1 - pred)
    )

    loss = -(pos_loss + neg_loss).sum()
    num_pos = pos_inds.sum()

    return loss / (num_pos + 1e-6)


# ============================================================================
# ALIKE DISTILLATION LOSS - XFeat Style Position-Aware Loss
# ============================================================================


def alike_distill_loss(student_logits, teacher_heatmap, grid_size=8, debug=False):
    """
    Distills keypoint positions from teacher heatmap to student's 65-channel logits.

    Uses CONFIDENCE-WEIGHTED cross-entropy to handle class imbalance:
    - Keypoint cells are weighted by their teacher confidence (stronger keypoints matter more)
    - Dustbin cells are down-weighted relative to keypoint cells

    Args:
        student_logits (torch.Tensor): [65, H/8, W/8] Raw logits from student
        teacher_heatmap (torch.Tensor): [1, H, W] Teacher's keypoint heatmap at full resolution
        grid_size (int): Pixel shuffle factor (default 8)
        debug (bool): If True, print diagnostic information

    Returns:
        loss (torch.Tensor): Scalar loss value
        accuracy (float): Classification accuracy (0-1)
    """
    C, Hc, Wc = student_logits.shape  # C=65, Hc=H/8, Wc=W/8

    # FIX: Force teacher heatmap to exact grid-aligned resolution
    target_H, target_W = Hc * grid_size, Wc * grid_size
    if teacher_heatmap.shape[-2:] != (target_H, target_W):
        if debug:
            logger.debug(
                f"Resizing teacher heatmap: {teacher_heatmap.shape} -> (1, {target_H}, {target_W})"
            )
        teacher_heatmap = F.interpolate(
            teacher_heatmap.unsqueeze(0),
            size=(target_H, target_W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
    _, H, W = teacher_heatmap.shape

    # Downsample teacher heatmap to coarse grid [1, H/8, W/8]
    with torch.no_grad():
        # Use MAX pooling to get the peak confidence in each cell
        teacher_coarse = F.max_pool2d(
            teacher_heatmap.unsqueeze(0), kernel_size=grid_size, stride=grid_size
        ).squeeze(0)  # [1, H/8, W/8]

        # INCREASED THRESHOLD: Only train on confident keypoint cells
        keypoint_mask = teacher_coarse > 0.1  # [1, H/8, W/8]
        keypoint_mask = keypoint_mask.squeeze(0)  # [H/8, W/8]

        # Get confidence values for weighting (squeeze channel dim)
        confidence = teacher_coarse.squeeze(0)  # [H/8, W/8]

    # For each cell with a keypoint, find the peak sub-pixel location
    with torch.no_grad():
        # Reshape teacher to [H/8, W/8, 8, 8] to see inside each cell
        teacher_cells = (
            teacher_heatmap.view(1, Hc, grid_size, Wc, grid_size)
            .permute(0, 1, 3, 2, 4)
            .reshape(Hc, Wc, grid_size * grid_size)
        )

        # Find peak position within each cell (0-63)
        peak_positions = teacher_cells.argmax(dim=-1)  # [H/8, W/8]

        # Create labels: Use peak position for keypoint cells, 64 (dustbin) for others
        labels = torch.where(
            keypoint_mask,
            peak_positions,
            torch.full_like(peak_positions, 64),  # Dustbin channel
        )  # [H/8, W/8]

        # Create per-pixel weights
        # Keypoint cells: weight = confidence (0.1 to 1.0)
        # Dustbin cells: weight = 0.1 (down-weighted to balance classes)
        weights = torch.where(
            keypoint_mask,
            confidence,  # Use teacher confidence as weight
            torch.full_like(confidence, 0.1),  # Low weight for dustbin
        )

        # Boost keypoint weights to ensure they dominate the loss
        # This gives keypoints ~10x more importance than dustbin
        weights = torch.where(
            keypoint_mask,
            weights * 10.0,
            weights,
        )

    # Compute WEIGHTED Cross-Entropy Loss
    logits_flat = student_logits.permute(1, 2, 0).reshape(-1, C)
    labels_flat = labels.reshape(-1)
    weights_flat = weights.reshape(-1)

    # Per-sample cross-entropy (no reduction)
    loss_per_sample = F.cross_entropy(logits_flat, labels_flat, reduction="none")

    # Apply weights and average
    loss = (loss_per_sample * weights_flat).sum() / (weights_flat.sum() + 1e-6)

    # Compute Accuracy (only on keypoint cells with confidence > 0.1)
    with torch.no_grad():
        predictions = logits_flat.argmax(dim=-1).reshape(Hc, Wc)

        if keypoint_mask.sum() > 0:
            correct = (predictions[keypoint_mask] == labels[keypoint_mask]).float()
            accuracy = correct.mean().item()
        else:
            accuracy = 0.0

        # DEBUG: Print diagnostic info
        if debug:
            num_kpts = keypoint_mask.sum().item()
            num_dustbin = (labels_flat == 64).sum().item()
            student_preds = (
                predictions[keypoint_mask] if num_kpts > 0 else torch.tensor([])
            )
            teacher_labels = labels[keypoint_mask] if num_kpts > 0 else torch.tensor([])

            logger.debug(
                f"Shapes: student={student_logits.shape}, teacher={teacher_heatmap.shape}"
            )
            logger.debug(
                f"Teacher heatmap: min={teacher_heatmap.min():.4f}, max={teacher_heatmap.max():.4f}"
            )
            logger.debug(
                f"Keypoint cells (conf>0.1): {num_kpts} / {Hc * Wc} ({100*num_kpts/(Hc*Wc):.1f}%)"
            )
            logger.debug(f"Dustbin labels: {num_dustbin} / {Hc * Wc}")
            logger.debug(
                f"Weight sum (kpts): {weights[keypoint_mask].sum():.2f}, Weight sum (dustbin): {weights[~keypoint_mask].sum():.2f}"
            )
            if num_kpts > 0:
                logger.debug(
                    f"Sample teacher labels (first 10): {teacher_labels[:10].tolist()}"
                )
                logger.debug(
                    f"Sample student preds (first 10): {student_preds[:10].tolist()}"
                )
                logger.debug(
                    f"Student predicts dustbin on kpts: {(student_preds == 64).sum().item()} / {num_kpts}"
                )
            logger.debug(f"Loss={loss.item():.4f}, Acc={accuracy:.4f}")

    return loss, accuracy


def batch_alike_distill_loss(
    student_logits_batch, teacher_heatmap_batch, grid_size=8, debug=False
):
    """
    Batch version of alike_distill_loss.

    Args:
        student_logits_batch (torch.Tensor): [B, 65, H/8, W/8]
        teacher_heatmap_batch (torch.Tensor): [B, 1, H, W]
        grid_size (int): Pixel shuffle factor
        debug (bool): If True, print diagnostic info for first batch item

    Returns:
        loss (torch.Tensor): Scalar loss
        accuracy (float): Average accuracy across batch
    """
    B = student_logits_batch.shape[0]

    total_loss = 0
    total_acc = 0
    valid_count = 0

    for b in range(B):
        # Process each sample in batch (only debug first item to avoid spam)
        loss, acc = alike_distill_loss(
            student_logits_batch[b],
            teacher_heatmap_batch[b],
            grid_size,
            debug=(debug and b == 0),
        )

        # Accumulate loss (even for samples without keypoints to train dustbin)
        total_loss += loss
        valid_count += 1

        # Only accumulate accuracy if there were keypoints
        if acc > 0:
            total_acc += acc

    # Average
    avg_loss = total_loss / max(valid_count, 1)
    avg_acc = total_acc / max(valid_count, 1) if valid_count > 0 else 0.0

    return avg_loss, avg_acc


def hybrid_heatmap_loss(
    student_logits, teacher_heatmap, grid_size=8, position_weight=1.0, focal_weight=1.0
):
    """
    Combines position-aware classification loss with focal loss.

    Args:
        student_logits (torch.Tensor): [B, 65, H/8, W/8] Raw student output
        teacher_heatmap (torch.Tensor): [B, 1, H, W] Teacher heatmap
        grid_size (int): Pixel shuffle factor
        position_weight (float): Weight for position classification loss
        focal_weight (float): Weight for focal loss

    Returns:
        loss (torch.Tensor): Combined loss
        metrics (dict): Dictionary with individual losses and accuracy
    """
    # Position Classification Loss (teaches WHERE in each cell)
    loss_pos, acc = batch_alike_distill_loss(student_logits, teacher_heatmap, grid_size)

    # Focal Loss (teaches soft confidence values)
    with torch.no_grad():
        student_probs = F.softmax(student_logits, dim=1)
        student_corners = student_probs[:, :-1, :, :]  # Remove dustbin
        student_heatmap = F.pixel_shuffle(student_corners, grid_size)

    loss_focal = superpoint_focal_loss_internal(
        student_heatmap, teacher_heatmap, grid_size
    )

    # Combine
    total_loss = position_weight * loss_pos + focal_weight * loss_focal

    metrics = {
        "position_loss": loss_pos.item(),
        "focal_loss": loss_focal.item(),
        "position_accuracy": acc,
        "total_loss": total_loss.item(),
    }

    return total_loss, metrics


def superpoint_focal_loss_internal(student_heat, teacher_heat, grid_size=8):
    """Internal focal loss helper."""
    pred = torch.clamp(student_heat, min=1e-6, max=1.0 - 1e-6)

    alpha = 2.0
    beta = 4.0

    pos_inds = teacher_heat.gt(0.01).float()
    neg_inds = 1.0 - pos_inds

    pos_loss = pos_inds * torch.pow(1 - pred, alpha) * torch.log(pred)
    neg_loss = (
        neg_inds
        * torch.pow(1 - teacher_heat, beta)
        * torch.pow(pred, alpha)
        * torch.log(1 - pred)
    )

    loss = -(pos_loss + neg_loss).sum()
    num_pos = pos_inds.sum()

    return loss / (num_pos + 1e-6)


# ============================================================================
# OTHER LOSS FUNCTIONS
# ============================================================================


def keypoint_position_loss(kpts1, kpts2, pts1, pts2, softmax_temp=1.0):
    """
    Computes coordinate classification loss.
    """
    C, H, W = kpts1.shape
    kpts1 = kpts1.permute(1, 2, 0) * softmax_temp
    kpts2 = kpts2.permute(1, 2, 0) * softmax_temp

    with torch.no_grad():
        x, y = torch.meshgrid(
            torch.arange(W, device=kpts1.device),
            torch.arange(H, device=kpts1.device),
            indexing="xy",
        )
        xy = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
        xy *= 8

        hashmap = (
            torch.ones((H * 8, W * 8, 2), dtype=torch.long, device=kpts1.device) * -1
        )

        p1y = torch.clamp(pts1[:, 1].long(), 0, H * 8 - 1)
        p1x = torch.clamp(pts1[:, 0].long(), 0, W * 8 - 1)
        hashmap[p1y, p1x, :] = pts2.long()

        _, kpts1_offsets = kpts1.max(dim=-1)
        kpts1_offsets_x = kpts1_offsets % 8
        kpts1_offsets_y = kpts1_offsets // 8
        kpts1_offsets_xy = torch.cat(
            [kpts1_offsets_x.unsqueeze(-1), kpts1_offsets_y.unsqueeze(-1)], dim=-1
        )

        kpts1_coords = xy + kpts1_offsets_xy
        kpts1_coords = kpts1_coords.view(-1, 2)

        valid_coords = (
            (kpts1_coords[:, 0] >= 0)
            & (kpts1_coords[:, 0] < W * 8)
            & (kpts1_coords[:, 1] >= 0)
            & (kpts1_coords[:, 1] < H * 8)
        )

        gt_12 = torch.ones_like(kpts1_coords) * -1
        gt_12[valid_coords] = hashmap[
            kpts1_coords[valid_coords, 1].long(), kpts1_coords[valid_coords, 0].long()
        ]

        mask_valid = torch.all(gt_12 >= 0, dim=-1)
        gt_12 = gt_12[mask_valid]

        labels2 = (gt_12 / 8) - (gt_12 / 8).long()
        labels2 = (labels2 * 8).long()
        labels2 = labels2[:, 0] + 8 * labels2[:, 1]

    p2y = (gt_12[:, 1] / 8).long().clamp(0, H - 1)
    p2x = (gt_12[:, 0] / 8).long().clamp(0, W - 1)
    kpts2_selected = kpts2[p2y, p2x]

    kpts1_selected = F.log_softmax(kpts1.reshape(-1, C)[mask_valid], dim=-1)
    kpts2_selected = F.log_softmax(kpts2_selected, dim=-1)

    with torch.no_grad():
        _, labels1 = kpts1_selected.max(dim=-1)
        predicted2 = kpts2_selected.max(dim=-1)[1]
        acc = (labels2 == predicted2).float().mean()

    loss = F.nll_loss(kpts1_selected, labels1, reduction="mean") + F.nll_loss(
        kpts2_selected, labels2, reduction="mean"
    )

    return loss, acc


def coordinate_classification_loss(coords1, pts1, pts2, conf):
    """Computes the fine coordinate classification loss."""
    with torch.no_grad():
        coords1_detached = pts1 * 8
        offsets1_detached = (coords1_detached / 8) - (coords1_detached / 8).long()
        offsets1_detached = (offsets1_detached * 8).long()
        labels1 = offsets1_detached[:, 0] + 8 * offsets1_detached[:, 1]

    coords1_log = F.log_softmax(coords1, dim=-1)

    with torch.no_grad():
        predicted = coords1.max(dim=-1)[1]
        acc = labels1 == predicted
        mask = conf > 0.1
        if mask.sum() > 0:
            acc = acc[mask].float().mean()
        else:
            acc = 0.0

    loss = F.nll_loss(coords1_log, labels1, reduction="none")

    if conf.sum() > 0:
        conf = conf / conf.sum()
        loss = (loss * conf).sum()
    else:
        loss = loss.mean()

    return loss * 2.0, acc


def keypoint_loss(reliability_pred, confidence_target):
    """
    Improved reliability loss using Binary Cross Entropy.

    Args:
        reliability_pred (torch.Tensor): [N] Predicted reliability scores (0-1, from sigmoid)
        confidence_target (torch.Tensor): [N] Target confidence scores (0-1, from dual_softmax)

    Returns:
        torch.Tensor: Scalar loss value
    """
    eps = 1e-6
    reliability_pred = torch.clamp(reliability_pred, eps, 1.0 - eps)
    confidence_target = torch.clamp(confidence_target, eps, 1.0 - eps)

    # Binary Cross Entropy
    bce_loss = -(
        confidence_target * torch.log(reliability_pred)
        + (1 - confidence_target) * torch.log(1 - reliability_pred)
    ).mean()

    return bce_loss * 3.0


def hard_triplet_loss(X, Y, margin=0.5):
    """Hard triplet loss for descriptor learning."""
    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError("Error: X and Y shapes must match and be 2D matrices")

    dist_mat = torch.cdist(X, Y, p=2.0)
    dist_pos = torch.diag(dist_mat)

    eye = torch.eye(dist_mat.size(0), device=dist_mat.device)
    dist_neg = dist_mat + eye * 100.0
    dist_neg = dist_neg + dist_neg.le(0.01).float() * 100.0

    hard_neg = torch.min(dist_neg, 1)[0]
    loss = torch.clamp(margin + dist_pos - hard_neg, min=0.0)

    return loss.mean()
