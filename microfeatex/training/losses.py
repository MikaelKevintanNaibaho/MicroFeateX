"""
training/losses.py
"""

import torch
import torch.nn.functional as F
from . import utils

# Optional imports depending on environment
try:
    from third_party.alike_wrapper import extract_alike_kpts
except ImportError:
    extract_alike_kpts = None


def dual_softmax_loss(X, Y, temp=0.2):
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
    diff = torch.abs(input - target)
    loss = torch.where(diff < beta, 0.5 * diff**2 / beta, diff - 0.5 * beta)
    return loss.mean() if size_average else loss.sum()


def fine_loss(f1, f2, pts1, pts2, fine_module, ws=7):
    """
    Compute Fine features and spatial loss
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
    # Ensure shapes match (Student might be slightly different due to padding, though unlikely with 640x480)
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
    # If teacher says YES: Penalize low prediction
    pos_loss = pos_inds * torch.pow(1 - pred, alpha) * torch.log(pred)

    # If teacher says NO: Penalize high prediction (weighted by distance to actual corner)
    neg_loss = (
        neg_inds
        * torch.pow(1 - t_heat, beta)
        * torch.pow(pred, alpha)
        * torch.log(1 - pred)
    )

    loss = -(pos_loss + neg_loss).sum()

    # Normalize by number of positive keypoints
    num_pos = pos_inds.sum()

    return loss / (num_pos + 1e-6)


def keypoint_position_loss(kpts1, kpts2, pts1, pts2, softmax_temp=1.0):
    """
    Computes coordinate classification loss, by re-interpreting the 64 bins to 8x8 grid and optimizing
    for correct offsets
    """
    C, H, W = kpts1.shape
    kpts1 = kpts1.permute(1, 2, 0) * softmax_temp
    kpts2 = kpts2.permute(1, 2, 0) * softmax_temp

    with torch.no_grad():
        # Generate meshgrid
        x, y = torch.meshgrid(
            torch.arange(W, device=kpts1.device),
            torch.arange(H, device=kpts1.device),
            indexing="xy",
        )
        xy = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
        xy *= 8

        # Generate collision map
        hashmap = (
            torch.ones((H * 8, W * 8, 2), dtype=torch.long, device=kpts1.device) * -1
        )

        # Safe indexing
        p1y = torch.clamp(pts1[:, 1].long(), 0, H * 8 - 1)
        p1x = torch.clamp(pts1[:, 0].long(), 0, W * 8 - 1)
        hashmap[p1y, p1x, :] = pts2.long()

        # Estimate offset of src kpts
        _, kpts1_offsets = kpts1.max(dim=-1)
        kpts1_offsets_x = kpts1_offsets % 8
        kpts1_offsets_y = kpts1_offsets // 8
        kpts1_offsets_xy = torch.cat(
            [kpts1_offsets_x.unsqueeze(-1), kpts1_offsets_y.unsqueeze(-1)], dim=-1
        )

        kpts1_coords = xy + kpts1_offsets_xy

        # find src -> tgt pts
        kpts1_coords = kpts1_coords.view(-1, 2)

        # Check bounds before indexing hashmap
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

        # find offset labels
        labels2 = (gt_12 / 8) - (gt_12 / 8).long()
        labels2 = (labels2 * 8).long()
        labels2 = labels2[:, 0] + 8 * labels2[:, 1]  # linear index

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
    """
    Computes the fine coordinate classification loss.
    """
    # Do not backprop coordinate warps
    with torch.no_grad():
        coords1_detached = pts1 * 8

        # find offset
        offsets1_detached = (coords1_detached / 8) - (coords1_detached / 8).long()
        offsets1_detached = (offsets1_detached * 8).long()
        labels1 = offsets1_detached[:, 0] + 8 * offsets1_detached[:, 1]

    coords1_log = F.log_softmax(coords1, dim=-1)

    with torch.no_grad():
        predicted = coords1.max(dim=-1)[1]
        acc = labels1 == predicted
        # Filter by confidence
        mask = conf > 0.1
        if mask.sum() > 0:
            acc = acc[mask].float().mean()
        else:
            acc = 0.0

    loss = F.nll_loss(coords1_log, labels1, reduction="none")

    # Weight loss by confidence
    if conf.sum() > 0:
        conf = conf / conf.sum()
        loss = (loss * conf).sum()
    else:
        loss = loss.mean()

    return loss * 2.0, acc


def keypoint_loss(heatmap, target):
    # Compute L1 loss
    # heatmap: [N], target: [N] (confidence)
    L1_loss = F.l1_loss(heatmap, target)
    return L1_loss * 3.0


def hard_triplet_loss(X, Y, margin=0.5):
    if X.size() != Y.size() or X.dim() != 2 or Y.dim() != 2:
        raise RuntimeError("Error: X and Y shapes must match and be 2D matrices")

    dist_mat = torch.cdist(X, Y, p=2.0)
    dist_pos = torch.diag(dist_mat)

    eye = torch.eye(dist_mat.size(0), device=dist_mat.device)
    dist_neg = dist_mat + eye * 100.0

    # filter repeated patches on negative distances
    dist_neg = dist_neg + dist_neg.le(0.01).float() * 100.0

    # Margin Ranking Loss
    hard_neg = torch.min(dist_neg, 1)[0]

    loss = torch.clamp(margin + dist_pos - hard_neg, min=0.0)

    return loss.mean()
