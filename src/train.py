import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia.geometry.transform as K

# Import your components
from src.model import EfficientFeatureExtractor
from src.teacher import SuperPointTeacher


def train_one_epoch(student, teacher, dataloader, optimizer, config):
    student.train()

    # Metrics tracking
    running_loss = 0.0

    for batch_idx, batch in enumerate(dataloader):
        # 1. Prepare Data
        # Image 1 (Original)
        img1 = batch["image"].to(config["device"])
        # Image 2 (Warped)
        img2 = batch["warped_image"].to(config["device"])
        # Homography (H) that relates img1 -> img2
        H_mat = batch["homography"].to(config["device"])

        # 2. Student Forward Pass (Siamese)
        # We process BOTH images through the student
        # Note: We share weights (same network), hence "Siamese"
        heatmap1, binary1_raw, proj1 = student(img1)
        heatmap2, binary2_raw, proj2 = student(img2)

        # 3. Teacher Forward Pass (Distillation)
        # We only need the teacher for the original image to guide the student
        with torch.no_grad():
            teacher_out = teacher(img1)
            teacher_desc = teacher_out["descriptors"]
            teacher_scores = teacher_out["scores"]

        # --- LOSS CALCULATION ---

        # A. Distillation Loss (Teacher -> Student)
        # Force student's projected float descriptors to match teacher's
        loss_distill_desc = 1 - F.cosine_similarity(proj1, teacher_desc, dim=1).mean()

        # B. DeepBit Hashing Loss (Quantization)
        # Ensure outputs are close to -1 or +1
        # We use binary1_raw (which is Tanh output)
        loss_quant = (binary1_raw.abs() - 1).pow(2).mean()

        # Ensure bits are balanced (Entropy)
        loss_entropy = binary1_raw.mean(dim=0).pow(2).mean()

        # C. Siamese Stability Loss (Student View 1 <-> Student View 2)
        # This is the "SLAM Loss". If the robot moves (homography),
        # the features should move exactly the same way.

        # 1. Scale Homography for Feature Map size (1/8th of input)
        # The network downsamples by 8, so translation in H must be divided by 8
        H_feat = H_mat.clone()
        H_feat[:, 0, 2] /= 8
        H_feat[:, 1, 2] /= 8

        # 2. Warp the descriptor map of View 1 to align with View 2
        # We use 'nearest' padding to avoid border artifacts affecting loss too much
        desc1_warped = K.warp_perspective(
            binary1_raw, H_feat, dsize=(binary1_raw.shape[2], binary1_raw.shape[3])
        )

        # 3. Mask out invalid border regions (pixels that disappeared after warp)
        # We create a mask of ones, warp it, and see what stays
        mask = torch.ones_like(binary1_raw[:, 0:1, :, :])
        mask_warped = K.warp_perspective(
            mask, H_feat, dsize=(binary1_raw.shape[2], binary1_raw.shape[3])
        )

        # 4. Compute Distance between (Warped View 1) and (View 2)
        # We use L2 distance on the Tanh values
        diff = (desc1_warped - binary2_raw) ** 2

        # Apply mask so we don't learn from black borders
        loss_siamese = (diff * mask_warped).sum() / (mask_warped.sum() + 1e-6)

        # --- OPTIMIZATION ---

        # Weighted Sum of losses
        total_loss = (
            (config["w_distill"] * loss_distill_desc)
            + (config["w_quant"] * loss_quant)
            + (config["w_entropy"] * loss_entropy)
            + (config["w_siamese"] * loss_siamese)
        )

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()

        if batch_idx % 10 == 0:
            print(
                f"Batch {batch_idx} | Total: {total_loss.item():.4f} | "
                f"Siam: {loss_siamese.item():.4f} | Distill: {loss_distill_desc.item():.4f}"
            )

    return running_loss / len(dataloader)
