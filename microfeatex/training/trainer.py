import argparse
import os
import time
import sys
import glob
import tqdm
import yaml

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np

from microfeatex.models.student import EfficientFeatureExtractor
from microfeatex.models.teacher import SuperPointTeacher
from microfeatex.data.augmentation import AugmentationPipe
from microfeatex.data.dataset import Dataset
from microfeatex.utils.visualization import Visualizer
from . import utils
from . import losses


def parse_arguments():
    parser = argparse.ArgumentParser(description="MicroFeatEX training script.")

    parser.add_argument(
        "--config",
        type=str,
        default="config/coco_train.yaml",
        help="Path to config file",
    )
    # RENAMED: generic 'dataset_root' instead of 'coco_root_path'
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="dataset/coco_20k",
        help="Path to the dataset root directory.",
    )
    parser.add_argument(
        "--ckpt_save_path",
        type=str,
        default="checkpoints/",
        help="Path to save the checkpoints.",
    )
    parser.add_argument(
        "--model_name", type=str, default="microfeatex_v1", help="Model name for logs."
    )
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size.")
    parser.add_argument(
        "--n_steps", type=int, default=160000, help="Number of training steps."
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use.")
    parser.add_argument(
        "--save_ckpt_every",
        type=int,
        default=500,
        help="Save checkpoints every N steps.",
    )

    args = parser.parse_args()
    return args


class Trainer:
    """
    Generic Trainer for MicroFeatEX.
    """

    def __init__(self, args):
        self.args = args
        self.dev = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # 1. Load Config
        self.config = {}
        if os.path.exists(args.config):
            with open(args.config, "r") as f:
                self.config = yaml.safe_load(f)
            print(f"Loaded configuration from {args.config}")

        # Override args with config
        if "training" in self.config:
            train_conf = self.config["training"]
            if "batch_size" in train_conf:
                self.args.batch_size = train_conf["batch_size"]
                print(f"Config Override: batch_size set to {self.args.batch_size}")
            if "lr" in train_conf:
                self.args.lr = train_conf["lr"]
                print(f"Config Override: lr set to {self.args.lr}")
            if "n_steps" in train_conf:
                self.args.n_steps = train_conf["n_steps"]
                print(f"Config Override: n_steps set to {self.args.n_steps}")

        # 2. Determine Dataset Path (Generic Logic)
        # Priority: Config 'dataset_root' > Config 'coco_root' > Args 'dataset_root'
        dataset_root = args.dataset_root
        paths_conf = self.config.get("paths", {})

        if "dataset_root" in paths_conf:
            dataset_root = paths_conf["dataset_root"]
        elif "coco_root" in paths_conf:
            # Fallback for old configs
            dataset_root = paths_conf["coco_root"]

        print(f"Dataset Root: {dataset_root}")

        # 3. Models
        self.net = EfficientFeatureExtractor(descriptor_dim=64, binary_bits=256).to(
            self.dev
        )

        print(f"Loading SuperPoint Teacher...")
        sp_weights = paths_conf.get("superpoint_weights", "dataset/superpoint_v1.pth")
        self.teacher = SuperPointTeacher(weights_path=sp_weights, device=str(self.dev))
        self.teacher.eval()

        # 4. Optimization
        self.opt = optim.Adam(
            filter(lambda x: x.requires_grad, self.net.parameters()), lr=self.args.lr
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt, step_size=30000, gamma=0.5
        )
        self.scaler = GradScaler("cuda")

        # 5. Data & Augmentation
        self.augmentor = AugmentationPipe(self.config, self.dev)

        # Initialize generic dataset
        dataset = Dataset(dataset_root, self.config)

        self.data_loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        self.data_iter = iter(self.data_loader)

        # 6. Logging
        os.makedirs(args.ckpt_save_path, exist_ok=True)
        log_dir = os.path.join(args.ckpt_save_path, "logdir")
        os.makedirs(log_dir, exist_ok=True)

        log_path = os.path.join(
            log_dir, f"{args.model_name}_{time.strftime('%Y_%m_%d-%H_%M_%S')}"
        )
        self.vis = Visualizer(log_dir=log_path)
        self.writer = self.vis.writer

        self.steps = args.n_steps

        loss_cfg = self.config.get("training", {}).get("loss_weights", {})
        self.w_heatmap = loss_cfg.get("heatmap", 10.0)
        self.w_distill = loss_cfg.get("distill", 1.0)
        self.w_reliability = loss_cfg.get("reliability", 0.1)
        # Default fine weight to 1.0 if not specified
        self.w_fine = loss_cfg.get("fine", 1.0)

        self.start_step = 0
        self.current_epoch = 0
        self.load_checkpoint()

    def load_checkpoint(self):
        """Loads the last checkpoint if it exists to resume training."""
        ckpt_path = os.path.join(self.args.ckpt_save_path, "last.pth")
        if os.path.exists(ckpt_path):
            print(f"🔄 Found checkpoint at {ckpt_path}. Resuming...")
            checkpoint = torch.load(ckpt_path, map_location=self.dev)

            # Load Model
            self.net.load_state_dict(checkpoint["model_state"])

            # Load Optimizer & Scaler (Crucial for correct resumption)
            if "optimizer_state" in checkpoint:
                self.opt.load_state_dict(checkpoint["optimizer_state"])
            if "scaler_state" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler_state"])
            if "step" in checkpoint:
                self.start_step = checkpoint["step"] + 1
            if "epoch" in checkpoint:
                self.current_epoch = checkpoint["epoch"]

            print(f"✅ Resumed from step {self.start_step}, epoch {self.current_epoch}")
        else:
            print("🚀 No checkpoint found. Starting from scratch.")

    @staticmethod
    def process_teacher_output(scores, target_size, return_logits=False):
        """
        Robustly processes teacher logits.
        1. Forces resize/transpose to exactly H/8, W/8 (Coarse Grid).
        2. If return_logits=True, returns here (for Loss).
        3. Else, applies Softmax -> Pixel Shuffle (for Vis).
        """
        H, W = target_size
        h_grid, w_grid = H // 8, W // 8

        # 1. Handle Transpose (Swap H/W if needed)
        if scores.shape[-1] == H and scores.shape[-2] == W:
            scores = scores.transpose(-1, -2)

        # 2. Force Resize to Coarse Grid
        if scores.shape[-2] != h_grid or scores.shape[-1] != w_grid:
            scores = F.interpolate(
                scores, size=(h_grid, w_grid), mode="bilinear", align_corners=False
            )

        # --- STOP HERE FOR LOSS ---
        if return_logits:
            return scores

        # 3. Process to Heatmap (For Visualization)
        probs = F.softmax(scores, dim=1)
        corners = probs[:, :-1, :, :]  # Remove dustbin
        heatmap = F.pixel_shuffle(corners, 8)  # Upsample

        return heatmap

    def train(self):
        self.net.train()

        print(f"Starting training on {self.dev} for {self.steps} steps...")

        step_counter = self.start_step

        with tqdm.tqdm(total=self.steps, initial=self.start_step) as pbar:
            while step_counter < self.steps:
                # 1. Get Batch
                try:
                    batch = next(self.data_iter)
                except StopIteration:
                    self.data_iter = iter(self.data_loader)
                    self.current_epoch += 1
                    batch = next(self.data_iter)

                with autocast("cuda"):
                    # 2. Augment
                    p1, p2, H1, H2 = utils.make_batch(
                        self.augmentor, batch, self.augmentor.difficulty
                    )

                    # 3. Forward Pass (Student)
                    out1 = self.net(p1)
                    out2 = self.net(p2)

                    # Extract heads
                    hmap1, desc1, rel1, off1 = (
                        out1["heatmap"],
                        out1["descriptors"],
                        out1["reliability"],
                        out1["offset"],
                    )
                    hmap2, desc2, rel2, off2 = (
                        out2["heatmap"],
                        out2["descriptors"],
                        out2["reliability"],
                        out2["offset"],
                    )
                    # 4. Forward Pass (Teacher)
                    with torch.no_grad():
                        t_out1 = self.teacher(p1)
                        t_out2 = self.teacher(p2)

                    target_shape = p1.shape[2:]

                    # 1. Get Clean Coarse LOGITS for the Loss function
                    # return_logits=True means we get [B, 65, 60, 80]
                    t_logits1 = self.process_teacher_output(
                        t_out1["scores"], target_shape, return_logits=True
                    )
                    t_logits2 = self.process_teacher_output(
                        t_out2["scores"], target_shape, return_logits=True
                    )
                    # 5. Distillation Loss (Dense)
                    loss_sp1 = losses.superpoint_distill_loss(hmap1, t_logits1)
                    loss_sp2 = losses.superpoint_distill_loss(hmap2, t_logits2)
                    loss_sp = (loss_sp1 + loss_sp2) / 2.0

                    # 6. Correspondences
                    h_coarse, w_coarse = p1.shape[-2] // 8, p1.shape[-1] // 8
                    negatives, positives = utils.get_corresponding_pts(
                        p1, p2, H1, H2, self.augmentor, h_coarse, w_coarse
                    )

                    # 7. Sparse Loss Loop
                    batch_loss_ds = 0
                    batch_loss_kp = 0
                    batch_loss_fine = 0
                    batch_acc = 0
                    batch_acc_fine = 0  # Accumulate fine accuracy
                    batch_matches = 0
                    valid_batches = 0

                    for b in range(len(positives)):
                        if len(positives[b]) < 10:
                            continue

                        pts1, pts2 = positives[b][:, :2], positives[b][:, 2:]

                        # --- Descriptor Loss ---

                        # pts1 and 2 is in coarse feature map coordinates (H/8, W/8), but the desc1 is also at H/8 resolution.
                        # the coordinates might still be float from the correspondence generation, so need to be ensured they're properly bounded
                        pts1_y = torch.clamp(pts1[:, 1].long(), 0, desc1.shape[2] - 1)
                        pts1_x = torch.clamp(pts1[:, 0].long(), 0, desc1.shape[3] - 1)
                        pts2_y = torch.clamp(pts2[:, 1].long(), 0, desc2.shape[2] - 1)
                        pts2_x = torch.clamp(pts2[:, 0].long(), 0, desc2.shape[3] - 1)

                        m1 = desc1[b, :, pts1_y, pts1_x].t()
                        m2 = desc2[b, :, pts2_y, pts2_x].t()
                        loss_ds, conf = losses.dual_softmax_loss(m1, m2, temp=0.2)

                        # --- Reliability/Keypoint Loss ---
                        # Sample reliability at coarse feature map resolution
                        # Reliability is [B, 1, H/8, W/8], same as descriptors
                        rel1_pred = rel1[b, 0, pts1_y, pts1_x]
                        rel2_pred = rel2[b, 0, pts2_y, pts2_x]

                        loss_kp = losses.keypoint_loss(
                            rel1_pred, conf
                        ) + losses.keypoint_loss(rel2_pred, conf)

                        # --- Fine / Offset Loss & Accuracy ---
                        # Sample predicted offsets
                        off1_pred = off1[b, :, pts1_y, pts1_x].t()
                        off2_pred = off2[b, :, pts2_y, pts2_x].t()

                        # Targets: Real Float Coord - Cell Center
                        off1_target = pts1 - (pts1.long().float() + 0.5)
                        off2_target = pts2 - (pts2.long().float() + 0.5)

                        loss_fine = F.l1_loss(off1_pred, off1_target) + F.l1_loss(
                            off2_pred, off2_target
                        )

                        # Calculate Fine Accuracy (% with L1 error < 0.5 pixels)
                        fine_diff = torch.norm(off1_pred - off1_target, dim=1)
                        pixel_dist = fine_diff * 8.0
                        acc_fine = (pixel_dist < 3).float().mean()

                        batch_loss_ds += loss_ds
                        batch_loss_kp += loss_kp
                        batch_loss_fine += loss_fine
                        batch_acc += utils.check_accuracy(m1, m2)
                        batch_acc_fine += acc_fine
                        batch_matches += len(positives[b])
                        valid_batches += 1

                    if valid_batches > 0:
                        # Average over batch
                        loss_ds = batch_loss_ds / valid_batches
                        loss_kp = batch_loss_kp / valid_batches
                        loss_fine = batch_loss_fine / valid_batches
                        acc = batch_acc / valid_batches
                        acc_fine = batch_acc_fine / valid_batches
                        avg_matches = batch_matches / valid_batches

                        total_loss = (
                            (self.w_distill * loss_ds)
                            + (self.w_reliability * loss_kp)
                            + (self.w_heatmap * loss_sp)
                            + (self.w_fine * loss_fine)
                        )

                        self.opt.zero_grad()
                        self.scaler.scale(total_loss).backward()
                        self.scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)

                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.opt)
                        self.scaler.update()
                        self.scheduler.step()

                        if step_counter % 10 == 0:
                            pbar.set_description(
                                f"Ep: {self.current_epoch} | "
                                f"Loss: {total_loss.item():.3f} | "
                                f"Acc: {acc:.3f} | "
                                f"Fine: {loss_fine.item():.3f}"
                            )

                            self.writer.add_scalar(
                                "Loss/total", total_loss.item(), step_counter
                            )
                            self.writer.add_scalar(
                                "Accuracy/coarse_synth", acc, step_counter
                            )

                            self.writer.add_scalar(
                                "Accuracy/kp_position", acc_fine, step_counter
                            )

                            self.writer.add_scalar(
                                "Loss/coarse", loss_ds.item(), step_counter
                            )
                            self.writer.add_scalar(
                                "Loss/fine", loss_fine.item(), step_counter
                            )
                            self.writer.add_scalar(
                                "Loss/reliability", loss_kp.item(), step_counter
                            )
                            self.writer.add_scalar(
                                "Loss/keypoint_pos", loss_fine.item(), step_counter
                            )

                            self.writer.add_scalar(
                                "Count/matches_coarse", avg_matches, step_counter
                            )
                            self.writer.add_scalar(
                                "Loss/heatmap_distill", loss_sp.item(), step_counter
                            )

                    else:
                        # Fallback when no valid matches
                        total_loss = self.w_heatmap * loss_sp
                        self.opt.zero_grad()
                        self.scaler.scale(total_loss).backward()

                        self.scaler.step(self.opt)
                        self.scaler.update()
                        self.scheduler.step()

                if step_counter % 500 == 0:
                    vis_t_heat = self.process_teacher_output(
                        t_out1["scores"], target_shape, return_logits=False
                    )
                    self.vis.log_advanced_visuals(
                        step=step_counter,
                        img1=p1,
                        img2=p2,
                        s_heat=hmap1,  # Student Heatmap
                        t_heat=vis_t_heat,  # Teacher Heatmap
                        desc1=desc1,  # Student Descriptors
                        desc2=desc2,
                        s_rel=out1["reliability"],  # Student Reliability
                    )

                if (step_counter + 1) % self.args.save_ckpt_every == 0:
                    save_path = os.path.join(self.args.ckpt_save_path, "last.pth")
                    torch.save(
                        {
                            "model_state": self.net.state_dict(),
                            "optimizer_state": self.opt.state_dict(),
                            "scaler_state": self.scaler.state_dict(),
                            "step": step_counter,
                        },
                        save_path,
                    )

                    # Also save indexed checkpoint for history
                    history_path = os.path.join(
                        self.args.ckpt_save_path,
                        f"{self.args.model_name}_{step_counter + 1}.pth",
                    )
                    torch.save(
                        self.net.state_dict(), history_path
                    )  # Save just weights for inference
                    print(f"Saved checkpoint to {save_path}")

                step_counter += 1
                pbar.update(1)


if __name__ == "__main__":
    args = parse_arguments()
    trainer = Trainer(args)
    trainer.train()
