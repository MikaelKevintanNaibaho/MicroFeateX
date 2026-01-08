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

# Modern AMP imports
from torch.amp import autocast, GradScaler

import numpy as np

# Project Imports
from microfeatex.models.student import EfficientFeatureExtractor
from microfeatex.models.teacher import SuperPointTeacher
from microfeatex.data.augmentation import AugmentationPipe
from microfeatex.data.dataset import Dataset

# Relative imports
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
    parser.add_argument(
        "--coco_root_path",
        type=str,
        default="dataset/coco_20k",
        help="Path to the COCO dataset root directory.",
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
    Trainer for MicroFeatEX using XFeat-style training loop.
    Focuses on Synthetic COCO Warps and SuperPoint Distillation.
    Includes AMP (Mixed Precision) for memory efficiency.
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
        # This ensures the YAML file takes precedence over argparse defaults
        if "training" in self.config:
            train_conf = self.config["training"]

            # 1. Batch Size
            if "batch_size" in train_conf:
                self.args.batch_size = train_conf["batch_size"]
                print(f"Config Override: batch_size set to {self.args.batch_size}")

            # 2. Learning Rate
            if "lr" in train_conf:
                self.args.lr = train_conf["lr"]
                print(f"Config Override: lr set to {self.args.lr}")

            # 3. Epochs / Steps (Optional but recommended)
            # If you want to control training length via config
            if "n_steps" in train_conf:
                self.args.n_steps = train_conf["n_steps"]
                print(f"Config Override: n_steps set to {self.args.n_steps}")

        # 2. Determine Dataset Path (Config > Args)
        coco_root = args.coco_root_path
        if self.config.get("paths", {}).get("coco_root"):
            coco_root = self.config["paths"]["coco_root"]

        print(f"Dataset Root: {coco_root}")

        # 3. Models
        self.net = EfficientFeatureExtractor(descriptor_dim=64, binary_bits=256).to(
            self.dev
        )

        print(f"Loading SuperPoint Teacher...")
        sp_weights = self.config.get("paths", {}).get(
            "superpoint_weights", "dataset/superpoint_v1.pth"
        )
        self.teacher = SuperPointTeacher(weights_path=sp_weights, device=str(self.dev))
        self.teacher.eval()

        # 4. Optimization
        self.opt = optim.Adam(
            filter(lambda x: x.requires_grad, self.net.parameters()), lr=args.lr
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt, step_size=30000, gamma=0.5
        )

        # 5. Initialize Scaler for AMP
        self.scaler = GradScaler("cuda")

        # 6. Data & Augmentation
        self.augmentor = AugmentationPipe(self.config, self.dev)

        dataset = Dataset(coco_root, self.config)
        self.data_loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        self.data_iter = iter(self.data_loader)

        # 7. Logging & Config
        os.makedirs(args.ckpt_save_path, exist_ok=True)
        log_dir = os.path.join(args.ckpt_save_path, "logdir")
        os.makedirs(log_dir, exist_ok=True)

        self.writer = SummaryWriter(
            os.path.join(
                log_dir, f"{args.model_name}_{time.strftime('%Y_%m_%d-%H_%M_%S')}"
            )
        )
        self.steps = args.n_steps

        loss_cfg = self.config.get("training", {}).get("loss_weights", {})
        self.w_heatmap = loss_cfg.get("heatmap", 10.0)
        self.w_distill = loss_cfg.get("distill", 1.0)
        self.w_reliability = 1.0

    def train(self):
        self.net.train()
        difficulty = 0.3

        print(f"Starting training on {self.dev} for {self.steps} steps...")

        with tqdm.tqdm(total=self.steps) as pbar:
            for i in range(self.steps):
                # 1. Get Batch
                try:
                    batch = next(self.data_iter)
                except StopIteration:
                    self.data_iter = iter(self.data_loader)
                    batch = next(self.data_iter)

                # --- MOVED: Start Autocast BEFORE Augmentation to save memory ---
                with autocast("cuda"):
                    # 2. Augment (Now uses Float16 where possible)
                    p1, p2, H1, H2 = utils.make_batch(self.augmentor, batch, difficulty)

                    # 3. Forward Pass (Student)
                    out1 = self.net(p1)
                    out2 = self.net(p2)

                    hmap1, desc1 = out1["heatmap"], out1["descriptors"]
                    hmap2, desc2 = out2["heatmap"], out2["descriptors"]

                    # 4. Forward Pass (Teacher)
                    with torch.no_grad():
                        t_out1 = self.teacher(p1)
                        t_out2 = self.teacher(p2)

                    # 5. Distillation Loss (Dense)
                    loss_sp1 = losses.superpoint_distill_loss(hmap1, t_out1["scores"])
                    loss_sp2 = losses.superpoint_distill_loss(hmap2, t_out2["scores"])
                    loss_sp = (loss_sp1 + loss_sp2) / 2.0

                    # 6. Compute Correspondences (No grads needed)
                    h_coarse, w_coarse = p1.shape[-2] // 8, p1.shape[-1] // 8
                    negatives, positives = utils.get_corresponding_pts(
                        p1, p2, H1, H2, self.augmentor, h_coarse, w_coarse
                    )

                    # 7. Sparse Loss Calculation
                    batch_loss_ds = 0
                    batch_loss_kp = 0
                    batch_acc = 0
                    valid_batches = 0

                    for b in range(len(positives)):
                        if len(positives[b]) < 10:
                            continue

                        pts1, pts2 = positives[b][:, :2], positives[b][:, 2:]

                        m1 = desc1[b, :, pts1[:, 1].long(), pts1[:, 0].long()].t()
                        m2 = desc2[b, :, pts2[:, 1].long(), pts2[:, 0].long()].t()

                        loss_ds, conf = losses.dual_softmax_loss(m1, m2, temp=0.1)

                        h1_pred = hmap1[b, 0, pts1[:, 1].long(), pts1[:, 0].long()]
                        h2_pred = hmap2[b, 0, pts2[:, 1].long(), pts2[:, 0].long()]

                        loss_kp = losses.keypoint_loss(
                            h1_pred, conf
                        ) + losses.keypoint_loss(h2_pred, conf)

                        batch_loss_ds += loss_ds
                        batch_loss_kp += loss_kp
                        batch_acc += utils.check_accuracy(m1, m2)
                        valid_batches += 1

                    if valid_batches > 0:
                        loss_ds = batch_loss_ds / valid_batches
                        loss_kp = batch_loss_kp / valid_batches
                        acc = batch_acc / valid_batches

                        total_loss = (
                            (self.w_distill * loss_ds)
                            + (self.w_reliability * loss_kp)
                            + (self.w_heatmap * loss_sp)
                        )

                        self.opt.zero_grad()
                        self.scaler.scale(total_loss).backward()
                        self.scaler.unscale_(self.opt)
                        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.0)
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.opt)
                        self.scaler.update()

                        # If scale decreased, it means the step was skipped (NaNs/Infs found)
                        # We must NOT step the scheduler in that case to avoid the warning.
                        scale_after = self.scaler.get_scale()
                        if scale_after >= scale_before:
                            self.scheduler.step()

                        if i % 10 == 0:
                            steps_per_epoch = len(self.data_loader)
                            epoch = i // steps_per_epoch

                            pbar.set_description(
                                f"Ep: {epoch} | "
                                f"Loss: {total_loss.item():.3f} | "
                                f"SP: {loss_sp.item():.3f} | "
                                f"DS: {loss_ds.item():.3f} | "
                                f"Acc: {acc:.3f}"
                            )

                            self.writer.add_scalar("Loss/total", total_loss.item(), i)
                            self.writer.add_scalar(
                                "Loss/superpoint_distill", loss_sp.item(), i
                            )
                            self.writer.add_scalar(
                                "Loss/desc_softmax", loss_ds.item(), i
                            )
                            self.writer.add_scalar("Accuracy/match", acc, i)
                    else:
                        total_loss = self.w_heatmap * loss_sp
                        self.opt.zero_grad()
                        self.scaler.scale(total_loss).backward()
                        self.scaler.step(self.opt)
                        self.scaler.update()

                if (i + 1) % self.args.save_ckpt_every == 0:
                    path = os.path.join(
                        self.args.ckpt_save_path, f"{self.args.model_name}_{i + 1}.pth"
                    )
                    torch.save(self.net.state_dict(), path)
                    torch.save(
                        self.net.state_dict(),
                        os.path.join(self.args.ckpt_save_path, "last.pth"),
                    )
                    print(f"Saved checkpoint to {path}")

                pbar.update(1)


if __name__ == "__main__":
    args = parse_arguments()
    trainer = Trainer(args)
    trainer.train()
