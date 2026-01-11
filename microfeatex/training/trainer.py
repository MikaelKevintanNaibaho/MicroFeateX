import os
import time
import tqdm
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from torch.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

from microfeatex.models.student import EfficientFeatureExtractor
from microfeatex.models.alike_teacher import AlikeTeacher
from microfeatex.models.teacher import SuperPointTeacher
from microfeatex.data.augmentation import AugmentationPipe
from microfeatex.data.dataset import Dataset
from microfeatex.utils.visualization import Visualizer
from microfeatex.utils.config import load_config, setup_logging_dir, seed_everything
from microfeatex.training.criterion import MicroFeatEXCriterion
from microfeatex.training import utils, losses


class Trainer:
    def __init__(self, args):
        self.args = args
        self.config = load_config(args.config)
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # Setup Reproducibility
        seed_everything(42)

        # Setup Paths & Logging
        self.ckpt_dir = self.args.ckpt_save_path
        self.log_dir = setup_logging_dir(self.ckpt_dir, self.args.model_name)
        vis_conf = self.config.get("visualization", {})
        cmap_name = vis_conf.get("colormap", "jet")

        self.vis = Visualizer(log_dir=self.log_dir, colormap_name=cmap_name)
        self.writer = self.vis.writer
        print(f"Artifacts will be saved to: {self.log_dir}")

        # Components
        self._build_data()
        self._build_models()
        self._build_optimizer()

        # Loss Module
        self.criterion = MicroFeatEXCriterion(self.config, str(self.device))

        # State
        self.start_step = 0
        self.current_epoch = 0
        self.total_steps = self.config.get("training", {}).get("n_steps", 160000)

        # Resume
        self._load_checkpoint()

    def _build_data(self):
        """Initializes DataLoaders."""
        paths_conf = self.config.get("paths", {})
        root = paths_conf.get(
            "dataset_root", paths_conf.get("coco_root", self.args.dataset_root)
        )

        print(f"Loading Dataset from: {root}")
        self.dataset = Dataset(root, self.config)
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=self.config.get("training", {}).get("batch_size", 8),
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True,
        )
        self.augmentor = AugmentationPipe(self.config, self.device)

    def _build_models(self):
        """Initializes Student and Teacher models."""
        model_conf = self.config.get("model", {})

        # Student
        self.student = EfficientFeatureExtractor(
            descriptor_dim=model_conf.get("descriptor_dim", 64),
            width_mult=model_conf.get("width_mult", 1.0),
            use_depthwise=model_conf.get("use_depthwise", False),
        ).to(self.device)

        self.student.profile()

        # Teacher
        teacher_type = model_conf.get("teacher", "superpoint").lower()
        if teacher_type == "alike":
            self.teacher = AlikeTeacher(
                model_name=model_conf.get("alike_model", "alike-t"),
                device=str(self.device),
            )
        else:
            weights = self.config.get("paths", {}).get(
                "superpoint_weights", "dataset/superpoint_v1.pth"
            )
            self.teacher = SuperPointTeacher(
                weights_path=weights, device=str(self.device)
            )

        self.teacher.eval()
        self.student.train()

    def _build_optimizer(self):
        train_conf = self.config.get("training", {})
        lr = train_conf.get("lr", 3e-4)

        self.optimizer = optim.Adam(self.student.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=30000, gamma=0.5
        )
        self.scaler = GradScaler("cuda")

    def _save_checkpoint(self, step):
        """Saves checkpoint safely."""
        state = {
            "model_state": self.student.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "step": step,
            "epoch": self.current_epoch,
        }
        # Save 'last'
        torch.save(state, os.path.join(self.ckpt_dir, "last.pth"))

        # Save history
        hist_path = os.path.join(self.ckpt_dir, f"{self.args.model_name}_{step}.pth")
        torch.save(self.student.state_dict(), hist_path)

    def _load_checkpoint(self):
        ckpt_path = os.path.join(self.ckpt_dir, "last.pth")
        if os.path.exists(ckpt_path):
            print(f"Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.student.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.scaler.load_state_dict(ckpt["scaler_state"])
            self.start_step = ckpt["step"] + 1
            self.current_epoch = ckpt.get("epoch", 0)

    def _train_step(self, batch):
        """
        Executes one training iteration.
        Returns loss dictionary and metrics.
        """
        with autocast("cuda"):
            # Augmentation
            p1, p2, H1, H2 = utils.make_batch(
                self.augmentor, batch, self.augmentor.difficulty
            )

            # Student Forward
            out1 = self.student(p1)
            out2 = self.student(p2)

            # Teacher Forward (No Grad)
            with torch.no_grad():
                t_out1 = self.teacher(p1.float())
                t_out2 = self.teacher(p2.float())

            # Criterion - Heatmap Distillation
            # Passing formatted tuple to criterion
            batch_imgs = (p1, p2, H1, H2)
            distill_metrics = self.criterion((out1, out2), (t_out1, t_out2), batch_imgs)
            loss_sp = distill_metrics["loss_heatmap"]
            acc_sp = distill_metrics["acc_heatmap"]

            # Geometric Losses (Sparse)
            h_c, w_c = p1.shape[-2] // 8, p1.shape[-1] // 8
            negatives, positives = utils.get_corresponding_pts(
                p1, p2, H1, H2, self.augmentor, h_c, w_c
            )

            # Interpolate ONCE for the whole batch (much faster and avoids dimension errors)
            rel1_shape = out1["reliability"].shape[-2:]  # (H/8, W/8)

            # Helper to get heatmap from teacher output (SuperPoint might not have it)
            if "heatmap" in t_out1:
                t_map1_batch = t_out1["heatmap"]
                t_map2_batch = t_out2["heatmap"]

                # Resize if needed (e.g. ALIKE returns full res, Student uses H/8 for reliability)
                if t_map1_batch.shape[-2:] != rel1_shape:
                    t_map1_batch = F.interpolate(
                        t_map1_batch,
                        size=rel1_shape,
                        mode="bilinear",
                        align_corners=False,
                    )
                    t_map2_batch = F.interpolate(
                        t_map2_batch,
                        size=rel1_shape,
                        mode="bilinear",
                        align_corners=False,
                    )
            else:
                # Fallback for SuperPoint if heatmap isn't pre-generated
                # (Ideally criterion handles this, but reliability loss needs it here)
                # Create dummy or handle SP conversion if needed.
                # For now assuming ALIKE or SP wrapper provides heatmap.
                t_map1_batch = out1["reliability"].detach()  # No-op placeholder
                t_map2_batch = out2["reliability"].detach()

            batch_stats = {
                "loss_ds": [],
                "loss_kp": [],
                "loss_fine": [],
                "acc_coarse": [],
                "acc_fine": [],
            }

            valid_batches = 0

            for b in range(len(positives)):
                if len(positives[b]) < 10:
                    continue

                # Unpack Student Outputs
                desc1, desc2 = out1["descriptors"], out2["descriptors"]
                rel1, rel2 = out1["reliability"], out2["reliability"]
                off1, off2 = out1["offset"], out2["offset"]

                # Extract Coordinates
                pts1, pts2 = positives[b][:, :2], positives[b][:, 2:]

                # --- Descriptor Loss ---
                # Safe indexing
                pts1_y = pts1[:, 1].long().clamp(0, desc1.shape[2] - 1)
                pts1_x = pts1[:, 0].long().clamp(0, desc1.shape[3] - 1)
                pts2_y = pts2[:, 1].long().clamp(0, desc2.shape[2] - 1)
                pts2_x = pts2[:, 0].long().clamp(0, desc2.shape[3] - 1)

                m1 = desc1[b, :, pts1_y, pts1_x].t()
                m2 = desc2[b, :, pts2_y, pts2_x].t()
                m1 = F.normalize(m1, dim=1)
                m2 = F.normalize(m2, dim=1)
                l_ds, _ = losses.dual_softmax_loss(m1, m2, temp=0.2)

                # --- Reliability Loss ---
                # Now using the pre-interpolated batch maps
                t_map1_b = t_map1_batch[b]
                t_map2_b = t_map2_batch[b]

                l_kp = F.mse_loss(rel1[b], t_map1_b) + F.mse_loss(rel2[b], t_map2_b)

                # --- Fine / Offset Loss ---
                off1_pred = off1[b, :, pts1_y, pts1_x].t()
                off2_pred = off2[b, :, pts2_y, pts2_x].t()
                off1_tgt = pts1 - (pts1.long().float() + 0.5)
                off2_tgt = pts2 - (pts2.long().float() + 0.5)

                l_fine = F.l1_loss(off1_pred, off1_tgt) + F.l1_loss(off2_pred, off2_tgt)

                # Fine Accuracy
                fine_diff = torch.norm(off1_pred - off1_tgt, dim=1)
                acc_fine = (fine_diff * 8.0 < 3).float().mean()

                # Accumulate
                batch_stats["loss_ds"].append(l_ds)
                batch_stats["loss_kp"].append(l_kp)
                batch_stats["loss_fine"].append(l_fine)
                batch_stats["acc_coarse"].append(utils.check_accuracy(m1, m2))
                batch_stats["acc_fine"].append(acc_fine)
                valid_batches += 1

            # Aggregate
            if valid_batches > 0:
                l_ds = torch.stack(batch_stats["loss_ds"]).mean()
                l_kp = torch.stack(batch_stats["loss_kp"]).mean()
                l_fine = torch.stack(batch_stats["loss_fine"]).mean()

                total_loss = (
                    (self.criterion.w_distill * l_ds)
                    + (self.criterion.w_reliability * l_kp)
                    + (self.criterion.w_heatmap * loss_sp)
                    + (self.criterion.w_fine * l_fine)
                )
                metrics = {
                    "loss/total": total_loss.item(),
                    "loss/coarse": l_ds.item(),
                    "loss/fine": l_fine.item(),
                    "loss/heatmap": loss_sp.item(),
                    "acc/coarse": sum(batch_stats["acc_coarse"]) / valid_batches,
                    "acc/fine": sum(batch_stats["acc_fine"]) / valid_batches,
                    "acc/heatmap": acc_sp,
                }
            else:
                # Fallback
                total_loss = self.criterion.w_heatmap * loss_sp
                metrics = {
                    "loss/total": total_loss.item(),
                    "loss/heatmap": loss_sp.item(),
                }

        return total_loss, metrics, (p1, p2, out1, t_out1)

    def train(self):
        """Main Training Loop."""
        print(f"Starting Training: {self.start_step} -> {self.total_steps} steps.")

        iterator = iter(self.train_loader)

        with tqdm.tqdm(total=self.total_steps, initial=self.start_step) as pbar:
            step = self.start_step
            while step < self.total_steps:
                try:
                    batch = next(iterator)
                except StopIteration:
                    self.current_epoch += 1
                    iterator = iter(self.train_loader)
                    batch = next(iterator)

                # Optimization Step
                self.optimizer.zero_grad()
                loss, metrics, vis_data = self._train_step(batch)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                # Logging
                if step % 10 == 0:
                    pbar.set_description(
                        f"Loss: {loss.item():.4f} | Acc: {metrics.get('acc/coarse', 0):.2f}"
                    )
                    for k, v in metrics.items():
                        self.writer.add_scalar(k, v, step)

                # Visualization
                if step % 500 == 0:
                    p1, p2, out1, t_out1 = vis_data

                    t_heat = t_out1.get("heatmap", None)
                    # If SuperPoint logic is needed here for vis, can handle it

                    self.vis.log_advanced_visuals(
                        step=step,
                        img1=p1,
                        img2=p2,
                        s_heat=out1["heatmap"],
                        t_heat=t_heat,
                        desc1=out1["descriptors"],
                        desc2=out1["descriptors"],  # Placeholder
                        s_rel=out1["reliability"],
                    )

                # Checkpointing
                if (step + 1) % self.args.save_ckpt_every == 0:
                    self._save_checkpoint(step)

                step += 1
                pbar.update(1)
