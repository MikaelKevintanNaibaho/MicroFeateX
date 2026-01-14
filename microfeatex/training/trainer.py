import os
import tqdm
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader

from torch.amp import autocast, GradScaler
import torch.nn.functional as F

from microfeatex.models.student import EfficientFeatureExtractor
from microfeatex.models.alike_teacher import AlikeTeacher
from microfeatex.models.teacher import SuperPointTeacher
from microfeatex.data.augmentation import AugmentationPipe
from microfeatex.data.dataset import ImageDataset
from microfeatex.utils.visualization import Visualizer
from microfeatex.utils.config import load_config, setup_logging_dir, seed_everything
from microfeatex.training.criterion import MicroFeatEXCriterion
from microfeatex.training.scheduler import HyperParamScheduler
from microfeatex.training import utils, losses
from microfeatex.utils.logger import get_logger

logger = get_logger(__name__)

# Scheduler Constants
WARMUP_START_FACTOR: float = 0.1
WARMUP_STEPS: int = 5000
LR_DECAY_STEP: int = 30000
LR_DECAY_GAMMA: float = 0.5


class Trainer:
    def __init__(self, args):
        self.args = args
        self.config = load_config(args.config)
        self.device = torch.device(args.device if torch.cuda.is_available() else "cpu")

        # Setup Reproducibility
        seed_everything(42)

        # Setup Paths & Logging
        self.ckpt_dir = self.args.ckpt_save_path
        os.makedirs(self.ckpt_dir, exist_ok=True)  # Create checkpoint dir if not exists

        # Check if user specified 'log_dir' in the YAML config
        paths_conf = self.config.get("paths", {})
        yaml_log_dir = paths_conf.get("log_dir", None)

        if yaml_log_dir:
            # Case 1: Use the specific log path from config (e.g., "logs/")
            logger.info(f"Using defined log directory: {yaml_log_dir}")
            self.log_dir = setup_logging_dir(
                yaml_log_dir, self.args.model_name, is_explicit_log_path=True
            )
        else:
            # Case 2: Fallback to creating a 'logdir' inside checkpoints folder
            self.log_dir = setup_logging_dir(
                self.ckpt_dir, self.args.model_name, is_explicit_log_path=False
            )
        vis_conf = self.config.get("visualization", {})
        cmap_name = vis_conf.get("colormap", "jet")

        self.vis = Visualizer(log_dir=self.log_dir, colormap_name=cmap_name)
        self.writer = self.vis.writer
        logger.info(f"Artifacts will be saved to: {self.log_dir}")

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

        # Gradient Accumulation
        self.grad_accum_steps = self.config.get("training", {}).get(
            "gradient_accumulation_steps", 1
        )
        if self.grad_accum_steps > 1:
            logger.info(
                f"Gradient Accumulation: {self.grad_accum_steps} steps (effective batch = {self.config.get('training', {}).get('batch_size', 8) * self.grad_accum_steps})"
            )

        self.hp_scheduler = HyperParamScheduler(self.total_steps)

        # Initial Hyperparams (Defaults)
        self.hp = {
            "conf_thresh": 0.0,
            "w_fine": 0.0,
        }

        # Resume
        self._load_checkpoint()

    def _update_hyperparameters(self, step):
        """
        Centralized control for all dynamic training parameters.
        Adjust these ranges based on your preference!
        """
        sched_cfg = self.config.get("training", {}).get("scheduler", {})
        # 1. Confidence Threshold Scheduler
        conf_cfg = sched_cfg.get("conf_thresh", {})
        self.hp["conf_thresh"] = self.hp_scheduler.get_value(
            step,
            start_val=conf_cfg.get("start_val", 0.0),
            end_val=conf_cfg.get("end_val", 0.15),
            start_step_pct=conf_cfg.get("start_pct", 0.0),
            end_step_pct=conf_cfg.get("end_pct", 0.5),
        )

        # 2. Fine Loss Scheduler
        # Target weight comes from the standard loss_weights config
        fine_cfg = sched_cfg.get("fine_loss", {})
        w_fine_target = (
            self.config.get("training", {}).get("loss_weights", {}).get("fine", 1.0)
        )

        current_w_fine = self.hp_scheduler.get_value(
            step,
            start_val=0.0,
            end_val=w_fine_target,
            start_step_pct=fine_cfg.get("start_pct", 0.2),
            end_step_pct=fine_cfg.get("end_pct", 0.4),
        )

        self.criterion.w_fine = current_w_fine
        self.hp["w_fine"] = current_w_fine

    def _build_data(self):
        """Initializes DataLoaders."""
        paths_conf = self.config.get("paths", {})
        root = paths_conf.get(
            "dataset_root", paths_conf.get("coco_root", self.args.dataset_root)
        )

        logger.info(f"Loading Dataset from: {root}")
        self.dataset = ImageDataset(root, self.config)
        self.train_loader = DataLoader(
            self.dataset,
            batch_size=self.config.get("training", {}).get("batch_size", 8),
            shuffle=True,
            num_workers=self.config.get("system", {}).get("num_workers", 4),
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
            use_hadamard=model_conf.get("use_hadamard", False),
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
        # --- 1. Warmup Scheduler ---
        warmup = LinearLR(
            self.optimizer, start_factor=WARMUP_START_FACTOR, total_iters=WARMUP_STEPS
        )

        # --- 2. Main Scheduler (Step Decay) ---
        main_scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=LR_DECAY_STEP, gamma=LR_DECAY_GAMMA
        )

        # --- 3. Sequential Combination ---
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup, main_scheduler],
            milestones=[WARMUP_STEPS],
        )
        self.scaler = GradScaler("cuda")

    def _save_checkpoint(self, step):
        """Saves checkpoint safely."""
        state = {
            "model_state": self.student.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scaler_state": self.scaler.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
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
            logger.info(f"Resuming from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location=self.device)
            self.student.load_state_dict(ckpt["model_state"])
            self.optimizer.load_state_dict(ckpt["optimizer_state"])
            self.scaler.load_state_dict(ckpt["scaler_state"])
            if "scheduler_state" in ckpt:
                self.scheduler.load_state_dict(ckpt["scheduler_state"])
            self.start_step = ckpt["step"] + 1
            self.current_epoch = ckpt.get("epoch", 0)

    # -------------------------------------------------------------------------
    # UPDATED: Added 'step' argument to enable logging inside the loop
    # -------------------------------------------------------------------------
    def _train_step(self, batch, step):
        """
        Executes one training iteration.
        Returns loss dictionary and metrics.
        """

        CONFIDENCE_THRESH = self.hp["conf_thresh"]
        with autocast("cuda"):
            # Augmentation
            p1, p2, H1, H2 = utils.make_batch(
                self.augmentor, batch, self.augmentor.difficulty
            )

            # Student Forward
            out1 = self.student(p1)
            out2 = self.student(p2)

            # Teacher Forward (No Grad) - Let AMP handle precision
            with torch.no_grad():
                t_out1 = self.teacher(p1)
                t_out2 = self.teacher(p2)

            # Criterion - Heatmap Distillation
            # Passing formatted tuple to criterion
            batch_imgs = (p1, p2, H1, H2)
            distill_metrics = self.criterion(
                (out1, out2), (t_out1, t_out2), batch_imgs, step=step
            )
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
            else:
                # SuperPoint Teacher returns 'scores' (65-ch logits).
                # We must convert this to a spatial heatmap [B, 1, H, W]
                with torch.no_grad():
                    # 1. Softmax to get probabilities
                    t_probs1 = F.softmax(t_out1["scores"], dim=1)
                    t_probs2 = F.softmax(t_out2["scores"], dim=1)

                    # 2. Drop the 65th "dustbin" channel (no keypoint)
                    # 3. Pixel Shuffle: [B, 64, H/8, W/8] -> [B, 1, H, W]
                    t_map1_batch = F.pixel_shuffle(t_probs1[:, :-1, :, :], 8)
                    t_map2_batch = F.pixel_shuffle(t_probs2[:, :-1, :, :], 8)

            # Resize to match Student's Reliability Head resolution (e.g. H/8)
            # This is critical because SuperPoint/ALIKE heatmaps are HxW (Full Res),
            # but your Student predicts reliability at H/8xW/8.
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

                # --- Reliability Loss (Keep this dense or semi-dense) ---
                # We use the interpolated teacher map we prepared earlier
                t_map1_b = t_map1_batch[b]
                t_map2_b = t_map2_batch[b]
                l_kp = F.mse_loss(rel1[b], t_map1_b) + F.mse_loss(rel2[b], t_map2_b)

                # --- Descriptor Loss ---
                # Safe indexing
                pts1_y = pts1[:, 1].long().clamp(0, desc1.shape[2] - 1)
                pts1_x = pts1[:, 0].long().clamp(0, desc1.shape[3] - 1)
                pts2_y = pts2[:, 1].long().clamp(0, desc2.shape[2] - 1)
                pts2_x = pts2[:, 0].long().clamp(0, desc2.shape[3] - 1)

                # Get Teacher (or Student) confidence at these specific points
                # We check if the point in View 1 is "interesting" according to the teacher
                score_1 = t_map1_b[0, pts1_y, pts1_x]
                valid_mask = score_1 > CONFIDENCE_THRESH

                # ----------------------------------------------------------------
                # LOGGING: Added Debug Logic Here
                # ----------------------------------------------------------------
                if step % 100 == 0 and b == 0:  # Log first batch every 100 steps
                    self.writer.add_scalar(
                        "debug/num_valid_pts", valid_mask.sum().item(), step
                    )
                    self.writer.add_scalar("debug/total_pts", len(pts1), step)
                    self.writer.add_scalar(
                        "debug/filter_ratio", valid_mask.float().mean().item(), step
                    )
                # ----------------------------------------------------------------

                # If too few points remain, skip descriptor loss for this batch item
                if valid_mask.sum() < 8:  # Minimum points to form a batch
                    # Only accumulate reliability loss
                    batch_stats["loss_kp"].append(l_kp)
                    # Add dummy zeros for others to avoid statistics errors or handle gracefully
                    continue

                # Filter coordinates and descriptors
                pts1_y, pts1_x = pts1_y[valid_mask], pts1_x[valid_mask]
                pts2_y, pts2_x = pts2_y[valid_mask], pts2_x[valid_mask]

                m1 = desc1[b, :, pts1_y, pts1_x].t()
                m2 = desc2[b, :, pts2_y, pts2_x].t()
                l_ds, _ = losses.dual_softmax_loss(m1, m2, temp=0.2)

                # --- Fine / Offset Loss (filtered) ---
                off1_pred = off1[b, :, pts1_y, pts1_x].t()
                off2_pred = off2[b, :, pts2_y, pts2_x].t()

                # Recalculate targets for filtered points
                pts1_filtered = pts1[valid_mask]
                pts2_filtered = pts2[valid_mask]

                off1_tgt = pts1_filtered - (pts1_filtered.long().float() + 0.5)
                off2_tgt = pts2_filtered - (pts2_filtered.long().float() + 0.5)

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

                # Calculate Weighted Components
                w_ds = l_ds * self.criterion.w_distill
                w_kp = l_kp * self.criterion.w_reliability
                w_sp = loss_sp * self.criterion.w_heatmap
                w_fine = l_fine * self.criterion.w_fine

                # Total Loss (Sum of weighted)
                total_loss = w_ds + w_kp + w_sp + w_fine

                metrics = {
                    "loss/total": total_loss.item(),
                    "loss/coarse": l_ds.item(),
                    "loss/fine": l_fine.item(),
                    "loss/heatmap": loss_sp.item(),
                    # --- Weighted Losses (Contribution to Gradient) ---
                    # Use these to check if a loss is disabled (should be 0.0)
                    "loss/weighted/coarse": w_ds.item(),
                    "loss/weighted/fine": w_fine.item(),
                    "loss/weighted/reliability": w_kp.item(),
                    "loss/weighted/heatmap": w_sp.item(),
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
        logger.info(
            f"Starting Training: {self.start_step} -> {self.total_steps} steps."
        )

        iterator = iter(self.train_loader)

        with tqdm.tqdm(total=self.total_steps, initial=self.start_step) as pbar:
            step = self.start_step
            accum_loss = 0.0
            accum_metrics = {}

            while step < self.total_steps:
                self.start_step = step

                # Gradient Accumulation Loop
                self.optimizer.zero_grad()

                for accum_step in range(self.grad_accum_steps):
                    try:
                        batch = next(iterator)
                    except StopIteration:
                        self.current_epoch += 1
                        iterator = iter(self.train_loader)
                        batch = next(iterator)

                    self._update_hyperparameters(step)

                    # Forward + Backward (accumulate gradients)
                    loss, metrics, vis_data = self._train_step(batch, step)

                    # Scale loss by accumulation steps for proper averaging
                    scaled_loss = loss / self.grad_accum_steps
                    self.scaler.scale(scaled_loss).backward()

                    # Accumulate for logging
                    accum_loss += loss.item() / self.grad_accum_steps
                    for k, v in metrics.items():
                        accum_metrics[k] = (
                            accum_metrics.get(k, 0) + v / self.grad_accum_steps
                        )

                # Now update weights (after accumulation)
                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.student.parameters(), 1.0
                )
                if step % 100 == 0:
                    self.writer.add_scalar("debug/grad_norm", grad_norm, step)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()

                # Logging (use accumulated metrics)
                if step % 10 == 0:
                    pbar.set_description(
                        f"Loss: {accum_loss:.4f} | Acc: {accum_metrics.get('acc/coarse', 0):.2f}"
                    )
                    for k, v in accum_metrics.items():
                        self.writer.add_scalar(k, v, step)

                # Reset accumulation
                accum_loss = 0.0
                accum_metrics = {}

                if step % 100 == 0:
                    self.writer.add_scalar(
                        "params/conf_thresh", self.hp["conf_thresh"], step
                    )
                    self.writer.add_scalar("params/w_fine", self.hp["w_fine"], step)
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
