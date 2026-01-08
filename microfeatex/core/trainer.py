import torch
import torch.optim as optim
import os
from tqdm import tqdm

# Import new modules
from microfeatex.utils.losses import MicroFeatEXLoss
from microfeatex.utils.visualization import Visualizer


class Trainer:
    def __init__(self, student, teacher, train_loader, val_loader, config, augmenter):
        self.student = student
        self.teacher = teacher
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.augmenter = augmenter
        self.device = config["system"]["device"]

        # Initialize Logic Components
        self.criterion = MicroFeatEXLoss(config).to(self.device)
        self.vis = Visualizer(config["paths"]["log_dir"])

        self.optimizer = optim.Adam(
            self.student.parameters(), lr=config["training"]["lr"]
        )

        self.start_epoch = 0
        self.best_loss = float("inf")
        os.makedirs(config["paths"]["checkpoint_dir"], exist_ok=True)

    def load_checkpoint(self, path):
        print(f"Loading checkpoint from {path}...")
        checkpoint = torch.load(path, map_location=self.device)
        self.student.load_state_dict(checkpoint["model_state"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state"])
        self.start_epoch = checkpoint["epoch"] + 1
        self.best_loss = checkpoint.get("loss", float("inf"))
        print(f"Resumed from epoch {self.start_epoch}")

    def save_checkpoint(self, epoch, loss, is_best=False):
        state = {
            "epoch": epoch,
            "model_state": self.student.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "loss": loss,
        }
        path = self.config["paths"]["checkpoint_dir"]
        torch.save(state, os.path.join(path, "last.pth"))

        if is_best:
            torch.save(state, os.path.join(path, "best_model.pth"))

    def train_epoch(self, epoch):
        self.student.train()
        self.teacher.eval()
        epoch_loss = 0.0

        progress = tqdm(self.train_loader, desc=f"Epoch {epoch}")

        for i, batch_img1 in enumerate(progress):
            step = epoch * len(self.train_loader) + i
            batch_img1 = batch_img1.to(self.device)

            # Augmentation
            img1, img2, H_mat, mask = self.augmenter(batch_img1)

            # Forward Pass
            outs1 = self.student(img1)
            outs2 = self.student(img2)
            with torch.no_grad():
                teacher_out = self.teacher(img1)

            # Loss Calculation
            loss_dict = self.criterion(outs1, outs2, teacher_out, H_mat, mask)

            # Optimization
            self.optimizer.zero_grad()
            loss_dict["total"].backward()
            self.optimizer.step()

            # Logging
            epoch_loss += loss_dict["total"].item()
            progress.set_postfix({"loss": loss_dict["total"].item()})

            if i % self.config["training"]["log_interval"] == 0:
                self.vis.log_scalars(loss_dict, step)

            # Visualization (Every N steps)
            if step % self.config["training"].get("image_log_interval", 100) == 0:
                self.vis.log_training_images(
                    step, img1, img2, mask, outs1[0], teacher_out["scores"]
                )

        return epoch_loss / len(self.train_loader)

    def validate(self):
        self.student.eval()
        self.augmenter.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_img1 in tqdm(self.val_loader, desc="Validating"):
                batch_img1 = batch_img1.to(self.device)
                img1, img2, H_mat, mask = self.augmenter(batch_img1)

                outs1 = self.student(img1)
                outs2 = self.student(img2)
                teacher_out = self.teacher(img1)

                loss_dict = self.criterion(outs1, outs2, teacher_out, H_mat, mask)
                total_loss += loss_dict["total"].item()

        self.augmenter.train()
        return total_loss / len(self.val_loader)

    def train_loop(self):
        print("Starting training loop...")
        try:
            for epoch in range(self.start_epoch, self.config["training"]["epochs"]):
                # ANNEALING SCHEDULE
                # Start at 1.0, end at 5.0 (very steep)
                progress_pct = epoch / self.config["training"]["epochs"]
                new_temp = 1.0 + (4.0 * progress_pct)

                # Access the hashing layer (adjust path based on your model structure)
                self.student.hashing.t = new_temp
                print(f"Epoch {epoch}: Quantization Temperature set to {new_temp:.2f}")

                avg_train_loss = self.train_epoch(epoch)
                avg_val_loss = self.validate()

                print(
                    f"Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}"
                )

                self.vis.log_scalars(
                    {"Train_Epoch": avg_train_loss, "Val_Epoch": avg_val_loss},
                    epoch,
                    prefix="Epoch",
                )

                is_best = avg_val_loss < self.best_loss
                if is_best:
                    self.best_loss = avg_val_loss

                self.save_checkpoint(epoch, avg_val_loss, is_best)

        except KeyboardInterrupt:
            print("Training interrupted. Saving checkpoint...")
            self.save_checkpoint(epoch, 0.0, is_best=False)
        finally:
            self.vis.close()
