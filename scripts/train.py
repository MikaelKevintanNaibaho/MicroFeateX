import sys
import os
import yaml
import torch
from torch.utils.data import DataLoader

# Add project root to python path so we can import 'microfeatex'
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from microfeatex.models.student import EfficientFeatureExtractor
from microfeatex.models.teacher import SuperPointTeacher
from microfeatex.data.dataset import Dataset
from microfeatex.data.augmentation import AugmentationPipe
from microfeatex.core.trainer import Trainer


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    # 1. Load Config
    config_path = os.path.join(os.path.dirname(__file__), "../config/coco_train.yaml")
    config = load_config(config_path)

    # Get the project root directory (one level up from scripts/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Join root with the path from config
    weights_path = os.path.join(project_root, config["paths"]["superpoint_weights"])

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"CRITICAL: Teacher weights not found at {weights_path}"
        )

    device = torch.device(config["system"]["device"])

    print(f"Initializing {config['experiment']['name']} on {device}")

    # 2. Dataset & DataLoader
    full_dataset = Dataset(config["paths"]["coco_root"], config)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["system"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],  # Can be larger if VRAM allows
        shuffle=False,  # Don't shuffle validation
        num_workers=config["system"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    # 3. Models
    print("Building models...")
    student = EfficientFeatureExtractor(
        descriptor_dim=config["model"]["descriptor_dim"],
        binary_bits=config["model"]["binary_bits"],
    ).to(device)

    teacher = SuperPointTeacher(device=device)  # Assumes weights exist

    # 4. Augmenter (GPU)
    augmenter = AugmentationPipe(config, device).to(device)

    # 5. Trainer
    trainer = Trainer(student, teacher, train_loader, val_loader, config, augmenter)

    # 6. Resume if needed
    resume_path = config["paths"]["resume_path"]
    if resume_path:
        # Construct full path if it's relative
        if not os.path.isabs(resume_path):
            resume_path = os.path.join(project_root, resume_path)

        if os.path.exists(resume_path):
            trainer.load_checkpoint(resume_path)
        else:
            print(f"WARNING: Checkpoint not found at {resume_path}")
            print("Starting fresh training session...")
    # 7. Start
    trainer.train_loop()


if __name__ == "__main__":
    main()
