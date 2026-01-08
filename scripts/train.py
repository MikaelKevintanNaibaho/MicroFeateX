import sys
import yaml
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from microfeatex.models.student import EfficientFeatureExtractor
from microfeatex.models.teacher import SuperPointTeacher
from microfeatex.data.dataset import Dataset
from microfeatex.data.augmentation import AugmentationPipe
from microfeatex.core.trainer import Trainer


def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/coco_train.yaml")
    args = parser.parse_args()

    # 1. Config & Paths
    config_path = PROJECT_ROOT / args.config
    config = load_config(config_path)

    # Resolve relative paths in config to absolute
    config["paths"]["coco_root"] = str(PROJECT_ROOT / config["paths"]["coco_root"])
    config["paths"]["log_dir"] = str(PROJECT_ROOT / config["paths"]["log_dir"])
    config["paths"]["checkpoint_dir"] = str(
        PROJECT_ROOT / config["paths"]["checkpoint_dir"]
    )

    device = torch.device(config["system"]["device"])
    print(f"Initializing {config['experiment']['name']} on {device}")

    # 2. Dataset
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
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["system"]["num_workers"],
        pin_memory=True,
    )

    # 3. Models
    student = EfficientFeatureExtractor(
        descriptor_dim=config["model"]["descriptor_dim"],
        binary_bits=config["model"]["binary_bits"],
    ).to(device)

    weights_path = PROJECT_ROOT / config["paths"]["superpoint_weights"]
    teacher = SuperPointTeacher(weights_path=str(weights_path), device=str(device))

    # 4. Trainer
    augmenter = AugmentationPipe(config, device).to(device)
    trainer = Trainer(student, teacher, train_loader, val_loader, config, augmenter)

    # 5. Resume
    if config["paths"].get("resume_path"):
        resume_path = PROJECT_ROOT / config["paths"]["resume_path"]
        if resume_path.exists():
            trainer.load_checkpoint(str(resume_path))

    trainer.train_loop()


if __name__ == "__main__":
    main()
