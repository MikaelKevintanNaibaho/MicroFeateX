import torch
import yaml
from torch.utils.data import DataLoader
import torch.optim as optim

# Import your modules
from src.model import EfficientFeatureExtractor
from src.teacher import SuperPointTeacher
from src.dataset import SLAMDataset
from src.train import train_one_epoch


def main():
    # 1. Configuration
    config = {
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "batch_size": 8,  # Low batch size for RTX 3050 (4GB)
        "epochs": 20,
        "lr": 0.001,
        "w_distill": 1.0,  # Teacher guidance
        "w_siamese": 1.5,  # Stability (Crucial for SLAM)
        "w_quant": 0.1,  # Hashing
        "w_entropy": 0.1,
    }

    print(f"Starting training on {config['device']}...")

    # 2. Data
    # Point this to your downloaded TUM folder
    dataset = SLAMDataset(
        root_dir="data/tum_rgbd/train/rgbd_dataset_freiburg1_desk/", mode="train"
    )
    loader = DataLoader(
        dataset, batch_size=config["batch_size"], shuffle=True, num_workers=4
    )

    # 3. Models
    student = EfficientFeatureExtractor(descriptor_dim=64, binary_bits=256).to(
        config["device"]
    )
    teacher = SuperPointTeacher(device=config["device"])  # Weights loaded automatically

    # 4. Optimizer
    optimizer = optim.Adam(student.parameters(), lr=config["lr"])

    # 5. Loop
    for epoch in range(config["epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{config['epochs']} ---")
        avg_loss = train_one_epoch(student, teacher, loader, optimizer, config)
        print(f"Epoch Summary: Avg Loss = {avg_loss:.4f}")

        # Save Checkpoint
        torch.save(student.state_dict(), f"checkpoints/hybrid_slam_epoch_{epoch}.pth")


if __name__ == "__main__":
    main()
