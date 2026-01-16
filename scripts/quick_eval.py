import torch
import os
import sys

# Add project root to path
sys.path.insert(0, os.getcwd())

from microfeatex.models.student import EfficientFeatureExtractor
from microfeatex.training.evaluation import run_evaluation
from microfeatex.utils.config import load_config


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path = "checkpoints/lite_hadamard/last.pth"
    config_path = "config/lite_hadamard.yaml"

    print(f"Loading checkpoint: {ckpt_path}")

    if not os.path.exists(ckpt_path):
        print(f"Checkpoint not found at {ckpt_path}")
        return

    # Load config and model
    config = load_config(config_path)
    model_conf = config.get("model", {})

    model = EfficientFeatureExtractor(
        descriptor_dim=model_conf.get("descriptor_dim", 64),
        width_mult=model_conf.get("width_mult", 1.0),
        use_depthwise=model_conf.get("use_depthwise", True),
        use_hadamard=model_conf.get("use_hadamard", True),
    )

    # Load weights
    ckpt = torch.load(ckpt_path, map_location=device)
    if "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"])
    else:
        model.load_state_dict(ckpt)

    model.to(device)
    model.eval()

    print("Running evaluation on 100 pairs...")
    aucs = run_evaluation(
        model=model,
        image_dir="dataset/megadepth_test_1500",
        json_path="assets/megadepth_1500.json",
        device=device,
        num_pairs=100,
        top_k=2000,
    )

    print("\nResults:")
    for k, v in aucs.items():
        print(f"{k}: {v * 100:.2f}")


if __name__ == "__main__":
    main()
