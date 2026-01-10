import argparse
import sys
from pathlib import Path

# Fix imports if running as script
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from microfeatex.training.trainer import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description="MicroFeatEX Training")
    parser.add_argument(
        "--config", type=str, default="config/config.yaml", help="Path to YAML config"
    )
    parser.add_argument(
        "--dataset_root", type=str, default=None, help="Override dataset root"
    )
    parser.add_argument(
        "--ckpt_save_path",
        type=str,
        default="checkpoints/",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--model_name", type=str, default="microfeatex", help="Name for logs/saves"
    )
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu")
    parser.add_argument(
        "--save_ckpt_every", type=int, default=1000, help="Steps between saves"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = Trainer(args)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving final state...")
        trainer._save_checkpoint(
            trainer.start_step
        )  # Accessing internal method for safety
        sys.exit(0)


if __name__ == "__main__":
    main()
