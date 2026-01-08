import sys
import os
from pathlib import Path

# Setup Project Root for Imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

# Import the Trainer from the microfeatex package
from microfeatex.training.trainer import Trainer, parse_arguments


def main():
    # 1. Parse Args
    args = parse_arguments()

    print(f"Initializing Trainer for {args.model_name}...")
    print(f"Config: {args.config}")

    # 2. Initialize
    trainer = Trainer(args)

    # 3. Train
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
