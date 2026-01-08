import glob
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, root_dir, config):
        """
        Initializes the dataset by recursively searching for images in root_dir.

        Args:
            root_dir (str): Path to the dataset root (from config).
            config (dict): Configuration dictionary.
        """
        # 1. Recursive Search
        # Uses 'recursive=True' with '**' to search all subdirectories
        # Matches .jpg, .jpeg, and .png
        patterns = [
            os.path.join(root_dir, "**", "*.jpg"),
            os.path.join(root_dir, "**", "*.jpeg"),
            os.path.join(root_dir, "**", "*.png"),
        ]

        self.files = []
        for p in patterns:
            self.files.extend(glob.glob(p, recursive=True))

        # Sort to ensure deterministic order across runs
        self.files = sorted(list(set(self.files)))

        if not self.files:
            raise ValueError(
                f"No images found in {root_dir}. Please check the 'coco_root' path in your config."
            )

        print(f"Dataset: Found {len(self.files)} images in {root_dir}")

        # 2. Get Resize Shape from Config
        self.resize_shape = tuple(config["augmentation"]["warp_resolution"])  # (W, H)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]

        # Load as Grayscale
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Handle corrupted images or read errors gracefully
            print(f"Warning: Could not read image: {path}. Skipping.")
            # Recursively try the next image (wrapping around)
            return self.__getitem__((idx + 1) % len(self))

        # Resize
        img = cv2.resize(img, self.resize_shape)

        # Normalize & Tensor
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # [1, H, W]

        return img_tensor
