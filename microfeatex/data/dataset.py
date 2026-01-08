import glob
import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class Dataset(Dataset):
    def __init__(self, root_dir, config):
        self.files = sorted(glob.glob(os.path.join(root_dir, "*.jpg"))) + sorted(
            glob.glob(os.path.join(root_dir, "*.png"))
        )

        if not self.files:
            raise ValueError(f"No images found in {root_dir}")

        self.resize_shape = tuple(config["augmentation"]["warp_resolution"])  # (W, H)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            # Handle corrupted images gracefully
            return self.__getitem__((idx + 1) % len(self))

        img = cv2.resize(img, self.resize_shape)
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # [1, H, W]

        return img_tensor
