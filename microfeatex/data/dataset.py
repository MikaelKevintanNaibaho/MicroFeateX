import glob
import os

import cv2
import torch
from torch.utils.data import Dataset as TorchDataset

from microfeatex.exceptions import DatasetError
from microfeatex.utils.logger import get_logger

__all__ = ["ImageDataset"]

logger = get_logger(__name__)


class ImageDataset(TorchDataset):
    def __init__(self, root_dir, config):
        """
        Initializes the dataset by recursively searching for images in root_dir.

        Args:
            root_dir (str): Path to the dataset root (from config).
            config (dict): Configuration dictionary.
        """
        # Recursive Search
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
            raise DatasetError(
                f"No images found in {root_dir}. Please check the 'coco_root' path in your config."
            )

        logger.info(f"Dataset: Found {len(self.files)} images in {root_dir}")

        # Get Resize Shape from Config
        self.resize_shape = tuple(config["augmentation"]["warp_resolution"])  # (W, H)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int, _retry_count: int = 0) -> torch.Tensor:
        """Load and preprocess an image.

        Args:
            idx: Index of the image to load.
            _retry_count: Internal counter for retry attempts (do not set manually).

        Returns:
            Preprocessed image tensor of shape [3, H, W].

        Raises:
            RuntimeError: If too many consecutive images fail to load.
        """
        path = self.files[idx]

        # Load as RGB (OpenCV loads as BGR, so we convert)
        img = cv2.imread(path, cv2.IMREAD_COLOR)

        if img is None:
            # Prevent infinite recursion with retry limit
            if _retry_count >= 10:
                raise DatasetError(
                    f"Too many consecutive failed image reads (10+) starting at index {idx}. "
                    f"Last failed path: {path}"
                )
            logger.warning(
                f"Could not read image: {path}. Skipping (retry {_retry_count + 1}/10)."
            )
            return self.__getitem__((idx + 1) % len(self), _retry_count + 1)

        # Convert BGR -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize
        img = cv2.resize(img, self.resize_shape)

        # Normalize & Tensor: [H, W, 3] -> [3, H, W]
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # [3, H, W]

        return img_tensor
