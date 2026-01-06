import torch
from torch.utils.data import Dataset
import glob
import cv2
import numpy as np
import os


class SLAMDataset(Dataset):
    def __init__(self, root_dir, height=480, width=640, mode="train"):
        """
        Args:
            root_dir: Path to the dataset (e.g., 'data/tum_rgbd/train')
            mode: 'train' or 'val'
        """
        # Find all rgb images in the TUM folder structure
        self.files = sorted(glob.glob(os.path.join(root_dir, "rgb", "*.png")))

        if len(self.files) == 0:
            raise ValueError(
                f"No images found in {root_dir}/rgb. Did you run the download script?"
            )

        self.height = height
        self.width = width
        self.mode = mode

        # Homography parameters (rho = max movement in pixels)
        self.rho = 32

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        # 1. Load Image
        img_path = self.files[idx]
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # Resize to target resolution (VGA)
        img = cv2.resize(img, (self.width, self.height))

        # Normalize [0, 1]
        img_tensor = torch.from_numpy(img).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0)  # [1, H, W]

        if self.mode == "train":
            # 2. Generate Robust Homography
            # Define original 4 corners
            src_pts = np.float32(
                [[0, 0], [self.width, 0], [self.width, self.height], [0, self.height]]
            )

            # Perturb corners randomly within range [-rho, rho]
            perturbation = np.random.randint(-self.rho, self.rho, size=(4, 2)).astype(
                np.float32
            )
            dst_pts = src_pts + perturbation

            # Compute Homography Matrix H
            H_cv = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # 3. Warp Image
            # We use OpenCV for warping because it's fast on CPU during dataloading
            warped_img_np = cv2.warpPerspective(
                img,
                H_cv,
                (self.width, self.height),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

            warped_tensor = torch.from_numpy(warped_img_np).float() / 255.0
            warped_tensor = warped_tensor.unsqueeze(0)

            # Convert H to torch tensor for the loss function later
            H_tensor = torch.from_numpy(H_cv).float()

            return {
                "image": img_tensor,  # View 1
                "warped_image": warped_tensor,  # View 2
                "homography": H_tensor,  # The ground truth transform
            }

        return {"image": img_tensor}
