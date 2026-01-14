import torch
import torch.nn as nn

from microfeatex.utils.logger import get_logger

logger = get_logger(__name__)


class SuperPointTeacher(nn.Module):
    """
    Wrapper for the standard SuperPoint model to serve as a frozen teacher.
    """

    def __init__(self, weights_path="data/superpoint_v1.pth", device="cuda"):
        super().__init__()
        self.device = device

        # Define standard SuperPoint architecture
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        c1, c2, c3, c4, c5, d1 = 64, 64, 128, 128, 256, 256
        self.conv1a = nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = nn.Conv2d(c1, c1, kernel_size=3, stride=1, padding=1)

        self.conv2a = nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = nn.Conv2d(c2, c2, kernel_size=3, stride=1, padding=1)

        self.conv3a = nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = nn.Conv2d(c3, c3, kernel_size=3, stride=1, padding=1)

        self.conv4a = nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        # Detector Head
        self.convPa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = nn.Conv2d(c5, 65, kernel_size=1, stride=1, padding=0)

        # Descriptor Head
        self.convDa = nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = nn.Conv2d(c5, d1, kernel_size=1, stride=1, padding=0)

        # Load weights
        try:
            self.load_state_dict(torch.load(weights_path), strict=True)
            logger.info(f"Loaded Teacher weights from {weights_path}")
        except FileNotFoundError:
            logger.warning(
                f"Teacher weights not found at {weights_path}. Please download superpoint_v1.pth"
            )

        self.to(device)
        self.eval()  # Always freeze the teacher!

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Input: Grayscale Image [B, 1, H, W]
        Output:
            - descriptors: [B, 256, H/8, W/8] (Dense Map)
            - semi_dense_scores: [B, 65, H/8, W/8] (Raw logits)
        """
        # Encoder
        x = self.relu(self.conv1a(x))
        x = self.relu(self.conv1b(x))
        x = self.pool(x)
        x = self.relu(self.conv2a(x))
        x = self.relu(self.conv2b(x))
        x = self.pool(x)
        x = self.relu(self.conv3a(x))
        x = self.relu(self.conv3b(x))
        x = self.pool(x)
        x = self.relu(self.conv4a(x))
        x = self.relu(self.conv4b(x))

        # Heads
        # Descriptor (Dense 256-dim map)
        cDa = self.relu(self.convDa(x))
        descriptors = self.convDb(cDa)
        dn = torch.norm(descriptors, p=2, dim=1, keepdim=True)
        descriptors = descriptors / (dn + 1e-6)

        # Detector
        cPa = self.relu(self.convPa(x))
        semi_dense_scores = self.convPb(cPa)

        return {"descriptors": descriptors, "scores": semi_dense_scores}
