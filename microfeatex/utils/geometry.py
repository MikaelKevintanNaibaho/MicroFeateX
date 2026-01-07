import torch
import kornia.geometry.transform as K
import torch.nn.functional as F


def scale_homography(H, stride=8.0):
    """
    Scales the translation components of a Homography matrix to match
    downsampled feature maps.

    Args:
        H (torch.Tensor): Homography matrix [B, 3, 3] for original image.
        stride (float): Downsampling factor (default 8 for XFeat/SuperPoint).

    Returns:
        torch.Tensor: Scaled homography matrix [B, 3, 3].
    """
    H_scaled = H.clone()
    # Scale translation in x and y
    H_scaled[:, 0, 2] /= stride
    H_scaled[:, 1, 2] /= stride
    return H_scaled


def warp_features(features, H, dsize):
    """
    Warps a feature map using a homography.

    Args:
        features (torch.Tensor): Feature map [B, C, H_feat, W_feat]
        H (torch.Tensor): Homography matrix valid for this feature map scale.
        dsize (tuple): Output size (H_out, W_out).

    Returns:
        torch.Tensor: Warped features.
    """
    return K.warp_perspective(
        features, H, dsize=dsize, align_corners=True, padding_mode="zeros"
    )


def create_valid_mask(height, width, H, device):
    """
    Creates a binary mask indicating which pixels in the warped image
    correspond to valid pixels in the source image (handles borders).

    Args:
        height (int): Height of the feature map.
        width (int): Width of the feature map.
        H (torch.Tensor): Scaled Homography matrix.
        device (torch.device): Device.

    Returns:
        torch.Tensor: Mask [B, 1, H, W] where 1 is valid, 0 is invalid.
    """
    # Create a mask of ones
    mask = torch.ones((H.shape[0], 1, height, width), device=device)

    # Warp it
    mask_warped = warp_features(mask, H, (height, width))

    # Threshold to binary (warping interpolation might create non-binary values at edges)
    return (mask_warped > 0.99).float()
