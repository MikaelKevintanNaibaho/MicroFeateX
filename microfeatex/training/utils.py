import torch
import numpy as np
import pdb
import matplotlib.pyplot as plt

debug_cnt = -1


def make_batch(augmentor, images, difficulty=0.3):
    """
    Adapts the MicroFeatEX AugmentationPipe output to the format expected by the XFeat trainer.

    Args:
        augmentor: The AugmentationPipe instance.
        images: [B, C, H, W] Batch of images from the dataloader.
        difficulty: distortion difficulty.

    Returns:
        p1: Source images (View 1)
        p2: Warped images (View 2)
        H1: (Identity Matrix, Mask1) for View 1
        H2: (Homography, Mask2) for View 2 (or tuple with TPS params if available)
    """
    # AugmentationPipe.forward(x) -> img1, img2, H_mat, mask
    # We assume img1 is the "Identity" view (just cropped/resized) and img2 is the warped view.
    p1, p2, H_mat, mask2 = augmentor(images)

    B = p1.shape[0]
    dev = p1.device

    # H1 is Identity because p1 is the canonical view (just cropped)
    H_identity = torch.eye(3, device=dev).unsqueeze(0).repeat(B, 1, 1)
    mask1 = torch.ones_like(mask2)  # p1 is fully valid

    H1 = (H_identity, mask1)

    # H2 is the Homography/Warp params
    # Note: If AugmentationPipe returns TPS params, we would pack them here.
    # Currently assuming H_mat and mask2.
    H2 = (H_mat, mask2)

    return p1, p2, H1, H2


def plot_corrs(p1, p2, src_pts, tgt_pts):
    p1 = p1.cpu()
    p2 = p2.cpu()
    src_pts = src_pts.cpu()
    tgt_pts = tgt_pts.cpu()

    if len(src_pts) > 200:
        rnd_idx = np.random.randint(len(src_pts), size=200)
        src_pts = src_pts[rnd_idx, ...]
        tgt_pts = tgt_pts[rnd_idx, ...]

    # Plot ground-truth correspondences
    fig, ax = plt.subplots(1, 2, figsize=(18, 12))
    colors = np.random.uniform(size=(len(tgt_pts), 3))

    # Src image
    img = p1.squeeze(0) if p1.dim() == 3 else p1
    if img.shape[0] == 1:
        img = img.repeat(3, 1, 1)
    ax[0].scatter(src_pts[:, 0], src_pts[:, 1], c=colors, s=10)
    ax[0].imshow(img.permute(1, 2, 0).numpy())
    ax[0].set_title("View 1 (Source)")

    # Target img
    img2 = p2.squeeze(0) if p2.dim() == 3 else p2
    if img2.shape[0] == 1:
        img2 = img2.repeat(3, 1, 1)
    ax[1].scatter(tgt_pts[:, 0], tgt_pts[:, 1], c=colors, s=10)
    ax[1].imshow(img2.permute(1, 2, 0).numpy())
    ax[1].set_title("View 2 (Target)")

    plt.tight_layout()
    plt.show()


def get_corresponding_pts(p1, p2, H1_struct, H2_struct, augmentor, h, w, crop=None):
    """
    Get dense corresponding points between p1 and p2.
    """
    global debug_cnt
    negatives, positives = [], []

    with torch.no_grad():
        # real input res of samples
        rh, rw = p1.shape[-2:]
        ratio = torch.tensor([rw / w, rh / h], device=p1.device)

        # Unpack Structures
        (H1, mask1) = H1_struct

        # Handle cases where H2 might have TPS params or just Homography
        if len(H2_struct) == 5:
            (H2, src_tps, W_tps, A_tps, mask2) = H2_struct
            is_tps = True
        else:
            (H2, mask2) = H2_struct
            is_tps = False

        # Generate meshgrid of target pts
        x, y = torch.meshgrid(
            torch.arange(w, device=p1.device),
            torch.arange(h, device=p1.device),
            indexing="xy",
        )
        mesh = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1)
        target_pts = mesh.view(-1, 2) * ratio  # Scale to image resolution

        # Pack all transformations into T
        for batch_idx in range(len(p1)):
            with torch.no_grad():
                # We need to map FROM target (p2) TO source (p1)
                # p2 = H * p1  =>  p1 = H_inv * p2

                # If using simple Homography:
                if not is_tps:
                    # FIX: Use augmentor.warp_points which handles the transformation correctly
                    # We need inverse mapping: target -> source
                    # So we invert H2 and pass it to warp_points
                    H_inv = torch.inverse(H2[batch_idx])

                    # warp_points expects pts in output coordinates and H on same device
                    # target_pts are already in the right coordinate system
                    src_pts = augmentor.warp_points(H_inv, target_pts)

                else:
                    # Use TPS unwarping if parameters exist
                    T = (
                        H1[batch_idx],
                        H2[batch_idx],
                        src_tps[batch_idx].unsqueeze(0),
                        W_tps[batch_idx].unsqueeze(0),
                        A_tps[batch_idx].unsqueeze(0),
                    )
                    # Note: This path needs proper implementation if TPS is used
                    # For now, fallback to simple inverse homography
                    H_inv = torch.inverse(H2[batch_idx])
                    src_pts = augmentor.warp_points(H_inv, target_pts)

                tgt_pts = target_pts.clone()

                # Check out of bounds points (Source Image Boundaries)
                mask_valid = (
                    (src_pts[:, 0] >= 0)
                    & (src_pts[:, 1] >= 0)
                    & (src_pts[:, 0] < rw)
                    & (src_pts[:, 1] < rh)
                )

                negatives.append(tgt_pts[~mask_valid])
                tgt_pts = tgt_pts[mask_valid]
                src_pts = src_pts[mask_valid]

                # Remove invalid pixels (Black Borders)
                # Ensure indices are within bounds for mask lookup
                src_x = torch.clamp(src_pts[:, 0].long(), 0, rw - 1)
                src_y = torch.clamp(src_pts[:, 1].long(), 0, rh - 1)
                tgt_x = torch.clamp(tgt_pts[:, 0].long(), 0, rw - 1)
                tgt_y = torch.clamp(tgt_pts[:, 1].long(), 0, rh - 1)

                mask_valid = (
                    mask1[batch_idx, src_y, src_x].bool()
                    & mask2[batch_idx, tgt_y, tgt_x].bool()
                )

                tgt_pts = tgt_pts[mask_valid]
                src_pts = src_pts[mask_valid]

                # Limit nb of matches if desired
                if crop is not None and len(src_pts) > crop:
                    rnd_idx = torch.randperm(len(src_pts), device=src_pts.device)[:crop]
                    src_pts = src_pts[rnd_idx]
                    tgt_pts = tgt_pts[rnd_idx]

                if debug_cnt >= 0 and debug_cnt < 4:
                    plot_corrs(p1[batch_idx], p2[batch_idx], src_pts, tgt_pts)
                    debug_cnt += 1

                # Normalize back to Feature Map Grid Coordinates (w, h)
                src_pts = src_pts / ratio
                tgt_pts = tgt_pts / ratio

                # Boundary Check on Feature Map
                padto = 10 if crop is not None else 2
                mask_valid1 = (
                    (src_pts[:, 0] >= padto)
                    & (src_pts[:, 1] >= padto)
                    & (src_pts[:, 0] < (w - padto))
                    & (src_pts[:, 1] < (h - padto))
                )
                mask_valid2 = (
                    (tgt_pts[:, 0] >= padto)
                    & (tgt_pts[:, 1] >= padto)
                    & (tgt_pts[:, 0] < (w - padto))
                    & (tgt_pts[:, 1] < (h - padto))
                )

                mask_valid = mask_valid1 & mask_valid2
                tgt_pts = tgt_pts[mask_valid]
                src_pts = src_pts[mask_valid]

                # Remove repeated correspondences using a LUT
                lut_mat = (
                    torch.ones((h, w, 4), device=src_pts.device, dtype=src_pts.dtype)
                    * -1
                )
                try:
                    # Store (x1, y1, x2, y2) at location (y1, x1)
                    lut_mat[src_pts[:, 1].long(), src_pts[:, 0].long()] = torch.cat(
                        [src_pts, tgt_pts], dim=1
                    )
                    # Filter: Keep only the last written value (or valid ones)
                    # This implicitly handles collisions by keeping one
                    mask_valid = torch.all(lut_mat >= 0, dim=-1)
                    points = lut_mat[mask_valid]
                    positives.append(points)
                except Exception as e:
                    print(f"Error in LUT creation: {e}")
                    pdb.set_trace()

    return negatives, positives


def crop_patches(tensor, coords, size=7):
    """
    Crop [size x size] patches around 2D coordinates from a tensor.
    """
    B, C, H, W = tensor.shape

    x, y = coords[:, 0], coords[:, 1]
    y = y.view(-1, 1, 1)
    x = x.view(-1, 1, 1)
    halfsize = size // 2

    # Create meshgrid for indexing
    x_offset, y_offset = torch.meshgrid(
        torch.arange(-halfsize, halfsize + 1),
        torch.arange(-halfsize, halfsize + 1),
        indexing="xy",
    )
    y_offset = y_offset.to(tensor.device)
    x_offset = x_offset.to(tensor.device)

    # Compute indices around each coordinate
    y_indices = (y + y_offset.view(1, size, size)).squeeze(0) + halfsize
    x_indices = (x + x_offset.view(1, size, size)).squeeze(0) + halfsize

    # Handle out-of-boundary indices with padding
    tensor_padded = torch.nn.functional.pad(
        tensor, (halfsize, halfsize, halfsize, halfsize), mode="constant"
    )

    # Index tensor to get patches
    # Note: Pad adds to left/top so indices shift by halfsize
    patches = tensor_padded[:, :, y_indices, x_indices]  # [B, C, N, H, W]
    return patches


def subpix_softmax2d(heatmaps, temp=0.25):
    N, H, W = heatmaps.shape
    heatmaps = torch.softmax(temp * heatmaps.view(-1, H * W), -1).view(-1, H, W)
    x, y = torch.meshgrid(
        torch.arange(W, device=heatmaps.device),
        torch.arange(H, device=heatmaps.device),
        indexing="xy",
    )
    x = x - (W // 2)
    y = y - (H // 2)

    coords_x = x[None, ...] * heatmaps
    coords_y = y[None, ...] * heatmaps
    coords = torch.cat([coords_x[..., None], coords_y[..., None]], -1).view(N, H * W, 2)
    coords = coords.sum(1)

    return coords


def check_accuracy(X, Y, pts1=None, pts2=None, plot=False):
    with torch.no_grad():
        # X: [N, D], Y: [N, D]
        # Similarity matrix
        dist_mat = X @ Y.t()
        nn = torch.argmax(dist_mat, dim=1)
        correct = nn == torch.arange(len(X), device=X.device)

        if pts1 is not None and plot:
            canvas = torch.zeros((60, 80), device=X.device)
            pts1_wrong = pts1[~correct]
            if len(pts1_wrong) > 0:
                canvas[pts1_wrong[:, 1].long(), pts1_wrong[:, 0].long()] = 1
                canvas = canvas.cpu().numpy()
                plt.imshow(canvas)
                plt.show()

        acc = correct.sum().item() / (len(X) + 1e-8)
        return acc


def get_nb_trainable_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    nb_params = sum([np.prod(p.size()) for p in model_parameters])
    print("Number of trainable parameters: {:d}".format(nb_params))
