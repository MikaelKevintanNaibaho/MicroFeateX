import torch
import numpy as np
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
    # Unpack the new 5-element tuple from augmentor
    p1, p2, H_mat, mask2, coords_map = augmentor(images)

    B = p1.shape[0]
    dev = p1.device

    # H1 is Identity because p1 is the canonical view (just cropped)
    H_identity = torch.eye(3, device=dev).unsqueeze(0).repeat(B, 1, 1)
    mask1 = torch.ones_like(mask2)  # p1 is fully valid
    H1 = (H_identity, mask1)

    # Pack the Coordinate Map into H2
    # This map contains the Dense Flow: img2 pixel -> img1 coordinate
    H2 = (H_mat, mask2, coords_map)

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
    Get dense corresponding points between p1 and p2 using precompute coordinate map.
    """
    global debug_cnt
    negatives, positives = [], []

    with torch.no_grad():
        # real input res of samples
        rh, rw = p1.shape[-2:]

        # Feature map scaling ratio (e.g /8)
        ratio_x = rw / w
        ratio_y = rh / h

        # Unpack
        (H1, mask1) = H1_struct
        (H_mat, mask2, coords_map) = H2_struct  # Here is our dense map [B, 2, H, W]

        # Generate meshgrid for Target Image (View 2) features
        # We want to find matches for every grid cell in the feature map
        x, y = torch.meshgrid(
            torch.arange(w, device=p1.device),
            torch.arange(h, device=p1.device),
            indexing="xy",
        )

        # Target points in Feature Grid coordinates
        tgt_grid_flat = torch.cat([x.unsqueeze(-1), y.unsqueeze(-1)], dim=-1).view(
            -1, 2
        )

        # Target points in Image coordinates (center of the receptive field)
        # Note: (x + 0.5) * ratio puts us in the center of the patch
        tgt_img_pts = (tgt_grid_flat + 0.5) * torch.tensor(
            [ratio_x, ratio_y], device=p1.device
        )

        # We need integer indices to look up the Coordinate Map
        # We simply sample the coordinate map at the center of the patches
        sample_x = tgt_img_pts[:, 0].long().clamp(0, rw - 1)
        sample_y = tgt_img_pts[:, 1].long().clamp(0, rh - 1)

        for batch_idx in range(len(p1)):
            # 1. Look up correspondences
            # coords_map contains [X, Y] coordinates of View 1
            # shape: [B, 2, H, W]

            # Get the map for this batch
            map_b = coords_map[batch_idx]  # [2, H, W]

            # Read the Source Coordinates (View 1) at the Target locations
            src_x = map_b[0, sample_y, sample_x]
            src_y = map_b[1, sample_y, sample_x]

            src_pts = torch.stack(
                [src_x, src_y], dim=1
            )  # [N, 2] in View 1 Image Coords
            tgt_pts = tgt_img_pts.clone()  # [N, 2] in View 2 Image Coords

            # 2. Filter Valid Points
            # Check 1: Must be inside View 1 boundaries
            mask_bound = (
                (src_pts[:, 0] >= 0)
                & (src_pts[:, 0] < rw)
                & (src_pts[:, 1] >= 0)
                & (src_pts[:, 1] < rh)
            )

            # Check 2: Must be valid in masks (not black border)
            # Check Source Mask (at projected location)
            s_ix = src_pts[:, 0].long().clamp(0, rw - 1)
            s_iy = src_pts[:, 1].long().clamp(0, rh - 1)

            # Check Target Mask (at sampling location)
            t_ix = sample_x
            t_iy = sample_y

            mask_valid_pixels = (
                mask1[batch_idx, s_iy, s_ix].bool()
                & mask2[batch_idx, t_iy, t_ix].bool()
            )

            total_mask = mask_bound & mask_valid_pixels

            # Store Negatives (optional, usually not used heavily)
            negatives.append(tgt_pts[~total_mask])

            # Keep Positives
            p_src = src_pts[total_mask]
            p_tgt = tgt_pts[total_mask]

            # 3. Scale back to Feature Grid coordinates
            p_src_grid = p_src / torch.tensor([ratio_x, ratio_y], device=p1.device)
            p_tgt_grid = p_tgt / torch.tensor([ratio_x, ratio_y], device=p1.device)

            # 4. Remove padding artifacts near edges of feature map
            pad = 2
            mask_edge = (
                (p_src_grid[:, 0] >= pad)
                & (p_src_grid[:, 0] < w - pad)
                & (p_src_grid[:, 1] >= pad)
                & (p_src_grid[:, 1] < h - pad)
            )

            p_src_grid = p_src_grid[mask_edge]
            p_tgt_grid = p_tgt_grid[mask_edge]

            # 5. Add to batch
            if len(p_src_grid) > 0:
                # Format: [x1, y1, x2, y2]
                matches = torch.cat([p_src_grid, p_tgt_grid], dim=1)

                # Optional: Limit number of matches to save memory
                if crop is not None and len(matches) > crop:
                    perm = torch.randperm(len(matches), device=matches.device)[:crop]
                    matches = matches[perm]

                positives.append(matches)

                if debug_cnt >= 0 and debug_cnt < 4:
                    plot_corrs(
                        p1[batch_idx],
                        p2[batch_idx],
                        p_src_grid * ratio_x,
                        p_tgt_grid * ratio_x,
                    )
                    debug_cnt += 1
            else:
                positives.append(torch.empty((0, 4), device=p1.device))

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
