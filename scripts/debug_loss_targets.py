import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from microfeatex.models.student import EfficientFeatureExtractor
from microfeatex.models.alike_teacher import AlikeTeacher


# sys.stdout = open("debug_output.txt", "w")


def debug_heatmap_targets():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")

    # 1. Load Inputs (Blank and Dot)
    _ = torch.zeros(1, 3, 480, 640).to(device)

    img_dot = torch.zeros(1, 3, 480, 640).to(device)
    # create a few bright spots
    img_dot[:, :, 200:210, 200:210] = 1.0
    img_dot[:, :, 300:305, 400:405] = 1.0

    # Select which one to test
    img = img_dot
    print("Testing with DOT Image (should see some keypoints)")

    # 2. Load Models
    student = EfficientFeatureExtractor(
        descriptor_dim=64,
        use_hadamard=True,  # Test with WHT
        use_depthwise=True,
    ).to(device)

    teacher = AlikeTeacher(device=device)

    # 3. Forward Pass
    with torch.no_grad():
        s_out = student(img)
        t_out = teacher(img)

    s_logits = s_out["keypoint_logits"]  # [1, 65, H/8, W/8]
    t_map = t_out["heatmap"]  # [1, 1, H, W]

    print(f"Student Logits: {s_logits.shape}", flush=True)
    print(
        f"Teacher Map: {t_map.shape} (Range: {t_map.min():.3f} - {t_map.max():.3f})",
        flush=True,
    )

    # 4. Re-run Loss Logic manually to get labels
    try:
        grid_size = 8
        C, Hc, Wc = s_logits.shape[1:]

        # Resize Teacher if needed
        target_H, target_W = Hc * grid_size, Wc * grid_size
        if t_map.shape[-2:] != (target_H, target_W):
            print(
                f"Resizing Teacher from {t_map.shape[-2:]} to {(target_H, target_W)}",
                flush=True,
            )
            t_map = F.interpolate(
                t_map, size=(target_H, target_W), mode="bilinear", align_corners=False
            )

        # Max Pool for Coarse Grid
        t_coarse = F.max_pool2d(t_map, kernel_size=grid_size, stride=grid_size)

        print(
            f"Teacher Coarse Stats: min={t_coarse.min().item():.4f}, max={t_coarse.max().item():.4f}, mean={t_coarse.mean().item():.4f}",
            flush=True,
        )

        # Check if 0.1 threshold is appropriate
        keypoint_mask = t_coarse > 0.1

        # FIX: Mask out 1-cell border (8px) to remove edge artifacts (SAME AS LOSSES.PY)
        border_mask = torch.ones_like(keypoint_mask, dtype=torch.bool)
        border_mask[..., 0, :] = False
        border_mask[..., -1, :] = False
        border_mask[..., :, 0] = False
        border_mask[..., :, -1] = False
        keypoint_mask = keypoint_mask & border_mask

        # Get Max Sub-pixel position
        # t_map is [1, 1, H, W]. Squeeze channel dim to get [1, H, W] matching losses.py input
        t_cells = (
            t_map.squeeze(1)
            .view(1, Hc, grid_size, Wc, grid_size)
            .permute(0, 1, 3, 2, 4)
            .reshape(1, Hc, Wc, -1)
        )
        peak_pos = t_cells.argmax(dim=-1)

        labels = torch.where(
            keypoint_mask.squeeze(1),
            peak_pos.squeeze(0),
            torch.full_like(peak_pos.squeeze(0), 64),
        )

        print(
            f"Keypoint Mask Ratio: {keypoint_mask.float().mean().item():.4f}",
            flush=True,
        )
        print(f"Valid Keypoint Cells: {keypoint_mask.sum().item()}", flush=True)

        # 5. Visualize
        # Convert to numpy
        t_map_np = t_map.squeeze().cpu().numpy()
        labels_np = labels.squeeze(0).cpu().numpy()

        # Visualize Labels (Colored by class 0-63, 64 is black/white)
        label_vis = np.zeros((Hc, Wc, 3), dtype=np.uint8)

        # Color Keypoints Red-ish
        # Map 0-63 to color
        for y in range(Hc):
            for x in range(Wc):
                lbl = labels_np[y, x]
                if lbl < 64:
                    # Tint based on sub-pixel pos
                    r = 255
                    g = int(lbl * 4)
                    b = 0
                    label_vis[y, x] = [r, g, b]
                else:
                    # Dustbin = Dark Gray
                    label_vis[y, x] = [20, 20, 20]

        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title("Teacher Heatmap (Input)")
        plt.imshow(t_map_np, cmap="jet")
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.title("Generated Training Labels (Target)")
        plt.imshow(label_vis)

        plt.subplot(1, 3, 3)
        plt.title("Student Init Heatmap (Random)")
        s_heat_np = s_out["heatmap"].detach().cpu().squeeze().numpy()
        plt.imshow(s_heat_np, cmap="jet")
        plt.colorbar()

        plt.savefig("debug_targets.png")
        print("Saved debug_targets.png", flush=True)

    except Exception as e:
        import traceback

        with open("debug_error.log", "w") as f:
            f.write(traceback.format_exc())
            f.write(str(e))
        traceback.print_exc()


if __name__ == "__main__":
    debug_heatmap_targets()
