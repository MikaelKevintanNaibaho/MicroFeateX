import argparse
import sys
from pathlib import Path
import torch

# Fix imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from microfeatex.models.student import EfficientFeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser(description="Export MicroFeatEX to ONNX")
    parser.add_argument(
        "--weights", type=str, required=False, help="Path to .pth model weights"
    )
    parser.add_argument(
        "--output", type=str, default="microfeatex.onnx", help="Output .onnx file path"
    )
    parser.add_argument("--opset", type=int, default=16, help="ONNX Opset Version")
    parser.add_argument(
        "--dynamic", action="store_true", help="Enable dynamic batch/height/width"
    )
    parser.add_argument(
        "--descriptor_dim", type=int, default=64, help="Descriptor dimension"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cpu")  # Export is usually safer on CPU

    print(f"Initializing Model (Descriptor Dim: {args.descriptor_dim})...")
    # Initialize model
    model = EfficientFeatureExtractor(
        descriptor_dim=args.descriptor_dim,
        width_mult=1.0,  # Assuming defaults; ideal would be to load from config if weights provided
        use_depthwise=False,  # Assuming standard; user might need to adjust or we load from config
        use_hadamard=False,
    )
    model.to(device)
    model.eval()

    if args.weights:
        print(f"Loading weights from {args.weights}...")
        ckpt = torch.load(args.weights, map_location=device)
        # Handle if it's a full checkpoint or just state_dict
        if "model_state" in ckpt:
            model.load_state_dict(ckpt["model_state"])
        else:
            model.load_state_dict(ckpt)
    else:
        print("Warning: No weights provided, exporting with random initialization.")

    # Dummy Input
    # Standard resolution for tracing
    dummy_input = torch.randn(1, 3, 480, 640).to(device)

    # I/O Names
    input_names = ["input"]
    output_names = ["heatmap", "descriptors", "reliability", "offset"]

    # Dynamic Axes
    dynamic_axes = {}
    if args.dynamic:
        print("Enabling dynamic axes for batch size and resolution...")
        dynamic_axes = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "heatmap": {0: "batch_size", 2: "height", 3: "width"},
            "descriptors": {0: "batch_size", 2: "height_d8", 3: "width_d8"},
            "reliability": {0: "batch_size", 2: "height_d8", 3: "width_d8"},
            "offset": {0: "batch_size", 2: "height_d8", 3: "width_d8"},
        }

    print(f"Exporting to {args.output} (Opset {args.opset})...")
    try:
        # Wrap model to filter output dict to tuple/list for ONNX if needed
        # Or trusted that recent torch.onnx handles dict outputs (it usually does, flattening keys)
        # However, for clearer downstream usage, let's wrap it to return a tuple.

        class OnnxWrapper(torch.nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, x):
                out = self.m(x)
                # Return in specific order
                return (
                    out["heatmap"],
                    out["descriptors"],
                    out["reliability"],
                    out["offset"],
                )

        wrapped_model = OnnxWrapper(model)

        torch.onnx.export(
            wrapped_model,
            dummy_input,
            args.output,
            input_names=input_names,
            output_names=output_names,
            opset_version=args.opset,
            dynamic_axes=dynamic_axes,
            do_constant_folding=True,
        )
        print("Export successful!")
    except Exception as e:
        print(f"Export failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
