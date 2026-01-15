import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import onnxruntime as ort

# Fix imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))

from microfeatex.models.student import EfficientFeatureExtractor


def parse_args():
    parser = argparse.ArgumentParser(description="Verify MicroFeatEX ONNX Export")
    parser.add_argument(
        "--onnx_model", type=str, required=True, help="Path to .onnx model"
    )
    parser.add_argument(
        "--torch_weights", type=str, default=None, help="Path to original .pth weights"
    )
    parser.add_argument(
        "--descriptor_dim", type=int, default=64, help="Descriptor dimension"
    )
    return parser.parse_args()


def to_numpy(tensor):
    return (
        tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    )


def main():
    args = parse_args()

    # 1. Setup PyTorch Model
    print("Setting up PyTorch model...")
    device = torch.device("cpu")
    pytorch_model = EfficientFeatureExtractor(
        descriptor_dim=args.descriptor_dim,
        width_mult=1.0,
        use_depthwise=False,
        use_hadamard=False,
    )
    pytorch_model.to(device)
    pytorch_model.eval()

    if args.torch_weights:
        print(f"Loading PyTorch weights from {args.torch_weights}")
        ckpt = torch.load(args.torch_weights, map_location=device)
        if "model_state" in ckpt:
            pytorch_model.load_state_dict(ckpt["model_state"])
        else:
            pytorch_model.load_state_dict(ckpt)
    else:
        print(
            "Using random weights for verification (ensure ONNX was exported with SAME random weights or this will fail)."
        )
        print(
            "Note: If you didn't provide weights during export, you can't verify correctness against a fresh random model."
        )
        print("Please export WITH weights or just check if ONNX runs without errors.")

    # 2. Setup ONNX Runtime
    print(f"Loading ONNX model: {args.onnx_model}")
    ort_session = ort.InferenceSession(
        args.onnx_model, providers=["CPUExecutionProvider"]
    )

    # 3. Create Dummy Input
    # Testing dynamic shape support
    H, W = 480, 640
    data = torch.randn(1, 3, H, W).to(device)

    # 4. PyTorch Inference
    print("Running PyTorch Inference...")
    t0 = time.time()
    with torch.no_grad():
        pt_out = pytorch_model(data)
    t_pt = time.time() - t0

    # Extract outputs in order
    pt_outputs = [
        to_numpy(pt_out["heatmap"]),
        to_numpy(pt_out["descriptors"]),
        to_numpy(pt_out["reliability"]),
        to_numpy(pt_out["offset"]),
    ]
    output_names = ["heatmap", "descriptors", "reliability", "offset"]

    # 5. ONNX Inference
    print("Running ONNX Inference...")
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(data)}

    t0 = time.time()
    ort_outs = ort_session.run(None, ort_inputs)
    t_onnx = time.time() - t0

    print(
        f"\nInference Time (Single Run): PyTorch={t_pt * 1000:.2f}ms, ONNX={t_onnx * 1000:.2f}ms"
    )

    # 6. Compare
    print("\nVerifying numerical correctness...")
    for name, pt_val, onnx_val in zip(output_names, pt_outputs, ort_outs):
        print(f"Checking {name}...")
        try:
            np.testing.assert_allclose(pt_val, onnx_val, rtol=1e-3, atol=1e-5)
            print("  [PASS] Output matches!")
        except AssertionError as e:
            print(f"  [FAIL] Output mismatch for {name}")
            print(e)

            # Print stats
            diff = np.abs(pt_val - onnx_val)
            print(f"  Max Diff: {diff.max()}")
            print(f"  Mean Diff: {diff.mean()}")


if __name__ == "__main__":
    main()
