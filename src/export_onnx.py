import torch
from src.model import EfficientFeatureExtractor


def export():
    model = EfficientFeatureExtractor()
    model.eval()

    # Create dummy input (Standard VGA)
    dummy = torch.randn(1, 1, 480, 640)

    # Export
    torch.onnx.export(
        model,
        dummy,
        "slam_features_quantized.onnx",
        input_names=["image_input"],
        output_names=["heatmap", "binary_desc", "unused_teacher_proj"],
        opset_version=11,
    )
    print("Model exported! Load this in TensorRT/OpenCV DNN.")


if __name__ == "__main__":
    export()
