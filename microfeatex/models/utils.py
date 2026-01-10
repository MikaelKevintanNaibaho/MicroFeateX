import torch
import torch.nn as nn
import copy


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def estimate_flops(model, input_size=(1, 1, 480, 640)):
    """
    Estimates FLOPs using a deepcopy to avoid polluting the main model state.
    """
    device = next(model.parameters()).device

    # Create a copy so we don't modify the actual model
    # thop adds 'total_ops' and 'total_params' buffers to the model
    model_copy = copy.deepcopy(model)
    dummy_input = torch.randn(*input_size).to(device)

    try:
        from thop import profile

        # Silence thop output
        macs, _ = profile(model_copy, inputs=(dummy_input,), verbose=False)
        return (macs * 2) / 1e9  # Convert MACs to GFLOPs
    except ImportError:
        pass
    except Exception as e:
        print(f"Warning: thop failed ({e}). Falling back to hooks.")

    total_flops = 0

    def conv_flops(module, input, output):
        nonlocal total_flops
        if isinstance(module, nn.Conv2d):
            batch_size = input[0].size(0)
            out_h, out_w = output.size(2), output.size(3)
            kernel_ops = module.kernel_size[0] * module.kernel_size[1]
            in_channels = module.in_channels // module.groups
            out_channels = module.out_channels

            flops = batch_size * out_channels * out_h * out_w * in_channels * kernel_ops
            total_flops += flops

    # Register hooks on the COPY
    hooks = []
    for module in model_copy.modules():
        if isinstance(module, nn.Conv2d):
            hooks.append(module.register_forward_hook(conv_flops))

    # Forward pass on COPY
    model_copy.eval()
    with torch.no_grad():
        model_copy(dummy_input)

    # (No need to remove hooks since we discard model_copy)

    return total_flops / 1e9


def print_model_stats(model, name="Model", input_size=(1, 1, 480, 640)):
    """Pretty prints model statistics."""
    params = count_parameters(model)
    flops = estimate_flops(model, input_size)

    print(f"\n📊 {name} Stats:")
    print(f"   • Resolution: {input_size[2]}x{input_size[3]}")
    print(f"   • Parameters: {params / 1e6:.2f} M")
    print(f"   • FLOPs:      {flops:.3f} G")

    pi4_fps = 15.0 / flops if flops > 0 else 0
    print(f"   • Est. FPS (Pi4): ~{int(pi4_fps)}")
