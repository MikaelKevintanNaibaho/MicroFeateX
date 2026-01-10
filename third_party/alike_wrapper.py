import sys
import os
import torch

# 1. Setup Path to the ALIKE folder
# The python files (alike.py, alnet.py) are directly inside third_party/ALIKE/
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ALIKE_REPO_PATH = os.path.join(CURRENT_DIR, "ALIKE")

if ALIKE_REPO_PATH not in sys.path:
    sys.path.append(ALIKE_REPO_PATH)

# 2. Import ALike class
# Now 'import alike' finds 'third_party/ALIKE/alike.py'
try:
    from alike import ALike
except ImportError as e:
    print(f"Wrapper Error: Could not import ALike from {ALIKE_REPO_PATH}")
    print(f"Details: {e}")
    # Common fix: user might need to install dependencies inside ALIKE
    print("Ensure you have 'thop' installed: pip install thop")
    raise


# 3. Helper to initialize model
def get_alike_model(model_name="alike-t", device="cuda", top_k=-1):
    # Config map matching the structure in alike.py
    c_map = {
        "alike-t": {
            "c1": 8,
            "c2": 16,
            "c3": 32,
            "c4": 64,
            "dim": 64,
            "single_head": True,
        },
        "alike-s": {
            "c1": 8,
            "c2": 16,
            "c3": 48,
            "c4": 96,
            "dim": 96,
            "single_head": True,
        },
        "alike-n": {
            "c1": 16,
            "c2": 32,
            "c3": 64,
            "c4": 128,
            "dim": 128,
            "single_head": True,
        },
        "alike-l": {
            "c1": 32,
            "c2": 64,
            "c3": 128,
            "c4": 128,
            "dim": 128,
            "single_head": False,
        },
    }

    if model_name not in c_map:
        raise ValueError(f"Unknown model: {model_name}")

    cfg = c_map[model_name]

    # Weights are located in 'third_party/ALIKE/models/'
    model_path = os.path.join(ALIKE_REPO_PATH, "models", f"{model_name}.pth")

    if not os.path.exists(model_path):
        print(f"WARNING: Model weights not found at {model_path}")
        print("Please verify the 'models' folder exists inside 'third_party/ALIKE/'")

    print(f"Initializing {model_name} on {device}...")

    # Initialize ALike with dense output settings
    model = ALike(
        **cfg, device=device, top_k=top_k, scores_th=0.0, model_path=model_path
    )
    model.eval()
    return model
