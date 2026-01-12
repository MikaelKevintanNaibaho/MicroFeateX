
import torch
import numpy as np
from scipy.linalg import hadamard

def check_hadamard_properties(in_c, out_c):
    print(f"\n--- Checking Dimensions: In={in_c}, Out={out_c} ---")
    
    # Replicate the logic from walsh_hadamard.py
    max_ch = max(in_c, out_c)
    base_dim = 2 ** (max_ch - 1).bit_length()
    print(f"Base Hadamard Dimension: {base_dim}")
    
    h_mat = hadamard(base_dim)
    
    # Slice
    # weights_np = h_mat[:out_channels, :in_channels]
    matrix = h_mat[:out_c, :in_c]
    
    # 1. Check Orthogonality of Rows (Output Channels)
    # A matrix has orthogonal rows if M @ M.T is diagonal
    gram = matrix @ matrix.T
    
    # Normalize for easier reading (should be 1s on diagonal if normalized)
    # But here we just check if off-diagonals are zero
    
    is_orthogonal = True
    off_diag_sum = 0.0
    for i in range(out_c):
        for j in range(out_c):
            if i != j:
                if abs(gram[i, j]) > 1e-9:
                    is_orthogonal = False
                    off_diag_sum += abs(gram[i, j])
    
    print(f"Rows Orthogonal?: {is_orthogonal}")
    if not is_orthogonal:
        print(f"  Sum of off-diagonal absolute values: {off_diag_sum:.2f}")
    
    # 2. Check Rank (Expressivity)
    rank = np.linalg.matrix_rank(matrix)
    print(f"Matrix Rank: {rank}")
    print(f"Full Rank possible: {min(in_c, out_c)}")
    
    if rank < min(in_c, out_c):
        print("  WARNING: Matrix is Rank Deficient! (Collapse of information)")
    else:
        print("  OK: Matrix is Full Rank.")

# Test cases relevant to the model
# Student model uses widths like 32, 64, 128 (powers of 2) usually, 
# but "LightweightBackbone" has channels:
# c(4), c(8), c(24), c(64), c(128)
# If width_mult=1.0:
# 4, 8, 24, 64, 128
# 24 is NOT a power of 2.

# Case 1: Power of 2 (Ideal)
check_hadamard_properties(64, 64)

# Case 2: Non-Power of 2 input (e.g. 24 -> 64)
check_hadamard_properties(24, 64)

# Case 3: Random odd sizes
check_hadamard_properties(32, 32)
