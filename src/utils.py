import torch
import numpy as np


def binary_descriptor_postprocess(descriptors):
    """
    Convert the [-1, 1] float descriptors from the network
    into packed binary bytes for storage/matching.
    """
    # 1. Binarize: x > 0 becomes 1, else 0
    binary_mask = (descriptors > 0).byte()

    # 2. Pack bits into bytes (numpy optimization)
    # This is critical for the C++ backend of your SLAM system later
    cpu_bits = binary_mask.detach().cpu().numpy()
    packed = np.packbits(cpu_bits, axis=1)

    return packed


def compute_hamming_dist(desc1, desc2):
    """
    Efficient XOR-based matching for validation.
    """
    # XOR and Popcount
    x = np.bitwise_xor(desc1, desc2)
    # Count set bits
    # (In Python 3.10+ use int.bit_count, for numpy use unpack)
    return np.unpackbits(x).sum()
