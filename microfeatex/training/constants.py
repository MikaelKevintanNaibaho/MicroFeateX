"""
Centralized constants for MicroFeatEX training.

This module contains hyperparameters and thresholds that were previously
hardcoded throughout the codebase. Centralizing them here makes the code
more maintainable and allows easier tuning.
"""

# =============================================================================
# FOCAL LOSS PARAMETERS
# =============================================================================

# Focal loss exponent for positive samples (higher = more focus on hard positives)
FOCAL_ALPHA: float = 2.0

# Focal loss exponent for negative samples (higher = more suppression of easy negatives)
FOCAL_BETA: float = 4.0


# =============================================================================
# KEYPOINT DETECTION THRESHOLDS
# =============================================================================

# Minimum confidence threshold for considering a cell as containing a keypoint
# Used in alike_distill_loss for creating the keypoint mask
KEYPOINT_CONFIDENCE_THRESHOLD: float = 0.1

# Multiplier applied to keypoint weights vs dustbin weights in distillation loss
# Higher values make keypoint cells dominate the loss more
KEYPOINT_WEIGHT_MULTIPLIER: float = 50.0

# Threshold for positive/negative region definition in focal loss
FOCAL_POSITIVE_THRESHOLD: float = 0.01


# =============================================================================
# TRAINING THRESHOLDS
# =============================================================================

# Minimum number of corresponding points required to compute geometric losses
MIN_CORRESPONDENCES: int = 10

# Minimum number of valid points after confidence filtering to compute descriptor loss
MIN_VALID_POINTS: int = 8
