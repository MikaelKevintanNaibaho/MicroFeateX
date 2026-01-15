"""
Custom exceptions for MicroFeatEX.

Provides specific exception types for better error handling and debugging.
"""


class MicroFeatEXError(Exception):
    """Base exception for all MicroFeatEX errors."""

    pass


class LossValidationError(MicroFeatEXError):
    """Raised when loss function inputs are invalid.

    Examples:
        - Mismatched tensor shapes
        - Invalid tensor dimensions
        - Empty inputs
    """

    pass


class DatasetError(MicroFeatEXError):
    """Raised when dataset operations fail.

    Examples:
        - Too many consecutive failed image reads
        - Empty dataset directory
        - Invalid image format
    """

    pass


class ConfigurationError(MicroFeatEXError):
    """Raised when configuration is invalid.

    Examples:
        - Missing required config keys
        - Invalid parameter values
    """

    pass
