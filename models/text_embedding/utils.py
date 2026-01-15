"""
Utility functions for Higress Text Embedding.

This module re-exports common utilities from the shared utils package.
"""

# Re-export common utilities for backward compatibility
from models.utils import (
    build_endpoint_url,
    get_embedding_protocol as get_embedding_model_protocol,
)

__all__ = [
    "build_endpoint_url",
    "get_embedding_model_protocol",
]
