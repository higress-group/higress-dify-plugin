"""
Utility functions for Higress Rerank.

This module re-exports common utilities from the shared utils package.
"""

# Re-export common utilities for backward compatibility
from models.utils import (
    build_endpoint_url,
    get_rerank_protocol as get_rerank_model_protocol,
)

__all__ = [
    "build_endpoint_url",
    "get_rerank_model_protocol",
]
