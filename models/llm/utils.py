"""
Utility functions for Higress LLM.

This module re-exports common utilities from the shared utils package.
"""

# Re-export common utilities for backward compatibility
from models.utils import (
    build_endpoint_url,
    get_llm_protocol as get_model_protocol,
    get_model_mode,
)

__all__ = [
    "build_endpoint_url",
    "get_model_protocol",
    "get_model_mode",
]
