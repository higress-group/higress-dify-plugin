"""
Common utilities for Higress AI Gateway models.
"""

from models.utils.url import build_endpoint_url
from models.utils.auth import (
    AuthContext,
    BaseAuthenticator,
    ApiKeyAuthenticator,
    JwtAuthenticator,
    HmacAuthenticator,
    ConsumerAuthManager,
    consumer_auth_manager,
    apply_consumer_auth,
    apply_consumer_auth_with_context,
)
from models.utils.protocol import (
    get_llm_protocol,
    get_embedding_protocol,
    get_rerank_protocol,
    get_model_mode,
)

__all__ = [
    # URL utilities
    "build_endpoint_url",
    # Auth utilities
    "AuthContext",
    "BaseAuthenticator",
    "ApiKeyAuthenticator",
    "JwtAuthenticator",
    "HmacAuthenticator",
    "ConsumerAuthManager",
    "consumer_auth_manager",
    "apply_consumer_auth",
    "apply_consumer_auth_with_context",
    # Protocol utilities
    "get_llm_protocol",
    "get_embedding_protocol",
    "get_rerank_protocol",
    "get_model_mode",
]
