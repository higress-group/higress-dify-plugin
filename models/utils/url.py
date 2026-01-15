"""
URL utility functions for Higress AI Gateway.
"""

import logging
from urllib.parse import urljoin

logger = logging.getLogger(__name__)


def build_endpoint_url(credentials: dict, api_path: str = "") -> str:
    """
    Build the complete endpoint URL from credentials.
    
    :param credentials: Model credentials containing endpoint_url (gateway address + route prefix)
    :param api_path: Optional API path to append (e.g., "chat/completions", "embeddings")
    :return: Complete endpoint URL
    """
    endpoint_url = credentials.get("endpoint_url", "")
    
    if not endpoint_url:
        raise ValueError("endpoint_url is required in credentials")
    
    # Ensure endpoint_url ends with /
    if not endpoint_url.endswith("/"):
        endpoint_url += "/"
    
    # Append API path if provided
    if api_path:
        endpoint_url = urljoin(endpoint_url, api_path)
    
    logger.debug(f"Built endpoint URL: {endpoint_url}")
    return endpoint_url
