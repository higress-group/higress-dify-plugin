"""
Consumer authentication module for Higress AI Gateway.

This module provides a pluggable authentication system that supports
multiple authentication methods (API Key, JWT, HMAC, etc.) and can be
used across different protocols (OpenAI-compatible, etc.).
"""

import base64
import hashlib
import hmac
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urlparse, parse_qs

logger = logging.getLogger(__name__)


@dataclass
class AuthContext:
    """
    Context object containing all information needed for authentication.
    
    This class encapsulates request details that different authentication
    methods may need to compute signatures or generate tokens.
    
    Attributes:
        headers: Current request headers
        credentials: Model credentials containing auth configuration
        method: HTTP method (GET, POST, etc.)
        url: Full request URL
        body: Request body (dict for JSON, str for raw body, or None)
        content_type: Content-Type of the request
    """
    headers: dict
    credentials: dict
    method: str = "POST"
    url: str = ""
    body: Optional[Any] = None
    content_type: str = "application/json"
    
    # Additional fields for extensibility
    extra: dict = field(default_factory=dict)


class BaseAuthenticator(ABC):
    """
    Abstract base class for consumer authenticators.
    
    All authentication methods should inherit from this class and implement
    the apply_auth method to add authentication to request headers.
    """
    
    @abstractmethod
    def get_auth_mode(self) -> str:
        """
        Get the authentication mode identifier.
        
        :return: Authentication mode string (e.g., "api_key", "jwt", "hmac")
        """
        pass
    
    @abstractmethod
    def apply_auth(self, ctx: AuthContext) -> dict:
        """
        Apply authentication to request headers.
        
        :param ctx: AuthContext containing all request information
        :return: Updated headers with authentication applied
        """
        pass
    
    def requires_body(self) -> bool:
        """
        Check if this authenticator requires request body for authentication.
        
        Override this method to return True for authenticators that need
        to sign or process the request body (e.g., HMAC).
        
        :return: True if request body is required for authentication
        """
        return False


class ApiKeyAuthenticator(BaseAuthenticator):
    """
    API Key authenticator.
    
    Adds Bearer token authentication to request headers using the
    api_key credential.
    """
    
    def get_auth_mode(self) -> str:
        return "api_key"
    
    def requires_body(self) -> bool:
        return False
    
    def apply_auth(self, ctx: AuthContext) -> dict:
        """
        Apply API Key authentication as Bearer token.
        
        :param ctx: AuthContext containing credentials
        :return: Updated headers with Authorization: Bearer <token>
        """
        api_key = ctx.credentials.get("api_key", "")
        
        if not api_key:
            logger.warning("API Key authentication enabled but api_key is empty")
            return ctx.headers
        
        # Create a copy of headers to avoid modifying the original
        updated_headers = dict(ctx.headers)
        updated_headers["Authorization"] = f"Bearer {api_key}"
        
        logger.debug("Applied API Key authentication to request headers")
        return updated_headers


class JwtAuthenticator(BaseAuthenticator):
    """
    JWT authenticator (placeholder for future implementation).
    
    Will add JWT token authentication to request headers.
    
    Note: JWT authentication typically requires the token to be generated
    by an authentication server, not by the client. This authenticator
    is reserved for future implementation where tokens may be obtained
    from an external auth service.
    """
    
    def get_auth_mode(self) -> str:
        return "jwt"
    
    def requires_body(self) -> bool:
        return False
    
    def apply_auth(self, ctx: AuthContext) -> dict:
        """
        Apply JWT authentication.
        
        :param ctx: AuthContext containing credentials and request info
        :return: Updated headers with JWT authentication
        """
        # TODO: Implement JWT authentication
        # JWT tokens are typically issued by an authentication server.
        # Possible implementation approaches:
        # 1. Use a pre-generated long-lived token from credentials
        # 2. Implement token refresh logic with an auth server
        # 3. Support OAuth 2.0 / OpenID Connect flows
        
        logger.warning("JWT authentication is not yet implemented")
        return ctx.headers


class HmacAuthenticator(BaseAuthenticator):
    """
    HMAC authenticator for Higress hmac-auth plugin.
    
    Implements HMAC-SHA256 signature authentication compatible with
    Higress API Gateway's hmac-auth plugin.
    
    This authenticator REQUIRES the request body to compute the signature
    (for Content-MD5 calculation on non-form bodies).
    
    The signature is computed based on (in order):
    1. HTTP Method (from :method pseudo-header)
    2. Accept header
    3. Content-MD5 header
    4. Content-Type header  
    5. Date header
    6. Custom headers (specified in x-ca-signature-headers, sorted alphabetically)
    7. Path and query parameters (sorted alphabetically)
    
    Note: For JSON body (application/json), the body content is NOT included
    in the signature string, only Content-MD5 hash is used. Form body parameters
    are only included for application/x-www-form-urlencoded content type.
    
    Required credentials:
    - hmac_key: Access key (used in x-ca-key header)
    - hmac_secret: Secret key (used for HMAC-SHA256 signature)
    """
    
    # Static headers that are always included in signature (in order)
    # These correspond to CHECK_HEADERS in server code
    STATIC_HEADERS = [
        ":method",      # HTTP method pseudo-header
        "accept",
        "content-md5",
        "content-type",
        "date",
    ]
    
    # Headers that should be excluded from dynamic header processing
    EXCLUDED_FROM_DYNAMIC = {
        "x-ca-signature",
        "x-ca-signature-headers",
        # Static headers are also excluded from dynamic processing
        "accept",
        "content-md5", 
        "content-type",
        "date",
    }
    
    # Headers to include in signature (will be listed in x-ca-signature-headers)
    DEFAULT_SIGNATURE_HEADERS = [
        "x-ca-key",
        "x-ca-nonce", 
        "x-ca-signature-method",
        "x-ca-timestamp",
    ]
    
    def get_auth_mode(self) -> str:
        return "hmac"
    
    def requires_body(self) -> bool:
        # HMAC authentication requires request body for signature
        return True
    
    def apply_auth(self, ctx: AuthContext) -> dict:
        """
        Apply HMAC authentication.
        
        :param ctx: AuthContext containing credentials, URL, method, and body
        :return: Updated headers with HMAC signature
        """
        # Extract credentials
        access_key = ctx.credentials.get("hmac_key", "")
        secret_key = ctx.credentials.get("hmac_secret", "")
        
        if not access_key or not secret_key:
            logger.warning("HMAC authentication enabled but hmac_key or hmac_secret is empty")
            return ctx.headers
        
        # Create a copy of headers to avoid modifying the original
        updated_headers = dict(ctx.headers)
        
        # Generate required headers
        timestamp = str(int(time.time() * 1000))  # milliseconds
        nonce = str(uuid.uuid4())
        date_str = datetime.now(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S GMT")
        
        # Set HMAC-specific headers
        updated_headers["x-ca-key"] = access_key
        updated_headers["x-ca-timestamp"] = timestamp
        updated_headers["x-ca-nonce"] = nonce
        updated_headers["x-ca-signature-method"] = "HmacSHA256"
        updated_headers["date"] = date_str
        
        # Note: Content-MD5 is OPTIONAL according to server implementation.
        # If we don't send content-md5 header, server won't verify it.
        # This avoids issues with JSON serialization format differences.
        # We intentionally skip content-md5 to simplify the implementation.
        
        # Build the string to sign
        string_to_sign = self._build_string_to_sign(
            method=ctx.method,
            url=ctx.url,
            headers=updated_headers,
            body=ctx.body,
            content_type=ctx.content_type,
        )
        
        # Calculate HMAC-SHA256 signature
        signature = self._calculate_signature(string_to_sign, secret_key)
        
        # Add signature headers
        updated_headers["x-ca-signature"] = signature
        # Sort the signature headers alphabetically (as server expects)
        sorted_sig_headers = sorted(self.DEFAULT_SIGNATURE_HEADERS)
        updated_headers["x-ca-signature-headers"] = ",".join(sorted_sig_headers)
        
        logger.debug("Applied HMAC authentication to request headers")
        return updated_headers
    
    def _calculate_content_md5(self, body: Any, content_type: str) -> str:
        """
        Calculate Content-MD5 for the request body.
        
        Only calculates MD5 for non-form body content.
        
        IMPORTANT: The MD5 must be calculated on the exact same bytes that will be
        sent in the HTTP request body. requests.post(json=data) uses json.dumps()
        with default separators.
        
        :param body: Request body (dict for JSON, str for raw body)
        :param content_type: Content-Type of the request
        :return: Base64-encoded MD5 hash, or empty string if not applicable
        """
        # Skip MD5 for form content
        if content_type and "application/x-www-form-urlencoded" in content_type:
            return ""
        
        if body is None:
            return ""
        
        # Convert body to string if it's a dict (JSON)
        # Use default json.dumps() format to match requests library behavior
        if isinstance(body, dict):
            body_str = json.dumps(body, ensure_ascii=False)
        elif isinstance(body, str):
            body_str = body
        else:
            body_str = str(body)
        
        # Calculate MD5 and encode in Base64
        md5_hash = hashlib.md5(body_str.encode("utf-8")).digest()
        md5_base64 = base64.b64encode(md5_hash).decode("utf-8")
        
        return md5_base64
    
    def _build_string_to_sign(
        self,
        method: str,
        url: str,
        headers: dict,
        body: Any,
        content_type: str,
    ) -> str:
        """
        Build the string to sign according to Higress hmac-auth specification.
        
        The string format is:
        HTTPMethod\n
        Accept\n
        Content-MD5\n
        Content-Type\n
        Date\n
        DynamicHeader1:Value1\n
        DynamicHeader2:Value2\n
        ...
        PathAndParameters
        
        Note: Each static header value is followed by \n.
        Dynamic headers are in format "key:value\n" (sorted alphabetically).
        PathAndParameters does NOT have trailing \n.
        
        :param method: HTTP method (GET, POST, etc.)
        :param url: Full request URL
        :param headers: Request headers
        :param body: Request body
        :param content_type: Content-Type of the request
        :return: String to sign
        """
        message = ""
        
        # Part 1-5: Static headers (CHECK_HEADERS in server code)
        # Format: "value\n" for each header
        
        # 1. HTTP Method (uppercase)
        http_method = method.upper()
        message += f"{http_method}\n"
        
        # 2. Accept header (can be empty)
        accept = self._get_header_value(headers, "accept")
        message += f"{accept}\n"
        
        # 3. Content-MD5 (can be empty)
        content_md5 = self._get_header_value(headers, "content-md5")
        message += f"{content_md5}\n"
        
        # 4. Content-Type (can be empty)
        content_type_header = self._get_header_value(headers, "content-type")
        message += f"{content_type_header}\n"
        
        # 5. Date header (can be empty if date_offset not configured on server)
        date_header = self._get_header_value(headers, "date")
        message += f"{date_header}\n"
        
        # Part 6: Dynamic headers (from x-ca-signature-headers)
        # Format: "key:value\n" for each header, sorted alphabetically
        headers_str = self._build_headers_string(headers)
        if headers_str:
            message += headers_str
        
        # Part 7: PathAndParameters (no trailing \n)
        path_and_params = self._build_path_and_parameters(url, body, content_type)
        message += path_and_params
        
        return message
    
    def _get_header_value(self, headers: dict, header_name: str) -> str:
        """
        Get header value case-insensitively.
        
        :param headers: Request headers
        :param header_name: Header name to find (lowercase)
        :return: Header value or empty string
        """
        for k, v in headers.items():
            if k.lower() == header_name.lower():
                return v
        return ""
    
    def _build_headers_string(self, headers: dict) -> str:
        """
        Build the dynamic headers string for signature.
        
        According to server implementation (getStringToSign):
        - Headers listed in x-ca-signature-headers are processed
        - Static headers (accept, content-md5, content-type, date) are excluded
        - x-ca-signature and x-ca-signature-headers are excluded
        - Headers are sorted alphabetically by key (lowercase)
        - Format: "key:value\n" for each header
        
        :param headers: Request headers
        :return: Headers string for signature (with trailing \n for each line)
        """
        signature_headers = []
        
        for header_key in self.DEFAULT_SIGNATURE_HEADERS:
            lower_key = header_key.lower()
            
            # Skip headers that are in the excluded set
            if lower_key in self.EXCLUDED_FROM_DYNAMIC:
                continue
            
            # Find header value (case-insensitive)
            header_value = self._get_header_value(headers, header_key)
            
            # Include header even if value is empty (server does this)
            signature_headers.append((lower_key, header_value))
        
        # Sort by key alphabetically
        signature_headers.sort(key=lambda x: x[0])
        
        # Build string with format "key:value\n" for each header
        result = ""
        for k, v in signature_headers:
            result += f"{k}:{v}\n"
        
        return result
    
    def _build_path_and_parameters(self, url: str, body: Any, content_type: str) -> str:
        """
        Build the path and parameters string for signature.
        
        According to server implementation (getStringToSignWithParam):
        - Parse query string from URL path
        - For form body (application/x-www-form-urlencoded), also parse body params
        - Sort all parameters alphabetically by key
        - Format: path?key1=value1&key2=value2...
        - If no params, just return the path
        
        Note: For JSON body (application/json), body parameters are NOT included,
        only query string parameters from URL.
        
        :param url: Full request URL
        :param body: Request body
        :param content_type: Content-Type of the request
        :return: Path and parameters string
        """
        parsed_url = urlparse(url)
        path = parsed_url.path or "/"
        
        # Collect all parameters as a list of (key, value) tuples
        # Using list to preserve order and handle duplicates properly
        params: dict[str, str] = {}
        
        # Parse query string parameters
        if parsed_url.query:
            query_params = parse_qs(parsed_url.query, keep_blank_values=True)
            for key, values in query_params.items():
                # Take the first value for array parameters (server behavior)
                params[key] = values[0] if values else ""
        
        # Parse form body parameters (ONLY for form content type)
        # For JSON body, we do NOT add body params to signature
        if content_type and "application/x-www-form-urlencoded" in content_type.lower():
            if isinstance(body, str):
                form_params = parse_qs(body, keep_blank_values=True)
                for key, values in form_params.items():
                    params[key] = values[0] if values else ""
            elif isinstance(body, dict):
                for key, value in body.items():
                    params[key] = str(value) if value is not None else ""
        
        # If no parameters, return just the path
        if not params:
            return path
        
        # Sort parameters by key alphabetically and build string
        # Format: key1=value1&key2=value2
        sorted_params = sorted(params.items(), key=lambda x: x[0])
        param_parts = []
        for key, value in sorted_params:
            # Always use key=value format (even for empty values based on server code)
            param_parts.append(f"{key}={value}")
        
        return f"{path}?{'&'.join(param_parts)}"
    
    def _calculate_signature(self, string_to_sign: str, secret_key: str) -> str:
        """
        Calculate HMAC-SHA256 signature.
        
        :param string_to_sign: The string to sign
        :param secret_key: Secret key for HMAC
        :return: Base64-encoded signature
        """
        h = hmac.new(
            secret_key.encode("utf-8"),
            string_to_sign.encode("utf-8"),
            hashlib.sha256
        )
        return base64.b64encode(h.digest()).decode("utf-8")


class ConsumerAuthManager:
    """
    Manager class for consumer authentication.
    
    This class provides a unified interface for applying authentication
    based on the configured auth mode. It maintains a registry of
    authenticators and selects the appropriate one based on credentials.
    
    Usage:
        auth_manager = ConsumerAuthManager()
        
        # Simple usage (API Key only)
        headers = auth_manager.apply_auth(headers, credentials)
        
        # Full context usage (supports all auth methods including HMAC)
        ctx = AuthContext(
            headers=headers,
            credentials=credentials,
            method="POST",
            url="https://api.example.com/v1/chat/completions",
            body={"model": "gpt-4", "messages": [...]}
        )
        headers = auth_manager.apply_auth_with_context(ctx)
    """
    
    # Registry of available authenticators
    _authenticators: dict[str, BaseAuthenticator] = {}
    
    def __init__(self):
        """Initialize the auth manager with default authenticators."""
        self._register_default_authenticators()
    
    def _register_default_authenticators(self):
        """Register the default set of authenticators."""
        self.register_authenticator(ApiKeyAuthenticator())
        self.register_authenticator(JwtAuthenticator())
        self.register_authenticator(HmacAuthenticator())
    
    def register_authenticator(self, authenticator: BaseAuthenticator):
        """
        Register a new authenticator.
        
        :param authenticator: Authenticator instance to register
        """
        auth_mode = authenticator.get_auth_mode()
        self._authenticators[auth_mode] = authenticator
        logger.debug(f"Registered authenticator for mode: {auth_mode}")
    
    def get_authenticator(self, auth_mode: str) -> Optional[BaseAuthenticator]:
        """
        Get an authenticator by auth mode.
        
        :param auth_mode: Authentication mode string
        :return: Authenticator instance or None if not found
        """
        return self._authenticators.get(auth_mode)
    
    def is_auth_enabled(self, credentials: dict) -> bool:
        """
        Check if consumer authentication is enabled.
        
        :param credentials: Model credentials
        :return: True if auth is enabled (not disabled)
        """
        auth_mode = credentials.get("consumer_auth_mode", "disabled")
        return auth_mode and auth_mode != "disabled"
    
    def requires_body(self, credentials: dict) -> bool:
        """
        Check if the configured auth method requires request body.
        
        :param credentials: Model credentials
        :return: True if the auth method needs request body
        """
        auth_mode = credentials.get("consumer_auth_mode", "disabled")
        if not auth_mode or auth_mode == "disabled":
            return False
        
        authenticator = self.get_authenticator(auth_mode)
        if not authenticator:
            return False
        
        return authenticator.requires_body()
    
    def apply_auth_with_context(self, ctx: AuthContext) -> dict:
        """
        Apply authentication using full context (supports all auth methods).
        
        :param ctx: AuthContext containing all request information
        :return: Updated headers with authentication applied (if enabled)
        """
        auth_mode = ctx.credentials.get("consumer_auth_mode", "disabled")
        
        # If auth is disabled, return headers unchanged
        if not auth_mode or auth_mode == "disabled":
            logger.debug("Consumer authentication is disabled")
            return ctx.headers
        
        # Get the appropriate authenticator
        authenticator = self.get_authenticator(auth_mode)
        
        if not authenticator:
            logger.warning(f"Unknown authentication mode: {auth_mode}, skipping auth")
            return ctx.headers
        
        # Apply authentication
        logger.debug(f"Applying {auth_mode} authentication")
        return authenticator.apply_auth(ctx)
    
    def apply_auth(self, headers: dict, credentials: dict) -> dict:
        """
        Apply authentication to headers (simplified interface for API Key).
        
        Note: This method creates a minimal AuthContext. For auth methods
        that require request body (like HMAC), use apply_auth_with_context().
        
        :param headers: Existing request headers
        :param credentials: Model credentials containing auth configuration
        :return: Updated headers with authentication applied (if enabled)
        """
        ctx = AuthContext(headers=headers, credentials=credentials)
        return self.apply_auth_with_context(ctx)


# Singleton instance for easy access across the codebase
consumer_auth_manager = ConsumerAuthManager()


def apply_consumer_auth(headers: dict, credentials: dict) -> dict:
    """
    Convenience function to apply consumer authentication (simplified).
    
    This is a shortcut for API Key authentication. For auth methods
    that require request body (like HMAC), use apply_consumer_auth_with_context().
    
    :param headers: Existing request headers
    :param credentials: Model credentials containing auth configuration
    :return: Updated headers with authentication applied (if enabled)
    """
    return consumer_auth_manager.apply_auth(headers, credentials)


def apply_consumer_auth_with_context(ctx: AuthContext) -> dict:
    """
    Convenience function to apply consumer authentication with full context.
    
    Use this function for auth methods that require request body (like HMAC).
    
    :param ctx: AuthContext containing all request information
    :return: Updated headers with authentication applied (if enabled)
    """
    return consumer_auth_manager.apply_auth_with_context(ctx)
