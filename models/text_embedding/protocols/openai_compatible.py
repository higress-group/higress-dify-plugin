"""
OpenAI Compatible protocol implementation for Text Embedding.
"""

import json
import logging
from typing import Optional

import requests

from dify_plugin.entities.model import EmbeddingInputType
from dify_plugin.entities.model.text_embedding import TextEmbeddingResult, EmbeddingUsage
from dify_plugin.errors.model import CredentialsValidateFailedError, InvokeError

from models.text_embedding.protocols.base import BaseTextEmbeddingProtocol
from models.utils import (
    build_endpoint_url,
    apply_consumer_auth,
    apply_consumer_auth_with_context,
    AuthContext,
    consumer_auth_manager,
)

logger = logging.getLogger(__name__)


class OpenAICompatibleTextEmbeddingProtocol(BaseTextEmbeddingProtocol):
    """
    OpenAI Compatible API protocol implementation for Text Embedding.
    Handles validation and API calls for OpenAI-compatible embedding endpoints.
    """

    PROTOCOL_NAME = "openai_compatible"

    # API path for embeddings
    EMBEDDINGS_PATH = "v1/embeddings"

    def get_protocol_name(self) -> str:
        return self.PROTOCOL_NAME

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate credentials by sending a test embedding request to the OpenAI-compatible API.

        :param model: Model name
        :param credentials: Model credentials
        :raises CredentialsValidateFailedError: If validation fails
        """
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "*/*",
            }

            endpoint_url = build_endpoint_url(credentials, self.EMBEDDINGS_PATH)
            request_model = credentials.get("gateway_model_name") or model

            data = {
                "input": "ping",
                "model": request_model,
            }

            # Apply consumer authentication
            if consumer_auth_manager.requires_body(credentials):
                auth_ctx = AuthContext(
                    headers=headers,
                    credentials=credentials,
                    method="POST",
                    url=endpoint_url,
                    body=data,
                    content_type="application/json",
                )
                headers = apply_consumer_auth_with_context(auth_ctx)
            else:
                headers = apply_consumer_auth(headers, credentials)

            response = requests.post(
                url=endpoint_url,
                headers=headers,
                data=json.dumps(data),
                timeout=(10, 300),
            )

            if response.status_code != 200:
                raise CredentialsValidateFailedError(
                    f"Credentials validation failed with status code {response.status_code}, "
                    f"endpoint: {endpoint_url}, response: {response.text}"
                )

            try:
                json_result = response.json()
            except json.JSONDecodeError as e:
                raise CredentialsValidateFailedError(
                    "Credentials validation failed: JSON decode error"
                ) from e

            if "model" not in json_result and "data" not in json_result:
                raise CredentialsValidateFailedError(
                    "Credentials validation failed: invalid response"
                )

            logger.info(f"Text embedding credentials validated successfully for model: {model}")

        except CredentialsValidateFailedError:
            raise
        except Exception as ex:
            logger.exception(f"Unexpected error during credentials validation: {ex}")
            raise CredentialsValidateFailedError(str(ex)) from ex

    def generate(
            self,
            model: str,
            credentials: dict,
            texts: list[str],
            user: Optional[str] = None,
            input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
            callbacks: Optional[dict] = None,
    ) -> TextEmbeddingResult:
        """
        Generate text embeddings using OpenAI-compatible API.

        :param model: Model name
        :param credentials: Model credentials
        :param texts: List of texts to embed
        :param user: Unique user id
        :param input_type: Input type (query or document)
        :param callbacks: Callback functions (get_num_tokens_by_gpt2, calc_response_usage, etc.)
        :return: Text embedding result
        """
        callbacks = callbacks or {}

        headers = {
            "Content-Type": "application/json",
            "Accept": "*/*",
        }

        endpoint_url = build_endpoint_url(credentials, self.EMBEDDINGS_PATH)
        request_model = credentials.get("gateway_model_name") or model

        extra_model_kwargs = {}
        if user:
            extra_model_kwargs["user"] = user
        extra_model_kwargs["encoding_format"] = "float"

        # Get model properties from callbacks
        get_context_size = callbacks.get("get_context_size")
        get_max_chunks = callbacks.get("get_max_chunks")
        get_num_tokens_by_gpt2 = callbacks.get("get_num_tokens_by_gpt2")

        context_size = get_context_size(model, credentials) if get_context_size else 8192
        max_chunks = get_max_chunks(model, credentials) if get_max_chunks else 1

        inputs = []
        indices = []
        used_tokens = 0

        for i, text in enumerate(texts):
            # Token count approximation based on GPT2 tokenizer
            if get_num_tokens_by_gpt2:
                num_tokens = get_num_tokens_by_gpt2(text)
            else:
                # Fallback: rough estimation ~4 characters per token
                num_tokens = len(text) // 4

            if num_tokens >= context_size:
                cutoff = int((len(text) * context_size) // num_tokens)
                # if num tokens is larger than context length, only use the start
                inputs.append(text[0:cutoff])
            else:
                inputs.append(text)
            indices += [i]

        batched_embeddings = []
        _iter = range(0, len(inputs), max_chunks)

        for i in _iter:
            data = {
                "input": inputs[i: i + max_chunks],
                "model": request_model,
                **extra_model_kwargs,
            }

            # Apply consumer authentication
            if consumer_auth_manager.requires_body(credentials):
                auth_ctx = AuthContext(
                    headers=headers.copy(),
                    credentials=credentials,
                    method="POST",
                    url=endpoint_url,
                    body=data,
                    content_type="application/json",
                )
                request_headers = apply_consumer_auth_with_context(auth_ctx)
            else:
                request_headers = apply_consumer_auth(headers.copy(), credentials)

            response = requests.post(
                endpoint_url,
                headers=request_headers,
                data=json.dumps(data),
                timeout=(10, 300),
            )

            if response.status_code != 200:
                raise InvokeError(
                    f"API request failed with status code {response.status_code}: {response.text}"
                )

            response.raise_for_status()
            response_data = response.json()

            # Extract embeddings and used tokens from the response
            embeddings_batch = [data["embedding"] for data in response_data["data"]]
            embedding_used_tokens = response_data.get("usage", {}).get("total_tokens", 0)

            used_tokens += embedding_used_tokens
            batched_embeddings += embeddings_batch

        # Calculate usage
        calc_response_usage = callbacks.get("calc_response_usage")
        if calc_response_usage:
            usage = calc_response_usage(model=model, credentials=credentials, tokens=used_tokens)
        else:
            usage = EmbeddingUsage(
                tokens=used_tokens,
                total_tokens=used_tokens,
                unit_price=0,
                price_unit=0,
                total_price=0,
                currency="USD",
                latency=0,
            )

        logger.debug(f"Text embedding generation completed: {len(batched_embeddings)} embeddings, {used_tokens} tokens")

        return TextEmbeddingResult(
            embeddings=batched_embeddings,
            usage=usage,
            model=request_model,
        )

    def get_num_tokens(
            self,
            model: str,
            credentials: dict,
            texts: list[str],
            callbacks: Optional[dict] = None,
    ) -> list[int]:
        """
        Get number of tokens for given texts.
        Uses GPT2 tokenizer approximation.

        :param model: Model name
        :param credentials: Model credentials
        :param texts: List of texts
        :param callbacks: Callback functions (get_num_tokens_by_gpt2)
        :return: List of token counts for each text
        """
        callbacks = callbacks or {}
        get_num_tokens_by_gpt2 = callbacks.get("get_num_tokens_by_gpt2")

        if get_num_tokens_by_gpt2:
            return [get_num_tokens_by_gpt2(text) for text in texts]
        else:
            # Fallback: rough estimation ~4 characters per token
            return [len(text) // 4 for text in texts]


# Singleton instance for easy access
openai_compatible_text_embedding_protocol = OpenAICompatibleTextEmbeddingProtocol()
