"""
Alibaba Cloud DashScope Rerank protocol implementation.
"""

import json
import logging
from typing import Optional

import requests

from dify_plugin.entities.model.rerank import RerankResult, RerankDocument
from dify_plugin.errors.model import CredentialsValidateFailedError, InvokeError

from models.rerank.protocols.base import BaseRerankProtocol
from models.utils import (
    build_endpoint_url,
    apply_consumer_auth,
    apply_consumer_auth_with_context,
    AuthContext,
    consumer_auth_manager,
)

logger = logging.getLogger(__name__)


class DashScopeRerankProtocol(BaseRerankProtocol):
    """
    Alibaba Cloud DashScope API protocol implementation for Rerank.
    Handles validation and API calls for DashScope rerank endpoints.
    """

    PROTOCOL_NAME = "dashscope_rerank"

    # API path for DashScope rerank
    RERANK_PATH = "api/v1/services/rerank/text-rerank/text-rerank"

    def get_protocol_name(self) -> str:
        return self.PROTOCOL_NAME

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate credentials by sending a test rerank request to the DashScope API.

        :param model: Model name
        :param credentials: Model credentials
        :raises CredentialsValidateFailedError: If validation fails
        """
        try:
            self.invoke(
                model=model,
                credentials=credentials,
                query="What is the capital of the United States?",
                docs=[
                    "Carson City is the capital city of the American state of Nevada. At the 2010 United States Census, Carson City had a population of 55,274.",
                    "The Commonwealth of the Northern Mariana Islands is a group of islands in the Pacific Ocean that are a political division controlled by the United States. Its capital is Saipan.",
                ],
                score_threshold=0.8,
            )
        except CredentialsValidateFailedError:
            raise
        except Exception as ex:
            raise CredentialsValidateFailedError(str(ex)) from ex

    def invoke(
            self,
            model: str,
            credentials: dict,
            query: str,
            docs: list[str],
            score_threshold: Optional[float] = None,
            top_n: Optional[int] = None,
            user: Optional[str] = None,
    ) -> RerankResult:
        """
        Invoke rerank model using DashScope API.

        :param model: Model name
        :param credentials: Model credentials
        :param query: Search query
        :param docs: Documents to rerank
        :param score_threshold: Score threshold for filtering results
        :param top_n: Top N results to return
        :param user: Unique user id
        :return: Rerank result
        """
        # Return empty result if no documents
        if len(docs) == 0:
            return RerankResult(model=model, docs=[])

        endpoint_url = build_endpoint_url(credentials, self.RERANK_PATH)
        request_model = credentials.get("gateway_model_name") or "gte-rerank-v2"

        headers = {
            "Content-Type": "application/json",
            "Accept": "*/*",
        }

        # Build request body based on model type
        # qwen3-rerank uses flat structure, other models use nested input/parameters
        if request_model == "qwen3-rerank":
            # qwen3-rerank: flat structure
            data = {
                "model": request_model,
                "query": query,
                "documents": docs,
            }
            if top_n is not None:
                data["top_n"] = top_n
        else:
            # gte-rerank-v2 and others: nested structure
            data = {
                "model": request_model,
                "input": {
                    "query": query,
                    "documents": docs,
                },
                "parameters": {
                    "return_documents": True,
                }
            }
            if top_n is not None:
                data["parameters"]["top_n"] = top_n

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

        try:
            response = requests.post(
                endpoint_url,
                headers=headers,
                data=json.dumps(data),
                timeout=(10, 300),
            )

            response_data = response.json()

            # Check for API error response
            if "code" in response_data:
                error_code = response_data.get("code", "Unknown")
                error_message = response_data.get("message", "Unknown error")
                raise CredentialsValidateFailedError(
                    f"DashScope API error: {error_code} - {error_message}"
                )

            if response.status_code != 200:
                raise InvokeError(
                    f"API request failed with status code {response.status_code}: {response.text}"
                )

            # Parse successful response
            output = response_data.get("output", {})
            results = output.get("results", [])

            rerank_documents = []
            for result in results:
                index = result.get("index", 0)
                relevance_score = result.get("relevance_score", 0.0)

                # Get document text from response or fallback to original docs
                document_obj = result.get("document", {})
                if document_obj and isinstance(document_obj, dict):
                    text = document_obj.get("text", docs[index] if index < len(docs) else "")
                else:
                    text = docs[index] if index < len(docs) else ""

                # Apply score threshold filter
                if score_threshold is None or relevance_score >= score_threshold:
                    rerank_document = RerankDocument(
                        index=index,
                        text=text,
                        score=relevance_score,
                    )
                    rerank_documents.append(rerank_document)

            # Results are already sorted by relevance_score from API,
            # but ensure sorting in descending order
            rerank_documents.sort(key=lambda doc: doc.score, reverse=True)

            return RerankResult(model=request_model, docs=rerank_documents)

        except CredentialsValidateFailedError:
            raise
        except InvokeError:
            raise
        except requests.exceptions.RequestException as e:
            raise InvokeError(f"Request failed: {str(e)}") from e
        except json.JSONDecodeError as e:
            raise InvokeError(f"Failed to parse response: {str(e)}") from e
        except Exception as e:
            raise InvokeError(f"Unexpected error: {str(e)}") from e


# Singleton instance for easy access
dashscope_rerank_protocol = DashScopeRerankProtocol()
