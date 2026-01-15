"""
Higress AI Gateway Text Embedding implementation.
"""

import logging
import time
from decimal import Decimal
from typing import Optional

from dify_plugin.entities import I18nObject
from dify_plugin.entities.model import (
    AIModelEntity,
    EmbeddingInputType,
    FetchFrom,
    ModelPropertyKey,
    ModelType,
    PriceConfig,
    PriceType,
)
from dify_plugin.entities.model.text_embedding import EmbeddingUsage, TextEmbeddingResult
from dify_plugin.errors.model import CredentialsValidateFailedError
from dify_plugin.interfaces.model.text_embedding_model import TextEmbeddingModel

from models._common import _CommonHigress
from models.text_embedding.utils import get_embedding_model_protocol
from models.text_embedding.protocols import OpenAICompatibleTextEmbeddingProtocol

logger = logging.getLogger(__name__)


class HigressTextEmbeddingModel(_CommonHigress, TextEmbeddingModel):
    """
    Model class for Higress AI Gateway text embedding model.
    Supports OpenAI-compatible protocol.
    """

    # Protocol handlers registry
    _protocol_handlers = {
        "openai_compatible": OpenAICompatibleTextEmbeddingProtocol(),
    }

    def _get_protocol_handler(self, credentials: dict):
        """
        Get the protocol handler based on credentials.
        
        :param credentials: Model credentials
        :return: Protocol handler instance
        :raises CredentialsValidateFailedError: If protocol is not supported
        """
        protocol = get_embedding_model_protocol(credentials)
        handler = self._protocol_handlers.get(protocol)
        
        if not handler:
            logger.error(f"Unsupported embedding model protocol: {protocol}")
            raise CredentialsValidateFailedError(
                f"Unsupported embedding model protocol: {protocol}"
            )
        
        return handler

    def _get_callbacks(self) -> dict:
        """
        Get callback functions to pass to protocol handlers.
        
        :return: Dictionary of callback functions
        """
        return {
            "get_num_tokens_by_gpt2": self._get_num_tokens_by_gpt2,
            "calc_response_usage": self._calc_response_usage,
            "get_context_size": self._get_context_size,
            "get_max_chunks": self._get_max_chunks,
        }

    def _get_context_size(self, model: str, credentials: dict) -> int:
        """
        Get context size for the model.
        
        :param model: Model name
        :param credentials: Model credentials
        :return: Context size
        """
        return int(credentials.get("context_size", 8192))

    def _get_max_chunks(self, model: str, credentials: dict) -> int:
        """
        Get max chunks for the model.
        
        :param model: Model name
        :param credentials: Model credentials
        :return: Max chunks
        """
        return int(credentials.get("max_chunks", 1))

    def _calc_response_usage(
        self,
        model: str,
        credentials: dict,
        tokens: int,
    ) -> EmbeddingUsage:
        """
        Calculate response usage.
        
        :param model: Model name
        :param credentials: Model credentials
        :param tokens: Number of tokens used
        :return: EmbeddingUsage object
        """
        # Get input price info
        input_price_info = self.get_price(
            model=model,
            credentials=credentials,
            price_type=PriceType.INPUT,
            tokens=tokens,
        )
        
        # Transform usage
        usage = EmbeddingUsage(
            tokens=tokens,
            total_tokens=tokens,
            unit_price=input_price_info.unit_price,
            price_unit=input_price_info.unit,
            total_price=input_price_info.total_amount,
            currency=input_price_info.currency,
            latency=time.perf_counter() - self.started_at,
        )
        
        return usage

    def _invoke(
        self,
        model: str,
        credentials: dict,
        texts: list[str],
        user: Optional[str] = None,
        input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> TextEmbeddingResult:
        """
        Invoke text embedding model.

        :param model: model name
        :param credentials: model credentials
        :param texts: texts to embed
        :param user: unique user id
        :param input_type: input type (query or document)
        :return: embeddings result
        """
        # Add prefix based on input type
        prefix = self._get_prefix(credentials, input_type)
        texts = self._add_prefix(texts, prefix)
        
        return self._generate(
            model=model,
            credentials=credentials,
            texts=texts,
            user=user,
            input_type=input_type,
        )

    def _get_prefix(self, credentials: dict, input_type: EmbeddingInputType) -> str:
        """
        Get prefix based on input type.
        
        :param credentials: Model credentials
        :param input_type: Input type (query or document)
        :return: Prefix string
        """
        if input_type == EmbeddingInputType.DOCUMENT:
            return credentials.get("document_prefix", "")
        
        if input_type == EmbeddingInputType.QUERY:
            return credentials.get("query_prefix", "")
        
        return ""

    def _add_prefix(self, texts: list[str], prefix: str) -> list[str]:
        """
        Add prefix to texts.
        
        :param texts: List of texts
        :param prefix: Prefix to add
        :return: List of texts with prefix
        """
        return [f"{prefix} {text}" for text in texts] if prefix else texts

    def _generate(
        self,
        model: str,
        credentials: dict,
        texts: list[str],
        user: Optional[str] = None,
        input_type: EmbeddingInputType = EmbeddingInputType.DOCUMENT,
    ) -> TextEmbeddingResult:
        """
        Generate text embeddings using the appropriate protocol handler.

        :param model: model name
        :param credentials: model credentials
        :param texts: texts to embed
        :param user: unique user id
        :param input_type: input type (query or document)
        :return: embeddings result
        """
        handler = self._get_protocol_handler(credentials)
        callbacks = self._get_callbacks()
        
        return handler.generate(
            model=model,
            credentials=credentials,
            texts=texts,
            user=user,
            input_type=input_type,
            callbacks=callbacks,
        )

    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        texts: list[str],
    ) -> list[int]:
        """
        Get number of tokens for given texts.

        :param model: model name
        :param credentials: model credentials
        :param texts: texts to tokenize
        :return: list of token counts for each text
        """
        handler = self._get_protocol_handler(credentials)
        callbacks = self._get_callbacks()
        
        return handler.get_num_tokens(
            model=model,
            credentials=credentials,
            texts=texts,
            callbacks=callbacks,
        )

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials based on the selected protocol.

        :param model: model name
        :param credentials: model credentials
        :raises CredentialsValidateFailedError: If validation fails
        """
        try:
            handler = self._get_protocol_handler(credentials)
            handler.validate_credentials(model, credentials)
            logger.info(f"Text embedding credentials validated successfully for model: {model}")
        except CredentialsValidateFailedError:
            raise
        except Exception as e:
            logger.exception(f"Unexpected error during credentials validation: {e}")
            raise

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        """
        Generate custom model entities from credentials.
        
        :param model: model name
        :param credentials: model credentials
        :return: AIModelEntity for the text embedding model
        """
        try:
            # Get context size and max chunks
            context_size = int(credentials.get("context_size", 8192))
            max_chunks = int(credentials.get("max_chunks", 1))
            
            entity = AIModelEntity(
                model=model,
                label=I18nObject(en_US=model),
                model_type=ModelType.TEXT_EMBEDDING,
                fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
                model_properties={
                    ModelPropertyKey.CONTEXT_SIZE: context_size,
                    ModelPropertyKey.MAX_CHUNKS: max_chunks,
                },
                parameter_rules=[],
                pricing=PriceConfig(
                    input=Decimal(credentials.get("input_price", 0)),
                    unit=Decimal(credentials.get("unit", 0)),
                    currency=credentials.get("currency", "USD"),
                ),
            )
            
            # Set display name if provided
            if "display_name" in credentials and credentials["display_name"] != "":
                entity.label = I18nObject(
                    en_US=credentials["display_name"], 
                    zh_Hans=credentials["display_name"]
                )
            
            return entity
            
        except Exception as e:
            logger.exception(f"Error in get_customizable_model_schema: {e}")
            raise
