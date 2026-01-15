"""
Base protocol class for Text Embedding model API implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional
import logging

from dify_plugin.entities.model import EmbeddingInputType
from dify_plugin.entities.model.text_embedding import TextEmbeddingResult

logger = logging.getLogger(__name__)


class BaseTextEmbeddingProtocol(ABC):
    """
    Abstract base class for Text Embedding protocol implementations.
    Each protocol (OpenAI, etc.) should extend this class.
    """
    
    @abstractmethod
    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate model credentials by sending a test request.
        
        :param model: Model name
        :param credentials: Model credentials
        :raises CredentialsValidateFailedError: If validation fails
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_protocol_name(self) -> str:
        """
        Get the protocol name identifier.
        
        :return: Protocol name (e.g., "openai_compatible")
        """
        raise NotImplementedError

    @abstractmethod
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
        Generate text embeddings based on the protocol.
        
        :param model: Model name
        :param credentials: Model credentials
        :param texts: List of texts to embed
        :param user: Unique user id
        :param input_type: Input type (query or document)
        :param callbacks: Callback functions for token calculation, usage calculation, etc.
        :return: Text embedding result
        """
        raise NotImplementedError

    @abstractmethod
    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        texts: list[str],
        callbacks: Optional[dict] = None,
    ) -> list[int]:
        """
        Get number of tokens for given texts.
        
        :param model: Model name
        :param credentials: Model credentials
        :param texts: List of texts
        :param callbacks: Callback functions for tokenization
        :return: List of token counts for each text
        """
        raise NotImplementedError
