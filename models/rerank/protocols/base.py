"""
Base protocol class for Rerank model API implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional
import logging

from dify_plugin.entities.model.rerank import RerankResult

logger = logging.getLogger(__name__)


class BaseRerankProtocol(ABC):
    """
    Abstract base class for Rerank protocol implementations.
    Each protocol (DashScope, etc.) should extend this class.
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

        :return: Protocol name (e.g., "dashscope_rerank")
        """
        raise NotImplementedError

    @abstractmethod
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
        Invoke rerank model based on the protocol.

        :param model: Model name
        :param credentials: Model credentials
        :param query: Search query
        :param docs: Documents to rerank
        :param score_threshold: Score threshold for filtering results
        :param top_n: Top N results to return
        :param user: Unique user id
        :return: Rerank result
        """
        raise NotImplementedError
