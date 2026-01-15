"""
Base protocol class for model API implementations.
"""

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Optional, Union
import logging

from dify_plugin.entities.model.llm import LLMResult
from dify_plugin.entities.model.message import PromptMessage, PromptMessageTool

logger = logging.getLogger(__name__)


class BaseProtocol(ABC):
    """
    Abstract base class for model protocol implementations.
    Each protocol (OpenAI, Anthropic, etc.) should extend this class.
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
        prompt_messages: list[PromptMessage],
        model_parameters: dict,
        tools: Optional[list[PromptMessageTool]] = None,
        stop: Optional[list[str]] = None,
        stream: bool = True,
        user: Optional[str] = None,
        callbacks: Optional[dict] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Generate LLM response based on the protocol.
        
        :param model: Model name
        :param credentials: Model credentials
        :param prompt_messages: Prompt messages
        :param model_parameters: Model parameters
        :param tools: Tools for tool calling
        :param stop: Stop words
        :param stream: Whether to stream the response
        :param user: Unique user id
        :param callbacks: Callback functions for token calculation, usage calculation, etc.
        :return: Full response or stream response chunk generator
        """
        raise NotImplementedError

    @abstractmethod
    def get_num_tokens(
        self,
        model: str,
        credentials: dict,
        prompt_messages: list[PromptMessage],
        tools: Optional[list[PromptMessageTool]] = None,
        callbacks: Optional[dict] = None,
    ) -> int:
        """
        Get number of tokens for given prompt messages.
        
        :param model: Model name
        :param credentials: Model credentials
        :param prompt_messages: Prompt messages
        :param tools: Tools for tool calling
        :param callbacks: Callback functions for tokenization
        :return: Number of tokens
        """
        raise NotImplementedError