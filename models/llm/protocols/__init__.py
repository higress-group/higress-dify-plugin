"""
Protocol implementations for LLM APIs.
"""

from models.llm.protocols.openai_compatible import OpenAICompatibleProtocol
from models.llm.protocols.dashscope_image_generation import DashScopeImageGenerationProtocol

__all__ = ["OpenAICompatibleProtocol", "DashScopeImageGenerationProtocol"]

