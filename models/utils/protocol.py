"""
Protocol utility functions for Higress AI Gateway.

This module provides functions to determine the protocol to use
based on model credentials and usage scenarios.
"""


def get_llm_protocol(credentials: dict) -> str:
    """
    Get the LLM protocol from credentials based on usage scenario.
    
    :param credentials: Model credentials
    :return: Model protocol (e.g., "openai_compatible", "dashscope_image_generation")
    """
    usage_scenario = credentials.get("llm_usage_scenario", "text_generation")
    
    if usage_scenario == "image_generation":
        return credentials.get("image_model_protocol", "dashscope_image_generation")
    else:
        return credentials.get("text_model_protocol", "openai_compatible")


def get_embedding_protocol(credentials: dict) -> str:
    """
    Get the embedding model protocol from credentials.
    
    :param credentials: Model credentials
    :return: Model protocol (e.g., "openai_compatible")
    """
    return credentials.get("embedding_model_protocol", "openai_compatible")


def get_rerank_protocol(credentials: dict) -> str:
    """
    Get the rerank model protocol from credentials.

    :param credentials: Model credentials
    :return: Model protocol (e.g., "dashscope_rerank")
    """
    return credentials.get("rerank_model_protocol", "dashscope_rerank")


def get_model_mode(credentials: dict) -> str:
    """
    Get the model mode from credentials with default value.
    
    :param credentials: Model credentials
    :return: Model mode ("chat" or "completion")
    """
    return credentials.get("mode", "chat")
