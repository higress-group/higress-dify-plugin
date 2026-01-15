"""
DashScope Image Generation protocol implementation.
百炼图片生成协议实现。
"""

import json
import logging
from collections.abc import Generator
from typing import Optional, Union

import requests

from dify_plugin.entities.model.llm import LLMResult, LLMUsage
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    PromptMessage,
    PromptMessageContentType,
    PromptMessageTool,
    UserPromptMessage,
)
from dify_plugin.errors.model import CredentialsValidateFailedError, InvokeError

from models.llm.protocols.base import BaseProtocol
from models.llm.utils import build_endpoint_url
from models.utils import (
    apply_consumer_auth,
    apply_consumer_auth_with_context,
    AuthContext,
    consumer_auth_manager,
)

logger = logging.getLogger(__name__)


class DashScopeImageGenerationProtocol(BaseProtocol):
    """
    DashScope Image Generation API protocol implementation.
    百炼图片生成 API 协议实现。
    
    Handles validation and API calls for DashScope multimodal generation endpoints.
    """
    
    PROTOCOL_NAME = "dashscope_image_generation"
    
    # API path for image generation
    IMAGE_GENERATION_PATH = "api/v1/services/aigc/multimodal-generation/generation"
    
    def get_protocol_name(self) -> str:
        return self.PROTOCOL_NAME
    
    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate credentials by sending a test image generation request.
        
        :param model: Model name
        :param credentials: Model credentials
        :raises CredentialsValidateFailedError: If validation fails
        """
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "*/*",
            }
            
            # Build the full endpoint URL
            endpoint_url = build_endpoint_url(credentials, self.IMAGE_GENERATION_PATH)
            
            # Build test data for validation request
            # Use gateway_model_name if provided, otherwise use default qwen-image-plus
            image_model = credentials.get("gateway_model_name") or "qwen-image-plus"
            data = {
                "model": image_model,
                "input": {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"text": "ping"}]
                        }
                    ]
                }
            }
            
            # Apply consumer authentication
            if consumer_auth_manager.requires_body(credentials):
                # For HMAC auth, we need request body for signature
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
                # Simple auth (API Key) doesn't need request body
                headers = apply_consumer_auth(headers, credentials)
            
            # Send a POST request to validate credentials
            response = requests.post(
                endpoint_url,
                headers=headers,
                json=data,
                timeout=(10, 300)
            )
            
            if response.status_code != 200:
                raise CredentialsValidateFailedError(
                    f"Credentials validation failed with status code {response.status_code} "
                    f"and response body {response.text}"
                )
            
            try:
                json_result = response.json()
            except json.JSONDecodeError:
                raise CredentialsValidateFailedError(
                    f"Credentials validation failed: JSON decode error, response body {response.text}"
                ) from None
            
            # Verify response contains expected output structure
            if "output" not in json_result or "choices" not in json_result.get("output", {}):
                raise CredentialsValidateFailedError(
                    f"Credentials validation failed: invalid response format, "
                    f"response body {response.text}"
                )
            
            logger.info(f"DashScope image generation credentials validated successfully for model: {model}")
                
        except CredentialsValidateFailedError:
            raise
        except Exception as ex:
            raise CredentialsValidateFailedError(
                f"An error occurred during credentials validation: {ex!s}"
            ) from ex

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
        Generate image using DashScope Image Generation API.
        
        :param model: Model name (e.g., "qwen-image-plus")
        :param credentials: Model credentials
        :param prompt_messages: Prompt messages containing the image generation prompt
        :param model_parameters: Model parameters (size, negative_prompt, etc.)
        :param tools: Not used for image generation
        :param stop: Not used for image generation
        :param stream: Not used for image generation
        :param user: Not used for image generation
        :param callbacks: Not used for image generation
        :return: LLMResult containing the generated image URL
        """
        
        headers = {
            "Content-Type": "application/json",
            "Accept": "*/*",
        }
        
        # Build endpoint URL
        endpoint_url = build_endpoint_url(credentials, self.IMAGE_GENERATION_PATH)
        
        # Extract prompt from messages
        prompt_text = self._extract_prompt_from_messages(prompt_messages)
        
        # Use gateway_model_name if provided, otherwise use default qwen-image-plus
        image_model = credentials.get("gateway_model_name") or "qwen-image-plus"
        
        # Build request data in DashScope format
        data = {
            "model": image_model,
            "input": {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"text": prompt_text}
                        ]
                    }
                ]
            },
            "parameters": self._build_parameters(model_parameters)
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
        
        logger.debug(f"DashScope Image Generation request to {endpoint_url}")
        
        # Send request
        response = requests.post(
            endpoint_url,
            headers=headers,
            json=data,
            timeout=(10, 300)
        )
        
        if response.encoding is None or response.encoding == "ISO-8859-1":
            response.encoding = "utf-8"
        
        if response.status_code != 200:
            raise InvokeError(
                f"DashScope API request failed with status code {response.status_code}: {response.text}"
            )
        
        # Image generation doesn't support streaming, return result directly
        return self._handle_generate_response(model, response)
    
    def _extract_prompt_from_messages(self, prompt_messages: list[PromptMessage]) -> str:
        """
        Extract the prompt text from prompt messages.
        
        :param prompt_messages: List of prompt messages
        :return: Combined prompt text
        """
        prompt_parts = []
        
        for message in prompt_messages:
            if isinstance(message, UserPromptMessage):
                if isinstance(message.content, str):
                    prompt_parts.append(message.content)
                elif isinstance(message.content, list):
                    for content in message.content:
                        if content.type == PromptMessageContentType.TEXT:
                            prompt_parts.append(content.data)
        
        return "\n".join(prompt_parts)
    
    def _build_parameters(self, model_parameters: dict) -> dict:
        """
        Build DashScope parameters from model parameters.
        
        :param model_parameters: Model parameters from request
        :return: DashScope format parameters
        """
        params = {}
        
        # Map common parameters to DashScope format
        if "negative_prompt" in model_parameters:
            params["negative_prompt"] = model_parameters["negative_prompt"]
        else:
            params["negative_prompt"] = ""
        
        if "prompt_extend" in model_parameters:
            params["prompt_extend"] = model_parameters["prompt_extend"]
        else:
            params["prompt_extend"] = True
        
        if "watermark" in model_parameters:
            params["watermark"] = model_parameters["watermark"]
        else:
            params["watermark"] = False
        
        if "size" in model_parameters:
            params["size"] = model_parameters["size"]
        else:
            params["size"] = "1328*1328"  # Default size (allowed: 1664*928, 1472*1140, 1328*1328, 1140*1472, 928*1664)
        
        return params
    
    def _handle_generate_response(
        self,
        model: str,
        response: requests.Response,
    ) -> LLMResult:
        """
        Handle the image generation response from DashScope API.
        
        Response format:
        {
            "output": {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": [
                                {"image": "https://..."}
                            ],
                            "role": "assistant"
                        }
                    }
                ],
                "task_metric": {...}
            },
            "usage": {
                "height": 1328,
                "image_count": 1,
                "width": 1328
            },
            "request_id": "..."
        }
        
        :param model: Model name
        :param response: API response
        :return: LLMResult with ImagePromptMessageContent
        """
        try:
            response_json = response.json()
        except json.JSONDecodeError as e:
            raise InvokeError(f"Failed to parse response JSON: {e}, response: {response.text}")
        
        logger.debug(f"DashScope response received for model: {model}")
        
        # Extract image URL from response
        output = response_json.get("output", {})
        choices = output.get("choices", [])
        
        if not choices:
            raise InvokeError(f"No choices in response: {response_json}")
        
        first_choice = choices[0]
        message = first_choice.get("message", {})
        content_list = message.get("content", [])
        
        # Extract image URLs from response
        image_urls = []
        for content_item in content_list:
            if "image" in content_item:
                image_urls.append(content_item["image"])
        
        if not image_urls:
            raise InvokeError(f"No images in response: {response_json}")
        
        # Return URLs as plain text (one URL per line if multiple)
        response_content = "\n".join(image_urls)
        
        # Create assistant message with plain text content
        assistant_message = AssistantPromptMessage(content=response_content, tool_calls=[])
        
        # Build result (image generation doesn't involve token billing, use zero usage)
        result = LLMResult(
            id=response_json.get("request_id", ""),
            model=model,
            message=assistant_message,
            usage=LLMUsage(
                prompt_tokens=0,
                prompt_unit_price=0,
                prompt_price_unit=0,
                prompt_price=0,
                completion_tokens=0,
                completion_unit_price=0,
                completion_price_unit=0,
                completion_price=0,
                total_tokens=0,
                total_price=0,
                currency="USD",
                latency=0,
            ),
        )
        
        return result
    
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
        Image generation doesn't involve token billing, always return 0.
        
        :param model: Model name
        :param credentials: Model credentials
        :param prompt_messages: Prompt messages
        :param tools: Not used for image generation
        :param callbacks: Callback functions
        :return: Always 0 for image generation
        """
        return 0


# Singleton instance for easy access
dashscope_image_generation_protocol = DashScopeImageGenerationProtocol()
