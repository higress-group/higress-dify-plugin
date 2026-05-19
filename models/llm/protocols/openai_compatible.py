"""
OpenAI Compatible protocol implementation.
"""

import re
import codecs
import json
import logging
import uuid
from contextlib import suppress
from collections.abc import Generator
from typing import Optional, Union, cast, Any, List

import requests
from pydantic import TypeAdapter, ValidationError

from dify_plugin.entities.model.llm import LLMMode, LLMResult, LLMResultChunk, LLMResultChunkDelta
from dify_plugin.entities.model.message import (
    AssistantPromptMessage,
    AudioPromptMessageContent,
    ImagePromptMessageContent,
    PromptMessage,
    PromptMessageContent,
    PromptMessageContentType,
    PromptMessageFunction,
    PromptMessageTool,
    SystemPromptMessage,
    ToolPromptMessage,
    UserPromptMessage,
    VideoPromptMessageContent,
)
from dify_plugin.errors.model import CredentialsValidateFailedError, InvokeError

from models.llm.protocols.base import BaseProtocol
from models.llm.utils import build_endpoint_url, get_model_mode
from models.utils import (
    apply_consumer_auth,
    apply_consumer_auth_with_context,
    AuthContext,
    consumer_auth_manager,
)

logger = logging.getLogger(__name__)


def _gen_tool_call_id() -> str:
    return f"chatcmpl-tool-{uuid.uuid4().hex!s}"


def _increase_tool_call(
        new_tool_calls: list[AssistantPromptMessage.ToolCall],
        existing_tools_calls: list[AssistantPromptMessage.ToolCall]
):
    """
    Merge incremental tool call updates into existing tool calls.
    
    :param new_tool_calls: List of new tool call deltas to be merged.
    :param existing_tools_calls: List of existing tool calls to be modified IN-PLACE.
    """

    def get_tool_call(tool_call_id: str):
        """
        Get or create a tool call by ID
        
        :param tool_call_id: tool call ID
        :return: existing or new tool call
        """
        if not tool_call_id:
            return existing_tools_calls[-1]

        _tool_call = next(
            (_tool_call for _tool_call in existing_tools_calls if _tool_call.id == tool_call_id),
            None
        )
        if _tool_call is None:
            _tool_call = AssistantPromptMessage.ToolCall(
                id=tool_call_id,
                type="function",
                function=AssistantPromptMessage.ToolCall.ToolCallFunction(name="", arguments=""),
            )
            existing_tools_calls.append(_tool_call)

        return _tool_call

    for new_tool_call in new_tool_calls:
        # generate ID for tool calls with function name but no ID to track them
        if new_tool_call.function.name and not new_tool_call.id:
            new_tool_call.id = _gen_tool_call_id()

        # get tool call
        tool_call = get_tool_call(new_tool_call.id)

        # update tool call
        if new_tool_call.id:
            tool_call.id = new_tool_call.id
        if new_tool_call.type:
            tool_call.type = new_tool_call.type
        if new_tool_call.function.name:
            tool_call.function.name = new_tool_call.function.name
        if new_tool_call.function.arguments:
            tool_call.function.arguments += new_tool_call.function.arguments


class OpenAICompatibleProtocol(BaseProtocol):
    """
    OpenAI Compatible API protocol implementation.
    Handles validation and API calls for OpenAI-compatible endpoints.
    """

    PROTOCOL_NAME = "openai_compatible"

    _THINK_PATTERN = re.compile(r"^<think>.*?</think>\s*", re.DOTALL)

    # API paths for different modes
    CHAT_COMPLETIONS_PATH = "v1/chat/completions"
    COMPLETIONS_PATH = "v1/completions"

    _DASHSCOPE_MODEL_PREFIXES = ("qwen", "qvq", "qwq")

    def get_protocol_name(self) -> str:
        return self.PROTOCOL_NAME

    @staticmethod
    def _is_dashscope_model(credentials: dict) -> bool:
        model_name = (credentials.get("gateway_model_name") or "").lower()
        return model_name.startswith(OpenAICompatibleProtocol._DASHSCOPE_MODEL_PREFIXES)

    @staticmethod
    def _guess_audio_format(data: str) -> str:
        if data.startswith("data:"):
            mime = data.split(";")[0].split(":")[1] if ";" in data else ""
            fmt = mime.split("/")[-1] if "/" in mime else ""
            if fmt:
                return fmt
        for ext in ("wav", "mp3", "aac", "flac", "ogg", "amr"):
            if ext in data.lower():
                return ext
        return "wav"

    def validate_credentials(self, model: str, credentials: dict) -> None:
        """
        Validate credentials by sending a ping request to the OpenAI-compatible API.
        
        :param model: Model name
        :param credentials: Model credentials
        :raises CredentialsValidateFailedError: If validation fails
        """
        try:
            headers = {
                "Content-Type": "application/json",
                "Accept": "*/*",  # Explicitly set Accept to avoid HMAC signature issues with empty value
            }

            # prepare the payload for a simple ping to the model
            # Use gateway_model_name if provided, otherwise use the original model name
            request_model = credentials.get("gateway_model_name") or model
            data = {"model": request_model}

            mode = get_model_mode(credentials)
            completion_type = LLMMode.value_of(mode)

            if completion_type is LLMMode.CHAT:
                data["messages"] = [{"role": "user", "content": "ping"}]
                endpoint_url = build_endpoint_url(credentials, self.CHAT_COMPLETIONS_PATH)
            elif completion_type is LLMMode.COMPLETION:
                data["prompt"] = "ping"
                endpoint_url = build_endpoint_url(credentials, self.COMPLETIONS_PATH)
            else:
                raise CredentialsValidateFailedError(
                    f"Unsupported completion mode: {mode}"
                )

            # Apply consumer authentication
            # Use full context for auth methods that require request body (e.g., HMAC)
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
                # Simple auth (API Key) doesn't need request body
                headers = apply_consumer_auth(headers, credentials)

            # ADD stream validate_credentials
            stream_mode_auth = credentials.get("stream_mode_auth", "use")
            if stream_mode_auth == "use":
                data["stream"] = True
                response = requests.post(
                    endpoint_url,
                    headers=headers,
                    json=data,
                    timeout=(10, 300),
                    stream=True
                )
                if response.status_code != 200:
                    raise CredentialsValidateFailedError(
                        f"Credentials validation failed with status code {response.status_code} "
                        f"and response body {response.text}"
                    )
                return

            # send a post request to validate the credentials
            response = requests.post(endpoint_url, headers=headers, json=data, timeout=(10, 300))

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

            if completion_type is LLMMode.CHAT and json_result.get("object", "") == "":
                json_result["object"] = "chat.completion"
            elif completion_type is LLMMode.COMPLETION and json_result.get("object", "") == "":
                json_result["object"] = "text_completion"

            if completion_type is LLMMode.CHAT and (
                    "object" not in json_result or json_result["object"] != "chat.completion"
            ):
                raise CredentialsValidateFailedError(
                    f"Credentials validation failed: invalid response object, "
                    f"must be 'chat.completion', response body {response.text}"
                )
            elif completion_type is LLMMode.COMPLETION and (
                    "object" not in json_result or json_result["object"] != "text_completion"
            ):
                raise CredentialsValidateFailedError(
                    f"Credentials validation failed: invalid response object, "
                    f"must be 'text_completion', response body {response.text}"
                )

            logger.info(f"LLM credentials validated successfully for model: {model}")

        except CredentialsValidateFailedError:
            raise
        except Exception as ex:
            raise CredentialsValidateFailedError(
                f"An error occurred during credentials validation: {ex!s}"
            ) from ex

    @classmethod
    def _drop_analyze_channel(self, prompt_messages: List[PromptMessage]) -> None:
        """
        Remove thinking content from assistant messages for better performance.

        Uses early exit and pre-compiled regex to minimize overhead.
        Args:
            prompt_messages:

        Returns:

        """
        for p in prompt_messages:
            # Early exit conditions
            if not isinstance(p, AssistantPromptMessage):
                continue
            if not isinstance(p.content, str):
                continue
            # Quick check to avoid regex if not needed
            if not p.content.startswith("<think>"):
                continue

            # Only perform regex substitution when necessary
            new_content = self._THINK_PATTERN.sub("", p.content, count=1)
            # Only update if changed
            if new_content != p.content:
                p.content = new_content

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

        # thinking mode preprocess
        agent_thought_support = credentials.get("agent_thought_support", "not_supported")
        enable_thinking_value = None
        if agent_thought_support == "only_thinking_supported":
            enable_thinking_value = True
        elif agent_thought_support == "not_supported":
            enable_thinking_value = False
        else:
            user_enable_thinking = model_parameters.pop("enable_thinking", None)
            if user_enable_thinking is not None:
                enable_thinking_value = bool(user_enable_thinking)

        # only when model support thinking, gen param about thinking
        if enable_thinking_value is not None and agent_thought_support in ["supported", "only_thinking_supported"]:
            chat_template_kwargs = model_parameters.setdefault("chat_template_kwargs", {})
            # Support vLLM/SGLang format (chat_template_kwargs)
            chat_template_kwargs["enable_thinking"] = enable_thinking_value
            chat_template_kwargs["thinking"] = enable_thinking_value

            # Support top-level `enable_thinking` parameter
            # This allows compatibility API format: {"enable_thinking": False/True}
            model_parameters["enable_thinking"] = enable_thinking_value

        # Remove thinking content from assistant messages for better performance.
        with suppress(Exception):
            self._drop_analyze_channel(prompt_messages)

        result = self._generate(model, credentials, prompt_messages, model_parameters, tools, stop, stream, user, callbacks)

        if enable_thinking_value is False:
            if stream:
                return self._filter_thinking_stream(result)
            else:
                return self._filter_thinking_result(result)

        return result

    def _generate(
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
        Generate LLM response using OpenAI-compatible API.
        
        :param model: Model name
        :param credentials: Model credentials
        :param prompt_messages: Prompt messages
        :param model_parameters: Model parameters
        :param tools: Tools for tool calling
        :param stop: Stop words
        :param stream: Whether to stream the response
        :param user: Unique user id
        :param callbacks: Callback functions (calc_response_usage, get_num_tokens_by_gpt2, etc.)
        :return: Full response or stream response chunk generator
        """
        callbacks = callbacks or {}
        headers = {
            "Content-Type": "application/json",
            "Accept-Charset": "utf-8",
            "Accept": "*/*",  # Explicitly set Accept to avoid HMAC signature issues with empty value
        }

        extra_headers = credentials.get("extra_headers")
        if extra_headers is not None:
            headers = {**headers, **extra_headers}

        # Handle response format
        response_format = model_parameters.get("response_format")
        if response_format:
            if response_format == "json_schema":
                json_schema = model_parameters.get("json_schema")
                if not json_schema:
                    raise ValueError("Must define JSON Schema when the response format is json_schema")
                try:
                    schema = TypeAdapter(dict[str, Any]).validate_json(json_schema)
                except Exception as exc:
                    raise ValueError(f"not correct json_schema format: {json_schema}") from exc
                model_parameters.pop("json_schema")
                model_parameters["response_format"] = {"type": "json_schema", "json_schema": schema}
            else:
                model_parameters["response_format"] = {"type": response_format}
        elif "json_schema" in model_parameters:
            del model_parameters["json_schema"]

        enable_web_search = model_parameters.pop("enable_web_search", None)

        # Use gateway_model_name if provided, otherwise use the original model name
        request_model = credentials.get("gateway_model_name") or model

        # Remove max_tokens from model_parameters to prevent it from being passed to the API
        filtered_model_parameters = {k: v for k, v in model_parameters.items() if
                                     k != "max_tokens" and k != "max_completion_tokens"}
        data = {"model": request_model, "stream": stream, **filtered_model_parameters}

        # Get completion type and build endpoint URL using our custom logic
        mode = get_model_mode(credentials)
        completion_type = LLMMode.value_of(mode)

        if completion_type is LLMMode.CHAT:
            endpoint_url = build_endpoint_url(credentials, self.CHAT_COMPLETIONS_PATH)
            data["messages"] = [self._convert_prompt_message_to_dict(m, credentials) for m in prompt_messages]
            if enable_web_search:
                data["web_search_options"] = {}
        elif completion_type is LLMMode.COMPLETION:
            endpoint_url = build_endpoint_url(credentials, self.COMPLETIONS_PATH)
            data["prompt"] = prompt_messages[0].content
        else:
            raise ValueError(f"Unsupported completion type for model configuration: {mode}")

        # annotate tools with names, descriptions, etc.
        function_calling_type = credentials.get("function_calling_type", "no_call")
        formatted_tools = []
        if tools:
            if function_calling_type == "function_call":
                data["functions"] = [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                    }
                    for tool in tools
                ]
            elif function_calling_type == "tool_call":
                data["tool_choice"] = "auto"
                for tool in tools:
                    formatted_tools.append(PromptMessageFunction(function=tool).model_dump())
                data["tools"] = formatted_tools

        if stop:
            data["stop"] = stop

        if user:
            data["user"] = user

        # Apply consumer authentication
        # Use full context for auth methods that require request body (e.g., HMAC)
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
            # Simple auth (API Key) doesn't need request body
            headers = apply_consumer_auth(headers, credentials)

        response = requests.post(
            endpoint_url,
            headers=headers,
            json=data,
            timeout=(10, 300),
            stream=stream
        )

        if response.encoding is None or response.encoding == "ISO-8859-1":
            response.encoding = "utf-8"

        if response.status_code != 200:
            raise InvokeError(f"API request failed with status code {response.status_code}: {response.text}")

        if stream:
            return self._handle_generate_stream_response(model, credentials, response, prompt_messages, callbacks)

        return self._handle_generate_response(model, credentials, response, prompt_messages, callbacks)

    def _create_final_llm_result_chunk(
            self,
            index: int,
            message: AssistantPromptMessage,
            finish_reason: str,
            usage: dict,
            model: str,
            prompt_messages: list[PromptMessage],
            credentials: dict,
            full_content: str,
            callbacks: dict,
    ) -> LLMResultChunk:
        """
        Create final LLM result chunk with usage information.
        """
        calc_response_usage = callbacks.get("calc_response_usage")

        # calculate num tokens
        prompt_tokens = usage and usage.get("prompt_tokens")
        if prompt_tokens is None:
            prompt_tokens = self._num_tokens_from_string(text=prompt_messages[0].content, callbacks=callbacks)

        completion_tokens = usage and usage.get("completion_tokens")
        if completion_tokens is None:
            completion_tokens = self._num_tokens_from_string(text=full_content, callbacks=callbacks)

        # transform usage
        usage_obj = None
        if calc_response_usage:
            usage_obj = calc_response_usage(model, credentials, prompt_tokens, completion_tokens)

        return LLMResultChunk(
            model=model,
            delta=LLMResultChunkDelta(
                index=index,
                message=message,
                finish_reason=finish_reason,
                usage=usage_obj
            ),
        )

    def _handle_generate_stream_response(
            self,
            model: str,
            credentials: dict,
            response: requests.Response,
            prompt_messages: list[PromptMessage],
            callbacks: dict,
    ) -> Generator:
        """
        Handle LLM stream response.

        :param model: model name
        :param credentials: model credentials
        :param response: streamed response
        :param prompt_messages: prompt messages
        :param callbacks: callback functions
        :return: llm response chunk generator
        """
        chunk_index = 0
        full_assistant_content = ""
        tools_calls: list[AssistantPromptMessage.ToolCall] = []
        finish_reason = None
        usage = None
        is_reasoning_started = False

        # delimiter for stream response, need unicode_escape
        delimiter = credentials.get("stream_mode_delimiter", "\n\n")
        delimiter = codecs.decode(delimiter, "unicode_escape")

        for chunk in response.iter_lines(decode_unicode=True, delimiter=delimiter):
            chunk = chunk.strip()
            if chunk:
                # ignore sse comments
                if chunk.startswith(":"):
                    continue
                decoded_chunk = chunk.strip().removeprefix("data:").lstrip()
                if decoded_chunk == "[DONE]":  # Some provider returns "data: [DONE]"
                    continue

                try:
                    chunk_json: dict = TypeAdapter(dict[str, Any]).validate_json(decoded_chunk)
                # stream ended
                except ValidationError:
                    yield self._create_final_llm_result_chunk(
                        index=chunk_index + 1,
                        message=AssistantPromptMessage(content=""),
                        finish_reason="Non-JSON encountered.",
                        usage=usage,
                        model=model,
                        credentials=credentials,
                        prompt_messages=prompt_messages,
                        full_content=full_assistant_content,
                        callbacks=callbacks,
                    )
                    break

                # handle the error here. for issue #11629
                if chunk_json.get("error") and chunk_json.get("choices") is None:
                    raise ValueError(chunk_json.get("error"))

                if chunk_json:  # noqa: SIM102
                    if u := chunk_json.get("usage"):
                        usage = u

                if not chunk_json or len(chunk_json["choices"]) == 0:
                    continue

                choice = chunk_json["choices"][0]
                finish_reason = chunk_json["choices"][0].get("finish_reason")
                chunk_index += 1

                if "delta" in choice:
                    delta = choice["delta"]
                    delta_content, is_reasoning_started = self._wrap_thinking_by_reasoning_content(
                        delta, is_reasoning_started
                    )

                    assistant_message_tool_calls = None

                    if "tool_calls" in delta and credentials.get("function_calling_type", "no_call") == "tool_call":
                        assistant_message_tool_calls = delta.get("tool_calls", None)
                    elif (
                            "function_call" in delta
                            and credentials.get("function_calling_type", "no_call") == "function_call"
                    ):
                        assistant_message_tool_calls = [
                            {"id": "tool_call_id", "type": "function", "function": delta.get("function_call", {})}
                        ]

                    # extract tool calls from response
                    if assistant_message_tool_calls:
                        tool_calls = self._extract_response_tool_calls(assistant_message_tool_calls)
                        _increase_tool_call(tool_calls, tools_calls)

                    if delta_content is None or delta_content == "":
                        continue

                    # transform assistant message to prompt message
                    assistant_prompt_message = AssistantPromptMessage(content=delta_content)
                    full_assistant_content += delta_content
                elif "text" in choice:
                    choice_text = choice.get("text", "")
                    if choice_text == "":
                        continue

                    # transform assistant message to prompt message
                    assistant_prompt_message = AssistantPromptMessage(content=choice_text)
                    full_assistant_content += choice_text
                else:
                    continue

                yield LLMResultChunk(
                    model=model,
                    delta=LLMResultChunkDelta(
                        index=chunk_index,
                        message=assistant_prompt_message,
                    ),
                )

            chunk_index += 1

        if tools_calls:
            yield LLMResultChunk(
                model=model,
                delta=LLMResultChunkDelta(
                    index=chunk_index,
                    message=AssistantPromptMessage(tool_calls=tools_calls, content=""),
                ),
            )

        yield self._create_final_llm_result_chunk(
            index=chunk_index,
            message=AssistantPromptMessage(content=""),
            finish_reason=finish_reason,
            usage=usage,
            model=model,
            credentials=credentials,
            prompt_messages=prompt_messages,
            full_content=full_assistant_content,
            callbacks=callbacks,
        )

    def _handle_generate_response(
            self,
            model: str,
            credentials: dict,
            response: requests.Response,
            prompt_messages: list[PromptMessage],
            callbacks: dict,
    ) -> LLMResult:
        """
        Handle non-streaming LLM response.
        """
        response_json: dict = response.json()

        mode = get_model_mode(credentials)
        completion_type = LLMMode.value_of(mode)

        output = response_json["choices"][0]
        message_id = response_json.get("id")

        response_content = ""
        tool_calls = None
        function_calling_type = credentials.get("function_calling_type", "no_call")

        if completion_type is LLMMode.CHAT:
            response_content = output.get("message", {})["content"]
            if function_calling_type == "tool_call":
                tool_calls = output.get("message", {}).get("tool_calls")
            elif function_calling_type == "function_call":
                tool_calls = output.get("message", {}).get("function_call")
        elif completion_type is LLMMode.COMPLETION:
            response_content = output["text"]

        assistant_message = AssistantPromptMessage(content=response_content, tool_calls=[])

        if tool_calls:
            if function_calling_type == "tool_call":
                assistant_message.tool_calls = self._extract_response_tool_calls(tool_calls)
            elif function_calling_type == "function_call":
                extracted = self._extract_response_function_call(tool_calls)
                if extracted:
                    assistant_message.tool_calls = [extracted]

        usage = response_json.get("usage")
        calc_response_usage = callbacks.get("calc_response_usage")

        if usage:
            # transform usage
            prompt_tokens = usage["prompt_tokens"]
            completion_tokens = usage["completion_tokens"]
        else:
            # calculate num tokens
            assert prompt_messages[0].content is not None
            prompt_tokens = self._num_tokens_from_string(prompt_messages[0].content, callbacks=callbacks)
            assert assistant_message.content is not None
            completion_tokens = self._num_tokens_from_string(assistant_message.content, callbacks=callbacks)

        # transform usage
        usage_obj = None
        if calc_response_usage:
            usage_obj = calc_response_usage(model, credentials, prompt_tokens, completion_tokens)

        # transform response
        result = LLMResult(
            id=message_id,
            model=response_json.get("model", model),
            message=assistant_message,
            usage=usage_obj,
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
        
        :param model: Model name
        :param credentials: Model credentials
        :param prompt_messages: Prompt messages
        :param tools: Tools for tool calling
        :param callbacks: Callback functions (get_num_tokens_by_gpt2)
        :return: Number of tokens
        """
        return self._num_tokens_from_messages(prompt_messages, tools, credentials, callbacks)

    def _convert_prompt_message_to_dict(self, message: PromptMessage, credentials: Optional[dict] = None) -> dict:
        """
        Convert PromptMessage to dict for OpenAI API format.
        """
        credentials = credentials or {}
        message_dict = {}

        if isinstance(message, UserPromptMessage):
            message = cast(UserPromptMessage, message)
            if isinstance(message.content, str):
                message_dict = {"role": "user", "content": message.content}
            else:
                sub_messages = []
                for message_content in message.content or []:
                    if message_content.type == PromptMessageContentType.TEXT:
                        message_content = cast(PromptMessageContent, message_content)
                        sub_message_dict = {
                            "type": "text",
                            "text": message_content.data,
                        }
                        sub_messages.append(sub_message_dict)
                    elif message_content.type == PromptMessageContentType.IMAGE:
                        message_content = cast(ImagePromptMessageContent, message_content)
                        sub_message_dict = {
                            "type": "image_url",
                            "image_url": {
                                "url": message_content.data,
                                "detail": message_content.detail.value,
                            },
                        }
                        sub_messages.append(sub_message_dict)
                    elif message_content.type == PromptMessageContentType.VIDEO:
                        message_content = cast(VideoPromptMessageContent, message_content)
                        if self._is_dashscope_model(credentials):
                            sub_messages.append(
                                {
                                    "type": "video_url",
                                    "video_url": {"url": message_content.data},
                                }
                            )
                        else:
                            sub_messages.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": message_content.data},
                                }
                            )
                    elif message_content.type == PromptMessageContentType.AUDIO:
                        message_content = cast(AudioPromptMessageContent, message_content)
                        if self._is_dashscope_model(credentials):
                            sub_messages.append(
                                {
                                    "type": "input_audio",
                                    "input_audio": {
                                        "data": message_content.data,
                                        "format": self._guess_audio_format(message_content.data),
                                    },
                                }
                            )
                        else:
                            sub_messages.append(
                                {
                                    "type": "image_url",
                                    "image_url": {"url": message_content.data},
                                }
                            )
                message_dict = {"role": "user", "content": sub_messages}
        elif isinstance(message, AssistantPromptMessage):
            message = cast(AssistantPromptMessage, message)
            message_dict = {"role": "assistant", "content": message.content}
            if message.tool_calls:
                function_calling_type = credentials.get("function_calling_type", "no_call")
                if function_calling_type == "tool_call":
                    message_dict["tool_calls"] = [tool_call.dict() for tool_call in message.tool_calls]
                elif function_calling_type == "function_call":
                    function_call = message.tool_calls[0]
                    message_dict["function_call"] = {
                        "name": function_call.function.name,
                        "arguments": function_call.function.arguments,
                    }
        elif isinstance(message, SystemPromptMessage):
            message = cast(SystemPromptMessage, message)
            message_dict = {"role": "system", "content": message.content}
        elif isinstance(message, ToolPromptMessage):
            message = cast(ToolPromptMessage, message)
            function_calling_type = credentials.get("function_calling_type", "no_call")
            if function_calling_type == "tool_call":
                message_dict = {
                    "role": "tool",
                    "content": message.content,
                    "tool_call_id": message.tool_call_id,
                }
            elif function_calling_type == "function_call":
                message_dict = {
                    "role": "function",
                    "content": message.content,
                    "name": message.tool_call_id,
                }
        else:
            raise ValueError(f"Got unknown type {message}")

        if message.name and message_dict.get("role", "") != "tool":
            message_dict["name"] = message.name

        return message_dict

    def _num_tokens_from_string(
            self,
            text: Union[str, list[PromptMessageContent]],
            tools: Optional[list[PromptMessageTool]] = None,
            callbacks: Optional[dict] = None,
    ) -> int:
        """
        Approximate num tokens for model with gpt2 tokenizer.
        
        :param text: prompt text
        :param tools: tools for tool calling
        :return: number of tokens
        """
        callbacks = callbacks or {}
        get_num_tokens_by_gpt2 = callbacks.get("get_num_tokens_by_gpt2")

        if isinstance(text, str):
            full_text = text
        else:
            full_text = ""
            for message_content in text:
                if message_content.type == PromptMessageContentType.TEXT:
                    message_content = cast(PromptMessageContent, message_content)
                    full_text += message_content.data

        num_tokens = 0
        if get_num_tokens_by_gpt2:
            num_tokens = get_num_tokens_by_gpt2(full_text)

        if tools:
            num_tokens += self._num_tokens_for_tools(tools, callbacks)

        return num_tokens

    def _num_tokens_from_messages(
            self,
            messages: list[PromptMessage],
            tools: Optional[list[PromptMessageTool]] = None,
            credentials: Optional[dict] = None,
            callbacks: Optional[dict] = None,
    ) -> int:
        """
        Approximate num tokens with GPT2 tokenizer.
        """
        callbacks = callbacks or {}
        get_num_tokens_by_gpt2 = callbacks.get("get_num_tokens_by_gpt2")

        tokens_per_message = 3
        tokens_per_name = 1

        num_tokens = 0
        messages_dict = [self._convert_prompt_message_to_dict(m, credentials) for m in messages]

        for message in messages_dict:
            num_tokens += tokens_per_message
            for key, value in message.items():
                # Cast str(value) in case the message value is not a string
                # This occurs with function messages
                # TODO: The current token calculation method for the image type is not implemented,
                #  which need to download the image and then get the resolution for calculation,
                #  and will increase the request delay
                if isinstance(value, list):
                    text = ""
                    for item in value:
                        if isinstance(item, dict) and item["type"] == "text":
                            text += item["text"]
                    value = text

                if key == "tool_calls":
                    for tool_call in value or []:
                        for t_key, t_value in tool_call.items():
                            if get_num_tokens_by_gpt2:
                                num_tokens += get_num_tokens_by_gpt2(t_key)
                            if t_key == "function":
                                for f_key, f_value in t_value.items():
                                    if get_num_tokens_by_gpt2:
                                        num_tokens += get_num_tokens_by_gpt2(f_key)
                                        num_tokens += get_num_tokens_by_gpt2(f_value)
                            else:
                                if get_num_tokens_by_gpt2:
                                    num_tokens += get_num_tokens_by_gpt2(t_key)
                                    num_tokens += get_num_tokens_by_gpt2(t_value)
                else:
                    if get_num_tokens_by_gpt2:
                        num_tokens += get_num_tokens_by_gpt2(str(value))

                if key == "name":
                    num_tokens += tokens_per_name

        # every reply is primed with <im_start>assistant
        num_tokens += 3

        if tools:
            num_tokens += self._num_tokens_for_tools(tools, callbacks)

        return num_tokens

    def _num_tokens_for_tools(
            self,
            tools: list[PromptMessageTool],
            callbacks: Optional[dict] = None,
    ) -> int:
        """
        Calculate num tokens for tool calling with tiktoken package.
        
        :param tools: tools for tool calling
        :return: number of tokens
        """
        callbacks = callbacks or {}
        get_num_tokens_by_gpt2 = callbacks.get("get_num_tokens_by_gpt2")

        if not get_num_tokens_by_gpt2:
            return 0

        num_tokens = 0
        for tool in tools:
            num_tokens += get_num_tokens_by_gpt2("type")
            num_tokens += get_num_tokens_by_gpt2("function")
            num_tokens += get_num_tokens_by_gpt2("function")

            # calculate num tokens for function object
            num_tokens += get_num_tokens_by_gpt2("name")
            if hasattr(tool, "name"):
                num_tokens += get_num_tokens_by_gpt2(tool.name)
            num_tokens += get_num_tokens_by_gpt2("description")
            if hasattr(tool, "description"):
                num_tokens += get_num_tokens_by_gpt2(tool.description)

            if hasattr(tool, "parameters"):
                parameters = tool.parameters
                num_tokens += get_num_tokens_by_gpt2("parameters")

                if "title" in parameters:
                    num_tokens += get_num_tokens_by_gpt2("title")
                    num_tokens += get_num_tokens_by_gpt2(parameters.get("title"))

                num_tokens += get_num_tokens_by_gpt2("type")
                num_tokens += get_num_tokens_by_gpt2(parameters.get("type"))

                if "properties" in parameters:
                    num_tokens += get_num_tokens_by_gpt2("properties")
                    for key, value in parameters.get("properties", {}).items():
                        num_tokens += get_num_tokens_by_gpt2(key)
                        for field_key, field_value in value.items():
                            num_tokens += get_num_tokens_by_gpt2(field_key)
                            if field_key == "enum":
                                for enum_field in field_value:
                                    num_tokens += 3
                                    num_tokens += get_num_tokens_by_gpt2(enum_field)
                            else:
                                num_tokens += get_num_tokens_by_gpt2(field_key)
                                num_tokens += get_num_tokens_by_gpt2(str(field_value))

                if "required" in parameters:
                    num_tokens += get_num_tokens_by_gpt2("required")
                    for required_field in parameters["required"]:
                        num_tokens += 3
                        num_tokens += get_num_tokens_by_gpt2(required_field)

        return num_tokens

    def _extract_response_tool_calls(self, response_tool_calls: list[dict]) -> list[AssistantPromptMessage.ToolCall]:
        """
        Extract tool calls from response.
        
        :param response_tool_calls: response tool calls
        :return: list of tool calls
        """
        tool_calls = []
        if response_tool_calls:
            for response_tool_call in response_tool_calls:
                if not response_tool_call.get("function"):
                    continue

                function = AssistantPromptMessage.ToolCall.ToolCallFunction(
                    name=response_tool_call.get("function", {}).get("name", ""),
                    arguments=response_tool_call.get("function", {}).get("arguments", ""),
                )
                tool_call = AssistantPromptMessage.ToolCall(
                    id=response_tool_call.get("id", ""),
                    type=response_tool_call.get("type", ""),
                    function=function,
                )
                tool_calls.append(tool_call)
        return tool_calls

    def _extract_response_function_call(self, response_function_call) -> Optional[AssistantPromptMessage.ToolCall]:
        """
        Extract function call from response.
        
        :param response_function_call: response function call
        :return: tool call
        """
        tool_call = None
        if response_function_call:
            function = AssistantPromptMessage.ToolCall.ToolCallFunction(
                name=response_function_call.get("name", ""),
                arguments=response_function_call.get("arguments", ""),
            )
            tool_call = AssistantPromptMessage.ToolCall(
                id=response_function_call.get("id", ""),
                type="function",
                function=function,
            )
        return tool_call

    def _wrap_thinking_by_reasoning_content(
            self,
            delta: dict,
            is_reasoning_started: bool
    ) -> tuple[str, bool]:
        """
        Wrap thinking content with <think> tags for reasoning models.
        
        :param delta: The delta object from the stream response
        :param is_reasoning_started: Whether reasoning has started
        :return: Tuple of (processed content, updated is_reasoning_started flag)
        """
        reasoning_content = delta.get("reasoning_content")
        content = delta.get("content", "")

        if reasoning_content:
            if not is_reasoning_started:
                # Start of reasoning
                return f"<think>{reasoning_content}", True
            else:
                # Continue reasoning
                return reasoning_content, True
        elif is_reasoning_started and content:
            # End of reasoning, start of actual content
            return f"</think>{content}", False
        else:
            return content or "", is_reasoning_started

    def _filter_thinking_result(self, result: LLMResult) -> LLMResult:
        """Filter thinking content from non-streaming result"""
        if result.message and result.message.content:
            content = result.message.content
            if isinstance(content, str) and content.startswith("<think>"):
                filtered_content = self._THINK_PATTERN.sub("", content, count=1)
                if filtered_content != content:
                    result.message.content = filtered_content
        return result

    def _filter_thinking_stream(self, stream: Generator) -> Generator:
        """Filter thinking content from streaming result"""
        buffer = ""
        in_thinking = False
        thinking_started = False

        for chunk in stream:
            if chunk.delta and chunk.delta.message and chunk.delta.message.content:
                content = chunk.delta.message.content
                buffer += content

                # Detect start of thinking block
                if not thinking_started and buffer.startswith("<think>"):
                    in_thinking = True
                    thinking_started = True
                    # Don't continue here - check for end tag in same iteration

                # Detect end of thinking block
                if in_thinking and "</think>" in buffer:
                    # Find the end of thinking block
                    end_idx = buffer.find("</think>") + len("</think>")
                    # Skip whitespace after </think>
                    while end_idx < len(buffer) and buffer[end_idx].isspace():
                        end_idx += 1
                    # Remove thinking block and continue with remaining content
                    buffer = buffer[end_idx:]
                    in_thinking = False
                    thinking_started = False
                    # Yield remaining content if any
                    if buffer:
                        chunk.delta.message.content = buffer
                        buffer = ""
                        yield chunk
                    continue

                # If not in thinking block, yield content
                if not in_thinking:
                    yield chunk
                    buffer = ""
            else:
                # Yield chunks without content as-is
                yield chunk


# Singleton instance for easy access
openai_compatible_protocol = OpenAICompatibleProtocol()
