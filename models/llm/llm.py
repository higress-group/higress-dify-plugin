"""
Higress AI Gateway LLM implementation.
"""

import logging
from collections.abc import Generator
from decimal import Decimal
from typing import Optional, Union

from dify_plugin.entities import I18nObject
from dify_plugin.entities.model import (
    AIModelEntity,
    DefaultParameterName,
    FetchFrom,
    ModelFeature,
    ModelPropertyKey,
    ModelType,
    ParameterRule,
    ParameterType,
    PriceConfig,
)
from dify_plugin.entities.model.llm import LLMMode, LLMResult
from dify_plugin.entities.model.message import PromptMessage, PromptMessageTool
from dify_plugin.errors.model import CredentialsValidateFailedError
from dify_plugin.interfaces.model.large_language_model import LargeLanguageModel

from models._common import _CommonHigress
from models.llm.utils import get_model_protocol
from models.llm.protocols import OpenAICompatibleProtocol, DashScopeImageGenerationProtocol

logger = logging.getLogger(__name__)


class HigressLargeLanguageModel(_CommonHigress, LargeLanguageModel):
    """
    Model class for Higress AI Gateway large language model.
    Supports multiple protocols with OpenAI-compatible as the default.
    """

    # Protocol handlers registry
    _protocol_handlers = {
        "openai_compatible": OpenAICompatibleProtocol(),
        "dashscope_image_generation": DashScopeImageGenerationProtocol(),
    }

    def _get_protocol_handler(self, credentials: dict):
        """
        Get the protocol handler based on credentials.
        
        :param credentials: Model credentials
        :return: Protocol handler instance
        :raises CredentialsValidateFailedError: If protocol is not supported
        """
        protocol = get_model_protocol(credentials)
        handler = self._protocol_handlers.get(protocol)
        
        if not handler:
            raise CredentialsValidateFailedError(
                f"Unsupported model protocol: {protocol}"
            )
        
        return handler

    def _get_callbacks(self) -> dict:
        """
        Get callback functions to pass to protocol handlers.
        
        :return: Dictionary of callback functions
        """
        return {
            "calc_response_usage": self._calc_response_usage,
            "get_num_tokens_by_gpt2": self._get_num_tokens_by_gpt2,
        }

    def _invoke(
            self,
            model: str,
            credentials: dict,
            prompt_messages: list[PromptMessage],
            model_parameters: dict,
            tools: Optional[list[PromptMessageTool]] = None,
            stop: Optional[list[str]] = None,
            stream: bool = True,
            user: Optional[str] = None,
    ) -> Union[LLMResult, Generator]:
        """
        Invoke large language model.

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param model_parameters: model parameters
        :param tools: tools for tool calling
        :param stop: stop words
        :param stream: is stream response
        :param user: unique user id
        :return: full response or stream response chunk generator result
        """
        return self._generate(
            model=model,
            credentials=credentials,
            prompt_messages=prompt_messages,
            model_parameters=model_parameters,
            tools=tools,
            stop=stop,
            stream=stream,
            user=user,
        )

    def get_num_tokens(
            self,
            model: str,
            credentials: dict,
            prompt_messages: list[PromptMessage],
            tools: Optional[list[PromptMessageTool]] = None,
    ) -> int:
        """
        Get number of tokens for given prompt messages.

        :param model: model name
        :param credentials: model credentials
        :param prompt_messages: prompt messages
        :param tools: tools for tool calling
        :return: number of tokens
        """
        handler = self._get_protocol_handler(credentials)
        callbacks = self._get_callbacks()
        
        return handler.get_num_tokens(
            model=model,
            credentials=credentials,
            prompt_messages=prompt_messages,
            tools=tools,
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
            logger.info(f"LLM credentials validated successfully for model: {model}")
        except CredentialsValidateFailedError:
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in validate_credentials: {e}")
            raise

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        """
        generate custom model entities from credentials
        """
        try:
            features = []

            function_calling_type = credentials.get("function_calling_type", "no_call")
            if function_calling_type == "function_call":
                features.append(ModelFeature.TOOL_CALL)
            elif function_calling_type == "tool_call":
                features.append(ModelFeature.MULTI_TOOL_CALL)

            stream_function_calling = credentials.get("stream_function_calling", "supported")
            if stream_function_calling == "supported":
                features.append(ModelFeature.STREAM_TOOL_CALL)

            vision_support = credentials.get("vision_support", "not_support")
            if vision_support == "support":
                features.append(ModelFeature.VISION)

            # Get mode with safe default
            mode = credentials.get("mode", "chat")
            if not mode:
                mode = "chat"

            entity = AIModelEntity(
            model=model,
            label=I18nObject(en_US=model),
            model_type=ModelType.LLM,
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            features=features,
            model_properties={
                ModelPropertyKey.CONTEXT_SIZE: int(credentials.get("context_size", "4096")),
                ModelPropertyKey.MODE: mode,
            },
            parameter_rules=[
                ParameterRule(
                    name=DefaultParameterName.TEMPERATURE.value,
                    label=I18nObject(en_US="Temperature", zh_Hans="温度"),
                    help=I18nObject(
                        en_US="Kernel sampling threshold. Used to determine the randomness of the results."
                        "The higher the value, the stronger the randomness."
                        "The higher the possibility of getting different answers to the same question.",
                        zh_Hans="核采样阈值。用于决定结果随机性，取值越高随机性越强即相同的问题得到的不同答案的可能性越高。",
                    ),
                    type=ParameterType.FLOAT,
                    default=float(credentials.get("temperature", 0.7)),
                    min=0,
                    max=2,
                    precision=2,
                ),
                ParameterRule(
                    name=DefaultParameterName.TOP_P.value,
                    label=I18nObject(en_US="Top P", zh_Hans="Top P"),
                    help=I18nObject(
                        en_US="The probability threshold of the nucleus sampling method during the generation process."
                        "The larger the value is, the higher the randomness of generation will be."
                        "The smaller the value is, the higher the certainty of generation will be.",
                        zh_Hans="生成过程中核采样方法概率阈值。取值越大，生成的随机性越高；取值越小，生成的确定性越高。",
                    ),
                    type=ParameterType.FLOAT,
                    default=float(credentials.get("top_p", 1)),
                    min=0,
                    max=1,
                    precision=2,
                ),
                ParameterRule(
                    name=DefaultParameterName.FREQUENCY_PENALTY.value,
                    label=I18nObject(en_US="Frequency Penalty", zh_Hans="频率惩罚"),
                    help=I18nObject(
                        en_US="For controlling the repetition rate of words used by the model."
                        "Increasing this can reduce the repetition of the same words in the model's output.",
                        zh_Hans="用于控制模型已使用字词的重复率。 提高此项可以降低模型在输出中重复相同字词的重复度。",
                    ),
                    type=ParameterType.FLOAT,
                    default=float(credentials.get("frequency_penalty", 0)),
                    min=-2,
                    max=2,
                ),
                ParameterRule(
                    name=DefaultParameterName.PRESENCE_PENALTY.value,
                    label=I18nObject(en_US="Presence Penalty", zh_Hans="存在惩罚"),
                    help=I18nObject(
                        en_US="Used to control the repetition rate when generating models."
                        "Increasing this can reduce the repetition rate of model generation.",
                        zh_Hans="用于控制模型生成时的重复度。提高此项可以降低模型生成的重复度。",
                    ),
                    type=ParameterType.FLOAT,
                    default=float(credentials.get("presence_penalty", 0)),
                    min=-2,
                    max=2,
                ),
                ParameterRule(
                    name=DefaultParameterName.MAX_TOKENS.value,
                    label=I18nObject(en_US="Max Tokens", zh_Hans="最大标记"),
                    help=I18nObject(
                        en_US="Maximum length of tokens for the model response.",
                        zh_Hans="模型回答的tokens的最大长度。",
                    ),
                    type=ParameterType.INT,
                    default=512,
                    min=1,
                    max=int(credentials.get("max_tokens_to_sample", 4096)),
                ),
            ],
            pricing=PriceConfig(
                input=Decimal(credentials.get("input_price", 0)),
                output=Decimal(credentials.get("output_price", 0)),
                unit=Decimal(credentials.get("unit", 0)),
                currency=credentials.get("currency", "USD"),
            ),
            )

            if mode == "chat":
                entity.model_properties[ModelPropertyKey.MODE] = LLMMode.CHAT.value
            elif mode == "completion":
                entity.model_properties[ModelPropertyKey.MODE] = LLMMode.COMPLETION.value
            else:
                raise ValueError(f"Unknown completion mode: {mode}")

            structured_output_support = credentials.get("structured_output_support", "not_supported")
            if structured_output_support == "supported":
                # ----
                # The following section should be added after the new version of `dify-plugin-sdks`
                # is released.
                # Related Commit:
                # https://github.com/langgenius/dify-plugin-sdks/commit/0690573a879caf43f92494bf411f45a1835d96f6
                # ----
                # try:
                #     entity.features.index(ModelFeature.STRUCTURED_OUTPUT)
                # except ValueError:
                #     entity.features.append(ModelFeature.STRUCTURED_OUTPUT)

                entity.parameter_rules.append(
                    ParameterRule(
                        name=DefaultParameterName.RESPONSE_FORMAT.value,
                        label=I18nObject(en_US="Response Format", zh_Hans="回复格式"),
                        help=I18nObject(
                            en_US="Specifying the format that the model must output.",
                            zh_Hans="指定模型必须输出的回复格式。",
                        ),
                        type=ParameterType.STRING,
                        options=["text", "json_object", "json_schema"],
                        required=False,
                    )
                )
                entity.parameter_rules.append(
                    ParameterRule(
                        name="reasoning_format",
                        label=I18nObject(en_US="Reasoning Format", zh_Hans="推理格式"),
                        help=I18nObject(
                            en_US="Specifying the format that the model must output reasoning.",
                            zh_Hans="指定模型必须输出的推理格式。",
                        ),
                        type=ParameterType.STRING,
                        options=["none", "auto", "deepseek", "deepseek-legacy"],
                        required=False,
                    )
                )
                entity.parameter_rules.append(
                    ParameterRule(
                        name=DefaultParameterName.JSON_SCHEMA.value,
                        use_template=DefaultParameterName.JSON_SCHEMA.value,
                    )
                )

            if "display_name" in credentials and credentials["display_name"] != "":
                entity.label = I18nObject(
                    en_US=credentials["display_name"], zh_Hans=credentials["display_name"]
                )

            # Configure thinking mode parameter based on model support
            agent_thought_support = credentials.get("agent_thought_support", "not_supported")

            # Add AGENT_THOUGHT feature if thinking mode is supported (either mode)
            if hasattr(ModelFeature, 'AGENT_THOUGHT'):
                if agent_thought_support in ["supported",
                                             "only_thinking_supported"] and ModelFeature.AGENT_THOUGHT not in entity.features:
                    entity.features.append(ModelFeature.AGENT_THOUGHT)

            # Only add the enable_thinking parameter if the model supports both modes
            # If only_thinking_supported, the parameter is not needed (forced behavior)
            if agent_thought_support == "supported":
                entity.parameter_rules.append(
                    ParameterRule(
                        name="enable_thinking",
                        label=I18nObject(en_US="Thinking mode", zh_Hans="思考模式"),
                        help=I18nObject(
                            en_US="Whether to enable thinking mode, applicable to various thinking mode models deployed on reasoning frameworks such as vLLM and SGLang, for example Qwen3.",
                            zh_Hans="是否开启思考模式，适用于vLLM和SGLang等推理框架部署的多种思考模式模型，例如Qwen3。",
                        ),
                        type=ParameterType.BOOLEAN,
                        required=False,
                    )
                )

            return entity
        
        except Exception as e:
            logger.exception(f"Error in get_customizable_model_schema: {e}")
            raise

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
    ) -> Union[LLMResult, Generator]:
        """
        Invoke LLM completion model using the appropriate protocol handler.

        :param model: model name
        :param credentials: credentials
        :param prompt_messages: prompt messages
        :param model_parameters: model parameters
        :param stop: stop words
        :param stream: is stream response
        :param user: unique user id
        :return: full response or stream response chunk generator result
        """
        handler = self._get_protocol_handler(credentials)
        callbacks = self._get_callbacks()
        
        return handler.generate(
            model=model,
            credentials=credentials,
            prompt_messages=prompt_messages,
            model_parameters=model_parameters,
            tools=tools,
            stop=stop,
            stream=stream,
            user=user,
            callbacks=callbacks,
        )
