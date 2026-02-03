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

            # tools support, default no_call
            function_calling_type = credentials.get("function_calling_type", "no_call")
            if function_calling_type == "function_call":
                features.append(ModelFeature.TOOL_CALL)
            elif function_calling_type == "tool_call":
                features.append(ModelFeature.MULTI_TOOL_CALL)
            # tools stream, default supported
            stream_function_calling = credentials.get("stream_function_calling", "supported")
            if stream_function_calling == "supported":
                features.append(ModelFeature.STREAM_TOOL_CALL)
            # vision support
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
                model_properties={  # model_properties will be shown on model list
                    ModelPropertyKey.MODE: mode,
                },
                parameter_rules=[
                    ParameterRule(
                        name=DefaultParameterName.TEMPERATURE.value,
                        use_template=DefaultParameterName.TEMPERATURE.value,
                        label=I18nObject(en_US="Temperature", zh_Hans="温度"),
                        help=I18nObject(
                            en_US="Used to control the degree of randomness and diversity. Specifically, the temperature value controls the degree to which the probability distribution of each candidate word is smoothed when generating text. A higher temperature value will reduce the peak value of the probability distribution, allowing more low-probability words to be selected, and the generated results will be more diverse; while a lower temperature value will enhance the peak value of the probability distribution, making it easier for high-probability words to be selected, the generated results are more certain.",
                            zh_Hans="用于控制随机性和多样性的程度。具体来说，temperature值控制了生成文本时对每个候选词的概率分布进行平滑的程度。较高的temperature值会降低概率分布的峰值，使得更多的低概率词被选择，生成结果更加多样化；而较低的temperature值则会增强概率分布的峰值，使得高概率词更容易被选择，生成结果更加确定。",
                        ),
                        type=ParameterType.FLOAT,
                        default=float(credentials.get("temperature", 0.7)),
                        min=0.0,
                        max=2.0,
                        precision=2,
                    ),
                    ParameterRule(
                        name=DefaultParameterName.TOP_P.value,
                        use_template=DefaultParameterName.TOP_P.value,
                        label=I18nObject(en_US="Top P", zh_Hans="Top P"),
                        help=I18nObject(
                            en_US="The probability threshold of the kernel sampling method during the generation process. For example, when the value is 0.8, only the smallest set of the most likely tokens with a sum of probabilities greater than or equal to 0.8 is retained as the candidate set. The value range is (0,1.0). The larger the value, the higher the randomness generated; the lower the value, the higher the certainty generated.",
                            zh_Hans="生成过程中核采样方法概率阈值，例如，取值为0.8时，仅保留概率加起来大于等于0.8的最可能token的最小集合作为候选集。取值范围为（0,1.0)，取值越大，生成的随机性越高；取值越低，生成的确定性越高。",
                        ),
                        type=ParameterType.FLOAT,
                        default=float(credentials.get("top_p", 0.8)),
                        min=0.1,
                        max=0.9,
                        precision=2,
                    ),
                    ParameterRule(
                        name=DefaultParameterName.TOP_K.value,
                        use_template=DefaultParameterName.TOP_K.value,
                        label=I18nObject(en_US="Top k", zh_Hans="取样数量"),
                        help=I18nObject(
                            en_US="The size of the sample candidate set when generated. For example, when the value is 50, only the 50 highest-scoring tokens in a single generation form a randomly sampled candidate set. The larger the value, the higher the randomness generated; the smaller the value, the higher the certainty generated.",
                            zh_Hans="生成时，采样候选集的大小。例如，取值为50时，仅将单次生成中得分最高的50个token组成随机采样的候选集。取值越大，生成的随机性越高；取值越小，生成的确定性越高。",
                        ),
                        type=ParameterType.INT,
                        default=int(credentials.get("top_k", 50)),
                        min=0,
                        max=99,
                    ),
                    ParameterRule(
                        name="seed",
                        label=I18nObject(en_US="Random seed", zh_Hans="随机种子"),
                        help=I18nObject(
                            en_US="The random number seed used when generating, the user controls the randomness of the content generated by the model. Supports unsigned 64-bit integers, default value is 1234. When using seed, the model will try its best to generate the same or similar results, but there is currently no guarantee that the results will be exactly the same every time.",
                            zh_Hans="生成时使用的随机数种子，用户控制模型生成内容的随机性。支持无符号64位整数，默认值为 1234。在使用seed时，模型将尽可能生成相同或相似的结果，但目前不保证每次生成的结果完全相同。",
                        ),
                        type=ParameterType.INT,
                        default=int(credentials.get("seed", 1234)),
                        required=False,
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
