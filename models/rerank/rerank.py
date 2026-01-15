"""
Higress AI Gateway Rerank implementation.
"""

import logging
from typing import Optional

from dify_plugin.entities import I18nObject
from dify_plugin.entities.model import AIModelEntity, FetchFrom, ModelType
from dify_plugin.entities.model.rerank import RerankResult
from dify_plugin.errors.model import CredentialsValidateFailedError
from dify_plugin.interfaces.model.rerank_model import RerankModel

from models._common import _CommonHigress
from models.rerank.utils import get_rerank_model_protocol
from models.rerank.protocols import DashScopeRerankProtocol

logger = logging.getLogger(__name__)


class HigressRerankModel(_CommonHigress, RerankModel):
    """
    Model class for Higress AI Gateway rerank model.
    Supports Alibaba Cloud DashScope rerank protocol.
    """

    # Protocol handlers registry
    _protocol_handlers = {
        "dashscope_rerank": DashScopeRerankProtocol(),
    }

    def _get_protocol_handler(self, credentials: dict):
        """
        Get the protocol handler based on credentials.

        :param credentials: Model credentials
        :return: Protocol handler instance
        :raises CredentialsValidateFailedError: If protocol is not supported
        """
        protocol = get_rerank_model_protocol(credentials)
        handler = self._protocol_handlers.get(protocol)

        if not handler:
            raise CredentialsValidateFailedError(
                f"Unsupported rerank model protocol: {protocol}"
            )

        return handler

    def _invoke(
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
        Invoke rerank model.

        :param model: model name
        :param credentials: model credentials
        :param query: search query
        :param docs: documents to rerank
        :param score_threshold: score threshold for filtering results
        :param top_n: top N results to return
        :param user: unique user id
        :return: rerank result
        """
        handler = self._get_protocol_handler(credentials)

        return handler.invoke(
            model=model,
            credentials=credentials,
            query=query,
            docs=docs,
            score_threshold=score_threshold,
            top_n=top_n,
            user=user,
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
            logger.info(f"Rerank credentials validated successfully for model: {model}")
        except CredentialsValidateFailedError:
            raise
        except Exception as e:
            logger.exception(f"Unexpected error in validate_credentials: {e}")
            raise

    def get_customizable_model_schema(self, model: str, credentials: dict) -> AIModelEntity:
        """
        Generate custom model entities from credentials.

        :param model: model name
        :param credentials: model credentials
        :return: AIModelEntity for the rerank model
        """

        entity = AIModelEntity(
            model=model,
            label=I18nObject(en_US=model),
            model_type=ModelType.RERANK,
            fetch_from=FetchFrom.CUSTOMIZABLE_MODEL,
            model_properties={},
        )

        # Set display name if provided
        if "display_name" in credentials and credentials["display_name"] != "":
            entity.label = I18nObject(
                en_US=credentials["display_name"],
                zh_Hans=credentials["display_name"]
            )

        return entity
