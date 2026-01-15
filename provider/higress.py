"""
Higress AI Gateway model provider.
"""

import logging
from collections.abc import Mapping

from dify_plugin import ModelProvider
from dify_plugin.errors.model import CredentialsValidateFailedError

logger = logging.getLogger(__name__)


class HigressModelProvider(ModelProvider):
    """
    Model provider for Higress AI Gateway.
    """

    def validate_provider_credentials(self, credentials: Mapping) -> None:
        """
        Validate provider credentials.
        
        Note: Higress uses model-level credentials, so provider-level
        validation is a no-op.

        :param credentials: provider credentials
        """
        try:
            # Higress uses model-level credentials validation
            pass
        except CredentialsValidateFailedError:
            raise
        except Exception as ex:
            logger.exception(
                f"{self.get_provider_schema().provider} credentials validate failed"
            )
            raise ex
