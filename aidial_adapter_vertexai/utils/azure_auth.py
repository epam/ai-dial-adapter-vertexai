import os
import time
from typing import Awaitable, Callable

from aidial_sdk.exceptions import HTTPException as DialException
from azure.core.credentials import AccessToken
from azure.core.exceptions import ClientAuthenticationError
from azure.identity.aio import DefaultAzureCredential

from aidial_adapter_vertexai.utils.log_config import app_logger as logger

EXPIRATION_WINDOW_IN_SEC: int = int(
    os.getenv("ACCESS_TOKEN_EXPIRATION_WINDOW", 10)
)
AZURE_OPEN_AI_SCOPE: str = os.getenv(
    "AZURE_OPEN_AI_SCOPE", "https://cognitiveservices.azure.com/.default"
)


def _get_azure_access_token() -> Callable[[], Awaitable[str]]:
    default_credential = DefaultAzureCredential()
    access_token: AccessToken | None = None

    async def _getter() -> str:
        now = int(time.time())
        nonlocal access_token

        if (
            access_token is None
            or now + EXPIRATION_WINDOW_IN_SEC > access_token.expires_on
        ):
            try:
                access_token = await default_credential.get_token(
                    AZURE_OPEN_AI_SCOPE
                )
                logger.debug(
                    f"Obtained new Azure access token, expires on {access_token.expires_on}"
                )
            except ClientAuthenticationError as e:
                logger.error(
                    f"Default Azure credential failed with the error: {e.message}"
                )
                raise DialException(
                    "Authentication failed", 401, "Unauthorized"
                )

        return access_token.token

    return _getter


get_azure_access_token = _get_azure_access_token()
