import time
from typing import Any, Callable, Coroutine, Mapping

from aidial_sdk.chat_completion import Response

from aidial_adapter_vertexai.deployments import ChatCompletionDeployment as D
from aidial_adapter_vertexai.utils.log_config import app_logger as log

_DIAL_CACHE_BREAKPOINT_PATH = "X-DIAL-CACHE-BREAKPOINT-PATH"
_DIAL_CACHE_EXPIRE_AT = "X-DIAL-CACHE-EXPIRE-AT"

# There is no clear indication for how long implicitly created caches are persisted:
# https://ai.google.dev/gemini-api/docs/caching?lang=python#implicit-caching
_DEFAULT_TTL_SEC = 5 * 60  # 5 minutes


def _get_prompt_tokens_threshold(deployment: D) -> int | None:
    """
    https://ai.google.dev/gemini-api/docs/caching?lang=python#implicit-caching

    1. Implicit caching is enabled by default for all Gemini 2.5 models.
    2. The minimum input token count for context caching is 1,024 for 2.5 Flash and 4,096 for 2.5 Pro.
    """
    value: str = deployment.value
    if "gemini-2.5-flash" in value:
        return 1_024

    if "gemini-2.5-pro" in value:
        return 4_096

    return None


def _get_last_message_idx(request_body: Any) -> int | None:
    if not isinstance(request_body, dict):
        return None

    messages = request_body.get("messages") or []
    if not isinstance(messages, list):
        return None

    if not messages:
        return None

    return len(messages) - 1


async def set_response_headers_for_caching(
    response: Response,
    *,
    deployment: D,
    request_headers: Mapping[str, str],
    request_body: Any,
    get_request_tokens: Callable[[], Coroutine[None, None, int]],
) -> None:
    # DIAL Core always sends this header if the deployment
    # is marked in listing as supporting auto-caching
    if request_headers.get(_DIAL_CACHE_BREAKPOINT_PATH) is None:
        return

    if (threshold := _get_prompt_tokens_threshold(deployment)) is None:
        return

    if (last_message_idx := _get_last_message_idx(request_body)) is None:
        return

    try:
        prompt_tokens = await get_request_tokens()
        if prompt_tokens < threshold:
            return
    except Exception:
        log.exception("Unable to compute prompt tokens")
        return

    path = f"prefix.body.messages[{last_message_idx}]"
    expire_at = str(int(time.time()) + _DEFAULT_TTL_SEC)

    response.append_header(_DIAL_CACHE_BREAKPOINT_PATH, path)
    response.append_header(_DIAL_CACHE_EXPIRE_AT, expire_at)
