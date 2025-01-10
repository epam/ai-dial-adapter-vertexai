# TODO: COMMON
from typing import Any, AsyncIterator, Callable

from aidial_adapter_vertexai.utils.log_config import vertex_ai_logger as log


class FinishReasonOtherError(Exception):
    def __init__(self, msg: str, retriable: bool):
        self.msg = msg
        self.retriable = retriable
        super().__init__(self.msg)


async def generate_with_retries(
    generator: Callable[[], AsyncIterator[Any]], max_retries: int
) -> AsyncIterator[Any]:
    retries = 0
    while True:
        try:
            async for content in generator():
                yield content
            break

        except FinishReasonOtherError as e:
            if not e.retriable:
                raise e

            retries += 1
            if retries > max_retries:
                log.debug(f"max retries exceeded ({max_retries})")
                raise e

            log.debug(f"retrying [{retries}/{max_retries}]")
