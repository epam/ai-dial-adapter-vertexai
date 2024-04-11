import logging
import os
import re
import sys
from logging import Filter, LogRecord

from aidial_sdk import logger as aidial_logger
from uvicorn.logging import DefaultFormatter

# By default (in prod) we don't want to print debug messages,
# because they typically contain prompts.
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

AIDIAL_LOG_LEVEL = os.getenv("AIDIAL_LOG_LEVEL", "WARNING")
aidial_logger.setLevel(AIDIAL_LOG_LEVEL)


class HealthCheckFilter(Filter):
    def filter(self, record: LogRecord):
        return not re.search(r"(\s+)/health(\s+)", record.getMessage())


def configure_loggers():
    # Making the uvicorn logger delegate logging to the root logger
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.handlers = []
    uvicorn_logger.setLevel(LOG_LEVEL)
    uvicorn_logger.propagate = True

    # Filter out health check requests from uvicorn logs
    logging.getLogger("uvicorn.access").addFilter(HealthCheckFilter())

    # Setting up log levels
    for name in ["app", "vertex-ai", "uvicorn", "__main__"]:
        logging.getLogger(name).setLevel(LOG_LEVEL)

    # Configuring the root logger
    root = logging.getLogger()

    root_has_stderr_handler = any(
        isinstance(handler, logging.StreamHandler)
        and handler.stream == sys.stderr
        for handler in root.handlers
    )

    if not root_has_stderr_handler:
        formatter = DefaultFormatter(
            fmt="%(levelprefix)s | %(asctime)s | %(process)d | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
            use_colors=True,
        )

        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        root.addHandler(handler)


# Loggers in order from high-level to low-level
# High-level logs from the adapter server
app_logger = logging.getLogger("app")

# LLM requests and responses
vertex_ai_logger = logging.getLogger("vertex-ai")
