from __future__ import annotations

import json
import re
from typing import Protocol

from aidial_sdk.deployment.from_request_mixin import FromRequestDeploymentMixin
from anthropic import (
    AsyncAnthropic,
    AsyncAnthropicFoundry,
    AsyncAnthropicVertex,
)
from google.genai.client import Client as GenAIClient
from pydantic import BaseModel

from aidial_adapter_vertexai.app_config import (
    DEFAULT_PROJECT,
    DEFAULT_PROJECT_ENV_VAR,
    DEFAULT_REGION,
    DEFAULT_REGION_ENV_VAR,
    get_anthropic_foundry_client,
    get_anthropic_vertex_client,
    get_genai_client,
)
from aidial_adapter_vertexai.utils.log_config import app_logger as log


class UpstreamConfig(Protocol):
    async def get_genai_client(self) -> GenAIClient: ...

    async def get_anthropic_client(
        self,
    ) -> AsyncAnthropicVertex | AsyncAnthropic | AsyncAnthropicFoundry: ...


def parse_upstream_config(
    request: FromRequestDeploymentMixin,
) -> UpstreamConfig:
    if (conf := _AzureFoundryUpstreamConfig.from_request(request)) is not None:
        log.debug("accessing deployment via Azure Foundry")
        return conf

    if (conf := _ApiKeyUpstreamConfig.from_request(request)) is not None:
        log.debug("accessing deployment via Platform API-key")
        return conf

    log.debug("accessing deployment via Google Cloud creds")
    return _CloudUpstreamConfig.from_request(request)


_UPSTREAM_CONFIG_HEADER_NAME = "x-upstream-extra-data"
_UPSTREAM_API_KEY_HEADER_NAME = "x-upstream-key"
_UPSTREAM_ENDPOINT_HEADER_NAME = "x-upstream-endpoint"


class _AzureFoundryUpstreamConfig(BaseModel):
    api_key: str | None

    base_url: str

    @classmethod
    def from_request(
        cls, request: FromRequestDeploymentMixin
    ) -> _AzureFoundryUpstreamConfig | None:
        api_key = request.headers.get(_UPSTREAM_API_KEY_HEADER_NAME)
        endpoint = request.headers.get(_UPSTREAM_ENDPOINT_HEADER_NAME)
        if (
            endpoint is None
            or (m := re.match(r"(.*/anthropic)/v1/messages", endpoint)) is None
        ):
            return None

        base_url = m.group(1)

        return cls(api_key=api_key, base_url=base_url)

    async def get_genai_client(self) -> GenAIClient:
        raise NotImplementedError(
            "Azure Foundry doesn't provide Google GenAI client"
        )

    async def get_anthropic_client(self) -> AsyncAnthropicFoundry:
        return await get_anthropic_foundry_client(self.api_key, self.base_url)


class _ApiKeyUpstreamConfig(BaseModel):
    api_key: str

    @classmethod
    def from_request(
        cls, request: FromRequestDeploymentMixin
    ) -> _ApiKeyUpstreamConfig | None:
        key = request.headers.get(_UPSTREAM_API_KEY_HEADER_NAME)
        return None if key is None else cls(api_key=key)

    async def get_genai_client(self) -> GenAIClient:
        return GenAIClient(api_key=self.api_key)

    async def get_anthropic_client(self) -> AsyncAnthropic:
        return AsyncAnthropic(api_key=self.api_key)


class _CloudUpstreamConfig(BaseModel):
    region: str
    project: str

    @classmethod
    def from_request(
        cls, request: FromRequestDeploymentMixin
    ) -> UpstreamConfig:
        conf = request.headers.get(_UPSTREAM_CONFIG_HEADER_NAME)
        try:
            conf = json.loads(conf or "{}")
        except Exception:
            raise ValueError(
                f"Header {_UPSTREAM_CONFIG_HEADER_NAME!r} isn't valid JSON"
            )

        if not isinstance(conf, dict):
            raise ValueError(
                f"Header {_UPSTREAM_CONFIG_HEADER_NAME!r} isn't valid JSON dictionary"
            )

        conf["region"] = conf.get("region") or DEFAULT_REGION
        conf["project"] = conf.get("project") or DEFAULT_PROJECT

        if not conf["region"]:
            raise ValueError(
                f"Region isn't specified neither in {_UPSTREAM_CONFIG_HEADER_NAME!r} header "
                f"nor in {DEFAULT_REGION_ENV_VAR!r} variable"
            )

        if not conf["project"]:
            raise ValueError(
                f"Project isn't specified neither in {_UPSTREAM_CONFIG_HEADER_NAME!r} header "
                f"nor in {DEFAULT_PROJECT_ENV_VAR!r} variable"
            )

        return cls.model_validate(conf)

    async def get_genai_client(self) -> GenAIClient:
        return await get_genai_client(self.project, self.region)

    async def get_anthropic_client(self) -> AsyncAnthropicVertex:
        return await get_anthropic_vertex_client(self.project, self.region)
