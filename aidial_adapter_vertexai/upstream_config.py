from __future__ import annotations

import json
import re
from typing import Protocol, TypeAlias

import pydantic
from aidial_sdk.deployment.from_request_mixin import FromRequestDeploymentMixin
from anthropic import (
    AsyncAnthropic,
    AsyncAnthropicFoundry,
    AsyncAnthropicVertex,
)
from fastapi import Request
from google.genai.client import Client as GenAIClient
from mistralai.client import Mistral
from mistralai.gcp.client import MistralGCP
from pydantic import BaseModel

from aidial_adapter_vertexai.app_config import (
    DEFAULT_PROJECT_ENV_VAR,
    DEFAULT_REGION_ENV_VAR,
    get_anthropic_foundry_client,
    get_anthropic_vertex_client,
    get_default_project,
    get_default_region,
    get_genai_client,
    get_mistral_gcp_client,
)
from aidial_adapter_vertexai.utils.log_config import app_logger as log

MistralClient: TypeAlias = Mistral | MistralGCP

AnthropicClient = AsyncAnthropicVertex | AsyncAnthropic | AsyncAnthropicFoundry


class UpstreamConfig(Protocol):
    async def get_genai_client(self) -> GenAIClient: ...
    async def get_anthropic_client(self) -> AnthropicClient: ...
    async def get_mistral_client(self) -> MistralClient: ...


def parse_upstream_config(request: Request) -> UpstreamConfig:
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
        cls, request: Request
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

    async def get_mistral_client(self) -> MistralClient:
        raise NotImplementedError(
            "Azure Foundry Mistral AI client is not supported."
        )


class _ApiKeyUpstreamConfig(BaseModel):
    api_key: str

    @classmethod
    def from_request(cls, request: Request) -> _ApiKeyUpstreamConfig | None:
        key = request.headers.get(_UPSTREAM_API_KEY_HEADER_NAME)
        return None if key is None else cls(api_key=key)

    async def get_genai_client(self) -> GenAIClient:
        return GenAIClient(api_key=self.api_key)

    async def get_anthropic_client(self) -> AsyncAnthropic:
        return AsyncAnthropic(api_key=self.api_key)

    async def get_mistral_client(self) -> Mistral:
        return Mistral(api_key=self.api_key)


class _CloudUpstreamConfig(BaseModel):
    region: str
    project: str

    @classmethod
    def from_request(cls, request: Request) -> UpstreamConfig:
        conf = request.headers.get(_UPSTREAM_CONFIG_HEADER_NAME)
        try:
            conf = json.loads(conf or "{}")
        except Exception:
            raise ValueError(
                f"Header {_UPSTREAM_CONFIG_HEADER_NAME!r} isn't valid JSON"
            ) from None

        if not isinstance(conf, dict):
            raise ValueError(
                f"Header {_UPSTREAM_CONFIG_HEADER_NAME!r} isn't valid JSON dictionary"
            )

        conf["region"] = conf.get("region") or get_default_region()
        conf["project"] = conf.get("project") or get_default_project()

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

    async def get_mistral_client(self) -> MistralGCP:
        return await get_mistral_gcp_client(self.project, self.region)


class CompatibleModelUpstreamConfig(BaseModel):
    compatible_model_id: str | None = None


def get_compatible_model_id(request: FromRequestDeploymentMixin) -> str | None:
    if (
        extra := request.original_request.headers.get(
            _UPSTREAM_CONFIG_HEADER_NAME
        )
    ) is None:
        return None

    try:
        conf = CompatibleModelUpstreamConfig.model_validate_json(extra)
    except pydantic.ValidationError as e:
        log.error(
            f"Request header {_UPSTREAM_CONFIG_HEADER_NAME!r} doesn't contain"
            f" valid override name configuration: {e}"
        )
        return None

    return conf.compatible_model_id
