from __future__ import annotations

import dataclasses
import json
import os

from aidial_adapter_vertexai.deployments import ChatCompletionDeployment as D
from tests.conftest import get_extra_headers
from tests.utils.openai import sanitize_test_name


@dataclasses.dataclass
class compat:
    upstream: str
    deployment: D

    @property
    def headers(self) -> dict[str, str]:
        return {
            "x-upstream-extra-data": json.dumps(
                {"compatible_model_id": self.deployment.value}
            )
        }


@dataclasses.dataclass
class supported:
    deployment: D

    @property
    def upstream(self) -> str:
        return self.deployment.value

    @property
    def headers(self) -> dict[str, str]:
        return {}


@dataclasses.dataclass
class vertexai:
    region: str

    @property
    def headers(self) -> dict[str, str]:
        return get_extra_headers(self.region)


@dataclasses.dataclass
class foundry:
    service_name: str

    @classmethod
    def create(cls) -> foundry | None:
        if name := os.getenv("INTEGRATION_TEST_AZURE_FOUNDRY_SERVICE_NAME"):
            return cls(service_name=name)
        return None

    @property
    def headers(self) -> dict[str, str]:
        return {
            "x-upstream-endpoint": f"https://{self.service_name}.services.ai.azure.com/anthropic/v1/messages"
        }


@dataclasses.dataclass
class DeploymentSpec:
    model: supported | compat
    source: vertexai | foundry

    @property
    def deployment(self) -> D:
        return self.model.deployment

    @property
    def upstream(self) -> str:
        return self.model.upstream

    @property
    def headers(self) -> dict[str, str]:
        return self.model.headers | self.source.headers

    def display(self) -> str:
        ret = ""
        if isinstance(self.model, compat):
            ret += f"{sanitize_test_name(self.model.upstream)}-compat"
        else:
            ret += sanitize_test_name(self.model.deployment.value)
        ret += "/"
        if isinstance(self.source, vertexai):
            ret += self.source.region
        else:
            ret += "foundry"
        return ret

    @classmethod
    def supported_vertexai(cls, deployment: D, region: str) -> DeploymentSpec:
        return cls(supported(deployment), vertexai(region))

    @classmethod
    def compat_foundry(
        cls, upstream: str, deployment: D
    ) -> DeploymentSpec | None:
        if conf := foundry.create():
            source = compat(upstream, deployment)
            return cls(source, conf)
        return None
