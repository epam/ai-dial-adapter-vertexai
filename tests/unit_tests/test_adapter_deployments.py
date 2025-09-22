from dataclasses import dataclass, field
from typing import Dict, List, Protocol

import pytest

from aidial_adapter_vertexai.adapter_deployments import AdapterDeployments
from aidial_adapter_vertexai.deployments import (
    ChatCompletionDeployment,
    EmbeddingsDeployment,
)


class Checker(Protocol):
    def check(self, deployments: AdapterDeployments): ...


@dataclass
class supported:
    deployment_id: ChatCompletionDeployment | EmbeddingsDeployment
    redirect: ChatCompletionDeployment | EmbeddingsDeployment | None = None

    def check(self, deployments: AdapterDeployments):
        deployment_name = self.deployment_id.value
        if isinstance(self.deployment_id, ChatCompletionDeployment):
            deployment = deployments.chat_completions.get(deployment_name)
        else:
            deployment = deployments.embeddings.get(deployment_name)

        assert deployment is not None
        assert deployment.adapter_deployment_id == deployment_name
        if self.redirect is not None:
            assert deployment.upstream_deployment_id == self.redirect.value
            assert deployment.reference_deployment_id == self.redirect
        else:
            assert deployment.upstream_deployment_id == deployment_name
            assert deployment.reference_deployment_id == self.deployment_id


@dataclass
class compat:
    deployment_id: str
    reference: ChatCompletionDeployment | EmbeddingsDeployment

    def check(self, deployments: AdapterDeployments):
        if isinstance(self.reference, ChatCompletionDeployment):
            deployment = deployments.chat_completions.get(self.deployment_id)
        else:
            deployment = deployments.embeddings.get(self.deployment_id)

        assert deployment is not None
        assert deployment.adapter_deployment_id == self.deployment_id
        assert deployment.upstream_deployment_id == self.deployment_id
        assert deployment.reference_deployment_id == self.reference


@dataclass
class TestCase:
    __test__ = False

    desc: str
    compat: Dict[str, str]

    error: str | None = None
    checks: List[Checker] = field(default_factory=list)


_chat_deployment = ChatCompletionDeployment.CLAUDE_3_5_HAIKU
_embedding_deployment = EmbeddingsDeployment.TEXT_EMBEDDING_4

test_cases: List[TestCase] = [
    TestCase(
        desc="invalid compat",
        compat={"xxx": "yyy", "zzz": "ddd"},
        error='None of the values in the following compatibility mapping corresponds to a VertexAI deployment supported by the adapter: {"xxx": "yyy", "zzz": "ddd"}. Remap the deployments to the supported VertexAI deployments to fix the error.',
    ),
    TestCase(
        desc="partially invalid compat",
        compat={
            "xxx": "yyy",
            "zzz": _chat_deployment.value,
        },
        error='None of the values in the following compatibility mapping corresponds to a VertexAI deployment supported by the adapter: {"xxx": "yyy"}. Remap the deployments to the supported VertexAI deployments to fix the error.',
    ),
    TestCase(
        desc="compat chat+embeddings",
        compat={
            "xxx": _chat_deployment.value,
            "yyy": _embedding_deployment.value,
        },
        checks=[
            supported(_chat_deployment),
            supported(_embedding_deployment),
            compat("xxx", _chat_deployment),
            compat("yyy", _embedding_deployment),
        ],
    ),
    TestCase(
        desc="compat supported deployment",
        compat={
            ChatCompletionDeployment.IMAGEN_005.value: _chat_deployment.value,
        },
        checks=[
            supported(_chat_deployment),
            compat(
                ChatCompletionDeployment.IMAGEN_005.value,
                _chat_deployment,
            ),
        ],
    ),
    TestCase(
        desc="compat mismatching supported deployments #1",
        compat={
            _chat_deployment.value: _embedding_deployment.value,
        },
        error="The chat completion deployment 'claude-3-5-haiku@20241022' is mapped onto the embeddings deployment 'text-embedding-004'",
    ),
    TestCase(
        desc="compat mismatching supported deployments #2",
        compat={
            _embedding_deployment.value: _chat_deployment.value,
        },
        error="The embeddings deployment 'text-embedding-004' is mapped onto the chat completion deployment 'claude-3-5-haiku@20241022'",
    ),
]


@pytest.mark.parametrize(
    "test_case", test_cases, ids=lambda t: t.desc.replace(" ", "_")
)
def test_compat_mapping(test_case: TestCase):
    if test_case.error is not None:
        with pytest.raises(ValueError, match=test_case.error):
            AdapterDeployments.create(compat_mapping=test_case.compat)
    else:
        deployments = AdapterDeployments.create(compat_mapping=test_case.compat)
        for checker in test_case.checks:
            checker.check(deployments)
