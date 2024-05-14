import re
from dataclasses import dataclass
from itertools import product
from typing import Callable, Dict, List

import pytest
from openai import AsyncAzureOpenAI
from openai.types import CreateEmbeddingResponse

from aidial_adapter_vertexai.deployments import EmbeddingsDeployment
from aidial_adapter_vertexai.dial_api.request import EmbeddingsType
from tests.conftest import DEFAULT_API_VERSION, TEST_SERVER_URL
from tests.utils.openai import sanitize_test_name

deployments: Dict[EmbeddingsDeployment, List[EmbeddingsType]] = {
    EmbeddingsDeployment.TEXT_EMBEDDING_GECKO_1: [EmbeddingsType.SYMMETRIC],
}


@dataclass
class TestCase:
    __test__ = False

    deployment: EmbeddingsDeployment
    input: str | List[str]
    headers: dict

    expected: Callable[[CreateEmbeddingResponse], None] | Exception

    def get_id(self):
        return sanitize_test_name(
            f"{self.deployment.value} {self.headers} {self.input}"
        )


def get_test_cases(
    deployment: EmbeddingsDeployment, allowed_types: List[EmbeddingsType]
) -> List[TestCase]:
    input = ["fish", "cat"]

    def test(resp: CreateEmbeddingResponse):
        assert resp.usage.prompt_tokens == len(input)
        assert resp.usage.total_tokens == len(input)
        assert len(resp.data) == len(input)
        assert len(resp.data[0].embedding) == 768

    ret: List[TestCase] = []

    for ty, instr in product(
        ["", "symmetric", "document", "query"], ["", "dummy"]
    ):
        headers = {}

        if instr:
            headers["X-DIAL-Instruction"] = instr

        if ty:
            headers["X-DIAL-Type"] = ty

        expected: Callable[[CreateEmbeddingResponse], None] | Exception = test
        if instr:
            expected = Exception("Instruction prompt is not supported")
        elif (ty or "symmetric") not in allowed_types:
            assert len(allowed_types) != 0
            allowed = ", ".join([e.value for e in allowed_types])
            expected = Exception(
                f"Embedding types other than {allowed} are not supported"
            )

        ret.append(
            TestCase(
                deployment=deployment,
                input=input,
                headers=headers,
                expected=expected,
            )
        )

    ret.append(
        TestCase(
            deployment=deployment,
            input=input,
            headers={"X-DIAL-Type": "FooBar"},
            expected=Exception("value is not a valid enumeration member"),
        )
    )

    return ret


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test",
    [
        test
        for model, types in deployments.items()
        for test in get_test_cases(model, types)
    ],
    ids=lambda test: test.get_id(),
)
async def test_embeddings_openai(server, test: TestCase):
    model_id = test.deployment.value

    client = AsyncAzureOpenAI(
        azure_endpoint=TEST_SERVER_URL,
        azure_deployment=model_id,
        api_version=DEFAULT_API_VERSION,
        api_key="dummy_key",
    )

    async def run() -> CreateEmbeddingResponse:
        return await client.embeddings.create(
            model=model_id, input=test.input, extra_headers=test.headers
        )

    if isinstance(test.expected, Exception):
        with pytest.raises(
            type(test.expected), match=re.escape(str(test.expected))
        ):
            await run()
    else:
        embeddings = await run()
        test.expected(embeddings)
