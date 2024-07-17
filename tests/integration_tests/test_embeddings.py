import re
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, List

import pytest
from openai import AsyncAzureOpenAI
from openai.types import CreateEmbeddingResponse

from aidial_adapter_vertexai.deployments import EmbeddingsDeployment
from tests.conftest import DEFAULT_API_VERSION, TEST_SERVER_URL
from tests.utils.openai import sanitize_test_name


@dataclass
class ModelSpec:
    deployment: EmbeddingsDeployment
    supports_types: List[str]
    supports_instr: bool
    default_dimension: int
    supports_dimensions: bool


specs: List[ModelSpec] = [
    ModelSpec(
        deployment=EmbeddingsDeployment.TEXT_EMBEDDING_GECKO_1,
        supports_types=[],
        supports_instr=False,
        default_dimension=768,
        supports_dimensions=False,
    ),
]


@dataclass
class TestCase:
    __test__ = False

    deployment: EmbeddingsDeployment
    input: str | List[str]
    extra_body: dict

    expected: Callable[[CreateEmbeddingResponse], None] | Exception

    def get_id(self):
        return sanitize_test_name(
            f"{self.deployment.value} {self.extra_body} {self.input}"
        )


gecko_title_test_case: TestCase = TestCase(
    deployment=EmbeddingsDeployment.TEXT_EMBEDDING_GECKO_1,
    input=[],
    extra_body=({"custom_input": [["title", "text"]]}),
    expected=Exception(
        "The model does not support inputs with titles "
        "unless the type is 'RETRIEVAL_DOCUMENT'"
    ),
)


def get_test_case(
    spec: ModelSpec,
    input: str | List[str],
    custom_input: list[Any] | None,
    encoding_format: str | None,
    embedding_type: str | None,
    embedding_instr: str | None,
    dimensions: int | None,
) -> TestCase:

    def check_response(resp: CreateEmbeddingResponse):
        n_inputs = 1 if isinstance(input, str) else len(input)
        if custom_input:
            n_inputs += len(custom_input)

        assert resp.usage.prompt_tokens == n_inputs
        assert resp.usage.total_tokens == n_inputs
        assert len(resp.data) == n_inputs
        assert (
            len(resp.data[0].embedding) == dimensions or spec.default_dimension
        )

    custom_fields = {}

    if embedding_instr:
        custom_fields["instruction"] = embedding_instr

    if embedding_type:
        custom_fields["type"] = embedding_type

    expected: Callable[[CreateEmbeddingResponse], None] | Exception = (
        check_response
    )

    if dimensions and not spec.supports_dimensions:
        expected = Exception("Request parameter 'dimensions' is not supported")
    elif embedding_instr and not spec.supports_instr:
        expected = Exception(
            "Request parameter 'custom_fields.instruction' is not supported"
        )
    elif embedding_type and embedding_type not in spec.supports_types:
        expected = Exception(
            "Request parameter 'custom_fields.type' is not supported"
        )

    return TestCase(
        deployment=spec.deployment,
        input=input,
        extra_body=(
            {
                "custom_input": custom_input,
                "custom_fields": custom_fields,
                "encoding_format": encoding_format,
                "dimensions": dimensions,
            }
        ),
        expected=expected,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "test",
    [
        get_test_case(spec, input, custom_input, format, ty, instr, dims)
        for spec, input, custom_input, format, ty, instr, dims in product(
            specs,
            ["dog", ["fish", "cat"]],
            [None, ["ball", "sun"]],
            ["base64", "float"],
            [
                None,
                "CLASSIFICATION",
                "CLUSTERING",
                "DEFAULT",
                "FACT_VERIFICATION",
                "QUESTION_ANSWERING",
                "RETRIEVAL_DOCUMENT",
                "RETRIEVAL_QUERY",
                "SEMANTIC_SIMILARITY",
            ],
            [None, "dummy"],
            [None, 512],
        )
    ],
    ids=lambda test: test.get_id(),
)
async def test_embeddings(server, test: TestCase):
    model_id = test.deployment.value

    client = AsyncAzureOpenAI(
        azure_endpoint=TEST_SERVER_URL,
        azure_deployment=model_id,
        api_version=DEFAULT_API_VERSION,
        api_key="dummy_key",
    )

    async def run() -> CreateEmbeddingResponse:
        return await client.embeddings.create(
            model=model_id, input=test.input, extra_body=test.extra_body
        )

    if isinstance(test.expected, Exception):
        with pytest.raises(
            type(test.expected), match=re.escape(str(test.expected))
        ):
            await run()
    else:
        embeddings = await run()
        test.expected(embeddings)
