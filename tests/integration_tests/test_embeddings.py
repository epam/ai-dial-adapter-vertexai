import itertools
import re
from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, List, Set

import numpy as np
import openai
import pytest
from aidial_sdk.chat_completion import Attachment
from openai.types import CreateEmbeddingResponse

from aidial_adapter_vertexai.deployments import EmbeddingsDeployment
from tests.utils.exception import expected_exception
from tests.utils.json import flatten_obj
from tests.utils.openai import sanitize_test_name


@dataclass
class ModelSpec:
    deployment: EmbeddingsDeployment
    supports_types: Set[str]
    supports_instr: bool
    default_dimensions: int
    supports_dimensions: bool


embedding_types: Set[str] = {
    "CLASSIFICATION",
    "CLUSTERING",
    "DEFAULT",
    "RETRIEVAL_DOCUMENT",
    "RETRIEVAL_QUERY",
    "SEMANTIC_SIMILARITY",
    "FACT_VERIFICATION",
    "QUESTION_ANSWERING",
}

specs: List[ModelSpec] = [
    ModelSpec(
        deployment=EmbeddingsDeployment.TEXT_GEMINI_EMBEDDING_1,
        supports_types=embedding_types,
        supports_instr=False,
        default_dimensions=3072,
        supports_dimensions=True,
    ),
    ModelSpec(
        deployment=EmbeddingsDeployment.TEXT_EMBEDDING_GECKO_3,
        supports_types=embedding_types,
        supports_instr=False,
        default_dimensions=768,
        supports_dimensions=False,
    ),
    ModelSpec(
        deployment=EmbeddingsDeployment.TEXT_EMBEDDING_4,
        supports_types=embedding_types,
        supports_instr=False,
        default_dimensions=768,
        supports_dimensions=True,
    ),
    ModelSpec(
        deployment=EmbeddingsDeployment.TEXT_EMBEDDING_GECKO_MULTILINGUAL_1,
        supports_types=embedding_types,
        supports_instr=False,
        default_dimensions=768,
        supports_dimensions=False,
    ),
    ModelSpec(
        deployment=EmbeddingsDeployment.TEXT_MULTILINGUAL_EMBEDDING_2,
        supports_types=embedding_types,
        supports_instr=False,
        default_dimensions=768,
        supports_dimensions=True,
    ),
]


@dataclass
class EmbeddingsTestCase:
    __test__ = False

    deployment: EmbeddingsDeployment
    input: str | List[str]
    extra_body: dict

    expected: Callable[[CreateEmbeddingResponse], None] | Exception

    def get_id(self):
        body_str = "/".join(
            [f"{path}:{val}" for path, val in flatten_obj(self.extra_body)]
        )
        input_str = (
            self.input if isinstance(self.input, str) else "_".join(self.input)
        )
        return sanitize_test_name(
            f"{self.deployment.value}/{body_str}/{input_str}"
        )


def display_deployment(dep: EmbeddingsDeployment):
    return sanitize_test_name(dep.value)


def check_embeddings_response(
    input: str | List[str],
    custom_input: list[Any] | None,
    dimensions: int,
) -> Callable[[CreateEmbeddingResponse], None]:
    def ret(resp: CreateEmbeddingResponse):
        n_inputs = 1 if isinstance(input, str) else len(input)
        n_inputs += len(custom_input) if custom_input else 0

        assert len(resp.data) == n_inputs
        assert len(resp.data[0].embedding) == dimensions

    return ret


def get_test_case(
    spec: ModelSpec,
    input: str | List[str],
    custom_input: list[Any] | None = None,
    encoding_format: str | None = None,
    embedding_type: str | None = None,
    embedding_instr: str | None = None,
    dimensions: int | None = None,
) -> EmbeddingsTestCase:

    has_titles = custom_input and any(isinstance(i, list) for i in custom_input)

    custom_fields = {}

    if embedding_instr:
        custom_fields["instruction"] = embedding_instr

    if embedding_type:
        custom_fields["type"] = embedding_type

    expected: Callable[[CreateEmbeddingResponse], None] | Exception = (
        check_embeddings_response(
            input, custom_input, dimensions or spec.default_dimensions
        )
    )

    if dimensions and not spec.supports_dimensions:
        expected = Exception("Dimensions parameter is not supported")
    elif embedding_instr and not spec.supports_instr:
        expected = Exception("Instruction prompt is not supported")
    elif embedding_type and len(spec.supports_types) == 0:
        expected = Exception(
            "The embedding model does not support embedding types"
        )
    elif has_titles and embedding_type != "RETRIEVAL_DOCUMENT":
        if spec.deployment == EmbeddingsDeployment.TEXT_GEMINI_EMBEDDING_1:
            expected = Exception(
                "the model does not support the title parameter unless the task_type is RETRIEVAL_DOCUMENT"
            )
        else:
            expected = Exception(
                "The model does not support inputs with titles "
                "unless the type is RETRIEVAL_DOCUMENT"
            )
    elif embedding_type and embedding_type not in spec.supports_types:
        # NOTE: error coming directly from Bedrock
        expected = Exception(
            f"Unable to submit request because the model does not support the task type {embedding_type}"
        )

    return EmbeddingsTestCase(
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


image_attachment = Attachment(
    type="image/png",
    url="https://www.google.com/images/branding/googlelogo/2x/googlelogo_color_92x30dp.png",
).dict()


def get_image_test_cases(
    input: str | List[str],
    custom_input: list[Any] | None,
    dimensions: int | None,
    exception: Exception | None,
) -> EmbeddingsTestCase:
    expected = exception or check_embeddings_response(
        input, custom_input, dimensions or 1408
    )

    return EmbeddingsTestCase(
        deployment=EmbeddingsDeployment.MULTI_MODAL_EMBEDDING_1,
        input=input,
        extra_body=(
            {
                "custom_input": custom_input,
                "dimensions": dimensions,
            }
        ),
        expected=expected,
    )


@pytest.mark.parametrize(
    "test",
    [
        get_image_test_cases(input, custom_input, dimensions, exception)
        for dimensions in [None, 512]
        for input, custom_input, exception in [
            ("dog", ["cat", image_attachment], None),
            ([], [["image title", image_attachment]], None),
            ([], [[image_attachment, "image title"]], None),
            (
                [],
                [["text1", "text2"]],
                Exception(
                    "The first element of a custom_input list element must be a string "
                    "and the second element must be an image attachment or vice versa"
                ),
            ),
            (
                [],
                [["image title 2", image_attachment, "image title 1"]],
                Exception(
                    "No more than two elements are allowed in an element of custom_input list"
                ),
            ),
        ]
    ]
    + [
        get_test_case(spec, input, custom_input, format, ty, instr, dims)
        for spec, input, custom_input, format, ty, instr, dims in product(
            specs,
            ["dog", ["fish", "cat"]],
            [None, ["ball", "sun"], [["title", "text"]]],
            ["base64", "float"],
            [None, *sorted(embedding_types)],
            [None, "instruction"],
            [None, 512],
        )
    ],
    ids=lambda test: test.get_id(),
)
async def test_embeddings(get_openai_client, test: EmbeddingsTestCase):
    model_id = test.deployment.value

    client: openai.AsyncAzureOpenAI = get_openai_client(model_id)

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


@pytest.mark.parametrize(
    "deployment",
    [EmbeddingsDeployment.TEXT_EMBEDDING_GECKO_1],
    ids=display_deployment,
)
async def test_retired_models(
    get_openai_client, deployment: EmbeddingsDeployment
):
    model_id = deployment.value
    client: openai.AsyncAzureOpenAI = get_openai_client(model_id)

    async with expected_exception(
        cls=openai.NotFoundError,
        status_code=404,
        message="not found",
    ):
        await client.embeddings.create(model=model_id, input="test")


@dataclass
class MultiInputTestCase:
    deployment: EmbeddingsDeployment
    input: List[str]

    def get_id(self):
        input_str = "/".join(self.input)
        return sanitize_test_name(f"{self.deployment.value}/{input_str}")


@pytest.mark.parametrize(
    "test",
    [
        MultiInputTestCase(deployment=spec.deployment, input=list(input))
        for input in itertools.product(["cat", "dog"], repeat=3)
        for spec in specs
    ],
    ids=lambda test: test.get_id(),
)
async def test_multi_input_embeddings(
    get_openai_client, test: MultiInputTestCase
):
    model_id = test.deployment.value
    client: openai.AsyncAzureOpenAI = get_openai_client(model_id)

    input = test.input

    response: CreateEmbeddingResponse = await client.embeddings.create(
        model=model_id, input=input, encoding_format="float"
    )
    vectors = [np.array(emb.embedding) for emb in response.data]

    assert len(input) == len(vectors)

    eps = 1e-8

    for i, a in enumerate(vectors):
        for j, b in enumerate(vectors):
            if i >= j:
                continue

            assert len(a) == len(b)
            dist = np.linalg.norm(a - b)
            if input[i] == input[j]:
                assert dist <= eps
            else:
                assert dist > eps
