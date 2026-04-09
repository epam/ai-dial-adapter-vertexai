import itertools
from typing import List

import numpy as np
import openai
import pytest
from aidial_sdk.chat_completion import Attachment
from openai import AsyncAzureOpenAI

from aidial_adapter_vertexai.deployments import EmbeddingsDeployment as D
from aidial_adapter_vertexai.utils.resource import Resource
from tests.integration_tests.constants import (
    JPG_RESOURCE,
    MP3_RESOURCE,
    MP4_RESOURCE,
    PDF_RESOURCE,
    XLSX_RESOURCE,
)
from tests.integration_tests.embeddings.specs import (
    EMBEDDING_TYPES,
    GEMINI_MULTI_MODAL_SPEC,
    MULTI_MODAL_SPEC,
    SPECS,
    ModelSpec,
)
from tests.integration_tests.embeddings.test_case import EmbeddingsTestCase
from tests.utils.exception import ExpectedException, expected_exception
from tests.utils.openai import sanitize_test_name


def _display_spec(spec: ModelSpec) -> str:
    return _display_deployment(spec.deployment)


def _display_deployment(dep: D):
    return sanitize_test_name(dep.value)


def _error_bad_request(message: str) -> ExpectedException:
    return ExpectedException(
        type=openai.BadRequestError,
        message=message,
        status_code=400,
    )


def _error_unprocessable_entity(message: str) -> ExpectedException:
    return ExpectedException(
        type=openai.UnprocessableEntityError,
        message=message,
        status_code=422,
    )


@pytest.fixture(params=SPECS, ids=_display_spec)
def spec(request) -> ModelSpec:
    return request.param


@pytest.fixture
def client(spec: ModelSpec, get_openai_client) -> AsyncAzureOpenAI:
    model_id = spec.deployment.value
    return get_openai_client(model_id)


async def test_embeddings_one_text(client, spec: ModelSpec):
    tc = EmbeddingsTestCase(spec=spec, input="text-input")
    await tc.run(client)


async def test_embeddings_two_texts(client, spec: ModelSpec):
    tc = EmbeddingsTestCase(spec=spec, input=["text-input1", "text-input2"])
    await tc.run(client)


async def test_embeddings_custom_input_two_texts(client, spec: ModelSpec):
    tc = EmbeddingsTestCase(
        spec=spec, custom_input=["text-input1", "text-input2"]
    )
    await tc.run(client)


async def test_embeddings_input_and_custom_input(client, spec: ModelSpec):
    tc = EmbeddingsTestCase(
        spec=spec,
        input=["text-input1", "text-input2"],
        custom_input=["text-input3", "text-input4"],
    )
    await tc.run(client)


async def test_embeddings_output_dimensions(client, spec: ModelSpec):
    tc = EmbeddingsTestCase(spec=spec, input="text-input", dimensions=512)

    error = None
    if not spec.supports_dimensions:
        error = _error_unprocessable_entity(
            "Dimensions parameter is not supported"
        )

    await tc.run(client, error)


async def test_embeddings_instruction_prompt(client, spec: ModelSpec):
    tc = EmbeddingsTestCase(
        spec=spec, input="text-input", embedding_instr="instruction"
    )

    error = None
    if not spec.supports_instr:
        error = _error_unprocessable_entity(
            "Instruction prompt is not supported"
        )

    await tc.run(client, error)


@pytest.mark.parametrize("embedding_type", EMBEDDING_TYPES)
async def test_embeddings_task_type(
    client, spec: ModelSpec, embedding_type: str
):
    tc = EmbeddingsTestCase(
        spec=spec, input="text-input", embedding_type=embedding_type
    )

    error = None
    if not spec.supports_types:
        error = _error_unprocessable_entity(
            "The embedding model does not support embedding types"
        )
    elif embedding_type not in spec.supports_types:
        error = _error_bad_request(
            f"Unable to submit request because the model does not support the task type {embedding_type}"
        )

    await tc.run(client, error)


async def test_embeddings_titles(client, spec: ModelSpec):
    if not spec.supports_titles:
        pytest.skip("The model doesn't support titles")

    tc = EmbeddingsTestCase(
        spec=spec, custom_input=[["test-title", "test-input"]]
    )

    error = None
    if spec.deployment in [
        D.TEXT_EMBEDDING_4,
        D.TEXT_EMBEDDING_5,
        D.TEXT_GEMINI_EMBEDDING_1,
        D.TEXT_MULTILINGUAL_EMBEDDING_2,
    ]:
        error = _error_bad_request(
            "the model does not support the title parameter unless the task_type is RETRIEVAL_DOCUMENT"
        )

    await tc.run(client, error)


async def test_embeddings_titles_with_retrieval_document(
    client, spec: ModelSpec
):
    if not spec.supports_titles:
        pytest.skip("The model doesn't support titles")

    tc = EmbeddingsTestCase(
        spec=spec,
        custom_input=[["test-title", "test-input"]],
        embedding_type="RETRIEVAL_DOCUMENT",
    )

    await tc.run(client)


@pytest.mark.parametrize("format", [None, "base64", "float"])
async def test_embeddings_encoding_format(
    client, spec: ModelSpec, format: str | None
):
    tc = EmbeddingsTestCase(
        spec=spec, custom_input=["test-input"], encoding_format=format
    )
    await tc.run(client)


def _create_attachment(resource: Resource) -> dict:
    return Attachment(
        type=resource.type, url=resource.to_data_url()
    ).model_dump()


_IMAGE = _create_attachment(JPG_RESOURCE)
_AUDIO = _create_attachment(MP3_RESOURCE)
_VIDEO = _create_attachment(MP4_RESOURCE)
_DOCUMENT = _create_attachment(PDF_RESOURCE)
_UNSUPPORTED_DOCUMENT = _create_attachment(XLSX_RESOURCE)


class TestMultiModalEmbeddings:
    @pytest.fixture
    def spec(self) -> ModelSpec:
        return MULTI_MODAL_SPEC

    @pytest.mark.parametrize(
        "input,custom_input",
        [
            ("dog", ["cat", _IMAGE]),
            ([], [["image title", _IMAGE]]),
            ([], [[_IMAGE, "image title"]]),
        ],
    )
    async def test_multi_modal_basic(
        self, client, spec: ModelSpec, input, custom_input
    ):
        tc = EmbeddingsTestCase(
            spec=spec, input=input, custom_input=custom_input
        )
        await tc.run(client)

    async def test_multi_invalid_input1(self, client, spec: ModelSpec):
        tc = EmbeddingsTestCase(spec=spec, custom_input=[["text1", "text2"]])
        error = _error_unprocessable_entity(
            "The first element of a custom_input list element must be a string "
            "and the second element must be an image attachment or vice versa"
        )
        await tc.run(client, error)

    async def test_multi_invalid_input2(self, client, spec: ModelSpec):
        tc = EmbeddingsTestCase(
            spec=spec, custom_input=[["image title 2", _IMAGE, "image title 1"]]
        )
        error = _error_unprocessable_entity(
            "No more than two elements are allowed in an element of custom_input list"
        )
        await tc.run(client, error)


class TestGeminiMultiModalEmbeddings:
    @pytest.fixture
    def spec(self):
        return GEMINI_MULTI_MODAL_SPEC

    @pytest.mark.parametrize(
        "custom_input",
        [
            [_IMAGE],
            [_VIDEO],
            [_AUDIO],
            [[_IMAGE]],
            [_IMAGE, _AUDIO, _VIDEO],
            [[_IMAGE, _AUDIO, _DOCUMENT]],
            [["image title", _IMAGE]],
        ],
    )
    async def test_multi_modal_gemini_basic(
        self, client, spec: ModelSpec, custom_input
    ):
        tc = EmbeddingsTestCase(spec=spec, custom_input=custom_input)
        await tc.run(client)

    async def test_multi_modal_gemini_unsupported_document(
        self, client, spec: ModelSpec
    ):
        tc = EmbeddingsTestCase(spec=spec, custom_input=[_UNSUPPORTED_DOCUMENT])
        error = _error_bad_request(
            f"Unable to submit request because it has a mimeType parameter with value {_UNSUPPORTED_DOCUMENT['type']}, which is not supported."
        )
        await tc.run(client, error)


# Find the list of retired models here:
# https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions#retired-models
# https://ai.google.dev/gemini-api/docs/deprecations
_RETIRED_MODELS = [
    D.TEXT_EMBEDDING_GECKO_1,
    D.TEXT_EMBEDDING_GECKO_3,
    D.TEXT_EMBEDDING_GECKO_MULTILINGUAL_1,
]


@pytest.mark.parametrize("deployment", _RETIRED_MODELS, ids=_display_deployment)
async def test_retired_embedding_models(get_openai_client, deployment: D):
    model_id = deployment.value
    client = get_openai_client(model_id)

    async with expected_exception(
        cls=openai.NotFoundError, status_code=404, message="not found"
    ):
        await client.embeddings.create(model=model_id, input="test")


@pytest.mark.parametrize(
    "input",
    itertools.product(["cat", "dog"], repeat=3),
    ids=lambda input: "/".join(input),
)
async def test_multi_input_embeddings(
    client, spec: ModelSpec, input: List[str]
):
    model_id = spec.deployment.value
    response = await client.embeddings.create(
        model=model_id, input=input, encoding_format="float"
    )
    vectors = [np.array(emb.embedding) for emb in response.data]

    assert len(input) == len(vectors)

    eps = 1e-6

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
