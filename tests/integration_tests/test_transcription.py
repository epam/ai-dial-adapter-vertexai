import openai
import pytest

from aidial_adapter_vertexai.deployments import ChatCompletionDeployment as D
from tests.integration_tests.constants import DOG_PICTURE, MP3_RESOURCE
from tests.utils.exception import expected_exception
from tests.utils.openai import (
    ai,
    chat_completion,
    sys,
    user,
    user_with_attachment_data,
)

ALL_TRANSCRIPTION_MODELS: dict[D, str] = {
    D.GEMINI_3_5_TRANSCRIBE_PREVIEW: "global",
}

_TRANSCRIPT_WORD = "ishmael"


@pytest.fixture(params=[True, False], ids=lambda b: "stream" if b else "block")
def stream(request) -> bool:
    return request.param


@pytest.fixture(
    params=ALL_TRANSCRIPTION_MODELS.items(), ids=lambda item: item[0].value
)
def openai_client(request, get_openai_client):
    deployment, region = request.param
    return get_openai_client(deployment.value, region=region)


async def test_transcription(openai_client, stream: bool):
    response = await chat_completion(
        openai_client,
        stream=stream,
        messages=[user_with_attachment_data(None, MP3_RESOURCE)],
    )

    assert _TRANSCRIPT_WORD in response.content.lower()


async def test_transcription_follow_up_message(openai_client, stream: bool):
    response = await chat_completion(
        openai_client,
        stream=stream,
        messages=[
            user_with_attachment_data(None, MP3_RESOURCE),
            ai("Call me Ishmael."),
            user_with_attachment_data(None, MP3_RESOURCE),
        ],
    )

    assert _TRANSCRIPT_WORD in response.content.lower()


async def test_transcription_system_message(openai_client):
    async with expected_exception(
        cls=openai.UnprocessableEntityError,
        message="System messages aren't supported by the transcription model",
        status_code=422,
    ):
        await chat_completion(
            openai_client,
            messages=[
                sys("Act as a transcriber"),
                user_with_attachment_data(None, MP3_RESOURCE),
            ],
        )


async def test_transcription_without_audio(openai_client):
    async with expected_exception(
        cls=openai.UnprocessableEntityError,
        message="Attach an audio file to transcribe",
        display_message="Attach an audio file to transcribe",
        status_code=422,
    ):
        await chat_completion(openai_client, messages=[user("")])


async def test_transcription_text_input(openai_client):
    message = "Transcription models don't support text input."
    async with expected_exception(
        cls=openai.UnprocessableEntityError,
        message=message,
        display_message=message,
        status_code=422,
    ):
        await chat_completion(
            openai_client,
            messages=[user_with_attachment_data("Transcribe", MP3_RESOURCE)],
        )


async def test_transcription_non_audio_attachment(openai_client):
    async with expected_exception(
        cls=openai.UnprocessableEntityError,
        message="isn't one of the supported types",
        status_code=422,
    ):
        await chat_completion(
            openai_client,
            messages=[user_with_attachment_data(None, DOG_PICTURE)],
        )
