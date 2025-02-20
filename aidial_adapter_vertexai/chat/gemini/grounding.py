from typing import assert_never

from aidial_sdk.chat_completion import Attachment
from google.genai.types import Candidate as GenAICandidate
from vertexai.preview.generative_models import Candidate

from aidial_adapter_vertexai.chat.consumer import Consumer
from aidial_adapter_vertexai.deployments import (
    ChatCompletionDeployment,
    Gemini2Deployment,
    GeminiDeployment,
)


def google_search_grounding_tokens(
    deployment: GeminiDeployment | Gemini2Deployment,
) -> int:
    # Grounding is $35 / 1K requests, so it's 0.035$ / 1 request
    match deployment:
        case (
            ChatCompletionDeployment.GEMINI_FLASH_1_5_V1
            | ChatCompletionDeployment.GEMINI_FLASH_1_5_V2
        ):
            # $0.30 / 1 million tokens
            # So 0.035$ = (0.035 / 0.3) * 1M tokens = 116,667 tokens
            return 116_667
        case (
            ChatCompletionDeployment.GEMINI_PRO_1_5_V1
            | ChatCompletionDeployment.GEMINI_PRO_1_5_V2
            | ChatCompletionDeployment.GEMINI_PRO_1_5_PREVIEW
        ):
            # $5.00 / 1 million tokens
            # So 0.035$ = (0.035 / 5) * 1M tokens = 7,000 tokens
            return 7_000
        case ChatCompletionDeployment.GEMINI_PRO_1:
            #  $1.50 / 1 million tokens
            # So 0.035$ = (0.035 / 1.5) * 1M tokens = 23,333 tokens
            return 23_333
        case (
            ChatCompletionDeployment.GEMINI_2_0_FLASH_EXP
            | ChatCompletionDeployment.GEMINI_2_0_FLASH_001
            | ChatCompletionDeployment.GEMINI_2_0_PRO_EXP_02_05
        ):
            # TODO: Add pricing, when it will be available.
            # Currently, while this models are in experimental mode, there is no pricing information.
            return 0
        case (
            ChatCompletionDeployment.GEMINI_PRO_VISION_1
            | ChatCompletionDeployment.GEMINI_2_0_FLASH_THINKING_EXP_01_21
            | ChatCompletionDeployment.GEMINI_2_0_FLASH_LITE_PREVIEW_02_05
        ):
            raise RuntimeError("Gemini Pro Vision 1 does not support grounding")

        case _:
            assert_never(deployment)


async def create_grounding(
    candidate: Candidate | GenAICandidate, consumer: Consumer
) -> bool:
    if not (metadata := candidate.grounding_metadata) or not (
        supports := metadata.grounding_supports
    ):
        return False

    grounding_added = False
    for support in supports:
        if not (chunk_indices := support.grounding_chunk_indices):
            continue

        for chunk_index in chunk_indices:
            if not metadata.grounding_chunks:
                continue
            chunk = metadata.grounding_chunks[chunk_index]
            if not chunk.web or not chunk.web.uri:
                continue
            await consumer.add_attachment(
                Attachment(
                    reference_url=chunk.web.uri,
                    data=support.segment.text if support.segment else None,
                    title=chunk.web.title,
                    type="text/markdown",
                )
            )
            grounding_added = True
    return grounding_added
