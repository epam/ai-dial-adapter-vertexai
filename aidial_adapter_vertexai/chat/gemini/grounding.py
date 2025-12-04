from aidial_sdk.chat_completion import Attachment
from google.genai.types import Candidate as GenAICandidate

from aidial_adapter_vertexai.chat.consumer import Consumer
from aidial_adapter_vertexai.deployments import GeminiDeployment


def google_search_grounding_tokens(deployment: GeminiDeployment) -> int:
    # https://cloud.google.com/vertex-ai/generative-ai/pricing:
    # Gemini 2.0 Flash, 2.5 Flash and 2.5 Flash-Lite include a combined 1,500 grounded prompts per day at no additional charge. Gemini 2.5 Pro includes 10,000 grounded prompts per day at no additional charge.
    # Grounded prompts exceeding those limits are billed at $35 per 1,000 grounded prompts.
    # NOTE: we need a storage to keep track of #prompts/day,
    # meantime, the grounding tokens aren't reported
    return 0


async def create_grounding(
    candidate: GenAICandidate, consumer: Consumer
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
                    reference_type="text/markdown",
                )
            )
            grounding_added = True
    return grounding_added
