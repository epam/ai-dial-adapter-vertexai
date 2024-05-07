import json
from typing import Optional

from aidial_sdk import DIALApp
from aidial_sdk.telemetry.types import TelemetryConfig
from fastapi import Body, Header, Path

from aidial_adapter_vertexai.adapters import get_embeddings_model
from aidial_adapter_vertexai.chat_completion import VertexAIChatCompletion
from aidial_adapter_vertexai.deployments import (
    ChatCompletionDeployment,
    EmbeddingsDeployment,
)
from aidial_adapter_vertexai.dial_api.exceptions import dial_exception_decorator
from aidial_adapter_vertexai.dial_api.request import (
    EmbeddingsQuery,
    EmbeddingsType,
)
from aidial_adapter_vertexai.dial_api.response import (
    ModelObject,
    ModelsResponse,
    make_embeddings_response,
)
from aidial_adapter_vertexai.utils.env import get_env
from aidial_adapter_vertexai.utils.log_config import app_logger as log
from aidial_adapter_vertexai.utils.log_config import configure_loggers

DEFAULT_REGION = get_env("DEFAULT_REGION")
GCP_PROJECT_ID = get_env("GCP_PROJECT_ID")

app = DIALApp(
    description="Google VertexAI adapter for DIAL API",
    telemetry_config=TelemetryConfig(),
    add_healthcheck=True,
)

# NOTE: configuring logger after the DIAL telemetry is initialized,
# because it may have configured the root logger on its own.
configure_loggers()


@app.get("/openai/models")
@dial_exception_decorator
async def models():
    models = [
        ModelObject(id=model.value, object="model")
        for model in ChatCompletionDeployment
    ]

    return ModelsResponse(data=models)


for deployment in ChatCompletionDeployment:
    app.add_chat_completion(
        deployment.get_model_id(),
        VertexAIChatCompletion(
            project_id=GCP_PROJECT_ID,
            region=DEFAULT_REGION,
        ),
    )


@app.post("/openai/deployments/{deployment}/embeddings")
@dial_exception_decorator
async def embeddings(
    embeddings_type: EmbeddingsType = Header(
        alias="X-DIAL-Type", default=EmbeddingsType.SYMMETRIC
    ),
    embeddings_instruction: Optional[str] = Header(
        alias="X-DIAL-Instruction", default=None
    ),
    deployment: EmbeddingsDeployment = Path(...),
    query: EmbeddingsQuery = Body(..., example=EmbeddingsQuery.example()),
):
    log.debug(f"query: {json.dumps(query.dict(exclude_none=True))}")

    model = await get_embeddings_model(
        location=DEFAULT_REGION,
        deployment=deployment,
        project_id=GCP_PROJECT_ID,
    )

    response = await model.embeddings(
        query.input, embeddings_instruction, embeddings_type
    )

    return make_embeddings_response(deployment, response)
