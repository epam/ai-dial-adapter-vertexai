from contextlib import asynccontextmanager

import vertexai
from aidial_sdk import DIALApp
from aidial_sdk.telemetry.types import TelemetryConfig

from aidial_adapter_vertexai.chat_completion import VertexAIChatCompletion
from aidial_adapter_vertexai.deployments import (
    ChatCompletionDeployment,
    EmbeddingsDeployment,
)
from aidial_adapter_vertexai.dial_api.exceptions import dial_exception_decorator
from aidial_adapter_vertexai.dial_api.response import (
    ModelObject,
    ModelsResponse,
)
from aidial_adapter_vertexai.embeddings import VertexAIEmbeddings
from aidial_adapter_vertexai.utils.env import get_env
from aidial_adapter_vertexai.utils.log_config import configure_loggers

DEFAULT_REGION = get_env("DEFAULT_REGION")
GCP_PROJECT_ID = get_env("GCP_PROJECT_ID")


@asynccontextmanager
async def lifespan(app: DIALApp):
    vertexai.init(project=GCP_PROJECT_ID, location=DEFAULT_REGION)
    yield


app = DIALApp(
    description="Google VertexAI adapter for DIAL API",
    telemetry_config=TelemetryConfig(),
    add_healthcheck=True,
    lifespan=lifespan,
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


for deployment in EmbeddingsDeployment:
    app.add_embeddings(
        deployment.get_model_id(),
        VertexAIEmbeddings(
            project_id=GCP_PROJECT_ID,
            region=DEFAULT_REGION,
        ),
    )
