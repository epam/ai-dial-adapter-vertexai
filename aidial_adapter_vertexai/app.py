from contextlib import asynccontextmanager

from aidial_sdk import DIALApp
from aidial_sdk.telemetry.types import TelemetryConfig

from aidial_adapter_vertexai.app_config import init_vertex_ai
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
from aidial_adapter_vertexai.utils.log_config import configure_loggers


@asynccontextmanager
async def lifespan(app: DIALApp):
    init_vertex_ai()
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
    app.add_chat_completion(deployment.get_model_id(), VertexAIChatCompletion())
for deployment in EmbeddingsDeployment:
    app.add_embeddings(deployment.get_model_id(), VertexAIEmbeddings())
