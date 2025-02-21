from contextlib import asynccontextmanager

from aidial_sdk import DIALApp
from aidial_sdk.telemetry.types import TelemetryConfig

from aidial_adapter_vertexai.adapter_deployments import AdapterDeployments
from aidial_adapter_vertexai.app_config import init_vertex_ai
from aidial_adapter_vertexai.chat_completion import VertexAIChatCompletion
from aidial_adapter_vertexai.dial_api.exceptions import dial_exception_decorator
from aidial_adapter_vertexai.dial_api.response import (
    ModelObject,
    ModelsResponse,
)
from aidial_adapter_vertexai.embeddings import VertexAIEmbeddings
from aidial_adapter_vertexai.utils.env import get_str_dict
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

deployments = AdapterDeployments.create(
    compat_mapping=get_str_dict("COMPATIBILITY_MAPPING")
)


@app.get("/openai/models")
@dial_exception_decorator
async def models():
    return ModelsResponse(
        data=list(
            map(ModelObject.chat_completions, deployments.chat_completions)
        )
        + list(map(ModelObject.embeddings, deployments.embeddings))
    )


for deployment_id, deployment in deployments.chat_completions.items():
    app.add_chat_completion(deployment_id, VertexAIChatCompletion(deployment))

for deployment_id, deployment in deployments.embeddings.items():
    app.add_embeddings(deployment_id, VertexAIEmbeddings(deployment))
