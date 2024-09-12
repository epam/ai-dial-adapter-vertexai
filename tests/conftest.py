import os

import httpx
import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from google.cloud.aiplatform.constants.base import DEFAULT_REGION
from httpx import ASGITransport
from openai import AsyncAzureOpenAI


@pytest.fixture(autouse=True)
def configure_unit_tests(monkeypatch, request):
    """
    Set up fake environment variables for unit tests.
    """
    if "tests/unit_tests" in request.node.nodeid:
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "test-creds")
        monkeypatch.setenv("DEFAULT_REGION", DEFAULT_REGION)
        monkeypatch.setenv("GCP_PROJECT_ID", "test-project-id")


@pytest.fixture(autouse=True)
def disable_aiocache():
    # It's important to disable caches defined in aidial_adapter_vertexai.vertex_ai,
    # because pytest recreates app for every test function,
    # but the caches are shared across the tests.
    #
    # A cached model holds a reference to an event loop
    # that is created for the first app instance.
    # Once the first app instance is released, the event loop is closed.
    # This reference to a closed event loop breaks follow-up tests.
    os.environ["AIOCACHE_DISABLE"] = "1"


@pytest_asyncio.fixture()
async def test_http_client():
    from aidial_adapter_vertexai.app import app

    async with LifespanManager(app):
        async with httpx.AsyncClient(
            transport=ASGITransport(app),  # type: ignore
            base_url="http://test-app.com",
        ) as client:
            yield client


@pytest.fixture
def get_openai_client(test_http_client: httpx.AsyncClient):
    def _get_client(deployment_id: str | None = None) -> AsyncAzureOpenAI:
        return AsyncAzureOpenAI(
            azure_endpoint=str(test_http_client.base_url),
            azure_deployment=deployment_id,
            api_version="",
            api_key="dummy_key",
            max_retries=2,
            timeout=30,
            http_client=test_http_client,
        )

    yield _get_client
