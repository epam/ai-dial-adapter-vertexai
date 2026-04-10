import json
from collections.abc import AsyncGenerator

import httpx
import openai
import pytest
from asgi_lifespan import LifespanManager
from google.cloud.aiplatform.constants.base import DEFAULT_REGION
from httpx import ASGITransport


@pytest.fixture(autouse=True)
def configure_unit_tests(monkeypatch, request):
    """
    Set up fake environment variables for unit tests.
    """
    if "tests/unit_tests" in request.node.nodeid:
        monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "test-creds")
        monkeypatch.setenv("DEFAULT_REGION", DEFAULT_REGION)
        monkeypatch.setenv("GCP_PROJECT_ID", "test-project-id")


@pytest.fixture()
async def test_http_client() -> AsyncGenerator[httpx.AsyncClient, None]:
    from aidial_adapter_vertexai.app import app

    async with (
        LifespanManager(app),
        httpx.AsyncClient(
            transport=ASGITransport(app),  # type: ignore
            base_url="http://test-app.com",
            params={"api-version": "dummy-version"},
            headers={"api-key": "dummy-key"},
        ) as client,
    ):
        yield client


def get_extra_headers(region: str) -> dict[str, str]:
    return {"x-upstream-extra-data": json.dumps({"region": region})}


class AsyncAzureOpenAI(openai.AsyncAzureOpenAI):
    def _should_retry(self, response: httpx.Response) -> bool:
        if response.status_code == 500:
            return False
        return super()._should_retry(response)


@pytest.fixture
def get_openai_client(test_http_client: httpx.AsyncClient):
    def _get_client(
        deployment_id: str | None = None,
        *,
        region: str | None = None,
        max_retries: int = 3,
        extra_headers: dict | None = None,
    ) -> openai.AsyncAzureOpenAI:
        default_headers = (extra_headers or {}) | (
            get_extra_headers(region) if region else {}
        )
        return AsyncAzureOpenAI(
            azure_endpoint=str(test_http_client.base_url),
            azure_deployment=deployment_id,
            api_version="dummy-version",
            api_key="dummy-key",
            max_retries=max_retries,
            timeout=30,
            http_client=test_http_client,
            default_headers=default_headers,
        )

    yield _get_client
