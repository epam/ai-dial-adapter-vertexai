import json
import logging
from typing import AsyncGenerator, Mapping

import httpx
import pytest
from asgi_lifespan import LifespanManager
from google.cloud.aiplatform.constants.base import DEFAULT_REGION
from httpx import ASGITransport
from openai import AsyncAzureOpenAI


def pytest_configure(config):
    # Filter out logs containing "Adapter deployments" because they are too verbose
    logging.getLogger("app").addFilter(
        lambda record: "Adapter deployments" not in record.getMessage()
    )


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

    async with LifespanManager(app):
        async with httpx.AsyncClient(
            transport=ASGITransport(app),  # type: ignore
            base_url="http://test-app.com",
            params={"api-version": "dummy-version"},
            headers={"api-key": "dummy-key"},
        ) as client:
            yield client


def get_extra_headers(region: str) -> Mapping[str, str]:
    return {"x-upstream-extra-data": json.dumps({"region": region})}


@pytest.fixture
def get_openai_client(test_http_client: httpx.AsyncClient):
    def _get_client(
        deployment_id: str | None = None,
        *,
        region: str | None = None,
        max_retries: int = 3,
    ) -> AsyncAzureOpenAI:
        return AsyncAzureOpenAI(
            azure_endpoint=str(test_http_client.base_url),
            azure_deployment=deployment_id,
            api_version="dummy-version",
            api_key="dummy-key",
            max_retries=max_retries,
            timeout=30,
            http_client=test_http_client,
            default_headers=get_extra_headers(region) if region else {},
        )

    yield _get_client
