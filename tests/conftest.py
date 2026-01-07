import json
import logging
import os
from typing import AsyncGenerator, Dict
from unittest.mock import patch

import httpx
import openai
import pytest
from asgi_lifespan import LifespanManager
from google.cloud.aiplatform.constants.base import DEFAULT_REGION
from httpx import ASGITransport


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
def compatibility_mapping() -> dict[str, str]:
    return {}


@pytest.fixture()
async def test_http_client(
    compatibility_mapping,
) -> AsyncGenerator[httpx.AsyncClient, None]:
    with patch.dict(
        os.environ, {"COMPATIBILITY_MAPPING": json.dumps(compatibility_mapping)}
    ):
        from aidial_adapter_vertexai.app import app

        async with LifespanManager(app):
            async with httpx.AsyncClient(
                transport=ASGITransport(app),  # type: ignore
                base_url="http://test-app.com",
                params={"api-version": "dummy-version"},
                headers={"api-key": "dummy-key"},
            ) as client:
                yield client


def get_extra_headers(region: str) -> Dict[str, str]:
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
        headers: Dict[str, str] | None = None,
        max_retries: int = 3,
    ) -> openai.AsyncAzureOpenAI:
        default_headers = headers or {}
        if region:
            default_headers.update(get_extra_headers(region))
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
