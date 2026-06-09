from types import SimpleNamespace

from mistralai.client import Mistral

import aidial_adapter_vertexai.upstream_config as upstream_config_module
from aidial_adapter_vertexai.upstream_config import parse_upstream_config


class _RequestStub:
    def __init__(self, headers: dict[str, str]):
        self.original_request = SimpleNamespace(headers=headers)


def _fake_get_genai_client(called: dict[str, str]):
    async def _wrapped(project: str, location: str):
        called["project"] = project
        called["location"] = location
        return object()

    return _wrapped


async def test_parse_upstream_config_returns_mistral_client_for_api_key():
    request = _RequestStub(headers={"x-upstream-key": "test-key"})

    config = parse_upstream_config(request)  # type: ignore[arg-type]
    client = await config.get_mistral_client()

    assert isinstance(client, Mistral)


async def test_parse_upstream_config_uses_region_and_project_from_header(
    monkeypatch,
):
    called: dict[str, str] = {}

    monkeypatch.setattr(
        upstream_config_module,
        "get_genai_client",
        _fake_get_genai_client(called),
    )
    request = _RequestStub(
        headers={
            "x-upstream-extra-data": '{"region":"eu","project":"my-project"}'
        }
    )

    config = parse_upstream_config(request)  # type: ignore[arg-type]
    await config.get_genai_client()

    assert called == {"project": "my-project", "location": "eu"}


async def test_parse_upstream_config_falls_back_to_default_env(
    monkeypatch,
):
    called: dict[str, str] = {}

    monkeypatch.setenv("DEFAULT_REGION", "global")
    monkeypatch.setenv("GCP_PROJECT_ID", "project_id")
    monkeypatch.setattr(
        upstream_config_module,
        "get_genai_client",
        _fake_get_genai_client(called),
    )
    request = _RequestStub(headers={})

    config = parse_upstream_config(request)  # type: ignore[arg-type]
    await config.get_genai_client()

    assert called == {"project": "project_id", "location": "global"}
