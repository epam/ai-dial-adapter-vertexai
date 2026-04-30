from types import SimpleNamespace

from mistralai.client import Mistral

from aidial_adapter_vertexai.upstream_config import parse_upstream_config


class _RequestStub:
    def __init__(self, headers: dict[str, str]):
        self.original_request = SimpleNamespace(headers=headers)


async def test_parse_upstream_config_returns_mistral_client_for_api_key():
    request = _RequestStub(headers={"x-upstream-key": "test-key"})

    config = parse_upstream_config(request)  # type: ignore[arg-type]
    client = await config.get_mistral_client()

    assert isinstance(client, Mistral)
