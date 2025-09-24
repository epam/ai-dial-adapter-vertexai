from aidial_sdk.chat_completion.request import AzureChatCompletionRequest

from aidial_adapter_vertexai.chat.static_tools import StaticToolsConfig


def test_no_tools():
    request = AzureChatCompletionRequest(messages=[])
    config = StaticToolsConfig.from_request(request)
    assert config.functions == []
