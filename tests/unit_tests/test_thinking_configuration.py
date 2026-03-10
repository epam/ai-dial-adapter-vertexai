from aidial_adapter_vertexai.chat.gemini.adapter import ThinkingConfig


def test_gemini_extra_fields_in_thinking_config():
    config_source = {
        "include_thoughts": True,
        "thinking_budget": 42,
        "foo": "bar",  # extra fields are preserved
    }

    config_obj = ThinkingConfig.model_validate(config_source)
    config_dict = config_obj.to_thinking_config()

    assert config_dict == config_source
