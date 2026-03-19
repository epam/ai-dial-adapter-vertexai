from aidial_adapter_vertexai.chat.gemini.adapter import ImageConfig


def test_gemini_extra_fields_in_image_config():
    config_source = {
        "aspect_ratio": "16:9",
        "image_size": "2K",
        "foo": "bar",  # extra fields are preserved
    }

    config_obj = ImageConfig.model_validate(config_source)
    config_dict = config_obj.to_image_config()

    assert config_dict == config_source
