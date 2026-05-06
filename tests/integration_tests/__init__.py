import tests.integration_tests.embeddings.test_embeddings as embeddings
from tests.integration_tests.test_chat_completion_generation import (
    _DEPLOYMENT_TO_REGION,
)
from tests.integration_tests.test_image_generation import ALL_IMAGE_GEN_MODELS
from tests.integration_tests.test_video_generation import ALL_VIDEO_GEN_MODELS
from tests.utils.validation import check_enum_completeness

_chat_deployments = (
    list(_DEPLOYMENT_TO_REGION.keys())
    + list(ALL_IMAGE_GEN_MODELS.keys())
    + list(ALL_VIDEO_GEN_MODELS.keys())
)
check_enum_completeness(_chat_deployments)

_embedding_deployments = [
    spec.deployment for spec in embeddings.SPECS
] + embeddings._RETIRED_MODELS

check_enum_completeness(_embedding_deployments)
