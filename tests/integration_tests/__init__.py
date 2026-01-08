from tests.integration_tests.test_chat_completion_generation import (
    _DEPLOYMENT_TO_REGION,
)
from tests.integration_tests.test_image_generation import _IMAGEN_MODELS
from tests.integration_tests.test_video_generation import _VEO_MODELS
from tests.utils.validation import check_enum_completeness

_deployments = (
    list(_DEPLOYMENT_TO_REGION.keys())
    + list(_IMAGEN_MODELS.keys())
    + list(_VEO_MODELS.keys())
)
check_enum_completeness(_deployments)
