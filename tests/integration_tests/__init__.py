from tests.integration_tests.test_chat_completion_generation import _DEPLOYMENTS
from tests.integration_tests.test_image_generation import _IMAGEN_MODELS
from tests.integration_tests.test_video_generation import _VEO_MODELS
from tests.utils.validation import check_enum_completeness

_deployments = (
    [d.deployment for d in _DEPLOYMENTS]
    + list(_IMAGEN_MODELS.keys())
    + list(_VEO_MODELS.keys())
)
check_enum_completeness(_deployments)
