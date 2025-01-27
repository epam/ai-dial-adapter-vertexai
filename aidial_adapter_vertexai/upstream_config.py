from aidial_sdk.deployment.from_request_mixin import FromRequestDeploymentMixin
from pydantic import BaseModel

from aidial_adapter_vertexai.app_config import DEFAULT_REGION
from aidial_adapter_vertexai.utils.log_config import app_logger as log

_UPSTREAM_CONFIG_HEADER_NAME = "x-upstream-extra-data"


class UpstreamConfig(BaseModel):

    region: str = DEFAULT_REGION

    @classmethod
    def from_request(
        cls, request: FromRequestDeploymentMixin
    ) -> "UpstreamConfig":
        conf = request.headers.get(_UPSTREAM_CONFIG_HEADER_NAME)
        return cls.model_validate_json(conf) if conf else cls()

    def dynamic_region_not_supported(self):
        if self.region != DEFAULT_REGION:
            log.warning(
                f"Per-request region configuration isn't supported for this deployment. The default region configured by the DEFAULT_REGION={DEFAULT_REGION!r} env var will be used instead of the requested {self.region!r}."
            )
