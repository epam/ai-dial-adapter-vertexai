import json
import os
import urllib.request

from google.auth import aws

from aidial_adapter_vertexai.utils.log_config import app_logger as _log

_RELATIVE = "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"
_FULL = "AWS_CONTAINER_CREDENTIALS_FULL_URI"
_AUTH = "AWS_CONTAINER_AUTHORIZATION_TOKEN"


class _ContainerSupplier(aws.AwsSecurityCredentialsSupplier):
    def get_aws_security_credentials(
        self, context, request
    ) -> aws.AwsSecurityCredentials:
        if _FULL in os.environ:
            url = os.environ[_FULL]
            headers = {}
            token = os.environ.get(_AUTH)
            if token:
                headers["Authorization"] = token
        else:
            url = "http://169.254.170.2" + os.environ[_RELATIVE]
            headers = {}
        # URL is the container credential provider endpoint - hardcoded
        # 169.254.170.2 (ECS) or AWS_CONTAINER_CREDENTIALS_FULL_URI set by the
        # runtime (EKS Pod Identity, ECS in some configurations). Not user input.
        req = urllib.request.Request(url, headers=headers)  # noqa: S310
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            c = json.loads(resp.read())
        # Observability: log a short AccessKeyId fingerprint + Expiration so
        # operators can confirm rotation is being picked up across refreshes.
        # The fingerprint avoids leaking the full key id to logs.
        akid = c["AccessKeyId"]
        _log.debug(
            "wif-aws supplier: AccessKeyId=%s***%s expires=%s",
            akid[:4],
            akid[-4:],
            c.get("Expiration", "?"),
        )
        return aws.AwsSecurityCredentials(
            akid, c["SecretAccessKey"], c["Token"]
        )

    def get_aws_region(self, context, request) -> str:
        return (
            os.environ.get("AWS_REGION")
            or os.environ.get("AWS_DEFAULT_REGION")
            or "us-east-1"
        )


def maybe_make_aws_credentials() -> aws.Credentials | None:
    if _FULL not in os.environ and _RELATIVE not in os.environ:
        return None
    cred_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not cred_path or not os.path.isfile(cred_path):
        return None
    with open(cred_path) as f:
        info = json.load(f)
    if info.get("type") != "external_account":
        return None
    return aws.Credentials(
        audience=info["audience"],
        subject_token_type=info["subject_token_type"],
        token_url=info.get("token_url", "https://sts.googleapis.com/v1/token"),
        service_account_impersonation_url=info.get(
            "service_account_impersonation_url"
        ),
        aws_security_credentials_supplier=_ContainerSupplier(),
        default_scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
