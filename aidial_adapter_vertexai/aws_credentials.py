import json
import os
import urllib.request
from urllib.parse import urljoin

from google.auth import aws

from aidial_adapter_vertexai.utils.log_config import app_logger as _log

# https://docs.aws.amazon.com/sdkref/latest/guide/feature-container-credentials.html
_RELATIVE = "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI"
_FULL = "AWS_CONTAINER_CREDENTIALS_FULL_URI"
_AUTH = "AWS_CONTAINER_AUTHORIZATION_TOKEN"
_ECS_AGENT = "http://169.254.170.2"


class _CredentialsSupplier(aws.AwsSecurityCredentialsSupplier):
    def __init__(self, cred_url: str) -> None:
        self._cred_url = cred_url

    def get_aws_security_credentials(
        self, context, request
    ) -> aws.AwsSecurityCredentials:
        headers = {}
        if (token := os.environ.get(_AUTH)) is not None:
            headers["Authorization"] = token
        # URL was resolved at startup from AWS_CONTAINER_CREDENTIALS_{FULL,RELATIVE}_URI
        # (link-local for ECS, FQDN for EKS Pod Identity). Not user input.
        req = urllib.request.Request(self._cred_url, headers=headers)  # noqa: S310
        with urllib.request.urlopen(req, timeout=5) as resp:  # noqa: S310
            c = json.loads(resp.read())
        # Observability: short AccessKeyId fingerprint + Expiration so operators
        # can confirm rotation is being picked up across refreshes.
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
    cred_url = os.getenv(_FULL) or os.getenv(_RELATIVE)
    if cred_url is None:
        return None
    cred_url = urljoin(_ECS_AGENT, cred_url)
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
        aws_security_credentials_supplier=_CredentialsSupplier(cred_url),
        default_scopes=["https://www.googleapis.com/auth/cloud-platform"],
    )
