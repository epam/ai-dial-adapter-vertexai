import io
import json
from unittest.mock import patch

from google.auth import aws

from aidial_adapter_vertexai.aws_credentials import (
    _CredentialsSupplier,
    maybe_make_aws_credentials,
)

_WIF_INFO = {
    "type": "external_account",
    "audience": "//iam.googleapis.com/projects/1/locations/global/workloadIdentityPools/p/providers/v",
    "subject_token_type": "urn:ietf:params:aws:token-type:aws4_request",
    "token_url": "https://sts.googleapis.com/v1/token",
    "service_account_impersonation_url": "https://iamcredentials.googleapis.com/v1/projects/-/serviceAccounts/x@y.iam.gserviceaccount.com:generateAccessToken",
    "credential_source": {
        "environment_id": "aws1",
        "regional_cred_verification_url": "https://sts.{region}.amazonaws.com?Action=GetCallerIdentity&Version=2011-06-15",
    },
}

_AWS_CREDS_JSON = {
    "AccessKeyId": "ASIATESTKEYID0000000",
    "SecretAccessKey": "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    "Token": "FQoGZXIvYXdzEXAMPLE+TOKEN/+=",
    "Expiration": "2026-05-08T12:00:00Z",
}


def _wif_file(tmp_path):
    path = tmp_path / "wif.json"
    path.write_text(json.dumps(_WIF_INFO))
    return str(path)


def test_returns_none_when_no_container_env_vars(monkeypatch, tmp_path):
    monkeypatch.delenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", raising=False)
    monkeypatch.delenv("AWS_CONTAINER_CREDENTIALS_FULL_URI", raising=False)
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", _wif_file(tmp_path))

    assert maybe_make_aws_credentials() is None


def test_returns_none_when_credentials_file_not_external_account(
    monkeypatch, tmp_path
):
    sa_path = tmp_path / "sa.json"
    sa_path.write_text(
        json.dumps({"type": "service_account", "project_id": "p"})
    )
    monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/v2/creds")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", str(sa_path))

    assert maybe_make_aws_credentials() is None


def test_returns_credentials_when_environment_complete(monkeypatch, tmp_path):
    monkeypatch.setenv("AWS_CONTAINER_CREDENTIALS_RELATIVE_URI", "/v2/creds")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", _wif_file(tmp_path))

    creds = maybe_make_aws_credentials()
    assert isinstance(creds, aws.Credentials)


def test_supplier_reads_from_relative_uri(monkeypatch):
    monkeypatch.delenv("AWS_CONTAINER_AUTHORIZATION_TOKEN", raising=False)
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        return io.BytesIO(json.dumps(_AWS_CREDS_JSON).encode())

    with patch("urllib.request.urlopen", fake_urlopen):
        supplier = _CredentialsSupplier("http://169.254.170.2/v2/creds")
        result = supplier.get_aws_security_credentials(None, None)

    assert captured["url"] == "http://169.254.170.2/v2/creds"
    assert "Authorization" not in captured["headers"]
    assert result.access_key_id == _AWS_CREDS_JSON["AccessKeyId"]
    assert result.session_token == _AWS_CREDS_JSON["Token"]


def test_supplier_reads_from_full_uri_with_auth_token(monkeypatch):
    monkeypatch.setenv("AWS_CONTAINER_AUTHORIZATION_TOKEN", "Bearer xyz")
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["url"] = req.full_url
        captured["headers"] = dict(req.header_items())
        return io.BytesIO(json.dumps(_AWS_CREDS_JSON).encode())

    with patch("urllib.request.urlopen", fake_urlopen):
        supplier = _CredentialsSupplier("http://eks.example/creds")
        supplier.get_aws_security_credentials(None, None)

    assert captured["url"] == "http://eks.example/creds"
    assert captured["headers"].get("Authorization") == "Bearer xyz"


def test_supplier_reads_auth_token_from_file(monkeypatch, tmp_path):
    token_path = tmp_path / "eks-pod-identity-token"
    token_path.write_text("eks-token\n")
    monkeypatch.delenv("AWS_CONTAINER_AUTHORIZATION_TOKEN", raising=False)
    monkeypatch.setenv(
        "AWS_CONTAINER_AUTHORIZATION_TOKEN_FILE", str(token_path)
    )
    captured = {}

    def fake_urlopen(req, timeout=None):
        captured["headers"] = dict(req.header_items())
        return io.BytesIO(json.dumps(_AWS_CREDS_JSON).encode())

    with patch("urllib.request.urlopen", fake_urlopen):
        supplier = _CredentialsSupplier("http://169.254.170.23/v1/credentials")
        supplier.get_aws_security_credentials(None, None)

    assert captured["headers"].get("Authorization") == "eks-token"
