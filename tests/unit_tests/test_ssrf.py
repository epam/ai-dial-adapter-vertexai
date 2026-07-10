from collections.abc import Mapping

import pytest

from aidial_adapter_vertexai.chat.errors import ValidationError
from aidial_adapter_vertexai.dial_api import storage as storage_module
from aidial_adapter_vertexai.dial_api.ssrf import validate_public_url
from aidial_adapter_vertexai.dial_api.storage import FileStorage, download_file


@pytest.mark.parametrize(
    "url",
    [
        # Cloud metadata endpoint (link-local).
        "http://169.254.169.254/metadata/v1/instanceinfo",
        "http://127.0.0.1/",
        "http://localhost/secret",
        "http://10.0.0.1/",
        "http://192.168.1.1/",
        "http://172.16.0.1/",
        "http://0.0.0.0/",  # noqa: S104
        "https://[::1]/",
        "http://[::ffff:169.254.169.254]/",
        # Decimal-encoded 127.0.0.1.
        "http://2130706433/",
    ],
)
async def test_rejects_non_public_address(url: str):
    with pytest.raises(ValidationError, match="non-public address"):
        await validate_public_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "file:///etc/passwd",
        "ftp://8.8.8.8/",
        "gopher://8.8.8.8/",
        "//8.8.8.8/",  # empty scheme
    ],
)
async def test_rejects_disallowed_scheme(url: str):
    with pytest.raises(ValidationError, match="scheme is not allowed"):
        await validate_public_url(url)


@pytest.mark.parametrize(
    "url",
    [
        "http://8.8.8.8/file.txt",
        "https://1.1.1.1/file.txt",
        "https://[2606:4700:4700::1111]/file.txt",
    ],
)
async def test_allows_public_address(url: str):
    await validate_public_url(url)


@pytest.mark.parametrize(
    "link",
    [
        # ``userinfo`` trick: prefix matches dial_url but the real host is
        # the cloud metadata endpoint.
        "http://dial-core@169.254.169.254/metadata/v1/instanceinfo",
        # A look-alike domain that merely starts with the dial_url string.
        "http://dial-core.attacker.example/10.0.0.1",
    ],
)
async def test_file_storage_does_not_trust_prefix_lookalikes(link: str):
    storage = FileStorage(dial_url="http://dial-core", api_key="secret")
    # A non-DIAL origin must be treated as untrusted and validated, so a
    # non-public target is rejected before any authenticated request is made.
    with pytest.raises(ValidationError):
        await storage.download_file(link)


class _FakeResponse:
    def __init__(self, *, status: int, headers: dict, body: bytes):
        self.status = status
        self.headers = headers
        self._body = body

    async def __aenter__(self) -> "_FakeResponse":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    def raise_for_status(self) -> None:
        if self.status >= 400:
            raise AssertionError(f"unexpected status {self.status}")

    async def read(self) -> bytes:
        return self._body


class _FakeSession:
    """Minimal stand-in for ``aiohttp.ClientSession`` used to exercise the
    manual redirect handling without touching the network."""

    def __init__(self, routes: dict[str, _FakeResponse]):
        self._routes = routes
        self.calls: list[tuple[str, Mapping[str, str]]] = []

    async def __aenter__(self) -> "_FakeSession":
        return self

    async def __aexit__(self, *exc) -> bool:
        return False

    def get(self, url, *, headers, allow_redirects):
        assert allow_redirects is False
        self.calls.append((url, headers))
        return self._routes[url]


def _install_fake_session(
    monkeypatch, routes: dict[str, _FakeResponse]
) -> _FakeSession:
    session = _FakeSession(routes)
    monkeypatch.setattr(
        storage_module.aiohttp, "ClientSession", lambda: session
    )
    return session


def _redirect(location: str) -> _FakeResponse:
    return _FakeResponse(status=302, headers={"Location": location}, body=b"")


def _ok(body: bytes) -> _FakeResponse:
    return _FakeResponse(status=200, headers={}, body=body)


async def test_download_follows_public_redirect(monkeypatch):
    _install_fake_session(
        monkeypatch,
        {
            "http://8.8.8.8/a": _redirect("http://1.1.1.1/b"),
            "http://1.1.1.1/b": _ok(b"redirected-bytes"),
        },
    )
    assert await download_file("http://8.8.8.8/a") == b"redirected-bytes"


async def test_download_blocks_redirect_into_internal(monkeypatch):
    _install_fake_session(
        monkeypatch,
        {"http://8.8.8.8/a": _redirect("http://169.254.169.254/secret")},
    )
    # The redirect target is re-validated before the next request is made.
    with pytest.raises(ValidationError, match="non-public address"):
        await download_file("http://8.8.8.8/a")


async def test_download_rejects_redirect_loop(monkeypatch):
    _install_fake_session(
        monkeypatch,
        {"http://8.8.8.8/loop": _redirect("http://8.8.8.8/loop")},
    )
    with pytest.raises(ValidationError, match="too many redirects"):
        await download_file("http://8.8.8.8/loop")


async def test_trusted_origin_skips_validation_and_receives_auth(monkeypatch):
    # The trusted storage origin may live on a private address and must stay
    # reachable, and it is the only origin allowed to receive the api-key.
    session = _install_fake_session(
        monkeypatch,
        {"http://10.0.0.5:1234/v1/files/x": _ok(b"trusted-bytes")},
    )
    storage = FileStorage(dial_url="http://10.0.0.5:1234", api_key="secret")

    assert await storage.download_file("files/x") == b"trusted-bytes"

    url, headers = session.calls[0]
    assert url == "http://10.0.0.5:1234/v1/files/x"
    assert headers == {"api-key": "secret"}


async def test_redirect_off_trusted_origin_drops_auth(monkeypatch):
    session = _install_fake_session(
        monkeypatch,
        {
            "http://10.0.0.5:1234/v1/files/x": _redirect(
                "http://8.8.8.8/public"
            ),
            "http://8.8.8.8/public": _ok(b"public-bytes"),
        },
    )
    storage = FileStorage(dial_url="http://10.0.0.5:1234", api_key="secret")

    assert await storage.download_file("files/x") == b"public-bytes"

    # First (trusted) hop carries the api-key; the redirect leaving the
    # trusted origin must not.
    assert session.calls[0][1] == {"api-key": "secret"}
    assert session.calls[1][0] == "http://8.8.8.8/public"
    assert session.calls[1][1] == {}
