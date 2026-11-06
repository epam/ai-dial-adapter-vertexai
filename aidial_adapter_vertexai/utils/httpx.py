from collections.abc import AsyncIterator
from contextvars import ContextVar

import aiohttp
import httpx
from httpx_aiohttp import AiohttpTransport, HttpxAiohttpClient


class _ReleasingStream(httpx.AsyncByteStream):
    """
    httpx binds the response stream to the response itself, which forms
    the reference cycle `Response -> BoundAsyncStream -> Response`.
    The response references the request, so the request body (e.g. inline
    attachments) survives until the cyclic garbage collection.
    The stream breaks the cycle by dropping the bound stream once closed.
    """

    def __init__(self, stream: httpx.AsyncByteStream) -> None:
        self._stream: httpx.AsyncByteStream | None = stream

    async def __aiter__(self) -> AsyncIterator[bytes]:
        if self._stream is None:
            raise httpx.StreamClosed()
        async for chunk in self._stream:
            yield chunk

    async def aclose(self) -> None:
        if (stream := self._stream) is not None:
            self._stream = None
            await stream.aclose()


class SharedAsyncClient(httpx.AsyncClient):
    async def send(
        self, request: httpx.Request, *, stream: bool = False, **kwargs
    ) -> httpx.Response:
        response = await super().send(request, stream=True, **kwargs)
        assert isinstance(response.stream, httpx.AsyncByteStream)
        response.stream = _ReleasingStream(response.stream)

        if not stream:
            try:
                await response.aread()
            except BaseException:
                await response.aclose()
                raise

        return response

    def build_request(
        self, *args, timeout=httpx.USE_CLIENT_DEFAULT, **kwargs
    ) -> httpx.Request:
        # Google GenAI SDK explicitly passes `timeout=None` to the httpx client
        # whenever `HttpOptions.timeout` isn't configured. Httpx treats it as
        # "no timeouts at all", which disables the pool timeout as well.
        # Fall back to the client-wide timeouts instead.
        if timeout is None:
            timeout = httpx.USE_CLIENT_DEFAULT
        return super().build_request(*args, timeout=timeout, **kwargs)


class SharedAiohttpClient(HttpxAiohttpClient, SharedAsyncClient):
    """
    The shared client on aiohttp's connection pool. httpcore's pool scans
    every pooled connection on each request event, which costs a lot of CPU
    with hundreds of concurrent streams.
    """


# Set while a request waits for a free pooled connection
_waiting_for_connection: ContextVar[list[bool]] = ContextVar(
    "_waiting_for_connection"
)


async def _on_connection_queued(*_) -> None:
    _waiting_for_connection.get()[0] = True


async def _on_connection_dequeued(*_) -> None:
    _waiting_for_connection.get()[0] = False


class SharedAiohttpTransport(AiohttpTransport):
    """
    aiohttp raises the same timeout for a request that waited too long for a
    pooled connection as for a slow connect. Report the first as
    `httpx.PoolTimeout`, as httpcore does, so it maps to 503.
    """

    def get_client(self) -> aiohttp.ClientSession:
        trace = aiohttp.TraceConfig()
        trace.on_connection_queued_start.append(_on_connection_queued)
        trace.on_connection_queued_end.append(_on_connection_dequeued)
        connector = aiohttp.TCPConnector(
            limit=self.limits.max_connections or 0,
            keepalive_timeout=self.limits.keepalive_expiry,
            ssl=self.ssl_context,
        )
        return aiohttp.ClientSession(connector=connector, trace_configs=[trace])

    async def handle_async_request(
        self, request: httpx.Request
    ) -> httpx.Response:
        waiting = [False]
        _waiting_for_connection.set(waiting)
        try:
            return await super().handle_async_request(request)
        except httpx.ConnectTimeout as e:
            if waiting[0]:
                raise httpx.PoolTimeout(str(e), request=request) from e
            raise
