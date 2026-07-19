from __future__ import annotations

import base64
import gzip
import socket
import traceback
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request

import pytest

from mcp_server_qdrant import safe_fetch

PUBLIC_IPV4 = "93.184.216.34"


def public_resolver(_host: str, _port: int, *, type: int) -> list[tuple[object, ...]]:
    assert type == socket.SOCK_STREAM
    return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (PUBLIC_IPV4, 443))]


class FakeResponse:
    def __init__(
        self,
        body: bytes = b"",
        *,
        status: int = 200,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status = status
        self.headers = headers or {}
        self._body = body
        self._offset = 0
        self.closed = False

    def read(self, size: int = -1) -> bytes:
        if size < 0:
            size = len(self._body) - self._offset
        chunk = self._body[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def close(self) -> None:
        self.closed = True


class FakeSocket:
    def __init__(self, *, peer_ip: str) -> None:
        self.peer_ip = peer_ip
        self.connected_to: tuple[object, ...] | None = None
        self.sent: list[bytes] = []
        self.closed = False
        self.timeout: object = None

    def settimeout(self, timeout: object) -> None:
        self.timeout = timeout

    def bind(self, _source_address: tuple[str, int]) -> None:
        return

    def connect(self, address: tuple[object, ...]) -> None:
        self.connected_to = address

    def getpeername(self) -> tuple[str, int]:
        assert self.connected_to is not None
        return self.peer_ip, int(self.connected_to[1])

    def setsockopt(self, *_args: object) -> None:
        return

    def sendall(self, data: bytes) -> None:
        self.sent.append(bytes(data))

    def close(self) -> None:
        self.closed = True


class FakeTLSContext:
    def __init__(self) -> None:
        self.server_hostnames: list[str] = []

    def wrap_socket(self, outbound_socket: FakeSocket, *, server_hostname: str) -> FakeSocket:
        self.server_hostnames.append(server_hostname)
        return outbound_socket


def install_responses(
    monkeypatch: pytest.MonkeyPatch,
    responses: list[FakeResponse],
    *,
    targets: list[object] | None = None,
) -> list[Request]:
    requests: list[Request] = []

    def fake_open(
        request: Request,
        *,
        timeout: float,
        target: object,
    ) -> FakeResponse:
        assert timeout > 0
        requests.append(request)
        if targets is not None:
            targets.append(target)
        if not responses:
            raise AssertionError("unexpected outbound request")
        return responses.pop(0)

    monkeypatch.setattr(safe_fetch, "_open_without_redirects", fake_open)
    return requests


def test_url_policy_fails_closed_without_allowlisted_hosts() -> None:
    with pytest.raises(safe_fetch.OutboundPolicyError, match="not allowlisted"):
        safe_fetch.validate_outbound_url(
            "https://example.test/document",
            allowed_hosts=set(),
            resolver=public_resolver,
        )


def test_url_policy_rejects_wildcard_allowlist_entries() -> None:
    with pytest.raises(safe_fetch.OutboundPolicyError, match="exact hostnames only"):
        safe_fetch.validate_outbound_url(
            "https://files.example.test/document",
            allowed_hosts={"*.example.test"},
            resolver=public_resolver,
        )


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1/document",
        "https://10.0.0.4/document",
        "https://169.254.169.254/latest/meta-data/",
        "https://224.0.0.1/document",
        "https://0.0.0.0/document",
        "https://240.0.0.1/document",
    ],
)
def test_url_policy_rejects_non_public_literal_addresses(url: str) -> None:
    host = url.split("/", 3)[2]
    with pytest.raises(safe_fetch.OutboundPolicyError, match="non-public"):
        safe_fetch.validate_outbound_url(url, allowed_hosts={host})


def test_url_policy_rejects_hostname_resolving_to_private_address() -> None:
    def private_resolver(_host: str, _port: int, *, type: int) -> list[tuple[object, ...]]:
        return [(socket.AF_INET, type, 6, "", ("192.168.1.10", 443))]

    with pytest.raises(safe_fetch.OutboundPolicyError, match="non-public"):
        safe_fetch.validate_outbound_url(
            "https://files.example.test/document",
            allowed_hosts={"files.example.test"},
            resolver=private_resolver,
        )


def test_url_policy_requires_https_and_allowlisted_port() -> None:
    with pytest.raises(safe_fetch.OutboundPolicyError, match="must use HTTPS"):
        safe_fetch.validate_outbound_url(
            "http://example.test/document",
            allowed_hosts={"example.test"},
            resolver=public_resolver,
        )

    assert (
        safe_fetch.validate_outbound_url(
            "http://example.test/document",
            allowed_hosts={"example.test"},
            allow_insecure_http=True,
            resolver=public_resolver,
        )
        == "http://example.test/document"
    )

    with pytest.raises(safe_fetch.OutboundPolicyError, match="standard HTTP port"):
        safe_fetch.validate_outbound_url(
            "http://example.test:8080/document",
            allowed_hosts={"example.test"},
            allowed_ports={443, 8443},
            allow_insecure_http=True,
            resolver=public_resolver,
        )

    with pytest.raises(safe_fetch.OutboundPolicyError, match="for HTTPS"):
        safe_fetch.validate_outbound_url(
            "https://example.test:80/document",
            allowed_hosts={"example.test"},
            allowed_ports={80, 443},
            resolver=public_resolver,
        )

    assert (
        safe_fetch.validate_outbound_url(
            "https://example.test:8443/document",
            allowed_hosts={"example.test"},
            allowed_ports={443, 8443},
            resolver=public_resolver,
        )
        == "https://example.test:8443/document"
    )


def test_pinned_https_connection_preserves_validated_ip_host_and_sni(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = safe_fetch._validate_outbound_target(  # noqa: SLF001
        "https://files.example.test/file",
        allowed_hosts={"files.example.test"},
        resolver=public_resolver,
    )
    outbound_socket = FakeSocket(peer_ip=PUBLIC_IPV4)
    created_with: list[tuple[int, int, int]] = []

    def socket_factory(family: int, socket_type: int, protocol: int) -> FakeSocket:
        created_with.append((family, socket_type, protocol))
        return outbound_socket

    def unexpected_dns(*_args: object, **_kwargs: object) -> list[tuple[object, ...]]:
        raise AssertionError("the pinned connection must not resolve the hostname again")

    monkeypatch.setattr(safe_fetch.socket, "socket", socket_factory)
    monkeypatch.setattr(safe_fetch.socket, "getaddrinfo", unexpected_dns)
    tls_context = FakeTLSContext()
    connection = safe_fetch._PinnedHTTPSConnection(  # noqa: SLF001
        target.hostname,
        target.port,
        validated_addresses=target.addresses,
        context=tls_context,
    )

    connection.request("GET", "/file")
    wire_request = b"".join(outbound_socket.sent)
    connection.close()

    assert created_with == [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP)]
    assert outbound_socket.connected_to == (PUBLIC_IPV4, 443)
    assert b"Host: files.example.test\r\n" in wire_request
    assert PUBLIC_IPV4.encode() not in wire_request
    assert tls_context.server_hostnames == ["files.example.test"]


def test_pinned_connection_rejects_unvalidated_actual_peer_before_request_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = safe_fetch._validate_outbound_target(  # noqa: SLF001
        "https://files.example.test/file",
        allowed_hosts={"files.example.test"},
        resolver=public_resolver,
    )
    outbound_socket = FakeSocket(peer_ip="127.0.0.1")
    monkeypatch.setattr(
        safe_fetch.socket,
        "socket",
        lambda *_args, **_kwargs: outbound_socket,
    )
    connection = safe_fetch._PinnedHTTPConnection(  # noqa: SLF001
        target.hostname,
        target.port,
        validated_addresses=target.addresses,
    )

    with pytest.raises(safe_fetch.OutboundPolicyError, match="non-public"):
        connection.request("GET", "/file")

    assert outbound_socket.connected_to == (PUBLIC_IPV4, 443)
    assert outbound_socket.sent == []
    assert outbound_socket.closed is True


def test_pinned_opener_rejects_request_target_origin_mismatch() -> None:
    target = safe_fetch._validate_outbound_target(  # noqa: SLF001
        "https://files.example.test/file",
        allowed_hosts={"files.example.test"},
        resolver=public_resolver,
    )

    with pytest.raises(safe_fetch.OutboundPolicyError, match="does not match"):
        safe_fetch._open_without_redirects(  # noqa: SLF001
            Request("https://other.example.test/file"),
            timeout=1,
            target=target,
        )


@pytest.mark.parametrize(
    "headers",
    [
        {"Host": "internal.example"},
        {"Content-Length": "1"},
        {"Transfer-Encoding": "chunked"},
        {"X-Test": "safe\r\nX-Injected: true"},
        {"Authorization": "one", "authorization": "two"},
    ],
)
def test_header_policy_rejects_unsafe_or_case_colliding_headers(
    headers: dict[str, str],
) -> None:
    with pytest.raises(safe_fetch.OutboundPolicyError):
        safe_fetch.validate_outbound_headers(headers)


def test_redirect_revalidates_destination_and_blocks_private_ip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        FakeResponse(
            status=302,
            headers={"Location": "https://169.254.169.254/latest/meta-data/"},
        )
    ]
    requests = install_responses(monkeypatch, responses)

    with pytest.raises(safe_fetch.OutboundPolicyError, match="non-public"):
        safe_fetch.fetch_bytes(
            "https://files.example.test/document",
            allowed_hosts={"files.example.test", "169.254.169.254"},
            max_bytes=1024,
            resolver=public_resolver,
        )
    assert len(requests) == 1


def test_redirect_resolves_and_pins_each_destination_independently(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    addresses = {
        "files.example.test": "93.184.216.34",
        "cdn.example.test": "151.101.1.69",
    }
    resolver_calls: list[tuple[str, int]] = []

    def redirect_resolver(
        host: str,
        port: int,
        *,
        type: int,
    ) -> list[tuple[object, ...]]:
        assert type == socket.SOCK_STREAM
        resolver_calls.append((host, port))
        return [(socket.AF_INET, type, socket.IPPROTO_TCP, "", (addresses[host], port))]

    targets: list[object] = []
    install_responses(
        monkeypatch,
        [
            FakeResponse(status=302, headers={"Location": "https://cdn.example.test/file"}),
            FakeResponse(b"safe"),
        ],
        targets=targets,
    )

    result = safe_fetch.fetch_bytes(
        "https://files.example.test/file",
        allowed_hosts=set(addresses),
        max_bytes=1024,
        resolver=redirect_resolver,
    )

    assert result.data == b"safe"
    assert resolver_calls == [
        ("files.example.test", 443),
        ("cdn.example.test", 443),
    ]
    assert [getattr(target, "hostname") for target in targets] == [
        "files.example.test",
        "cdn.example.test",
    ]
    assert [getattr(target, "addresses")[0].ip for target in targets] == [
        "93.184.216.34",
        "151.101.1.69",
    ]


def test_cross_origin_redirect_drops_credentials_and_sanitizes_final_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    responses = [
        FakeResponse(
            status=302,
            headers={"Location": "https://cdn.example.test/file?token=redirect-secret"},
        ),
        FakeResponse(b"safe", headers={"Content-Type": "text/plain"}),
    ]
    requests = install_responses(monkeypatch, responses)

    result = safe_fetch.fetch_bytes(
        "https://files.example.test/file?token=original-secret",
        headers={
            "Authorization": "Bearer top-secret",
            "Cookie": "session=top-secret",
            "X-API-Key": "top-secret",
            "User-Agent": "qdrant-test",
        },
        allowed_hosts={"files.example.test", "cdn.example.test"},
        max_bytes=1024,
        resolver=public_resolver,
    )

    assert result.data == b"safe"
    assert result.metadata.final_url == "https://cdn.example.test/"
    assert len(requests) == 2
    first_headers = {key.lower(): value for key, value in requests[0].header_items()}
    second_headers = {key.lower(): value for key, value in requests[1].header_items()}
    assert first_headers["authorization"] == "Bearer top-secret"
    assert first_headers["cookie"] == "session=top-secret"
    assert first_headers["x-api-key"] == "top-secret"
    assert "authorization" not in second_headers
    assert "cookie" not in second_headers
    assert "x-api-key" not in second_headers
    assert second_headers["user-agent"] == "qdrant-test"


def test_content_length_and_streaming_size_limits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_responses(
        monkeypatch,
        [FakeResponse(b"ignored", headers={"Content-Length": "100"})],
    )
    with pytest.raises(safe_fetch.OutboundSizeLimitError) as declared_error:
        safe_fetch.fetch_bytes(
            "https://files.example.test/file",
            allowed_hosts={"files.example.test"},
            max_bytes=10,
            resolver=public_resolver,
        )
    assert declared_error.value.limit == 10
    assert declared_error.value.actual == 100

    install_responses(monkeypatch, [FakeResponse(b"x" * 11)])
    with pytest.raises(safe_fetch.OutboundSizeLimitError) as streamed_error:
        safe_fetch.fetch_bytes(
            "https://files.example.test/file",
            allowed_hosts={"files.example.test"},
            max_bytes=10,
            chunk_bytes=4,
            resolver=public_resolver,
        )
    assert streamed_error.value.actual == 11


def test_decompressed_size_limit_blocks_compression_bomb(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    compressed = gzip.compress(b"x" * 4096)
    install_responses(
        monkeypatch,
        [FakeResponse(compressed, headers={"Content-Encoding": "gzip"})],
    )

    with pytest.raises(safe_fetch.OutboundSizeLimitError):
        safe_fetch.fetch_bytes(
            "https://files.example.test/file",
            allowed_hosts={"files.example.test"},
            max_bytes=1024,
            resolver=public_resolver,
        )


@pytest.mark.parametrize(
    "compressed",
    [
        gzip.compress(b"safe content")[:-2],
        gzip.compress(b"safe content") + b"trailing-data",
        gzip.compress(b"first") + gzip.compress(b"second"),
    ],
)
def test_compressed_response_rejects_truncation_and_trailing_members(
    compressed: bytes,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_responses(
        monkeypatch,
        [FakeResponse(compressed, headers={"Content-Encoding": "gzip"})],
    )

    with pytest.raises(
        safe_fetch.OutboundRequestError,
        match="incomplete or trailing compressed content",
    ):
        safe_fetch.fetch_bytes(
            "https://files.example.test/file",
            allowed_hosts={"files.example.test"},
            max_bytes=1024,
            resolver=public_resolver,
        )


def test_environment_proxies_are_disabled_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_handlers: list[object] = []

    class FakeOpener:
        def open(self, _request: Request, *, timeout: float) -> FakeResponse:
            assert timeout == 1
            return FakeResponse()

    def fake_build_opener(*handlers: object) -> FakeOpener:
        captured_handlers.extend(handlers)
        return FakeOpener()

    monkeypatch.setenv("HTTP_PROXY", "http://proxy.example.test:8080")
    monkeypatch.setenv("HTTPS_PROXY", "http://proxy.example.test:8080")
    monkeypatch.setattr(safe_fetch, "build_opener", fake_build_opener)

    safe_fetch._open_without_redirects(  # pylint: disable=protected-access
        Request("https://files.example.test/file"),
        timeout=1,
        target=safe_fetch._validate_outbound_target(  # noqa: SLF001
            "https://files.example.test/file",
            allowed_hosts={"files.example.test"},
            resolver=public_resolver,
        ),
    )

    proxy_handlers = [
        handler for handler in captured_handlers if isinstance(handler, safe_fetch.ProxyHandler)
    ]
    assert len(proxy_handlers) == 1
    assert proxy_handlers[0].proxies == {}
    assert any(isinstance(handler, safe_fetch._PinnedHTTPHandler) for handler in captured_handlers)
    assert any(isinstance(handler, safe_fetch._PinnedHTTPSHandler) for handler in captured_handlers)


def test_content_type_is_bounded_and_control_character_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_responses(
        monkeypatch,
        [FakeResponse(b"safe", headers={"Content-Type": "text/plain; charset=utf-8"})],
    )
    safe_result = safe_fetch.fetch_bytes(
        "https://files.example.test/file",
        allowed_hosts={"files.example.test"},
        max_bytes=1024,
        resolver=public_resolver,
    )
    assert safe_result.metadata.content_type == "text/plain; charset=utf-8"

    for unsafe_value in ("text/plain\x00secret", "x" * 257):
        install_responses(
            monkeypatch,
            [FakeResponse(b"safe", headers={"Content-Type": unsafe_value})],
        )
        unsafe_result = safe_fetch.fetch_bytes(
            "https://files.example.test/file",
            allowed_hosts={"files.example.test"},
            max_bytes=1024,
            resolver=public_resolver,
        )
        assert unsafe_result.metadata.content_type is None


def test_failed_path_fetch_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    destination = tmp_path / "download.pdf"
    install_responses(monkeypatch, [FakeResponse(b"x" * 11)])

    with pytest.raises(safe_fetch.OutboundSizeLimitError):
        safe_fetch.fetch_to_path(
            "https://files.example.test/file",
            destination,
            allowed_hosts={"files.example.test"},
            max_bytes=10,
            chunk_bytes=4,
            resolver=public_resolver,
        )

    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))


def test_bounded_base64_decode_preflights_and_validates() -> None:
    assert (
        safe_fetch.decode_base64_bounded(base64.b64encode(b"safe").decode("ascii"), max_bytes=4)
        == b"safe"
    )

    with pytest.raises(safe_fetch.OutboundSizeLimitError):
        safe_fetch.decode_base64_bounded(
            base64.b64encode(b"oversized").decode("ascii"), max_bytes=4
        )
    with pytest.raises(safe_fetch.OutboundPolicyError, match="invalid"):
        safe_fetch.decode_base64_bounded("not-base64!", max_bytes=100)


def test_errors_and_retained_reference_never_include_url_credentials() -> None:
    credentialed_url = "https://agent:password@example.test/document?token=query-secret#fragment"
    with pytest.raises(safe_fetch.OutboundPolicyError) as error:
        safe_fetch.validate_outbound_url(
            credentialed_url,
            allowed_hosts={"example.test"},
            resolver=public_resolver,
        )
    serialized_error = str(error.value)
    assert "agent" not in serialized_error
    assert "password" not in serialized_error
    assert "query-secret" not in serialized_error

    assert safe_fetch.sanitize_source_reference(credentialed_url) == "https://example.test/"


def test_network_exception_chain_does_not_reveal_url_or_path_material(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sensitive_value = "SENSITIVE_CHAIN_VALUE_123"
    synthetic_private_path = "/" + "home/example/private-download.pdf"
    sensitive_reason = (
        f"https://files.example.test/file?credential={sensitive_value} {synthetic_private_path}"
    )

    def fail_open(
        _request: Request,
        *,
        timeout: float,
        target: object,
    ) -> FakeResponse:
        assert timeout > 0
        assert target is not None
        raise URLError(sensitive_reason)

    monkeypatch.setattr(safe_fetch, "_open_without_redirects", fail_open)

    with pytest.raises(safe_fetch.OutboundRequestError) as caught:
        safe_fetch.fetch_bytes(
            "https://files.example.test/file",
            allowed_hosts={"files.example.test"},
            max_bytes=1024,
            resolver=public_resolver,
        )

    formatted = "".join(
        traceback.format_exception(
            type(caught.value),
            caught.value,
            caught.value.__traceback__,
        )
    )
    assert sensitive_value not in formatted
    assert synthetic_private_path not in formatted
