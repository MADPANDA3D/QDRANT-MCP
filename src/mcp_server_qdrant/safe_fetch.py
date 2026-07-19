from __future__ import annotations

import base64
import binascii
import hashlib
import http.client
import io
import ipaddress
import re
import socket
import tempfile
import zlib
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, BinaryIO, Protocol
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.request import (
    HTTPHandler,
    HTTPRedirectHandler,
    HTTPSHandler,
    ProxyHandler,
    Request,
    build_opener,
)

DEFAULT_FETCH_TIMEOUT_SECONDS = 30
DEFAULT_MAX_REDIRECTS = 3
DEFAULT_STREAM_CHUNK_BYTES = 64 * 1024
MAX_URL_CHARS = 4096
MAX_HEADER_COUNT = 32
MAX_HEADER_NAME_CHARS = 128
MAX_HEADER_VALUE_CHARS = 8192
MAX_HEADER_BLOCK_CHARS = 16 * 1024
MAX_CONTENT_TYPE_CHARS = 256

_HEADER_NAME_RE = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_BLOCKED_REQUEST_HEADERS = {
    "connection",
    "content-length",
    "host",
    "proxy-authorization",
    "proxy-connection",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
_CROSS_ORIGIN_REDIRECT_HEADERS = {
    "accept",
    "accept-encoding",
    "accept-language",
    "cache-control",
    "user-agent",
}
_BLOCKED_HOSTS = {
    "instance-data",
    "instance-data.ec2.internal",
    "localhost",
    "localhost.localdomain",
    "metadata",
    "metadata.aws.internal",
    "metadata.azure.internal",
    "metadata.google.internal",
}
_BLOCKED_HOST_SUFFIXES = (".internal", ".local", ".localdomain", ".localhost")


class OutboundRequestError(ValueError):
    """Base error for sanitized outbound request failures."""


class OutboundPolicyError(OutboundRequestError):
    """Raised when a URL or header violates the outbound network policy."""


class OutboundSizeLimitError(OutboundRequestError):
    """Raised when an encoded, transferred, or decoded body exceeds its limit."""

    def __init__(self, *, limit: int, actual: int | None = None) -> None:
        self.limit = limit
        self.actual = actual
        detail = f"limit={limit}"
        if actual is not None:
            detail += f", actual={actual}"
        super().__init__(f"Outbound content exceeds the configured size limit ({detail}).")


@dataclass(frozen=True)
class FetchMetadata:
    content_type: str | None
    final_url: str
    size: int
    sha256: str
    redirects: int


@dataclass(frozen=True)
class FetchResult:
    data: bytes
    metadata: FetchMetadata


@dataclass(frozen=True)
class _ResolvedAddress:
    family: int
    socket_type: int
    protocol: int
    ip: str
    sockaddr: tuple[Any, ...]


@dataclass(frozen=True)
class _ValidatedTarget:
    url: str
    hostname: str
    port: int
    addresses: tuple[_ResolvedAddress, ...]


class _Response(Protocol):
    headers: object

    def read(self, size: int = -1) -> bytes: ...

    def close(self) -> None: ...


Resolver = Callable[..., list[tuple[object, ...]]]
_SOCKET_DEFAULT_TIMEOUT = getattr(socket, "_GLOBAL_DEFAULT_TIMEOUT", object())


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(  # type: ignore[override]
        self,
        req: Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> None:
        return None


class _PinnedConnectionMixin:
    """Connect only to addresses retained by outbound policy validation."""

    def __init__(
        self,
        *args: Any,
        validated_addresses: tuple[_ResolvedAddress, ...],
        **kwargs: Any,
    ) -> None:
        self._validated_addresses = validated_addresses
        super().__init__(*args, **kwargs)
        self._create_connection = self._create_validated_connection

    def _create_validated_connection(
        self,
        _address: tuple[str, int],
        timeout: object = _SOCKET_DEFAULT_TIMEOUT,
        source_address: tuple[str, int] | None = None,
    ) -> socket.socket:
        last_error: OSError | None = None
        for candidate in self._validated_addresses:
            outbound_socket = socket.socket(
                candidate.family,
                candidate.socket_type,
                candidate.protocol,
            )
            try:
                if timeout is not _SOCKET_DEFAULT_TIMEOUT:
                    outbound_socket.settimeout(timeout)  # type: ignore[arg-type]
                if source_address is not None:
                    outbound_socket.bind(source_address)
                outbound_socket.connect(candidate.sockaddr)
                peer = outbound_socket.getpeername()
                peer_ip = str(peer[0])
                _validate_resolved_address(peer_ip)
                if ipaddress.ip_address(peer_ip) != ipaddress.ip_address(candidate.ip):
                    raise OutboundPolicyError(
                        "Outbound connection peer did not match the validated destination."
                    )
                return outbound_socket
            except OutboundRequestError:
                outbound_socket.close()
                raise
            except OSError as exc:
                outbound_socket.close()
                last_error = exc

        if last_error is not None:
            raise last_error
        raise OutboundPolicyError("Outbound destination has no validated public addresses.")


class _PinnedHTTPConnection(_PinnedConnectionMixin, http.client.HTTPConnection):
    pass


class _PinnedHTTPSConnection(_PinnedConnectionMixin, http.client.HTTPSConnection):
    pass


class _PinnedHTTPHandler(HTTPHandler):
    def __init__(self, addresses: tuple[_ResolvedAddress, ...]) -> None:
        super().__init__()
        self._addresses = addresses

    def http_open(self, request: Request) -> _Response:
        connection = partial(
            _PinnedHTTPConnection,
            validated_addresses=self._addresses,
        )
        return self.do_open(connection, request)  # type: ignore[return-value]


class _PinnedHTTPSHandler(HTTPSHandler):
    def __init__(self, addresses: tuple[_ResolvedAddress, ...]) -> None:
        super().__init__()
        self._addresses = addresses

    def https_open(self, request: Request) -> _Response:
        connection = partial(
            _PinnedHTTPSConnection,
            validated_addresses=self._addresses,
        )
        return self.do_open(  # type: ignore[return-value]
            connection,
            request,
            context=self._context,
        )


def _open_without_redirects(
    request: Request,
    *,
    timeout: float,
    target: _ValidatedTarget,
) -> _Response:
    if not target.addresses:
        raise OutboundPolicyError("Outbound destination has no validated public addresses.")
    if _origin(request.get_full_url()) != _origin(target.url):
        raise OutboundPolicyError("Outbound request does not match its validated destination.")
    return build_opener(
        ProxyHandler({}),
        _PinnedHTTPHandler(target.addresses),
        _PinnedHTTPSHandler(target.addresses),
        _NoRedirectHandler(),
    ).open(
        request,
        timeout=timeout,
    )


def _normalize_hostname(hostname: str) -> str:
    value = hostname.rstrip(".").lower()
    if not value or len(value) > 253:
        raise OutboundPolicyError("Outbound URL must contain a valid hostname.")
    if "%" in value:
        raise OutboundPolicyError("Scoped IPv6 destinations are not allowed.")
    try:
        return value.encode("idna").decode("ascii")
    except UnicodeError:
        raise OutboundPolicyError("Outbound URL must contain a valid hostname.") from None


def _normalize_allowed_hosts(allowed_hosts: Collection[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for raw_host in allowed_hosts:
        if not isinstance(raw_host, str):
            raise OutboundPolicyError("Outbound host allowlist entries must be strings.")
        host = raw_host.strip().lower().rstrip(".")
        if not host or "*" in host:
            raise OutboundPolicyError(
                "Outbound host allowlist entries must be exact hostnames only."
            )
        normalized.append(_normalize_hostname(host))
    return tuple(normalized)


def _host_is_allowed(hostname: str, allowed_hosts: tuple[str, ...]) -> bool:
    return hostname in allowed_hosts


def _validate_resolved_address(value: str) -> None:
    try:
        address = ipaddress.ip_address(value.split("%", 1)[0])
    except ValueError:
        raise OutboundPolicyError("Outbound destination resolved to an invalid address.") from None
    if (
        not address.is_global
        or address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_unspecified
        or address.is_reserved
    ):
        raise OutboundPolicyError("Outbound destination resolved to a non-public address.")


def _resolved_address(
    *,
    value: str,
    family: int,
    socket_type: int,
    protocol: int,
    port: int,
) -> _ResolvedAddress:
    _validate_resolved_address(value)
    parsed_address = ipaddress.ip_address(value.split("%", 1)[0])
    expected_family = socket.AF_INET6 if parsed_address.version == 6 else socket.AF_INET
    if (
        family != expected_family
        or socket_type != socket.SOCK_STREAM
        or protocol not in {0, socket.IPPROTO_TCP}
    ):
        raise OutboundPolicyError("Outbound destination returned an invalid DNS result.")
    normalized_address = str(parsed_address)
    if family == socket.AF_INET6:
        sockaddr: tuple[Any, ...] = (normalized_address, port, 0, 0)
    else:
        sockaddr = (normalized_address, port)
    return _ResolvedAddress(
        family=family,
        socket_type=socket_type,
        protocol=protocol,
        ip=normalized_address,
        sockaddr=sockaddr,
    )


def _validate_outbound_target(
    url: str,
    *,
    allowed_hosts: Collection[str],
    allowed_ports: Collection[int] | None = None,
    allow_insecure_http: bool = False,
    resolve_dns: bool = True,
    resolver: Resolver | None = None,
) -> _ValidatedTarget:
    """Validate a URL and retain the exact public addresses approved for connection."""

    if not isinstance(url, str):
        raise OutboundPolicyError("Outbound URL must be a string.")
    if not url or url != url.strip() or len(url) > MAX_URL_CHARS:
        raise OutboundPolicyError("Outbound URL is invalid or too long.")
    if any(ord(char) < 32 or ord(char) == 127 for char in url):
        raise OutboundPolicyError("Outbound URL contains prohibited characters.")

    try:
        parsed = urlsplit(url)
        port = parsed.port
        hostname = parsed.hostname
    except ValueError:
        raise OutboundPolicyError("Outbound URL is invalid.") from None

    scheme = parsed.scheme.lower()
    if scheme != "https" and not (allow_insecure_http and scheme == "http"):
        raise OutboundPolicyError(
            "Outbound URL must use HTTPS; development HTTP requires explicit opt-in."
        )
    if parsed.username is not None or parsed.password is not None:
        raise OutboundPolicyError("Credentials are not allowed in outbound URLs.")
    if parsed.fragment:
        raise OutboundPolicyError("Outbound URL fragments are not allowed.")
    if not hostname:
        raise OutboundPolicyError("Outbound URL must contain a hostname.")
    if port is not None and not (1 <= port <= 65535):
        raise OutboundPolicyError("Outbound URL contains an invalid port.")

    configured_ports = (443,) if allowed_ports is None else tuple(allowed_ports)
    if not configured_ports:
        raise OutboundPolicyError("Outbound port allowlist must not be empty.")
    if any(
        not isinstance(allowed_port, int) or not (1 <= allowed_port <= 65535)
        for allowed_port in configured_ports
    ):
        raise OutboundPolicyError("Outbound port allowlist contains an invalid entry.")
    effective_port = port or (443 if scheme == "https" else 80)
    if scheme == "http":
        if effective_port != 80:
            raise OutboundPolicyError("Development HTTP is restricted to the standard HTTP port.")
    elif effective_port == 80 or effective_port not in configured_ports:
        raise OutboundPolicyError("Outbound destination port is not allowlisted for HTTPS.")

    normalized_host = _normalize_hostname(hostname)
    if normalized_host in _BLOCKED_HOSTS or normalized_host.endswith(_BLOCKED_HOST_SUFFIXES):
        raise OutboundPolicyError("Outbound destination is prohibited.")

    normalized_allowlist = _normalize_allowed_hosts(allowed_hosts)
    if not normalized_allowlist or not _host_is_allowed(normalized_host, normalized_allowlist):
        raise OutboundPolicyError("Outbound destination is not allowlisted.")

    try:
        literal_address = ipaddress.ip_address(normalized_host)
    except ValueError:
        literal_address = None
    resolved_addresses: list[_ResolvedAddress] = []
    if literal_address is not None:
        resolved_addresses.append(
            _resolved_address(
                value=str(literal_address),
                family=(socket.AF_INET6 if literal_address.version == 6 else socket.AF_INET),
                socket_type=socket.SOCK_STREAM,
                protocol=socket.IPPROTO_TCP,
                port=effective_port,
            )
        )
    elif resolve_dns:
        resolve = resolver or socket.getaddrinfo
        try:
            answers = resolve(
                normalized_host,
                effective_port,
                type=socket.SOCK_STREAM,
            )
        except (OSError, socket.gaierror):
            raise OutboundPolicyError(
                "Outbound destination could not be resolved safely."
            ) from None
        if not answers:
            raise OutboundPolicyError("Outbound destination could not be resolved safely.")
        for answer in answers:
            try:
                family = int(answer[0])
                socket_type = int(answer[1])
                protocol = int(answer[2])
                sockaddr = answer[4]
                address = sockaddr[0]  # type: ignore[index]
            except (IndexError, TypeError, ValueError):
                raise OutboundPolicyError(
                    "Outbound destination returned an invalid DNS result."
                ) from None
            candidate = _resolved_address(
                value=str(address),
                family=family,
                socket_type=socket_type,
                protocol=protocol,
                port=effective_port,
            )
            if candidate not in resolved_addresses:
                resolved_addresses.append(candidate)

    try:
        is_ipv6 = isinstance(ipaddress.ip_address(normalized_host), ipaddress.IPv6Address)
    except ValueError:
        is_ipv6 = False
    netloc = f"[{normalized_host}]" if is_ipv6 else normalized_host
    if port is not None:
        netloc = f"{netloc}:{port}"
    normalized_url = urlunsplit((scheme, netloc, parsed.path or "/", parsed.query, ""))
    return _ValidatedTarget(
        url=normalized_url,
        hostname=normalized_host,
        port=effective_port,
        addresses=tuple(resolved_addresses),
    )


def validate_outbound_url(
    url: str,
    *,
    allowed_hosts: Collection[str],
    allowed_ports: Collection[int] | None = None,
    allow_insecure_http: bool = False,
    resolve_dns: bool = True,
    resolver: Resolver | None = None,
) -> str:
    """Validate and normalize an outbound HTTP URL without exposing it in errors."""

    return _validate_outbound_target(
        url,
        allowed_hosts=allowed_hosts,
        allowed_ports=allowed_ports,
        allow_insecure_http=allow_insecure_http,
        resolve_dns=resolve_dns,
        resolver=resolver,
    ).url


def sanitize_source_reference(url: str | None) -> str | None:
    """Return a non-secret source reference suitable for metadata and telemetry."""

    if not isinstance(url, str) or not url:
        return None
    try:
        parsed = urlsplit(url)
        scheme = parsed.scheme.lower()
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if scheme == "upload":
        upload_id = parsed.hostname or parsed.path.lstrip("/")
        return f"upload://{upload_id}" if upload_id else None
    if scheme not in {"http", "https"} or not hostname:
        return None
    try:
        normalized_host = _normalize_hostname(hostname)
    except OutboundPolicyError:
        return None
    try:
        is_ipv6 = isinstance(ipaddress.ip_address(normalized_host), ipaddress.IPv6Address)
    except ValueError:
        is_ipv6 = False
    netloc = f"[{normalized_host}]" if is_ipv6 else normalized_host
    if port is not None:
        netloc = f"{netloc}:{port}"
    return urlunsplit((scheme, netloc, "/", "", ""))


def validate_outbound_headers(
    headers: Mapping[str, str] | None,
) -> dict[str, str]:
    """Validate caller-supplied headers and return a copy safe for urllib."""

    if headers is None:
        return {}
    if not isinstance(headers, Mapping):
        raise OutboundPolicyError("Outbound headers must be a mapping.")
    if len(headers) > MAX_HEADER_COUNT:
        raise OutboundPolicyError("Outbound request contains too many headers.")

    normalized: dict[str, str] = {}
    seen_names: set[str] = set()
    total_chars = 0
    for raw_name, raw_value in headers.items():
        if not isinstance(raw_name, str) or not isinstance(raw_value, str):
            raise OutboundPolicyError("Outbound header names and values must be strings.")
        name = raw_name.strip()
        value = raw_value.strip()
        lower_name = name.lower()
        if not name or len(name) > MAX_HEADER_NAME_CHARS or not _HEADER_NAME_RE.fullmatch(name):
            raise OutboundPolicyError("Outbound request contains an invalid header name.")
        if lower_name in _BLOCKED_REQUEST_HEADERS:
            raise OutboundPolicyError("Outbound request contains a prohibited header.")
        if lower_name in seen_names:
            raise OutboundPolicyError("Outbound request contains duplicate headers.")
        if len(value) > MAX_HEADER_VALUE_CHARS or "\r" in value or "\n" in value or "\x00" in value:
            raise OutboundPolicyError("Outbound request contains an invalid header value.")
        total_chars += len(name) + len(value)
        if total_chars > MAX_HEADER_BLOCK_CHARS:
            raise OutboundPolicyError("Outbound request headers exceed the size limit.")
        seen_names.add(lower_name)
        normalized[name] = value
    return normalized


def decode_base64_bounded(value: str, *, max_bytes: int) -> bytes:
    """Strictly decode base64 after an encoded-size preflight and decoded-size check."""

    if not isinstance(value, str) or not value.strip():
        raise OutboundPolicyError("Base64 content must be a non-empty string.")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive.")

    payload = value.strip()
    if payload.lower().startswith("data:"):
        prefix, separator, encoded = payload.partition(",")
        if not separator or ";base64" not in prefix.lower():
            raise OutboundPolicyError("Base64 data URI is invalid.")
        if len(prefix) > 512:
            raise OutboundPolicyError("Base64 data URI metadata is too long.")
        payload = encoded
    elif "base64," in payload:
        payload = payload.split("base64,", 1)[1]

    encoded_limit = ((max_bytes + 2) // 3) * 4
    if len(payload) > encoded_limit + 1024:
        estimated = (len(payload) * 3) // 4
        raise OutboundSizeLimitError(limit=max_bytes, actual=estimated)
    compact_payload = "".join(payload.split())
    if len(compact_payload) > encoded_limit:
        estimated = (len(compact_payload) * 3) // 4
        raise OutboundSizeLimitError(limit=max_bytes, actual=estimated)
    try:
        decoded = base64.b64decode(compact_payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise OutboundPolicyError("Base64 content is invalid.") from exc
    if len(decoded) > max_bytes:
        raise OutboundSizeLimitError(limit=max_bytes, actual=len(decoded))
    return decoded


def _response_header(headers: object, name: str) -> str | None:
    getter = getattr(headers, "get", None)
    if callable(getter):
        value = getter(name)
        if value is not None:
            return str(value).strip()
    items = getattr(headers, "items", None)
    if callable(items):
        for key, value in items():
            if str(key).lower() == name.lower():
                return str(value).strip()
    return None


def _response_content_type(headers: object) -> str | None:
    value = _response_header(headers, "Content-Type")
    if value is None or len(value) > MAX_CONTENT_TYPE_CHARS:
        return None
    if any(ord(character) < 32 or ord(character) > 126 for character in value):
        return None
    return value


def _response_status(response: _Response) -> int:
    status = getattr(response, "status", None)
    if status is None:
        getter = getattr(response, "getcode", None)
        status = getter() if callable(getter) else 200
    try:
        return int(status)
    except (TypeError, ValueError):
        return 200


def _origin(url: str) -> tuple[str, str, int]:
    parsed = urlsplit(url)
    scheme = parsed.scheme.lower()
    host = _normalize_hostname(parsed.hostname or "")
    port = parsed.port or (443 if scheme == "https" else 80)
    return scheme, host, port


def _redirect_headers(
    headers: Mapping[str, str], *, current_url: str, next_url: str
) -> dict[str, str]:
    if _origin(current_url) == _origin(next_url):
        return dict(headers)
    return {
        name: value
        for name, value in headers.items()
        if name.lower() in _CROSS_ORIGIN_REDIRECT_HEADERS
    }


def _validate_size_settings(
    *, max_bytes: int, timeout_seconds: float, max_redirects: int, chunk_bytes: int
) -> None:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive.")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive.")
    if max_redirects < 0:
        raise ValueError("max_redirects must be non-negative.")
    if chunk_bytes <= 0:
        raise ValueError("chunk_bytes must be positive.")


def _stream_response(
    response: _Response,
    *,
    destination: BinaryIO,
    max_bytes: int,
    chunk_bytes: int,
) -> tuple[int, str]:
    content_length = _response_header(response.headers, "Content-Length")
    if content_length:
        try:
            declared_size = int(content_length)
        except ValueError:
            raise OutboundRequestError(
                "Outbound response contains an invalid Content-Length."
            ) from None
        if declared_size < 0:
            raise OutboundRequestError("Outbound response contains an invalid Content-Length.")
        if declared_size > max_bytes:
            raise OutboundSizeLimitError(limit=max_bytes, actual=declared_size)

    encoding = _response_header(response.headers, "Content-Encoding") or "identity"
    encoding = encoding.lower().strip()
    if encoding in {"", "identity"}:
        decompressor = None
    elif encoding in {"gzip", "x-gzip"}:
        decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    elif encoding == "deflate":
        decompressor = zlib.decompressobj()
    else:
        raise OutboundRequestError("Outbound response uses an unsupported encoding.")

    encoded_total = 0
    decoded_total = 0
    digest = hashlib.sha256()
    try:
        while True:
            raw_chunk = response.read(chunk_bytes)
            if not raw_chunk:
                break
            encoded_total += len(raw_chunk)
            if encoded_total > max_bytes:
                raise OutboundSizeLimitError(limit=max_bytes, actual=encoded_total)
            if decompressor is None:
                decoded_chunk = raw_chunk
            else:
                remaining = max_bytes - decoded_total
                decoded_chunk = decompressor.decompress(raw_chunk, remaining + 1)
                if decompressor.unconsumed_tail:
                    raise OutboundSizeLimitError(limit=max_bytes, actual=max_bytes + 1)
            decoded_total += len(decoded_chunk)
            if decoded_total > max_bytes:
                raise OutboundSizeLimitError(limit=max_bytes, actual=decoded_total)
            destination.write(decoded_chunk)
            digest.update(decoded_chunk)

        if decompressor is not None:
            if not decompressor.eof or decompressor.unused_data:
                raise OutboundRequestError(
                    "Outbound response contains incomplete or trailing compressed content."
                )
            remaining = max_bytes - decoded_total
            final_chunk = decompressor.flush(remaining + 1)
            decoded_total += len(final_chunk)
            if decoded_total > max_bytes:
                raise OutboundSizeLimitError(limit=max_bytes, actual=decoded_total)
            destination.write(final_chunk)
            digest.update(final_chunk)
    except zlib.error:
        raise OutboundRequestError(
            "Outbound response contains invalid compressed content."
        ) from None
    return decoded_total, digest.hexdigest()


def _fetch_to_destination(
    url: str,
    *,
    destination: BinaryIO,
    headers: Mapping[str, str] | None,
    allowed_hosts: Collection[str],
    allowed_ports: Collection[int] | None,
    allow_insecure_http: bool,
    max_bytes: int,
    timeout_seconds: float,
    max_redirects: int,
    chunk_bytes: int,
    resolver: Resolver | None,
) -> FetchMetadata:
    _validate_size_settings(
        max_bytes=max_bytes,
        timeout_seconds=timeout_seconds,
        max_redirects=max_redirects,
        chunk_bytes=chunk_bytes,
    )
    current_target = _validate_outbound_target(
        url,
        allowed_hosts=allowed_hosts,
        allowed_ports=allowed_ports,
        allow_insecure_http=allow_insecure_http,
        resolver=resolver,
    )
    current_url = current_target.url
    current_headers = validate_outbound_headers(headers)
    if not any(name.lower() == "accept-encoding" for name in current_headers):
        current_headers["Accept-Encoding"] = "identity"

    for redirect_count in range(max_redirects + 1):
        request = Request(current_url, headers=current_headers, method="GET")
        response: _Response | None = None
        try:
            try:
                response = _open_without_redirects(
                    request,
                    timeout=timeout_seconds,
                    target=current_target,
                )
            except HTTPError as exc:
                if 300 <= exc.code < 400:
                    response = exc
                else:
                    raise OutboundRequestError(
                        f"Outbound server returned HTTP status {exc.code}."
                    ) from None
            except (TimeoutError, socket.timeout):
                raise OutboundRequestError("Outbound request timed out.") from None
            except (URLError, OSError):
                raise OutboundRequestError("Outbound request could not be completed.") from None
            except OutboundRequestError:
                raise
            except Exception:
                raise OutboundRequestError("Outbound request could not be completed.") from None

            status = _response_status(response)
            if 300 <= status < 400:
                location = _response_header(response.headers, "Location")
                if not location:
                    raise OutboundRequestError("Outbound redirect did not include a destination.")
                if redirect_count >= max_redirects:
                    raise OutboundRequestError("Outbound request exceeded the redirect limit.")
                candidate_url = urljoin(current_url, location)
                next_target = _validate_outbound_target(
                    candidate_url,
                    allowed_hosts=allowed_hosts,
                    allowed_ports=allowed_ports,
                    allow_insecure_http=allow_insecure_http,
                    resolver=resolver,
                )
                next_url = next_target.url
                current_headers = _redirect_headers(
                    current_headers,
                    current_url=current_url,
                    next_url=next_url,
                )
                current_target = next_target
                current_url = next_url
                continue
            if status >= 400:
                raise OutboundRequestError(f"Outbound server returned HTTP status {status}.")

            try:
                size, digest = _stream_response(
                    response,
                    destination=destination,
                    max_bytes=max_bytes,
                    chunk_bytes=chunk_bytes,
                )
            except OutboundRequestError:
                raise
            except (TimeoutError, socket.timeout):
                raise OutboundRequestError("Outbound response timed out while reading.") from None
            except Exception:
                raise OutboundRequestError("Outbound response could not be read safely.") from None
            return FetchMetadata(
                content_type=_response_content_type(response.headers),
                final_url=sanitize_source_reference(current_url) or "",
                size=size,
                sha256=digest,
                redirects=redirect_count,
            )
        finally:
            if response is not None:
                close = getattr(response, "close", None)
                if callable(close):
                    close()

    raise OutboundRequestError("Outbound request exceeded the redirect limit.")


def fetch_bytes(
    url: str,
    *,
    headers: Mapping[str, str] | None = None,
    allowed_hosts: Collection[str],
    allowed_ports: Collection[int] | None = None,
    allow_insecure_http: bool = False,
    max_bytes: int,
    timeout_seconds: float = DEFAULT_FETCH_TIMEOUT_SECONDS,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    chunk_bytes: int = DEFAULT_STREAM_CHUNK_BYTES,
    resolver: Resolver | None = None,
) -> FetchResult:
    """Fetch a URL into memory while enforcing URL, redirect, and size policy."""

    destination = io.BytesIO()
    metadata = _fetch_to_destination(
        url,
        destination=destination,
        headers=headers,
        allowed_hosts=allowed_hosts,
        allowed_ports=allowed_ports,
        allow_insecure_http=allow_insecure_http,
        max_bytes=max_bytes,
        timeout_seconds=timeout_seconds,
        max_redirects=max_redirects,
        chunk_bytes=chunk_bytes,
        resolver=resolver,
    )
    return FetchResult(data=destination.getvalue(), metadata=metadata)


def fetch_to_path(
    url: str,
    path: str | Path,
    *,
    headers: Mapping[str, str] | None = None,
    allowed_hosts: Collection[str],
    allowed_ports: Collection[int] | None = None,
    allow_insecure_http: bool = False,
    max_bytes: int,
    timeout_seconds: float = DEFAULT_FETCH_TIMEOUT_SECONDS,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    chunk_bytes: int = DEFAULT_STREAM_CHUNK_BYTES,
    resolver: Resolver | None = None,
) -> FetchMetadata:
    """Fetch a URL to a caller-owned path while enforcing the outbound policy."""

    try:
        destination_path = Path(path)
    except (OSError, TypeError, ValueError):
        raise OutboundRequestError(
            "Outbound destination file could not be prepared safely."
        ) from None
    temporary_path: Path | None = None

    def remove_partial_file() -> None:
        if temporary_path is None:
            return
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            return

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination_path.parent,
            prefix=f".{destination_path.name}.",
            suffix=".part",
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            metadata = _fetch_to_destination(
                url,
                destination=destination,
                headers=headers,
                allowed_hosts=allowed_hosts,
                allowed_ports=allowed_ports,
                allow_insecure_http=allow_insecure_http,
                max_bytes=max_bytes,
                timeout_seconds=timeout_seconds,
                max_redirects=max_redirects,
                chunk_bytes=chunk_bytes,
                resolver=resolver,
            )
            destination.flush()
        temporary_path.replace(destination_path)
        return metadata
    except OutboundRequestError:
        remove_partial_file()
        raise
    except OSError:
        remove_partial_file()
        raise OutboundRequestError(
            "Outbound destination file could not be written safely."
        ) from None
    except Exception:
        remove_partial_file()
        raise
