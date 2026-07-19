"""Fail-closed HTTP access controls for the Qdrant MCP runtime."""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import re
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal
from urllib.parse import urlsplit

from starlette.responses import JSONResponse

VALID_MCP_MODES = frozenset({"portal", "standalone"})
DEFAULT_PORTAL_GRANT_HEADER = "X-MADPANDA-PORTAL-GRANT"
DEFAULT_TENANT_ID_HEADER = "X-MADPANDA-USER-ID"
DEFAULT_REQUEST_BODY_MAX_BYTES = 1_048_576
DEFAULT_REQUEST_BODY_TIMEOUT_SECONDS = 10
DEFAULT_ALLOWED_HOSTS = (
    "localhost:*",
    "127.0.0.1:*",
    "[::1]:*",
    "qdrant-mcp:*",
    "mcp-qdrant:*",
    "qdrant-mcp-standalone:*",
    "qdrant-mcp-portal:*",
)
MIN_SERVICE_TOKEN_LENGTH = 32
MAX_SERVICE_TOKEN_LENGTH = 512
MAX_REQUEST_BODY_MAX_BYTES = 16_777_216
MAX_REQUEST_HEADER_COUNT = 64
MAX_REQUEST_HEADER_BYTES = 32_768
MAX_TENANT_ID_LENGTH = 128
_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_SERVICE_TOKEN = re.compile(r"^[\x21-\x7e]+$")
_TENANT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$")
_DNS_HOST = re.compile(
    r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)*"
    r"[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$"
)
_SINGLETON_SECURITY_HEADERS = frozenset(
    {
        "authorization",
        "content-length",
        "host",
        "origin",
        "transfer-encoding",
    }
)


@dataclass(frozen=True)
class RequestIdentity:
    """Authenticated request identity without any credential material."""

    kind: Literal["standalone", "portal"]
    subject: str = field(repr=False)


_request_identity: ContextVar[RequestIdentity | None] = ContextVar(
    "qdrant_mcp_request_identity",
    default=None,
)


class RuntimeConfigurationError(RuntimeError):
    """Raised when the selected HTTP access boundary is incomplete."""


@dataclass(frozen=True)
class RuntimeSecurityConfig:
    """Immutable startup-selected service access policy."""

    mode: str
    standalone_access_token: str = field(repr=False)
    portal_grant_token: str = field(repr=False)
    portal_grant_header: str
    tenant_id_header: str
    request_body_max_bytes: int
    request_body_timeout_seconds: int
    allowed_hosts: tuple[str, ...]
    allowed_origins: tuple[str, ...]


def get_request_identity() -> RequestIdentity | None:
    """Return the typed authenticated request identity, if one is active."""

    return _request_identity.get()


def get_request_principal() -> str | None:
    """Return the authenticated request principal without exposing credentials.

    Portal requests return the validated tenant identifier. Standalone HTTP
    requests return the constant ``standalone``. Local stdio calls and code
    outside an authenticated request return ``None``.
    """

    identity = get_request_identity()
    return identity.subject if identity is not None else None


def _required_service_token(environment: Mapping[str, str], name: str) -> str:
    value = str(environment.get(name, "") or "")
    if (
        len(value) < MIN_SERVICE_TOKEN_LENGTH
        or len(value) > MAX_SERVICE_TOKEN_LENGTH
        or not _SERVICE_TOKEN.fullmatch(value)
    ):
        raise RuntimeConfigurationError(
            f"{name} must contain {MIN_SERVICE_TOKEN_LENGTH}-{MAX_SERVICE_TOKEN_LENGTH} "
            "visible ASCII characters without whitespace."
        )
    return value


def _parse_positive_int(
    raw: str,
    *,
    name: str,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    value = str(raw or "").strip()
    if not value:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeConfigurationError(f"{name} must be an integer.") from exc
    if parsed < minimum or parsed > maximum:
        raise RuntimeConfigurationError(f"{name} must be between {minimum} and {maximum}.")
    return parsed


def _normalize_origin(candidate: str) -> str:
    if candidate != candidate.strip():
        raise RuntimeConfigurationError("Origins must not contain leading or trailing whitespace.")
    try:
        parsed = urlsplit(candidate)
        port = parsed.port
    except ValueError as exc:
        raise RuntimeConfigurationError("Origin contains an invalid port.") from exc
    scheme = parsed.scheme.lower()
    hostname = parsed.hostname
    if (
        scheme not in {"http", "https"}
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise RuntimeConfigurationError(
            "Origins must be exact HTTP(S) origins without credentials, paths, queries, or fragments."
        )
    host = hostname.lower()
    if ":" in host:
        host = f"[{host}]"
    default_port = 80 if scheme == "http" else 443
    port_suffix = f":{port}" if port is not None and port != default_port else ""
    return f"{scheme}://{host}{port_suffix}"


def _parse_allowed_origins(raw: str) -> tuple[str, ...]:
    origins: list[str] = []
    for candidate in str(raw or "").split(","):
        origin = candidate.strip()
        if not origin:
            continue
        if origin == "*":
            raise RuntimeConfigurationError(
                "MCP_ALLOWED_ORIGINS must contain exact HTTP(S) origins; wildcards are forbidden."
            )
        origins.append(_normalize_origin(origin))
    return tuple(dict.fromkeys(origins))


def _normalize_allowed_host(candidate: str) -> str:
    host = candidate.strip().lower()
    if not host or any(character.isspace() for character in host):
        raise RuntimeConfigurationError(
            "MCP_ALLOWED_HOSTS must contain exact Host values or host:* port patterns."
        )
    hostname: str
    port: str | None
    if host.startswith("["):
        closing = host.find("]")
        if closing < 0:
            raise RuntimeConfigurationError(
                "MCP_ALLOWED_HOSTS IPv6 literals must use bracket notation."
            )
        hostname = host[1:closing]
        suffix = host[closing + 1 :]
        if suffix and not suffix.startswith(":"):
            raise RuntimeConfigurationError(
                "MCP_ALLOWED_HOSTS must contain exact Host values or host:* port patterns."
            )
        port = suffix[1:] if suffix else None
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError as exc:
            raise RuntimeConfigurationError(
                "MCP_ALLOWED_HOSTS contains an invalid IPv6 literal."
            ) from exc
        if not isinstance(address, ipaddress.IPv6Address):
            raise RuntimeConfigurationError(
                "MCP_ALLOWED_HOSTS bracket notation is reserved for IPv6 literals."
            )
        normalized_hostname = f"[{address.compressed}]"
    else:
        if host.count(":") > 1:
            raise RuntimeConfigurationError(
                "MCP_ALLOWED_HOSTS IPv6 literals must use bracket notation."
            )
        hostname, separator, port = host.rpartition(":")
        if not separator:
            hostname = host
            port = None
        elif not hostname:
            raise RuntimeConfigurationError(
                "MCP_ALLOWED_HOSTS must contain exact Host values or host:* port patterns."
            )
        try:
            address = ipaddress.ip_address(hostname)
        except ValueError:
            if not _DNS_HOST.fullmatch(hostname):
                raise RuntimeConfigurationError("MCP_ALLOWED_HOSTS contains an invalid hostname.")
            normalized_hostname = hostname
        else:
            normalized_hostname = address.compressed
    if port is not None and port != "*":
        if not port.isdecimal() or not 1 <= int(port) <= 65_535:
            raise RuntimeConfigurationError("MCP_ALLOWED_HOSTS contains an invalid port.")
    if "*" in hostname:
        raise RuntimeConfigurationError(
            "MCP_ALLOWED_HOSTS must contain exact Host values or host:* port patterns."
        )
    return normalized_hostname + (f":{port}" if port is not None else "")


def _parse_allowed_hosts(raw: str) -> tuple[str, ...]:
    candidates = str(raw or "").split(",")
    hosts = tuple(
        dict.fromkeys(
            _normalize_allowed_host(candidate) for candidate in candidates if candidate.strip()
        )
    )
    return hosts or DEFAULT_ALLOWED_HOSTS


def _validate_header_name(value: str, *, variable: str) -> str:
    header = str(value or "").strip().lower()
    if not _HEADER_NAME.fullmatch(header):
        raise RuntimeConfigurationError(f"{variable} is not a valid HTTP header name.")
    return header


def load_runtime_security_config(
    environment: Mapping[str, str],
) -> RuntimeSecurityConfig:
    """Load and validate one authenticated HTTP runtime mode at startup."""

    mode = str(environment.get("MCP_MODE", "") or "").strip().lower()
    if mode not in VALID_MCP_MODES:
        choices = ", ".join(sorted(VALID_MCP_MODES))
        raise RuntimeConfigurationError(f"MCP_MODE must be exactly one of: {choices}.")

    portal_grant_header = _validate_header_name(
        str(
            environment.get("MCP_PORTAL_GRANT_HEADER", DEFAULT_PORTAL_GRANT_HEADER)
            or DEFAULT_PORTAL_GRANT_HEADER
        ),
        variable="MCP_PORTAL_GRANT_HEADER",
    )
    tenant_id_header = _validate_header_name(
        str(
            environment.get("MCP_TENANT_ID_HEADER", DEFAULT_TENANT_ID_HEADER)
            or DEFAULT_TENANT_ID_HEADER
        ),
        variable="MCP_TENANT_ID_HEADER",
    )
    reserved = set(_SINGLETON_SECURITY_HEADERS)
    if portal_grant_header in reserved:
        raise RuntimeConfigurationError(
            "MCP_PORTAL_GRANT_HEADER must not reuse a reserved HTTP security header."
        )
    reserved.add(portal_grant_header)
    if tenant_id_header in reserved:
        raise RuntimeConfigurationError(
            "MCP_TENANT_ID_HEADER must use a unique, non-reserved HTTP security header."
        )

    standalone_access_token = ""
    portal_grant_token = ""
    if mode == "standalone":
        standalone_access_token = _required_service_token(environment, "MCP_ACCESS_TOKEN")
    else:
        portal_grant_token = _required_service_token(environment, "MCP_PORTAL_GRANT_TOKEN")

    return RuntimeSecurityConfig(
        mode=mode,
        standalone_access_token=standalone_access_token,
        portal_grant_token=portal_grant_token,
        portal_grant_header=portal_grant_header,
        tenant_id_header=tenant_id_header,
        request_body_max_bytes=_parse_positive_int(
            str(environment.get("MCP_REQUEST_BODY_MAX_BYTES", "") or ""),
            name="MCP_REQUEST_BODY_MAX_BYTES",
            default=DEFAULT_REQUEST_BODY_MAX_BYTES,
            minimum=1_024,
            maximum=MAX_REQUEST_BODY_MAX_BYTES,
        ),
        request_body_timeout_seconds=_parse_positive_int(
            str(environment.get("MCP_REQUEST_BODY_TIMEOUT_SECONDS", "") or ""),
            name="MCP_REQUEST_BODY_TIMEOUT_SECONDS",
            default=DEFAULT_REQUEST_BODY_TIMEOUT_SECONDS,
            minimum=1,
            maximum=60,
        ),
        allowed_hosts=_parse_allowed_hosts(str(environment.get("MCP_ALLOWED_HOSTS", "") or "")),
        allowed_origins=_parse_allowed_origins(
            str(environment.get("MCP_ALLOWED_ORIGINS", "") or "")
        ),
    )


def validate_request_header_configuration(
    config: RuntimeSecurityConfig,
    configured_headers: Mapping[str, str],
) -> tuple[str, ...]:
    """Return normalized provider headers or fail on invalid/colliding names."""

    seen = set(_SINGLETON_SECURITY_HEADERS)
    seen.update({config.portal_grant_header, config.tenant_id_header})
    normalized_headers: list[str] = []
    for variable, configured_header in configured_headers.items():
        header = _validate_header_name(configured_header, variable=variable)
        if header in seen:
            raise RuntimeConfigurationError(
                f"{variable} must use a unique, non-reserved HTTP security header."
            )
        seen.add(header)
        normalized_headers.append(header)
    return tuple(normalized_headers)


def _normalized_headers(
    scope: Mapping[str, Any],
    config: RuntimeSecurityConfig,
    additional_singleton_headers: frozenset[str],
) -> tuple[dict[str, str], set[str]]:
    normalized: dict[str, str] = {}
    duplicates: set[str] = set()
    singleton_headers = (
        _SINGLETON_SECURITY_HEADERS
        | {config.portal_grant_header, config.tenant_id_header}
        | additional_singleton_headers
    )
    for key, value in scope.get("headers", []):
        name = key.decode("latin-1").lower()
        if name in normalized and name in singleton_headers:
            duplicates.add(name)
        normalized[name] = value.decode("latin-1")
    return normalized, duplicates


def _validate_bearer(authorization: str, expected: str) -> str | None:
    if not authorization:
        return "missing_access_token"
    scheme, separator, provided = authorization.partition(" ")
    if separator != " " or scheme.lower() != "bearer" or not provided or " " in provided:
        return "invalid_access_token"
    if not hmac.compare_digest(provided, expected):
        return "invalid_access_token"
    return None


def _validate_tenant_id(value: str) -> str | None:
    if not value:
        return "missing_tenant_id"
    if len(value) > MAX_TENANT_ID_LENGTH or not _TENANT_ID.fullmatch(value):
        return "invalid_tenant_id"
    return None


def validate_service_access(
    headers: Mapping[str, str],
    config: RuntimeSecurityConfig,
) -> tuple[str | None, str | None]:
    """Return ``(error_code, principal)`` without reflecting credentials."""

    if config.mode == "standalone":
        error = _validate_bearer(
            headers.get("authorization", ""),
            config.standalone_access_token,
        )
        return error, "standalone" if error is None else None

    provided = headers.get(config.portal_grant_header, "")
    if not provided:
        return "missing_portal_grant", None
    if not hmac.compare_digest(provided, config.portal_grant_token):
        return "invalid_portal_grant", None
    tenant_id = headers.get(config.tenant_id_header, "")
    tenant_error = _validate_tenant_id(tenant_id)
    if tenant_error is not None:
        return tenant_error, None
    return None, tenant_id


def _host_allowed(host: str, allowed_hosts: tuple[str, ...]) -> bool:
    if not host or host != host.strip() or host.endswith(":*"):
        return False
    try:
        normalized = _normalize_allowed_host(host)
    except RuntimeConfigurationError:
        return False
    if normalized in allowed_hosts:
        return True
    return any(
        allowed.endswith(":*") and normalized.startswith(f"{allowed[:-2]}:")
        for allowed in allowed_hosts
    )


def _error_response(code: str, status_code: int) -> JSONResponse:
    return JSONResponse(
        {
            "ok": False,
            "error": {
                "type": ("permission_denied" if status_code in {401, 403} else "invalid_request"),
                "code": code,
                "message": "Qdrant MCP request rejected by the service access boundary.",
            },
        },
        status_code=status_code,
        headers={
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
        },
    )


async def _read_bounded_body(receive: Any, maximum: int) -> list[dict[str, Any]] | None:
    messages: list[dict[str, Any]] = []
    total = 0
    while True:
        message = await receive()
        messages.append(message)
        if message.get("type") != "http.request":
            return messages
        total += len(message.get("body", b""))
        if total > maximum:
            return None
        if not message.get("more_body", False):
            return messages


def _replay_receive(messages: list[dict[str, Any]], original_receive: Any) -> Any:
    pending = list(messages)

    async def replay() -> dict[str, Any]:
        if pending:
            return pending.pop(0)
        return await original_receive()

    return replay


class AccessControlMiddleware:
    """Authenticate and bound every non-health HTTP request before parsing."""

    def __init__(
        self,
        app: Any,
        config: RuntimeSecurityConfig,
        singleton_headers: tuple[str, ...] = (),
    ) -> None:
        self.app = app
        self.config = config
        normalized_singletons = frozenset(
            str(header or "").strip().lower() for header in singleton_headers
        )
        if any(not _HEADER_NAME.fullmatch(header) for header in normalized_singletons):
            raise RuntimeConfigurationError(
                "Request credential and policy header names must be valid HTTP headers."
            )
        self.singleton_headers = normalized_singletons

    def __getattr__(self, name: str) -> Any:
        """Preserve FastMCP app attributes such as ``state`` and lifespan hooks."""

        return getattr(self.app, name)

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        public_health = (
            str(scope.get("path", "")) == "/health"
            and str(scope.get("method", "")).upper() == "GET"
        )

        raw_headers = scope.get("headers", [])
        if (
            not isinstance(raw_headers, (list, tuple))
            or len(raw_headers) > MAX_REQUEST_HEADER_COUNT
        ):
            await _error_response("request_headers_too_large", 431)(
                scope,
                receive,
                send,
            )
            return
        header_bytes = 0
        for raw_name, raw_value in raw_headers:
            header_bytes += len(raw_name) + len(raw_value)
            if header_bytes > MAX_REQUEST_HEADER_BYTES:
                await _error_response("request_headers_too_large", 431)(
                    scope,
                    receive,
                    send,
                )
                return

        headers, duplicate_headers = _normalized_headers(
            scope,
            self.config,
            self.singleton_headers,
        )
        if duplicate_headers:
            await _error_response("duplicate_security_header", 400)(scope, receive, send)
            return
        if headers.get("content-length") and headers.get("transfer-encoding"):
            await _error_response("ambiguous_request_framing", 400)(scope, receive, send)
            return

        principal: str | None = None
        if not public_health:
            access_error, principal = validate_service_access(headers, self.config)
            if access_error is not None or principal is None:
                await _error_response(access_error or "invalid_access_token", 401)(
                    scope,
                    receive,
                    send,
                )
                return

        if not _host_allowed(headers.get("host", ""), self.config.allowed_hosts):
            await _error_response("host_not_allowed", 421)(scope, receive, send)
            return

        origin = headers.get("origin", "")
        if origin:
            try:
                normalized_origin = _normalize_origin(origin)
            except RuntimeConfigurationError:
                normalized_origin = ""
            if normalized_origin not in self.config.allowed_origins:
                await _error_response("origin_not_allowed", 403)(scope, receive, send)
                return

        raw_content_length = headers.get("content-length", "")
        if raw_content_length:
            if not raw_content_length.isdecimal():
                await _error_response("invalid_content_length", 400)(scope, receive, send)
                return
            content_length = int(raw_content_length)
            if content_length > self.config.request_body_max_bytes:
                await _error_response("request_body_too_large", 413)(scope, receive, send)
                return

        try:
            async with asyncio.timeout(self.config.request_body_timeout_seconds):
                buffered = await _read_bounded_body(
                    receive,
                    self.config.request_body_max_bytes,
                )
        except TimeoutError:
            await _error_response("request_body_timeout", 408)(scope, receive, send)
            return
        if buffered is None:
            await _error_response("request_body_too_large", 413)(scope, receive, send)
            return
        receive = _replay_receive(buffered, receive)

        if public_health:
            await self.app(scope, receive, send)
            return

        identity_kind: Literal["standalone", "portal"] = (
            "portal" if self.config.mode == "portal" else "standalone"
        )
        identity_token = _request_identity.set(
            RequestIdentity(kind=identity_kind, subject=principal)
        )
        try:
            await self.app(scope, receive, send)
        finally:
            _request_identity.reset(identity_token)
