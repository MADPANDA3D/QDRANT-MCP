from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

import pytest
from fastmcp import FastMCP
from httpx import Headers
from qdrant_client.http.exceptions import UnexpectedResponse
from starlette.testclient import TestClient

from mcp_server_qdrant import __version__
from mcp_server_qdrant.embeddings.openai import OpenAIProvider
from mcp_server_qdrant.hosted_server import HostedQdrantMCPServer
from mcp_server_qdrant.qdrant import QdrantConnector
from mcp_server_qdrant.runtime_security import (
    AccessControlMiddleware,
    RequestIdentity,
    RuntimeConfigurationError,
    get_request_identity,
    get_request_principal,
    load_runtime_security_config,
    validate_request_header_configuration,
)
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    MemorySettings,
    QdrantSettings,
    RequestOverrideSettings,
    ToolSettings,
)
from tests.test_collection_management import DummyEmbeddingProvider

ACCESS_TOKEN = "standalone-access-token-0000000000000001"
PORTAL_GRANT = "portal-grant-token-000000000000000001"


class ConfiguredFastEmbedProvider(DummyEmbeddingProvider):
    provider_type = "fastembed"
    model_name = "configured/fastembed-model"
    version = "configured/fastembed-model"


class RecordingApp:
    def __init__(self) -> None:
        self.calls = 0
        self.body = b""
        self.principals: list[str | None] = []
        self.identities: list[RequestIdentity | None] = []

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        self.calls += 1
        self.principals.append(get_request_principal())
        self.identities.append(get_request_identity())
        if scope.get("type") == "http":
            while True:
                message = await receive()
                if message.get("type") != "http.request":
                    break
                self.body += message.get("body", b"")
                if not message.get("more_body", False):
                    break
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [(b"content-type", b"application/json")],
            }
        )
        await send({"type": "http.response.body", "body": b'{"ok":true}'})


def security_config(mode: str = "standalone", **overrides: str):
    environment = {
        "MCP_MODE": mode,
        "MCP_ACCESS_TOKEN": ACCESS_TOKEN,
        "MCP_PORTAL_GRANT_TOKEN": PORTAL_GRANT,
        "MCP_REQUEST_BODY_MAX_BYTES": "1024",
        "MCP_ALLOWED_HOSTS": "localhost:*,[::1]:*,qdrant-mcp-portal:*",
    }
    environment.update(overrides)
    return load_runtime_security_config(environment)


async def invoke(
    app: Any,
    *,
    headers: Mapping[str, str] | None = None,
    extra_headers: list[tuple[bytes, bytes]] | None = None,
    chunks: list[tuple[bytes, bool]] | None = None,
    path: str = "/mcp",
    method: str = "POST",
) -> tuple[int, dict[str, Any], int]:
    request_headers = {"host": "localhost:8085", **(headers or {})}
    pending = list(chunks or [(b"", False)])
    receive_calls = 0
    sent: list[dict[str, Any]] = []

    async def receive() -> dict[str, Any]:
        nonlocal receive_calls
        receive_calls += 1
        if pending:
            body, more_body = pending.pop(0)
            return {
                "type": "http.request",
                "body": body,
                "more_body": more_body,
            }
        return {"type": "http.disconnect"}

    async def send(message: dict[str, Any]) -> None:
        sent.append(message)

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "scheme": "http",
            "method": method,
            "path": path,
            "raw_path": path.encode("ascii"),
            "query_string": b"",
            "server": ("localhost", 8085),
            "client": ("127.0.0.1", 50000),
            "headers": [
                (key.lower().encode("latin-1"), value.encode("latin-1"))
                for key, value in request_headers.items()
            ]
            + list(extra_headers or []),
        },
        receive,
        send,
    )
    status = next(message["status"] for message in sent if message["type"] == "http.response.start")
    raw_body = b"".join(
        message.get("body", b"") for message in sent if message["type"] == "http.response.body"
    )
    return status, json.loads(raw_body or b"{}"), receive_calls


@pytest.mark.parametrize("mode", ["", "public", "portal typo"])
def test_http_mode_is_startup_selected_and_fail_closed(mode: str) -> None:
    with pytest.raises(RuntimeConfigurationError, match="MCP_MODE"):
        load_runtime_security_config({"MCP_MODE": mode})


@pytest.mark.parametrize(
    "mode,variable,value",
    [
        ("standalone", "MCP_ACCESS_TOKEN", "too-short"),
        ("standalone", "MCP_ACCESS_TOKEN", f" {ACCESS_TOKEN}"),
        ("standalone", "MCP_ACCESS_TOKEN", ACCESS_TOKEN + " "),
        ("portal", "MCP_PORTAL_GRANT_TOKEN", "x" * 513),
        ("portal", "MCP_PORTAL_GRANT_TOKEN", "x" * 31 + "\n"),
    ],
)
def test_selected_service_token_is_visible_ascii_without_whitespace(
    mode: str,
    variable: str,
    value: str,
) -> None:
    environment = {"MCP_MODE": mode, variable: value}
    with pytest.raises(RuntimeConfigurationError, match=variable):
        load_runtime_security_config(environment)


@pytest.mark.parametrize(
    "allowed_hosts",
    [
        "localhost:* 127.0.0.1:*",
        "::1:*",
        "[::1",
        "[127.0.0.1]:*",
        "localhost:0",
        "*.example.com",
    ],
)
def test_allowed_hosts_use_strict_comma_separated_host_patterns(
    allowed_hosts: str,
) -> None:
    with pytest.raises(RuntimeConfigurationError, match="MCP_ALLOWED_HOSTS"):
        security_config(MCP_ALLOWED_HOSTS=allowed_hosts)


def test_allowed_hosts_preserve_bracketed_ipv6_and_normalize_case() -> None:
    config = security_config(
        MCP_ALLOWED_HOSTS="LOCALHOST:8085, [0:0:0:0:0:0:0:1]:*,qdrant-mcp-portal:*"
    )
    assert config.allowed_hosts == (
        "localhost:8085",
        "[::1]:*",
        "qdrant-mcp-portal:*",
    )


def test_security_and_provider_header_names_must_be_unique() -> None:
    with pytest.raises(RuntimeConfigurationError, match="reserved"):
        security_config(MCP_PORTAL_GRANT_HEADER="Authorization")
    with pytest.raises(RuntimeConfigurationError, match="unique"):
        security_config(MCP_TENANT_ID_HEADER="X-MADPANDA-PORTAL-GRANT")

    config = security_config()
    with pytest.raises(RuntimeConfigurationError, match="unique"):
        validate_request_header_configuration(
            config,
            {
                "MCP_QDRANT_URL_HEADER": "x-provider-config",
                "MCP_QDRANT_API_KEY_HEADER": "X-PROVIDER-CONFIG",
            },
        )
    with pytest.raises(RuntimeConfigurationError, match="unique"):
        validate_request_header_configuration(
            config,
            {"MCP_QDRANT_URL_HEADER": "authorization"},
        )


def test_origins_are_exact_and_wildcards_are_forbidden() -> None:
    with pytest.raises(RuntimeConfigurationError, match="wildcards"):
        security_config(MCP_ALLOWED_ORIGINS="*")
    config = security_config(
        MCP_ALLOWED_ORIGINS="HTTPS://Console.Example:443,http://localhost:3000"
    )
    assert config.allowed_origins == (
        "https://console.example",
        "http://localhost:3000",
    )


@pytest.mark.asyncio
async def test_standalone_uses_strict_bearer_and_only_get_health_is_public() -> None:
    downstream = RecordingApp()
    app = AccessControlMiddleware(downstream, security_config())

    public_status, _, _ = await invoke(app, path="/health", method="GET")
    assert public_status == 200
    assert downstream.principals == [None]

    for path, method in (("/health", "POST"), ("/health/", "GET"), ("/other", "GET")):
        status, payload, receive_calls = await invoke(app, path=path, method=method)
        assert status == 401
        assert payload["error"]["code"] == "missing_access_token"
        assert receive_calls == 0

    for authorization in ("Basic value", f"Bearer  {ACCESS_TOKEN}", "Bearer wrong"):
        status, payload, _ = await invoke(
            app,
            headers={"authorization": authorization},
        )
        assert status == 401
        assert payload["error"]["code"] == "invalid_access_token"
        assert ACCESS_TOKEN not in json.dumps(payload)

    status, _, _ = await invoke(
        app,
        headers={
            "authorization": f"Bearer {ACCESS_TOKEN}",
            "x-madpanda-user-id": "spoofed-tenant",
        },
    )
    assert status == 200
    assert downstream.principals[-1] == "standalone"
    assert get_request_principal() is None


@pytest.mark.asyncio
async def test_portal_grant_requires_safe_tenant_principal_and_resets_context() -> None:
    config = security_config("portal")
    downstream = RecordingApp()
    app = AccessControlMiddleware(downstream, config)

    cases = [
        ({}, "missing_portal_grant"),
        ({config.portal_grant_header: "wrong"}, "invalid_portal_grant"),
        ({config.portal_grant_header: PORTAL_GRANT}, "missing_tenant_id"),
        (
            {
                config.portal_grant_header: PORTAL_GRANT,
                config.tenant_id_header: "../../tenant",
            },
            "invalid_tenant_id",
        ),
    ]
    for headers, code in cases:
        status, payload, receive_calls = await invoke(app, headers=headers)
        assert status == 401
        assert payload["error"]["code"] == code
        assert receive_calls == 0
        assert PORTAL_GRANT not in json.dumps(payload)

    status, _, _ = await invoke(
        app,
        headers={
            config.portal_grant_header: PORTAL_GRANT,
            config.tenant_id_header: "tenant_01:project-a",
        },
    )
    assert status == 200
    assert downstream.principals == ["tenant_01:project-a"]
    assert get_request_principal() is None


@pytest.mark.asyncio
async def test_typed_identity_distinguishes_standalone_from_matching_portal_subject() -> None:
    standalone_downstream = RecordingApp()
    standalone_app = AccessControlMiddleware(standalone_downstream, security_config())
    standalone_status, _, _ = await invoke(
        standalone_app,
        headers={"authorization": f"Bearer {ACCESS_TOKEN}"},
    )
    assert get_request_identity() is None
    assert get_request_principal() is None

    portal_config = security_config("portal")
    portal_downstream = RecordingApp()
    portal_app = AccessControlMiddleware(portal_downstream, portal_config)
    portal_status, _, _ = await invoke(
        portal_app,
        headers={
            portal_config.portal_grant_header: PORTAL_GRANT,
            portal_config.tenant_id_header: "standalone",
        },
    )

    assert standalone_status == portal_status == 200
    assert standalone_downstream.principals == ["standalone"]
    assert portal_downstream.principals == ["standalone"]
    assert standalone_downstream.identities == [
        RequestIdentity(kind="standalone", subject="standalone")
    ]
    assert portal_downstream.identities == [RequestIdentity(kind="portal", subject="standalone")]
    assert standalone_downstream.identities != portal_downstream.identities
    assert get_request_identity() is None
    assert get_request_principal() is None


def test_request_identity_repr_redacts_subject_without_changing_value_semantics() -> None:
    identity = RequestIdentity(kind="portal", subject="tenant-sensitive")

    assert identity == RequestIdentity(kind="portal", subject="tenant-sensitive")
    assert identity.subject == "tenant-sensitive"
    assert "tenant-sensitive" not in repr(identity)
    assert "kind='portal'" in repr(identity)


@pytest.mark.parametrize(
    ("mode", "selected_token"),
    [("standalone", ACCESS_TOKEN), ("portal", PORTAL_GRANT)],
)
def test_runtime_security_repr_never_contains_service_tokens(
    mode: str,
    selected_token: str,
) -> None:
    config = security_config(mode)

    if mode == "standalone":
        assert config.standalone_access_token == selected_token
    else:
        assert config.portal_grant_token == selected_token
    for rendered in (repr(config), str(config)):
        assert ACCESS_TOKEN not in rendered
        assert PORTAL_GRANT not in rendered


@pytest.mark.asyncio
async def test_duplicate_security_and_provider_headers_reject_before_body_read() -> None:
    config = security_config()
    downstream = RecordingApp()
    app = AccessControlMiddleware(
        downstream,
        config,
        singleton_headers=("x-qdrant-url",),
    )
    base = {"authorization": f"Bearer {ACCESS_TOKEN}"}

    for duplicate in (
        (b"authorization", f"Bearer {ACCESS_TOKEN}".encode()),
        (b"x-qdrant-url", b"https://second.example"),
    ):
        headers = dict(base)
        if duplicate[0] == b"x-qdrant-url":
            headers["x-qdrant-url"] = "https://first.example"
        status, payload, receive_calls = await invoke(
            app,
            headers=headers,
            extra_headers=[duplicate],
        )
        assert status == 400
        assert payload["error"]["code"] == "duplicate_security_header"
        assert receive_calls == 0
    assert downstream.calls == 0


@pytest.mark.asyncio
async def test_oversized_header_count_and_bytes_reject_before_auth_or_body() -> None:
    downstream = RecordingApp()
    app = AccessControlMiddleware(downstream, security_config())
    auth = {"authorization": f"Bearer {ACCESS_TOKEN}"}

    status, payload, receive_calls = await invoke(
        app,
        headers=auth,
        extra_headers=[(f"x-padding-{index}".encode(), b"value") for index in range(64)],
    )
    assert status == 431
    assert payload["error"]["code"] == "request_headers_too_large"
    assert receive_calls == 0

    oversized_secret = "s" * 33_000
    status, payload, receive_calls = await invoke(
        app,
        headers={**auth, "x-openai-api-key": oversized_secret},
    )
    assert status == 431
    assert payload["error"]["code"] == "request_headers_too_large"
    assert oversized_secret not in json.dumps(payload)
    assert receive_calls == 0
    assert downstream.calls == 0


@pytest.mark.asyncio
async def test_host_origin_and_request_framing_are_checked_before_protocol() -> None:
    config = security_config(MCP_ALLOWED_ORIGINS="https://console.example")
    downstream = RecordingApp()
    app = AccessControlMiddleware(downstream, config)
    auth = {"authorization": f"Bearer {ACCESS_TOKEN}"}

    cases = [
        ({**auth, "host": "attacker.example"}, 421, "host_not_allowed"),
        ({**auth, "host": "localhost:evil"}, 421, "host_not_allowed"),
        ({**auth, "host": "localhost:8000:extra"}, 421, "host_not_allowed"),
        ({**auth, "host": "[::1]:evil"}, 421, "host_not_allowed"),
        ({**auth, "origin": "https://evil.example"}, 403, "origin_not_allowed"),
        ({**auth, "content-length": "invalid"}, 400, "invalid_content_length"),
        ({**auth, "content-length": "1025"}, 413, "request_body_too_large"),
        (
            {**auth, "content-length": "1", "transfer-encoding": "chunked"},
            400,
            "ambiguous_request_framing",
        ),
    ]
    for headers, expected_status, expected_code in cases:
        status, payload, receive_calls = await invoke(app, headers=headers)
        assert status == expected_status
        assert payload["error"]["code"] == expected_code
        assert receive_calls == 0
    assert downstream.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("method", ["GET", "POST"])
async def test_chunked_bodies_are_bounded_for_every_http_method(method: str) -> None:
    downstream = RecordingApp()
    app = AccessControlMiddleware(downstream, security_config())
    status, payload, receive_calls = await invoke(
        app,
        method=method,
        headers={"authorization": f"Bearer {ACCESS_TOKEN}"},
        chunks=[(b"a" * 800, True), (b"b" * 225, False)],
    )
    assert status == 413
    assert payload["error"]["code"] == "request_body_too_large"
    assert receive_calls == 2
    assert downstream.calls == 0


@pytest.mark.asyncio
async def test_public_health_still_enforces_host_and_chunked_get_body_limits() -> None:
    downstream = RecordingApp()
    app = AccessControlMiddleware(downstream, security_config())

    bad_host, payload, receive_calls = await invoke(
        app,
        path="/health",
        method="GET",
        headers={"host": "attacker.example"},
    )
    assert bad_host == 421
    assert payload["error"]["code"] == "host_not_allowed"
    assert receive_calls == 0

    too_large, payload, receive_calls = await invoke(
        app,
        path="/health",
        method="GET",
        chunks=[(b"a" * 800, True), (b"b" * 225, False)],
    )
    assert too_large == 413
    assert payload["error"]["code"] == "request_body_too_large"
    assert receive_calls == 2
    assert downstream.calls == 0


@pytest.mark.asyncio
async def test_valid_body_is_replayed_once_to_downstream() -> None:
    downstream = RecordingApp()
    app = AccessControlMiddleware(downstream, security_config())
    status, _, receive_calls = await invoke(
        app,
        headers={"authorization": f"Bearer {ACCESS_TOKEN}"},
        chunks=[(b"hello ", True), (b"world", False)],
    )
    assert status == 200
    assert receive_calls == 2
    assert downstream.body == b"hello world"


def make_hosted(
    request_settings: RequestOverrideSettings,
    *,
    qdrant_configured: bool = True,
    embedding_provider: DummyEmbeddingProvider | None = None,
) -> HostedQdrantMCPServer:
    qdrant = (
        QdrantSettings.model_validate({"QDRANT_LOCAL_PATH": ":memory:"})
        if qdrant_configured
        else QdrantSettings()
    )
    return HostedQdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=qdrant,
        request_override_settings=request_settings,
        memory_settings=MemorySettings(),
        embedding_provider=embedding_provider or DummyEmbeddingProvider(),
        version=__version__,
    )


def standalone_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_MODE", "standalone")
    monkeypatch.setenv("MCP_ACCESS_TOKEN", ACCESS_TOKEN)
    monkeypatch.setenv("MCP_ALLOWED_HOSTS", "testserver,localhost:*")
    monkeypatch.delenv("MCP_ALLOWED_ORIGINS", raising=False)


def test_stdio_construction_does_not_require_http_auth_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for variable in (
        "MCP_MODE",
        "MCP_ACCESS_TOKEN",
        "MCP_PORTAL_GRANT_TOKEN",
        "QDRANT_CREDENTIAL_MODE",
    ):
        monkeypatch.delenv(variable, raising=False)
    server = make_hosted(RequestOverrideSettings())
    assert server._runtime_security_config is None  # noqa: SLF001


def test_standalone_server_http_starts_and_health_uses_safe_build_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    standalone_environment(monkeypatch)
    monkeypatch.setenv("MCP_SERVER_VERSION", "must-not-be-used")
    monkeypatch.setenv("MCP_BUILD_SHA", "not-a-commit")
    monkeypatch.setenv("MCP_SOURCE_FINGERPRINT", "f" * 64)
    settings = RequestOverrideSettings.model_validate({"QDRANT_CREDENTIAL_MODE": "server"})
    server = make_hosted(settings)

    with TestClient(server.http_app(path="/mcp")) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["version"] == "2.0.0"
    assert payload["build_sha"] == "unknown"
    assert payload["source_fingerprint"] == "f" * 64
    assert payload["configuration_ready"] is True
    assert payload["configuration_readiness"]["mode"] == "standalone"
    assert payload["configuration_readiness"]["credential_mode"] == "server"


def test_health_derives_digest_from_exact_runtime_image_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    standalone_environment(monkeypatch)
    digest = "sha256:" + "a" * 64
    reference = f"ghcr.io/example/qdrant-mcp@{digest}"
    monkeypatch.setenv("MCP_IMAGE_REFERENCE", reference)
    monkeypatch.setenv("MCP_IMAGE_DIGEST", "sha256:" + "b" * 64)
    settings = RequestOverrideSettings.model_validate({"QDRANT_CREDENTIAL_MODE": "server"})
    server = make_hosted(settings)

    with TestClient(server.http_app(path="/mcp")) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["image_reference"] == reference
    assert payload["image_digest"] == digest


def test_portal_request_mode_allows_local_fastembed_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_MODE", "portal")
    monkeypatch.setenv("MCP_PORTAL_GRANT_TOKEN", PORTAL_GRANT)
    monkeypatch.setenv("MCP_ALLOWED_HOSTS", "testserver")
    settings = RequestOverrideSettings.model_validate(
        {
            "QDRANT_CREDENTIAL_MODE": "request",
            "MCP_ALLOW_REQUEST_OVERRIDES": True,
            "MCP_REQUIRE_REQUEST_QDRANT_URL": True,
            "MCP_DISABLE_DEFAULT_QDRANT_FALLBACK": True,
            "MCP_DISABLE_DEFAULT_EMBEDDING_FALLBACK": False,
            "MCP_QDRANT_HOST_ALLOWLIST": "tenant-qdrant.example",
        }
    )
    server = make_hosted(settings)
    app = server.http_app(path="/mcp")
    assert app.config.mode == "portal"
    assert server._runtime_credential_mode == "request"  # noqa: SLF001


@pytest.mark.parametrize(
    "mode,settings_data,qdrant_configured,error",
    [
        (
            "portal",
            {"QDRANT_CREDENTIAL_MODE": "server"},
            True,
            "portal requires QDRANT_CREDENTIAL_MODE=request",
        ),
        (
            "standalone",
            {
                "QDRANT_CREDENTIAL_MODE": "request",
                "MCP_QDRANT_HOST_ALLOWLIST": "tenant-qdrant.example",
            },
            True,
            "requires fail-closed request credential flags",
        ),
        (
            "standalone",
            {"QDRANT_CREDENTIAL_MODE": "server"},
            False,
            "requires QDRANT_URL or QDRANT_LOCAL_PATH",
        ),
    ],
)
def test_http_startup_rejects_incomplete_or_contradictory_modes(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    settings_data: dict[str, Any],
    qdrant_configured: bool,
    error: str,
) -> None:
    if mode == "portal":
        monkeypatch.setenv("MCP_MODE", "portal")
        monkeypatch.setenv("MCP_PORTAL_GRANT_TOKEN", PORTAL_GRANT)
    else:
        standalone_environment(monkeypatch)
    server = make_hosted(
        RequestOverrideSettings.model_validate(settings_data),
        qdrant_configured=qdrant_configured,
    )
    with pytest.raises(RuntimeConfigurationError, match=error):
        server.http_app(path="/mcp")


def test_request_qdrant_and_openai_urls_use_separate_fail_closed_policies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = RequestOverrideSettings.model_validate(
        {
            "QDRANT_CREDENTIAL_MODE": "request",
            "MCP_ALLOW_REQUEST_OVERRIDES": True,
            "MCP_REQUIRE_REQUEST_QDRANT_URL": True,
            "MCP_REQUIRE_REQUEST_COLLECTION": False,
            "MCP_DISABLE_DEFAULT_QDRANT_FALLBACK": True,
            "MCP_QDRANT_HOST_ALLOWLIST": "qdrant.example",
            "MCP_QDRANT_ALLOWED_PORTS": "443,6333",
            "MCP_OPENAI_HOST_ALLOWLIST": "api.openai.example",
            "MCP_OPENAI_ALLOWED_PORTS": "443",
        }
    )
    server = make_hosted(settings)
    calls: list[dict[str, Any]] = []

    def validate(url: str, **policy: Any) -> str:
        calls.append({"url": url, **policy})
        return url

    monkeypatch.setattr("mcp_server_qdrant.hosted_server.validate_outbound_url", validate)
    first = server._build_request_overrides(  # noqa: SLF001
        {
            "x-qdrant-url": "https://qdrant.example:6333",
            "x-embedding-provider": "openai",
            "x-embedding-model": "text-embedding-3-small",
            "x-openai-api-key": "request-key-one",
            "x-openai-base-url": "https://api.openai.example/v1",
        }
    )
    second = server._build_request_overrides(  # noqa: SLF001
        {
            "x-qdrant-url": "https://qdrant.example:6333",
            "x-embedding-provider": "openai",
            "x-embedding-model": "text-embedding-3-small",
            "x-openai-api-key": "request-key-one",
            "x-openai-base-url": "https://api.openai.example/v1",
        }
    )

    assert first is not None and second is not None
    assert first.embedding_provider is not second.embedding_provider
    assert calls[0]["allowed_hosts"] == ["api.openai.example"]
    assert calls[0]["allowed_ports"] == [443]
    assert calls[1]["allowed_hosts"] == ["qdrant.example"]
    assert calls[1]["allowed_ports"] == [443, 6333]
    assert all(call["allow_insecure_http"] is False for call in calls)


def test_request_fastembed_rejects_arbitrary_models_without_construction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = make_hosted(
        RequestOverrideSettings(),
        embedding_provider=ConfiguredFastEmbedProvider(),
    )
    constructor_calls = 0

    def forbidden_constructor(_settings: Any) -> None:
        nonlocal constructor_calls
        constructor_calls += 1
        raise AssertionError("request FastEmbed must not construct or download a model")

    monkeypatch.setattr(
        "mcp_server_qdrant.hosted_server.create_embedding_provider",
        forbidden_constructor,
    )
    with pytest.raises(ValueError, match="exact configured server model"):
        server._resolve_request_embedding_provider(  # noqa: SLF001
            {
                "x-embedding-provider": "fastembed",
                "x-embedding-model": "attacker/arbitrary-model",
            }
        )
    assert constructor_calls == 0


def test_request_fastembed_reuses_exact_default_provider_and_dimension() -> None:
    configured = ConfiguredFastEmbedProvider()
    server = make_hosted(
        RequestOverrideSettings(),
        embedding_provider=configured,
    )
    headers = {
        "x-embedding-provider": "fastembed",
        "x-embedding-model": configured.model_name,
        "x-embedding-vector-size": "3",
    }

    first, first_settings = server._resolve_request_embedding_provider(  # noqa: SLF001
        headers
    )
    second, _ = server._resolve_request_embedding_provider(headers)  # noqa: SLF001
    assert first is configured
    assert second is configured
    assert first_settings is not None
    assert first_settings.vector_size == 3

    with pytest.raises(ValueError, match="vector size must match"):
        server._resolve_request_embedding_provider(  # noqa: SLF001
            {**headers, "x-embedding-vector-size": "4"}
        )


@pytest.mark.asyncio
async def test_named_vector_creation_never_reflects_provider_error_content() -> None:
    secret = "provider-api-key-do-not-reflect"
    provider_url = "https://qdrant.example/private/collection"
    connector = QdrantConnector(
        qdrant_url=":memory:",
        qdrant_api_key=secret,
        collection_name="memory",
        embedding_provider=DummyEmbeddingProvider(),
    )

    class FailingClient:
        async def create_vector_name(self, **_kwargs: Any) -> None:
            raise UnexpectedResponse(
                status_code=500,
                reason_phrase="Internal Server Error",
                content=f"token={secret} url={provider_url}".encode(),
                headers=Headers(),
            )

    connector._client = FailingClient()  # type: ignore[assignment]  # noqa: SLF001
    with pytest.raises(ValueError) as captured:
        await connector.create_dense_vector_name(
            collection_name="private-path",
            vector_name="target",
            size=3,
            distance="Cosine",
        )

    message = str(captured.value)
    assert message == "Qdrant named vector creation failed with HTTP 500."
    assert secret not in message
    assert provider_url not in message
    assert "private-path" not in message
    assert captured.value.__cause__ is None


@pytest.mark.asyncio
async def test_qdrant_connector_aclose_is_idempotent_and_repr_safe() -> None:
    secret = "synthetic-request-qdrant-key"
    connector = QdrantConnector(
        qdrant_url=":memory:",
        qdrant_api_key=secret,
        collection_name="memory",
        embedding_provider=DummyEmbeddingProvider(),
    )
    await connector._client.close()  # noqa: SLF001

    class ClosingClient:
        def __init__(self) -> None:
            self.close_calls = 0

        async def close(self) -> None:
            self.close_calls += 1

    client = ClosingClient()
    connector._client = client  # type: ignore[assignment]  # noqa: SLF001

    assert secret not in repr(connector)
    assert secret not in str(connector)
    assert secret not in repr(vars(connector))
    await connector.aclose()
    await connector.aclose()

    assert connector.closed is True
    assert connector._qdrant_api_key is None  # noqa: SLF001
    assert client.close_calls == 1


@pytest.mark.asyncio
async def test_hosted_request_overrides_are_built_once_and_context_is_reset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = RequestOverrideSettings.model_validate(
        {
            "QDRANT_CREDENTIAL_MODE": "request",
            "MCP_ALLOW_REQUEST_OVERRIDES": True,
            "MCP_REQUIRE_REQUEST_QDRANT_URL": True,
            "MCP_REQUIRE_REQUEST_COLLECTION": False,
            "MCP_DISABLE_DEFAULT_QDRANT_FALLBACK": True,
            "MCP_QDRANT_HOST_ALLOWLIST": "qdrant.example",
        }
    )
    server = make_hosted(settings)
    headers = {
        "x-qdrant-url": "https://qdrant.example:6333",
        "x-embedding-provider": "openai",
        "x-embedding-model": "text-embedding-3-small",
        "x-openai-api-key": "one-call-request-key",
    }
    monkeypatch.setattr(
        "mcp_server_qdrant.hosted_server.validate_outbound_url",
        lambda url, **_policy: url,
    )
    monkeypatch.setattr(
        "mcp_server_qdrant.hosted_server.get_http_headers",
        lambda: headers,
    )
    original_build = server._build_request_overrides  # noqa: SLF001
    build_calls = 0

    def counted_build(request_headers: Mapping[str, Any] | None):
        nonlocal build_calls
        build_calls += 1
        return original_build(request_headers)

    monkeypatch.setattr(server, "_build_request_overrides", counted_build)
    observed: dict[str, Any] = {}

    async def delegate(
        _server: FastMCP,
        _key: str,
        _arguments: dict[str, Any],
    ) -> list[Any]:
        observed["overrides"] = server._request_overrides_var.get()  # noqa: SLF001
        observed["connector"] = server._connector_var.get()  # noqa: SLF001
        observed["provider"] = server._embedding_provider_var.get()  # noqa: SLF001
        observed["info"] = server._embedding_info_var.get()  # noqa: SLF001
        return []

    monkeypatch.setattr(FastMCP, "_call_tool_mcp", delegate)
    await server._call_tool_mcp("qdrant-list-collections", {})  # noqa: SLF001

    assert build_calls == 1
    assert observed["overrides"] is not None
    assert observed["connector"] is not None
    assert observed["provider"] is not None
    assert observed["info"].provider == "openai"
    assert server._request_overrides_var.get() is None  # noqa: SLF001
    assert server._connector_var.get() is None  # noqa: SLF001
    assert server._embedding_provider_var.get() is None  # noqa: SLF001
    assert server._embedding_info_var.get() is None  # noqa: SLF001
    assert observed["connector"].closed is True
    assert observed["provider"].closed is True


@pytest.mark.asyncio
async def test_hosted_claimed_request_resources_are_not_closed_before_background_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = RequestOverrideSettings.model_validate(
        {
            "QDRANT_CREDENTIAL_MODE": "request",
            "MCP_ALLOW_REQUEST_OVERRIDES": True,
            "MCP_REQUIRE_REQUEST_QDRANT_URL": True,
            "MCP_REQUIRE_REQUEST_COLLECTION": False,
            "MCP_DISABLE_DEFAULT_QDRANT_FALLBACK": True,
            "MCP_QDRANT_HOST_ALLOWLIST": "qdrant.example",
        }
    )
    server = make_hosted(settings)
    monkeypatch.setattr(
        "mcp_server_qdrant.hosted_server.validate_outbound_url",
        lambda url, **_policy: url,
    )
    monkeypatch.setattr(
        "mcp_server_qdrant.hosted_server.get_http_headers",
        lambda: {
            "x-qdrant-url": "https://qdrant.example:6333",
            "x-embedding-provider": "openai",
            "x-embedding-model": "text-embedding-3-small",
            "x-openai-api-key": "background-request-key",
        },
    )
    observed: dict[str, Any] = {}

    async def delegate(
        _server: FastMCP,
        _key: str,
        _arguments: dict[str, Any],
    ) -> list[Any]:
        overrides = server._request_overrides_var.get()  # noqa: SLF001
        assert overrides is not None
        overrides.resources_claimed = True
        observed["connector"] = server._connector_var.get()  # noqa: SLF001
        observed["provider"] = server._embedding_provider_var.get()  # noqa: SLF001
        return []

    monkeypatch.setattr(FastMCP, "_call_tool_mcp", delegate)
    await server._call_tool_mcp("qdrant-list-collections", {})  # noqa: SLF001

    connector = observed["connector"]
    provider = observed["provider"]
    assert connector.closed is False
    assert provider.closed is False
    assert server._request_overrides_var.get() is None  # noqa: SLF001
    assert server._connector_var.get() is None  # noqa: SLF001
    assert server._embedding_provider_var.get() is None  # noqa: SLF001
    assert server._embedding_info_var.get() is None  # noqa: SLF001

    await server._close_request_resources(connector, provider)  # noqa: SLF001
    assert connector.closed is True
    assert provider.closed is True


@pytest.mark.asyncio
async def test_hosted_lifespan_closes_owned_server_openai_client() -> None:
    secret = "synthetic-server-openai-key"
    server = HostedQdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=QdrantSettings.model_validate({"QDRANT_LOCAL_PATH": ":memory:"}),
        request_override_settings=RequestOverrideSettings(),
        memory_settings=MemorySettings(),
        embedding_provider_settings=EmbeddingProviderSettings.model_validate(
            {
                "EMBEDDING_PROVIDER": "openai",
                "EMBEDDING_MODEL": "text-embedding-3-small",
                "OPENAI_API_KEY": secret,
            }
        ),
        version=__version__,
    )
    provider = server._default_embedding_provider  # noqa: SLF001
    assert isinstance(provider, OpenAIProvider)

    async with server._lifespan(server):  # noqa: SLF001
        await provider._get_client()  # noqa: SLF001
        http_client = provider._http_client  # noqa: SLF001
        assert http_client is not None
        assert http_client.is_closed is False

    assert provider.closed is True
    assert http_client.is_closed is True
    assert secret not in repr(provider)
