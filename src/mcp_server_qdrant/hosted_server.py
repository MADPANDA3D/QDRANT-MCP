from __future__ import annotations

import os
import re
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from typing import Any, Literal

from starlette.requests import Request
from starlette.responses import JSONResponse

try:  # FastMCP >= 2.2.11
    from fastmcp.server.dependencies import get_http_headers
except ImportError:  # pragma: no cover - older FastMCP

    def get_http_headers() -> dict[str, str]:
        return {}


from mcp.types import EmbeddedResource, ImageContent, TextContent

from mcp_server_qdrant import __version__
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.mcp_server import QdrantMCPServer, RequestQdrantOverrides
from mcp_server_qdrant.qdrant import QdrantConnector
from mcp_server_qdrant.runtime_security import (
    AccessControlMiddleware,
    RuntimeConfigurationError,
    RuntimeSecurityConfig,
    load_runtime_security_config,
    validate_request_header_configuration,
)
from mcp_server_qdrant.safe_fetch import validate_outbound_url
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    MemorySettings,
    QdrantSettings,
    RequestOverrideSettings,
    ToolSettings,
)
from mcp_server_qdrant.tool_manifest import CATALOG_VERSION, LOCAL_NAVIGATION_TOOLS

_COMMIT_SHA = re.compile(r"^[0-9a-f]{7,64}$")
_SOURCE_FINGERPRINT = re.compile(r"^[0-9a-f]{64}$")
_IMAGE_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_IMAGE_REFERENCE = re.compile(r"^[A-Za-z0-9._/@:+-]{1,256}$")


def _validated_build_value(name: str, pattern: re.Pattern[str]) -> str:
    value = str(os.getenv(name, "") or "").strip()
    return value if pattern.fullmatch(value) else "unknown"


def _resolved_image_identity() -> tuple[str, str]:
    image_reference = _validated_build_value("MCP_IMAGE_REFERENCE", _IMAGE_REFERENCE)
    image_digest = _validated_build_value("MCP_IMAGE_DIGEST", _IMAGE_DIGEST)
    if image_reference != "unknown":
        _, separator, reference_digest = image_reference.rpartition("@")
        if separator and _IMAGE_DIGEST.fullmatch(reference_digest):
            image_digest = reference_digest
    return image_reference, image_digest


class HostedQdrantMCPServer(QdrantMCPServer):
    """
    Qdrant MCP server with per-request connection overrides via HTTP headers.
    """

    _MUTATION_TOOL_NAMES = {
        "qdrant-store",
        "qdrant-cache-memory",
        "qdrant-promote-short-term",
        "qdrant-ingest-with-validation",
        "qdrant-ingest-document",
        "qdrant-ingest-manifest",
        "qdrant-ingest-class-manifest",
        "qdrant-ingest-school-manifest",
        "qdrant-ingest-textbook",
        "qdrant-ensure-payload-indexes",
        "qdrant-create-collection",
        "qdrant-backfill-memory-contract",
        "qdrant-update-point",
        "qdrant-patch-payload",
        "qdrant-tag-memories",
        "qdrant-link-memories",
        "qdrant-reembed-points",
        "qdrant-migrate-collection-embedding",
        "qdrant-bulk-patch",
        "qdrant-dedupe-memories",
        "qdrant-merge-duplicates",
        "qdrant-expire-memories",
        "qdrant-expire-short-term",
        "qdrant-update-optimizer-config",
        "qdrant-delete-points",
        "qdrant-delete-by-filter",
        "qdrant-delete-document",
        "qdrant-create-snapshot",
        "qdrant-submit-job",
        "qdrant-cancel-job",
        "qdrant-cancel-ingest",
    }
    _EMBEDDING_REQUIRED_READ_TOOLS = {
        "qdrant-find",
        "qdrant-find-short-term",
    }
    _LOCAL_NAVIGATION_TOOL_NAMES = LOCAL_NAVIGATION_TOOLS

    def __init__(
        self,
        tool_settings: ToolSettings,
        qdrant_settings: QdrantSettings,
        request_override_settings: RequestOverrideSettings | None = None,
        memory_settings: MemorySettings | None = None,
        embedding_provider_settings: EmbeddingProviderSettings | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        name: str = "mcp-server-qdrant",
        instructions: str | None = None,
        **settings: Any,
    ) -> None:
        inherited_lifespan = settings.pop("lifespan", None)
        settings["lifespan"] = self._managed_resource_lifespan(inherited_lifespan)
        self._owns_default_embedding_provider = embedding_provider_settings is not None
        self._owned_default_connectors: list[QdrantConnector] = []
        self._server_resources_closed = False
        self.request_override_settings = request_override_settings or RequestOverrideSettings()
        self._default_qdrant_url = qdrant_settings.location
        self._default_qdrant_api_key = qdrant_settings.api_key
        self._default_collection_name = qdrant_settings.collection_name
        self._default_vector_name = qdrant_settings.vector_name
        self._default_local_path = qdrant_settings.local_path
        self._runtime_security_config: RuntimeSecurityConfig | None = None
        self._runtime_credential_mode: Literal["server", "request"] | None = None

        if self.request_override_settings.allow_request_overrides:
            qdrant_settings = qdrant_settings.model_copy()
            qdrant_settings.collection_name = None

        super().__init__(
            tool_settings=tool_settings,
            qdrant_settings=qdrant_settings,
            request_override_settings=self.request_override_settings,
            memory_settings=memory_settings,
            embedding_provider_settings=embedding_provider_settings,
            embedding_provider=embedding_provider,
            name=name,
            instructions=instructions,
            **settings,
        )
        self._owned_default_connectors.append(self._default_qdrant_connector)
        self._register_health_route()

        if self.request_override_settings.allow_request_overrides:
            connector = QdrantConnector(
                self._default_qdrant_url,
                self._default_qdrant_api_key,
                self._default_collection_name,
                self.embedding_provider,
                self._default_vector_name,
                self._default_local_path,
                self.payload_indexes,
            )
            self.qdrant_connector = connector
            self._owned_default_connectors.append(connector)

    def _managed_resource_lifespan(self, inherited_lifespan: Any) -> Any:
        @asynccontextmanager
        async def managed_lifespan(server: Any) -> AsyncIterator[dict[str, Any]]:
            try:
                if inherited_lifespan is None:
                    yield {}
                else:
                    async with inherited_lifespan(server) as context:
                        yield context or {}
            finally:
                await self._close_server_resources()

        return managed_lifespan

    async def _close_server_resources(self) -> None:
        if self._server_resources_closed:
            return
        self._server_resources_closed = True
        close_failed = False
        seen: set[int] = set()
        for connector in reversed(self._owned_default_connectors):
            if id(connector) in seen:
                continue
            seen.add(id(connector))
            try:
                await connector.aclose()
            except Exception:
                close_failed = True

        if self._owns_default_embedding_provider:
            close_provider = getattr(self._default_embedding_provider, "aclose", None)
            if callable(close_provider):
                try:
                    await close_provider()
                except Exception:
                    close_failed = True
        if close_failed:
            raise RuntimeError("Hosted server resource shutdown failed.") from None

    @staticmethod
    async def _close_request_resources(
        connector: QdrantConnector | None,
        provider: EmbeddingProvider | None,
    ) -> None:
        close_failed = False
        if connector is not None:
            try:
                await connector.aclose()
            except Exception:
                close_failed = True
        close_provider = getattr(provider, "aclose", None)
        if callable(close_provider):
            try:
                await close_provider()
            except Exception:
                close_failed = True
        if close_failed:
            raise RuntimeError("Request-scoped client shutdown failed.") from None

    def _provider_header_configuration(self) -> dict[str, str]:
        configured = self.request_override_settings
        return {
            "MCP_QDRANT_URL_HEADER": configured.qdrant_url_header,
            "MCP_QDRANT_API_KEY_HEADER": configured.qdrant_api_key_header,
            "MCP_COLLECTION_NAME_HEADER": configured.collection_name_header,
            "MCP_QDRANT_VECTOR_NAME_HEADER": configured.vector_name_header,
            "MCP_EMBEDDING_PROVIDER_HEADER": configured.embedding_provider_header,
            "MCP_EMBEDDING_MODEL_HEADER": configured.embedding_model_header,
            "MCP_EMBEDDING_VECTOR_SIZE_HEADER": (configured.embedding_vector_size_header),
            "MCP_OPENAI_API_KEY_HEADER": configured.openai_api_key_header,
            "MCP_OPENAI_BASE_URL_HEADER": configured.openai_base_url_header,
            "MCP_OPENAI_ORG_HEADER": configured.openai_organization_header,
            "MCP_OPENAI_PROJECT_HEADER": configured.openai_project_header,
        }

    def _validate_http_credential_configuration(
        self,
        security: RuntimeSecurityConfig,
    ) -> Literal["server", "request"]:
        configured = self.request_override_settings
        credential_mode = configured.credential_mode
        if credential_mode is None:
            raise RuntimeConfigurationError(
                "QDRANT_CREDENTIAL_MODE must be exactly 'server' or 'request' for HTTP."
            )
        if security.mode == "portal" and credential_mode != "request":
            raise RuntimeConfigurationError(
                "MCP_MODE=portal requires QDRANT_CREDENTIAL_MODE=request."
            )

        if credential_mode == "request":
            required_flags = {
                "MCP_ALLOW_REQUEST_OVERRIDES": configured.allow_request_overrides,
                "MCP_REQUIRE_REQUEST_QDRANT_URL": configured.require_request_qdrant_url,
                "MCP_DISABLE_DEFAULT_QDRANT_FALLBACK": (configured.disable_default_qdrant_fallback),
            }
            disabled = [name for name, enabled in required_flags.items() if not enabled]
            if disabled:
                raise RuntimeConfigurationError(
                    "QDRANT_CREDENTIAL_MODE=request requires fail-closed request "
                    "credential flags: " + ", ".join(disabled) + "."
                )
            if not configured.qdrant_host_allowlist:
                raise RuntimeConfigurationError(
                    "QDRANT_CREDENTIAL_MODE=request requires a non-empty MCP_QDRANT_HOST_ALLOWLIST."
                )
            if (
                str(self.embedding_info.provider).lower() in {"openai", "openaiprovider"}
                and not configured.disable_default_embedding_fallback
            ):
                raise RuntimeConfigurationError(
                    "Request mode with a server OpenAI provider requires "
                    "MCP_DISABLE_DEFAULT_EMBEDDING_FALLBACK=true so tenant requests "
                    "cannot consume a server credential."
                )
        else:
            contradictory: list[str] = []
            if configured.allow_request_overrides:
                contradictory.append("MCP_ALLOW_REQUEST_OVERRIDES")
            if configured.disable_default_qdrant_fallback:
                contradictory.append("MCP_DISABLE_DEFAULT_QDRANT_FALLBACK")
            if configured.disable_default_embedding_fallback:
                contradictory.append("MCP_DISABLE_DEFAULT_EMBEDDING_FALLBACK")
            if contradictory:
                raise RuntimeConfigurationError(
                    "QDRANT_CREDENTIAL_MODE=server conflicts with request-only flags: "
                    + ", ".join(contradictory)
                    + "."
                )
            if not (self._default_qdrant_url or self._default_local_path):
                raise RuntimeConfigurationError(
                    "QDRANT_CREDENTIAL_MODE=server requires QDRANT_URL or QDRANT_LOCAL_PATH."
                )
        return credential_mode

    def http_app(
        self,
        path: str | None = None,
        middleware: list[Any] | None = None,
        json_response: bool | None = None,
        stateless_http: bool | None = None,
        transport: Literal["http", "streamable-http", "sse"] = "http",
        event_store: Any | None = None,
        retry_interval: int | None = None,
        host_origin_protection: bool | Literal["auto"] | None = None,
        allowed_hosts: list[str] | None = None,
        allowed_origins: list[str] | None = None,
    ) -> Any:
        del host_origin_protection, allowed_hosts, allowed_origins
        security = load_runtime_security_config(os.environ)
        if security.portal_grant_header != self.request_override_settings.portal_grant_header:
            raise RuntimeConfigurationError(
                "MCP_PORTAL_GRANT_HEADER must match the hosted request configuration."
            )
        if security.tenant_id_header != self.request_override_settings.tenant_id_header:
            raise RuntimeConfigurationError(
                "MCP_TENANT_ID_HEADER must match the hosted request configuration."
            )
        singleton_headers = validate_request_header_configuration(
            security,
            self._provider_header_configuration(),
        )
        credential_mode = self._validate_http_credential_configuration(security)
        if self._runtime_security_config not in {None, security}:
            raise RuntimeConfigurationError(
                "HTTP access configuration changed after startup; restart the service."
            )
        if self._runtime_credential_mode not in {None, credential_mode}:
            raise RuntimeConfigurationError(
                "QDRANT_CREDENTIAL_MODE changed after startup; restart the service."
            )
        app = super().http_app(
            path=path,
            middleware=middleware,
            json_response=json_response,
            stateless_http=stateless_http,
            transport=transport,
            event_store=event_store,
            retry_interval=retry_interval,
            host_origin_protection=True,
            allowed_hosts=list(security.allowed_hosts),
            allowed_origins=list(security.allowed_origins),
        )
        self._runtime_security_config = security
        self._runtime_credential_mode = credential_mode
        return AccessControlMiddleware(app, security, singleton_headers)

    def _register_health_route(self) -> None:
        @self.custom_route("/health", methods=["GET"])
        async def health(_: Request) -> JSONResponse:
            manifest = self.get_tool_manifest()
            count = int(manifest["counts"]["raw"])
            default_qdrant_configured = bool(self._default_qdrant_url or self._default_local_path)
            request_overrides_enabled = bool(self.request_override_settings.allow_request_overrides)
            security = self._runtime_security_config
            credential_mode = self._runtime_credential_mode or "unconfigured"
            access_boundary_configured = security is not None
            credential_boundary_configured = (
                credential_mode == "request"
                and request_overrides_enabled
                and bool(self.request_override_settings.qdrant_host_allowlist)
            ) or (credential_mode == "server" and default_qdrant_configured)
            configuration_ready = access_boundary_configured and credential_boundary_configured
            build_sha = _validated_build_value("MCP_BUILD_SHA", _COMMIT_SHA)
            source_fingerprint = _validated_build_value(
                "MCP_SOURCE_FINGERPRINT",
                _SOURCE_FINGERPRINT,
            )
            image_reference, image_digest = _resolved_image_identity()
            return JSONResponse(
                {
                    "ok": True,
                    "service": "qdrant-mcp",
                    "status": "ok" if configuration_ready else "needs_configuration",
                    "version": __version__,
                    "build_sha": build_sha,
                    "source_fingerprint": source_fingerprint,
                    "image_reference": image_reference,
                    "image_digest": image_digest,
                    "catalog_version": CATALOG_VERSION,
                    "tool_count": count,
                    "raw_tool_count": manifest["counts"]["raw"],
                    "exposed_tool_count": count,
                    "agent_ready_tool_count": manifest["counts"]["agentReady"],
                    "configuration_ready": configuration_ready,
                    "configuration_readiness": {
                        "access_boundary_configured": access_boundary_configured,
                        "mode": security.mode if security is not None else "unconfigured",
                        "credential_mode": credential_mode,
                        "default_qdrant_configured": default_qdrant_configured,
                        "request_overrides_enabled": request_overrides_enabled,
                        "request_qdrant_allowlist_configured": bool(
                            self.request_override_settings.qdrant_host_allowlist
                        ),
                    },
                    "tools": {
                        "total": count,
                        "raw": manifest["counts"]["raw"],
                        "exposed": count,
                        "agent_ready": manifest["counts"]["agentReady"],
                        "legacy": manifest["counts"]["legacy"],
                        "hidden": manifest["counts"]["hidden"],
                    },
                }
            )

    @property
    def qdrant_connector(self) -> QdrantConnector:
        connector = self._connector_var.get()
        if connector is not None:
            return connector
        if self._default_qdrant_connector is None:  # pragma: no cover - setup guard
            raise ValueError("Qdrant connector is not initialized.")
        return self._default_qdrant_connector

    @qdrant_connector.setter
    def qdrant_connector(self, value: QdrantConnector) -> None:
        self._default_qdrant_connector = value

    def _normalize_headers(self, headers: Mapping[str, Any] | None) -> dict[str, str]:
        if not headers:
            return {}
        normalized: dict[str, str] = {}
        for key, value in headers.items():
            if value is None:
                continue
            if isinstance(value, (list, tuple)):
                if not value:
                    continue
                value = value[0]
            if isinstance(value, bytes):
                value = value.decode("utf-8", "ignore")
            normalized[str(key).lower()] = str(value).strip()
        return normalized

    def _resolve_request_embedding_provider(
        self, normalized: dict[str, str]
    ) -> tuple[EmbeddingProvider | None, EmbeddingProviderSettings | None]:
        provider_raw = normalized.get(
            self.request_override_settings.embedding_provider_header, ""
        ).lower()
        model_name = normalized.get(self.request_override_settings.embedding_model_header, "")
        vector_size_raw = normalized.get(
            self.request_override_settings.embedding_vector_size_header, ""
        )
        openai_api_key = normalized.get(self.request_override_settings.openai_api_key_header, "")
        openai_base_url = normalized.get(self.request_override_settings.openai_base_url_header, "")
        openai_org = normalized.get(self.request_override_settings.openai_organization_header, "")
        openai_project = normalized.get(self.request_override_settings.openai_project_header, "")

        embedding_requested = any(
            [
                provider_raw,
                model_name,
                vector_size_raw,
                openai_api_key,
                openai_base_url,
                openai_org,
                openai_project,
            ]
        )

        if not embedding_requested:
            if self.request_override_settings.disable_default_embedding_fallback:
                raise ValueError(
                    "Missing required header(s): "
                    + ", ".join(
                        [
                            self.request_override_settings.embedding_provider_header,
                            self.request_override_settings.embedding_model_header,
                        ]
                    )
                    + "."
                )
            return None, None

        missing_required: list[str] = []
        if not provider_raw:
            missing_required.append(self.request_override_settings.embedding_provider_header)
        if not model_name:
            missing_required.append(self.request_override_settings.embedding_model_header)

        if provider_raw and provider_raw not in {
            EmbeddingProviderType.OPENAI.value,
            EmbeddingProviderType.FASTEMBED.value,
        }:
            raise ValueError(
                f"Unsupported embedding provider '{provider_raw}'. "
                "Supported providers: openai, fastembed."
            )

        if provider_raw == EmbeddingProviderType.OPENAI.value and not openai_api_key:
            missing_required.append(self.request_override_settings.openai_api_key_header)

        if missing_required:
            raise ValueError("Missing required header(s): " + ", ".join(missing_required) + ".")

        if openai_base_url:
            openai_base_url = validate_outbound_url(
                openai_base_url,
                allowed_hosts=self.request_override_settings.openai_host_allowlist,
                allowed_ports=self.request_override_settings.openai_allowed_ports,
                allow_insecure_http=False,
                resolve_dns=True,
            )

        vector_size: int | None = None
        if vector_size_raw:
            try:
                vector_size = int(vector_size_raw)
            except ValueError as exc:
                raise ValueError(
                    f"{self.request_override_settings.embedding_vector_size_header} must be an integer."
                ) from exc
            if vector_size <= 0:
                raise ValueError(
                    f"{self.request_override_settings.embedding_vector_size_header} must be positive."
                )

        provider_type = EmbeddingProviderType(provider_raw)
        if provider_type == EmbeddingProviderType.FASTEMBED:
            if any([openai_api_key, openai_base_url, openai_org, openai_project]):
                raise ValueError("OpenAI credential headers require x-embedding-provider=openai.")
            configured_provider = str(self.embedding_info.provider).lower()
            configured_model = self.embedding_info.model
            configured_dimension = self.embedding_info.dim
            if configured_provider not in {"fastembed", "fastembedprovider"}:
                raise ValueError(
                    "Request FastEmbed is unavailable because the server default "
                    "embedding provider is not FastEmbed."
                )
            if model_name != configured_model:
                raise ValueError("Request FastEmbed must use the exact configured server model.")
            if vector_size is not None and vector_size != configured_dimension:
                raise ValueError(
                    "Request FastEmbed vector size must match the configured server model."
                )
            settings = EmbeddingProviderSettings.model_validate(
                {
                    "EMBEDDING_PROVIDER": provider_type.value,
                    "EMBEDDING_MODEL": configured_model,
                    "EMBEDDING_VECTOR_SIZE": configured_dimension,
                }
            )
            return self.embedding_provider, settings

        settings = EmbeddingProviderSettings.model_validate(
            {
                "EMBEDDING_PROVIDER": provider_type.value,
                "EMBEDDING_MODEL": model_name,
                "EMBEDDING_VECTOR_SIZE": vector_size,
                "OPENAI_API_KEY": (openai_api_key or None)
                if provider_type == EmbeddingProviderType.OPENAI
                else None,
                "OPENAI_BASE_URL": (openai_base_url or None)
                if provider_type == EmbeddingProviderType.OPENAI
                else None,
                "OPENAI_ORG": (openai_org or None)
                if provider_type == EmbeddingProviderType.OPENAI
                else None,
                "OPENAI_PROJECT": (openai_project or None)
                if provider_type == EmbeddingProviderType.OPENAI
                else None,
            }
        )

        # Request BYOK clients carry raw credentials; keep both the key and
        # OpenAI client scoped to this one MCP tool call.
        provider = create_embedding_provider(settings)

        return provider, settings

    def _build_request_overrides(
        self, headers: Mapping[str, Any] | None
    ) -> RequestQdrantOverrides | None:
        if not self.request_override_settings.allow_request_overrides:
            return None

        normalized = self._normalize_headers(headers)
        url = normalized.get(self.request_override_settings.qdrant_url_header, "")
        api_key = normalized.get(self.request_override_settings.qdrant_api_key_header, "")
        collection_name = normalized.get(self.request_override_settings.collection_name_header, "")
        vector_name = normalized.get(self.request_override_settings.vector_name_header, "")
        missing_required: list[str] = []
        if self.request_override_settings.require_request_qdrant_url and not url:
            missing_required.append(self.request_override_settings.qdrant_url_header)
        if self.request_override_settings.require_request_qdrant_api_key and not api_key:
            missing_required.append(self.request_override_settings.qdrant_api_key_header)
        if self.request_override_settings.require_request_collection and not collection_name:
            missing_required.append(self.request_override_settings.collection_name_header)
        if missing_required:
            raise ValueError("Missing required header(s): " + ", ".join(missing_required) + ".")

        (
            embedding_provider,
            embedding_provider_settings,
        ) = self._resolve_request_embedding_provider(normalized)
        embedding_info = (
            self._resolve_embedding_info(
                provider=embedding_provider,
                settings=embedding_provider_settings,
            )
            if embedding_provider is not None
            else None
        )

        if not any(
            [
                url,
                api_key,
                collection_name,
                vector_name,
                embedding_provider is not None,
            ]
        ):
            return None

        if self.request_override_settings.disable_default_qdrant_fallback:
            effective_url = url or None
            effective_api_key = api_key or None
            effective_collection = collection_name or None
            effective_local_path = None
        else:
            effective_url = url or self._default_qdrant_url
            effective_api_key = api_key or self._default_qdrant_api_key
            effective_collection = collection_name or self._default_collection_name
            effective_local_path = None if effective_url else self._default_local_path

        if effective_url:
            effective_url = validate_outbound_url(
                effective_url,
                allowed_hosts=self.request_override_settings.qdrant_host_allowlist,
                allowed_ports=self.request_override_settings.qdrant_allowed_ports,
                allow_insecure_http=False,
                resolve_dns=True,
            )
        elif effective_local_path is None:
            raise ValueError("Qdrant URL is required.")

        return RequestQdrantOverrides(
            url=effective_url,
            api_key=effective_api_key,
            collection_name=effective_collection,
            vector_name=vector_name or None,
            embedding_provider=embedding_provider,
            embedding_provider_settings=embedding_provider_settings,
            embedding_info=embedding_info,
        )

    def _inject_collection_name(
        self, key: str, arguments: dict[str, Any], collection_name: str
    ) -> dict[str, Any]:
        tools = getattr(self, "_registered_tools", {})
        tool = tools.get(key) if isinstance(tools, dict) else None
        if tool is None:
            return arguments
        allowed = set(tool.parameters.get("properties", {}).keys())
        if "collection_name" not in allowed or "collection_name" in arguments:
            return arguments
        updated = dict(arguments)
        updated["collection_name"] = collection_name
        return updated

    async def _call_tool_mcp(
        self, key: str, arguments: dict[str, Any]
    ) -> list[TextContent | ImageContent | EmbeddedResource]:
        # Catalog/configuration navigation is provider-local and must remain usable
        # after Portal-grant authorization but before a user configures Qdrant.
        # Data tools still pass through the strict per-request connector checks below.
        if key in self._LOCAL_NAVIGATION_TOOL_NAMES:
            return await super()._call_tool_mcp(key, arguments)

        connector_token = None
        overrides_token = None
        embedding_provider_token = None
        embedding_info_token = None
        overrides: RequestQdrantOverrides | None = None
        connector: QdrantConnector | None = None
        request_provider: EmbeddingProvider | None = None

        try:
            overrides = self._build_request_overrides(get_http_headers())
            if overrides is not None:
                if overrides.embedding_provider is None:
                    if key in self._MUTATION_TOOL_NAMES:
                        raise ValueError(
                            "Embedding headers are missing. This session is read-only; "
                            "set x-embedding-provider and x-embedding-model to enable write tools."
                        )
                    if key in self._EMBEDDING_REQUIRED_READ_TOOLS:
                        raise ValueError(
                            "Embedding headers are required for semantic search tools; "
                            "set x-embedding-provider and x-embedding-model."
                        )
                    if (
                        key == "qdrant-health-check"
                        and isinstance(arguments, dict)
                        and bool(arguments.get("warm_all"))
                    ):
                        raise ValueError(
                            "Embedding headers are required when qdrant-health-check warm_all=true."
                        )

                effective_local_path = None if overrides.url else self._default_local_path
                runtime_embedding_provider = overrides.embedding_provider or self.embedding_provider
                if (
                    overrides.embedding_provider is not None
                    and overrides.embedding_provider is not self._default_embedding_provider
                ):
                    request_provider = overrides.embedding_provider
                connector = QdrantConnector(
                    overrides.url,
                    overrides.api_key,
                    overrides.collection_name,
                    runtime_embedding_provider,
                    overrides.vector_name,
                    effective_local_path,
                    self.payload_indexes,
                )
                connector_token = self._connector_var.set(connector)
                overrides_token = self._request_overrides_var.set(overrides)
                if overrides.embedding_provider is not None:
                    embedding_provider_token = self._embedding_provider_var.set(
                        overrides.embedding_provider
                    )
                if overrides.embedding_info is not None:
                    embedding_info_token = self._embedding_info_var.set(overrides.embedding_info)
                if isinstance(arguments, dict) and overrides.collection_name:
                    arguments = self._inject_collection_name(
                        key, arguments, overrides.collection_name
                    )
            elif (
                self._default_collection_name
                and not self.request_override_settings.require_request_collection
                and not self.request_override_settings.disable_default_qdrant_fallback
                and isinstance(arguments, dict)
            ):
                arguments = self._inject_collection_name(
                    key, arguments, self._default_collection_name
                )

            return await super()._call_tool_mcp(key, arguments)
        finally:
            if embedding_info_token is not None:
                self._embedding_info_var.reset(embedding_info_token)
            if embedding_provider_token is not None:
                self._embedding_provider_var.reset(embedding_provider_token)
            if overrides_token is not None:
                self._request_overrides_var.reset(overrides_token)
            if connector_token is not None:
                self._connector_var.reset(connector_token)
            if overrides is not None and not overrides.resources_claimed:
                await self._close_request_resources(connector, request_provider)
