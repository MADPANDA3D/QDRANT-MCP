from __future__ import annotations

from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

try:  # FastMCP >= 2.2.11
    from fastmcp.server.dependencies import get_http_headers
except ImportError:  # pragma: no cover - older FastMCP

    def get_http_headers() -> dict[str, str]:
        return {}


from mcp.types import EmbeddedResource, ImageContent, TextContent

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.mcp_server import QdrantMCPServer, RequestQdrantOverrides
from mcp_server_qdrant.qdrant import QdrantConnector
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    MemorySettings,
    QdrantSettings,
    RequestOverrideSettings,
    ToolSettings,
)


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
        "qdrant-ingest-textbook",
        "qdrant-ensure-payload-indexes",
        "qdrant-backfill-memory-contract",
        "qdrant-update-point",
        "qdrant-patch-payload",
        "qdrant-tag-memories",
        "qdrant-link-memories",
        "qdrant-reembed-points",
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
        "qdrant-restore-snapshot",
        "qdrant-submit-job",
        "qdrant-cancel-job",
        "qdrant-cancel-ingest",
    }
    _EMBEDDING_REQUIRED_READ_TOOLS = {
        "qdrant-find",
        "qdrant-find-short-term",
    }

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
        self.request_override_settings = (
            request_override_settings or RequestOverrideSettings()
        )
        self._default_qdrant_url = qdrant_settings.location
        self._default_qdrant_api_key = qdrant_settings.api_key
        self._default_collection_name = qdrant_settings.collection_name
        self._default_vector_name = qdrant_settings.vector_name
        self._default_local_path = qdrant_settings.local_path
        self._embedding_provider_cache: dict[tuple[str, ...], EmbeddingProvider] = {}

        if self.request_override_settings.allow_request_overrides:
            qdrant_settings = qdrant_settings.model_copy()
            qdrant_settings.collection_name = None

        super().__init__(
            tool_settings=tool_settings,
            qdrant_settings=qdrant_settings,
            memory_settings=memory_settings,
            embedding_provider_settings=embedding_provider_settings,
            embedding_provider=embedding_provider,
            name=name,
            instructions=instructions,
            **settings,
        )

        if self.request_override_settings.allow_request_overrides:
            self.qdrant_connector = QdrantConnector(
                self._default_qdrant_url,
                self._default_qdrant_api_key,
                self._default_collection_name,
                self.embedding_provider,
                self._default_vector_name,
                self._default_local_path,
                self.payload_indexes,
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

    def _host_allowed(self, host: str) -> bool:
        allowlist = self.request_override_settings.qdrant_host_allowlist
        if not allowlist:
            return True
        host = host.lower()
        for allowed in allowlist:
            if allowed.startswith("*.") and host.endswith(allowed[1:]):
                return True
            if host == allowed:
                return True
        return False

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
        openai_api_key = normalized.get(
            self.request_override_settings.openai_api_key_header, ""
        )
        openai_base_url = normalized.get(
            self.request_override_settings.openai_base_url_header, ""
        )
        openai_org = normalized.get(
            self.request_override_settings.openai_organization_header, ""
        )
        openai_project = normalized.get(
            self.request_override_settings.openai_project_header, ""
        )

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
            missing_required.append(
                self.request_override_settings.embedding_provider_header
            )
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
            raise ValueError(
                "Missing required header(s): " + ", ".join(missing_required) + "."
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

        cache_key = (
            provider_type.value,
            model_name,
            str(vector_size or ""),
            settings.openai_api_key or "",
            settings.openai_base_url or "",
            settings.openai_organization or "",
            settings.openai_project or "",
        )
        provider = self._embedding_provider_cache.get(cache_key)
        if provider is None:
            provider = create_embedding_provider(settings)
            if len(self._embedding_provider_cache) >= 64:
                self._embedding_provider_cache.clear()
            self._embedding_provider_cache[cache_key] = provider

        return provider, settings

    def _build_request_overrides(
        self, headers: Mapping[str, Any] | None
    ) -> RequestQdrantOverrides | None:
        if not self.request_override_settings.allow_request_overrides:
            return None

        normalized = self._normalize_headers(headers)
        url = normalized.get(self.request_override_settings.qdrant_url_header, "")
        api_key = normalized.get(
            self.request_override_settings.qdrant_api_key_header, ""
        )
        collection_name = normalized.get(
            self.request_override_settings.collection_name_header, ""
        )
        vector_name = normalized.get(
            self.request_override_settings.vector_name_header, ""
        )
        missing_required: list[str] = []
        if self.request_override_settings.require_request_qdrant_url and not url:
            missing_required.append(self.request_override_settings.qdrant_url_header)
        if self.request_override_settings.require_request_qdrant_api_key and not api_key:
            missing_required.append(
                self.request_override_settings.qdrant_api_key_header
            )
        if (
            self.request_override_settings.require_request_collection
            and not collection_name
        ):
            missing_required.append(
                self.request_override_settings.collection_name_header
            )
        if missing_required:
            raise ValueError(
                "Missing required header(s): " + ", ".join(missing_required) + "."
            )

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
            parsed = urlparse(effective_url)
            if parsed.scheme not in {"http", "https"}:
                raise ValueError("Qdrant URL must start with http:// or https://")
            host = parsed.hostname
            if not host:
                raise ValueError("Qdrant URL must include a hostname.")
            if not self._host_allowed(host):
                raise ValueError("Qdrant host is not allowed.")
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
        if not self._tool_manager.has_tool(key):
            return arguments
        tool = self._tool_manager.get_tool(key)
        allowed = set(tool.parameters.get("properties", {}).keys())
        if "collection_name" not in allowed or "collection_name" in arguments:
            return arguments
        updated = dict(arguments)
        updated["collection_name"] = collection_name
        return updated

    async def _mcp_call_tool(
        self, key: str, arguments: dict[str, Any]
    ) -> list[TextContent | ImageContent | EmbeddedResource]:
        connector_token = None
        overrides_token = None
        embedding_provider_token = None
        embedding_info_token = None
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
            runtime_embedding_provider = (
                overrides.embedding_provider or self.embedding_provider
            )
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
                embedding_info_token = self._embedding_info_var.set(
                    overrides.embedding_info
                )
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

        try:
            return await super()._mcp_call_tool(key, arguments)
        finally:
            if embedding_info_token is not None:
                self._embedding_info_var.reset(embedding_info_token)
            if embedding_provider_token is not None:
                self._embedding_provider_var.reset(embedding_provider_token)
            if overrides_token is not None:
                self._request_overrides_var.reset(overrides_token)
            if connector_token is not None:
                self._connector_var.reset(connector_token)
