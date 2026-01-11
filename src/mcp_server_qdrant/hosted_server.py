from __future__ import annotations

from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

try:  # FastMCP >= 2.2.11
    from fastmcp.server.dependencies import get_http_headers
except ImportError:  # pragma: no cover - older FastMCP
    def get_http_headers() -> dict[str, str]:
        return {}

from mcp.types import EmbeddedResource, ImageContent, TextContent
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.qdrant import QdrantConnector
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    MemorySettings,
    QdrantSettings,
    RequestOverrideSettings,
    ToolSettings,
)


@dataclass
class RequestQdrantOverrides:
    url: str | None
    api_key: str | None
    collection_name: str | None
    vector_name: str | None
    local_path: str | None


class HostedQdrantMCPServer(QdrantMCPServer):
    """
    Qdrant MCP server with per-request connection overrides via HTTP headers.
    """

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
        self._connector_var: ContextVar[QdrantConnector | None] = ContextVar(
            "qdrant_connector",
            default=None,
        )
        self._request_overrides_var: ContextVar[
            RequestQdrantOverrides | None
        ] = ContextVar("qdrant_request_overrides", default=None)
        self._default_qdrant_connector: QdrantConnector | None = None
        self._default_qdrant_url = qdrant_settings.location
        self._default_qdrant_api_key = qdrant_settings.api_key
        self._default_collection_name = qdrant_settings.collection_name
        self._default_vector_name = qdrant_settings.vector_name
        self._default_local_path = qdrant_settings.local_path

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
        if self.request_override_settings.require_request_collection and not collection_name:
            missing_required.append(
                self.request_override_settings.collection_name_header
            )
        if missing_required:
            raise ValueError(
                "Missing required header(s): " + ", ".join(missing_required) + "."
            )

        if not any([url, api_key, collection_name, vector_name]):
            return None

        effective_url = url or self._default_qdrant_url
        effective_api_key = api_key or self._default_qdrant_api_key
        effective_collection = collection_name or self._default_collection_name
        effective_local_path = None if url else self._default_local_path

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
            local_path=effective_local_path,
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
        overrides = self._build_request_overrides(get_http_headers())
        if overrides is not None:
            connector = QdrantConnector(
                overrides.url,
                overrides.api_key,
                overrides.collection_name,
                self.embedding_provider,
                overrides.vector_name,
                overrides.local_path,
                self.payload_indexes,
            )
            connector_token = self._connector_var.set(connector)
            overrides_token = self._request_overrides_var.set(overrides)
            if isinstance(arguments, dict) and overrides.collection_name:
                arguments = self._inject_collection_name(
                    key, arguments, overrides.collection_name
                )
        elif (
            self._default_collection_name
            and not self.request_override_settings.require_request_collection
            and isinstance(arguments, dict)
        ):
            arguments = self._inject_collection_name(
                key, arguments, self._default_collection_name
            )

        try:
            return await super()._mcp_call_tool(key, arguments)
        finally:
            if overrides_token is not None:
                self._request_overrides_var.reset(overrides_token)
            if connector_token is not None:
                self._connector_var.reset(connector_token)
