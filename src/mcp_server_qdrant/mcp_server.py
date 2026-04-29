import asyncio
import base64
import difflib
import hashlib
import json
import logging
import math
import tempfile
import time
import uuid
from collections import OrderedDict
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Annotated, Any, Literal
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from fastmcp import Context, FastMCP

try:  # FastMCP >= 2.2.11
    from fastmcp.server.dependencies import get_http_headers
except ImportError:  # pragma: no cover - older FastMCP

    def get_http_headers() -> dict[str, str]:
        return {}


from mcp.types import EmbeddedResource, ImageContent, TextContent, ToolAnnotations
from pydantic import Field
from qdrant_client import models

from mcp_server_qdrant.common.filters import make_indexes
from mcp_server_qdrant.common.func_tools import make_partial_function
from mcp_server_qdrant.common.telemetry import (
    SERVER_INSTANCE_ID,
    SERVER_START,
    finish_request,
    new_request,
)
from mcp_server_qdrant.common.wrap_filters import wrap_filters
from mcp_server_qdrant.document_ingest import (
    chunk_text_with_overlap,
    detect_pdf_chapter_markers,
    extract_document_sections,
    normalize_chapter_map,
    normalize_text_for_chunking,
    resolve_chapter_metadata_for_page,
)
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.memory import (
    ALLOWED_MEMORY_KEYS,
    DEFAULT_CONFIDENCE,
    DEFAULT_MEMORY_TYPE,
    DEFAULT_SCOPE,
    DEFAULT_SOURCE,
    FILTER_FIELDS,
    REQUIRED_FIELDS,
    EmbeddingInfo,
    MemoryFilterInput,
    build_memory_backfill_patch,
    build_memory_filter,
    compute_text_hash,
    default_memory_indexes,
    normalize_memory_input,
)
from mcp_server_qdrant.qdrant import ArbitraryFilter, Entry, Metadata, QdrantConnector
from mcp_server_qdrant.settings import (
    METADATA_PATH,
    EmbeddingProviderSettings,
    MemorySettings,
    QdrantSettings,
    RequestOverrideSettings,
    ToolSettings,
)

logger = logging.getLogger(__name__)


@dataclass
class RequestQdrantOverrides:
    url: str | None
    api_key: str | None
    collection_name: str | None
    vector_name: str | None
    embedding_provider: EmbeddingProvider | None = None
    embedding_provider_settings: EmbeddingProviderSettings | None = None
    embedding_info: EmbeddingInfo | None = None


class StructuredIngestError(Exception):
    def __init__(
        self,
        *,
        error_code: str,
        suggested_http_status: int,
        stage: str,
        message: str,
        limit_name: str | None = None,
        limit_value: int | None = None,
        actual_value: int | None = None,
    ) -> None:
        super().__init__(message)
        self.error_code = error_code
        self.suggested_http_status = suggested_http_status
        self.stage = stage
        self.message = message
        self.limit_name = limit_name
        self.limit_value = limit_value
        self.actual_value = actual_value

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "error_code": self.error_code,
            "suggested_http_status": self.suggested_http_status,
            "stage": self.stage,
            "message": self.message,
        }
        if self.limit_name is not None:
            payload["limit_name"] = self.limit_name
        if self.limit_value is not None:
            payload["limit_value"] = self.limit_value
        if self.actual_value is not None:
            payload["actual_value"] = self.actual_value
        return payload


# FastMCP is an alternative interface for declaring the capabilities
# of the server. Its API is based on FastAPI.
class QdrantMCPServer(FastMCP):
    """
    A MCP server for Qdrant.
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
    ):
        self.tool_settings = tool_settings
        self.qdrant_settings = qdrant_settings
        self.request_override_settings = (
            request_override_settings or RequestOverrideSettings()
        )
        self.memory_settings = memory_settings or MemorySettings()
        self._embedding_provider_var: ContextVar[EmbeddingProvider | None] = ContextVar(
            "embedding_provider",
            default=None,
        )
        self._embedding_info_var: ContextVar[EmbeddingInfo | None] = ContextVar(
            "embedding_info",
            default=None,
        )

        if embedding_provider_settings and embedding_provider:
            raise ValueError(
                "Cannot provide both embedding_provider_settings and embedding_provider"
            )

        if not embedding_provider_settings and not embedding_provider:
            raise ValueError(
                "Must provide either embedding_provider_settings or embedding_provider"
            )

        self.embedding_provider_settings: EmbeddingProviderSettings | None
        self.embedding_provider: EmbeddingProvider

        if embedding_provider_settings:
            self.embedding_provider_settings = embedding_provider_settings
            self.embedding_provider = create_embedding_provider(
                embedding_provider_settings
            )
        elif embedding_provider:
            self.embedding_provider_settings = None
            self.embedding_provider = embedding_provider
        else:
            raise ValueError(
                "Must provide either embedding_provider_settings or embedding_provider"
            )
        self.embedding_info = self._resolve_embedding_info()

        field_indexes = default_memory_indexes()
        field_indexes.update(make_indexes(qdrant_settings.filterable_fields_dict()))
        self.payload_indexes = dict(field_indexes)

        self._default_qdrant_connector = QdrantConnector(
            qdrant_settings.location,
            qdrant_settings.api_key,
            qdrant_settings.collection_name,
            self.embedding_provider,
            qdrant_settings.vector_name,
            qdrant_settings.local_path,
            field_indexes,
        )
        self._connector_var: ContextVar[QdrantConnector | None] = ContextVar(
            "qdrant_connector",
            default=None,
        )
        self._request_overrides_var: ContextVar[RequestQdrantOverrides | None] = (
            ContextVar("qdrant_request_overrides", default=None)
        )
        self._jobs: dict[str, dict[str, Any]] = {}
        self._job_tasks: dict[str, asyncio.Task] = {}
        self._query_embedding_cache: OrderedDict[
            tuple[str, str, int, str, str], tuple[float, list[float]]
        ] = OrderedDict()

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    @property
    def qdrant_connector(self) -> QdrantConnector:
        connector = self._connector_var.get()
        return connector or self._default_qdrant_connector

    @property
    def embedding_provider(self) -> EmbeddingProvider:
        provider = self._embedding_provider_var.get()
        if provider is not None:
            return provider
        return self._default_embedding_provider

    @embedding_provider.setter
    def embedding_provider(self, value: EmbeddingProvider) -> None:
        self._default_embedding_provider = value

    @property
    def embedding_info(self) -> EmbeddingInfo:
        info = self._embedding_info_var.get()
        if info is not None:
            return info
        return self._default_embedding_info

    @embedding_info.setter
    def embedding_info(self, value: EmbeddingInfo) -> None:
        self._default_embedding_info = value

    def _query_embedding_cache_key(self, query: str) -> tuple[str, str, int, str, str]:
        info = self.embedding_info
        query_hash = hashlib.sha256(query.encode("utf-8")).hexdigest()
        provider_identity = str(id(self.embedding_provider))
        return (info.provider, info.model, info.dim, provider_identity, query_hash)

    async def _embed_query_cached(self, query: str) -> list[float]:
        cache_size = self.memory_settings.query_embedding_cache_size
        cache_ttl = self.memory_settings.query_embedding_cache_ttl_seconds
        if cache_size <= 0 or cache_ttl <= 0:
            return await self.embedding_provider.embed_query(query)

        key = self._query_embedding_cache_key(query)
        now = time.monotonic()
        cached = self._query_embedding_cache.get(key)
        if cached is not None:
            cached_at, vector = cached
            if now - cached_at <= cache_ttl:
                self._query_embedding_cache.move_to_end(key)
                return list(vector)
            self._query_embedding_cache.pop(key, None)

        vector = await self.embedding_provider.embed_query(query)
        self._query_embedding_cache[key] = (now, list(vector))
        self._query_embedding_cache.move_to_end(key)
        while len(self._query_embedding_cache) > cache_size:
            self._query_embedding_cache.popitem(last=False)
        return vector

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

        if not any([url, api_key, collection_name, vector_name]):
            return None

        if url:
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"}:
                raise ValueError("Qdrant URL must start with http:// or https://")
            host = parsed.hostname
            if not host:
                raise ValueError("Qdrant URL must include a hostname.")
            if not self._host_allowed(host):
                raise ValueError("Qdrant host is not allowed.")

        return RequestQdrantOverrides(
            url=url or None,
            api_key=api_key or None,
            collection_name=collection_name or None,
            vector_name=vector_name or None,
        )

    def _get_default_collection_name(self) -> str | None:
        overrides = self._request_overrides_var.get()
        if overrides and overrides.collection_name:
            return overrides.collection_name
        return self.qdrant_settings.collection_name

    async def _mcp_call_tool(
        self, key: str, arguments: dict[str, Any]
    ) -> list[TextContent | ImageContent | EmbeddedResource]:
        """
        Normalize tool arguments before validation to tolerate MCP clients
        that wrap arguments inside Airtable-like records (id/createdTime/fields).
        """
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
                None,
                self.payload_indexes,
            )
            connector_token = self._connector_var.set(connector)
            overrides_token = self._request_overrides_var.set(overrides)

        if isinstance(arguments, dict) and self._tool_manager.has_tool(key):
            tool = self._tool_manager.get_tool(key)
            allowed = set(tool.parameters.get("properties", {}).keys())
            filtered: dict[str, Any] = {}
            unknown: set[str] = set()
            fields = arguments.get("fields")
            if isinstance(fields, dict):
                for arg_key, arg_value in fields.items():
                    if arg_key in allowed:
                        filtered[arg_key] = arg_value
                    else:
                        unknown.add(arg_key)
            for arg_key, arg_value in arguments.items():
                if arg_key in allowed:
                    filtered[arg_key] = arg_value
                elif arg_key != "fields":
                    unknown.add(arg_key)
            if unknown and self.memory_settings.strict_params:
                raise ValueError(f"Unknown tool parameters: {sorted(unknown)}")
            arguments = filtered

        try:
            return await super()._mcp_call_tool(key, arguments)
        finally:
            if overrides_token is not None:
                self._request_overrides_var.reset(overrides_token)
            if connector_token is not None:
                self._connector_var.reset(connector_token)

    def format_entry(self, entry: Entry) -> str:
        """
        Feel free to override this method in your subclass to customize the format of the entry.
        """
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        return f"<entry><content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>"

    def _resolve_embedding_info(
        self,
        provider: EmbeddingProvider | None = None,
        settings: EmbeddingProviderSettings | None = None,
    ) -> EmbeddingInfo:
        model_name = "unknown"
        provider_name = "unknown"
        version = "unknown"

        provider_obj = provider or self.embedding_provider
        settings_obj = (
            settings if settings is not None else self.embedding_provider_settings
        )

        if settings_obj:
            provider_name = settings_obj.provider_type.value
            model_name = settings_obj.model_name
            version = settings_obj.version or model_name
        else:
            provider_name = (
                getattr(provider_obj, "provider_type", None)
                or provider_obj.__class__.__name__.lower()
            )
            model_name = getattr(provider_obj, "model_name", "unknown")
            version = getattr(provider_obj, "version", None) or model_name

        dim = provider_obj.get_vector_size()
        return EmbeddingInfo(
            provider=provider_name,
            model=model_name,
            dim=dim,
            version=version,
        )

    def setup_tools(self):
        """
        Register the tools in the server.
        """

        missing_collection_message = (
            "Tool requires a collection, provide `collection_name` arg or "
            "`X-Collection-Name` header."
        )

        def resolve_collection_name(collection_name: str | None) -> str:
            name = collection_name.strip() if collection_name else ""
            if name:
                return name
            default_name = self._get_default_collection_name()
            if default_name:
                return default_name
            raise ValueError(missing_collection_message)

        def resolve_health_collection(collection_name: str | None) -> str:
            name = collection_name.strip() if collection_name else ""
            if name:
                return name
            default_name = self._get_default_collection_name()
            if default_name:
                return default_name
            if self.memory_settings.health_check_collection:
                return self.memory_settings.health_check_collection
            return "jarvis-knowledge-base"

        def merge_filters(filters: list[models.Filter | None]) -> models.Filter | None:
            must = []
            should = []
            must_not = []
            for current in filters:
                if not current:
                    continue
                if current.must:
                    must.extend(current.must)
                if current.should:
                    should.extend(current.should)
                if current.must_not:
                    must_not.extend(current.must_not)
            if not must and not should and not must_not:
                return None
            return models.Filter(
                must=must or None,
                should=should or None,
                must_not=must_not or None,
            )

        def extract_payload_text(payload: dict[str, Any]) -> str | None:
            if not payload:
                return None
            for key in ("document", "content", "text"):
                value = payload.get(key)
                if isinstance(value, str):
                    return value
            metadata = payload.get(METADATA_PATH) or payload.get("metadata")
            if isinstance(metadata, dict):
                value = metadata.get("text")
                if isinstance(value, str):
                    return value
            return None

        def serialize_model(value: Any) -> Any:
            if value is None:
                return None
            if hasattr(value, "model_dump"):
                return value.model_dump()
            if hasattr(value, "dict"):
                return value.dict()
            return value

        def make_snippet(text: str | None, max_length: int = 160) -> str:
            if not text:
                return ""
            cleaned = " ".join(str(text).split())
            if len(cleaned) <= max_length:
                return cleaned
            return cleaned[: max_length - 3] + "..."

        def safe_json_size(value: Any) -> int:
            try:
                return len(json.dumps(value, default=str).encode("utf-8"))
            except TypeError:
                return 0

        DRY_RUN_PREVIEW_LIMIT = 5
        DRY_RUN_GROUP_FIELDS = ("scope", "type", "labels", "source", "doc_id")
        SEARCH_COMPACT_METADATA_FIELDS = (
            "type",
            "class",
            "subject",
            "module",
            "week",
            "status",
            "year",
            "material_type",
            "title",
            "author",
            "edition",
            "isbn",
            "publisher",
            "chapter",
            "chapter_title",
            "labels",
            "doc_id",
            "doc_title",
            "file_type",
            "page_start",
            "page_end",
            "section_heading",
        )
        CONTEXT_METADATA_FIELDS = (
            "type",
            "class",
            "subject",
            "module",
            "week",
            "status",
            "year",
            "material_type",
            "title",
            "doc_title",
            "author",
            "doc_id",
            "source_url",
            "file_name",
            "file_type",
            "chapter",
            "chapter_title",
            "page_start",
            "page_end",
            "section_heading",
            "labels",
        )
        DRY_RUN_PREVIEW_FIELDS = (
            "text",
            "type",
            "scope",
            "source",
            "labels",
            "doc_id",
            "doc_title",
            "created_at",
            "updated_at",
            "expires_at",
            "confidence",
            "embedding_version",
            "embedding_model",
            "embedding_provider",
            "text_hash",
            "merged_into",
            "merged_from",
        )
        DRY_RUN_MAX_LIST_ITEMS = 10
        JOB_LOG_LIMIT = 200
        PREVIEW_SCAN_LIMIT = 2000
        TEXTBOOK_REQUIRED_METADATA = (
            "class",
            "material_type",
            "title",
            "author",
            "edition",
            "isbn",
        )
        TEXTBOOK_OPTIONAL_METADATA = ("publisher", "chapter", "chapter_title")

        def extract_vector(
            raw_vector: Any, vector_name: str | None
        ) -> list[float] | None:
            if isinstance(raw_vector, dict):
                if vector_name and vector_name in raw_vector:
                    return raw_vector[vector_name]
                if len(raw_vector) == 1:
                    return next(iter(raw_vector.values()))
                return None
            if isinstance(raw_vector, list):
                return raw_vector
            return None

        def extract_metadata(payload: dict[str, Any]) -> dict[str, Any]:
            if not payload:
                return {}
            metadata = payload.get(METADATA_PATH) or payload.get("metadata") or {}
            return metadata if isinstance(metadata, dict) else {}

        def compact_value(value: Any) -> Any:
            if isinstance(value, str):
                return make_snippet(value, max_length=200)
            if isinstance(value, list):
                if len(value) <= DRY_RUN_MAX_LIST_ITEMS:
                    return [compact_value(item) for item in value]
                trimmed = [
                    compact_value(item) for item in value[:DRY_RUN_MAX_LIST_ITEMS]
                ]
                trimmed.append(f"...(+{len(value) - DRY_RUN_MAX_LIST_ITEMS})")
                return trimmed
            return value

        def compact_metadata(
            metadata: dict[str, Any], keys: tuple[str, ...] | set[str] | None = None
        ) -> dict[str, Any]:
            if not isinstance(metadata, dict):
                return {}
            if keys is None:
                keys = set(metadata.keys())
            result: dict[str, Any] = {}
            for key in keys:
                if key in metadata:
                    result[key] = compact_value(metadata.get(key))
            return result

        def clamp_snippet_chars(value: int) -> int:
            return max(40, min(value, 1200))

        def clamp_max_output_chars(value: int | None) -> int | None:
            if value is None:
                return None
            return max(800, min(value, 50000))

        def normalize_metadata_fields(
            fields: list[str] | tuple[str, ...] | None,
        ) -> tuple[str, ...] | None:
            if not fields:
                return None
            normalized: list[str] = []
            for raw_field in fields:
                if not isinstance(raw_field, str):
                    continue
                field = raw_field.strip()
                if not field:
                    continue
                if field.startswith(f"{METADATA_PATH}."):
                    field = field.removeprefix(f"{METADATA_PATH}.")
                if field not in normalized:
                    normalized.append(field)
                if len(normalized) >= 40:
                    break
            return tuple(normalized) if normalized else None

        def point_group_key(point: models.ScoredPoint) -> str:
            metadata = extract_metadata(point.payload or {})
            for key in (
                "doc_id",
                "doc_title",
                "title",
                "source_url",
                "file_name",
                "text_hash",
            ):
                value = metadata.get(key)
                if value is not None and str(value).strip():
                    return f"{key}:{value}"
            return f"id:{point.id}"

        def select_points_for_output(
            points: list[models.ScoredPoint],
            *,
            limit: int,
            group_by_doc: bool,
            max_chunks_per_doc: int,
            min_score: float | None,
        ) -> list[models.ScoredPoint]:
            if limit <= 0:
                return []
            selected: list[models.ScoredPoint] = []
            group_counts: dict[str, int] = {}
            chunk_limit = max(1, max_chunks_per_doc)
            for point in points:
                if min_score is not None and point.score < min_score:
                    continue
                if group_by_doc:
                    group_key = point_group_key(point)
                    current_count = group_counts.get(group_key, 0)
                    if current_count >= chunk_limit:
                        continue
                    group_counts[group_key] = current_count + 1
                selected.append(point)
                if len(selected) >= limit:
                    break
            return selected

        def trim_results_to_budget(
            results: list[dict[str, Any]],
            *,
            base_data: dict[str, Any],
            max_output_chars: int | None,
        ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
            budget = clamp_max_output_chars(max_output_chars)
            if budget is None:
                estimated = safe_json_size({**base_data, "results": results})
                return results, {
                    "max_output_chars": None,
                    "estimated_output_chars": estimated,
                    "truncated": False,
                }

            kept: list[dict[str, Any]] = []
            for result in results:
                candidate = [*kept, result]
                estimated = safe_json_size({**base_data, "results": candidate})
                if estimated <= budget or not kept:
                    kept = candidate
                    continue
                break

            return kept, {
                "max_output_chars": budget,
                "estimated_output_chars": safe_json_size(
                    {**base_data, "results": kept}
                ),
                "truncated": len(kept) < len(results),
            }

        def format_search_result(
            point: models.ScoredPoint,
            *,
            response_mode: str,
            snippet_chars: int,
            metadata_fields: tuple[str, ...] | None = None,
        ) -> dict[str, Any]:
            payload = point.payload or {}
            metadata = extract_metadata(payload)
            text = extract_payload_text(payload)
            result: dict[str, Any] = {
                "id": str(point.id),
                "score": point.score,
                "snippet": make_snippet(text, max_length=snippet_chars),
            }
            if response_mode == "compact":
                selected = compact_metadata(
                    metadata, metadata_fields or SEARCH_COMPACT_METADATA_FIELDS
                )
                if selected:
                    result["metadata"] = selected
            elif response_mode == "metadata":
                result["metadata"] = compact_metadata(metadata, metadata_fields)
            elif response_mode == "payload":
                result["payload"] = payload
            else:
                raise ValueError(
                    "response_mode must be one of: compact, metadata, payload."
                )
            return result

        def metadata_value_from_payload(
            payload: dict[str, Any], field_name: str
        ) -> Any:
            field = field_name.strip()
            if field.startswith(f"{METADATA_PATH}."):
                field = field.removeprefix(f"{METADATA_PATH}.")
            metadata = extract_metadata(payload)
            if field in metadata:
                return metadata[field]
            if field in payload:
                return payload[field]
            return None

        def add_sample_value(
            samples: dict[str, list[Any]],
            field_name: str,
            value: Any,
            *,
            max_values: int,
        ) -> None:
            if value is None:
                return
            values = value if isinstance(value, list) else [value]
            existing = samples.setdefault(field_name, [])
            seen = {str(item).lower() for item in existing}
            for item in values:
                if item is None:
                    continue
                compacted = compact_value(item)
                key = str(compacted).strip().lower()
                if not key or key in seen:
                    continue
                existing.append(compacted)
                seen.add(key)
                if len(existing) >= max_values:
                    break

        def memory_filter_key_for_field(field_name: str) -> str:
            if field_name == "class":
                return "class_code"
            return field_name

        def build_context_text(items: list[dict[str, Any]]) -> str:
            lines: list[str] = []
            for item in items:
                metadata = item.get("metadata") or {}
                title = (
                    metadata.get("title")
                    or metadata.get("doc_title")
                    or metadata.get("file_name")
                    or metadata.get("source_url")
                    or item.get("id")
                )
                score = item.get("score")
                score_text = f" score={score:.4f}" if isinstance(score, float) else ""
                lines.append(
                    f"{item['citation']} {title} ({item['id']}{score_text})\n"
                    f"{item.get('snippet', '')}"
                )
            return "\n\n".join(lines)

        def trim_context_to_budget(
            items: list[dict[str, Any]],
            *,
            base_data: dict[str, Any],
            max_output_chars: int,
        ) -> tuple[list[dict[str, Any]], str, dict[str, Any]]:
            budget = clamp_max_output_chars(max_output_chars) or 4000
            kept: list[dict[str, Any]] = []
            context_text = ""
            for item in items:
                candidate = [*kept, item]
                candidate_text = build_context_text(candidate)
                estimated_data = {
                    **base_data,
                    "context": candidate,
                    "context_text": candidate_text,
                }
                estimated = safe_json_size(estimated_data)
                if estimated <= budget or not kept:
                    kept = candidate
                    context_text = candidate_text
                    continue
                break
            return (
                kept,
                context_text,
                {
                    "max_output_chars": budget,
                    "estimated_output_chars": safe_json_size(
                        {
                            **base_data,
                            "context": kept,
                            "context_text": context_text,
                        }
                    ),
                    "truncated": len(kept) < len(items),
                },
            )

        def build_structured_error_payload(
            *,
            error_code: str,
            suggested_http_status: int,
            stage: str,
            message: str,
            limit_name: str | None = None,
            limit_value: int | None = None,
            actual_value: int | None = None,
        ) -> dict[str, Any]:
            payload: dict[str, Any] = {
                "error_code": error_code,
                "suggested_http_status": suggested_http_status,
                "stage": stage,
                "message": message,
            }
            if limit_name is not None:
                payload["limit_name"] = limit_name
            if limit_value is not None:
                payload["limit_value"] = limit_value
            if actual_value is not None:
                payload["actual_value"] = actual_value
            return payload

        def normalized_required_text(
            metadata: dict[str, Any],
            key: str,
            *,
            stage: str,
        ) -> str:
            value = metadata.get(key)
            if not isinstance(value, str) or not value.strip():
                raise StructuredIngestError(
                    error_code="textbook_validation_error",
                    suggested_http_status=422,
                    stage=stage,
                    message=(
                        f"metadata.{key} is required and must be a non-empty string."
                    ),
                )
            return value.strip()

        def normalize_textbook_metadata(
            metadata: Metadata | None,
            *,
            stage: str = "validate_input",
        ) -> dict[str, Any]:
            if not isinstance(metadata, dict):
                raise StructuredIngestError(
                    error_code="textbook_validation_error",
                    suggested_http_status=422,
                    stage=stage,
                    message="metadata is required and must be an object.",
                )

            normalized = dict(metadata)
            for key in TEXTBOOK_REQUIRED_METADATA:
                normalized[key] = normalized_required_text(normalized, key, stage=stage)

            for key in TEXTBOOK_OPTIONAL_METADATA:
                value = normalized.get(key)
                if value is None:
                    continue
                if isinstance(value, str):
                    normalized[key] = value.strip()
                elif key == "chapter" and isinstance(value, int):
                    normalized[key] = value
                else:
                    expected = "a string or integer" if key == "chapter" else "a string"
                    raise StructuredIngestError(
                        error_code="textbook_validation_error",
                        suggested_http_status=422,
                        stage=stage,
                        message=f"metadata.{key} must be {expected}.",
                    )
            return normalized

        def build_textbook_doc_id(metadata: dict[str, Any]) -> str:
            key_parts = [
                str(metadata[field]).strip().lower()
                for field in TEXTBOOK_REQUIRED_METADATA
            ]
            raw = "|".join(key_parts)
            digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
            return f"textbook-{digest[:24]}"

        def build_textbook_ingest_fingerprint(
            *,
            source_url: str,
            doc_hash: str,
            metadata: dict[str, Any],
        ) -> str:
            key_parts = [
                source_url.strip(),
                doc_hash.strip(),
                *[
                    str(metadata[field]).strip().lower()
                    for field in TEXTBOOK_REQUIRED_METADATA
                ],
            ]
            return hashlib.sha256("|".join(key_parts).encode("utf-8")).hexdigest()

        def raise_limit_error(
            *,
            stage: str,
            limit_name: str,
            limit_value: int,
            actual_value: int,
            message: str | None = None,
        ) -> None:
            detail = message or (
                f"{limit_name} exceeded. limit={limit_value}, actual={actual_value}."
            )
            raise StructuredIngestError(
                error_code="textbook_limit_exceeded",
                suggested_http_status=413,
                stage=stage,
                message=detail,
                limit_name=limit_name,
                limit_value=limit_value,
                actual_value=actual_value,
            )

        def chunks_of(items: list[Any], size: int) -> list[list[Any]]:
            if size <= 0:
                return [items]
            return [items[i : i + size] for i in range(0, len(items), size)]

        def diff_metadata(
            before: dict[str, Any],
            after: dict[str, Any],
            fallback_keys: tuple[str, ...] | None = None,
        ) -> tuple[dict[str, Any], dict[str, Any]]:
            before = before or {}
            after = after or {}
            keys = {
                key
                for key in set(before.keys()) | set(after.keys())
                if before.get(key) != after.get(key)
            }
            if not keys and fallback_keys:
                keys = set(fallback_keys)
            return compact_metadata(before, keys), compact_metadata(after, keys)

        def init_dry_run_diff(
            sample_limit: int = DRY_RUN_PREVIEW_LIMIT,
        ) -> dict[str, Any]:
            return {
                "sample_limit": sample_limit,
                "samples": [],
                "id_sample": [],
                "group_counts": {field: {} for field in DRY_RUN_GROUP_FIELDS},
                "action_counts": {},
                "affected": 0,
            }

        def add_group_count(
            group_counts: dict[str, dict[str, int]],
            field: str,
            value: str,
        ) -> None:
            bucket = group_counts.setdefault(field, {})
            bucket[value] = bucket.get(value, 0) + 1

        def update_group_counts(
            group_counts: dict[str, dict[str, int]],
            metadata: dict[str, Any],
        ) -> None:
            for field in DRY_RUN_GROUP_FIELDS:
                raw = metadata.get(field)
                if field == "labels":
                    if isinstance(raw, list) and raw:
                        for item in raw:
                            add_group_count(group_counts, field, str(item))
                    elif raw:
                        add_group_count(group_counts, field, str(raw))
                    else:
                        add_group_count(group_counts, field, "(missing)")
                    continue
                if raw is None or raw == "":
                    add_group_count(group_counts, field, "(missing)")
                else:
                    add_group_count(group_counts, field, str(raw))

        def record_dry_run_action(
            diff: dict[str, Any],
            action: str,
            point_id: str,
            before_metadata: dict[str, Any] | None,
            after_metadata: dict[str, Any] | None,
        ) -> None:
            diff["affected"] += 1
            diff["action_counts"][action] = diff["action_counts"].get(action, 0) + 1
            metadata_for_group = before_metadata or after_metadata or {}
            update_group_counts(diff["group_counts"], metadata_for_group)
            if len(diff["id_sample"]) < diff["sample_limit"]:
                diff["id_sample"].append(str(point_id))
            if len(diff["samples"]) >= diff["sample_limit"]:
                return
            if action == "delete":
                preview = compact_metadata(
                    before_metadata or {}, set(DRY_RUN_PREVIEW_FIELDS)
                )
                diff["samples"].append(
                    {
                        "id": str(point_id),
                        "action": action,
                        "before": {"metadata": preview},
                    }
                )
                return
            before_preview, after_preview = diff_metadata(
                before_metadata or {},
                after_metadata or {},
                fallback_keys=DRY_RUN_PREVIEW_FIELDS,
            )
            diff["samples"].append(
                {
                    "id": str(point_id),
                    "action": action,
                    "before": {"metadata": before_preview},
                    "after": {"metadata": after_preview},
                }
            )

        def parse_offset(offset: str | int | None) -> str | int | None:
            if offset is None:
                return None
            if isinstance(offset, int):
                return offset
            if isinstance(offset, str):
                text = offset.strip()
                if not text:
                    return None
                if text.isdigit() or (text.startswith("-") and text[1:].isdigit()):
                    try:
                        return int(text)
                    except ValueError:
                        return text
                return text
            return offset

        def resolve_combined_filter(
            memory_filter: MemoryFilterInput | None,
            query_filter: ArbitraryFilter | None,
            warnings: list[str],
        ) -> models.Filter | None:
            memory_filter_obj = build_memory_filter(
                memory_filter,
                strict=self.memory_settings.strict_params,
                warnings=warnings,
            )

            query_filter_obj = None
            if query_filter:
                if not self.qdrant_settings.allow_arbitrary_filter:
                    if self.memory_settings.strict_params:
                        raise ValueError("query_filter is not allowed.")
                    warnings.append("query_filter ignored (not allowed).")
                else:
                    query_filter_obj = models.Filter(**query_filter)

            return merge_filters([memory_filter_obj, query_filter_obj])

        def merge_list_values(
            existing: list[Any] | None, incoming: list[Any]
        ) -> list[Any]:
            if not existing:
                return list(incoming)
            merged: list[Any] = []
            seen: set[str] = set()
            for item in [*existing, *incoming]:
                try:
                    key = json.dumps(item, sort_keys=True, default=str)
                except TypeError:
                    key = str(item)
                if key in seen:
                    continue
                seen.add(key)
                merged.append(item)
            return merged

        async def perform_store(
            *,
            information: str,
            collection_name: str,
            metadata: Metadata | None,
            dedupe_action: str | None,
            warnings: list[str],
            strict: bool | None = None,
        ) -> dict[str, Any]:
            collection = resolve_collection_name(collection_name)
            strict_mode = (
                self.memory_settings.strict_params if strict is None else strict
            )

            records, normalize_warnings = normalize_memory_input(
                information=information,
                metadata=metadata,
                memory=None,
                embedding_info=self.embedding_info,
                strict=strict_mode,
                max_text_length=self.memory_settings.max_text_length,
            )
            warnings.extend(normalize_warnings)

            action = (dedupe_action or self.memory_settings.dedupe_action).lower()
            if action not in {"update", "skip"}:
                if self.memory_settings.strict_params:
                    raise ValueError("dedupe_action must be 'update' or 'skip'.")
                warnings.append(
                    f"Unknown dedupe_action '{action}', defaulted to update."
                )
                action = "update"

            results: list[dict[str, Any]] = []
            now = datetime.now(timezone.utc)
            now_iso = now.isoformat()
            now_ms = int(now.timestamp() * 1000)

            for record in records:
                scope = record.metadata.get("scope")
                text_hash = record.metadata.get("text_hash")
                chunk_index = record.metadata.get("chunk_index")
                chunk_count = record.metadata.get("chunk_count")
                duplicate = None

                if text_hash and scope:
                    duplicate_filter = models.Filter(
                        must=[
                            models.FieldCondition(
                                key=f"{METADATA_PATH}.text_hash",
                                match=models.MatchValue(value=text_hash),
                            ),
                            models.FieldCondition(
                                key=f"{METADATA_PATH}.scope",
                                match=models.MatchValue(value=scope),
                            ),
                        ]
                    )
                    matches = await self.qdrant_connector.scroll_points(
                        collection_name=collection,
                        query_filter=duplicate_filter,
                        limit=1,
                    )
                    if matches:
                        duplicate = matches[0]

                if duplicate:
                    if action == "skip":
                        result = {
                            "status": "skipped",
                            "id": str(duplicate.id),
                            "text_hash": text_hash,
                            "scope": scope,
                        }
                    else:
                        existing_payload = duplicate.payload or {}
                        existing_metadata = existing_payload.get(METADATA_PATH) or {}
                        try:
                            count = int(existing_metadata.get("reinforcement_count", 0))
                        except (TypeError, ValueError):
                            count = 0
                        merged_metadata = dict(existing_metadata)
                        merged_metadata.update(
                            {
                                "last_seen_at": now_iso,
                                "last_seen_at_ts": now_ms,
                                "reinforcement_count": count + 1,
                                "updated_at": now_iso,
                                "updated_at_ts": now_ms,
                            }
                        )
                        new_payload = dict(existing_payload)
                        new_payload[METADATA_PATH] = merged_metadata
                        await self.qdrant_connector.overwrite_payload(
                            [str(duplicate.id)],
                            new_payload,
                            collection_name=collection,
                        )
                        result = {
                            "status": "updated",
                            "id": str(duplicate.id),
                            "text_hash": text_hash,
                            "scope": scope,
                            "reinforcement_count": merged_metadata.get(
                                "reinforcement_count"
                            ),
                        }
                else:
                    entry = Entry(content=record.text, metadata=record.metadata)
                    point_id = await self.qdrant_connector.store(
                        entry, collection_name=collection
                    )
                    result = {
                        "status": "inserted",
                        "id": point_id,
                        "text_hash": text_hash,
                        "scope": scope,
                        "reinforcement_count": record.metadata.get(
                            "reinforcement_count"
                        ),
                    }

                if chunk_index is not None:
                    result["chunk_index"] = chunk_index
                if chunk_count is not None:
                    result["chunk_count"] = chunk_count

                results.append(result)

            return {
                "collection_name": collection,
                "dedupe_action": action,
                "results": results,
            }

        def build_validation_report(
            information: str | None,
            metadata: Metadata | None,
        ) -> dict[str, Any]:
            raw: dict[str, Any] = dict(metadata or {})
            if information:
                raw["text"] = information
            if "text" not in raw:
                for fallback in ("content", "document"):
                    if fallback in raw:
                        raw["text"] = raw[fallback]
                        break

            errors: list[str] = []
            missing_required: list[str] = []
            for field in REQUIRED_FIELDS:
                value = raw.get(field)
                if value is None or value == "":
                    missing_required.append(field)
            if missing_required:
                errors.append(f"Missing required fields: {sorted(missing_required)}.")

            try:
                normalize_memory_input(
                    information=information,
                    metadata=metadata,
                    memory=None,
                    embedding_info=self.embedding_info,
                    strict=True,
                    max_text_length=self.memory_settings.max_text_length,
                )
            except ValueError as exc:
                errors.append(str(exc))

            now_iso = datetime.now(timezone.utc).isoformat()
            suggested_metadata: dict[str, Any] = {}
            if "text" in missing_required:
                suggested_metadata["text"] = information or ""
            if "type" in missing_required:
                suggested_metadata["type"] = DEFAULT_MEMORY_TYPE
            if "entities" in missing_required:
                suggested_metadata["entities"] = []
            if "source" in missing_required:
                suggested_metadata["source"] = DEFAULT_SOURCE
            if "scope" in missing_required:
                suggested_metadata["scope"] = DEFAULT_SCOPE
            if "confidence" in missing_required:
                suggested_metadata["confidence"] = DEFAULT_CONFIDENCE
            if "created_at" in missing_required:
                suggested_metadata["created_at"] = now_iso
            if "updated_at" in missing_required:
                suggested_metadata["updated_at"] = now_iso

            return {
                "valid": not errors,
                "errors": errors,
                "missing_required": sorted(set(missing_required)),
                "suggested_metadata": suggested_metadata,
            }

        def coerce_int(value: Any) -> int | None:
            if isinstance(value, int):
                return value
            if isinstance(value, float) and value.is_integer():
                return int(value)
            if isinstance(value, str) and value.isdigit():
                return int(value)
            return None

        def ensure_mutations_allowed() -> None:
            if (
                self.tool_settings.mutations_require_admin
                and not self.tool_settings.admin_tools_enabled
            ):
                raise ValueError(
                    "Mutating operations require admin access. "
                    "Enable MCP_ADMIN_TOOLS_ENABLED or disable MCP_MUTATIONS_REQUIRE_ADMIN."
                )

        def enforce_batch_size(value: int, name: str = "batch_size") -> None:
            if value > self.tool_settings.max_batch_size:
                raise ValueError(
                    f"{name} exceeds max {self.tool_settings.max_batch_size}."
                )

        def enforce_point_ids(point_ids: list[str], name: str = "point_ids") -> None:
            if len(point_ids) > self.tool_settings.max_point_ids:
                raise ValueError(
                    f"{name} exceeds max {self.tool_settings.max_point_ids}."
                )

        def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
            if not vec_a or not vec_b or len(vec_a) != len(vec_b):
                return 0.0
            dot = sum(a * b for a, b in zip(vec_a, vec_b))
            norm_a = math.sqrt(sum(a * a for a in vec_a))
            norm_b = math.sqrt(sum(b * b for b in vec_b))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        def average_vectors(vectors: list[list[float]]) -> list[float]:
            if not vectors:
                raise ValueError("vectors cannot be empty.")
            dimension = len(vectors[0])
            sums = [0.0] * dimension
            for vector in vectors:
                if len(vector) != dimension:
                    raise ValueError("vectors must share the same dimension.")
                for index, value in enumerate(vector):
                    sums[index] += float(value)
            return [value / len(vectors) for value in sums]

        def mmr_select(
            query_vector: list[float],
            points: list[models.ScoredPoint],
            top_k: int,
            lambda_mult: float,
            vector_name: str | None,
        ) -> list[models.ScoredPoint] | None:
            candidates: list[tuple[models.ScoredPoint, list[float]]] = []
            for point in points:
                vector = extract_vector(point.vector, vector_name)
                if vector is None:
                    return None
                candidates.append((point, vector))

            sim_to_query = [
                cosine_similarity(query_vector, vector) for _, vector in candidates
            ]
            selected: list[int] = []
            candidate_indices = list(range(len(candidates)))
            while candidate_indices and len(selected) < top_k:
                if not selected:
                    best_index = max(candidate_indices, key=lambda i: sim_to_query[i])
                else:
                    best_index = max(
                        candidate_indices,
                        key=lambda i: lambda_mult * sim_to_query[i]
                        - (1 - lambda_mult)
                        * max(
                            cosine_similarity(candidates[i][1], candidates[j][1])
                            for j in selected
                        ),
                    )
                selected.append(best_index)
                candidate_indices.remove(best_index)

            return [candidates[i][0] for i in selected]

        def normalize_file_type(
            *,
            file_type: str | None,
            file_name: str | None,
            mime_type: str | None,
            has_text: bool,
        ) -> str:
            mime_map = {
                "text/plain": "txt",
                "text/markdown": "md",
                "text/csv": "csv",
                "application/csv": "csv",
                "application/vnd.ms-excel": "csv",
                "application/pdf": "pdf",
                "application/msword": "doc",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
            }

            candidate = None
            if file_type:
                normalized = file_type.strip().lower()
                if normalized.startswith("."):
                    normalized = normalized[1:]
                if "/" in normalized:
                    normalized = mime_map.get(normalized, normalized)
                candidate = normalized

            if not candidate and file_name:
                suffix = Path(file_name).suffix.lower().lstrip(".")
                if suffix:
                    candidate = suffix

            if not candidate and mime_type:
                normalized_mime = mime_type.split(";", 1)[0].strip().lower()
                candidate = mime_map.get(normalized_mime)

            if not candidate and has_text:
                candidate = "txt"

            if candidate in {"markdown"}:
                candidate = "md"
            if candidate in {"text"}:
                candidate = "txt"

            allowed = {"txt", "md", "csv", "pdf", "doc", "docx"}
            if candidate not in allowed:
                raise ValueError(
                    "file_type must be one of: txt, md, csv, pdf, doc, docx."
                )
            return candidate

        def parse_base64_payload(value: str) -> bytes:
            payload = value.strip()
            if "base64," in payload:
                payload = payload.split("base64,", 1)[1]
            try:
                return base64.b64decode(payload)
            except Exception as exc:
                raise ValueError("content_base64 is not valid base64 data.") from exc

        async def fetch_url_data(
            url: str, headers: dict[str, str] | None = None
        ) -> tuple[bytes, str | None]:
            def _fetch() -> tuple[bytes, str | None]:
                request = Request(url, headers=headers or {})
                with urlopen(request, timeout=30) as response:
                    data = response.read()
                    content_type = response.headers.get("Content-Type")
                return data, content_type

            return await asyncio.to_thread(_fetch)

        async def fetch_url_data_streaming(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            max_bytes: int,
            timeout_seconds: int,
            chunk_bytes: int = 1024 * 1024,
        ) -> tuple[bytes, str | None, int]:
            def _fetch() -> tuple[bytes, str | None, int]:
                request = Request(url, headers=headers or {})
                total = 0
                parts: list[bytes] = []
                with urlopen(request, timeout=timeout_seconds) as response:
                    content_type = response.headers.get("Content-Type")
                    while True:
                        chunk = response.read(chunk_bytes)
                        if not chunk:
                            break
                        total += len(chunk)
                        if total > max_bytes:
                            raise_limit_error(
                                stage="download",
                                limit_name="textbook_max_file_bytes",
                                limit_value=max_bytes,
                                actual_value=total,
                            )
                        parts.append(chunk)
                return b"".join(parts), content_type, total

            return await asyncio.to_thread(_fetch)

        async def fetch_url_to_tempfile_streaming(
            url: str,
            *,
            headers: dict[str, str] | None = None,
            max_bytes: int,
            timeout_seconds: int,
            suffix: str = ".pdf",
            chunk_bytes: int = 1024 * 1024,
        ) -> tuple[Path, str | None, int, str]:
            def _fetch() -> tuple[Path, str | None, int, str]:
                request = Request(url, headers=headers or {})
                total = 0
                content_type: str | None = None
                hasher = hashlib.sha256()
                temp_path_obj: Path | None = None
                try:
                    with tempfile.NamedTemporaryFile(
                        delete=False, suffix=suffix
                    ) as temp_file:
                        temp_path_obj = Path(temp_file.name)
                        with urlopen(request, timeout=timeout_seconds) as response:
                            content_type = response.headers.get("Content-Type")
                            while True:
                                chunk = response.read(chunk_bytes)
                                if not chunk:
                                    break
                                total += len(chunk)
                                if total > max_bytes:
                                    raise_limit_error(
                                        stage="download",
                                        limit_name="textbook_max_file_bytes",
                                        limit_value=max_bytes,
                                        actual_value=total,
                                    )
                                temp_file.write(chunk)
                                hasher.update(chunk)
                        temp_file.flush()
                except Exception:
                    if temp_path_obj and temp_path_obj.exists():
                        temp_path_obj.unlink(missing_ok=True)
                    raise
                if temp_path_obj is None:
                    raise RuntimeError("Failed to create temporary download file.")
                return temp_path_obj, content_type, total, hasher.hexdigest()

            return await asyncio.to_thread(_fetch)

        async def health_check(
            ctx: Context,
            collection_name: Annotated[
                str | None,
                Field(description="Collection to inspect for health."),
            ] = None,
            warm_all: Annotated[
                bool,
                Field(description="Warm up Qdrant and embedding clients."),
            ] = False,
        ) -> dict[str, Any]:
            state = new_request(
                ctx, {"collection_name": collection_name, "warm_all": warm_all}
            )
            name = resolve_health_collection(collection_name)
            checks: dict[str, Any] = {}
            ok = True
            collections: list[str] = []

            try:
                collections = await self.qdrant_connector.get_collection_names()
                checks["connection"] = {
                    "ok": True,
                    "collection_count": len(collections),
                }
            except Exception as exc:  # pragma: no cover - transport errors vary
                ok = False
                checks["connection"] = {"ok": False, "error": str(exc)}

            exists = False
            vector_indexed: bool | None = None
            vector_index_coverage: float | None = None
            unindexed_vectors_count: int | None = None
            payload_indexes_ok: bool | None = None
            optimizer_ok: bool | None = None
            try:
                exists = await self.qdrant_connector.collection_exists(name)
                collection_check: dict[str, Any] = {
                    "ok": exists,
                    "collection_name": name,
                }
                if not exists and collections:
                    collection_check["available_collections"] = collections[:20]
                    collection_check["suggested_collections"] = (
                        difflib.get_close_matches(
                            name,
                            collections,
                            n=3,
                            cutoff=0.45,
                        )
                    )
                checks["collection_exists"] = collection_check
                if not exists:
                    ok = False
            except Exception as exc:  # pragma: no cover
                ok = False
                checks["collection_exists"] = {"ok": False, "error": str(exc)}

            if exists:
                try:
                    info = await self.qdrant_connector.get_collection_info(name)
                    optimizer_ok = str(info.optimizer_status).lower() == "ok"
                    if info.indexed_vectors_count is not None:
                        if info.points_count and info.points_count > 0:
                            vector_index_coverage = (
                                info.indexed_vectors_count / info.points_count
                            )
                        else:
                            vector_index_coverage = 1.0
                        unindexed_vectors_count = max(
                            info.points_count - info.indexed_vectors_count, 0
                        )
                        vector_indexed = info.points_count == info.indexed_vectors_count
                    checks["collection_status"] = {
                        "ok": True,
                        "status": str(info.status),
                        "optimizer_status": str(info.optimizer_status),
                        "points_count": info.points_count,
                        "indexed_vectors_count": info.indexed_vectors_count,
                        "segments_count": info.segments_count,
                        "vector_indexed": vector_indexed,
                        "vector_index_coverage": vector_index_coverage,
                        "unindexed_vectors_count": unindexed_vectors_count,
                    }
                except Exception as exc:  # pragma: no cover
                    ok = False
                    checks["collection_status"] = {"ok": False, "error": str(exc)}

                try:
                    vectors = await self.qdrant_connector.get_collection_vectors(name)
                    checks["vectors"] = {"ok": True, "vectors": vectors}
                    vector_name = await self.qdrant_connector.resolve_vector_name(name)
                    checks["vector_name"] = {
                        "ok": True,
                        "vector_name": vector_name,
                        "embedding_dim": self.embedding_info.dim,
                    }
                except Exception as exc:  # pragma: no cover
                    ok = False
                    checks["vectors"] = {"ok": False, "error": str(exc)}

                try:
                    schema = await self.qdrant_connector.get_collection_payload_schema(
                        name
                    )
                    checks["payload_schema"] = {"ok": True, "payload_schema": schema}
                    expected = set(self.payload_indexes.keys())
                    missing = sorted(expected - set(schema.keys()))
                    payload_indexes_ok = not missing
                    if missing:
                        state.warnings.append(
                            f"Payload schema missing expected indexes: {missing}"
                        )
                except Exception as exc:  # pragma: no cover
                    ok = False
                    checks["payload_schema"] = {"ok": False, "error": str(exc)}
                    payload_indexes_ok = None

                if "collection_status" in checks:
                    checks["collection_status"]["payload_indexes_ok"] = (
                        payload_indexes_ok
                    )
                    if (
                        vector_indexed is not None
                        and payload_indexes_ok is not None
                        and optimizer_ok is not None
                    ):
                        checks["collection_status"]["fully_indexed"] = bool(
                            vector_indexed and payload_indexes_ok and optimizer_ok
                        )
                    else:
                        checks["collection_status"]["fully_indexed"] = None

            warmup: dict[str, Any] = {}
            if warm_all:
                try:
                    await self.embedding_provider.embed_query("warmup")
                    warmup["embedding"] = "ok"
                except Exception as exc:  # pragma: no cover
                    ok = False
                    warmup["embedding"] = f"error: {exc}"
                try:
                    await self.qdrant_connector.get_collection_names()
                    warmup["qdrant"] = "ok"
                except Exception as exc:  # pragma: no cover
                    ok = False
                    warmup["qdrant"] = f"error: {exc}"

            data = {
                "ok": ok,
                "collection_name": name,
                "embedding": self.embedding_info.__dict__,
                "checks": checks,
            }
            if warmup:
                data["warmup"] = warmup
            return finish_request(state, data)

        async def store(
            ctx: Context,
            information: Annotated[str, Field(description="Text to store")],
            collection_name: Annotated[
                str, Field(description="The collection to store the information in")
            ],
            # The `metadata` parameter is defined as non-optional, but it can be None.
            # If we set it to be optional, some of the MCP clients, like Cursor, cannot
            # handle the optional parameter correctly.
            metadata: Annotated[
                Metadata | None,
                Field(
                    description=(
                        "Memory metadata (type, entities, source, scope, timestamps, confidence)."
                    )
                ),
            ] = None,
            dedupe_action: Annotated[
                str | None,
                Field(
                    description="How to handle duplicates: update or skip. Defaults to MCP_DEDUPE_ACTION."
                ),
            ] = None,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "information": information,
                    "collection_name": collection_name,
                    "metadata": metadata,
                    "dedupe_action": dedupe_action,
                },
            )
            ensure_mutations_allowed()
            await ctx.debug(f"Storing information {information} in Qdrant")
            data = await perform_store(
                information=information,
                collection_name=collection_name,
                metadata=metadata,
                dedupe_action=dedupe_action,
                warnings=state.warnings,
            )
            return finish_request(state, data)

        async def cache_memory(
            ctx: Context,
            information: Annotated[
                str, Field(description="Short-term memory text to store.")
            ],
            collection_name: Annotated[
                str | None,
                Field(
                    description=(
                        "Optional short-term collection override. Defaults to "
                        "MCP_SHORT_TERM_COLLECTION."
                    )
                ),
            ] = None,
            metadata: Annotated[
                Metadata | None,
                Field(description="Optional memory metadata overrides."),
            ] = None,
            ttl_days: Annotated[
                int | None,
                Field(
                    description=(
                        "TTL in days for short-term memory "
                        "(defaults to MCP_SHORT_TERM_TTL_DAYS)."
                    )
                ),
            ] = None,
            dedupe_action: Annotated[
                str | None,
                Field(
                    description="How to handle duplicates: update or skip. Defaults to MCP_DEDUPE_ACTION."
                ),
            ] = None,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "information": information,
                    "collection_name": collection_name,
                    "metadata": metadata,
                    "ttl_days": ttl_days,
                    "dedupe_action": dedupe_action,
                },
            )
            ensure_mutations_allowed()

            resolved_collection = (
                collection_name or self.memory_settings.short_term_collection
            )
            if not resolved_collection:
                raise ValueError("Short-term collection is not configured.")

            resolved_ttl = ttl_days
            if resolved_ttl is None and metadata and "ttl_days" in metadata:
                raw_ttl = metadata.get("ttl_days")
                if isinstance(raw_ttl, int):
                    resolved_ttl = raw_ttl
                else:
                    coerced = coerce_int(raw_ttl)
                    if coerced is not None:
                        state.warnings.append("ttl_days coerced to int.")
                        resolved_ttl = coerced
                    else:
                        state.warnings.append("ttl_days ignored due to invalid value.")
            if resolved_ttl is None:
                resolved_ttl = self.memory_settings.short_term_ttl_days
            if resolved_ttl <= 0:
                raise ValueError("ttl_days must be a positive integer.")

            cache_metadata = dict(metadata or {})
            cache_metadata["ttl_days"] = resolved_ttl
            cache_metadata.setdefault("scope", "short_term")
            cache_metadata.setdefault("source", "short_term_cache")

            data = await perform_store(
                information=information,
                collection_name=resolved_collection,
                metadata=cache_metadata,
                dedupe_action=dedupe_action,
                warnings=state.warnings,
            )
            data["collection_name"] = resolved_collection
            data["ttl_days"] = resolved_ttl
            return finish_request(state, data)

        async def promote_short_term(
            ctx: Context,
            point_ids: Annotated[
                list[str],
                Field(description="Point ids to promote from short-term memory."),
            ],
            source_collection: Annotated[
                str | None,
                Field(
                    description="Optional short-term collection override (defaults to MCP_SHORT_TERM_COLLECTION)."
                ),
            ] = None,
            target_collection: Annotated[
                str | None,
                Field(
                    description="Target collection for long-term storage (defaults to server collection)."
                ),
            ] = None,
            metadata_patch: Annotated[
                Metadata | None,
                Field(description="Optional metadata overrides to apply on promote."),
            ] = None,
            clear_ttl: Annotated[
                bool,
                Field(
                    description="Clear expires_at/ttl_days fields when promoting to long-term."
                ),
            ] = True,
            remove_source: Annotated[
                bool,
                Field(description="Delete promoted points from short-term collection."),
            ] = False,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "point_ids": point_ids,
                    "source_collection": source_collection,
                    "target_collection": target_collection,
                    "metadata_patch": metadata_patch,
                    "clear_ttl": clear_ttl,
                    "remove_source": remove_source,
                },
            )
            ensure_mutations_allowed()
            if not point_ids:
                raise ValueError("point_ids cannot be empty.")
            enforce_point_ids(point_ids)

            source = source_collection or self.memory_settings.short_term_collection
            if not source:
                raise ValueError("Short-term collection is not configured.")
            target = resolve_collection_name(target_collection or "")

            records = await self.qdrant_connector.retrieve_points(
                point_ids,
                collection_name=source,
                with_payload=True,
                with_vectors=True,
            )
            found_ids = {str(record.id) for record in records}
            missing_ids = [pid for pid in point_ids if pid not in found_ids]

            await self.qdrant_connector.ensure_collection_exists(target)
            source_vector_name = await self.qdrant_connector.resolve_vector_name(source)
            target_vector_name = await self.qdrant_connector.resolve_vector_name(target)

            now = datetime.now(timezone.utc)
            promoted_ids: list[str] = []
            reembedded_ids: list[str] = []
            skipped_ids: list[str] = []
            points_to_upsert: list[models.PointStruct] = []
            patch = dict(metadata_patch or {})

            for record in records:
                record_id = str(record.id)
                payload = dict(record.payload or {})
                metadata = dict(payload.get(METADATA_PATH) or {})
                if clear_ttl:
                    metadata.pop("ttl_days", None)
                    metadata.pop("expires_at", None)
                    metadata.pop("expires_at_ts", None)
                if patch:
                    metadata.update(patch)
                metadata["updated_at"] = now.isoformat()
                metadata["updated_at_ts"] = int(now.timestamp() * 1000)
                payload[METADATA_PATH] = metadata

                vector = extract_vector(record.vector, source_vector_name)
                if vector is None:
                    text = extract_payload_text(payload)
                    if not text:
                        skipped_ids.append(record_id)
                        state.warnings.append(
                            f"Missing text/vector for {record_id}; skipped."
                        )
                        continue
                    entry = Entry(content=text, metadata=metadata)
                    await self.qdrant_connector.store(
                        entry, collection_name=target, point_id=record_id
                    )
                    reembedded_ids.append(record_id)
                    continue

                if target_vector_name is None:
                    vector_payload: list[float] | dict[str, list[float]] = vector
                else:
                    vector_payload = {target_vector_name: vector}

                points_to_upsert.append(
                    models.PointStruct(
                        id=record_id,
                        vector=vector_payload,
                        payload=payload,
                    )
                )
                promoted_ids.append(record_id)

            if points_to_upsert:
                await self.qdrant_connector.upsert_points(
                    points_to_upsert,
                    collection_name=target,
                )

            if remove_source and (promoted_ids or reembedded_ids):
                await self.qdrant_connector.delete_points(
                    [*promoted_ids, *reembedded_ids],
                    collection_name=source,
                )

            data = {
                "status": "promoted",
                "source_collection": source,
                "target_collection": target,
                "promoted": promoted_ids,
                "reembedded": reembedded_ids,
                "skipped": skipped_ids,
                "missing": missing_ids,
                "removed_from_source": remove_source,
            }
            return finish_request(state, data)

        async def validate_memory(
            ctx: Context,
            information: Annotated[
                str | None, Field(description="Memory text to validate.")
            ] = None,
            metadata: Annotated[
                Metadata | None,
                Field(description="Memory metadata to validate."),
            ] = None,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "information": information,
                    "metadata": metadata,
                },
            )
            report = build_validation_report(information, metadata)
            return finish_request(state, report)

        async def ingest_with_validation(
            ctx: Context,
            information: Annotated[str, Field(description="Text to store.")],
            collection_name: Annotated[
                str, Field(description="The collection to store the information in")
            ],
            metadata: Annotated[
                Metadata | None,
                Field(description="Memory metadata."),
            ] = None,
            dedupe_action: Annotated[
                str | None,
                Field(
                    description="How to handle duplicates: update or skip. Defaults to MCP_DEDUPE_ACTION."
                ),
            ] = None,
            on_invalid: Annotated[
                str | None,
                Field(
                    description="What to do if validation fails: allow, reject, quarantine."
                ),
            ] = None,
            quarantine_collection: Annotated[
                str | None,
                Field(description="Override quarantine collection name."),
            ] = None,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "information": information,
                    "collection_name": collection_name,
                    "metadata": metadata,
                    "dedupe_action": dedupe_action,
                    "on_invalid": on_invalid,
                    "quarantine_collection": quarantine_collection,
                },
            )
            mode = (on_invalid or self.memory_settings.ingest_validation_mode).lower()
            if mode not in {"allow", "reject", "quarantine"}:
                raise ValueError("on_invalid must be allow, reject, or quarantine.")

            report = build_validation_report(information, metadata)
            if report["valid"]:
                ensure_mutations_allowed()
                data = await perform_store(
                    information=information,
                    collection_name=collection_name,
                    metadata=metadata,
                    dedupe_action=dedupe_action,
                    warnings=state.warnings,
                )
                data["status"] = "stored"
                data["validation"] = report
                return finish_request(state, data)

            if mode == "reject":
                data = {
                    "status": "rejected",
                    "collection_name": resolve_collection_name(collection_name),
                    "validation": report,
                }
                return finish_request(state, data)

            if mode == "quarantine":
                ensure_mutations_allowed()
                quarantine_name = (
                    quarantine_collection or self.memory_settings.quarantine_collection
                )
                quarantine_metadata = dict(metadata or {})
                labels = quarantine_metadata.get("labels")
                if not isinstance(labels, list):
                    labels = []
                if "needs_review" not in labels:
                    labels.append("needs_review")
                quarantine_metadata["labels"] = labels
                quarantine_metadata["validation_status"] = "needs_review"
                quarantine_metadata["validation_errors"] = report["errors"]
                data = await perform_store(
                    information=information,
                    collection_name=quarantine_name,
                    metadata=quarantine_metadata,
                    dedupe_action=dedupe_action,
                    warnings=state.warnings,
                    strict=False,
                )
                data["status"] = "quarantined"
                data["quarantine_collection"] = quarantine_name
                data["validation"] = report
                return finish_request(state, data)

            ensure_mutations_allowed()
            data = await perform_store(
                information=information,
                collection_name=collection_name,
                metadata=metadata,
                dedupe_action=dedupe_action,
                warnings=state.warnings,
            )
            data["status"] = "stored_unvalidated"
            data["validation"] = report
            return finish_request(state, data)

        async def ingest_document(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The collection to store the document in")
            ],
            file_name: Annotated[
                str | None,
                Field(
                    description="Original file name (used to infer file type and title)."
                ),
            ] = None,
            file_type: Annotated[
                str | None,
                Field(description="File type/extension: txt, md, csv, pdf, doc, docx."),
            ] = None,
            mime_type: Annotated[
                str | None,
                Field(description="Optional MIME type for file type inference."),
            ] = None,
            content_base64: Annotated[
                str | None,
                Field(description="Base64-encoded file content."),
            ] = None,
            text: Annotated[
                str | None,
                Field(description="Raw text content (for txt/md uploads)."),
            ] = None,
            source_url: Annotated[
                str | None,
                Field(description="URL to fetch the document from."),
            ] = None,
            source_url_headers: Annotated[
                dict[str, str] | None,
                Field(
                    description=(
                        "Optional headers to use when fetching source_url "
                        "(e.g., User-Agent, Authorization)."
                    )
                ),
            ] = None,
            doc_id: Annotated[
                str | None,
                Field(description="Document id for update/delete workflows."),
            ] = None,
            doc_title: Annotated[
                str | None,
                Field(description="Document title stored with each chunk."),
            ] = None,
            metadata: Annotated[
                Metadata | None,
                Field(
                    description=(
                        "Base memory metadata overrides (type, entities, source, scope,"
                        " confidence, etc.)."
                    )
                ),
            ] = None,
            chunk_size: Annotated[
                int | None,
                Field(
                    description="Chunk size in characters (defaults to MCP_MAX_TEXT_LENGTH)."
                ),
            ] = None,
            chunk_overlap: Annotated[
                int | None,
                Field(description="Chunk overlap in characters (default 200)."),
            ] = None,
            ocr: Annotated[
                bool,
                Field(description="Enable OCR for PDF extraction (default: true)."),
            ] = True,
            dedupe_action: Annotated[
                str | None,
                Field(
                    description="How to handle existing doc_id: update or skip. Defaults to MCP_DEDUPE_ACTION."
                ),
            ] = None,
            return_chunk_ids: Annotated[
                bool,
                Field(description="Return IDs for the stored chunks."),
            ] = False,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "collection_name": collection_name,
                    "file_name": file_name,
                    "file_type": file_type,
                    "mime_type": mime_type,
                    "content_base64": bool(content_base64),
                    "text": bool(text),
                    "source_url": source_url,
                    "source_url_headers": (
                        list(source_url_headers.keys()) if source_url_headers else None
                    ),
                    "doc_id": doc_id,
                    "doc_title": doc_title,
                    "metadata": metadata,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "ocr": ocr,
                    "dedupe_action": dedupe_action,
                    "return_chunk_ids": return_chunk_ids,
                },
            )
            ensure_mutations_allowed()

            warning_set: set[str] = set()

            def add_warning(message: str) -> None:
                if message not in warning_set:
                    warning_set.add(message)
                    state.warnings.append(message)

            if not content_base64 and not text and not source_url:
                raise ValueError(
                    "Provide content_base64, text, or source_url for ingestion."
                )

            if content_base64 and text:
                add_warning(
                    "Both content_base64 and text provided; using content_base64."
                )

            if file_name is not None:
                file_name = file_name.strip() or None

            fetched_mime = None
            file_bytes = None
            if content_base64:
                file_bytes = parse_base64_payload(content_base64)
                if source_url:
                    add_warning(
                        "source_url ignored because content_base64 was provided."
                    )
                if source_url_headers:
                    add_warning(
                        "source_url_headers ignored because content_base64 was provided."
                    )
            elif source_url:
                if text:
                    add_warning("text ignored because source_url was provided.")
                resolved_headers = {
                    "User-Agent": "Mozilla/5.0 (compatible; mcp-server-qdrant)",
                    "Accept": "*/*",
                    "Accept-Language": "en-US,en;q=0.9",
                }
                if source_url_headers:
                    for key, value in source_url_headers.items():
                        if not isinstance(key, str) or not isinstance(value, str):
                            add_warning("source_url_headers coerced to strings.")
                            key = str(key)
                            value = str(value)
                        resolved_headers[key] = value
                file_bytes, fetched_mime = await fetch_url_data(
                    source_url, headers=resolved_headers
                )

            if source_url and not file_name:
                parsed_url = urlparse(source_url)
                if parsed_url.path:
                    parsed_name = Path(parsed_url.path).name
                    file_name = parsed_name or file_name

            resolved_mime = mime_type or fetched_mime
            resolved_file_type = normalize_file_type(
                file_type=file_type,
                file_name=file_name,
                mime_type=resolved_mime,
                has_text=bool(text),
            )

            if resolved_file_type in {"pdf", "doc", "docx"} and file_bytes is None:
                raise ValueError(f"{resolved_file_type} ingestion requires file bytes.")

            if (
                resolved_file_type in {"txt", "md", "csv"}
                and text is None
                and file_bytes is None
            ):
                raise ValueError("txt/md/csv ingestion requires text or file bytes.")

            extraction_result = await asyncio.to_thread(
                extract_document_sections,
                resolved_file_type,
                text=text,
                data=file_bytes,
                ocr=ocr,
            )
            for warning in extraction_result.warnings:
                add_warning(warning)

            doc_text = "\n\n".join(
                section.text for section in extraction_result.sections if section.text
            ).strip()

            if file_bytes:
                doc_hash = hashlib.sha256(file_bytes).hexdigest()
            else:
                doc_hash = hashlib.sha256(doc_text.encode("utf-8")).hexdigest()

            base_metadata = dict(metadata or {})
            resolved_doc_id = doc_id or base_metadata.get("doc_id") or doc_hash
            resolved_doc_title = doc_title or base_metadata.get("doc_title")
            if not resolved_doc_title:
                resolved_doc_title = (
                    extraction_result.title_hint
                    or (Path(file_name).stem if file_name else None)
                    or "document"
                )

            if not extraction_result.sections:
                data = {
                    "status": "no_text_extracted",
                    "collection_name": resolve_collection_name(collection_name),
                    "doc_id": resolved_doc_id,
                    "doc_title": resolved_doc_title,
                    "doc_hash": doc_hash,
                    "file_name": file_name,
                    "file_type": resolved_file_type,
                    "source_url": source_url,
                    "pages": extraction_result.page_count,
                    "chunks_count": 0,
                    "warnings": list(warning_set),
                }
                return finish_request(state, data)

            base_metadata.setdefault("type", "document")
            base_metadata.setdefault("source", "document")
            base_metadata.setdefault("scope", resolved_doc_id)
            base_metadata.setdefault("entities", [])
            base_metadata.setdefault("confidence", 0.5)

            base_metadata["doc_id"] = resolved_doc_id
            base_metadata["doc_title"] = resolved_doc_title
            base_metadata["doc_hash"] = doc_hash
            base_metadata["file_type"] = resolved_file_type
            if source_url:
                base_metadata["source_url"] = source_url
            if file_name:
                base_metadata["file_name"] = file_name

            resolved_chunk_size = chunk_size or self.memory_settings.max_text_length
            if resolved_chunk_size <= 0:
                raise ValueError("chunk_size must be positive.")
            resolved_overlap = 200 if chunk_overlap is None else chunk_overlap
            if resolved_overlap < 0:
                raise ValueError("chunk_overlap must be >= 0.")
            if resolved_overlap >= resolved_chunk_size:
                add_warning("chunk_overlap reduced to chunk_size - 1.")
                resolved_overlap = max(0, resolved_chunk_size - 1)

            chunk_specs: list[dict[str, Any]] = []
            for section in extraction_result.sections:
                for chunk in chunk_text_with_overlap(
                    section.text, resolved_chunk_size, resolved_overlap
                ):
                    if not chunk:
                        continue
                    chunk_specs.append(
                        {
                            "text": chunk,
                            "page_start": section.page_start,
                            "page_end": section.page_end,
                            "section_heading": section.section_heading,
                        }
                    )

            if not chunk_specs:
                add_warning("No non-empty chunks produced from document.")
                data = {
                    "status": "no_chunks",
                    "collection_name": resolve_collection_name(collection_name),
                    "doc_id": resolved_doc_id,
                    "doc_title": resolved_doc_title,
                    "doc_hash": doc_hash,
                    "file_name": file_name,
                    "file_type": resolved_file_type,
                    "source_url": source_url,
                    "pages": extraction_result.page_count,
                    "chunks_count": 0,
                    "warnings": list(warning_set),
                }
                return finish_request(state, data)

            chunk_count = len(chunk_specs)
            parent_text_hash = compute_text_hash(doc_text) if chunk_count > 1 else None

            entries: list[Entry] = []
            for index, spec in enumerate(chunk_specs):
                chunk_metadata = dict(base_metadata)
                if chunk_count > 1:
                    chunk_metadata["chunk_index"] = index
                    chunk_metadata["chunk_count"] = chunk_count
                    if parent_text_hash:
                        chunk_metadata["parent_text_hash"] = parent_text_hash
                if spec.get("page_start") is not None:
                    chunk_metadata["page_start"] = spec["page_start"]
                if spec.get("page_end") is not None:
                    chunk_metadata["page_end"] = spec["page_end"]
                if spec.get("section_heading"):
                    chunk_metadata["section_heading"] = spec["section_heading"]

                records, warnings = normalize_memory_input(
                    information=spec["text"],
                    metadata=chunk_metadata,
                    memory=None,
                    embedding_info=self.embedding_info,
                    strict=self.memory_settings.strict_params,
                    max_text_length=max(resolved_chunk_size, len(spec["text"])),
                )
                for warning in warnings:
                    add_warning(warning)
                for record in records:
                    entries.append(Entry(content=record.text, metadata=record.metadata))

            action = (dedupe_action or self.memory_settings.dedupe_action).lower()
            if action not in {"update", "skip"}:
                if self.memory_settings.strict_params:
                    raise ValueError("dedupe_action must be 'update' or 'skip'.")
                add_warning(f"Unknown dedupe_action '{action}', defaulted to update.")
                action = "update"

            collection = resolve_collection_name(collection_name)
            existing_count = 0
            deleted_existing = False
            skip_doc_filter = False
            if resolved_doc_id:
                if await self.qdrant_connector.collection_exists(collection):
                    doc_id_key = f"{METADATA_PATH}.doc_id"
                    try:
                        schema = (
                            await self.qdrant_connector.get_collection_payload_schema(
                                collection
                            )
                        )
                    except Exception as exc:  # pragma: no cover - transport errors vary
                        add_warning(f"Failed to read payload schema: {exc}")
                        schema = {}

                    if doc_id_key not in schema:
                        try:
                            created = (
                                await self.qdrant_connector.ensure_payload_indexes(
                                    collection_name=collection,
                                    indexes={
                                        doc_id_key: models.PayloadSchemaType.KEYWORD
                                    },
                                )
                            )
                            if doc_id_key in created:
                                add_warning(
                                    "Created payload index for metadata.doc_id."
                                )
                        except (
                            Exception
                        ) as exc:  # pragma: no cover - transport errors vary
                            add_warning(
                                "Payload index for metadata.doc_id missing and "
                                "could not be created."
                            )
                            add_warning(f"doc_id dedupe skipped: {exc}")
                            skip_doc_filter = True

                    if skip_doc_filter:
                        doc_filter = None
                    else:
                        doc_filter = models.Filter(
                            must=[
                                models.FieldCondition(
                                    key=doc_id_key,
                                    match=models.MatchValue(value=resolved_doc_id),
                                )
                            ]
                        )
                    if doc_filter is not None:
                        existing_count = await self.qdrant_connector.count_points(
                            collection_name=collection,
                            query_filter=doc_filter,
                        )
                        if existing_count > 0:
                            if action == "skip":
                                data = {
                                    "status": "skipped",
                                    "collection_name": collection,
                                    "doc_id": resolved_doc_id,
                                    "doc_title": resolved_doc_title,
                                    "doc_hash": doc_hash,
                                    "file_name": file_name,
                                    "file_type": resolved_file_type,
                                    "source_url": source_url,
                                    "pages": extraction_result.page_count,
                                    "chunks_count": 0,
                                    "existing_count": existing_count,
                                    "warnings": list(warning_set),
                                }
                                return finish_request(state, data)
                            await self.qdrant_connector.delete_by_filter(
                                doc_filter, collection_name=collection
                            )
                            deleted_existing = True

            point_ids = await self.qdrant_connector.store_entries(
                entries, collection_name=collection
            )

            data = {
                "status": "ingested",
                "collection_name": collection,
                "doc_id": resolved_doc_id,
                "doc_title": resolved_doc_title,
                "doc_hash": doc_hash,
                "file_name": file_name,
                "file_type": resolved_file_type,
                "source_url": source_url,
                "pages": extraction_result.page_count,
                "chunks_count": chunk_count,
                "dedupe_action": action,
                "existing_count": existing_count,
                "replaced_existing": deleted_existing,
                "warnings": list(warning_set),
            }
            if return_chunk_ids:
                data["chunk_ids"] = point_ids
            return finish_request(state, data)

        async def find(
            ctx: Context,
            query: Annotated[str, Field(description="What to search for")],
            collection_name: Annotated[
                str | None, Field(description="The collection to search in")
            ] = None,
            memory_filter: Annotated[
                MemoryFilterInput | None,
                Field(description="Structured filters for memory fields."),
            ] = None,
            query_filter: ArbitraryFilter | None = None,
            top_k: Annotated[
                int | None,
                Field(description="Max number of results to return. Start with 3-5."),
            ] = None,
            response_mode: Annotated[
                Literal["compact", "metadata", "payload"],
                Field(
                    description=(
                        "Result detail level. compact returns ids, scores, snippets, "
                        "and selected metadata; payload returns full Qdrant payloads."
                    )
                ),
            ] = "compact",
            snippet_chars: Annotated[
                int,
                Field(
                    description="Max snippet length in characters, clamped to 40-1200."
                ),
            ] = 240,
            max_output_chars: Annotated[
                int | None,
                Field(
                    description=(
                        "Optional approximate response budget. When set, results are "
                        "trimmed to fit 800-50000 characters."
                    )
                ),
            ] = None,
            metadata_fields: Annotated[
                list[str] | None,
                Field(
                    description=(
                        "Optional metadata field allowlist for compact/metadata modes, "
                        "for example ['title','doc_id','source_url']."
                    )
                ),
            ] = None,
            group_by_doc: Annotated[
                bool,
                Field(
                    description=(
                        "When true, limit repeated chunks from the same doc/title/source."
                    )
                ),
            ] = False,
            max_chunks_per_doc: Annotated[
                int,
                Field(description="Max chunks per document when group_by_doc=true."),
            ] = 2,
            min_score: Annotated[
                float | None,
                Field(
                    description="Optional minimum similarity score for returned hits."
                ),
            ] = None,
            use_mmr: Annotated[
                bool, Field(description="Enable MMR for diverse retrieval.")
            ] = False,
            mmr_lambda: Annotated[
                float,
                Field(description="MMR trade-off between relevance and diversity."),
            ] = 0.5,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "query": query,
                    "collection_name": collection_name,
                    "memory_filter": memory_filter,
                    "query_filter": query_filter,
                    "top_k": top_k,
                    "response_mode": response_mode,
                    "snippet_chars": snippet_chars,
                    "max_output_chars": max_output_chars,
                    "metadata_fields": metadata_fields,
                    "group_by_doc": group_by_doc,
                    "max_chunks_per_doc": max_chunks_per_doc,
                    "min_score": min_score,
                    "use_mmr": use_mmr,
                    "mmr_lambda": mmr_lambda,
                },
            )

            memory_filter_obj = build_memory_filter(
                memory_filter,
                strict=self.memory_settings.strict_params,
                warnings=state.warnings,
            )

            query_filter_obj = None
            if query_filter:
                if not self.qdrant_settings.allow_arbitrary_filter:
                    if self.memory_settings.strict_params:
                        raise ValueError("query_filter is not allowed.")
                    state.warnings.append("query_filter ignored (not allowed).")
                else:
                    query_filter_obj = models.Filter(**query_filter)

            combined_filter = merge_filters([memory_filter_obj, query_filter_obj])

            limit = top_k or self.qdrant_settings.search_limit
            if limit <= 0:
                raise ValueError("top_k must be positive.")
            if max_chunks_per_doc <= 0:
                raise ValueError("max_chunks_per_doc must be positive.")
            snippet_chars = clamp_snippet_chars(snippet_chars)
            selected_metadata_fields = normalize_metadata_fields(metadata_fields)
            candidate_limit = limit
            if group_by_doc or min_score is not None:
                candidate_limit = min(max(limit * 3, limit), 100)

            collection = resolve_collection_name(collection_name)
            filter_applied = combined_filter is not None
            query_vector_dim = self.embedding_provider.get_vector_size()
            query_vector = await self._embed_query_cached(query)

            points: list[models.ScoredPoint]
            if use_mmr:
                if mmr_lambda < 0 or mmr_lambda > 1:
                    if self.memory_settings.strict_params:
                        raise ValueError("mmr_lambda must be between 0 and 1.")
                    state.warnings.append("mmr_lambda clamped to [0,1].")
                    mmr_lambda = max(0.0, min(1.0, mmr_lambda))

                vector_name = await self.qdrant_connector.resolve_vector_name(
                    collection
                )
                candidate_limit = min(max(candidate_limit, limit * 4), 100)
                points = await self.qdrant_connector.query_points(
                    query_vector,
                    collection_name=collection,
                    limit=candidate_limit,
                    query_filter=combined_filter,
                    with_vectors=True,
                )
                selected = mmr_select(
                    query_vector,
                    points,
                    top_k=candidate_limit,
                    lambda_mult=mmr_lambda,
                    vector_name=vector_name,
                )
                if selected is None:
                    state.warnings.append("MMR disabled due to missing vectors.")
                    points = points[:limit]
                else:
                    points = selected
            else:
                points = await self.qdrant_connector.query_points(
                    query_vector,
                    collection_name=collection,
                    limit=candidate_limit,
                    query_filter=combined_filter,
                )

            points = select_points_for_output(
                points,
                limit=limit,
                group_by_doc=group_by_doc,
                max_chunks_per_doc=max_chunks_per_doc,
                min_score=min_score,
            )
            results = []
            for point in points:
                results.append(
                    format_search_result(
                        point,
                        response_mode=response_mode,
                        snippet_chars=snippet_chars,
                        metadata_fields=selected_metadata_fields,
                    )
                )

            base_data: dict[str, Any] = {
                "query": query,
                "collection_name": collection,
            }
            results, budget = trim_results_to_budget(
                results,
                base_data=base_data,
                max_output_chars=max_output_chars,
            )
            data = {**base_data, "results": results}
            extra_meta = {
                "top_k": limit,
                "candidate_limit": candidate_limit,
                "filter_applied": filter_applied,
                "query_vector_dim": query_vector_dim,
                "mmr": use_mmr,
                "response_mode": response_mode,
                "metadata_fields": selected_metadata_fields,
                "group_by_doc": group_by_doc,
                "max_chunks_per_doc": max_chunks_per_doc if group_by_doc else None,
                "min_score": min_score,
                "budget": budget,
            }
            return finish_request(state, data, extra_meta=extra_meta)

        async def study_search(
            ctx: Context,
            query: Annotated[
                str,
                Field(
                    description=(
                        "Study query to search semantically across school materials."
                    )
                ),
            ],
            collection_name: Annotated[
                str | None,
                Field(
                    description=(
                        "Study collection to search. Defaults to MCP_STUDY_COLLECTION."
                    )
                ),
            ] = None,
            class_code: Annotated[
                str | None,
                Field(
                    description=(
                        "Optional exact course/class code filter, such as MUS327."
                    )
                ),
            ] = None,
            subject: Annotated[
                str | None,
                Field(description="Optional exact subject filter."),
            ] = None,
            module: Annotated[
                str | int | None,
                Field(description="Optional exact module number or label filter."),
            ] = None,
            week: Annotated[
                str | int | None,
                Field(description="Optional exact accelerated course week filter."),
            ] = None,
            status: Annotated[
                str | None,
                Field(description="Optional exact material status filter."),
            ] = None,
            material_type: Annotated[
                str | None,
                Field(
                    description=(
                        "Optional exact material type filter, such as textbook or lesson."
                    )
                ),
            ] = None,
            title: Annotated[
                str | None,
                Field(description="Optional exact title filter."),
            ] = None,
            author: Annotated[
                str | None,
                Field(description="Optional exact author filter."),
            ] = None,
            doc_id: Annotated[
                str | None,
                Field(description="Optional exact document id filter."),
            ] = None,
            chapter: Annotated[
                str | int | None,
                Field(description="Optional exact chapter number or label filter."),
            ] = None,
            top_k: Annotated[
                int,
                Field(description="Max compact results to return. Use 3-5 first."),
            ] = 5,
            response_mode: Annotated[
                Literal["compact", "metadata", "payload"],
                Field(
                    description=(
                        "Result detail level. Keep compact unless full payload is required."
                    )
                ),
            ] = "compact",
            snippet_chars: Annotated[
                int,
                Field(
                    description="Max snippet length in characters, clamped to 40-1200."
                ),
            ] = 320,
            max_output_chars: Annotated[
                int | None,
                Field(
                    description=(
                        "Optional approximate response budget. When set, results are "
                        "trimmed to fit 800-50000 characters."
                    )
                ),
            ] = None,
            metadata_fields: Annotated[
                list[str] | None,
                Field(
                    description="Optional metadata field allowlist for compact output."
                ),
            ] = None,
            group_by_doc: Annotated[
                bool,
                Field(description="Limit repeated chunks from the same document."),
            ] = False,
            max_chunks_per_doc: Annotated[
                int,
                Field(description="Max chunks per document when group_by_doc=true."),
            ] = 2,
            min_score: Annotated[
                float | None,
                Field(
                    description="Optional minimum similarity score for returned hits."
                ),
            ] = None,
            use_mmr: Annotated[
                bool,
                Field(description="Enable MMR for diverse study retrieval."),
            ] = False,
        ) -> dict[str, Any]:
            filters: dict[str, Any] = {}
            optional_filters = {
                "class": class_code,
                "subject": subject,
                "module": module,
                "week": week,
                "status": status,
                "material_type": material_type,
                "title": title,
                "author": author,
                "doc_id": doc_id,
                "chapter": chapter,
            }
            for key, value in optional_filters.items():
                if value is not None and str(value).strip():
                    filters[key] = value

            study_filter = (
                MemoryFilterInput.model_validate(filters) if filters else None
            )
            return await find(
                ctx,
                query=query,
                collection_name=(
                    collection_name or self.memory_settings.study_collection
                ),
                memory_filter=study_filter,
                query_filter=None,
                top_k=top_k,
                response_mode=response_mode,
                snippet_chars=snippet_chars,
                max_output_chars=max_output_chars,
                metadata_fields=metadata_fields,
                group_by_doc=group_by_doc,
                max_chunks_per_doc=max_chunks_per_doc,
                min_score=min_score,
                use_mmr=use_mmr,
            )

        async def build_context(
            ctx: Context,
            query: Annotated[
                str,
                Field(
                    description=(
                        "Task, question, or search phrase to build a compact context pack for."
                    )
                ),
            ],
            collection_name: Annotated[
                str | None,
                Field(description="Collection to search for second-brain context."),
            ] = None,
            memory_filter: Annotated[
                MemoryFilterInput | None,
                Field(description="Structured exact filters to apply before fallback."),
            ] = None,
            query_filter: ArbitraryFilter | None = None,
            top_k: Annotated[
                int,
                Field(description="Max cited context items to return. Start with 3-5."),
            ] = 5,
            candidate_k: Annotated[
                int | None,
                Field(
                    description=(
                        "Candidate hits to inspect before dedupe. Defaults to top_k*3 "
                        "when grouping by document."
                    )
                ),
            ] = None,
            max_output_chars: Annotated[
                int,
                Field(
                    description=(
                        "Approximate response budget for the packed context, clamped "
                        "to 800-50000 characters."
                    )
                ),
            ] = 4000,
            snippet_chars: Annotated[
                int,
                Field(description="Max characters per snippet, clamped to 40-1200."),
            ] = 360,
            metadata_fields: Annotated[
                list[str] | None,
                Field(
                    description=(
                        "Metadata fields to include. Defaults to citation-friendly "
                        "identity, document, source, and section fields."
                    )
                ),
            ] = None,
            group_by_doc: Annotated[
                bool,
                Field(
                    description="Avoid repeated chunks from the same document/source."
                ),
            ] = True,
            max_chunks_per_doc: Annotated[
                int,
                Field(description="Max chunks per document/source when grouping."),
            ] = 2,
            min_score: Annotated[
                float | None,
                Field(description="Optional minimum similarity score."),
            ] = None,
            fallback_strategy: Annotated[
                Literal["none", "relax_filters"],
                Field(
                    description=(
                        "Fallback behavior when filtered search returns no context. "
                        "relax_filters retries the same query without exact filters."
                    )
                ),
            ] = "relax_filters",
            use_mmr: Annotated[
                bool,
                Field(description="Enable MMR diversity for the candidate set."),
            ] = False,
            mmr_lambda: Annotated[
                float,
                Field(description="MMR trade-off between relevance and diversity."),
            ] = 0.5,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "query": query,
                    "collection_name": collection_name,
                    "memory_filter": memory_filter,
                    "query_filter": query_filter,
                    "top_k": top_k,
                    "candidate_k": candidate_k,
                    "max_output_chars": max_output_chars,
                    "snippet_chars": snippet_chars,
                    "metadata_fields": metadata_fields,
                    "group_by_doc": group_by_doc,
                    "max_chunks_per_doc": max_chunks_per_doc,
                    "min_score": min_score,
                    "fallback_strategy": fallback_strategy,
                    "use_mmr": use_mmr,
                    "mmr_lambda": mmr_lambda,
                },
            )
            if top_k <= 0:
                raise ValueError("top_k must be positive.")
            if candidate_k is not None and candidate_k <= 0:
                raise ValueError("candidate_k must be positive when provided.")
            if max_chunks_per_doc <= 0:
                raise ValueError("max_chunks_per_doc must be positive.")
            if fallback_strategy not in {"none", "relax_filters"}:
                raise ValueError("fallback_strategy must be none or relax_filters.")
            if mmr_lambda < 0 or mmr_lambda > 1:
                if self.memory_settings.strict_params:
                    raise ValueError("mmr_lambda must be between 0 and 1.")
                state.warnings.append("mmr_lambda clamped to [0,1].")
                mmr_lambda = max(0.0, min(1.0, mmr_lambda))

            collection = resolve_collection_name(collection_name)
            snippet_chars = clamp_snippet_chars(snippet_chars)
            normalized_metadata_fields = (
                normalize_metadata_fields(metadata_fields) or CONTEXT_METADATA_FIELDS
            )
            limit = min(top_k, 25)
            if top_k > limit:
                state.warnings.append("top_k clamped to 25 for context packing.")
            resolved_candidate_k = candidate_k
            if resolved_candidate_k is None:
                resolved_candidate_k = limit * 3 if group_by_doc else limit
            resolved_candidate_k = min(max(resolved_candidate_k, limit), 100)

            combined_filter = resolve_combined_filter(
                memory_filter,
                query_filter,
                state.warnings,
            )
            query_vector = await self._embed_query_cached(query)
            vector_name = (
                await self.qdrant_connector.resolve_vector_name(collection)
                if use_mmr
                else None
            )
            search_attempts: list[dict[str, Any]] = []

            async def run_context_attempt(
                strategy: str, active_filter: models.Filter | None
            ) -> list[models.ScoredPoint]:
                points = await self.qdrant_connector.query_points(
                    query_vector,
                    collection_name=collection,
                    limit=resolved_candidate_k,
                    query_filter=active_filter,
                    with_vectors=use_mmr,
                )
                raw_count = len(points)
                if use_mmr and points:
                    mmr_points = mmr_select(
                        query_vector,
                        points,
                        top_k=resolved_candidate_k,
                        lambda_mult=mmr_lambda,
                        vector_name=vector_name,
                    )
                    if mmr_points is None:
                        state.warnings.append("MMR disabled due to missing vectors.")
                    else:
                        points = mmr_points

                selected = select_points_for_output(
                    points,
                    limit=limit,
                    group_by_doc=group_by_doc,
                    max_chunks_per_doc=max_chunks_per_doc,
                    min_score=min_score,
                )
                search_attempts.append(
                    {
                        "strategy": strategy,
                        "filter_applied": active_filter is not None,
                        "candidate_count": raw_count,
                        "selected_count": len(selected),
                    }
                )
                return selected

            selected_points = await run_context_attempt("primary", combined_filter)
            fallback_used = False
            if (
                not selected_points
                and combined_filter is not None
                and fallback_strategy == "relax_filters"
            ):
                fallback_used = True
                selected_points = await run_context_attempt("relax_filters", None)

            context_items: list[dict[str, Any]] = []
            for index, point in enumerate(selected_points, start=1):
                item = format_search_result(
                    point,
                    response_mode="compact",
                    snippet_chars=snippet_chars,
                    metadata_fields=normalized_metadata_fields,
                )
                item["citation"] = f"[{index}]"
                context_items.append(item)

            base_data: dict[str, Any] = {
                "query": query,
                "collection_name": collection,
                "search_attempts": search_attempts,
                "fallback_used": fallback_used,
            }
            context_items, context_text, budget = trim_context_to_budget(
                context_items,
                base_data=base_data,
                max_output_chars=max_output_chars,
            )
            data = {
                **base_data,
                "context": context_items,
                "context_text": context_text,
                "result_count": len(context_items),
                "usage": (
                    "Use citation labels from context_text. Fetch payloads only for "
                    "selected ids that need full source text."
                ),
            }
            extra_meta = {
                "top_k": limit,
                "candidate_limit": resolved_candidate_k,
                "query_vector_dim": len(query_vector),
                "response_mode": "compact",
                "metadata_fields": normalized_metadata_fields,
                "group_by_doc": group_by_doc,
                "max_chunks_per_doc": max_chunks_per_doc if group_by_doc else None,
                "min_score": min_score,
                "budget": budget,
            }
            return finish_request(state, data, extra_meta=extra_meta)

        async def recommend_memories(
            ctx: Context,
            positive_ids: Annotated[
                list[str], Field(description="Point ids to treat as positive examples.")
            ],
            collection_name: Annotated[
                str, Field(description="The collection to search in")
            ],
            negative_ids: Annotated[
                list[str] | None,
                Field(description="Optional point ids to treat as negatives."),
            ] = None,
            memory_filter: Annotated[
                MemoryFilterInput | None,
                Field(description="Structured filters for memory fields."),
            ] = None,
            query_filter: ArbitraryFilter | None = None,
            top_k: Annotated[
                int | None,
                Field(description="Max number of results to return. Start with 3-5."),
            ] = None,
            response_mode: Annotated[
                Literal["compact", "metadata", "payload"],
                Field(
                    description=(
                        "Result detail level. compact returns ids, scores, snippets, "
                        "and selected metadata; payload returns full Qdrant payloads."
                    )
                ),
            ] = "compact",
            snippet_chars: Annotated[
                int,
                Field(
                    description="Max snippet length in characters, clamped to 40-1200."
                ),
            ] = 240,
            max_output_chars: Annotated[
                int | None,
                Field(
                    description=(
                        "Optional approximate response budget. When set, results are "
                        "trimmed to fit 800-50000 characters."
                    )
                ),
            ] = None,
            metadata_fields: Annotated[
                list[str] | None,
                Field(
                    description="Optional metadata field allowlist for compact output."
                ),
            ] = None,
            group_by_doc: Annotated[
                bool,
                Field(description="Limit repeated chunks from the same document."),
            ] = False,
            max_chunks_per_doc: Annotated[
                int,
                Field(description="Max chunks per document when group_by_doc=true."),
            ] = 2,
            min_score: Annotated[
                float | None,
                Field(
                    description="Optional minimum similarity score for returned hits."
                ),
            ] = None,
            negative_weight: Annotated[
                float,
                Field(description="Weight for negative vectors in the blend."),
            ] = 1.0,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "positive_ids": positive_ids,
                    "negative_ids": negative_ids,
                    "collection_name": collection_name,
                    "memory_filter": memory_filter,
                    "query_filter": query_filter,
                    "top_k": top_k,
                    "response_mode": response_mode,
                    "snippet_chars": snippet_chars,
                    "max_output_chars": max_output_chars,
                    "metadata_fields": metadata_fields,
                    "group_by_doc": group_by_doc,
                    "max_chunks_per_doc": max_chunks_per_doc,
                    "min_score": min_score,
                    "negative_weight": negative_weight,
                },
            )

            if not positive_ids:
                raise ValueError("positive_ids cannot be empty.")
            enforce_point_ids(positive_ids, name="positive_ids")
            if negative_ids is not None:
                if not negative_ids:
                    raise ValueError("negative_ids cannot be empty.")
                enforce_point_ids(negative_ids, name="negative_ids")
            if negative_weight < 0:
                raise ValueError("negative_weight must be >= 0.")

            positive_set = {str(pid) for pid in positive_ids}
            negative_set = {str(pid) for pid in (negative_ids or [])}
            overlap = positive_set & negative_set
            if overlap:
                state.warnings.append(
                    "positive_ids and negative_ids overlap; removing from negatives."
                )
                negative_set -= overlap

            memory_filter_obj = build_memory_filter(
                memory_filter,
                strict=self.memory_settings.strict_params,
                warnings=state.warnings,
            )

            query_filter_obj = None
            if query_filter:
                if not self.qdrant_settings.allow_arbitrary_filter:
                    if self.memory_settings.strict_params:
                        raise ValueError("query_filter is not allowed.")
                    state.warnings.append("query_filter ignored (not allowed).")
                else:
                    query_filter_obj = models.Filter(**query_filter)

            combined_filter = merge_filters([memory_filter_obj, query_filter_obj])

            limit = top_k or self.qdrant_settings.search_limit
            if limit <= 0:
                raise ValueError("top_k must be positive.")
            if max_chunks_per_doc <= 0:
                raise ValueError("max_chunks_per_doc must be positive.")
            snippet_chars = clamp_snippet_chars(snippet_chars)
            selected_metadata_fields = normalize_metadata_fields(metadata_fields)

            collection = resolve_collection_name(collection_name)
            vector_name = await self.qdrant_connector.resolve_vector_name(collection)
            requested_ids = [*positive_set, *negative_set]
            records = await self.qdrant_connector.retrieve_points(
                requested_ids,
                collection_name=collection,
                with_payload=False,
                with_vectors=True,
            )
            found_ids = {str(record.id) for record in records}
            missing_ids = [pid for pid in requested_ids if pid not in found_ids]

            positive_vectors: list[list[float]] = []
            negative_vectors: list[list[float]] = []
            for record in records:
                record_id = str(record.id)
                vector = extract_vector(record.vector, vector_name)
                if vector is None:
                    state.warnings.append(
                        f"Missing vector for {record_id}; skipped in blend."
                    )
                    continue
                if record_id in positive_set:
                    positive_vectors.append(vector)
                elif record_id in negative_set:
                    negative_vectors.append(vector)

            if not positive_vectors:
                raise ValueError("No vectors found for positive_ids.")
            if negative_set and not negative_vectors:
                state.warnings.append("No vectors found for negative_ids.")

            query_vector = average_vectors(positive_vectors)
            if negative_vectors and negative_weight > 0:
                negative_vector = average_vectors(negative_vectors)
                query_vector = [
                    pos - (negative_weight * neg)
                    for pos, neg in zip(query_vector, negative_vector)
                ]

            exclude_ids = positive_set | negative_set
            fetch_limit = min(limit + len(exclude_ids), limit * 2 + 50)
            if group_by_doc or min_score is not None:
                fetch_limit = min(max(fetch_limit, limit * 3), 100)
            points = await self.qdrant_connector.query_points(
                query_vector,
                collection_name=collection,
                limit=fetch_limit,
                query_filter=combined_filter,
                with_vectors=False,
            )

            results = []
            candidate_points = [
                point for point in points if str(point.id) not in exclude_ids
            ]
            selected_points = select_points_for_output(
                candidate_points,
                limit=limit,
                group_by_doc=group_by_doc,
                max_chunks_per_doc=max_chunks_per_doc,
                min_score=min_score,
            )
            for point in selected_points:
                results.append(
                    format_search_result(
                        point,
                        response_mode=response_mode,
                        snippet_chars=snippet_chars,
                        metadata_fields=selected_metadata_fields,
                    )
                )

            base_data: dict[str, Any] = {
                "collection_name": collection,
                "missing": missing_ids,
                "excluded": sorted(exclude_ids),
            }
            results, budget = trim_results_to_budget(
                results,
                base_data=base_data,
                max_output_chars=max_output_chars,
            )
            data = {**base_data, "results": results}
            extra_meta = {
                "top_k": limit,
                "candidate_limit": fetch_limit,
                "filter_applied": combined_filter is not None,
                "query_vector_dim": len(query_vector),
                "response_mode": response_mode,
                "metadata_fields": selected_metadata_fields,
                "group_by_doc": group_by_doc,
                "max_chunks_per_doc": max_chunks_per_doc if group_by_doc else None,
                "min_score": min_score,
                "budget": budget,
            }
            return finish_request(state, data, extra_meta=extra_meta)

        async def list_points(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The collection to list points from.")
            ],
            memory_filter: Annotated[
                MemoryFilterInput | None,
                Field(description="Structured filters for memory fields."),
            ] = None,
            query_filter: ArbitraryFilter | None = None,
            limit: Annotated[
                int | None, Field(description="Max points to return.")
            ] = 50,
            offset: Annotated[
                str | int | None, Field(description="Scroll offset to resume from.")
            ] = None,
            include_payload: Annotated[
                bool, Field(description="Include payload data.")
            ] = True,
            include_vectors: Annotated[
                bool, Field(description="Include vector data.")
            ] = False,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "collection_name": collection_name,
                    "memory_filter": memory_filter,
                    "query_filter": query_filter,
                    "limit": limit,
                    "offset": offset,
                    "include_payload": include_payload,
                    "include_vectors": include_vectors,
                },
            )
            resolved_limit = 50 if limit is None else limit
            if resolved_limit <= 0:
                raise ValueError("limit must be positive.")
            enforce_batch_size(resolved_limit, name="limit")

            combined_filter = resolve_combined_filter(
                memory_filter, query_filter, state.warnings
            )
            collection = resolve_collection_name(collection_name)

            points, next_offset = await self.qdrant_connector.scroll_points_page(
                collection_name=collection,
                query_filter=combined_filter,
                limit=resolved_limit,
                with_payload=include_payload,
                with_vectors=include_vectors,
                offset=parse_offset(offset),
            )

            items: list[dict[str, Any]] = []
            for point in points:
                item: dict[str, Any] = {"id": str(point.id)}
                if include_payload:
                    item["payload"] = point.payload
                if include_vectors:
                    item["vector"] = point.vector
                items.append(item)

            data = {
                "collection_name": collection,
                "points": items,
                "count": len(items),
                "next_offset": str(next_offset) if next_offset is not None else None,
            }
            return finish_request(state, data)

        async def get_points(
            ctx: Context,
            point_ids: Annotated[
                list[str], Field(description="Point ids to retrieve.")
            ],
            collection_name: Annotated[
                str, Field(description="The collection containing the points.")
            ],
            include_payload: Annotated[
                bool, Field(description="Include payload data.")
            ] = True,
            include_vectors: Annotated[
                bool, Field(description="Include vector data.")
            ] = False,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "point_ids": point_ids,
                    "collection_name": collection_name,
                    "include_payload": include_payload,
                    "include_vectors": include_vectors,
                },
            )
            if not point_ids:
                raise ValueError("point_ids cannot be empty.")
            enforce_point_ids(point_ids)
            enforce_point_ids(point_ids)

            collection = resolve_collection_name(collection_name)
            records = await self.qdrant_connector.retrieve_points(
                point_ids,
                collection_name=collection,
                with_payload=include_payload,
                with_vectors=include_vectors,
            )

            items: list[dict[str, Any]] = []
            found_ids: set[str] = set()
            for record in records:
                record_id = str(record.id)
                found_ids.add(record_id)
                item: dict[str, Any] = {"id": record_id}
                if include_payload:
                    item["payload"] = record.payload
                if include_vectors:
                    item["vector"] = record.vector
                items.append(item)

            missing = [pid for pid in point_ids if str(pid) not in found_ids]
            data = {
                "collection_name": collection,
                "points": items,
                "count": len(items),
                "missing_ids": missing,
            }
            return finish_request(state, data)

        async def count_points(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The collection to count points in.")
            ],
            memory_filter: Annotated[
                MemoryFilterInput | None,
                Field(description="Structured filters for memory fields."),
            ] = None,
            query_filter: ArbitraryFilter | None = None,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "collection_name": collection_name,
                    "memory_filter": memory_filter,
                    "query_filter": query_filter,
                },
            )
            combined_filter = resolve_combined_filter(
                memory_filter, query_filter, state.warnings
            )
            collection = resolve_collection_name(collection_name)
            count = await self.qdrant_connector.count_points(
                collection_name=collection,
                query_filter=combined_filter,
            )
            data = {"collection_name": collection, "count": count}
            return finish_request(state, data)

        async def audit_memories(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="Collection to audit.")
            ] = "",
            memory_filter: Annotated[
                MemoryFilterInput | None,
                Field(description="Structured filters for memory fields."),
            ] = None,
            query_filter: ArbitraryFilter | None = None,
            batch_size: Annotated[
                int, Field(description="Batch size for scanning.")
            ] = 100,
            max_points: Annotated[
                int | None, Field(description="Max points to scan (None for all).")
            ] = None,
            include_samples: Annotated[
                bool, Field(description="Include sample point ids for issues.")
            ] = False,
            sample_limit: Annotated[
                int, Field(description="Max samples per issue type.")
            ] = 5,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "collection_name": collection_name,
                    "memory_filter": memory_filter,
                    "query_filter": query_filter,
                    "batch_size": batch_size,
                    "max_points": max_points,
                    "include_samples": include_samples,
                    "sample_limit": sample_limit,
                },
            )
            if batch_size <= 0:
                raise ValueError("batch_size must be positive.")
            enforce_batch_size(batch_size)
            if sample_limit < 0:
                raise ValueError("sample_limit must be non-negative.")
            enforce_batch_size(batch_size)

            combined_filter = resolve_combined_filter(
                memory_filter, query_filter, state.warnings
            )
            collection = resolve_collection_name(collection_name)

            scanned = 0
            missing_payload = 0
            missing_text = 0
            missing_metadata = 0
            missing_required = 0
            needs_backfill = 0

            duplicate_stats: dict[tuple[str, str], dict[str, Any]] = {}

            samples: dict[str, list[Any]] = {}
            if include_samples:
                samples = {
                    "missing_payload": [],
                    "missing_text": [],
                    "missing_metadata": [],
                    "missing_required_fields": [],
                    "duplicate_groups": [],
                }

            offset = None
            stop = False

            if isinstance(ctx, JobContext):
                ctx.set_phase("scanning")
                ctx.set_total(max_points)

            while True:
                points, offset = await self.qdrant_connector.scroll_points_page(
                    collection_name=collection,
                    query_filter=combined_filter,
                    limit=batch_size,
                    with_payload=True,
                    offset=offset,
                )
                if not points:
                    break

                for point in points:
                    scanned += 1
                    if max_points is not None and scanned > max_points:
                        stop = True
                        break

                    payload = point.payload or {}
                    if not payload:
                        missing_payload += 1
                        if (
                            include_samples
                            and len(samples["missing_payload"]) < sample_limit
                        ):
                            samples["missing_payload"].append(str(point.id))

                    metadata = payload.get(METADATA_PATH) or payload.get("metadata")
                    if not isinstance(metadata, dict):
                        missing_metadata += 1
                        metadata = {}
                        if (
                            include_samples
                            and len(samples["missing_metadata"]) < sample_limit
                        ):
                            samples["missing_metadata"].append(str(point.id))

                    text = extract_payload_text(payload)
                    if not text:
                        missing_text += 1
                        if (
                            include_samples
                            and len(samples["missing_text"]) < sample_limit
                        ):
                            samples["missing_text"].append(str(point.id))

                    if metadata:
                        missing_fields = sorted(
                            field for field in REQUIRED_FIELDS if field not in metadata
                        )
                    else:
                        missing_fields = sorted(REQUIRED_FIELDS)

                    if missing_fields:
                        missing_required += 1
                        if (
                            include_samples
                            and len(samples["missing_required_fields"]) < sample_limit
                        ):
                            samples["missing_required_fields"].append(
                                {"id": str(point.id), "missing": missing_fields}
                            )

                    if text:
                        patch, _ = build_memory_backfill_patch(
                            text=text,
                            metadata=metadata,
                            embedding_info=self.embedding_info,
                            strict=False,
                        )
                        if patch:
                            needs_backfill += 1

                    text_hash = metadata.get("text_hash")
                    scope = metadata.get("scope")
                    if isinstance(text_hash, str) and isinstance(scope, str):
                        key = (scope, text_hash)
                        entry = duplicate_stats.get(key)
                        if entry is None:
                            duplicate_stats[key] = {
                                "count": 1,
                                "ids": [str(point.id)],
                            }
                        else:
                            entry["count"] += 1
                            if include_samples and len(entry["ids"]) < sample_limit:
                                entry["ids"].append(str(point.id))

                if isinstance(ctx, JobContext):
                    ctx.advance(len(points))
                if stop or offset is None:
                    break

            duplicate_groups = 0
            duplicate_points = 0
            if duplicate_stats:
                for (scope, text_hash), entry in duplicate_stats.items():
                    count = entry["count"]
                    if count > 1:
                        duplicate_groups += 1
                        duplicate_points += count - 1
                        if (
                            include_samples
                            and len(samples["duplicate_groups"]) < sample_limit
                        ):
                            samples["duplicate_groups"].append(
                                {
                                    "scope": scope,
                                    "text_hash": text_hash,
                                    "count": count,
                                    "ids": entry["ids"],
                                }
                            )

            data: dict[str, Any] = {
                "collection_name": collection,
                "scanned": scanned,
                "missing_payload": missing_payload,
                "missing_text": missing_text,
                "missing_metadata": missing_metadata,
                "missing_required_fields": missing_required,
                "needs_backfill": needs_backfill,
                "duplicate_groups": duplicate_groups,
                "duplicate_points": duplicate_points,
            }
            if include_samples:
                data["samples"] = samples
            data["next_offset"] = str(offset) if offset is not None and stop else None
            if max_points is not None:
                data["max_points"] = max_points
            return finish_request(state, data)

        async def find_near_duplicates(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="Collection to scan.")
            ] = "",
            memory_filter: Annotated[
                MemoryFilterInput | None,
                Field(description="Structured filters for memory fields."),
            ] = None,
            query_filter: ArbitraryFilter | None = None,
            batch_size: Annotated[
                int, Field(description="Batch size for scanning.")
            ] = 100,
            max_points: Annotated[
                int | None, Field(description="Max points to scan (None for all).")
            ] = None,
            threshold: Annotated[
                float, Field(description="Cosine similarity threshold.")
            ] = 0.985,
            group_by: Annotated[
                list[str] | None, Field(description="Metadata fields to group by.")
            ] = None,
            include_missing_group: Annotated[
                bool, Field(description="Include points missing group fields.")
            ] = False,
            max_group_size: Annotated[
                int, Field(description="Skip groups larger than this size.")
            ] = 200,
            include_snippets: Annotated[
                bool, Field(description="Include text snippets for review.")
            ] = True,
            max_clusters: Annotated[
                int, Field(description="Max clusters to return.")
            ] = 100,
            max_pairs_per_cluster: Annotated[
                int, Field(description="Max pair samples per cluster.")
            ] = 10,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "collection_name": collection_name,
                    "memory_filter": memory_filter,
                    "query_filter": query_filter,
                    "batch_size": batch_size,
                    "max_points": max_points,
                    "threshold": threshold,
                    "group_by": group_by,
                    "include_missing_group": include_missing_group,
                    "max_group_size": max_group_size,
                    "include_snippets": include_snippets,
                    "max_clusters": max_clusters,
                    "max_pairs_per_cluster": max_pairs_per_cluster,
                },
            )
            if batch_size <= 0:
                raise ValueError("batch_size must be positive.")
            enforce_batch_size(batch_size)
            if threshold < 0 or threshold > 1:
                raise ValueError("threshold must be between 0 and 1.")
            if max_group_size < 2:
                raise ValueError("max_group_size must be >= 2.")
            if max_clusters <= 0:
                raise ValueError("max_clusters must be positive.")
            if max_pairs_per_cluster < 0:
                raise ValueError("max_pairs_per_cluster must be non-negative.")

            def serialize_group_value(value: Any) -> str:
                if value is None:
                    return ""
                if isinstance(value, (str, int, float, bool)):
                    return str(value)
                try:
                    return json.dumps(value, sort_keys=True, default=str)
                except TypeError:
                    return str(value)

            group_fields = group_by if group_by is not None else ["doc_id"]

            combined_filter = resolve_combined_filter(
                memory_filter, query_filter, state.warnings
            )
            collection = resolve_collection_name(collection_name)
            vector_name = await self.qdrant_connector.resolve_vector_name(collection)

            scanned = 0
            missing_vectors = 0
            missing_group = 0
            oversize_groups = 0
            oversize_samples: list[dict[str, Any]] = []

            groups: dict[tuple[str, ...], list[dict[str, Any]]] = {}
            group_values: dict[tuple[str, ...], dict[str, Any]] = {}
            oversize_keys: set[tuple[str, ...]] = set()

            offset = None
            stop = False

            if isinstance(ctx, JobContext):
                ctx.set_phase("scanning")
                ctx.set_total(max_points)

            while True:
                points, offset = await self.qdrant_connector.scroll_points_page(
                    collection_name=collection,
                    query_filter=combined_filter,
                    limit=batch_size,
                    with_payload=True,
                    with_vectors=True,
                    offset=offset,
                )
                if not points:
                    break

                for point in points:
                    scanned += 1
                    if max_points is not None and scanned > max_points:
                        stop = True
                        break

                    vector = extract_vector(point.vector, vector_name)
                    if vector is None:
                        missing_vectors += 1
                        continue

                    payload = point.payload or {}
                    metadata = (
                        payload.get(METADATA_PATH) or payload.get("metadata") or {}
                    )
                    if not isinstance(metadata, dict):
                        metadata = {}

                    if group_fields:
                        values: list[str] = []
                        missing = False
                        for field in group_fields:
                            value = metadata.get(field)
                            if value is None:
                                missing = True
                            values.append(serialize_group_value(value))
                        if missing and not include_missing_group:
                            missing_group += 1
                            continue
                        group_key = tuple(values)
                        if group_key not in group_values:
                            group_values[group_key] = {
                                field: metadata.get(field) for field in group_fields
                            }
                    else:
                        group_key = ("__all__",)
                        group_values.setdefault(group_key, {})

                    if group_key in oversize_keys:
                        continue

                    bucket = groups.setdefault(group_key, [])
                    if len(bucket) >= max_group_size:
                        oversize_keys.add(group_key)
                        groups.pop(group_key, None)
                        oversize_groups += 1
                        if len(oversize_samples) < 5:
                            oversize_samples.append(group_values.get(group_key, {}))
                        continue

                    text = extract_payload_text(payload)
                    snippet = make_snippet(text) if include_snippets else None
                    bucket.append(
                        {
                            "id": str(point.id),
                            "vector": vector,
                            "snippet": snippet,
                        }
                    )

                if isinstance(ctx, JobContext):
                    ctx.advance(len(points))
                if stop or offset is None:
                    break

            clusters: list[dict[str, Any]] = []
            total_clusters = 0
            total_candidate_points = 0

            for group_key, items in groups.items():
                if len(items) < 2:
                    continue

                n = len(items)
                parent = list(range(n))

                def find_index(index: int) -> int:
                    while parent[index] != index:
                        parent[index] = parent[parent[index]]
                        index = parent[index]
                    return index

                def union(a: int, b: int) -> None:
                    root_a = find_index(a)
                    root_b = find_index(b)
                    if root_a != root_b:
                        parent[root_b] = root_a

                pair_edges: list[tuple[int, int, float]] = []
                for i in range(n):
                    vec_i = items[i]["vector"]
                    for j in range(i + 1, n):
                        score = cosine_similarity(vec_i, items[j]["vector"])
                        if score >= threshold:
                            union(i, j)
                            pair_edges.append((i, j, score))

                if not pair_edges:
                    continue

                clusters_map: dict[int, list[int]] = {}
                for i in range(n):
                    root = find_index(i)
                    clusters_map.setdefault(root, []).append(i)

                pair_samples: dict[int, list[dict[str, Any]]] = {}
                if max_pairs_per_cluster > 0:
                    for i, j, score in pair_edges:
                        root = find_index(i)
                        if root != find_index(j):
                            continue
                        bucket = pair_samples.setdefault(root, [])
                        if len(bucket) >= max_pairs_per_cluster:
                            continue
                        bucket.append(
                            {
                                "a": items[i]["id"],
                                "b": items[j]["id"],
                                "score": round(score, 6),
                            }
                        )

                for root, indices in clusters_map.items():
                    if len(indices) < 2:
                        continue
                    total_clusters += 1
                    total_candidate_points += len(indices)
                    if len(clusters) >= max_clusters:
                        continue

                    cluster = {
                        "group": group_values.get(group_key, {}),
                        "ids": [items[i]["id"] for i in indices],
                        "count": len(indices),
                    }
                    if include_snippets:
                        cluster["snippets"] = [
                            {"id": items[i]["id"], "snippet": items[i]["snippet"]}
                            for i in indices
                        ]
                    samples = pair_samples.get(root)
                    if samples:
                        cluster["pairs"] = samples
                    clusters.append(cluster)

            data: dict[str, Any] = {
                "collection_name": collection,
                "scanned": scanned,
                "threshold": threshold,
                "group_by": group_fields,
                "groups_scanned": len(groups),
                "missing_vectors": missing_vectors,
                "missing_group_values": missing_group,
                "oversize_groups": oversize_groups,
                "clusters_count": total_clusters,
                "candidate_points": total_candidate_points,
                "clusters_returned": len(clusters),
                "clusters": clusters,
            }
            if oversize_samples:
                data["oversize_group_samples"] = oversize_samples
            if max_points is not None:
                data["max_points"] = max_points
            if stop and offset is not None:
                data["next_offset"] = str(offset)
            if total_clusters > len(clusters):
                data["truncated"] = True
            return finish_request(state, data)

        async def update_point(
            ctx: Context,
            point_id: Annotated[str, Field(description="Point id to update.")],
            information: Annotated[str, Field(description="Updated memory text.")],
            collection_name: Annotated[
                str, Field(description="The collection containing the point.")
            ],
            metadata: Annotated[
                Metadata | None,
                Field(description="Updated memory metadata."),
            ] = None,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "point_id": point_id,
                    "information": information,
                    "collection_name": collection_name,
                    "metadata": metadata,
                },
            )
            ensure_mutations_allowed()
            collection = resolve_collection_name(collection_name)

            records, warnings = normalize_memory_input(
                information=information,
                metadata=metadata,
                memory=None,
                embedding_info=self.embedding_info,
                strict=self.memory_settings.strict_params,
                max_text_length=self.memory_settings.max_text_length,
            )
            state.warnings.extend(warnings)

            if len(records) != 1:
                raise ValueError("update_point does not support chunked payloads.")

            entry = Entry(content=records[0].text, metadata=records[0].metadata)
            await self.qdrant_connector.store(
                entry, collection_name=collection, point_id=point_id
            )
            data = {
                "status": "updated",
                "id": point_id,
                "collection_name": collection,
            }
            return finish_request(state, data)

        async def patch_payload(
            ctx: Context,
            point_id: Annotated[str, Field(description="Point id to patch.")],
            collection_name: Annotated[
                str, Field(description="The collection containing the point.")
            ],
            metadata_patch: Annotated[
                Metadata | None,
                Field(description="Partial metadata patch."),
            ] = None,
            payload_patch: Annotated[
                Metadata | None,
                Field(description="Partial top-level payload patch."),
            ] = None,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "point_id": point_id,
                    "collection_name": collection_name,
                    "metadata_patch": metadata_patch,
                    "payload_patch": payload_patch,
                },
            )
            ensure_mutations_allowed()
            if not metadata_patch and not payload_patch:
                raise ValueError("metadata_patch or payload_patch is required.")

            if metadata_patch and "text" in metadata_patch:
                raise ValueError("Use qdrant-update-point to change text.")
            if payload_patch and "document" in payload_patch:
                raise ValueError("Use qdrant-update-point to change document.")

            if metadata_patch and self.memory_settings.strict_params:
                extras = set(metadata_patch.keys()) - ALLOWED_MEMORY_KEYS
                if extras:
                    raise ValueError(f"Unknown metadata keys: {sorted(extras)}")

            collection = resolve_collection_name(collection_name)
            records = await self.qdrant_connector.retrieve_points(
                [point_id], collection_name=collection
            )
            if not records:
                raise ValueError(f"Point {point_id} not found.")

            existing_payload = records[0].payload or {}
            new_payload = dict(existing_payload)

            if metadata_patch:
                merged_metadata = dict(existing_payload.get(METADATA_PATH) or {})
                merged_metadata.update(metadata_patch)
                now = datetime.now(timezone.utc)
                merged_metadata["updated_at"] = now.isoformat()
                merged_metadata["updated_at_ts"] = int(now.timestamp() * 1000)
                new_payload[METADATA_PATH] = merged_metadata

            if payload_patch:
                new_payload.update(payload_patch)

            await self.qdrant_connector.overwrite_payload(
                [point_id],
                new_payload,
                collection_name=collection,
            )

            data = {
                "status": "patched",
                "id": point_id,
                "collection_name": collection,
            }
            return finish_request(state, data)

        async def tag_memories(
            ctx: Context,
            point_ids: Annotated[
                list[str], Field(description="Point ids to tag with labels.")
            ],
            labels: Annotated[
                list[str], Field(description="Labels to add to the points.")
            ],
            collection_name: Annotated[
                str, Field(description="The collection containing the points.")
            ],
            replace: Annotated[
                bool,
                Field(
                    description="Replace existing labels instead of merging.",
                ),
            ] = False,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "point_ids": point_ids,
                    "labels": labels,
                    "collection_name": collection_name,
                    "replace": replace,
                },
            )
            ensure_mutations_allowed()
            if not point_ids:
                raise ValueError("point_ids cannot be empty.")
            enforce_point_ids(point_ids)

            normalized_labels = [
                str(label).strip() for label in labels if str(label).strip()
            ]
            if not normalized_labels:
                raise ValueError("labels cannot be empty.")

            collection = resolve_collection_name(collection_name)
            records = await self.qdrant_connector.retrieve_points(
                point_ids, collection_name=collection
            )
            found_ids = {str(record.id) for record in records}
            missing_ids = [pid for pid in point_ids if pid not in found_ids]

            updated_ids: list[str] = []
            now = datetime.now(timezone.utc)
            for record in records:
                existing_payload = record.payload or {}
                existing_metadata = dict(existing_payload.get(METADATA_PATH) or {})
                existing_labels = existing_metadata.get("labels")

                if replace:
                    merged_labels = normalized_labels
                else:
                    if isinstance(existing_labels, list):
                        merged_labels = merge_list_values(
                            existing_labels, normalized_labels
                        )
                    elif existing_labels:
                        merged_labels = merge_list_values(
                            [str(existing_labels)], normalized_labels
                        )
                    else:
                        merged_labels = list(normalized_labels)

                existing_metadata["labels"] = merged_labels
                existing_metadata["updated_at"] = now.isoformat()
                existing_metadata["updated_at_ts"] = int(now.timestamp() * 1000)
                new_payload = dict(existing_payload)
                new_payload[METADATA_PATH] = existing_metadata

                await self.qdrant_connector.overwrite_payload(
                    [str(record.id)],
                    new_payload,
                    collection_name=collection,
                )
                updated_ids.append(str(record.id))

            data = {
                "status": "tagged",
                "collection_name": collection,
                "updated": updated_ids,
                "missing": missing_ids,
                "labels": normalized_labels,
                "replace": replace,
            }
            return finish_request(state, data)

        async def link_memories(
            ctx: Context,
            source_id: Annotated[
                str, Field(description="Source point id to link from.")
            ],
            target_ids: Annotated[
                list[str], Field(description="Target point ids to link to.")
            ],
            collection_name: Annotated[
                str, Field(description="The collection containing the points.")
            ],
            relation: Annotated[
                str | None,
                Field(description="Optional relation label for the association."),
            ] = None,
            bidirectional: Annotated[
                bool,
                Field(description="Apply links in both directions."),
            ] = True,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "source_id": source_id,
                    "target_ids": target_ids,
                    "collection_name": collection_name,
                    "relation": relation,
                    "bidirectional": bidirectional,
                },
            )
            ensure_mutations_allowed()
            if not source_id:
                raise ValueError("source_id is required.")
            if not target_ids:
                raise ValueError("target_ids cannot be empty.")
            enforce_point_ids([source_id], name="source_id")
            enforce_point_ids(target_ids, name="target_ids")

            normalized_targets = [str(pid) for pid in target_ids if str(pid).strip()]
            if source_id in normalized_targets:
                normalized_targets = [
                    pid for pid in normalized_targets if pid != source_id
                ]
                state.warnings.append("source_id removed from target_ids.")
            if not normalized_targets:
                raise ValueError("target_ids cannot be empty after normalization.")

            collection = resolve_collection_name(collection_name)
            ids_to_fetch = [source_id, *normalized_targets]
            records = await self.qdrant_connector.retrieve_points(
                ids_to_fetch,
                collection_name=collection,
                with_payload=True,
                with_vectors=False,
            )
            record_map = {str(record.id): record for record in records}
            missing_ids = [pid for pid in ids_to_fetch if pid not in record_map]
            if source_id not in record_map:
                raise ValueError(f"Source point {source_id} not found.")

            updated_ids: list[str] = []
            now = datetime.now(timezone.utc)

            def build_related(existing: Any, additions: list[str]) -> list[str]:
                if isinstance(existing, list):
                    base = [str(item).strip() for item in existing if str(item).strip()]
                elif existing:
                    base = [str(existing).strip()]
                else:
                    base = []
                return merge_list_values(base, additions)

            def build_associations(
                existing: Any, additions: list[dict[str, Any]]
            ) -> list[Any]:
                if isinstance(existing, list):
                    base = list(existing)
                elif existing:
                    base = [existing]
                else:
                    base = []
                return merge_list_values(base, additions)

            update_targets = [pid for pid in normalized_targets if pid in record_map]
            updates: dict[str, list[str]] = {source_id: update_targets}
            if bidirectional:
                for target_id in update_targets:
                    updates[target_id] = [source_id]

            for record_id, add_ids in updates.items():
                record = record_map[record_id]
                payload = dict(record.payload or {})
                metadata = dict(payload.get(METADATA_PATH) or {})
                metadata["related_ids"] = build_related(
                    metadata.get("related_ids"), add_ids
                )

                if relation:
                    assoc_items = [
                        {"id": target_id, "relation": relation} for target_id in add_ids
                    ]
                    metadata["associations"] = build_associations(
                        metadata.get("associations"), assoc_items
                    )

                metadata["updated_at"] = now.isoformat()
                metadata["updated_at_ts"] = int(now.timestamp() * 1000)
                payload[METADATA_PATH] = metadata

                await self.qdrant_connector.overwrite_payload(
                    [record_id],
                    payload,
                    collection_name=collection,
                )
                updated_ids.append(record_id)

            data = {
                "status": "linked",
                "collection_name": collection,
                "source_id": source_id,
                "targets": update_targets,
                "updated": updated_ids,
                "missing": missing_ids,
                "relation": relation,
                "bidirectional": bidirectional,
            }
            return finish_request(state, data)

        async def reembed_points(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="Collection to re-embed.")
            ] = "",
            point_ids: Annotated[
                list[str] | None, Field(description="Optional point ids to re-embed.")
            ] = None,
            memory_filter: Annotated[
                MemoryFilterInput | None,
                Field(description="Structured filters for memory fields."),
            ] = None,
            query_filter: ArbitraryFilter | None = None,
            batch_size: Annotated[
                int, Field(description="Batch size for scanning and updates.")
            ] = 64,
            max_points: Annotated[
                int | None, Field(description="Max points to scan (None for all).")
            ] = None,
            target_version: Annotated[
                str | None, Field(description="Target embedding version to enforce.")
            ] = None,
            recompute_text_hash: Annotated[
                bool, Field(description="Recompute text_hash when re-embedding.")
            ] = False,
            dry_run: Annotated[
                bool, Field(description="Report changes without writing.")
            ] = True,
            confirm: Annotated[
                bool, Field(description="Confirm writes when dry_run is false.")
            ] = False,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "collection_name": collection_name,
                    "point_ids": point_ids,
                    "memory_filter": memory_filter,
                    "query_filter": query_filter,
                    "batch_size": batch_size,
                    "max_points": max_points,
                    "target_version": target_version,
                    "recompute_text_hash": recompute_text_hash,
                    "dry_run": dry_run,
                    "confirm": confirm,
                },
            )
            if batch_size <= 0:
                raise ValueError("batch_size must be positive.")
            enforce_batch_size(batch_size)
            if point_ids is not None and not point_ids:
                raise ValueError("point_ids cannot be empty.")
            if point_ids:
                enforce_point_ids(point_ids)
            if point_ids and (memory_filter or query_filter):
                raise ValueError("Provide either point_ids or filters, not both.")
            if point_ids is None and memory_filter is None and query_filter is None:
                state.warnings.append(
                    "No filters provided; re-embed applies to all points."
                )

            resolved_version = target_version or self.embedding_info.version
            if target_version and target_version != self.embedding_info.version:
                raise ValueError(
                    "target_version must match the current embedding version."
                )

            if not dry_run and not confirm:
                state.warnings.append("confirm=true required to apply re-embed.")
                response = {
                    "scanned": 0,
                    "updated": 0,
                    "skipped_version_match": 0,
                    "skipped_missing_text": 0,
                    "dry_run": True,
                }
                return finish_request(state, response)
            if not dry_run:
                ensure_mutations_allowed()

            collection = resolve_collection_name(collection_name)
            vector_name = await self.qdrant_connector.resolve_vector_name(collection)

            scanned = 0
            updated = 0
            skipped_version = 0
            skipped_missing_text = 0
            missing_ids: list[str] = []
            updated_ids_sample: list[str] = []
            skipped_ids_sample: list[str] = []
            dry_run_diff = init_dry_run_diff() if dry_run else None

            if isinstance(ctx, JobContext):
                ctx.set_phase("scanning")
                total = len(point_ids) if point_ids else max_points
                ctx.set_total(total)

            def make_vector_payload(
                embedding: list[float],
            ) -> list[float] | dict[str, list[float]]:
                if vector_name is None:
                    return embedding
                return {vector_name: embedding}

            async def process_records(records: list[models.Record]) -> None:
                nonlocal scanned, updated, skipped_version, skipped_missing_text
                to_embed: list[
                    tuple[models.Record, str, dict[str, Any], dict[str, Any]]
                ] = []

                for record in records:
                    scanned += 1
                    payload = record.payload or {}
                    metadata = (
                        payload.get(METADATA_PATH) or payload.get("metadata") or {}
                    )
                    if not isinstance(metadata, dict):
                        metadata = {}

                    current_version = metadata.get("embedding_version")
                    if current_version == resolved_version:
                        skipped_version += 1
                        if len(skipped_ids_sample) < 20:
                            skipped_ids_sample.append(str(record.id))
                        continue

                    text = extract_payload_text(payload)
                    if not text:
                        skipped_missing_text += 1
                        continue

                    to_embed.append((record, text, metadata, payload))

                if not to_embed:
                    if isinstance(ctx, JobContext):
                        ctx.advance(len(records))
                    return

                if dry_run:
                    now = datetime.now(timezone.utc)
                    now_iso = now.isoformat()
                    now_ms = int(now.timestamp() * 1000)
                    for record, _text, _metadata, _payload in to_embed:
                        metadata_before = _metadata
                        metadata_after = dict(_metadata)
                        metadata_after.update(
                            {
                                "embedding_provider": self.embedding_info.provider,
                                "embedding_model": self.embedding_info.model,
                                "embedding_dim": self.embedding_info.dim,
                                "embedding_version": resolved_version,
                                "updated_at": now_iso,
                                "updated_at_ts": now_ms,
                            }
                        )
                        if recompute_text_hash:
                            metadata_after["text_hash"] = compute_text_hash(_text)
                        updated += 1
                        if len(updated_ids_sample) < 20:
                            updated_ids_sample.append(str(record.id))
                        if dry_run_diff is not None:
                            record_dry_run_action(
                                dry_run_diff,
                                "reembed",
                                str(record.id),
                                metadata_before,
                                metadata_after,
                            )
                    if isinstance(ctx, JobContext):
                        ctx.advance(len(records))
                    return

                embeddings = await self.embedding_provider.embed_documents(
                    [item[1] for item in to_embed]
                )
                now = datetime.now(timezone.utc)
                now_iso = now.isoformat()
                now_ms = int(now.timestamp() * 1000)

                points: list[models.PointStruct] = []
                for (record, text, metadata, payload), embedding in zip(
                    to_embed, embeddings
                ):
                    new_metadata = dict(metadata)
                    new_metadata.update(
                        {
                            "embedding_provider": self.embedding_info.provider,
                            "embedding_model": self.embedding_info.model,
                            "embedding_dim": self.embedding_info.dim,
                            "embedding_version": resolved_version,
                            "updated_at": now_iso,
                            "updated_at_ts": now_ms,
                        }
                    )
                    if recompute_text_hash:
                        new_metadata["text_hash"] = compute_text_hash(text)

                    new_payload = dict(payload)
                    new_payload[METADATA_PATH] = new_metadata
                    points.append(
                        models.PointStruct(
                            id=record.id,
                            vector=make_vector_payload(embedding),
                            payload=new_payload,
                        )
                    )
                    updated += 1
                    if len(updated_ids_sample) < 20:
                        updated_ids_sample.append(str(record.id))

                if not dry_run:
                    await self.qdrant_connector.upsert_points(
                        points, collection_name=collection
                    )
                if isinstance(ctx, JobContext):
                    ctx.advance(len(records))

            if point_ids:
                for start in range(0, len(point_ids), batch_size):
                    chunk = point_ids[start : start + batch_size]
                    records = await self.qdrant_connector.retrieve_points(
                        chunk, collection_name=collection, with_payload=True
                    )
                    found_ids = {str(record.id) for record in records}
                    missing_ids.extend(
                        [pid for pid in chunk if str(pid) not in found_ids]
                    )
                    await process_records(records)
            else:
                combined_filter = resolve_combined_filter(
                    memory_filter, query_filter, state.warnings
                )
                offset = None
                stop = False
                next_offset_override = None
                while True:
                    points, offset = await self.qdrant_connector.scroll_points_page(
                        collection_name=collection,
                        query_filter=combined_filter,
                        limit=batch_size,
                        with_payload=True,
                        offset=offset,
                    )
                    if not points:
                        break

                    if max_points is not None:
                        remaining = max_points - scanned
                        if remaining <= 0:
                            stop = True
                            break
                        if remaining < len(points):
                            points = points[:remaining]
                            stop = True
                            next_offset_override = points[-1].id

                    await process_records(points)

                    if stop or offset is None:
                        break

            data: dict[str, Any] = {
                "collection_name": collection,
                "scanned": scanned,
                "updated": updated,
                "skipped_version_match": skipped_version,
                "skipped_missing_text": skipped_missing_text,
                "embedding_version": resolved_version,
                "dry_run": dry_run,
                "updated_ids_sample": updated_ids_sample,
            }
            if dry_run and dry_run_diff is not None:
                data["dry_run_diff"] = dry_run_diff
            if skipped_ids_sample:
                data["skipped_ids_sample"] = skipped_ids_sample
            if missing_ids:
                data["missing_ids"] = missing_ids
            if max_points is not None:
                data["max_points"] = max_points
            if point_ids is None and stop:
                next_offset = (
                    next_offset_override if next_offset_override is not None else offset
                )
                if next_offset is not None:
                    data["next_offset"] = str(next_offset)
            return finish_request(state, data)

        async def bulk_patch(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The collection to patch.")
            ] = "",
            point_ids: Annotated[
                list[str] | None, Field(description="Optional point ids to patch.")
            ] = None,
            memory_filter: Annotated[
                MemoryFilterInput | None,
                Field(description="Structured filters for memory fields."),
            ] = None,
            query_filter: ArbitraryFilter | None = None,
            metadata_patch: Annotated[
                Metadata | None,
                Field(description="Partial metadata patch."),
            ] = None,
            payload_patch: Annotated[
                Metadata | None,
                Field(description="Partial top-level payload patch."),
            ] = None,
            merge_lists: Annotated[
                bool, Field(description="Merge list fields instead of replacing.")
            ] = True,
            batch_size: Annotated[
                int, Field(description="Batch size for scanning.")
            ] = 100,
            max_points: Annotated[
                int | None, Field(description="Max points to scan (None for all).")
            ] = None,
            dry_run: Annotated[
                bool, Field(description="Report changes without writing.")
            ] = True,
            confirm: Annotated[
                bool, Field(description="Confirm writes when dry_run is false.")
            ] = False,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "collection_name": collection_name,
                    "point_ids": point_ids,
                    "memory_filter": memory_filter,
                    "query_filter": query_filter,
                    "metadata_patch": metadata_patch,
                    "payload_patch": payload_patch,
                    "merge_lists": merge_lists,
                    "batch_size": batch_size,
                    "max_points": max_points,
                    "dry_run": dry_run,
                    "confirm": confirm,
                },
            )
            if not metadata_patch and not payload_patch:
                raise ValueError("metadata_patch or payload_patch is required.")
            if metadata_patch and "text" in metadata_patch:
                raise ValueError("Use qdrant-update-point to change text.")
            if payload_patch and "document" in payload_patch:
                raise ValueError("Use qdrant-update-point to change document.")
            if payload_patch and METADATA_PATH in payload_patch:
                raise ValueError("Use metadata_patch to edit metadata.")
            if batch_size <= 0:
                raise ValueError("batch_size must be positive.")
            enforce_batch_size(batch_size)

            if metadata_patch and self.memory_settings.strict_params:
                extras = set(metadata_patch.keys()) - ALLOWED_MEMORY_KEYS
                if extras:
                    raise ValueError(f"Unknown metadata keys: {sorted(extras)}")

            if point_ids is not None and not point_ids:
                raise ValueError("point_ids cannot be empty.")
            if point_ids:
                enforce_point_ids(point_ids)
            if point_ids and (memory_filter or query_filter):
                raise ValueError("Provide either point_ids or filters, not both.")
            if point_ids is None and memory_filter is None and query_filter is None:
                state.warnings.append(
                    "No filters provided; patch applies to all points."
                )

            if not dry_run and not confirm:
                state.warnings.append("confirm=true required to apply bulk patch.")
                response = {
                    "matched": 0,
                    "updated": 0,
                    "skipped": 0,
                    "dry_run": True,
                }
                return finish_request(state, response)
            if not dry_run:
                ensure_mutations_allowed()

            collection = resolve_collection_name(collection_name)
            updated = 0
            skipped = 0
            scanned = 0
            updated_ids_sample: list[str] = []
            missing_ids: list[str] = []
            offset = None
            stop = False
            dry_run_diff = init_dry_run_diff() if dry_run else None

            now = datetime.now(timezone.utc)
            now_iso = now.isoformat()
            now_ms = int(now.timestamp() * 1000)

            if isinstance(ctx, JobContext):
                ctx.set_phase("scanning")
                total = len(point_ids) if point_ids else max_points
                ctx.set_total(total)

            async def apply_patch_to_point(point: models.Record) -> None:
                nonlocal updated, skipped
                payload = point.payload or {}
                new_payload = dict(payload)

                before_metadata = extract_metadata(payload)
                merged_metadata = before_metadata
                metadata_changed = False
                if metadata_patch:
                    existing_metadata = payload.get(METADATA_PATH) or {}
                    if not isinstance(existing_metadata, dict):
                        existing_metadata = {}
                    merged_metadata = dict(existing_metadata)
                    for key, value in metadata_patch.items():
                        if (
                            merge_lists
                            and isinstance(value, list)
                            and isinstance(existing_metadata.get(key), list)
                        ):
                            merged_metadata[key] = merge_list_values(
                                existing_metadata.get(key), value
                            )
                        else:
                            merged_metadata[key] = value
                    if merged_metadata != existing_metadata:
                        metadata_changed = True
                        merged_metadata["updated_at"] = now_iso
                        merged_metadata["updated_at_ts"] = now_ms
                        new_payload[METADATA_PATH] = merged_metadata

                payload_changed = False
                if payload_patch:
                    for key, value in payload_patch.items():
                        if new_payload.get(key) != value:
                            payload_changed = True
                    if payload_changed:
                        new_payload.update(payload_patch)

                if not metadata_changed and not payload_changed:
                    skipped += 1
                    return

                after_metadata = (
                    merged_metadata if metadata_changed else before_metadata
                )
                updated += 1
                if len(updated_ids_sample) < 20:
                    updated_ids_sample.append(str(point.id))
                if dry_run and dry_run_diff is not None:
                    record_dry_run_action(
                        dry_run_diff,
                        "patch",
                        str(point.id),
                        before_metadata,
                        after_metadata,
                    )
                if not dry_run:
                    await self.qdrant_connector.overwrite_payload(
                        [str(point.id)],
                        new_payload,
                        collection_name=collection,
                    )

            if point_ids:
                records = await self.qdrant_connector.retrieve_points(
                    point_ids,
                    collection_name=collection,
                    with_payload=True,
                )
                found_ids = {str(record.id) for record in records}
                missing_ids = [pid for pid in point_ids if str(pid) not in found_ids]

                for record in records:
                    scanned += 1
                    await apply_patch_to_point(record)
                if isinstance(ctx, JobContext):
                    ctx.advance(len(records))
            else:
                combined_filter = resolve_combined_filter(
                    memory_filter, query_filter, state.warnings
                )
                while True:
                    points, offset = await self.qdrant_connector.scroll_points_page(
                        collection_name=collection,
                        query_filter=combined_filter,
                        limit=batch_size,
                        with_payload=True,
                        offset=offset,
                    )
                    if not points:
                        break

                    for point in points:
                        scanned += 1
                        if max_points is not None and scanned > max_points:
                            stop = True
                            break
                        await apply_patch_to_point(point)
                    if isinstance(ctx, JobContext):
                        ctx.advance(len(points))

                    if stop or offset is None:
                        break

            data: dict[str, Any] = {
                "collection_name": collection,
                "matched": scanned,
                "updated": updated,
                "skipped": skipped,
                "dry_run": dry_run,
                "updated_ids_sample": updated_ids_sample,
            }
            if dry_run and dry_run_diff is not None:
                data["dry_run_diff"] = dry_run_diff
            if missing_ids:
                data["missing_ids"] = missing_ids
            if max_points is not None:
                data["max_points"] = max_points
            if point_ids is None:
                data["next_offset"] = (
                    str(offset) if stop and offset is not None else None
                )
            return finish_request(state, data)

        async def dedupe_memories(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="Collection to dedupe.")
            ] = "",
            memory_filter: Annotated[
                MemoryFilterInput | None,
                Field(description="Structured filters for memory fields."),
            ] = None,
            query_filter: ArbitraryFilter | None = None,
            batch_size: Annotated[
                int, Field(description="Batch size for scanning and deletes.")
            ] = 100,
            max_points: Annotated[
                int | None, Field(description="Max points to scan (None for all).")
            ] = None,
            keep: Annotated[
                str,
                Field(
                    description="Which duplicate to keep: newest, oldest, first, last."
                ),
            ] = "newest",
            merge_metadata: Annotated[
                bool, Field(description="Merge metadata into kept point.")
            ] = False,
            dry_run: Annotated[
                bool, Field(description="Report changes without writing.")
            ] = True,
            confirm: Annotated[
                bool, Field(description="Confirm deletes when dry_run is false.")
            ] = False,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "collection_name": collection_name,
                    "memory_filter": memory_filter,
                    "query_filter": query_filter,
                    "batch_size": batch_size,
                    "max_points": max_points,
                    "keep": keep,
                    "merge_metadata": merge_metadata,
                    "dry_run": dry_run,
                    "confirm": confirm,
                },
            )
            if batch_size <= 0:
                raise ValueError("batch_size must be positive.")
            enforce_batch_size(batch_size)
            if keep not in {"newest", "oldest", "first", "last"}:
                raise ValueError("keep must be newest, oldest, first, or last.")

            if not dry_run and not confirm:
                state.warnings.append("confirm=true required to apply dedupe.")
                response = {
                    "scanned": 0,
                    "duplicate_groups": 0,
                    "duplicate_points": 0,
                    "dry_run": True,
                }
                return finish_request(state, response)
            if not dry_run:
                ensure_mutations_allowed()

            combined_filter = resolve_combined_filter(
                memory_filter, query_filter, state.warnings
            )
            collection = resolve_collection_name(collection_name)

            scanned = 0
            groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
            offset = None
            stop = False
            dry_run_diff = init_dry_run_diff() if dry_run else None

            if isinstance(ctx, JobContext):
                ctx.set_phase("scanning")
                ctx.set_total(max_points)

            while True:
                points, offset = await self.qdrant_connector.scroll_points_page(
                    collection_name=collection,
                    query_filter=combined_filter,
                    limit=batch_size,
                    with_payload=True,
                    offset=offset,
                )
                if not points:
                    break

                for point in points:
                    scanned += 1
                    if max_points is not None and scanned > max_points:
                        stop = True
                        break

                    payload = point.payload or {}
                    metadata = payload.get(METADATA_PATH) or payload.get("metadata")
                    if not isinstance(metadata, dict):
                        metadata = {}

                    text_hash = metadata.get("text_hash")
                    scope = metadata.get("scope")
                    if not isinstance(text_hash, str) or not isinstance(scope, str):
                        continue

                    entry = {
                        "id": str(point.id),
                        "payload": payload,
                        "metadata": metadata,
                        "updated_at_ts": coerce_int(metadata.get("updated_at_ts")),
                        "created_at_ts": coerce_int(metadata.get("created_at_ts")),
                        "last_seen_at_ts": coerce_int(metadata.get("last_seen_at_ts")),
                    }
                    groups.setdefault((scope, text_hash), []).append(entry)

                if isinstance(ctx, JobContext):
                    ctx.advance(len(points))
                if stop or offset is None:
                    break

            duplicate_groups = 0
            duplicate_points = 0
            delete_ids: list[str] = []
            kept_ids_sample: list[str] = []
            delete_ids_sample: list[str] = []
            updated_ids_sample: list[str] = []

            now = datetime.now(timezone.utc)
            now_iso = now.isoformat()
            now_ms = int(now.timestamp() * 1000)

            def pick_keep(entries: list[dict[str, Any]]) -> dict[str, Any]:
                if keep == "first":
                    return entries[0]
                if keep == "last":
                    return entries[-1]

                def score(entry: dict[str, Any]) -> int:
                    return entry.get("updated_at_ts") or entry.get("created_at_ts") or 0

                return (
                    max(entries, key=score)
                    if keep == "newest"
                    else min(entries, key=score)
                )

            for entries in groups.values():
                if len(entries) <= 1:
                    continue
                duplicate_groups += 1
                duplicate_points += len(entries) - 1

                keep_entry = pick_keep(entries)
                keep_id = keep_entry["id"]
                if len(kept_ids_sample) < 20:
                    kept_ids_sample.append(keep_id)

                for entry in entries:
                    if entry["id"] != keep_id:
                        delete_ids.append(entry["id"])
                        if len(delete_ids_sample) < 20:
                            delete_ids_sample.append(entry["id"])
                        if dry_run and dry_run_diff is not None:
                            record_dry_run_action(
                                dry_run_diff,
                                "delete",
                                entry["id"],
                                entry.get("metadata") or {},
                                None,
                            )

                if merge_metadata:
                    merged_metadata = dict(keep_entry["metadata"])

                    for list_key in ("entities", "labels"):
                        incoming: list[Any] = []
                        for entry in entries:
                            value = entry["metadata"].get(list_key)
                            if isinstance(value, list):
                                incoming.extend(value)
                        if incoming:
                            existing_list = (
                                merged_metadata.get(list_key)
                                if isinstance(merged_metadata.get(list_key), list)
                                else None
                            )
                            merged_metadata[list_key] = merge_list_values(
                                existing_list, incoming
                            )

                    reinforcement_total = 0
                    for entry in entries:
                        value = entry["metadata"].get("reinforcement_count")
                        count = coerce_int(value)
                        reinforcement_total += count if count and count > 0 else 1
                    merged_metadata["reinforcement_count"] = max(reinforcement_total, 1)

                    last_seen_candidates: list[int] = []
                    for entry in entries:
                        for ts_value in (
                            entry.get("last_seen_at_ts"),
                            entry.get("updated_at_ts"),
                            entry.get("created_at_ts"),
                        ):
                            if ts_value is not None:
                                last_seen_candidates.append(ts_value)
                    last_seen_ts = (
                        max(last_seen_candidates) if last_seen_candidates else now_ms
                    )
                    merged_metadata["last_seen_at_ts"] = last_seen_ts
                    merged_metadata["last_seen_at"] = datetime.fromtimestamp(
                        last_seen_ts / 1000, tz=timezone.utc
                    ).isoformat()
                    merged_metadata["updated_at"] = now_iso
                    merged_metadata["updated_at_ts"] = now_ms

                    new_payload = dict(keep_entry["payload"])
                    new_payload[METADATA_PATH] = merged_metadata

                    if not dry_run:
                        await self.qdrant_connector.overwrite_payload(
                            [keep_id],
                            new_payload,
                            collection_name=collection,
                        )
                    if len(updated_ids_sample) < 20:
                        updated_ids_sample.append(keep_id)
                    if dry_run and dry_run_diff is not None:
                        record_dry_run_action(
                            dry_run_diff,
                            "merge",
                            keep_id,
                            keep_entry.get("metadata") or {},
                            merged_metadata,
                        )

            if not dry_run and delete_ids:
                for start in range(0, len(delete_ids), batch_size):
                    chunk = delete_ids[start : start + batch_size]
                    await self.qdrant_connector.delete_points(
                        chunk, collection_name=collection
                    )

            data = {
                "collection_name": collection,
                "scanned": scanned,
                "duplicate_groups": duplicate_groups,
                "duplicate_points": duplicate_points,
                "kept_ids_sample": kept_ids_sample,
                "deleted_ids_sample": delete_ids_sample,
                "merged_ids_sample": updated_ids_sample if merge_metadata else [],
                "dry_run": dry_run,
            }
            if dry_run and dry_run_diff is not None:
                data["dry_run_diff"] = dry_run_diff
            if max_points is not None:
                data["max_points"] = max_points
            data["next_offset"] = str(offset) if offset is not None and stop else None
            return finish_request(state, data)

        async def expire_memories(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="Collection to expire from.")
            ] = "",
            batch_size: Annotated[
                int, Field(description="Batch size for scanning and deletes.")
            ] = 200,
            max_points: Annotated[
                int | None, Field(description="Max points to scan (None for all).")
            ] = None,
            archive_collection: Annotated[
                str | None, Field(description="Optional archive collection name.")
            ] = None,
            dry_run: Annotated[
                bool, Field(description="Report changes without writing.")
            ] = True,
            confirm: Annotated[
                bool, Field(description="Confirm deletes when dry_run is false.")
            ] = False,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "collection_name": collection_name,
                    "batch_size": batch_size,
                    "max_points": max_points,
                    "archive_collection": archive_collection,
                    "dry_run": dry_run,
                    "confirm": confirm,
                },
            )
            if batch_size <= 0:
                raise ValueError("batch_size must be positive.")
            enforce_batch_size(batch_size)

            if not dry_run and not confirm:
                state.warnings.append("confirm=true required to expire memories.")
                early_response = {"matched": 0, "deleted": 0, "dry_run": True}
                return finish_request(state, early_response)
            if not dry_run:
                ensure_mutations_allowed()

            collection = resolve_collection_name(collection_name)
            now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
            expire_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=f"{METADATA_PATH}.expires_at_ts",
                        range=models.Range(lt=now_ms),
                    )
                ]
            )

            matched = await self.qdrant_connector.count_points(
                collection_name=collection,
                query_filter=expire_filter,
            )

            if isinstance(ctx, JobContext):
                ctx.set_phase("counting")
                ctx.set_total(max_points)

            if dry_run or not confirm:
                dry_run_diff = init_dry_run_diff()
                scanned = 0
                preview_target = matched
                if max_points is not None:
                    preview_target = min(preview_target, max_points)
                if preview_target > PREVIEW_SCAN_LIMIT:
                    preview_target = PREVIEW_SCAN_LIMIT
                    state.warnings.append(
                        "dry_run preview truncated; increase max_points to scan more."
                    )

                offset = None
                stop = False
                while True:
                    points, offset = await self.qdrant_connector.scroll_points_page(
                        collection_name=collection,
                        query_filter=expire_filter,
                        limit=batch_size,
                        with_payload=True,
                        offset=offset,
                    )
                    if not points:
                        break

                    for point in points:
                        scanned += 1
                        record_dry_run_action(
                            dry_run_diff,
                            "delete",
                            str(point.id),
                            extract_metadata(point.payload or {}),
                            None,
                        )
                        if preview_target and scanned >= preview_target:
                            stop = True
                            break
                    if isinstance(ctx, JobContext):
                        ctx.advance(len(points))
                    if stop or offset is None:
                        break

                response = {
                    "collection_name": collection,
                    "matched": matched,
                    "deleted": 0,
                    "dry_run": True,
                    "dry_run_diff": dry_run_diff,
                    "preview_scanned": scanned,
                    "preview_truncated": scanned < matched,
                }
                return finish_request(state, response)

            deleted = 0
            archived = 0
            scanned = 0
            skipped_missing_vectors = 0
            next_offset = None

            if archive_collection:
                await self.qdrant_connector.ensure_collection_exists(archive_collection)

                offset = None
                stop = False
                while True:
                    points, offset = await self.qdrant_connector.scroll_points_page(
                        collection_name=collection,
                        query_filter=expire_filter,
                        limit=batch_size,
                        with_payload=True,
                        with_vectors=True,
                        offset=offset,
                    )
                    if not points:
                        break

                    if max_points is not None:
                        remaining = max_points - scanned
                        if remaining <= 0:
                            stop = True
                            break
                        if remaining < len(points):
                            points = points[:remaining]
                            stop = True
                            next_offset = str(points[-1].id)

                    for point in points:
                        scanned += 1

                    archive_batch: list[models.PointStruct] = []
                    delete_ids: list[str] = []
                    for point in points:
                        vector = point.vector
                        if vector is None:
                            skipped_missing_vectors += 1
                            continue
                        archive_batch.append(
                            models.PointStruct(
                                id=point.id,
                                vector=vector,
                                payload=point.payload or {},
                            )
                        )
                        delete_ids.append(str(point.id))

                    if archive_batch:
                        await self.qdrant_connector.upsert_points(
                            archive_batch, collection_name=archive_collection
                        )
                        archived += len(archive_batch)

                    if delete_ids:
                        await self.qdrant_connector.delete_points(
                            delete_ids, collection_name=collection
                        )
                        deleted += len(delete_ids)

                    if isinstance(ctx, JobContext):
                        ctx.advance(len(points))

                    if stop or offset is None:
                        if stop and next_offset is None and offset is not None:
                            next_offset = str(offset)
                        break
            else:
                await self.qdrant_connector.delete_by_filter(
                    expire_filter,
                    collection_name=collection,
                )
                deleted = matched

            data: dict[str, Any] = {
                "collection_name": collection,
                "matched": matched,
                "deleted": deleted,
                "archived": archived,
                "skipped_missing_vectors": skipped_missing_vectors,
                "dry_run": False,
            }
            if archive_collection:
                data["archive_collection"] = archive_collection
            if max_points is not None:
                data["max_points"] = max_points
            if next_offset is not None:
                data["next_offset"] = next_offset
            return finish_request(state, data)

        async def merge_duplicates(
            ctx: Context,
            canonical_id: Annotated[
                str, Field(description="Canonical point id to keep.")
            ],
            duplicate_ids: Annotated[
                list[str], Field(description="Duplicate point ids to merge.")
            ],
            collection_name: Annotated[
                str, Field(description="Collection containing the points.")
            ],
            delete_duplicates: Annotated[
                bool, Field(description="Delete duplicates after merge.")
            ] = False,
            mark_merged: Annotated[
                bool, Field(description="Mark duplicates with merged_into.")
            ] = True,
            merge_lists: Annotated[
                bool, Field(description="Merge list fields like entities/labels.")
            ] = True,
            dry_run: Annotated[
                bool, Field(description="Report changes without writing.")
            ] = True,
            confirm: Annotated[
                bool, Field(description="Confirm changes when dry_run is false.")
            ] = False,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "canonical_id": canonical_id,
                    "duplicate_ids": duplicate_ids,
                    "collection_name": collection_name,
                    "delete_duplicates": delete_duplicates,
                    "mark_merged": mark_merged,
                    "merge_lists": merge_lists,
                    "dry_run": dry_run,
                    "confirm": confirm,
                },
            )
            if not canonical_id:
                raise ValueError("canonical_id is required.")
            if not duplicate_ids:
                raise ValueError("duplicate_ids cannot be empty.")
            if canonical_id in duplicate_ids:
                raise ValueError("canonical_id cannot be in duplicate_ids.")

            all_ids = [canonical_id, *duplicate_ids]
            enforce_point_ids(all_ids, name="point_ids")

            if not dry_run and not confirm:
                state.warnings.append("confirm=true required to merge duplicates.")
                response = {
                    "dry_run": True,
                    "updated_canonical": False,
                    "marked_duplicates": 0,
                    "deleted_duplicates": 0,
                }
                return finish_request(state, response)
            if not dry_run:
                ensure_mutations_allowed()

            dry_run_diff = init_dry_run_diff() if dry_run else None
            if isinstance(ctx, JobContext):
                ctx.set_phase("merging")
                ctx.set_total(len(all_ids))

            collection = resolve_collection_name(collection_name)
            records = await self.qdrant_connector.retrieve_points(
                all_ids, collection_name=collection, with_payload=True
            )
            record_map = {str(record.id): record for record in records}
            missing_ids = [pid for pid in all_ids if str(pid) not in record_map]
            if canonical_id not in record_map:
                raise ValueError(f"Canonical point {canonical_id} not found.")

            canonical_record = record_map[canonical_id]
            canonical_payload = canonical_record.payload or {}
            canonical_metadata = canonical_payload.get(METADATA_PATH) or {}
            if not isinstance(canonical_metadata, dict):
                canonical_metadata = {}

            merged_metadata = dict(canonical_metadata)
            if merge_lists:
                for list_key in ("entities", "labels", "merged_from"):
                    incoming: list[Any] = []
                    for dup_id in duplicate_ids:
                        record = record_map.get(dup_id)
                        if record is None:
                            continue
                        metadata = (record.payload or {}).get(METADATA_PATH) or {}
                        if not isinstance(metadata, dict):
                            continue
                        value = metadata.get(list_key)
                        if isinstance(value, list):
                            incoming.extend(value)
                    if list_key == "merged_from":
                        incoming.extend(duplicate_ids)
                    if incoming:
                        existing_list = (
                            merged_metadata.get(list_key)
                            if isinstance(merged_metadata.get(list_key), list)
                            else None
                        )
                        merged_metadata[list_key] = merge_list_values(
                            existing_list, incoming
                        )
                if "merged_from" not in merged_metadata:
                    merged_metadata["merged_from"] = duplicate_ids
            else:
                merged_metadata["merged_from"] = duplicate_ids

            reinforcement_total = 0
            for record_id in all_ids:
                record = record_map.get(record_id)
                if record is None:
                    continue
                metadata = (record.payload or {}).get(METADATA_PATH) or {}
                if not isinstance(metadata, dict):
                    continue
                count = coerce_int(metadata.get("reinforcement_count"))
                reinforcement_total += count if count and count > 0 else 1
            merged_metadata["reinforcement_count"] = max(reinforcement_total, 1)

            now = datetime.now(timezone.utc)
            now_iso = now.isoformat()
            now_ms = int(now.timestamp() * 1000)
            last_seen_candidates: list[int] = []
            for record_id in all_ids:
                record = record_map.get(record_id)
                if record is None:
                    continue
                metadata = (record.payload or {}).get(METADATA_PATH) or {}
                if not isinstance(metadata, dict):
                    continue
                for key in ("last_seen_at_ts", "updated_at_ts", "created_at_ts"):
                    ts_value = coerce_int(metadata.get(key))
                    if ts_value is not None:
                        last_seen_candidates.append(ts_value)
            last_seen_ts = max(last_seen_candidates) if last_seen_candidates else now_ms
            merged_metadata["last_seen_at_ts"] = last_seen_ts
            merged_metadata["last_seen_at"] = datetime.fromtimestamp(
                last_seen_ts / 1000, tz=timezone.utc
            ).isoformat()
            merged_metadata["updated_at"] = now_iso
            merged_metadata["updated_at_ts"] = now_ms

            updated_canonical = merged_metadata != canonical_metadata
            if updated_canonical and not dry_run:
                new_payload = dict(canonical_payload)
                new_payload[METADATA_PATH] = merged_metadata
                await self.qdrant_connector.overwrite_payload(
                    [canonical_id],
                    new_payload,
                    collection_name=collection,
                )

            marked_duplicates = 0
            if mark_merged and duplicate_ids:
                for dup_id in duplicate_ids:
                    record = record_map.get(dup_id)
                    if record is None:
                        continue
                    dup_payload = record.payload or {}
                    dup_metadata = dup_payload.get(METADATA_PATH) or {}
                    if not isinstance(dup_metadata, dict):
                        dup_metadata = {}
                    dup_metadata["merged_into"] = canonical_id
                    dup_metadata["updated_at"] = now_iso
                    dup_metadata["updated_at_ts"] = now_ms
                    marked_duplicates += 1
                    if not dry_run:
                        new_payload = dict(dup_payload)
                        new_payload[METADATA_PATH] = dup_metadata
                        await self.qdrant_connector.overwrite_payload(
                            [dup_id],
                            new_payload,
                            collection_name=collection,
                        )
                    if dry_run and dry_run_diff is not None:
                        record_dry_run_action(
                            dry_run_diff,
                            "mark",
                            dup_id,
                            extract_metadata(dup_payload),
                            dup_metadata,
                        )

            deleted_duplicates = 0
            if delete_duplicates and duplicate_ids:
                if not dry_run:
                    await self.qdrant_connector.delete_points(
                        duplicate_ids, collection_name=collection
                    )
                deleted_duplicates = len(duplicate_ids)
                if dry_run and dry_run_diff is not None:
                    for dup_id in duplicate_ids:
                        record = record_map.get(dup_id)
                        if record is None:
                            continue
                        record_dry_run_action(
                            dry_run_diff,
                            "delete",
                            dup_id,
                            extract_metadata(record.payload or {}),
                            None,
                        )

            data: dict[str, Any] = {
                "collection_name": collection,
                "canonical_id": canonical_id,
                "updated_canonical": updated_canonical,
                "marked_duplicates": marked_duplicates,
                "deleted_duplicates": deleted_duplicates,
                "dry_run": dry_run,
            }
            if dry_run and dry_run_diff is not None:
                if updated_canonical:
                    record_dry_run_action(
                        dry_run_diff,
                        "merge",
                        canonical_id,
                        canonical_metadata,
                        merged_metadata,
                    )
                data["dry_run_diff"] = dry_run_diff
            if missing_ids:
                data["missing_ids"] = missing_ids
            if isinstance(ctx, JobContext):
                ctx.advance(len(all_ids))
            return finish_request(state, data)

        async def delete_points(
            ctx: Context,
            point_ids: Annotated[
                list[str], Field(description="List of point ids to delete.")
            ],
            collection_name: Annotated[
                str, Field(description="The collection containing the points.")
            ],
            confirm: Annotated[
                bool, Field(description="Confirm deletion (required).")
            ] = False,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "point_ids": point_ids,
                    "collection_name": collection_name,
                    "confirm": confirm,
                },
            )
            if not point_ids:
                raise ValueError("point_ids cannot be empty.")

            if not confirm:
                state.warnings.append("confirm=true required to delete points.")
                collection = resolve_collection_name(collection_name)
                dry_run_diff = init_dry_run_diff()
                records = await self.qdrant_connector.retrieve_points(
                    point_ids,
                    collection_name=collection,
                    with_payload=True,
                )
                found_ids = {str(record.id) for record in records}
                missing_ids = [pid for pid in point_ids if str(pid) not in found_ids]
                for record in records:
                    record_dry_run_action(
                        dry_run_diff,
                        "delete",
                        str(record.id),
                        extract_metadata(record.payload or {}),
                        None,
                    )
                data = {
                    "deleted": 0,
                    "requested": len(point_ids),
                    "dry_run": True,
                    "dry_run_diff": dry_run_diff,
                }
                if missing_ids:
                    data["missing_ids"] = missing_ids
                return finish_request(state, data)

            ensure_mutations_allowed()
            collection = resolve_collection_name(collection_name)
            await self.qdrant_connector.delete_points(
                point_ids, collection_name=collection
            )
            data = {"deleted": len(point_ids), "requested": len(point_ids)}
            return finish_request(state, data)

        async def delete_by_filter(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The collection to delete from.")
            ],
            memory_filter: Annotated[
                MemoryFilterInput | None,
                Field(description="Structured filters for memory fields."),
            ] = None,
            query_filter: ArbitraryFilter | None = None,
            confirm: Annotated[
                bool, Field(description="Confirm deletion (required).")
            ] = False,
            dry_run: Annotated[
                bool, Field(description="Return count without deleting.")
            ] = True,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "collection_name": collection_name,
                    "memory_filter": memory_filter,
                    "query_filter": query_filter,
                    "confirm": confirm,
                    "dry_run": dry_run,
                },
            )

            memory_filter_obj = build_memory_filter(
                memory_filter,
                strict=self.memory_settings.strict_params,
                warnings=state.warnings,
            )
            query_filter_obj = None
            if query_filter:
                if not self.qdrant_settings.allow_arbitrary_filter:
                    if self.memory_settings.strict_params:
                        raise ValueError("query_filter is not allowed.")
                    state.warnings.append("query_filter ignored (not allowed).")
                else:
                    query_filter_obj = models.Filter(**query_filter)

            merged_filter = merge_filters([memory_filter_obj, query_filter_obj])
            if merged_filter is None:
                if self.memory_settings.strict_params:
                    raise ValueError(
                        "delete_by_filter requires a filter in strict mode."
                    )
                state.warnings.append(
                    "No filter provided; operation targets entire collection."
                )
                merged_filter = models.Filter()

            collection = resolve_collection_name(collection_name)
            matched = await self.qdrant_connector.count_points(
                collection_name=collection,
                query_filter=merged_filter,
            )

            if dry_run or not confirm:
                if not confirm:
                    state.warnings.append("confirm=true required to delete points.")
                dry_run_diff = init_dry_run_diff()
                preview_target = matched
                if preview_target > PREVIEW_SCAN_LIMIT:
                    preview_target = PREVIEW_SCAN_LIMIT
                    state.warnings.append(
                        "dry_run preview truncated; refine filter for more detail."
                    )

                scanned = 0
                offset = None
                stop = False
                preview_batch = min(200, self.tool_settings.max_batch_size)
                while True:
                    points, offset = await self.qdrant_connector.scroll_points_page(
                        collection_name=collection,
                        query_filter=merged_filter,
                        limit=preview_batch,
                        with_payload=True,
                        offset=offset,
                    )
                    if not points:
                        break
                    for point in points:
                        scanned += 1
                        record_dry_run_action(
                            dry_run_diff,
                            "delete",
                            str(point.id),
                            extract_metadata(point.payload or {}),
                            None,
                        )
                        if preview_target and scanned >= preview_target:
                            stop = True
                            break
                    if stop or offset is None:
                        break
                data = {
                    "matched": matched,
                    "deleted": 0,
                    "dry_run": True,
                    "dry_run_diff": dry_run_diff,
                    "preview_scanned": scanned,
                    "preview_truncated": scanned < matched,
                }
                return finish_request(state, data)

            ensure_mutations_allowed()
            await self.qdrant_connector.delete_by_filter(
                merged_filter,
                collection_name=collection,
            )
            data = {"matched": matched, "deleted": matched, "dry_run": False}
            return finish_request(state, data)

        async def delete_document(
            ctx: Context,
            doc_id: Annotated[str, Field(description="Document id to delete.")],
            collection_name: Annotated[
                str, Field(description="The collection containing the document.")
            ],
            confirm: Annotated[
                bool, Field(description="Confirm deletion (required).")
            ] = False,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "doc_id": doc_id,
                    "collection_name": collection_name,
                    "confirm": confirm,
                },
            )
            if not doc_id:
                raise ValueError("doc_id cannot be empty.")

            collection = resolve_collection_name(collection_name)
            doc_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key=f"{METADATA_PATH}.doc_id",
                        match=models.MatchValue(value=doc_id),
                    )
                ]
            )
            matched = await self.qdrant_connector.count_points(
                collection_name=collection,
                query_filter=doc_filter,
            )

            if not confirm:
                state.warnings.append("confirm=true required to delete document.")
                dry_run_diff = init_dry_run_diff()
                preview_target = matched
                if preview_target > PREVIEW_SCAN_LIMIT:
                    preview_target = PREVIEW_SCAN_LIMIT
                    state.warnings.append(
                        "dry_run preview truncated; doc_id has many chunks."
                    )
                scanned = 0
                offset = None
                stop = False
                preview_batch = min(200, self.tool_settings.max_batch_size)
                while True:
                    points, offset = await self.qdrant_connector.scroll_points_page(
                        collection_name=collection,
                        query_filter=doc_filter,
                        limit=preview_batch,
                        with_payload=True,
                        offset=offset,
                    )
                    if not points:
                        break
                    for point in points:
                        scanned += 1
                        record_dry_run_action(
                            dry_run_diff,
                            "delete",
                            str(point.id),
                            extract_metadata(point.payload or {}),
                            None,
                        )
                        if preview_target and scanned >= preview_target:
                            stop = True
                            break
                    if stop or offset is None:
                        break
                data = {
                    "doc_id": doc_id,
                    "collection_name": collection,
                    "matched": matched,
                    "deleted": 0,
                    "dry_run": True,
                    "dry_run_diff": dry_run_diff,
                    "preview_scanned": scanned,
                    "preview_truncated": scanned < matched,
                }
                return finish_request(state, data)

            ensure_mutations_allowed()
            await self.qdrant_connector.delete_by_filter(
                doc_filter,
                collection_name=collection,
            )
            data = {
                "doc_id": doc_id,
                "collection_name": collection,
                "matched": matched,
                "deleted": matched,
                "dry_run": False,
            }
            return finish_request(state, data)

        async def list_collections(ctx: Context) -> dict[str, Any]:
            state = new_request(ctx, {})
            collections = await self.qdrant_connector.get_collection_names()
            data = {"collections": collections, "count": len(collections)}
            return finish_request(state, data)

        async def create_collection(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="Name of the collection to create.")
            ],
            vector_name: Annotated[
                str | None,
                Field(
                    description=(
                        "Optional vector name override. Use empty string for an "
                        "unnamed/default vector."
                    )
                ),
            ] = None,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {"collection_name": collection_name, "vector_name": vector_name},
            )
            ensure_mutations_allowed()
            result = await self.qdrant_connector.create_collection(
                collection_name,
                vector_name=vector_name,
            )
            result["indexes_applied_count"] = len(result.get("indexes_applied", []))
            return finish_request(state, result)

        async def collection_exists(
            ctx: Context, collection_name: str = ""
        ) -> dict[str, Any]:
            state = new_request(ctx, {"collection_name": collection_name})
            name = resolve_collection_name(collection_name)
            exists = await self.qdrant_connector.collection_exists(name)
            data = {"collection_name": name, "exists": exists}
            return finish_request(state, data)

        async def collection_info(
            ctx: Context, collection_name: str = ""
        ) -> dict[str, Any]:
            state = new_request(ctx, {"collection_name": collection_name})
            name = resolve_collection_name(collection_name)
            summary = await self.qdrant_connector.get_collection_summary(name)
            summary["collection_name"] = name
            return finish_request(state, summary)

        async def collection_stats(
            ctx: Context, collection_name: str = ""
        ) -> dict[str, Any]:
            state = new_request(ctx, {"collection_name": collection_name})
            name = resolve_collection_name(collection_name)
            info = await self.qdrant_connector.get_collection_info(name)
            data = {
                "collection_name": name,
                "status": str(info.status),
                "optimizer_status": str(info.optimizer_status),
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "segments_count": info.segments_count,
            }
            if info.warnings:
                data["warnings"] = [str(warning) for warning in info.warnings]
            return finish_request(state, data)

        async def collection_vectors(
            ctx: Context, collection_name: str = ""
        ) -> dict[str, Any]:
            state = new_request(ctx, {"collection_name": collection_name})
            name = resolve_collection_name(collection_name)
            vectors = await self.qdrant_connector.get_collection_vectors(name)
            data = {"collection_name": name, "vectors": vectors}
            return finish_request(state, data)

        async def collection_payload_schema(
            ctx: Context, collection_name: str = ""
        ) -> dict[str, Any]:
            state = new_request(ctx, {"collection_name": collection_name})
            name = resolve_collection_name(collection_name)
            schema = await self.qdrant_connector.get_collection_payload_schema(name)
            data = {"collection_name": name, "payload_schema": schema}
            return finish_request(state, data)

        async def optimizer_status(
            ctx: Context, collection_name: str = ""
        ) -> dict[str, Any]:
            state = new_request(ctx, {"collection_name": collection_name})
            name = resolve_collection_name(collection_name)
            info = await self.qdrant_connector.get_collection_info(name)

            vector_indexed = None
            vector_index_coverage = None
            unindexed_vectors_count = None
            if info.indexed_vectors_count is not None:
                if info.points_count and info.points_count > 0:
                    vector_index_coverage = (
                        info.indexed_vectors_count / info.points_count
                    )
                else:
                    vector_index_coverage = 1.0
                unindexed_vectors_count = max(
                    info.points_count - info.indexed_vectors_count, 0
                )
                vector_indexed = info.points_count == info.indexed_vectors_count

            data = {
                "collection_name": name,
                "status": str(info.status),
                "optimizer_status": str(info.optimizer_status),
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "segments_count": info.segments_count,
                "vector_indexed": vector_indexed,
                "vector_index_coverage": vector_index_coverage,
                "unindexed_vectors_count": unindexed_vectors_count,
                "optimizer_config": serialize_model(info.config.optimizer_config),
                "hnsw_config": serialize_model(info.config.hnsw_config),
            }
            if info.warnings:
                data["warnings"] = [str(warning) for warning in info.warnings]
            return finish_request(state, data)

        async def metrics_snapshot(
            ctx: Context, collection_name: str = ""
        ) -> dict[str, Any]:
            state = new_request(ctx, {"collection_name": collection_name})
            name = resolve_collection_name(collection_name)
            summary = await self.qdrant_connector.get_collection_summary(name)

            points_count = summary.get("points_count") or 0
            indexed_vectors_count = summary.get("indexed_vectors_count")
            vector_index_coverage = None
            unindexed_vectors_count = None
            if indexed_vectors_count is not None:
                if points_count > 0:
                    vector_index_coverage = indexed_vectors_count / points_count
                else:
                    vector_index_coverage = 1.0
                unindexed_vectors_count = max(points_count - indexed_vectors_count, 0)

            payload_schema = summary.get("payload_schema") or {}
            payload_fields = (
                sorted(payload_schema.keys())
                if isinstance(payload_schema, dict)
                else []
            )

            data = {
                "collection_name": name,
                "snapshot_at": datetime.now(timezone.utc).isoformat(),
                "status": summary.get("status"),
                "optimizer_status": summary.get("optimizer_status"),
                "points_count": points_count,
                "indexed_vectors_count": indexed_vectors_count,
                "segments_count": summary.get("segments_count"),
                "vector_index_coverage": vector_index_coverage,
                "unindexed_vectors_count": unindexed_vectors_count,
                "vectors": summary.get("vectors"),
                "payload_index_count": len(payload_fields),
                "payload_index_fields": payload_fields,
            }
            if "warnings" in summary:
                data["warnings"] = summary["warnings"]
            if "sparse_vectors" in summary:
                data["sparse_vectors"] = summary["sparse_vectors"]
            return finish_request(state, data)

        async def get_vector_name(
            ctx: Context, collection_name: str = ""
        ) -> dict[str, Any]:
            state = new_request(ctx, {"collection_name": collection_name})
            name = resolve_collection_name(collection_name)
            vector_name = await self.qdrant_connector.resolve_vector_name(name)
            data = {
                "collection_name": name,
                "vector_name": vector_name,
                "label": "(default)" if vector_name is None else vector_name,
            }
            return finish_request(state, data)

        async def update_optimizer_config(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="Collection to update optimizer settings for.")
            ] = "",
            indexing_threshold: Annotated[
                int | None,
                Field(
                    description=(
                        "Indexing threshold (vectors per segment). "
                        "Lower values force indexing sooner."
                    )
                ),
            ] = None,
            max_optimization_threads: Annotated[
                int | None,
                Field(
                    description=(
                        "Maximum optimizer threads. Higher values may increase load."
                    )
                ),
            ] = None,
            dry_run: Annotated[
                bool,
                Field(description="Report planned changes without applying them."),
            ] = True,
            confirm: Annotated[
                bool,
                Field(
                    description=(
                        "Confirm optimizer update when dry_run is false (required)."
                    )
                ),
            ] = False,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "collection_name": collection_name,
                    "indexing_threshold": indexing_threshold,
                    "max_optimization_threads": max_optimization_threads,
                    "dry_run": dry_run,
                    "confirm": confirm,
                },
            )
            if self.qdrant_settings.read_only:
                raise ValueError("Server is read-only; optimizer updates are disabled.")

            name = resolve_collection_name(collection_name)
            info = await self.qdrant_connector.get_collection_info(name)
            current_config = serialize_model(info.config.optimizer_config)

            requested: dict[str, Any] = {}
            if indexing_threshold is not None:
                if indexing_threshold < 0:
                    raise ValueError("indexing_threshold must be >= 0.")
                requested["indexing_threshold"] = indexing_threshold
                if info.points_count and indexing_threshold <= info.points_count:
                    state.warnings.append(
                        "indexing_threshold below points_count may increase load."
                    )
            if max_optimization_threads is not None:
                if max_optimization_threads < 0:
                    raise ValueError("max_optimization_threads must be >= 0.")
                requested["max_optimization_threads"] = max_optimization_threads
                if max_optimization_threads > 1:
                    state.warnings.append(
                        "max_optimization_threads > 1 may increase load."
                    )

            if not requested:
                state.warnings.append("No optimizer config changes requested.")
                data = {
                    "collection_name": name,
                    "dry_run": True,
                    "current_config": current_config,
                    "requested_config": requested,
                }
                return finish_request(state, data)

            if dry_run or not confirm:
                if not confirm:
                    state.warnings.append(
                        "confirm=true required to apply optimizer changes."
                    )
                data = {
                    "collection_name": name,
                    "dry_run": True,
                    "current_config": current_config,
                    "requested_config": requested,
                }
                return finish_request(state, data)

            diff = models.OptimizersConfigDiff(**requested)
            applied = await self.qdrant_connector.update_optimizer_config(
                collection_name=name,
                optimizers_config=diff,
            )
            updated = await self.qdrant_connector.get_collection_info(name)
            data = {
                "collection_name": name,
                "dry_run": False,
                "applied": applied,
                "requested_config": requested,
                "optimizer_status": str(updated.optimizer_status),
                "optimizer_config": serialize_model(updated.config.optimizer_config),
            }
            return finish_request(state, data)

        async def ensure_payload_indexes(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="Collection to ensure payload indexes for.")
            ] = "",
        ) -> dict[str, Any]:
            state = new_request(ctx, {"collection_name": collection_name})
            ensure_mutations_allowed()
            name = resolve_collection_name(collection_name)
            created = await self.qdrant_connector.ensure_payload_indexes(
                collection_name=name,
                indexes=self.payload_indexes,
            )
            data = {
                "collection_name": name,
                "created_indexes": created,
                "created_count": len(created),
            }
            return finish_request(state, data)

        async def backfill_memory_contract(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="Collection to backfill.")
            ] = "",
            batch_size: Annotated[
                int, Field(description="Batch size for scanning.")
            ] = 100,
            max_points: Annotated[
                int | None,
                Field(description="Max points to scan (None for all)."),
            ] = None,
            dry_run: Annotated[
                bool, Field(description="Report changes without writing.")
            ] = True,
            confirm: Annotated[
                bool, Field(description="Confirm writes when dry_run is false.")
            ] = False,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "collection_name": collection_name,
                    "batch_size": batch_size,
                    "max_points": max_points,
                    "dry_run": dry_run,
                    "confirm": confirm,
                },
            )
            if batch_size <= 0:
                raise ValueError("batch_size must be positive.")

            if not dry_run and not confirm:
                state.warnings.append("confirm=true required to apply backfill.")
                data: dict[str, Any] = {
                    "scanned": 0,
                    "updated": 0,
                    "skipped": 0,
                    "dry_run": True,
                }
                return finish_request(state, data)
            if not dry_run:
                ensure_mutations_allowed()

            collection = resolve_collection_name(collection_name)
            scanned = 0
            updated = 0
            skipped = 0
            offset = None
            warning_set: set[str] = set()
            stop = False
            dry_run_diff = init_dry_run_diff() if dry_run else None

            if isinstance(ctx, JobContext):
                ctx.set_phase("scanning")
                ctx.set_total(max_points)

            while True:
                points, offset = await self.qdrant_connector.scroll_points_page(
                    collection_name=collection,
                    limit=batch_size,
                    with_payload=True,
                    offset=offset,
                )
                if not points:
                    break

                for point in points:
                    scanned += 1
                    if max_points is not None and scanned > max_points:
                        stop = True
                        break

                    payload = point.payload or {}
                    metadata = (
                        payload.get(METADATA_PATH) or payload.get("metadata") or {}
                    )
                    text = extract_payload_text(payload)

                    patch, patch_warnings = build_memory_backfill_patch(
                        text=text,
                        metadata=metadata,
                        embedding_info=self.embedding_info,
                        strict=self.memory_settings.strict_params,
                    )
                    warning_set.update(patch_warnings)

                    if not patch:
                        skipped += 1
                        continue

                    updated += 1
                    if dry_run and dry_run_diff is not None:
                        merged_metadata = dict(metadata)
                        merged_metadata.update(patch)
                        record_dry_run_action(
                            dry_run_diff,
                            "backfill",
                            str(point.id),
                            metadata,
                            merged_metadata,
                        )
                    if not dry_run:
                        merged_metadata = dict(metadata)
                        merged_metadata.update(patch)
                        new_payload = dict(payload)
                        new_payload[METADATA_PATH] = merged_metadata
                        await self.qdrant_connector.overwrite_payload(
                            [str(point.id)],
                            new_payload,
                            collection_name=collection,
                        )

                if isinstance(ctx, JobContext):
                    ctx.advance(len(points))
                if stop or offset is None:
                    break

            state.warnings.extend(sorted(warning_set))
            data = {
                "collection_name": collection,
                "scanned": scanned,
                "updated": updated,
                "skipped": skipped,
                "dry_run": dry_run,
                "next_offset": str(offset) if offset is not None and not stop else None,
            }
            if dry_run and dry_run_diff is not None:
                data["dry_run_diff"] = dry_run_diff
            if max_points is not None:
                data["max_points"] = max_points
            return finish_request(state, data)

        async def list_aliases(ctx: Context) -> dict[str, Any]:
            state = new_request(ctx, {})
            aliases = await self.qdrant_connector.list_aliases()
            data: list[dict[str, str]] = [
                {
                    "alias_name": alias.alias_name,
                    "collection_name": alias.collection_name,
                }
                for alias in aliases
            ]
            return finish_request(state, {"aliases": data, "count": len(data)})

        async def collection_aliases(
            ctx: Context, collection_name: str = ""
        ) -> dict[str, Any]:
            state = new_request(ctx, {"collection_name": collection_name})
            name = resolve_collection_name(collection_name)
            aliases = await self.qdrant_connector.list_collection_aliases(name)
            data: list[dict[str, str]] = [
                {
                    "alias_name": alias.alias_name,
                    "collection_name": alias.collection_name,
                }
                for alias in aliases
            ]
            return finish_request(
                state,
                {"collection_name": name, "aliases": data, "count": len(data)},
            )

        async def collection_cluster_info(
            ctx: Context, collection_name: str = ""
        ) -> dict[str, Any]:
            state = new_request(ctx, {"collection_name": collection_name})
            name = resolve_collection_name(collection_name)
            info = await self.qdrant_connector.get_collection_cluster_info(name)
            data = info.model_dump() if hasattr(info, "model_dump") else info.dict()
            data["collection_name"] = name
            return finish_request(state, data)

        async def list_snapshots(
            ctx: Context, collection_name: str = ""
        ) -> dict[str, Any]:
            state = new_request(ctx, {"collection_name": collection_name})
            name = resolve_collection_name(collection_name)
            snapshots = await self.qdrant_connector.list_snapshots(name)
            data = [
                {
                    "name": snap.name,
                    "creation_time": str(snap.creation_time),
                    "size": snap.size,
                    "checksum": snap.checksum,
                }
                for snap in snapshots
            ]
            return finish_request(
                state, {"collection_name": name, "snapshots": data, "count": len(data)}
            )

        async def list_full_snapshots(ctx: Context) -> dict[str, Any]:
            state = new_request(ctx, {})
            snapshots = await self.qdrant_connector.list_full_snapshots()
            data = [
                {
                    "name": snap.name,
                    "creation_time": str(snap.creation_time),
                    "size": snap.size,
                    "checksum": snap.checksum,
                }
                for snap in snapshots
            ]
            return finish_request(state, {"snapshots": data, "count": len(data)})

        async def create_snapshot(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="Collection to snapshot.")
            ] = "",
            confirm: Annotated[
                bool, Field(description="Confirm snapshot creation.")
            ] = False,
        ) -> dict[str, Any]:
            state = new_request(
                ctx, {"collection_name": collection_name, "confirm": confirm}
            )
            ensure_mutations_allowed()
            if not self.tool_settings.admin_tools_enabled:
                raise ValueError("Snapshot creation requires admin access.")
            if not confirm:
                state.warnings.append("confirm=true required to create snapshot.")
                data = {"collection_name": collection_name, "dry_run": True}
                return finish_request(state, data)

            name = resolve_collection_name(collection_name)
            snapshot = await self.qdrant_connector.create_snapshot(name)
            data = {
                "collection_name": name,
                "snapshot": serialize_model(snapshot),
            }
            return finish_request(state, data)

        async def restore_snapshot(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="Collection to restore.")
            ] = "",
            snapshot_location: Annotated[
                str, Field(description="Snapshot URL/path to restore from.")
            ] = "",
            snapshot_checksum: Annotated[
                str | None, Field(description="Optional snapshot checksum.")
            ] = None,
            api_key: Annotated[
                str | None, Field(description="Optional API key for snapshot URL.")
            ] = None,
            wait: Annotated[
                bool, Field(description="Wait for restore to complete.")
            ] = True,
            priority: Annotated[
                str | None, Field(description="Optional restore priority.")
            ] = None,
            confirm: Annotated[
                bool, Field(description="Confirm snapshot restore.")
            ] = False,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "collection_name": collection_name,
                    "snapshot_location": snapshot_location,
                    "snapshot_checksum": snapshot_checksum,
                    "api_key": api_key,
                    "wait": wait,
                    "priority": priority,
                    "confirm": confirm,
                },
            )
            ensure_mutations_allowed()
            if not self.tool_settings.admin_tools_enabled:
                raise ValueError("Snapshot restore requires admin access.")
            if self.qdrant_settings.read_only:
                raise ValueError("Server is read-only; snapshot restore is disabled.")
            if not snapshot_location:
                raise ValueError("snapshot_location is required.")
            if not confirm:
                state.warnings.append("confirm=true required to restore snapshot.")
                data = {"collection_name": collection_name, "dry_run": True}
                return finish_request(state, data)

            name = resolve_collection_name(collection_name)
            result = await self.qdrant_connector.recover_snapshot(
                name,
                location=snapshot_location,
                api_key=api_key,
                checksum=snapshot_checksum,
                wait=wait,
                priority=priority,
            )
            data = {
                "collection_name": name,
                "result": serialize_model(result),
            }
            return finish_request(state, data)

        async def list_shard_snapshots(
            ctx: Context, shard_id: int, collection_name: str = ""
        ) -> dict[str, Any]:
            state = new_request(
                ctx, {"shard_id": shard_id, "collection_name": collection_name}
            )
            name = resolve_collection_name(collection_name)
            if shard_id < 0:
                raise ValueError("shard_id must be a non-negative integer")
            snapshots = await self.qdrant_connector.list_shard_snapshots(name, shard_id)
            data = [
                {
                    "name": snap.name,
                    "creation_time": str(snap.creation_time),
                    "size": snap.size,
                    "checksum": snap.checksum,
                }
                for snap in snapshots
            ]
            return finish_request(
                state,
                {
                    "collection_name": name,
                    "shard_id": shard_id,
                    "snapshots": data,
                    "count": len(data),
                },
            )

        job_store_dir = Path(self.memory_settings.textbook_job_state_dir)
        job_store_enabled = True
        try:
            job_store_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            job_store_enabled = False
            logger.warning(
                "Failed to initialize textbook job store directory '%s': %s",
                job_store_dir,
                exc,
            )

        job_state_retention_delta = timedelta(
            hours=self.memory_settings.textbook_job_state_retention_hours
        )

        def parse_iso_datetime(value: Any) -> datetime | None:
            if not value:
                return None
            try:
                parsed = datetime.fromisoformat(str(value))
            except ValueError:
                return None
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)

        def is_terminal_status(status: Any) -> bool:
            return str(status) in {"completed", "failed", "cancelled"}

        def job_record_path(job_id: str) -> Path:
            return job_store_dir / f"{job_id}.json"

        def persist_job_record(record: dict[str, Any]) -> None:
            if not job_store_enabled:
                return
            job_id = str(record.get("job_id") or "").strip()
            if not job_id:
                return
            path = job_record_path(job_id)
            tmp_path = path.with_suffix(".json.tmp")
            try:
                payload = json.dumps(record, default=str, ensure_ascii=False)
                tmp_path.write_text(payload, encoding="utf-8")
                tmp_path.replace(path)
            except Exception as exc:
                logger.warning("Failed to persist job record '%s': %s", job_id, exc)
                tmp_path.unlink(missing_ok=True)

        def prune_job_store() -> None:
            if not job_store_enabled:
                return
            now = datetime.now(timezone.utc)
            for path in job_store_dir.glob("*.json"):
                try:
                    parsed = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue
                if not isinstance(parsed, dict):
                    continue
                if not is_terminal_status(parsed.get("status")):
                    continue
                finished_at = parse_iso_datetime(parsed.get("finished_at"))
                created_at = parse_iso_datetime(parsed.get("created_at"))
                reference_time = finished_at or created_at
                if reference_time is None:
                    continue
                if now - reference_time > job_state_retention_delta:
                    path.unlink(missing_ok=True)

        def init_job_progress(record: dict[str, Any]) -> dict[str, Any]:
            progress = record.get("progress")
            if isinstance(progress, dict):
                return progress
            progress = {
                "phase": "queued",
                "items_done": 0,
                "items_total": None,
                "percent": None,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            record["progress"] = progress
            return progress

        def update_job_progress(
            record: dict[str, Any],
            *,
            phase: str | None = None,
            items_total: int | None = None,
            items_done: int | None = None,
            advance: int | None = None,
        ) -> None:
            progress = init_job_progress(record)
            if phase is not None:
                progress["phase"] = phase
            if items_total is not None:
                progress["items_total"] = items_total
            if items_done is not None:
                progress["items_done"] = items_done
            if advance is not None:
                current = progress.get("items_done") or 0
                progress["items_done"] = current + advance
            total = progress.get("items_total")
            done = progress.get("items_done")
            if total and done is not None and total > 0:
                progress["percent"] = round(min(done / total, 1.0) * 100, 2)
            else:
                progress["percent"] = None
            progress["updated_at"] = datetime.now(timezone.utc).isoformat()
            persist_job_record(record)

        def init_job_metrics(record: dict[str, Any]) -> dict[str, Any]:
            metrics = record.get("metrics")
            if isinstance(metrics, dict):
                return metrics
            metrics = {
                "bytes_downloaded": 0,
                "pages_extracted": 0,
                "chars_extracted": 0,
                "chunks_total": 0,
                "chunks_embedded": 0,
                "chunks_upserted": 0,
                "stages": {},
            }
            record["metrics"] = metrics
            return metrics

        def set_job_metric(record: dict[str, Any], key: str, value: Any) -> None:
            metrics = init_job_metrics(record)
            metrics[key] = value
            persist_job_record(record)

        def increment_job_metric(
            record: dict[str, Any], key: str, amount: int | float
        ) -> None:
            metrics = init_job_metrics(record)
            current = metrics.get(key)
            if not isinstance(current, (int, float)):
                current = 0
            metrics[key] = current + amount
            persist_job_record(record)

        def set_job_stage_duration(
            record: dict[str, Any], stage: str, duration_ms: int
        ) -> None:
            metrics = init_job_metrics(record)
            stages = metrics.get("stages")
            if not isinstance(stages, dict):
                stages = {}
                metrics["stages"] = stages
            stages[f"{stage}_ms"] = duration_ms
            persist_job_record(record)

        def append_job_log(
            record: dict[str, Any],
            message: str,
            *,
            stage: str | None = None,
            event: str = "log",
        ) -> None:
            logs = record.setdefault("logs", [])
            if not isinstance(logs, list):
                logs = []
                record["logs"] = logs
            logs.append(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "job_id": record.get("job_id"),
                    "stage": stage,
                    "event": event,
                    "message": str(message),
                }
            )
            if len(logs) > JOB_LOG_LIMIT:
                del logs[: len(logs) - JOB_LOG_LIMIT]
            persist_job_record(record)

        def hydrate_jobs_from_store() -> None:
            if not job_store_enabled:
                return
            now_iso = datetime.now(timezone.utc).isoformat()
            prune_job_store()
            for path in job_store_dir.glob("*.json"):
                try:
                    parsed = json.loads(path.read_text(encoding="utf-8"))
                except Exception as exc:
                    logger.warning(
                        "Failed to read persisted job file '%s': %s", path, exc
                    )
                    continue
                if not isinstance(parsed, dict):
                    continue
                job_id = str(parsed.get("job_id") or "").strip()
                if not job_id:
                    continue
                parsed["job_id"] = job_id
                init_job_progress(parsed)
                init_job_metrics(parsed)
                if str(parsed.get("status")) in {"queued", "running"}:
                    progress = parsed.get("progress") or {}
                    stage = (
                        str(progress.get("phase"))
                        if isinstance(progress, dict) and progress.get("phase")
                        else "runtime"
                    )
                    message = "Job was interrupted by server restart before completion."
                    parsed["status"] = "failed"
                    parsed["error"] = message
                    parsed["structured_error"] = build_structured_error_payload(
                        error_code="server_restarted",
                        suggested_http_status=422,
                        stage=stage,
                        message=message,
                    )
                    parsed["finished_at"] = now_iso
                    update_job_progress(parsed, phase="failed")
                    append_job_log(
                        parsed,
                        "job marked failed after server restart",
                        stage=stage,
                        event="server_restarted",
                    )
                self._jobs[job_id] = parsed
                persist_job_record(parsed)

        hydrate_jobs_from_store()

        class JobContext:
            def __init__(self, request_id: str, record: dict[str, Any]):
                self.request_id = request_id
                self._record = record

            async def debug(self, _message: str) -> None:
                return None

            def log(
                self, message: str, *, stage: str | None = None, event: str = "log"
            ) -> None:
                append_job_log(self._record, message, stage=stage, event=event)

            def set_phase(self, phase: str) -> None:
                update_job_progress(self._record, phase=phase)

            def set_total(self, total: int | None) -> None:
                update_job_progress(self._record, items_total=total)

            def advance(self, count: int) -> None:
                update_job_progress(self._record, advance=count)

            def set_metric(self, key: str, value: Any) -> None:
                set_job_metric(self._record, key, value)

            def increment_metric(self, key: str, amount: int | float) -> None:
                increment_job_metric(self._record, key, amount)

            def set_stage_duration(self, stage: str, duration_ms: int) -> None:
                set_job_stage_duration(self._record, stage, duration_ms)

        async def ingest_textbook_job(
            ctx: JobContext,
            *,
            collection_name: str,
            source_url: str,
            source_url_headers: dict[str, str] | None = None,
            metadata: Metadata | None = None,
            chapter_map: list[dict[str, Any]] | None = None,
            chunk_size: int | None = None,
            chunk_overlap: int | None = None,
            return_chunk_ids: bool = False,
            ocr: bool = True,
        ) -> dict[str, Any]:
            stage = "validate_input"
            warnings: list[str] = []
            chunk_ids: list[str] = [] if return_chunk_ids else []
            replaced_existing = False
            existing_doc_count = 0
            existing_fingerprint_count = 0
            downloaded_pdf_path: Path | None = None
            doc_hash = ""
            ingest_fingerprint = ""
            chunk_count = 0
            parent_text_hash = ""
            inferred_file_name = "textbook.pdf"
            base_metadata: dict[str, Any] = {}
            detected_markers: list[dict[str, Any]] = []

            def start_stage(next_stage: str) -> float:
                nonlocal stage
                stage = next_stage
                ctx.set_phase(next_stage)
                ctx.log(
                    f"stage started: {next_stage}",
                    stage=next_stage,
                    event="stage_start",
                )
                return time.perf_counter()

            def end_stage(current_stage: str, start_time: float) -> None:
                duration_ms = int((time.perf_counter() - start_time) * 1000)
                ctx.set_stage_duration(current_stage, duration_ms)
                ctx.log(
                    f"stage completed: {current_stage} ({duration_ms}ms)",
                    stage=current_stage,
                    event="stage_complete",
                )

            try:
                async with asyncio.timeout(
                    self.memory_settings.textbook_job_timeout_seconds
                ):
                    stage_start = start_stage("validate_input")
                    try:
                        normalized_metadata = normalize_textbook_metadata(metadata)
                        try:
                            resolved_chapter_map = normalize_chapter_map(chapter_map)
                        except ValueError as exc:
                            raise StructuredIngestError(
                                error_code="textbook_validation_error",
                                suggested_http_status=422,
                                stage="validate_input",
                                message=str(exc),
                            ) from exc

                        parsed = urlparse(source_url or "")
                        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                            raise StructuredIngestError(
                                error_code="textbook_validation_error",
                                suggested_http_status=422,
                                stage="validate_input",
                                message="source_url must be a valid http(s) URL.",
                            )

                        if chunk_size is None:
                            resolved_chunk_size = min(
                                self.memory_settings.max_text_length, 1500
                            )
                        else:
                            resolved_chunk_size = chunk_size
                        if resolved_chunk_size <= 0:
                            raise StructuredIngestError(
                                error_code="textbook_validation_error",
                                suggested_http_status=422,
                                stage="validate_input",
                                message="chunk_size must be positive.",
                            )

                        resolved_overlap = (
                            200 if chunk_overlap is None else chunk_overlap
                        )
                        if resolved_overlap < 0:
                            raise StructuredIngestError(
                                error_code="textbook_validation_error",
                                suggested_http_status=422,
                                stage="validate_input",
                                message="chunk_overlap must be >= 0.",
                            )
                        if resolved_overlap >= resolved_chunk_size:
                            resolved_overlap = max(0, resolved_chunk_size - 1)
                            warnings.append(
                                "chunk_overlap reduced to chunk_size - 1 for textbook ingest."
                            )

                        resolved_collection = resolve_collection_name(collection_name)
                        resolved_doc_id = build_textbook_doc_id(normalized_metadata)
                    finally:
                        end_stage("validate_input", stage_start)

                    stage_start = start_stage("download")
                    try:
                        request_headers = {
                            "User-Agent": "Mozilla/5.0 (compatible; mcp-server-qdrant)",
                            "Accept": "*/*",
                            "Accept-Language": "en-US,en;q=0.9",
                        }
                        if source_url_headers:
                            request_headers.update(
                                {
                                    str(key): str(value)
                                    for key, value in source_url_headers.items()
                                }
                            )
                        (
                            downloaded_pdf_path,
                            fetched_mime,
                            bytes_downloaded,
                            doc_hash,
                        ) = await fetch_url_to_tempfile_streaming(
                            source_url,
                            headers=request_headers,
                            max_bytes=self.memory_settings.textbook_max_file_bytes,
                            timeout_seconds=min(
                                120, self.memory_settings.textbook_job_timeout_seconds
                            ),
                            suffix=".pdf",
                        )
                        ctx.set_metric("bytes_downloaded", bytes_downloaded)
                        inferred_file_name = (
                            Path(urlparse(source_url).path).name or "textbook.pdf"
                        )
                        resolved_file_type = normalize_file_type(
                            file_type=None,
                            file_name=inferred_file_name,
                            mime_type=fetched_mime,
                            has_text=False,
                        )
                        if resolved_file_type != "pdf":
                            raise StructuredIngestError(
                                error_code="textbook_validation_error",
                                suggested_http_status=422,
                                stage="download",
                                message="qdrant-ingest-textbook currently supports PDF source_url inputs only.",
                            )
                    finally:
                        end_stage("download", stage_start)

                    stage_start = start_stage("extract")
                    try:
                        extraction_result = await asyncio.to_thread(
                            extract_document_sections,
                            "pdf",
                            text=None,
                            data=None,
                            file_path=downloaded_pdf_path,
                            ocr=ocr,
                            ocr_low_text_threshold_chars=self.memory_settings.textbook_ocr_low_text_threshold_chars,
                            ocr_min_coverage_ratio=self.memory_settings.textbook_ocr_min_coverage_ratio,
                            ocr_max_pages=self.memory_settings.textbook_ocr_max_pages,
                            ocr_max_page_ratio=self.memory_settings.textbook_ocr_max_page_ratio,
                        )
                        warnings.extend(extraction_result.warnings)
                        page_count = extraction_result.page_count or 0
                        ctx.set_metric("pages_extracted", page_count)
                        ctx.set_metric(
                            "coverage_ratio",
                            extraction_result.coverage_ratio,
                        )
                        ctx.set_metric(
                            "coverage_pages_total",
                            extraction_result.coverage_pages_total,
                        )
                        ctx.set_metric(
                            "coverage_pages_meeting_threshold",
                            extraction_result.coverage_pages_meeting_threshold,
                        )
                        ctx.set_metric(
                            "ocr_attempted_pages",
                            extraction_result.ocr_attempted_pages,
                        )
                        ctx.set_metric(
                            "ocr_budget_pages",
                            extraction_result.ocr_budget_pages,
                        )
                        if page_count > self.memory_settings.textbook_max_pages:
                            raise_limit_error(
                                stage="extract",
                                limit_name="textbook_max_pages",
                                limit_value=self.memory_settings.textbook_max_pages,
                                actual_value=page_count,
                            )
                        if (
                            ocr
                            and page_count > 0
                            and extraction_result.coverage_ratio is not None
                            and extraction_result.coverage_ratio
                            < self.memory_settings.textbook_ocr_min_coverage_ratio
                        ):
                            raise StructuredIngestError(
                                error_code="textbook_ocr_coverage_unmet",
                                suggested_http_status=422,
                                stage="extract",
                                message=(
                                    "OCR coverage target not met within OCR budget. "
                                    f"coverage={extraction_result.coverage_ratio:.3f}, "
                                    f"target={self.memory_settings.textbook_ocr_min_coverage_ratio:.3f}, "
                                    f"budget={extraction_result.ocr_budget_pages}, "
                                    f"attempted={extraction_result.ocr_attempted_pages}. "
                                    "Retry with chapter-scoped ingest or a smaller source file."
                                ),
                            )
                        if not extraction_result.sections:
                            raise StructuredIngestError(
                                error_code="textbook_validation_error",
                                suggested_http_status=422,
                                stage="extract",
                                message="No text could be extracted from the PDF.",
                            )
                    finally:
                        end_stage("extract", stage_start)

                    stage_start = start_stage("chunk")
                    try:
                        detected_markers = detect_pdf_chapter_markers(
                            extraction_result.sections
                        )
                        extracted_chars = 0
                        parent_text_hasher = hashlib.sha256()
                        parent_hash_has_content = False
                        chunk_count = 0

                        for section in extraction_result.sections:
                            normalized_text = normalize_text_for_chunking(section.text)
                            if not normalized_text:
                                continue
                            extracted_chars += len(normalized_text)
                            section_chunks = chunk_text_with_overlap(
                                normalized_text,
                                resolved_chunk_size,
                                resolved_overlap,
                            )
                            for chunk in section_chunks:
                                if not chunk:
                                    continue
                                if parent_hash_has_content:
                                    parent_text_hasher.update(b"\n\n")
                                parent_text_hasher.update(chunk.encode("utf-8"))
                                parent_hash_has_content = True
                                chunk_count += 1

                        ctx.set_metric("chars_extracted", extracted_chars)
                        if (
                            extracted_chars
                            > self.memory_settings.textbook_max_extracted_chars
                        ):
                            raise_limit_error(
                                stage="chunk",
                                limit_name="textbook_max_extracted_chars",
                                limit_value=self.memory_settings.textbook_max_extracted_chars,
                                actual_value=extracted_chars,
                            )

                        ctx.set_metric("chunks_total", chunk_count)
                        if chunk_count == 0:
                            raise StructuredIngestError(
                                error_code="textbook_validation_error",
                                suggested_http_status=422,
                                stage="chunk",
                                message="No chunks were produced from extracted textbook text.",
                            )
                        if chunk_count > self.memory_settings.textbook_max_chunks:
                            raise_limit_error(
                                stage="chunk",
                                limit_name="textbook_max_chunks",
                                limit_value=self.memory_settings.textbook_max_chunks,
                                actual_value=chunk_count,
                            )
                        parent_text_hash = parent_text_hasher.hexdigest()
                    finally:
                        end_stage("chunk", stage_start)

                    stage_start = start_stage("embed")
                    try:
                        ingest_fingerprint = build_textbook_ingest_fingerprint(
                            source_url=source_url,
                            doc_hash=doc_hash,
                            metadata=normalized_metadata,
                        )

                        base_metadata = dict(normalized_metadata)
                        base_metadata.setdefault("type", "document")
                        base_metadata.setdefault("source", "document")
                        base_metadata.setdefault("scope", resolved_doc_id)
                        base_metadata.setdefault("entities", [])
                        base_metadata.setdefault("confidence", 0.5)
                        base_metadata["doc_id"] = resolved_doc_id
                        base_metadata["doc_title"] = normalized_metadata["title"]
                        base_metadata["doc_hash"] = doc_hash
                        base_metadata["file_type"] = "pdf"
                        base_metadata["source_url"] = source_url
                        base_metadata["file_name"] = inferred_file_name
                        base_metadata["ingest_fingerprint"] = ingest_fingerprint
                        ctx.set_metric("chunks_total", chunk_count)
                    finally:
                        end_stage("embed", stage_start)

                    stage_start = start_stage("upsert")
                    try:
                        await self.qdrant_connector.ensure_collection_exists(
                            resolved_collection
                        )

                        doc_id_key = f"{METADATA_PATH}.doc_id"
                        fingerprint_key = f"{METADATA_PATH}.ingest_fingerprint"
                        for key_name in (doc_id_key, fingerprint_key):
                            try:
                                await self.qdrant_connector.ensure_payload_indexes(
                                    collection_name=resolved_collection,
                                    indexes={
                                        key_name: models.PayloadSchemaType.KEYWORD
                                    },
                                )
                            except Exception as exc:
                                warnings.append(
                                    f"Unable to ensure payload index for {key_name}: {exc}"
                                )

                        doc_filter = models.Filter(
                            must=[
                                models.FieldCondition(
                                    key=doc_id_key,
                                    match=models.MatchValue(value=resolved_doc_id),
                                )
                            ]
                        )
                        fingerprint_filter = models.Filter(
                            must=[
                                models.FieldCondition(
                                    key=doc_id_key,
                                    match=models.MatchValue(value=resolved_doc_id),
                                ),
                                models.FieldCondition(
                                    key=fingerprint_key,
                                    match=models.MatchValue(value=ingest_fingerprint),
                                ),
                            ]
                        )
                        existing_doc_count = await self.qdrant_connector.count_points(
                            collection_name=resolved_collection,
                            query_filter=doc_filter,
                        )
                        existing_fingerprint_count = (
                            await self.qdrant_connector.count_points(
                                collection_name=resolved_collection,
                                query_filter=fingerprint_filter,
                            )
                        )

                        if existing_fingerprint_count >= chunk_count:
                            ctx.set_metric("chunks_upserted", 0)
                            return {
                                "status": "already_ingested",
                                "collection_name": resolved_collection,
                                "doc_id": resolved_doc_id,
                                "doc_hash": doc_hash,
                                "ingest_fingerprint": ingest_fingerprint,
                                "source_url": source_url,
                                "existing_count": existing_doc_count,
                                "chunks_count": chunk_count,
                                "replaced_existing": False,
                                "warnings": sorted(set(warnings)),
                            }

                        if existing_doc_count > 0:
                            await self.qdrant_connector.delete_by_filter(
                                doc_filter, collection_name=resolved_collection
                            )
                            replaced_existing = True

                        vector_name = await self.qdrant_connector.resolve_vector_name(
                            resolved_collection
                        )

                        ctx.set_total(chunk_count)
                        semaphore = asyncio.Semaphore(
                            max(1, self.memory_settings.textbook_max_concurrency)
                        )
                        max_inflight = max(
                            1, self.memory_settings.textbook_max_concurrency
                        )
                        effective_chunk_total = chunk_count

                        async def process_embed_batch(
                            batch_entries: list[Entry],
                        ) -> list[str]:
                            async with semaphore:
                                texts = [entry.content for entry in batch_entries]
                                embeddings = (
                                    await self.embedding_provider.embed_documents(texts)
                                )
                                points: list[models.PointStruct] = []
                                point_ids: list[str] = [] if return_chunk_ids else []
                                for entry, embedding in zip(batch_entries, embeddings):
                                    point_id = uuid.uuid4().hex
                                    if return_chunk_ids:
                                        point_ids.append(point_id)
                                    payload = {
                                        "document": entry.content,
                                        METADATA_PATH: entry.metadata,
                                    }
                                    if vector_name is None:
                                        vector_payload: (
                                            list[float] | dict[str, list[float]]
                                        ) = embedding
                                    else:
                                        vector_payload = {vector_name: embedding}
                                    points.append(
                                        models.PointStruct(
                                            id=point_id,
                                            vector=vector_payload,
                                            payload=payload,
                                        )
                                    )
                                ctx.increment_metric("chunks_embedded", len(points))
                                for point_batch in chunks_of(
                                    points,
                                    self.memory_settings.textbook_upsert_batch_size,
                                ):
                                    await self.qdrant_connector.upsert_points(
                                        point_batch,
                                        collection_name=resolved_collection,
                                    )
                                    ctx.increment_metric(
                                        "chunks_upserted", len(point_batch)
                                    )
                                    ctx.advance(len(point_batch))
                                return point_ids

                        inflight: set[asyncio.Task[list[str]]] = set()
                        pending_entries: list[Entry] = []
                        chunk_index = 0
                        normalized_chunk_total = 0

                        async def drain_inflight(*, wait_all: bool) -> None:
                            nonlocal inflight
                            if not inflight:
                                return
                            done, pending = await asyncio.wait(
                                inflight,
                                return_when=(
                                    asyncio.ALL_COMPLETED
                                    if wait_all
                                    else asyncio.FIRST_COMPLETED
                                ),
                            )
                            inflight = set(pending)
                            for task in done:
                                point_ids = await task
                                if return_chunk_ids:
                                    chunk_ids.extend(point_ids)

                        async def submit_pending_entries() -> None:
                            nonlocal pending_entries
                            if not pending_entries:
                                return
                            batch = pending_entries
                            pending_entries = []
                            inflight.add(
                                asyncio.create_task(process_embed_batch(batch))
                            )
                            if len(inflight) >= max_inflight:
                                await drain_inflight(wait_all=False)

                        for section in extraction_result.sections:
                            normalized_text = normalize_text_for_chunking(section.text)
                            if not normalized_text:
                                continue
                            chapter_num, chapter_title = (
                                resolve_chapter_metadata_for_page(
                                    section.page_start,
                                    chapter_map=resolved_chapter_map,
                                    detected_markers=detected_markers,
                                )
                            )
                            section_chunks = chunk_text_with_overlap(
                                normalized_text,
                                resolved_chunk_size,
                                resolved_overlap,
                            )
                            for chunk in section_chunks:
                                if not chunk:
                                    continue
                                chunk_metadata = dict(base_metadata)
                                chunk_metadata["chunk_index"] = chunk_index
                                chunk_metadata["chunk_count"] = chunk_count
                                chunk_metadata["parent_text_hash"] = parent_text_hash
                                if section.page_start is not None:
                                    chunk_metadata["page_start"] = section.page_start
                                if section.page_end is not None:
                                    chunk_metadata["page_end"] = section.page_end
                                if section.section_heading:
                                    chunk_metadata["section_heading"] = (
                                        section.section_heading
                                    )
                                if chapter_num is not None:
                                    chunk_metadata["chapter"] = chapter_num
                                if chapter_title:
                                    chunk_metadata["chapter_title"] = chapter_title

                                records, record_warnings = normalize_memory_input(
                                    information=chunk,
                                    metadata=chunk_metadata,
                                    memory=None,
                                    embedding_info=self.embedding_info,
                                    strict=self.memory_settings.strict_params,
                                    max_text_length=max(
                                        resolved_chunk_size, len(chunk)
                                    ),
                                )
                                warnings.extend(record_warnings)
                                for record in records:
                                    normalized_chunk_total += 1
                                    if (
                                        normalized_chunk_total
                                        > self.memory_settings.textbook_max_chunks
                                    ):
                                        raise_limit_error(
                                            stage="upsert",
                                            limit_name="textbook_max_chunks",
                                            limit_value=self.memory_settings.textbook_max_chunks,
                                            actual_value=normalized_chunk_total,
                                            message=(
                                                "textbook_max_chunks exceeded after normalization "
                                                "and memory-contract chunk expansion."
                                            ),
                                        )
                                    pending_entries.append(
                                        Entry(
                                            content=record.text,
                                            metadata=record.metadata,
                                        )
                                    )
                                if normalized_chunk_total > effective_chunk_total:
                                    effective_chunk_total = normalized_chunk_total
                                    ctx.set_total(effective_chunk_total)
                                    ctx.set_metric(
                                        "chunks_total", effective_chunk_total
                                    )
                                if (
                                    len(pending_entries)
                                    >= self.memory_settings.textbook_embed_batch_size
                                ):
                                    await submit_pending_entries()
                                chunk_index += 1

                        if normalized_chunk_total == 0:
                            raise StructuredIngestError(
                                error_code="textbook_validation_error",
                                suggested_http_status=422,
                                stage="upsert",
                                message="No chunks remained after normalization.",
                            )
                        await submit_pending_entries()
                        await drain_inflight(wait_all=True)
                        if normalized_chunk_total > 0:
                            chunk_count = normalized_chunk_total
                            ctx.set_metric("chunks_total", chunk_count)
                            ctx.set_total(chunk_count)
                    finally:
                        end_stage("upsert", stage_start)

                    stage_start = start_stage("finalize")
                    try:
                        result: dict[str, Any] = {
                            "status": "ingested",
                            "collection_name": resolved_collection,
                            "doc_id": resolved_doc_id,
                            "doc_hash": doc_hash,
                            "ingest_fingerprint": ingest_fingerprint,
                            "source_url": source_url,
                            "chunks_count": chunk_count,
                            "existing_count": existing_doc_count,
                            "existing_fingerprint_count": existing_fingerprint_count,
                            "replaced_existing": replaced_existing,
                            "warnings": sorted(set(warnings)),
                        }
                        if return_chunk_ids:
                            result["chunk_ids"] = chunk_ids
                        return result
                    finally:
                        end_stage("finalize", stage_start)
            except TimeoutError as exc:
                raise StructuredIngestError(
                    error_code="textbook_ingest_timeout",
                    suggested_http_status=422,
                    stage=stage,
                    message=(
                        "Textbook ingest exceeded timeout "
                        f"({self.memory_settings.textbook_job_timeout_seconds}s)."
                    ),
                ) from exc
            finally:
                if downloaded_pdf_path and downloaded_pdf_path.exists():
                    downloaded_pdf_path.unlink(missing_ok=True)

        job_handlers: dict[str, Any] = {
            "audit-memories": audit_memories,
            "backfill-memory-contract": backfill_memory_contract,
            "bulk-patch": bulk_patch,
            "dedupe-memories": dedupe_memories,
            "expire-memories": expire_memories,
            "find-near-duplicates": find_near_duplicates,
            "ingest-textbook": ingest_textbook_job,
            "merge-duplicates": merge_duplicates,
            "reembed-points": reembed_points,
        }

        def submit_background_job(job_key: str, args: dict[str, Any]) -> dict[str, Any]:
            handler = job_handlers.get(job_key)
            if handler is None:
                raise ValueError(f"Unknown job_type '{job_key}'.")

            if "collection_name" not in args:
                default_collection = self._get_default_collection_name()
                if default_collection:
                    args["collection_name"] = default_collection

            job_id = uuid.uuid4().hex
            now = datetime.now(timezone.utc).isoformat()
            record: dict[str, Any] = {
                "job_id": job_id,
                "job_type": job_key,
                "status": "queued",
                "created_at": now,
                "started_at": None,
                "finished_at": None,
                "error": None,
            }
            init_job_progress(record)
            init_job_metrics(record)
            append_job_log(
                record,
                f"queued job_type={job_key}",
                stage="queued",
                event="queued",
            )
            self._jobs[job_id] = record
            persist_job_record(record)

            async def run_job() -> None:
                record["status"] = "running"
                record["started_at"] = datetime.now(timezone.utc).isoformat()
                update_job_progress(record, phase="running")
                append_job_log(
                    record,
                    f"started job_type={job_key}",
                    stage="running",
                    event="started",
                )
                job_ctx = JobContext(job_id, record)
                try:
                    result = await handler(job_ctx, **args)
                    record["status"] = "completed"
                    record["result"] = result
                    update_job_progress(record, phase="completed")
                    append_job_log(
                        record,
                        f"completed job_type={job_key}",
                        stage="completed",
                        event="completed",
                    )
                except StructuredIngestError as exc:
                    record["status"] = "failed"
                    record["error"] = exc.message
                    record["structured_error"] = exc.to_dict()
                    update_job_progress(record, phase="failed")
                    append_job_log(
                        record,
                        f"failed job_type={job_key}: {exc.message}",
                        stage=exc.stage,
                        event="failed",
                    )
                except Exception as exc:  # pragma: no cover - runtime safety
                    record["status"] = "failed"
                    record["error"] = str(exc)
                    record["structured_error"] = build_structured_error_payload(
                        error_code="internal_error",
                        suggested_http_status=422,
                        stage="runtime",
                        message=str(exc),
                    )
                    update_job_progress(record, phase="failed")
                    append_job_log(
                        record,
                        f"failed job_type={job_key}: {exc}",
                        stage="runtime",
                        event="failed",
                    )
                record["finished_at"] = datetime.now(timezone.utc).isoformat()
                persist_job_record(record)
                prune_job_store()

            task = asyncio.create_task(run_job())
            self._job_tasks[job_id] = task
            return {"job_id": job_id, "status": "queued", "job_type": job_key}

        async def submit_job(
            ctx: Context,
            job_type: Annotated[str, Field(description="Job type to run.")],
            job_args: Annotated[
                dict[str, Any] | None,
                Field(description="Arguments for the job."),
            ] = None,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {"job_type": job_type, "job_args": job_args},
            )
            job_key = job_type.strip()
            if job_key.startswith("qdrant-"):
                job_key = job_key[len("qdrant-") :]
            args = dict(job_args or {})
            data = submit_background_job(job_key, args)
            return finish_request(state, data)

        async def ingest_textbook(
            ctx: Context,
            collection_name: Annotated[
                str, Field(description="The collection to store the textbook in.")
            ],
            source_url: Annotated[
                str, Field(description="HTTP(S) URL to the textbook PDF.")
            ],
            metadata: Annotated[
                Metadata | None,
                Field(
                    description=(
                        "Required metadata keys: class, material_type, title, author, "
                        "edition, isbn. Optional: publisher, chapter, chapter_title."
                    )
                ),
            ] = None,
            source_url_headers: Annotated[
                dict[str, str] | None,
                Field(description="Optional request headers used while downloading."),
            ] = None,
            chapter_map: Annotated[
                list[dict[str, Any]] | None,
                Field(
                    description=(
                        "Optional chapter overrides with start_page/end_page/chapter/chapter_title."
                    )
                ),
            ] = None,
            chunk_size: Annotated[
                int | None,
                Field(description="Chunk size in characters (defaults to 1500)."),
            ] = None,
            chunk_overlap: Annotated[
                int | None,
                Field(description="Chunk overlap in characters (defaults to 200)."),
            ] = None,
            return_chunk_ids: Annotated[
                bool,
                Field(description="Include chunk point ids in job result."),
            ] = False,
            ocr: Annotated[
                bool,
                Field(description="Enable OCR for textbook PDF extraction."),
            ] = True,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "collection_name": collection_name,
                    "source_url": source_url,
                    "metadata": metadata,
                    "source_url_headers": (
                        list(source_url_headers.keys()) if source_url_headers else None
                    ),
                    "chapter_map": chapter_map,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "return_chunk_ids": return_chunk_ids,
                    "ocr": ocr,
                },
            )
            ensure_mutations_allowed()

            try:
                normalized_metadata = normalize_textbook_metadata(metadata)
                parsed = urlparse(source_url or "")
                if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                    raise StructuredIngestError(
                        error_code="textbook_validation_error",
                        suggested_http_status=422,
                        stage="validate_input",
                        message="source_url must be a valid http(s) URL.",
                    )
                normalized_chapter_map = normalize_chapter_map(chapter_map)
                if chunk_size is not None and chunk_size <= 0:
                    raise StructuredIngestError(
                        error_code="textbook_validation_error",
                        suggested_http_status=422,
                        stage="validate_input",
                        message="chunk_size must be positive.",
                    )
                if chunk_overlap is not None and chunk_overlap < 0:
                    raise StructuredIngestError(
                        error_code="textbook_validation_error",
                        suggested_http_status=422,
                        stage="validate_input",
                        message="chunk_overlap must be >= 0.",
                    )

                job_args: dict[str, Any] = {
                    "collection_name": collection_name,
                    "source_url": source_url,
                    "source_url_headers": source_url_headers or {},
                    "metadata": normalized_metadata,
                    "chapter_map": normalized_chapter_map,
                    "chunk_size": chunk_size,
                    "chunk_overlap": chunk_overlap,
                    "return_chunk_ids": return_chunk_ids,
                    "ocr": ocr,
                }
                data = submit_background_job("ingest-textbook", job_args)
                data["doc_id"] = build_textbook_doc_id(normalized_metadata)
                return finish_request(state, data)
            except StructuredIngestError as exc:
                data = {"status": "rejected", "error": exc.to_dict()}
                return finish_request(state, data)
            except ValueError as exc:
                structured = build_structured_error_payload(
                    error_code="textbook_validation_error",
                    suggested_http_status=422,
                    stage="validate_input",
                    message=str(exc),
                )
                data = {"status": "rejected", "error": structured}
                return finish_request(state, data)

        async def get_ingest_status(
            ctx: Context,
            job_id: Annotated[str, Field(description="Textbook ingest job id.")],
            include_logs: Annotated[
                bool, Field(description="Include job logs in the response.")
            ] = False,
            logs_tail: Annotated[
                int, Field(description="Max log lines when include_logs=true.")
            ] = 50,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "job_id": job_id,
                    "include_logs": include_logs,
                    "logs_tail": logs_tail,
                },
            )
            record = self._jobs.get(job_id)
            if not record:
                raise ValueError(f"Job {job_id} not found.")
            if record.get("job_type") != "ingest-textbook":
                raise ValueError(
                    f"Job {job_id} is type '{record.get('job_type')}', not ingest-textbook."
                )
            logs = record.get("logs") or []
            if include_logs:
                logs = logs[-max(logs_tail, 0) :] if logs_tail else []
            else:
                logs = []
            data = {
                "job_id": job_id,
                "job_type": record.get("job_type"),
                "status": record.get("status"),
                "created_at": record.get("created_at"),
                "started_at": record.get("started_at"),
                "finished_at": record.get("finished_at"),
                "error": record.get("error"),
                "structured_error": record.get("structured_error"),
                "progress": record.get("progress") or {},
                "metrics": record.get("metrics") or {},
                "result": record.get("result")
                if record.get("status") == "completed"
                else None,
                "logs": logs,
                "server_instance_id": SERVER_INSTANCE_ID,
                "server_uptime_ms": int((time.monotonic() - SERVER_START) * 1000),
            }
            return finish_request(state, data)

        async def cancel_ingest(
            ctx: Context,
            job_id: Annotated[str, Field(description="Textbook ingest job id.")],
        ) -> dict[str, Any]:
            state = new_request(ctx, {"job_id": job_id})
            task = self._job_tasks.get(job_id)
            record = self._jobs.get(job_id)
            if record is None:
                raise ValueError(f"Job {job_id} not found.")
            if record.get("job_type") != "ingest-textbook":
                raise ValueError(
                    f"Job {job_id} is type '{record.get('job_type')}', not ingest-textbook."
                )
            if task is None or task.done():
                data = {
                    "job_id": job_id,
                    "status": record.get("status"),
                    "cancelled": False,
                }
                return finish_request(state, data)
            task.cancel()
            record["status"] = "cancelled"
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            update_job_progress(record, phase="cancelled")
            append_job_log(
                record,
                "cancelled job_type=ingest-textbook",
                stage="cancelled",
                event="cancelled",
            )
            prune_job_store()
            data = {"job_id": job_id, "status": "cancelled", "cancelled": True}
            return finish_request(state, data)

        async def job_status(
            ctx: Context,
            job_id: Annotated[str, Field(description="Job id to inspect.")],
        ) -> dict[str, Any]:
            state = new_request(ctx, {"job_id": job_id})
            record = self._jobs.get(job_id)
            if not record:
                raise ValueError(f"Job {job_id} not found.")
            data = {
                key: value
                for key, value in record.items()
                if key not in {"result", "logs"}
            }
            return finish_request(state, data)

        async def job_progress(
            ctx: Context,
            job_id: Annotated[str, Field(description="Job id to inspect.")],
        ) -> dict[str, Any]:
            state = new_request(ctx, {"job_id": job_id})
            record = self._jobs.get(job_id)
            if not record:
                raise ValueError(f"Job {job_id} not found.")
            progress = record.get("progress") or {}
            data = {
                "job_id": job_id,
                "status": record.get("status"),
                "progress": progress,
            }
            return finish_request(state, data)

        async def job_logs(
            ctx: Context,
            job_id: Annotated[str, Field(description="Job id to inspect.")],
            tail: Annotated[
                int, Field(description="Max log lines to return.")
            ] = JOB_LOG_LIMIT,
        ) -> dict[str, Any]:
            state = new_request(ctx, {"job_id": job_id, "tail": tail})
            record = self._jobs.get(job_id)
            if not record:
                raise ValueError(f"Job {job_id} not found.")
            logs = record.get("logs") or []
            if tail < 0:
                raise ValueError("tail must be non-negative.")
            if tail:
                logs = logs[-tail:]
            else:
                logs = []
            data = {
                "job_id": job_id,
                "status": record.get("status"),
                "logs": logs,
                "count": len(logs),
            }
            return finish_request(state, data)

        async def job_result(
            ctx: Context,
            job_id: Annotated[str, Field(description="Job id to fetch result for.")],
        ) -> dict[str, Any]:
            state = new_request(ctx, {"job_id": job_id})
            record = self._jobs.get(job_id)
            if not record:
                raise ValueError(f"Job {job_id} not found.")
            if record.get("status") != "completed":
                data = {
                    "job_id": job_id,
                    "status": record.get("status"),
                    "error": record.get("error"),
                    "structured_error": record.get("structured_error"),
                }
                return finish_request(state, data)
            data = {
                "job_id": job_id,
                "status": record.get("status"),
                "result": record.get("result"),
            }
            return finish_request(state, data)

        async def cancel_job(
            ctx: Context,
            job_id: Annotated[str, Field(description="Job id to cancel.")],
        ) -> dict[str, Any]:
            state = new_request(ctx, {"job_id": job_id})
            task = self._job_tasks.get(job_id)
            record = self._jobs.get(job_id)
            if record is None:
                raise ValueError(f"Job {job_id} not found.")
            if task is None or task.done():
                data = {
                    "job_id": job_id,
                    "status": record.get("status"),
                    "cancelled": False,
                }
                return finish_request(state, data)
            task.cancel()
            record["status"] = "cancelled"
            record["finished_at"] = datetime.now(timezone.utc).isoformat()
            update_job_progress(record, phase="cancelled")
            append_job_log(record, f"cancelled job_type={record.get('job_type')}")
            data = {"job_id": job_id, "status": "cancelled", "cancelled": True}
            return finish_request(state, data)

        async def check_configuration(ctx: Context) -> dict[str, Any]:
            state = new_request(ctx, {})
            qdrant_configured = bool(
                self.qdrant_settings.location or self.qdrant_settings.local_path
            )
            portal_grant_configured = bool(
                self.request_override_settings.portal_grant_token
            )
            missing: list[str] = []
            data = {
                "service": "qdrant-mcp",
                "portal_grant_configured": portal_grant_configured,
                "request_overrides_enabled": self.request_override_settings.allow_request_overrides,
                "default_qdrant_configured": qdrant_configured,
                "default_collection_configured": bool(
                    self.qdrant_settings.collection_name
                ),
                "embedding_provider": self.embedding_info.provider,
                "embedding_model": self.embedding_info.model,
                "admin_tools_enabled": self.tool_settings.admin_tools_enabled,
                "read_only": self.qdrant_settings.read_only,
                "missing": missing,
                "query_embedding_cache": {
                    "size": self.memory_settings.query_embedding_cache_size,
                    "ttl_seconds": self.memory_settings.query_embedding_cache_ttl_seconds,
                    "entries": len(self._query_embedding_cache),
                },
                "accepted_headers": [
                    self.request_override_settings.portal_grant_header,
                    self.request_override_settings.qdrant_url_header,
                    self.request_override_settings.qdrant_api_key_header,
                    self.request_override_settings.collection_name_header,
                    self.request_override_settings.embedding_provider_header,
                    self.request_override_settings.embedding_model_header,
                    self.request_override_settings.openai_api_key_header,
                ],
            }
            if not portal_grant_configured:
                missing.append("MCP_PORTAL_GRANT_TOKEN")
            if (
                not qdrant_configured
                and not self.request_override_settings.allow_request_overrides
            ):
                missing.append("QDRANT_URL or QDRANT_LOCAL_PATH")
            return finish_request(state, data)

        async def list_capabilities(ctx: Context) -> dict[str, Any]:
            state = new_request(ctx, {})
            data = {
                "service": "qdrant-mcp",
                "tool_count": len(getattr(self._tool_manager, "_tools", {})),
                "groups": [
                    {
                        "name": "memory",
                        "tools": [
                            "qdrant-build-context",
                            "qdrant-study-search",
                            "qdrant-find",
                            "qdrant-store",
                            "qdrant-ingest-document",
                            "qdrant-ingest-textbook",
                        ],
                        "guidance": "Use compact search with top_k=3-5 before requesting payloads.",
                    },
                    {
                        "name": "collections",
                        "tools": [
                            "qdrant-list-collections",
                            "qdrant-describe-collection",
                            "qdrant-summarize-collection-schema",
                            "qdrant-suggest-filters",
                            "qdrant-create-collection",
                            "qdrant-collection-info",
                        ],
                    },
                    {
                        "name": "maintenance",
                        "tools": [
                            "qdrant-audit-memories",
                            "qdrant-dedupe-memories",
                            "qdrant-reembed-points",
                        ],
                        "guidance": "Run mutating tools with dry_run first and confirm only after reviewing the diff.",
                    },
                    {
                        "name": "navigation",
                        "tools": [
                            "qdrant-check-configuration",
                            "qdrant-list-capabilities",
                            "qdrant-get-endpoint-coverage",
                            "qdrant-get-tool-usage",
                        ],
                    },
                ],
            }
            return finish_request(state, data)

        async def get_endpoint_coverage(ctx: Context) -> dict[str, Any]:
            state = new_request(ctx, {})
            data = {
                "source": "docs/endpoint-coverage.md",
                "provider_docs": [
                    "https://api.qdrant.tech/master/api-reference",
                    "https://api.qdrant.tech/api-reference/collections",
                    "https://api.qdrant.tech/api-reference/points",
                    "https://api.qdrant.tech/api-reference/aliases/get-collections-aliases",
                    "https://api.qdrant.tech/api-reference/snapshots/list-snapshots",
                    "https://api.qdrant.tech/api-reference/service",
                ],
                "summary": {
                    "collections": "covered",
                    "points": "covered for memory workflows",
                    "payload_indexes": "covered",
                    "aliases": "read-only coverage",
                    "snapshots": "admin coverage with exclusions for downloads/uploads",
                    "cluster": "read-only collection cluster info",
                    "service": "partially covered by qdrant-health-check and /health",
                },
                "excluded": [
                    "Snapshot file download/upload endpoints are excluded to avoid returning binary payloads through MCP.",
                    "Alias mutation endpoints are excluded until alias-switch workflows have explicit dry-run and confirmation UX.",
                    "Cluster-wide service telemetry is excluded by default because it can be token-heavy and infrastructure-sensitive.",
                ],
            }
            return finish_request(state, data)

        async def get_tool_usage(
            ctx: Context,
            tool_name: Annotated[str, Field(description="Tool name to inspect.")],
        ) -> dict[str, Any]:
            state = new_request(ctx, {"tool_name": tool_name})
            normalized_name = tool_name.strip()
            if "." in normalized_name:
                normalized_name = normalized_name.rsplit(".", maxsplit=1)[1]
            if not self._tool_manager.has_tool(normalized_name):
                raise ValueError(f"Unknown tool '{tool_name}'.")
            tool = self._tool_manager.get_tool(normalized_name)
            search_guidance_tools = {
                "qdrant-build-context",
                "qdrant-find",
                "qdrant-study-search",
                "qdrant-recommend-memories",
            }
            data = {
                "name": normalized_name,
                "description": tool.description,
                "input_schema": tool.parameters,
                "annotations": tool.annotations.model_dump()
                if tool.annotations is not None
                else None,
                "guidance": (
                    "For search tools, start with response_mode='compact' and top_k=3-5. "
                    "Use response_mode='payload' only when the full document/payload is required."
                )
                if normalized_name in search_guidance_tools
                else None,
            }
            return finish_request(state, data)

        async def describe_collection(
            ctx: Context,
            collection_name: Annotated[
                str | None,
                Field(description="Collection to describe for agent navigation."),
            ] = None,
        ) -> dict[str, Any]:
            state = new_request(ctx, {"collection_name": collection_name})
            name = resolve_collection_name(collection_name)
            summary = await self.qdrant_connector.get_collection_summary(name)
            payload_schema = summary.get("payload_schema") or {}
            schema_fields = (
                sorted(payload_schema.keys())
                if isinstance(payload_schema, dict)
                else []
            )
            normalized_fields = [
                field.removeprefix(f"{METADATA_PATH}.") for field in schema_fields
            ]
            filterable_fields = [
                field for field in normalized_fields if field in FILTER_FIELDS
            ]
            data = {
                "collection_name": name,
                "status": summary.get("status"),
                "optimizer_status": summary.get("optimizer_status"),
                "points_count": summary.get("points_count"),
                "vectors": summary.get("vectors"),
                "payload_index_count": len(schema_fields),
                "payload_index_fields": schema_fields,
                "known_memory_filter_fields": sorted(filterable_fields),
                "suggested_tools": [
                    "qdrant-build-context",
                    "qdrant-suggest-filters",
                    "qdrant-find",
                    "qdrant-get-points",
                ],
            }
            if "warnings" in summary:
                data["warnings"] = summary["warnings"]
            return finish_request(state, data)

        async def summarize_collection_schema(
            ctx: Context,
            collection_name: Annotated[
                str | None,
                Field(
                    description="Collection whose payload schema should be summarized."
                ),
            ] = None,
        ) -> dict[str, Any]:
            state = new_request(ctx, {"collection_name": collection_name})
            name = resolve_collection_name(collection_name)
            payload_schema = await self.qdrant_connector.get_collection_payload_schema(
                name
            )
            schema_fields = sorted(payload_schema.keys())
            normalized_fields = [
                field.removeprefix(f"{METADATA_PATH}.") for field in schema_fields
            ]
            memory_filter_fields = [
                field for field in normalized_fields if field in FILTER_FIELDS
            ]
            data = {
                "collection_name": name,
                "indexed_field_count": len(schema_fields),
                "indexed_fields": schema_fields,
                "memory_filter_fields": sorted(memory_filter_fields),
                "memory_filter_keys": {
                    field: memory_filter_key_for_field(field)
                    for field in sorted(memory_filter_fields)
                },
                "unindexed_note": (
                    "Qdrant payload schema lists indexed fields. Use qdrant-suggest-filters "
                    "to sample values from stored payloads."
                ),
            }
            return finish_request(state, data)

        async def suggest_filters(
            ctx: Context,
            collection_name: Annotated[
                str | None,
                Field(description="Collection to inspect for useful exact filters."),
            ] = None,
            query: Annotated[
                str | None,
                Field(
                    description=(
                        "Optional task/query text. Sample values found inside this "
                        "text are returned as recommended filters."
                    )
                ),
            ] = None,
            fields: Annotated[
                list[str] | None,
                Field(
                    description=(
                        "Optional metadata fields to inspect. Defaults to indexed "
                        "fields that qdrant memory filters understand."
                    )
                ),
            ] = None,
            sample_limit: Annotated[
                int,
                Field(description="Max payload samples to inspect, clamped to 0-100."),
            ] = 25,
            max_values_per_field: Annotated[
                int,
                Field(
                    description="Max distinct sample values per field, clamped to 1-20."
                ),
            ] = 8,
        ) -> dict[str, Any]:
            state = new_request(
                ctx,
                {
                    "collection_name": collection_name,
                    "query": query,
                    "fields": fields,
                    "sample_limit": sample_limit,
                    "max_values_per_field": max_values_per_field,
                },
            )
            name = resolve_collection_name(collection_name)
            payload_schema = await self.qdrant_connector.get_collection_payload_schema(
                name
            )
            schema_fields = sorted(payload_schema.keys())
            normalized_schema_fields = [
                field.removeprefix(f"{METADATA_PATH}.") for field in schema_fields
            ]
            requested_fields = normalize_metadata_fields(fields)
            if requested_fields is None:
                selected_fields = [
                    field
                    for field in normalized_schema_fields
                    if field in FILTER_FIELDS
                ]
            else:
                selected_fields = [
                    field for field in requested_fields if field in FILTER_FIELDS
                ]
                ignored = [
                    field for field in requested_fields if field not in FILTER_FIELDS
                ]
                if ignored:
                    state.warnings.append(
                        f"Fields without memory_filter support ignored: {ignored}"
                    )
            selected_fields = sorted(dict.fromkeys(selected_fields))
            sample_limit = max(0, min(sample_limit, 100))
            max_values_per_field = max(1, min(max_values_per_field, 20))

            samples: dict[str, list[Any]] = {field: [] for field in selected_fields}
            sampled_points = 0
            if selected_fields and sample_limit > 0:
                points, _ = await self.qdrant_connector.scroll_points_page(
                    collection_name=name,
                    limit=sample_limit,
                    with_payload=True,
                    with_vectors=False,
                )
                sampled_points = len(points)
                for point in points:
                    payload = point.payload or {}
                    for field in selected_fields:
                        add_sample_value(
                            samples,
                            field,
                            metadata_value_from_payload(payload, field),
                            max_values=max_values_per_field,
                        )

            recommended_filters: dict[str, Any] = {}
            query_text = (query or "").lower()
            if query_text:
                for field, values in samples.items():
                    for value in values:
                        value_text = str(value).strip()
                        if value_text and value_text.lower() in query_text:
                            recommended_filters[memory_filter_key_for_field(field)] = (
                                value
                            )
                            break

            field_rows = []
            indexed = set(schema_fields) | set(normalized_schema_fields)
            for field in selected_fields:
                field_rows.append(
                    {
                        "field": field,
                        "memory_filter_key": memory_filter_key_for_field(field),
                        "indexed": (
                            field in indexed or f"{METADATA_PATH}.{field}" in indexed
                        ),
                        "sample_values": samples.get(field, []),
                    }
                )

            data = {
                "collection_name": name,
                "sampled_points": sampled_points,
                "fields": field_rows,
                "recommended_filters": recommended_filters,
                "usage": (
                    "Pass recommended_filters as memory_filter to qdrant-build-context "
                    "or qdrant-find, then keep compact output unless full payloads are needed."
                ),
            }
            return finish_request(state, data)

        find_foo = find
        study_search_foo = study_search
        build_context_foo = build_context
        recommend_memories_foo = recommend_memories
        store_foo = store
        cache_memory_foo = cache_memory
        promote_short_term_foo = promote_short_term
        update_foo = update_point
        patch_foo = patch_payload
        tag_memories_foo = tag_memories
        link_memories_foo = link_memories
        list_points_foo = list_points
        get_points_foo = get_points
        count_points_foo = count_points
        audit_memories_foo = audit_memories
        find_near_duplicates_foo = find_near_duplicates
        reembed_points_foo = reembed_points
        validate_memory_foo = validate_memory
        ingest_with_validation_foo = ingest_with_validation
        expire_memories_foo = expire_memories
        merge_duplicates_foo = merge_duplicates
        bulk_patch_foo = bulk_patch
        dedupe_memories_foo = dedupe_memories
        delete_points_foo = delete_points
        delete_filter_foo = delete_by_filter

        filterable_conditions = (
            self.qdrant_settings.filterable_fields_dict_with_conditions()
        )

        if len(filterable_conditions) > 0:
            find_foo = wrap_filters(find_foo, filterable_conditions)
            recommend_memories_foo = wrap_filters(
                recommend_memories_foo, filterable_conditions
            )
        elif not self.qdrant_settings.allow_arbitrary_filter:
            find_foo = make_partial_function(find_foo, {"query_filter": None})
            recommend_memories_foo = make_partial_function(
                recommend_memories_foo, {"query_filter": None}
            )

        short_term_find_foo = None
        if self.memory_settings.short_term_collection:
            short_term_find_foo = make_partial_function(
                find_foo,
                {"collection_name": self.memory_settings.short_term_collection},
            )

        short_term_expire_foo = None
        if self.memory_settings.short_term_collection:
            short_term_expire_foo = make_partial_function(
                expire_memories_foo,
                {"collection_name": self.memory_settings.short_term_collection},
            )

        if (
            self.qdrant_settings.collection_name
            and not self.request_override_settings.allow_request_overrides
        ):
            find_foo = make_partial_function(
                find_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            build_context_foo = make_partial_function(
                build_context_foo,
                {"collection_name": self.qdrant_settings.collection_name},
            )
            recommend_memories_foo = make_partial_function(
                recommend_memories_foo,
                {"collection_name": self.qdrant_settings.collection_name},
            )
            store_foo = make_partial_function(
                store_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            update_foo = make_partial_function(
                update_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            patch_foo = make_partial_function(
                patch_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            tag_memories_foo = make_partial_function(
                tag_memories_foo,
                {"collection_name": self.qdrant_settings.collection_name},
            )
            link_memories_foo = make_partial_function(
                link_memories_foo,
                {"collection_name": self.qdrant_settings.collection_name},
            )
            list_points_foo = make_partial_function(
                list_points_foo,
                {"collection_name": self.qdrant_settings.collection_name},
            )
            get_points_foo = make_partial_function(
                get_points_foo,
                {"collection_name": self.qdrant_settings.collection_name},
            )
            count_points_foo = make_partial_function(
                count_points_foo,
                {"collection_name": self.qdrant_settings.collection_name},
            )
            audit_memories_foo = make_partial_function(
                audit_memories_foo,
                {"collection_name": self.qdrant_settings.collection_name},
            )
            find_near_duplicates_foo = make_partial_function(
                find_near_duplicates_foo,
                {"collection_name": self.qdrant_settings.collection_name},
            )
            reembed_points_foo = make_partial_function(
                reembed_points_foo,
                {"collection_name": self.qdrant_settings.collection_name},
            )
            ingest_with_validation_foo = make_partial_function(
                ingest_with_validation_foo,
                {"collection_name": self.qdrant_settings.collection_name},
            )
            expire_memories_foo = make_partial_function(
                expire_memories_foo,
                {"collection_name": self.qdrant_settings.collection_name},
            )
            merge_duplicates_foo = make_partial_function(
                merge_duplicates_foo,
                {"collection_name": self.qdrant_settings.collection_name},
            )
            bulk_patch_foo = make_partial_function(
                bulk_patch_foo,
                {"collection_name": self.qdrant_settings.collection_name},
            )
            dedupe_memories_foo = make_partial_function(
                dedupe_memories_foo,
                {"collection_name": self.qdrant_settings.collection_name},
            )
            delete_points_foo = make_partial_function(
                delete_points_foo,
                {"collection_name": self.qdrant_settings.collection_name},
            )
            delete_filter_foo = make_partial_function(
                delete_filter_foo,
                {"collection_name": self.qdrant_settings.collection_name},
            )

        collection_required_tag = "[Requires collection name]"
        collection_required_tool_names = {
            "qdrant-build-context",
            "qdrant-describe-collection",
            "qdrant-summarize-collection-schema",
            "qdrant-suggest-filters",
            "qdrant-find",
            "qdrant-recommend-memories",
            "qdrant-list-points",
            "qdrant-get-points",
            "qdrant-count-points",
            "qdrant-audit-memories",
            "qdrant-find-near-duplicates",
            "qdrant-ingest-textbook",
            "qdrant-collection-exists",
            "qdrant-collection-info",
            "qdrant-collection-stats",
            "qdrant-collection-vectors",
            "qdrant-collection-payload-schema",
            "qdrant-optimizer-status",
            "qdrant-metrics-snapshot",
            "qdrant-get-vector-name",
            "qdrant-collection-aliases",
            "qdrant-collection-cluster-info",
            "qdrant-list-snapshots",
            "qdrant-create-snapshot",
            "qdrant-restore-snapshot",
            "qdrant-list-shard-snapshots",
            "qdrant-create-collection",
            "qdrant-store",
            "qdrant-promote-short-term",
            "qdrant-ingest-with-validation",
            "qdrant-ingest-document",
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
            "qdrant-update-optimizer-config",
            "qdrant-delete-points",
            "qdrant-delete-by-filter",
            "qdrant-delete-document",
        }
        navigation_tool_names = {
            "qdrant-check-configuration",
            "qdrant-list-capabilities",
            "qdrant-get-endpoint-coverage",
            "qdrant-get-tool-usage",
        }
        common_parameter_descriptions = {
            "collection_name": "Qdrant collection name to use for this operation.",
            "query_filter": (
                "Raw Qdrant filter object. Prefer memory_filter unless arbitrary "
                "filters are explicitly enabled."
            ),
            "shard_id": "Qdrant shard id to inspect.",
        }
        destructive_tool_names = {
            "qdrant-delete-points",
            "qdrant-delete-by-filter",
            "qdrant-delete-document",
            "qdrant-expire-memories",
            "qdrant-expire-short-term",
            "qdrant-merge-duplicates",
            "qdrant-restore-snapshot",
        }
        mutation_tool_names = {
            "qdrant-store",
            "qdrant-cache-memory",
            "qdrant-promote-short-term",
            "qdrant-ingest-with-validation",
            "qdrant-ingest-document",
            "qdrant-ingest-textbook",
            "qdrant-ensure-payload-indexes",
            "qdrant-create-collection",
            "qdrant-create-snapshot",
            "qdrant-restore-snapshot",
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
            "qdrant-submit-job",
            "qdrant-cancel-job",
            "qdrant-cancel-ingest",
        }

        def register_tool(
            tool_fn: Any,
            *,
            name: str,
            description: str,
        ) -> None:
            final_description = description
            if (
                name in collection_required_tool_names
                and collection_required_tag not in final_description
            ):
                final_description = f"{description} {collection_required_tag}"
            read_only = name not in mutation_tool_names
            destructive = name in destructive_tool_names
            open_world = name not in navigation_tool_names
            self.tool(
                tool_fn,
                name=name,
                description=final_description,
                annotations=ToolAnnotations(
                    title=name,
                    readOnlyHint=read_only,
                    destructiveHint=destructive,
                    idempotentHint=True if read_only else None,
                    openWorldHint=open_world,
                ),
            )
            tool = self._tool_manager.get_tool(name)
            properties = tool.parameters.get("properties", {})
            for (
                property_name,
                fallback_description,
            ) in common_parameter_descriptions.items():
                property_schema = properties.get(property_name)
                if isinstance(property_schema, dict) and not property_schema.get(
                    "description"
                ):
                    property_schema["description"] = fallback_description

        register_tool(
            check_configuration,
            name="qdrant-check-configuration",
            description=(
                "Check Qdrant MCP configuration readiness without returning secrets."
            ),
        )
        register_tool(
            list_capabilities,
            name="qdrant-list-capabilities",
            description=(
                "List Qdrant MCP capability groups and common agent workflows."
            ),
        )
        register_tool(
            get_endpoint_coverage,
            name="qdrant-get-endpoint-coverage",
            description=(
                "Summarize Qdrant API endpoint coverage and documented exclusions."
            ),
        )
        register_tool(
            get_tool_usage,
            name="qdrant-get-tool-usage",
            description=(
                "Explain when to use a Qdrant MCP tool, its inputs, annotations, and token-safe guidance."
            ),
        )

        register_tool(
            health_check,
            name="qdrant-health-check",
            description="Run health checks against Qdrant and embedding clients.",
        )

        register_tool(
            build_context_foo,
            name="qdrant-build-context",
            description=(
                "Build a collection-agnostic, token-budgeted second-brain context "
                "pack with cited snippets, deduping, optional filters, and safe "
                "relax-filters fallback."
            ),
        )

        register_tool(
            study_search_foo,
            name="qdrant-study-search",
            description=(
                "Search school/study materials with compact results by default. "
                "Defaults to MCP_STUDY_COLLECTION and supports course, subject, "
                "module, week, status, material_type, title, author, doc_id, and "
                "chapter filters to avoid noisy cross-course retrieval."
            ),
        )

        register_tool(
            find_foo,
            name="qdrant-find",
            description=(
                self.tool_settings.tool_find_description
                + " Returns compact results by default; start with top_k=3-5 and filters, "
                "then request response_mode='payload' only when full payloads are needed."
            ),
        )
        if short_term_find_foo is not None:
            register_tool(
                short_term_find_foo,
                name="qdrant-find-short-term",
                description=(
                    "Search the short-term memory cache collection. Returns compact "
                    "results by default; start with top_k=3-5."
                ),
            )
        register_tool(
            recommend_memories_foo,
            name="qdrant-recommend-memories",
            description=(
                "Recommend memories using positive/negative example ids. Returns compact "
                "results by default; use response_mode='payload' only when full payloads are needed."
            ),
        )
        register_tool(
            validate_memory_foo,
            name="qdrant-validate-memory",
            description="Validate memory contract fields before ingest.",
        )
        register_tool(
            list_points_foo,
            name="qdrant-list-points",
            description="List points with pagination (scroll).",
        )
        register_tool(
            get_points_foo,
            name="qdrant-get-points",
            description="Retrieve points by id.",
        )
        register_tool(
            count_points_foo,
            name="qdrant-count-points",
            description="Count points matching an optional filter.",
        )
        register_tool(
            audit_memories_foo,
            name="qdrant-audit-memories",
            description="Audit memory payloads for missing fields and duplicates.",
        )
        register_tool(
            find_near_duplicates_foo,
            name="qdrant-find-near-duplicates",
            description="Find near-duplicate points using vector similarity.",
        )
        register_tool(
            submit_job,
            name="qdrant-submit-job",
            description="Submit a long-running housekeeping job.",
        )
        register_tool(
            ingest_textbook,
            name="qdrant-ingest-textbook",
            description=(
                "Submit an asynchronous textbook PDF ingest job from source_url. "
                "Returns immediately with a job_id."
            ),
        )
        register_tool(
            get_ingest_status,
            name="qdrant-get-ingest-status",
            description="Get status, progress, metrics, and structured errors for a textbook ingest job.",
        )
        register_tool(
            cancel_ingest,
            name="qdrant-cancel-ingest",
            description="Cancel a running textbook ingest job.",
        )
        register_tool(
            job_status,
            name="qdrant-job-status",
            description="Check status for a submitted job.",
        )
        register_tool(
            job_progress,
            name="qdrant-job-progress",
            description="Get progress for a submitted job.",
        )
        register_tool(
            job_logs,
            name="qdrant-job-logs",
            description="Fetch recent logs for a submitted job.",
        )
        register_tool(
            job_result,
            name="qdrant-job-result",
            description="Fetch the result for a completed job.",
        )
        register_tool(
            cancel_job,
            name="qdrant-cancel-job",
            description="Cancel a running job.",
        )

        register_tool(
            list_collections,
            name="qdrant-list-collections",
            description="List all Qdrant collections.",
        )
        if not self.qdrant_settings.read_only:
            register_tool(
                create_collection,
                name="qdrant-create-collection",
                description=(
                    "Create a collection with embedding-compatible vector settings "
                    "and default payload indexes."
                ),
            )
        register_tool(
            collection_exists,
            name="qdrant-collection-exists",
            description="Check if a collection exists.",
        )
        register_tool(
            describe_collection,
            name="qdrant-describe-collection",
            description=(
                "Describe a collection for agents: size, vectors, indexed fields, "
                "and useful retrieval tools."
            ),
        )
        register_tool(
            summarize_collection_schema,
            name="qdrant-summarize-collection-schema",
            description=(
                "Summarize indexed payload schema and memory_filter keys for a collection."
            ),
        )
        register_tool(
            suggest_filters,
            name="qdrant-suggest-filters",
            description=(
                "Sample a collection's indexed metadata values and suggest exact "
                "memory_filter fields for cheaper, cleaner retrieval."
            ),
        )
        register_tool(
            collection_info,
            name="qdrant-collection-info",
            description="Get collection details including vectors and payload schema.",
        )
        register_tool(
            collection_stats,
            name="qdrant-collection-stats",
            description="Get basic collection statistics (points, segments, status).",
        )
        register_tool(
            collection_vectors,
            name="qdrant-collection-vectors",
            description="List vector names and sizes for a collection.",
        )
        register_tool(
            collection_payload_schema,
            name="qdrant-collection-payload-schema",
            description="Get payload schema for a collection.",
        )
        register_tool(
            optimizer_status,
            name="qdrant-optimizer-status",
            description="Get optimizer config and index coverage for a collection.",
        )
        register_tool(
            metrics_snapshot,
            name="qdrant-metrics-snapshot",
            description="Snapshot collection stats and index coverage metrics.",
        )
        register_tool(
            get_vector_name,
            name="qdrant-get-vector-name",
            description="Resolve the vector name used by this MCP server.",
        )
        register_tool(
            list_aliases,
            name="qdrant-list-aliases",
            description="List all collection aliases.",
        )
        register_tool(
            collection_aliases,
            name="qdrant-collection-aliases",
            description="List aliases for a specific collection.",
        )
        register_tool(
            collection_cluster_info,
            name="qdrant-collection-cluster-info",
            description="Get cluster info for a collection.",
        )
        register_tool(
            list_snapshots,
            name="qdrant-list-snapshots",
            description="List snapshots for a collection.",
        )
        register_tool(
            create_snapshot,
            name="qdrant-create-snapshot",
            description="Create a collection snapshot (admin-only, confirm required).",
        )
        register_tool(
            restore_snapshot,
            name="qdrant-restore-snapshot",
            description="Restore a collection snapshot (admin-only, confirm required).",
        )
        register_tool(
            list_full_snapshots,
            name="qdrant-list-full-snapshots",
            description="List full cluster snapshots.",
        )
        register_tool(
            list_shard_snapshots,
            name="qdrant-list-shard-snapshots",
            description="List snapshots for a specific shard.",
        )

        if not self.qdrant_settings.read_only:
            register_tool(
                store_foo,
                name="qdrant-store",
                description=self.tool_settings.tool_store_description,
            )
            register_tool(
                cache_memory_foo,
                name="qdrant-cache-memory",
                description="Store short-term memory with a TTL in a cache collection.",
            )
            register_tool(
                promote_short_term_foo,
                name="qdrant-promote-short-term",
                description="Promote short-term memories into the long-term collection.",
            )
            register_tool(
                ingest_with_validation_foo,
                name="qdrant-ingest-with-validation",
                description="Store memory with contract validation and quarantine support.",
            )
            register_tool(
                ingest_document,
                name="qdrant-ingest-document",
                description=(
                    "Ingest documents (txt, md, csv, pdf, doc, docx) by extracting text and storing chunks."
                ),
            )
            register_tool(
                ensure_payload_indexes,
                name="qdrant-ensure-payload-indexes",
                description="Ensure expected payload indexes exist for a collection.",
            )
            register_tool(
                backfill_memory_contract,
                name="qdrant-backfill-memory-contract",
                description="Backfill missing memory contract fields for existing points.",
            )
            register_tool(
                update_foo,
                name="qdrant-update-point",
                description="Update an existing point (re-embeds content).",
            )
            register_tool(
                patch_foo,
                name="qdrant-patch-payload",
                description="Patch payload metadata for a point.",
            )
            register_tool(
                tag_memories_foo,
                name="qdrant-tag-memories",
                description="Append or replace labels for a set of points.",
            )
            register_tool(
                link_memories_foo,
                name="qdrant-link-memories",
                description="Link memories via related_ids and optional associations.",
            )
            register_tool(
                reembed_points_foo,
                name="qdrant-reembed-points",
                description="Re-embed points when embedding version changes.",
            )
            register_tool(
                bulk_patch_foo,
                name="qdrant-bulk-patch",
                description="Apply metadata/payload patches to points by id or filter.",
            )
            register_tool(
                dedupe_memories_foo,
                name="qdrant-dedupe-memories",
                description="Find and optionally delete duplicate memories.",
            )
            register_tool(
                merge_duplicates_foo,
                name="qdrant-merge-duplicates",
                description="Merge duplicate points into a canonical point.",
            )
            register_tool(
                expire_memories_foo,
                name="qdrant-expire-memories",
                description="Expire memories by expires_at_ts (optional archive).",
            )
            if short_term_expire_foo is not None:
                register_tool(
                    short_term_expire_foo,
                    name="qdrant-expire-short-term",
                    description="Expire short-term memories by expires_at_ts.",
                )
            if self.tool_settings.admin_tools_enabled:
                register_tool(
                    update_optimizer_config,
                    name="qdrant-update-optimizer-config",
                    description=(
                        "Update optimizer config (admin; confirm + dry_run=false "
                        "required)."
                    ),
                )
            register_tool(
                delete_points_foo,
                name="qdrant-delete-points",
                description="Delete points by id (confirm required).",
            )
            register_tool(
                delete_filter_foo,
                name="qdrant-delete-by-filter",
                description="Delete points by filter (confirm required).",
            )
            register_tool(
                delete_document,
                name="qdrant-delete-document",
                description="Delete all chunks for a document by doc_id (confirm required).",
            )
