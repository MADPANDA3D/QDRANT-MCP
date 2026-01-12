import asyncio
import base64
import hashlib
import json
import logging
import math
import uuid
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from fastmcp import Context, FastMCP

try:  # FastMCP >= 2.2.11
    from fastmcp.server.dependencies import get_http_headers
except ImportError:  # pragma: no cover - older FastMCP

    def get_http_headers() -> dict[str, str]:
        return {}


from mcp.types import EmbeddedResource, ImageContent, TextContent
from pydantic import Field
from qdrant_client import models

from mcp_server_qdrant.common.filters import make_indexes
from mcp_server_qdrant.common.func_tools import make_partial_function
from mcp_server_qdrant.common.telemetry import finish_request, new_request
from mcp_server_qdrant.common.wrap_filters import wrap_filters
from mcp_server_qdrant.document_ingest import (
    chunk_text_with_overlap,
    extract_document_sections,
)
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.memory import (
    ALLOWED_MEMORY_KEYS,
    DEFAULT_CONFIDENCE,
    DEFAULT_MEMORY_TYPE,
    DEFAULT_SCOPE,
    DEFAULT_SOURCE,
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

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    @property
    def qdrant_connector(self) -> QdrantConnector:
        connector = self._connector_var.get()
        return connector or self._default_qdrant_connector

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

    def _resolve_embedding_info(self) -> EmbeddingInfo:
        model_name = "unknown"
        provider_name = "unknown"
        version = "unknown"

        if self.embedding_provider_settings:
            provider_name = self.embedding_provider_settings.provider_type.value
            model_name = self.embedding_provider_settings.model_name
            version = self.embedding_provider_settings.version or model_name
        else:
            provider_name = (
                getattr(self.embedding_provider, "provider_type", None)
                or self.embedding_provider.__class__.__name__.lower()
            )
            model_name = getattr(self.embedding_provider, "model_name", "unknown")
            version = getattr(self.embedding_provider, "version", None) or model_name

        dim = self.embedding_provider.get_vector_size()
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

        def resolve_collection_name(collection_name: str) -> str:
            name = collection_name.strip() if collection_name else ""
            if name:
                return name
            default_name = self._get_default_collection_name()
            if default_name:
                return default_name
            raise ValueError("collection_name is required")

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

        DRY_RUN_PREVIEW_LIMIT = 5
        DRY_RUN_GROUP_FIELDS = ("scope", "type", "labels", "source", "doc_id")
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
            metadata: dict[str, Any], keys: set[str] | None = None
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

            allowed = {"txt", "md", "pdf", "doc", "docx"}
            if candidate not in allowed:
                raise ValueError("file_type must be one of: txt, md, pdf, doc, docx.")
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
                checks["collection_exists"] = {"ok": exists, "collection_name": name}
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
                Field(description="File type/extension: txt, md, pdf, doc, docx."),
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
                Field(description="Enable OCR for PDF pages without text."),
            ] = False,
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
                resolved_file_type in {"txt", "md"}
                and text is None
                and file_bytes is None
            ):
                raise ValueError("txt/md ingestion requires text or file bytes.")

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
                str, Field(description="The collection to search in")
            ],
            memory_filter: Annotated[
                MemoryFilterInput | None,
                Field(description="Structured filters for memory fields."),
            ] = None,
            query_filter: ArbitraryFilter | None = None,
            top_k: Annotated[
                int | None, Field(description="Max number of results to return.")
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

            collection = resolve_collection_name(collection_name)
            filter_applied = combined_filter is not None
            query_vector_dim = self.embedding_provider.get_vector_size()

            points: list[models.ScoredPoint]
            if use_mmr:
                if mmr_lambda < 0 or mmr_lambda > 1:
                    if self.memory_settings.strict_params:
                        raise ValueError("mmr_lambda must be between 0 and 1.")
                    state.warnings.append("mmr_lambda clamped to [0,1].")
                    mmr_lambda = max(0.0, min(1.0, mmr_lambda))

                query_vector = await self.embedding_provider.embed_query(query)
                vector_name = await self.qdrant_connector.resolve_vector_name(
                    collection
                )
                candidate_limit = min(max(limit * 4, limit), 100)
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
                    top_k=limit,
                    lambda_mult=mmr_lambda,
                    vector_name=vector_name,
                )
                if selected is None:
                    state.warnings.append("MMR disabled due to missing vectors.")
                    points = points[:limit]
                else:
                    points = selected
            else:
                points = await self.qdrant_connector.search_points(
                    query,
                    collection_name=collection,
                    limit=limit,
                    query_filter=combined_filter,
                )

            results = []
            for point in points:
                payload = point.payload or {}
                text = extract_payload_text(payload)
                results.append(
                    {
                        "id": str(point.id),
                        "score": point.score,
                        "payload": payload,
                        "snippet": make_snippet(text),
                    }
                )

            data = {
                "query": query,
                "collection_name": collection,
                "results": results,
            }
            extra_meta = {
                "top_k": limit,
                "filter_applied": filter_applied,
                "query_vector_dim": query_vector_dim,
                "mmr": use_mmr,
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

        def append_job_log(record: dict[str, Any], message: str) -> None:
            logs = record.setdefault("logs", [])
            if not isinstance(logs, list):
                logs = []
                record["logs"] = logs
            logs.append(
                {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "message": str(message),
                }
            )
            if len(logs) > JOB_LOG_LIMIT:
                del logs[: len(logs) - JOB_LOG_LIMIT]

        class JobContext:
            def __init__(self, request_id: str, record: dict[str, Any]):
                self.request_id = request_id
                self._record = record

            async def debug(self, _message: str) -> None:
                return None

            def log(self, message: str) -> None:
                append_job_log(self._record, message)

            def set_phase(self, phase: str) -> None:
                update_job_progress(self._record, phase=phase)

            def set_total(self, total: int | None) -> None:
                update_job_progress(self._record, items_total=total)

            def advance(self, count: int) -> None:
                update_job_progress(self._record, advance=count)

        job_handlers: dict[str, Any] = {
            "audit-memories": audit_memories,
            "backfill-memory-contract": backfill_memory_contract,
            "bulk-patch": bulk_patch,
            "dedupe-memories": dedupe_memories,
            "expire-memories": expire_memories,
            "find-near-duplicates": find_near_duplicates,
            "merge-duplicates": merge_duplicates,
            "reembed-points": reembed_points,
        }

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
            handler = job_handlers.get(job_key)
            if handler is None:
                raise ValueError(f"Unknown job_type '{job_type}'.")

            args = dict(job_args or {})
            if "collection_name" not in args:
                default_collection = self._get_default_collection_name()
                if default_collection:
                    args["collection_name"] = default_collection

            job_id = uuid.uuid4().hex
            now = datetime.now(timezone.utc).isoformat()
            record = {
                "job_id": job_id,
                "job_type": job_key,
                "status": "queued",
                "created_at": now,
                "started_at": None,
                "finished_at": None,
                "error": None,
            }
            init_job_progress(record)
            append_job_log(record, f"queued job_type={job_key}")
            self._jobs[job_id] = record

            async def run_job() -> None:
                record["status"] = "running"
                record["started_at"] = datetime.now(timezone.utc).isoformat()
                update_job_progress(record, phase="running")
                append_job_log(record, f"started job_type={job_key}")
                job_ctx = JobContext(job_id, record)
                try:
                    result = await handler(job_ctx, **args)
                    record["status"] = "completed"
                    record["result"] = result
                    update_job_progress(record, phase="completed")
                    append_job_log(record, f"completed job_type={job_key}")
                except Exception as exc:  # pragma: no cover - runtime safety
                    record["status"] = "failed"
                    record["error"] = str(exc)
                    update_job_progress(record, phase="failed")
                    append_job_log(record, f"failed job_type={job_key}: {exc}")
                record["finished_at"] = datetime.now(timezone.utc).isoformat()

            task = asyncio.create_task(run_job())
            self._job_tasks[job_id] = task
            data = {"job_id": job_id, "status": "queued"}
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

        find_foo = find
        store_foo = store
        update_foo = update_point
        patch_foo = patch_payload
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
        elif not self.qdrant_settings.allow_arbitrary_filter:
            find_foo = make_partial_function(find_foo, {"query_filter": None})

        if (
            self.qdrant_settings.collection_name
            and not self.request_override_settings.allow_request_overrides
        ):
            find_foo = make_partial_function(
                find_foo, {"collection_name": self.qdrant_settings.collection_name}
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

        self.tool(
            health_check,
            name="qdrant-health-check",
            description="Run health checks against Qdrant and embedding clients.",
        )

        self.tool(
            find_foo,
            name="qdrant-find",
            description=self.tool_settings.tool_find_description,
        )
        self.tool(
            validate_memory_foo,
            name="qdrant-validate-memory",
            description="Validate memory contract fields before ingest.",
        )
        self.tool(
            list_points_foo,
            name="qdrant-list-points",
            description="List points with pagination (scroll).",
        )
        self.tool(
            get_points_foo,
            name="qdrant-get-points",
            description="Retrieve points by id.",
        )
        self.tool(
            count_points_foo,
            name="qdrant-count-points",
            description="Count points matching an optional filter.",
        )
        self.tool(
            audit_memories_foo,
            name="qdrant-audit-memories",
            description="Audit memory payloads for missing fields and duplicates.",
        )
        self.tool(
            find_near_duplicates_foo,
            name="qdrant-find-near-duplicates",
            description="Find near-duplicate points using vector similarity.",
        )
        self.tool(
            submit_job,
            name="qdrant-submit-job",
            description="Submit a long-running housekeeping job.",
        )
        self.tool(
            job_status,
            name="qdrant-job-status",
            description="Check status for a submitted job.",
        )
        self.tool(
            job_progress,
            name="qdrant-job-progress",
            description="Get progress for a submitted job.",
        )
        self.tool(
            job_logs,
            name="qdrant-job-logs",
            description="Fetch recent logs for a submitted job.",
        )
        self.tool(
            job_result,
            name="qdrant-job-result",
            description="Fetch the result for a completed job.",
        )
        self.tool(
            cancel_job,
            name="qdrant-cancel-job",
            description="Cancel a running job.",
        )

        self.tool(
            list_collections,
            name="qdrant-list-collections",
            description="List all Qdrant collections.",
        )
        self.tool(
            collection_exists,
            name="qdrant-collection-exists",
            description="Check if a collection exists.",
        )
        self.tool(
            collection_info,
            name="qdrant-collection-info",
            description="Get collection details including vectors and payload schema.",
        )
        self.tool(
            collection_stats,
            name="qdrant-collection-stats",
            description="Get basic collection statistics (points, segments, status).",
        )
        self.tool(
            collection_vectors,
            name="qdrant-collection-vectors",
            description="List vector names and sizes for a collection.",
        )
        self.tool(
            collection_payload_schema,
            name="qdrant-collection-payload-schema",
            description="Get payload schema for a collection.",
        )
        self.tool(
            optimizer_status,
            name="qdrant-optimizer-status",
            description="Get optimizer config and index coverage for a collection.",
        )
        self.tool(
            metrics_snapshot,
            name="qdrant-metrics-snapshot",
            description="Snapshot collection stats and index coverage metrics.",
        )
        self.tool(
            get_vector_name,
            name="qdrant-get-vector-name",
            description="Resolve the vector name used by this MCP server.",
        )
        self.tool(
            list_aliases,
            name="qdrant-list-aliases",
            description="List all collection aliases.",
        )
        self.tool(
            collection_aliases,
            name="qdrant-collection-aliases",
            description="List aliases for a specific collection.",
        )
        self.tool(
            collection_cluster_info,
            name="qdrant-collection-cluster-info",
            description="Get cluster info for a collection.",
        )
        self.tool(
            list_snapshots,
            name="qdrant-list-snapshots",
            description="List snapshots for a collection.",
        )
        self.tool(
            create_snapshot,
            name="qdrant-create-snapshot",
            description="Create a collection snapshot (admin-only, confirm required).",
        )
        self.tool(
            restore_snapshot,
            name="qdrant-restore-snapshot",
            description="Restore a collection snapshot (admin-only, confirm required).",
        )
        self.tool(
            list_full_snapshots,
            name="qdrant-list-full-snapshots",
            description="List full cluster snapshots.",
        )
        self.tool(
            list_shard_snapshots,
            name="qdrant-list-shard-snapshots",
            description="List snapshots for a specific shard.",
        )

        if not self.qdrant_settings.read_only:
            self.tool(
                store_foo,
                name="qdrant-store",
                description=self.tool_settings.tool_store_description,
            )
            self.tool(
                ingest_with_validation_foo,
                name="qdrant-ingest-with-validation",
                description="Store memory with contract validation and quarantine support.",
            )
            self.tool(
                ingest_document,
                name="qdrant-ingest-document",
                description=(
                    "Ingest documents (txt, md, pdf, doc, docx) by extracting text and storing chunks."
                ),
            )
            self.tool(
                ensure_payload_indexes,
                name="qdrant-ensure-payload-indexes",
                description="Ensure expected payload indexes exist for a collection.",
            )
            self.tool(
                backfill_memory_contract,
                name="qdrant-backfill-memory-contract",
                description="Backfill missing memory contract fields for existing points.",
            )
            self.tool(
                update_foo,
                name="qdrant-update-point",
                description="Update an existing point (re-embeds content).",
            )
            self.tool(
                patch_foo,
                name="qdrant-patch-payload",
                description="Patch payload metadata for a point.",
            )
            self.tool(
                reembed_points_foo,
                name="qdrant-reembed-points",
                description="Re-embed points when embedding version changes.",
            )
            self.tool(
                bulk_patch_foo,
                name="qdrant-bulk-patch",
                description="Apply metadata/payload patches to points by id or filter.",
            )
            self.tool(
                dedupe_memories_foo,
                name="qdrant-dedupe-memories",
                description="Find and optionally delete duplicate memories.",
            )
            self.tool(
                merge_duplicates_foo,
                name="qdrant-merge-duplicates",
                description="Merge duplicate points into a canonical point.",
            )
            self.tool(
                expire_memories_foo,
                name="qdrant-expire-memories",
                description="Expire memories by expires_at_ts (optional archive).",
            )
            if self.tool_settings.admin_tools_enabled:
                self.tool(
                    update_optimizer_config,
                    name="qdrant-update-optimizer-config",
                    description=(
                        "Update optimizer config (admin; confirm + dry_run=false "
                        "required)."
                    ),
                )
            self.tool(
                delete_points_foo,
                name="qdrant-delete-points",
                description="Delete points by id (confirm required).",
            )
            self.tool(
                delete_filter_foo,
                name="qdrant-delete-by-filter",
                description="Delete points by filter (confirm required).",
            )
            self.tool(
                delete_document,
                name="qdrant-delete-document",
                description="Delete all chunks for a document by doc_id (confirm required).",
            )
