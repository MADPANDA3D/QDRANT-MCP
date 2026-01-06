import asyncio
import base64
import hashlib
import json
import logging
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Any
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from fastmcp import Context, FastMCP
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
    ToolSettings,
)

logger = logging.getLogger(__name__)


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
        memory_settings: MemorySettings | None = None,
        embedding_provider_settings: EmbeddingProviderSettings | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        name: str = "mcp-server-qdrant",
        instructions: str | None = None,
        **settings: Any,
    ):
        self.tool_settings = tool_settings
        self.qdrant_settings = qdrant_settings
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

        self.qdrant_connector = QdrantConnector(
            qdrant_settings.location,
            qdrant_settings.api_key,
            qdrant_settings.collection_name,
            self.embedding_provider,
            qdrant_settings.vector_name,
            qdrant_settings.local_path,
            field_indexes,
        )

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    async def _mcp_call_tool(
        self, key: str, arguments: dict[str, Any]
    ) -> list[TextContent | ImageContent | EmbeddedResource]:
        """
        Normalize tool arguments before validation to tolerate MCP clients
        that wrap arguments inside Airtable-like records (id/createdTime/fields).
        """
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

        return await super()._mcp_call_tool(key, arguments)

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
            if self.qdrant_settings.collection_name:
                return self.qdrant_settings.collection_name
            raise ValueError("collection_name is required")

        def resolve_health_collection(collection_name: str | None) -> str:
            name = collection_name.strip() if collection_name else ""
            if name:
                return name
            if self.qdrant_settings.collection_name:
                return self.qdrant_settings.collection_name
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
                raise ValueError(
                    "file_type must be one of: txt, md, pdf, doc, docx."
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
            await ctx.debug(f"Storing information {information} in Qdrant")
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

            action = (dedupe_action or self.memory_settings.dedupe_action).lower()
            if action not in {"update", "skip"}:
                if self.memory_settings.strict_params:
                    raise ValueError("dedupe_action must be 'update' or 'skip'.")
                state.warnings.append(
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

            data = {
                "collection_name": collection,
                "dedupe_action": action,
                "results": results,
            }
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
                        list(source_url_headers.keys())
                        if source_url_headers
                        else None
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
                    add_warning("source_url ignored because content_base64 was provided.")
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

            if resolved_file_type in {"txt", "md"} and text is None and file_bytes is None:
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
                        schema = await self.qdrant_connector.get_collection_payload_schema(
                            collection
                        )
                    except Exception as exc:  # pragma: no cover - transport errors vary
                        add_warning(f"Failed to read payload schema: {exc}")
                        schema = {}

                    if doc_id_key not in schema:
                        try:
                            created = await self.qdrant_connector.ensure_payload_indexes(
                                collection_name=collection,
                                indexes={
                                    doc_id_key: models.PayloadSchemaType.KEYWORD
                                },
                            )
                            if doc_id_key in created:
                                add_warning(
                                    "Created payload index for metadata.doc_id."
                                )
                        except Exception as exc:  # pragma: no cover - transport errors vary
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
                data = {
                    "deleted": 0,
                    "requested": len(point_ids),
                    "dry_run": True,
                }
                return finish_request(state, data)

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
                data = {"matched": matched, "deleted": 0, "dry_run": True}
                return finish_request(state, data)

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
                data = {
                    "doc_id": doc_id,
                    "collection_name": collection,
                    "matched": matched,
                    "deleted": 0,
                    "dry_run": True,
                }
                return finish_request(state, data)

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

            collection = resolve_collection_name(collection_name)
            scanned = 0
            updated = 0
            skipped = 0
            offset = None
            warning_set: set[str] = set()
            stop = False

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

        find_foo = find
        store_foo = store
        update_foo = update_point
        patch_foo = patch_payload
        delete_points_foo = delete_points
        delete_filter_foo = delete_by_filter

        filterable_conditions = (
            self.qdrant_settings.filterable_fields_dict_with_conditions()
        )

        if len(filterable_conditions) > 0:
            find_foo = wrap_filters(find_foo, filterable_conditions)
        elif not self.qdrant_settings.allow_arbitrary_filter:
            find_foo = make_partial_function(find_foo, {"query_filter": None})

        if self.qdrant_settings.collection_name:
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
