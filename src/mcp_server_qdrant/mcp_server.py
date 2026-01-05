import json
import logging
import math
from datetime import datetime, timezone
from typing import Annotated, Any, Optional

from fastmcp import Context, FastMCP
from mcp.types import EmbeddedResource, ImageContent, TextContent
from pydantic import Field
from qdrant_client import models

from mcp_server_qdrant.common.filters import make_indexes
from mcp_server_qdrant.common.telemetry import finish_request, new_request
from mcp_server_qdrant.common.func_tools import make_partial_function
from mcp_server_qdrant.common.wrap_filters import wrap_filters
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.memory import (
    ALLOWED_MEMORY_KEYS,
    EmbeddingInfo,
    MemoryFilterInput,
    build_memory_filter,
    default_memory_indexes,
    normalize_memory_input,
)
from mcp_server_qdrant.qdrant import ArbitraryFilter, Entry, Metadata, QdrantConnector
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    MemorySettings,
    METADATA_PATH,
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
        embedding_provider_settings: Optional[EmbeddingProviderSettings] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
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

        self.embedding_provider_settings: Optional[EmbeddingProviderSettings] = None
        self.embedding_provider: Optional[EmbeddingProvider] = None

        if embedding_provider_settings:
            self.embedding_provider_settings = embedding_provider_settings
            self.embedding_provider = create_embedding_provider(
                embedding_provider_settings
            )
        else:
            self.embedding_provider_settings = None
            self.embedding_provider = embedding_provider

        assert self.embedding_provider is not None, "Embedding provider is required"
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
                    checks["collection_status"] = {
                        "ok": True,
                        "status": str(info.status),
                        "optimizer_status": str(info.optimizer_status),
                        "points_count": info.points_count,
                        "indexed_vectors_count": info.indexed_vectors_count,
                        "segments_count": info.segments_count,
                    }
                    if info.points_count and info.indexed_vectors_count is not None:
                        checks["collection_status"]["fully_indexed"] = (
                            info.points_count == info.indexed_vectors_count
                        )
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
                    expected = set(default_memory_indexes().keys())
                    missing = sorted(expected - set(schema.keys()))
                    if missing:
                        state.warnings.append(
                            f"Payload schema missing expected indexes: {missing}"
                        )
                except Exception as exc:  # pragma: no cover
                    ok = False
                    checks["payload_schema"] = {"ok": False, "error": str(exc)}

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
                vector_name = await self.qdrant_connector.resolve_vector_name(collection)
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
                    raise ValueError("delete_by_filter requires a filter in strict mode.")
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

        async def list_collections(ctx: Context) -> dict[str, Any]:
            state = new_request(ctx, {})
            collections = await self.qdrant_connector.get_collection_names()
            data = {"collections": collections, "count": len(collections)}
            return finish_request(state, data)

        async def collection_exists(ctx: Context, collection_name: str = "") -> dict[str, Any]:
            state = new_request(ctx, {"collection_name": collection_name})
            name = resolve_collection_name(collection_name)
            exists = await self.qdrant_connector.collection_exists(name)
            data = {"collection_name": name, "exists": exists}
            return finish_request(state, data)

        async def collection_info(ctx: Context, collection_name: str = "") -> dict[str, Any]:
            state = new_request(ctx, {"collection_name": collection_name})
            name = resolve_collection_name(collection_name)
            summary = await self.qdrant_connector.get_collection_summary(name)
            summary["collection_name"] = name
            return finish_request(state, summary)

        async def collection_stats(ctx: Context, collection_name: str = "") -> dict[str, Any]:
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

        async def collection_vectors(ctx: Context, collection_name: str = "") -> dict[str, Any]:
            state = new_request(ctx, {"collection_name": collection_name})
            name = resolve_collection_name(collection_name)
            vectors = await self.qdrant_connector.get_collection_vectors(name)
            data = {"collection_name": name, "vectors": vectors}
            return finish_request(state, data)

        async def collection_payload_schema(ctx: Context, collection_name: str = "") -> dict[str, Any]:
            state = new_request(ctx, {"collection_name": collection_name})
            name = resolve_collection_name(collection_name)
            schema = await self.qdrant_connector.get_collection_payload_schema(name)
            data = {"collection_name": name, "payload_schema": schema}
            return finish_request(state, data)

        async def get_vector_name(ctx: Context, collection_name: str = "") -> dict[str, Any]:
            state = new_request(ctx, {"collection_name": collection_name})
            name = resolve_collection_name(collection_name)
            vector_name = await self.qdrant_connector.resolve_vector_name(name)
            data = {
                "collection_name": name,
                "vector_name": vector_name,
                "label": "(default)" if vector_name is None else vector_name,
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

        async def list_aliases(ctx: Context) -> dict[str, Any]:
            state = new_request(ctx, {})
            aliases = await self.qdrant_connector.list_aliases()
            data = [
                {"alias_name": alias.alias_name, "collection_name": alias.collection_name}
                for alias in aliases
            ]
            return finish_request(state, {"aliases": data, "count": len(data)})

        async def collection_aliases(ctx: Context, collection_name: str = "") -> dict[str, Any]:
            state = new_request(ctx, {"collection_name": collection_name})
            name = resolve_collection_name(collection_name)
            aliases = await self.qdrant_connector.list_collection_aliases(name)
            data = [
                {"alias_name": alias.alias_name, "collection_name": alias.collection_name}
                for alias in aliases
            ]
            return finish_request(
                state,
                {"collection_name": name, "aliases": data, "count": len(data)},
            )

        async def collection_cluster_info(ctx: Context, collection_name: str = "") -> dict[str, Any]:
            state = new_request(ctx, {"collection_name": collection_name})
            name = resolve_collection_name(collection_name)
            info = await self.qdrant_connector.get_collection_cluster_info(name)
            data = info.model_dump() if hasattr(info, "model_dump") else info.dict()
            data["collection_name"] = name
            return finish_request(state, data)

        async def list_snapshots(ctx: Context, collection_name: str = "") -> dict[str, Any]:
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
                ensure_payload_indexes,
                name="qdrant-ensure-payload-indexes",
                description="Ensure expected payload indexes exist for a collection.",
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
                delete_points_foo,
                name="qdrant-delete-points",
                description="Delete points by id (confirm required).",
            )
            self.tool(
                delete_filter_foo,
                name="qdrant-delete-by-filter",
                description="Delete points by filter (confirm required).",
            )
