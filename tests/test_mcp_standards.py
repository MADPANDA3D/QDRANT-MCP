import base64
import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from qdrant_client import models

from mcp_server_qdrant.document_ingest import DocumentSection, ExtractionResult
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.memory import MemoryFilterInput
from mcp_server_qdrant.settings import (
    METADATA_PATH,
    EmbeddingProviderSettings,
    MemorySettings,
    QdrantSettings,
    RequestOverrideSettings,
    ToolSettings,
)
from mcp_server_qdrant.tool_manifest import READ_ONLY_TOOLS


class CountingEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.query_calls = 0
        self.document_calls = 0

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [[0.1, 0.2, 0.3] for _ in documents]

    async def embed_query(self, query: str) -> list[float]:
        self.query_calls += 1
        return [0.1, 0.2, 0.3]

    def get_vector_name(self) -> str:
        return "text"

    def get_vector_size(self) -> int:
        return 3


class FakeSearchConnector:
    def __init__(self) -> None:
        self.query_calls = 0
        self.query_vectors: list[list[float]] = []
        self.collection_names: list[str | None] = []
        self.query_filters: list[models.Filter | None] = []
        self.payload = {
            "document": "This is the full stored document. " * 40,
            METADATA_PATH: {
                "class": "COURSE101",
                "subject": "Example Studies",
                "module": "1",
                "week": "1",
                "status": "active",
                "material_type": "lesson",
                "title": "Module 1 Lesson 1",
                "author": "Example Author",
                "type": "note",
                "scope": "global",
                "source": "test",
                "labels": ["alpha", "beta"],
                "doc_id": "doc-1",
                "unknown_large": "not returned in compact mode",
            },
        }

    async def query_points(
        self,
        query_vector: list[float],
        *,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
        with_vectors: bool = False,
    ) -> list[models.ScoredPoint]:
        self.query_calls += 1
        self.query_vectors.append(query_vector)
        self.collection_names.append(collection_name)
        self.query_filters.append(query_filter)
        return [
            models.ScoredPoint(
                id="point-1",
                version=1,
                score=0.98,
                payload=self.payload,
                vector=None,
            )
        ]

    async def resolve_vector_name(self, collection_name: str) -> str | None:
        return None

    async def retrieve_points(
        self,
        point_ids: list[str],
        *,
        collection_name: str | None = None,
        with_payload: bool = True,
        with_vectors: bool = False,
    ) -> list[models.Record]:
        return [
            models.Record(id=point_id, payload=None, vector=[0.1, 0.2, 0.3])
            for point_id in point_ids
        ]


class FakeContextConnector(FakeSearchConnector):
    def __init__(self, *, empty_filtered: bool = True) -> None:
        super().__init__()
        self.empty_filtered = empty_filtered
        self.payloads = [
            {
                "document": "First document chunk about the example CRM plan. " * 20,
                METADATA_PATH: {
                    "title": "CRM Operating Plan",
                    "doc_id": "doc-1",
                    "source_url": "https://example.test/crm",
                    "status": "active",
                    "type": "note",
                },
            },
            {
                "document": "Second chunk from the same CRM operating plan. " * 20,
                METADATA_PATH: {
                    "title": "CRM Operating Plan",
                    "doc_id": "doc-1",
                    "source_url": "https://example.test/crm",
                    "status": "active",
                    "type": "note",
                },
            },
            {
                "document": "Separate brand memory about positioning and offers. " * 20,
                METADATA_PATH: {
                    "title": "Brand Positioning",
                    "doc_id": "doc-2",
                    "source_url": "https://example.test/brand",
                    "status": "active",
                    "type": "note",
                },
            },
        ]

    async def query_points(
        self,
        query_vector: list[float],
        *,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
        with_vectors: bool = False,
    ) -> list[models.ScoredPoint]:
        self.query_calls += 1
        self.query_vectors.append(query_vector)
        self.collection_names.append(collection_name)
        self.query_filters.append(query_filter)
        if query_filter is not None and self.empty_filtered:
            return []
        points = []
        for index, payload in enumerate(self.payloads[:limit], start=1):
            points.append(
                models.ScoredPoint(
                    id=f"context-{index}",
                    version=1,
                    score=0.99 - (index / 100),
                    payload=payload,
                    vector=[0.1, 0.2, 0.3] if with_vectors else None,
                )
            )
        return points


class FakeSchemaConnector(FakeSearchConnector):
    async def get_collection_summary(self, collection_name: str | None = None) -> dict[str, Any]:
        return {
            "status": "green",
            "optimizer_status": "ok",
            "points_count": 3,
            "vectors": {"text": {"size": 3, "distance": "Cosine"}},
            "payload_schema": await self.get_collection_payload_schema(
                collection_name or "memories"
            ),
        }

    async def get_collection_payload_schema(
        self, collection_name: str | None = None
    ) -> dict[str, str]:
        return {
            "metadata.class": "keyword",
            "metadata.module": "keyword",
            "metadata.status": "keyword",
            "metadata.title": "keyword",
            "metadata.doc_id": "keyword",
            "metadata.text_hash": "keyword",
        }

    async def scroll_points_page(
        self,
        *,
        collection_name: str | None = None,
        query_filter: models.Filter | None = None,
        limit: int = 10,
        with_payload: bool = True,
        with_vectors: bool = False,
        offset: Any | None = None,
    ) -> tuple[list[models.Record], Any | None]:
        payloads = [
            {
                METADATA_PATH: {
                    "class": "COURSE101",
                    "module": "1",
                    "status": "active",
                    "title": "Example Module 1",
                    "doc_id": "example-1",
                    "text_hash": "hash-1",
                }
            },
            {
                METADATA_PATH: {
                    "class": "COURSE303",
                    "module": "2",
                    "status": "archived",
                    "title": "Example Module 2",
                    "doc_id": "example-2",
                    "text_hash": "hash-2",
                }
            },
        ]
        records = [
            models.Record(id=f"sample-{index}", payload=payload, vector=None)
            for index, payload in enumerate(payloads[:limit], start=1)
        ]
        return records, None


class FakeMigrationConnector(FakeSearchConnector):
    def __init__(self) -> None:
        super().__init__()
        self.created_vectors: list[tuple[str, str, int]] = []
        self.updated_vectors: list[models.PointVectors] = []
        self.payload_updates: list[tuple[list[str], dict[str, Any]]] = []
        self.scroll_offsets: list[Any | None] = []

    async def get_collection_summary(self, collection_name: str | None = None) -> dict[str, Any]:
        return {
            "status": "green",
            "optimizer_status": "ok",
            "points_count": 437,
            "vectors": {"(default)": {"size": 1536, "distance": "Cosine"}},
            "payload_schema": {},
        }

    async def scroll_points_page(
        self,
        *,
        collection_name: str | None = None,
        query_filter: models.Filter | None = None,
        limit: int = 10,
        with_payload: bool = True,
        with_vectors: bool = False,
        offset: Any | None = None,
    ) -> tuple[list[models.Record], Any | None]:
        self.scroll_offsets.append(offset)
        if offset is not None:
            return [], None
        records = [
            models.Record(
                id="legacy-1",
                payload={
                    "document": "Legacy barn memory",
                    METADATA_PATH: {
                        "embedding_model": "text-embedding-3-small",
                        "embedding_dim": 1536,
                        "embedding_provider": "openai",
                    },
                },
                vector=None,
            )
        ]
        return records[:limit], None

    async def create_dense_vector_name(
        self,
        *,
        collection_name: str,
        vector_name: str,
        size: int,
        distance: str,
    ) -> None:
        self.created_vectors.append((collection_name, vector_name, size))

    async def update_point_vectors(
        self,
        vectors: list[models.PointVectors],
        *,
        collection_name: str | None = None,
    ) -> None:
        self.updated_vectors.extend(vectors)

    async def set_payload(
        self,
        point_ids: list[str],
        payload: dict[str, Any],
        *,
        collection_name: str | None = None,
    ) -> None:
        self.payload_updates.append((point_ids, payload))


class FakeGovernanceStoreConnector:
    def __init__(self) -> None:
        self.stored_entries: list[Any] = []

    async def scroll_points(
        self,
        *,
        collection_name: str | None = None,
        query_filter: models.Filter | None = None,
        limit: int = 10,
    ) -> list[models.Record]:
        return []

    async def store(self, entry: Any, collection_name: str | None = None) -> str:
        self.stored_entries.append(entry)
        return "stored-1"


class FakeGovernanceBackfillConnector:
    def __init__(self) -> None:
        self.payload_updates: list[tuple[list[str], dict[str, Any]]] = []

    async def scroll_points_page(
        self,
        *,
        collection_name: str | None = None,
        query_filter: models.Filter | None = None,
        limit: int = 10,
        with_payload: bool = True,
        with_vectors: bool = False,
        offset: Any | None = None,
    ) -> tuple[list[models.Record], Any | None]:
        if offset is not None:
            return [], None
        return [
            models.Record(
                id="memory-1",
                payload={
                    "document": "Imported operational checkpoint.",
                    METADATA_PATH: {
                        "text": "Imported operational checkpoint.",
                        "type": "project_checkpoint",
                        "project": "example-project",
                        "source": "import",
                        "scope": "global",
                        "entities": [],
                        "confidence": 0.5,
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "updated_at": "2026-01-01T00:00:00+00:00",
                    },
                },
                vector=None,
            )
        ], None

    async def overwrite_payload(
        self,
        point_ids: list[str],
        payload: dict[str, Any],
        *,
        collection_name: str | None = None,
    ) -> None:
        self.payload_updates.append((point_ids, payload))


class FakeTextbookIngestConnector:
    def __init__(self) -> None:
        self.collections: list[str] = []
        self.indexes: list[dict[str, Any]] = []
        self.upserted_points: list[models.PointStruct] = []

    async def ensure_collection_exists(self, collection_name: str) -> None:
        self.collections.append(collection_name)

    async def ensure_payload_indexes(
        self,
        *,
        collection_name: str | None = None,
        indexes: dict[str, Any] | None = None,
    ) -> None:
        self.indexes.append({"collection_name": collection_name, "indexes": indexes})

    async def count_points(
        self,
        *,
        collection_name: str | None = None,
        query_filter: models.Filter | None = None,
    ) -> int:
        return 0

    async def delete_by_filter(
        self,
        query_filter: models.Filter,
        *,
        collection_name: str | None = None,
    ) -> None:
        raise AssertionError("delete_by_filter should not run for a fresh ingest")

    async def resolve_vector_name(self, collection_name: str) -> str | None:
        return None

    async def upsert_points(
        self,
        points: list[models.PointStruct],
        *,
        collection_name: str | None = None,
    ) -> None:
        self.upserted_points.extend(points)


class FakeSchoolManifestConnector:
    def __init__(self) -> None:
        self.store_calls: list[dict[str, Any]] = []
        self.query_calls: list[dict[str, Any]] = []

    async def collection_exists(self, collection_name: str) -> bool:
        return False

    async def get_collection_payload_schema(
        self, collection_name: str | None = None
    ) -> dict[str, str]:
        return {}

    async def ensure_payload_indexes(
        self,
        *,
        collection_name: str | None = None,
        indexes: dict[str, Any] | None = None,
    ) -> list[str]:
        return list(indexes or {})

    async def count_points(
        self,
        *,
        collection_name: str | None = None,
        query_filter: models.Filter | None = None,
    ) -> int:
        return 0

    async def delete_by_filter(
        self,
        query_filter: models.Filter,
        *,
        collection_name: str | None = None,
    ) -> None:
        raise AssertionError("delete_by_filter should not run for fresh manifest items")

    async def store_entries(
        self,
        entries: list[Any],
        *,
        collection_name: str | None = None,
    ) -> list[str]:
        call_index = len(self.store_calls)
        point_ids = [f"manifest-{call_index}-{index}" for index, _ in enumerate(entries)]
        self.store_calls.append(
            {
                "collection_name": collection_name,
                "entries": entries,
                "point_ids": point_ids,
            }
        )
        return point_ids

    async def resolve_vector_name(self, collection_name: str) -> str | None:
        return None

    async def query_points(
        self,
        query_vector: list[float],
        *,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
        with_vectors: bool = False,
    ) -> list[models.ScoredPoint]:
        self.query_calls.append(
            {
                "collection_name": collection_name,
                "limit": limit,
                "query_filter": query_filter,
            }
        )
        doc_id = None
        if query_filter is not None:
            for condition in query_filter.must or []:
                if condition.key == "metadata.doc_id":
                    doc_id = condition.match.value
                    break
        for store_call in self.store_calls:
            for point_id, entry in zip(store_call["point_ids"], store_call["entries"]):
                metadata = entry.metadata or {}
                if doc_id is None or metadata.get("doc_id") == doc_id:
                    return [
                        models.ScoredPoint(
                            id=point_id,
                            version=1,
                            score=0.97,
                            payload={
                                "document": entry.content,
                                METADATA_PATH: metadata,
                            },
                            vector=None,
                        )
                    ]
        return []


class FakeHealthConnector:
    async def get_collection_names(self) -> list[str]:
        return ["study", "knowledgebase", "memories"]

    async def collection_exists(self, collection_name: str) -> bool:
        return False


class FakeUrlopenResponse:
    def __init__(self, data: bytes, content_type: str | None) -> None:
        self._data = data
        self._offset = 0
        self.headers = {"Content-Type": content_type} if content_type else {}

    def __enter__(self) -> "FakeUrlopenResponse":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def close(self) -> None:
        return None

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = len(self._data) - self._offset
        chunk = self._data[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk


def make_server(
    *,
    collection_name: str | None = "memories",
    read_only: bool = False,
    memory_settings: MemorySettings | None = None,
) -> tuple[QdrantMCPServer, CountingEmbeddingProvider]:
    provider = CountingEmbeddingProvider()
    server = QdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=QdrantSettings.model_validate(
            {
                "QDRANT_LOCAL_PATH": ":memory:",
                "COLLECTION_NAME": collection_name,
                "QDRANT_READ_ONLY": read_only,
            }
        ),
        request_override_settings=RequestOverrideSettings(),
        memory_settings=memory_settings or MemorySettings(),
        embedding_provider=provider,
    )
    return server, provider


def test_every_registered_tool_has_required_annotations() -> None:
    server, _ = make_server()
    tools = server._registered_tools.values()  # pylint: disable=protected-access

    for tool in tools:
        annotations = tool.annotations
        assert annotations is not None, tool.name
        assert annotations.readOnlyHint is not None, tool.name
        assert annotations.destructiveHint is not None, tool.name
        assert annotations.openWorldHint is not None, tool.name
        if annotations.readOnlyHint:
            assert annotations.idempotentHint is True, tool.name

    delete_tool = server._registered_tools["qdrant-delete-points"]
    assert delete_tool.annotations.destructiveHint is True
    find_tool = server._registered_tools["qdrant-find"]
    assert find_tool.annotations.readOnlyHint is True
    assert find_tool.annotations.destructiveHint is False

    for tool in tools:
        for property_name, property_schema in tool.parameters.get("properties", {}).items():
            assert property_schema.get("description"), f"{tool.name}.{property_name}"


def test_read_only_registry_matches_the_explicit_allowlist_exactly() -> None:
    server, _ = make_server(read_only=True)

    assert set(server._registered_tools) == READ_ONLY_TOOLS
    assert "qdrant-submit-job" not in server._registered_tools
    assert "qdrant-create-snapshot" not in server._registered_tools
    assert "qdrant-restore-snapshot" not in server._registered_tools


@pytest.mark.asyncio
async def test_read_only_dispatch_rejects_stale_mutating_tool_invocations() -> None:
    server, _ = make_server(read_only=True)

    with pytest.raises(ValueError, match="unavailable in read-only mode"):
        await server._call_tool_mcp(  # pylint: disable=protected-access
            "qdrant-submit-job",
            {"job_type": "audit-memories", "job_args": {}},
        )

    with pytest.raises(ValueError, match="unavailable in read-only mode"):
        await server._call_tool_mcp(  # pylint: disable=protected-access
            "qdrant-restore-snapshot",
            {},
        )


@pytest.mark.asyncio
async def test_background_job_failures_redact_exception_material(
    tmp_path: Path,
) -> None:
    sensitive_value = "SENSITIVE_VALUE_123"
    synthetic_private_path = "/" + "home/example/private-job.json"
    sensitive_error = (
        f"credential={sensitive_value} "
        f"https://private.example/job?credential={sensitive_value} "
        f"{synthetic_private_path}"
    )

    class FailingJobConnector:
        async def scroll_points_page(self, **_kwargs: Any) -> tuple[list[Any], None]:
            raise RuntimeError(sensitive_error)

    server, _ = make_server(
        memory_settings=MemorySettings.model_validate({"MCP_TEXTBOOK_JOB_STATE_DIR": str(tmp_path)})
    )
    server._default_qdrant_connector = cast(  # pylint: disable=protected-access
        Any,
        FailingJobConnector(),
    )
    ctx = SimpleNamespace(request_id="redacted-job-failure-test")

    submitted = await server._registered_tools[  # pylint: disable=protected-access
        "qdrant-submit-job"
    ].fn(
        ctx,
        job_type="audit-memories",
        job_args={"collection_name": "memories"},
    )
    job_id = submitted["data"]["job_id"]
    task = server._job_tasks[job_id]  # pylint: disable=protected-access
    await task
    status = await server._registered_tools[  # pylint: disable=protected-access
        "qdrant-job-status"
    ].fn(ctx, job_id=job_id)
    logs = await server._registered_tools[  # pylint: disable=protected-access
        "qdrant-job-logs"
    ].fn(ctx, job_id=job_id)

    persisted = (tmp_path / f"{job_id}.json").read_text(encoding="utf-8")
    serialized = f"{status!r}\n{logs!r}\n{persisted}"
    assert status["data"]["status"] == "failed"
    assert status["data"]["error"] == "Background job failed safely."
    assert status["data"]["structured_error"]["suggested_http_status"] == 500
    assert sensitive_value not in serialized
    assert "private.example" not in serialized
    assert synthetic_private_path not in serialized


@pytest.mark.asyncio
async def test_validate_memory_returns_governance_recommendation() -> None:
    server, _ = make_server()
    validate_tool = server._registered_tools["qdrant-validate-memory"]

    response = await validate_tool.fn(
        SimpleNamespace(request_id="governance-validation-test"),
        information="Example project checkpoint.",
        metadata={
            "type": "project_checkpoint",
            "project": "example-project",
            "source": "manual",
            "scope": "global",
            "entities": [],
            "confidence": 0.5,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
    )

    governance = response["data"]["governance"]
    assert governance["recommended_metadata"]["knowledge_domain"] == "project_memory"
    assert governance["recommended_metadata"]["trust_level"] == "needs_review"
    assert governance["missing_governance_fields"] == [
        "knowledge_domain",
        "knowledge_role",
        "trust_level",
        "retrieval_tier",
        "lifecycle_state",
        "evaluation_status",
        "evaluation_sequence",
        "review_status",
    ]


@pytest.mark.asyncio
async def test_ingest_with_validation_can_apply_governance_metadata() -> None:
    server, _ = make_server()
    connector = FakeGovernanceStoreConnector()
    server._default_qdrant_connector = cast(  # pylint: disable=protected-access
        Any,
        connector,
    )
    ingest_tool = server._registered_tools["qdrant-ingest-with-validation"]

    response = await ingest_tool.fn(
        SimpleNamespace(request_id="governance-ingest-test"),
        information="Example project checkpoint.",
        collection_name="example-knowledge-base",
        metadata={
            "type": "project_checkpoint",
            "project": "example-project",
            "source": "manual",
            "scope": "global",
            "entities": [],
            "confidence": 0.5,
            "created_at": "2026-01-01T00:00:00+00:00",
            "updated_at": "2026-01-01T00:00:00+00:00",
        },
        apply_governance_metadata=True,
    )

    assert response["data"]["status"] == "stored"
    stored_metadata = connector.stored_entries[0].metadata
    assert stored_metadata["knowledge_domain"] == "project_memory"
    assert stored_metadata["knowledge_role"] == "checkpoint"
    assert stored_metadata["trust_level"] == "needs_review"
    assert stored_metadata["evaluation_status"] == "pending_review"


@pytest.mark.asyncio
async def test_backfill_memory_contract_dry_run_can_include_governance_patch() -> None:
    server, _ = make_server()
    connector = FakeGovernanceBackfillConnector()
    server._default_qdrant_connector = cast(  # pylint: disable=protected-access
        Any,
        connector,
    )
    backfill_tool = server._registered_tools["qdrant-backfill-memory-contract"]

    response = await backfill_tool.fn(
        SimpleNamespace(request_id="governance-backfill-test"),
        collection_name="example-knowledge-base",
        dry_run=True,
        include_governance_metadata=True,
    )

    data = response["data"]
    assert data["updated"] == 1
    assert connector.payload_updates == []
    sample = data["dry_run_diff"]["samples"][0]
    assert sample["after"]["metadata"]["knowledge_domain"] == "project_memory"
    assert sample["after"]["metadata"]["trust_level"] == "needs_review"


@pytest.mark.asyncio
async def test_uploaded_textbook_session_submits_and_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _ = make_server(
        memory_settings=MemorySettings.model_validate({"MCP_TEXTBOOK_JOB_STATE_DIR": str(tmp_path)})
    )
    connector = FakeTextbookIngestConnector()
    server._default_qdrant_connector = cast(  # pylint: disable=protected-access
        Any,
        connector,
    )
    monkeypatch.setattr("mcp_server_qdrant.mcp_server.get_pdf_page_count", lambda _: 1)
    monkeypatch.setattr(
        "mcp_server_qdrant.mcp_server.extract_document_sections",
        lambda *_args, **_kwargs: ExtractionResult(
            sections=[
                DocumentSection(
                    text="Chapter 1: Sample Topic\nFoundations and methods.",
                    page_start=1,
                    page_end=1,
                )
            ],
            page_count=1,
            coverage_ratio=1.0,
            coverage_pages_total=1,
            coverage_pages_meeting_threshold=1,
        ),
    )

    pdf_bytes = b"%PDF-1.7\nfake local textbook bytes\n%%EOF"
    expected_hash = hashlib.sha256(pdf_bytes).hexdigest()
    ctx = SimpleNamespace(request_id="local-textbook-upload-test")

    started = await server._registered_tools.get(  # pylint: disable=protected-access
        "qdrant-start-file-upload"
    ).fn(
        ctx,
        file_name="COURSE202.pdf",
        expected_size=len(pdf_bytes),
        expected_sha256=expected_hash,
        purpose="textbook",
    )
    upload_id = started["data"]["upload_id"]

    appended = await server._registered_tools.get(  # pylint: disable=protected-access
        "qdrant-append-file-upload"
    ).fn(
        ctx,
        upload_id=upload_id,
        chunk_base64=base64.b64encode(pdf_bytes).decode("ascii"),
        offset=0,
    )
    assert appended["data"]["bytes_received"] == len(pdf_bytes)

    finished = await server._registered_tools.get(  # pylint: disable=protected-access
        "qdrant-finish-file-upload"
    ).fn(ctx, upload_id=upload_id)
    upload_uri = finished["data"]["uploaded_file_uri"]
    assert upload_uri == f"upload://{upload_id}"
    assert finished["data"]["upload_uri"] == upload_uri
    assert finished["data"]["source_url"] == upload_uri
    assert finished["data"]["uri"] == upload_uri

    finalized_again = await server._registered_tools.get(  # pylint: disable=protected-access
        "qdrant-finish-file-upload"
    ).fn(ctx, upload_id=upload_id)
    assert finalized_again["data"]["uploaded_file_uri"] == upload_uri
    assert finalized_again["data"]["upload_uri"] == upload_uri
    assert finalized_again["data"]["source_url"] == upload_uri
    assert finalized_again["data"]["uri"] == upload_uri

    submitted = await server._registered_tools.get(  # pylint: disable=protected-access
        "qdrant-ingest-textbook"
    ).fn(
        ctx,
        collection_name="school",
        source_url=upload_uri,
        metadata={
            "class": "COURSE202",
            "material_type": "textbook",
            "title": "Synthetic Course Reader",
            "author": "Example Author",
            "edition": "2",
            "isbn": "0000000000000",
        },
        ocr=False,
    )
    assert submitted["data"]["status"] == "queued"

    job_id = submitted["data"]["job_id"]
    task = server._job_tasks[job_id]  # pylint: disable=protected-access
    await task
    status = await server._registered_tools.get(  # pylint: disable=protected-access
        "qdrant-get-ingest-status"
    ).fn(ctx, job_id=job_id)

    result = status["data"]["result"]
    assert status["data"]["status"] == "completed", status["data"]
    assert result["source_url"] == upload_uri
    assert result["chunks_count"] == 1
    assert connector.collections == ["school"]
    assert len(connector.upserted_points) == 1
    assert upload_id not in server._file_upload_sessions  # pylint: disable=protected-access
    upload_dir = tmp_path / "uploads"
    assert not any(upload_dir.iterdir())


@pytest.mark.asyncio
async def test_upload_bridge_advertises_and_enforces_broker_safe_chunk_limit(
    tmp_path: Path,
) -> None:
    server, _ = make_server(
        memory_settings=MemorySettings.model_validate({"MCP_TEXTBOOK_JOB_STATE_DIR": str(tmp_path)})
    )
    ctx = SimpleNamespace(request_id="upload-chunk-limit-test")

    configuration = await server._registered_tools.get(  # pylint: disable=protected-access
        "qdrant-check-configuration"
    ).fn(ctx)
    assert configuration["data"]["file_uploads"]["max_chunk_bytes"] == 512 * 1024

    started = await server._registered_tools.get(  # pylint: disable=protected-access
        "qdrant-start-file-upload"
    ).fn(
        ctx,
        file_name="COURSE202.pdf",
        expected_size=(512 * 1024) + 1,
        purpose="textbook",
    )
    assert started["data"]["max_chunk_bytes"] == 512 * 1024

    with pytest.raises(ValueError, match="decoded upload chunk exceeds"):
        await server._registered_tools.get(  # pylint: disable=protected-access
            "qdrant-append-file-upload"
        ).fn(
            ctx,
            upload_id=started["data"]["upload_id"],
            chunk_base64=base64.b64encode(b"x" * ((512 * 1024) + 1)).decode("ascii"),
            offset=0,
        )


@pytest.mark.asyncio
async def test_textbook_file_url_rejection_points_to_upload_bridge() -> None:
    server, _ = make_server()
    response = await server._registered_tools.get(  # pylint: disable=protected-access
        "qdrant-ingest-textbook"
    ).fn(
        SimpleNamespace(request_id="file-url-rejection-test"),
        collection_name="school",
        source_url="file:///tmp/example.pdf",
        metadata={
            "class": "COURSE202",
            "material_type": "textbook",
            "title": "Synthetic Course Reader",
            "author": "Example Author",
            "edition": "2",
            "isbn": "0000000000000",
        },
    )

    assert response["data"]["status"] == "rejected"
    message = response["data"]["error"]["message"]
    assert "qdrant-start-file-upload" in message
    assert "upload://" in message


@pytest.mark.asyncio
async def test_textbook_drive_pdf_url_without_extension_completes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server, _ = make_server(
        memory_settings=MemorySettings.model_validate(
            {
                "MCP_TEXTBOOK_JOB_STATE_DIR": str(tmp_path),
                "MCP_OUTBOUND_HOST_ALLOWLIST": ["drive.google.com"],
            }
        )
    )
    connector = FakeTextbookIngestConnector()
    server._default_qdrant_connector = cast(  # pylint: disable=protected-access
        Any,
        connector,
    )
    pdf_bytes = b"%PDF-1.7\nfake drive textbook bytes\n%%EOF"

    def fake_urlopen(*_args: object, **_kwargs: object) -> FakeUrlopenResponse:
        return FakeUrlopenResponse(pdf_bytes, "application/octet-stream")

    monkeypatch.setattr(
        "mcp_server_qdrant.safe_fetch._open_without_redirects",
        fake_urlopen,
    )
    monkeypatch.setattr(
        "mcp_server_qdrant.safe_fetch.socket.getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("93.184.216.34", 443))],
    )
    monkeypatch.setattr("mcp_server_qdrant.mcp_server.get_pdf_page_count", lambda _: 1)
    monkeypatch.setattr(
        "mcp_server_qdrant.mcp_server.extract_document_sections",
        lambda *_args, **_kwargs: ExtractionResult(
            sections=[
                DocumentSection(
                    text="Chapter 1: Sample Topic\nFoundations and methods.",
                    page_start=1,
                    page_end=1,
                )
            ],
            page_count=1,
            coverage_ratio=1.0,
            coverage_pages_total=1,
            coverage_pages_meeting_threshold=1,
        ),
    )

    ctx = SimpleNamespace(request_id="drive-textbook-url-test")
    submitted = await server._registered_tools.get(  # pylint: disable=protected-access
        "qdrant-ingest-textbook"
    ).fn(
        ctx,
        collection_name="school",
        source_url="https://drive.google.com/uc?id=public-file&export=download",
        metadata={
            "class": "COURSE202",
            "material_type": "textbook",
            "title": "Synthetic Course Reader",
            "author": "Example Author",
            "edition": "2",
            "isbn": "0000000000000",
        },
        ocr=False,
    )
    assert submitted["data"]["status"] == "queued"

    job_id = submitted["data"]["job_id"]
    task = server._job_tasks[job_id]  # pylint: disable=protected-access
    await task
    status = await server._registered_tools.get(  # pylint: disable=protected-access
        "qdrant-get-ingest-status"
    ).fn(ctx, job_id=job_id)

    assert status["data"]["status"] == "completed", status["data"]
    result = status["data"]["result"]
    assert result["chunks_count"] == 1
    assert len(connector.upserted_points) == 1
    stored_metadata = connector.upserted_points[0].payload[METADATA_PATH]
    assert stored_metadata["file_type"] == "pdf"


@pytest.mark.asyncio
async def test_compact_search_default_excludes_payload_and_cache_reuses_embedding() -> None:
    server, provider = make_server()
    connector = FakeSearchConnector()
    server._default_qdrant_connector = cast(  # pylint: disable=protected-access
        Any,
        connector,
    )
    find_tool = server._registered_tools["qdrant-find"]
    ctx = SimpleNamespace(request_id="search-test")

    compact = await find_tool.fn(ctx, query="same query", top_k=3)
    result = compact["data"]["results"][0]
    assert result["id"] == "point-1"
    assert result["score"] == 0.98
    assert len(result["snippet"]) <= 240
    assert "payload" not in result
    assert "document" not in result
    assert result["metadata"] == {
        "class": "COURSE101",
        "subject": "Example Studies",
        "module": "1",
        "week": "1",
        "status": "active",
        "material_type": "lesson",
        "title": "Module 1 Lesson 1",
        "author": "Example Author",
        "type": "note",
        "labels": ["alpha", "beta"],
        "doc_id": "doc-1",
    }

    payload = await find_tool.fn(
        ctx,
        query="same query",
        top_k=3,
        response_mode="payload",
    )
    payload_result = payload["data"]["results"][0]
    assert payload_result["payload"] == connector.payload
    assert provider.query_calls == 1
    assert provider.document_calls == 0
    assert connector.query_calls == 2


@pytest.mark.asyncio
async def test_study_search_defaults_to_study_collection_and_compact_filters() -> None:
    server, _ = make_server()
    connector = FakeSearchConnector()
    server._default_qdrant_connector = cast(  # pylint: disable=protected-access
        Any,
        connector,
    )
    study_tool = server._registered_tools["qdrant-study-search"]

    response = await study_tool.fn(
        SimpleNamespace(request_id="study-test"),
        query="forced retrieval melody",
        class_code="COURSE101",
        subject="Example Studies",
        module="1",
        week="1",
        status="active",
        material_type="lesson",
        author="Example Author",
        top_k=3,
    )

    assert connector.collection_names == ["study"]
    assert connector.query_filters[0] is not None
    filter_keys = {condition.key for condition in connector.query_filters[0].must}
    assert "metadata.class" in filter_keys
    assert "metadata.subject" in filter_keys
    assert "metadata.module" in filter_keys
    assert "metadata.week" in filter_keys
    assert "metadata.status" in filter_keys
    assert "metadata.material_type" in filter_keys
    assert "metadata.author" in filter_keys
    result = response["data"]["results"][0]
    assert "payload" not in result
    assert result["metadata"]["class"] == "COURSE101"


@pytest.mark.asyncio
async def test_school_manifest_ingests_batch_and_verifies_items() -> None:
    server, _ = make_server()
    connector = FakeSchoolManifestConnector()
    server._default_qdrant_connector = cast(  # pylint: disable=protected-access
        Any,
        connector,
    )
    manifest_tool = server._registered_tools["qdrant-ingest-school-manifest"]
    alias_tool = server._registered_tools["qdrant-ingest-class-manifest"]
    generic_alias_tool = server._registered_tools["qdrant-ingest-manifest"]
    assert alias_tool.parameters == manifest_tool.parameters
    assert generic_alias_tool.parameters == manifest_tool.parameters

    response = await manifest_tool.fn(
        SimpleNamespace(request_id="school-manifest-test"),
        default_metadata={
            "subject": "Example Studies",
            "source": "browser_capture",
            "status": "active",
        },
        manifest=[
            {
                "id": "module-overview",
                "class_code": "COURSE202",
                "module": "1",
                "week": "1",
                "material_type": "class_capture",
                "title": "COURSE202 Module 1 Overview",
                "file_name": "module-1-overview.md",
                "text": "COURSE202 module one overview and required reading notes.",
            },
            {
                "id": "discussion-rubric",
                "class_code": "COURSE202",
                "module": "1",
                "week": "1",
                "material_type": "rubric",
                "title": "COURSE202 Discussion Rubric",
                "text": "Discussion rubric with peer response expectations.",
            },
        ],
    )

    data = response["data"]
    assert data["status"] == "completed"
    assert data["collection_name"] == "study"
    assert data["items_total"] == 2
    assert data["ingested"] == 2
    assert data["verified"] == 2
    assert len(data["results"]) == 2
    assert {item["verification"]["status"] for item in data["results"]} == {"passed"}

    assert [call["collection_name"] for call in connector.store_calls] == [
        "study",
        "study",
    ]
    first_metadata = connector.store_calls[0]["entries"][0].metadata
    assert first_metadata["class"] == "COURSE202"
    assert first_metadata["subject"] == "Example Studies"
    assert first_metadata["source"] == "browser_capture"
    assert first_metadata["module"] == "1"
    assert first_metadata["week"] == "1"
    assert first_metadata["material_type"] == "class_capture"
    assert first_metadata["title"] == "COURSE202 Module 1 Overview"
    assert first_metadata["manifest_item_id"] == "module-overview"

    assert len(connector.query_calls) == 2
    first_filter = connector.query_calls[0]["query_filter"]
    assert first_filter is not None
    filter_keys = {condition.key for condition in first_filter.must}
    assert "metadata.doc_id" in filter_keys
    assert "metadata.class" in filter_keys
    assert "metadata.module" in filter_keys
    assert "metadata.week" in filter_keys
    assert "metadata.material_type" in filter_keys


@pytest.mark.asyncio
async def test_find_supports_metadata_fields_grouping_and_budget() -> None:
    server, _ = make_server()
    connector = FakeContextConnector(empty_filtered=False)
    server._default_qdrant_connector = cast(  # pylint: disable=protected-access
        Any,
        connector,
    )
    find_tool = server._registered_tools["qdrant-find"]

    response = await find_tool.fn(
        SimpleNamespace(request_id="find-shape-test"),
        query="crm operating plan",
        top_k=3,
        metadata_fields=["title", "doc_id"],
        group_by_doc=True,
        max_chunks_per_doc=1,
        max_output_chars=1600,
    )

    results = response["data"]["results"]
    assert len(results) == 2
    assert [result["metadata"]["doc_id"] for result in results] == ["doc-1", "doc-2"]
    assert set(results[0]["metadata"].keys()) == {"title", "doc_id"}
    assert "payload" not in results[0]
    assert response["meta"]["budget"]["max_output_chars"] == 1600


@pytest.mark.asyncio
async def test_build_context_relaxes_filters_dedupes_and_returns_citations() -> None:
    server, provider = make_server()
    connector = FakeContextConnector(empty_filtered=True)
    server._default_qdrant_connector = cast(  # pylint: disable=protected-access
        Any,
        connector,
    )
    context_tool = server._registered_tools["qdrant-build-context"]

    response = await context_tool.fn(
        SimpleNamespace(request_id="context-test"),
        query="crm operating plan",
        memory_filter=MemoryFilterInput.model_validate({"status": "missing"}),
        top_k=3,
        metadata_fields=["title", "doc_id", "source_url"],
        group_by_doc=True,
        max_chunks_per_doc=1,
        max_output_chars=4000,
    )

    data = response["data"]
    assert data["fallback_used"] is True
    assert [attempt["strategy"] for attempt in data["search_attempts"]] == [
        "primary",
        "relax_filters",
    ]
    assert len(data["context"]) == 2
    assert data["context"][0]["citation"] == "[1]"
    assert "[1] CRM Operating Plan" in data["context_text"]
    assert {item["metadata"]["doc_id"] for item in data["context"]} == {
        "doc-1",
        "doc-2",
    }
    assert "payload" not in data["context"][0]
    assert provider.query_calls == 1
    assert connector.query_calls == 2
    assert response["meta"]["budget"]["max_output_chars"] == 4000


@pytest.mark.asyncio
async def test_recommend_memories_compact_default_excludes_payload() -> None:
    server, _ = make_server()
    connector = FakeSearchConnector()
    server._default_qdrant_connector = cast(  # pylint: disable=protected-access
        Any,
        connector,
    )
    recommend_tool = server._registered_tools["qdrant-recommend-memories"]

    response = await recommend_tool.fn(
        SimpleNamespace(request_id="recommend-test"),
        positive_ids=["positive-1"],
        top_k=3,
    )

    result = response["data"]["results"][0]
    assert "payload" not in result
    assert result["metadata"]["title"] == "Module 1 Lesson 1"
    assert "source" not in result["metadata"]


@pytest.mark.asyncio
async def test_navigation_tools_return_stable_compact_outputs() -> None:
    server, _ = make_server()
    ctx = SimpleNamespace(request_id="navigation-test")

    configuration = await server._registered_tools.get(  # pylint: disable=protected-access
        "qdrant-check-configuration"
    ).fn(ctx)
    assert configuration["data"]["service"] == "qdrant-mcp"
    assert configuration["data"]["credential_mode"] is None
    assert configuration["data"]["query_embedding_cache"]["size"] == 256

    capabilities = await server._registered_tools.get(  # pylint: disable=protected-access
        "qdrant-list-capabilities"
    ).fn(ctx)
    assert capabilities["data"]["tool_count"] >= 1
    assert any(group["name"] == "navigation" for group in capabilities["data"]["groups"])

    coverage = await server._registered_tools.get(  # pylint: disable=protected-access
        "qdrant-get-endpoint-coverage"
    ).fn(ctx)
    assert coverage["data"]["source"] == "docs/endpoint-coverage.md"
    assert coverage["data"]["summary"]["points"] == "covered for memory workflows"

    usage = await server._registered_tools.get(  # pylint: disable=protected-access
        "qdrant-get-tool-usage"
    ).fn(ctx, tool_name="qdrant-build-context")
    assert usage["data"]["name"] == "qdrant-build-context"
    assert usage["data"]["annotations"]["readOnlyHint"] is True
    assert "top_k=3-5" in usage["data"]["guidance"]

    alias_usage = await server._registered_tools.get(  # pylint: disable=protected-access
        "qdrant-get-tool-usage"
    ).fn(ctx, tool="qdrant-find")
    assert alias_usage["data"]["name"] == "qdrant-find"

    upload_usage = await server._registered_tools.get(  # pylint: disable=protected-access
        "qdrant-get-tool-usage"
    ).fn(ctx, tool_name="qdrant-finish-file-upload")
    assert "upload_uri" in upload_usage["data"]["guidance"]
    assert "source_url" in upload_usage["data"]["guidance"]

    with pytest.raises(ValueError, match="tool_name or tool is required"):
        await server._registered_tools.get(  # pylint: disable=protected-access
            "qdrant-get-tool-usage"
        ).fn(ctx)


@pytest.mark.asyncio
async def test_embedding_migration_dry_run_reports_legacy_collection_gap() -> None:
    server, _ = make_server()
    connector = FakeMigrationConnector()
    server._default_qdrant_connector = cast(  # pylint: disable=protected-access
        Any,
        connector,
    )

    response = await server._registered_tools.get(  # pylint: disable=protected-access
        "qdrant-migrate-collection-embedding"
    ).fn(
        SimpleNamespace(request_id="migration-test"),
        collection_name="sample-collection",
        max_points=1,
        dry_run=True,
    )

    data = response["data"]
    assert data["collection_name"] == "sample-collection"
    assert data["target_vector_name"] == "text"
    assert data["target_vector_size"] == 3
    assert data["needs_vector_schema"] is True
    assert data["scanned"] == 1
    assert data["would_update"] == 1
    assert data["dry_run"] is True
    assert connector.created_vectors == []
    assert connector.updated_vectors == []
    assert connector.scroll_offsets == [None]


@pytest.mark.asyncio
async def test_embedding_migration_accepts_scroll_offset() -> None:
    server, _ = make_server()
    connector = FakeMigrationConnector()
    server._default_qdrant_connector = cast(  # pylint: disable=protected-access
        Any,
        connector,
    )

    response = await server._registered_tools.get(  # pylint: disable=protected-access
        "qdrant-migrate-collection-embedding"
    ).fn(
        SimpleNamespace(request_id="migration-offset-test"),
        collection_name="sample-collection",
        offset="legacy-1",
        max_points=1,
        dry_run=True,
    )

    data = response["data"]
    assert data["offset"] == "legacy-1"
    assert data["scanned"] == 0
    assert connector.scroll_offsets == ["legacy-1"]


@pytest.mark.asyncio
async def test_collection_navigation_describes_schema_and_suggests_filters() -> None:
    server, _ = make_server()
    connector = FakeSchemaConnector()
    server._default_qdrant_connector = cast(  # pylint: disable=protected-access
        Any,
        connector,
    )
    ctx = SimpleNamespace(request_id="collection-navigation-test")

    describe = await server._registered_tools.get(  # pylint: disable=protected-access
        "qdrant-describe-collection"
    ).fn(ctx)
    assert describe["data"]["collection_name"] == "memories"
    assert describe["data"]["points_count"] == 3
    assert "class" in describe["data"]["known_memory_filter_fields"]

    schema = await server._registered_tools.get(  # pylint: disable=protected-access
        "qdrant-summarize-collection-schema"
    ).fn(ctx)
    assert schema["data"]["memory_filter_keys"]["class"] == "class_code"
    assert "metadata.status" in schema["data"]["indexed_fields"]

    suggested = await server._registered_tools.get(  # pylint: disable=protected-access
        "qdrant-suggest-filters"
    ).fn(ctx, query="Need COURSE101 module 1 active context")
    assert suggested["data"]["recommended_filters"] == {
        "class_code": "COURSE101",
        "module": "1",
        "status": "active",
    }
    field_names = {field["field"] for field in suggested["data"]["fields"]}
    assert {"class", "module", "status"}.issubset(field_names)
    assert "text_hash" not in field_names
    assert all(len(field["sample_values"]) <= 4 for field in suggested["data"]["fields"])


@pytest.mark.asyncio
async def test_health_check_suggests_close_collection_names() -> None:
    server, _ = make_server(collection_name="knowledge-base")
    server._default_qdrant_connector = cast(  # pylint: disable=protected-access
        Any,
        FakeHealthConnector(),
    )
    health_tool = server._registered_tools["qdrant-health-check"]

    response = await health_tool.fn(SimpleNamespace(request_id="health-test"))

    check = response["data"]["checks"]["collection_exists"]
    assert check["ok"] is False
    assert check["collection_name"] == "knowledge-base"
    assert "knowledgebase" in check["suggested_collections"]


def test_env_example_and_endpoint_coverage_docs_are_present(monkeypatch) -> None:
    env_example = Path(".env.example").read_text()
    required_names = {
        "MCP_PORTAL_GRANT_TOKEN",
        "MCP_PORTAL_GRANT_HEADER",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "COLLECTION_NAME",
        "MCP_REQUIRE_REQUEST_COLLECTION",
        "MCP_DISABLE_DEFAULT_EMBEDDING_FALLBACK",
        "MCP_QDRANT_URL_HEADER",
        "MCP_QDRANT_API_KEY_HEADER",
        "MCP_COLLECTION_NAME_HEADER",
        "MCP_EMBEDDING_PROVIDER_HEADER",
        "MCP_EMBEDDING_MODEL_HEADER",
        "MCP_OPENAI_API_KEY_HEADER",
        "FASTEMBED_MODEL_PATH",
        "FASTEMBED_MODEL_REVISION",
        "MCP_QUERY_EMBEDDING_CACHE_SIZE",
        "MCP_QUERY_EMBEDDING_CACHE_TTL_SECONDS",
        "MCP_STUDY_COLLECTION",
        "MCP_HOST_PORT",
    }
    for name in required_names:
        assert f"{name}=" in env_example
    for line in env_example.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        monkeypatch.setenv(key, value)
    QdrantSettings()
    EmbeddingProviderSettings()
    MemorySettings()
    RequestOverrideSettings()

    endpoint_coverage = Path("docs/endpoint-coverage.md")
    assert endpoint_coverage.exists()
    assert "qdrant-find" in endpoint_coverage.read_text()
