from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from qdrant_client import models
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.hosted_server import PortalGrantMiddleware, validate_portal_grant
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
                "class": "MUS327",
                "subject": "World Music",
                "module": "1",
                "week": "1",
                "status": "active",
                "material_type": "lesson",
                "title": "Module 1 Lesson 1",
                "author": "Leo Lara",
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
                "document": "First document chunk about the user's CRM plan. " * 20,
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
    async def get_collection_summary(
        self, collection_name: str | None = None
    ) -> dict[str, Any]:
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
                    "class": "MUS327",
                    "module": "1",
                    "status": "active",
                    "title": "Music Module 1",
                    "doc_id": "music-1",
                    "text_hash": "hash-1",
                }
            },
            {
                METADATA_PATH: {
                    "class": "ILR260",
                    "module": "2",
                    "status": "archived",
                    "title": "Labor Relations Module 2",
                    "doc_id": "labor-2",
                    "text_hash": "hash-2",
                }
            },
        ]
        records = [
            models.Record(id=f"sample-{index}", payload=payload, vector=None)
            for index, payload in enumerate(payloads[:limit], start=1)
        ]
        return records, None


class FakeHealthConnector:
    async def get_collection_names(self) -> list[str]:
        return ["school", "jarvis-knowledgebase", "memories"]

    async def collection_exists(self, collection_name: str) -> bool:
        return False


def make_server(
    *,
    collection_name: str | None = "memories",
    portal_grant_token: str | None = "portal-secret",
    memory_settings: MemorySettings | None = None,
) -> tuple[QdrantMCPServer, CountingEmbeddingProvider]:
    provider = CountingEmbeddingProvider()
    server = QdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=QdrantSettings.model_validate(
            {
                "QDRANT_LOCAL_PATH": ":memory:",
                "COLLECTION_NAME": collection_name,
            }
        ),
        request_override_settings=RequestOverrideSettings.model_validate(
            {"MCP_PORTAL_GRANT_TOKEN": portal_grant_token}
        ),
        memory_settings=memory_settings or MemorySettings(),
        embedding_provider=provider,
    )
    return server, provider


def make_test_app(grant_token: str | None) -> Starlette:
    async def health(_: Any) -> JSONResponse:
        return JSONResponse({"ok": True})

    async def mcp(_: Any) -> JSONResponse:
        return JSONResponse({"ok": True})

    return Starlette(
        routes=[Route("/health", health), Route("/mcp", mcp)],
        middleware=[
            Middleware(
                PortalGrantMiddleware,
                grant_token=grant_token,
                grant_header="x-madpanda-portal-grant",
            )
        ],
    )


def test_portal_grant_validation_fails_closed_without_leaking_token() -> None:
    token = "super-secret-token"
    assert (
        validate_portal_grant(
            headers={},
            expected_token=None,
            header_name="x-madpanda-portal-grant",
        )
        == "portal_grant_not_configured"
    )
    assert (
        validate_portal_grant(
            headers={},
            expected_token=token,
            header_name="x-madpanda-portal-grant",
        )
        == "missing_portal_grant"
    )
    assert (
        validate_portal_grant(
            headers={"x-madpanda-portal-grant": "wrong"},
            expected_token=token,
            header_name="x-madpanda-portal-grant",
        )
        == "invalid_portal_grant"
    )
    assert (
        validate_portal_grant(
            headers={"x-madpanda-portal-grant": token},
            expected_token=token,
            header_name="x-madpanda-portal-grant",
        )
        is None
    )


def test_portal_grant_middleware_allows_health_and_protects_mcp() -> None:
    token = "super-secret-token"
    client = TestClient(make_test_app(token))

    assert client.get("/health").status_code == 200

    missing = client.get("/mcp")
    assert missing.status_code == 401
    assert missing.json()["error"] == "missing_portal_grant"
    assert token not in missing.text

    invalid = client.get("/mcp", headers={"X-MADPANDA-PORTAL-GRANT": "wrong"})
    assert invalid.status_code == 401
    assert invalid.json()["error"] == "invalid_portal_grant"
    assert token not in invalid.text

    valid = client.get("/mcp", headers={"X-MADPANDA-PORTAL-GRANT": token})
    assert valid.status_code == 200
    assert token not in valid.text

    unconfigured = TestClient(make_test_app(None)).get("/mcp")
    assert unconfigured.status_code == 503
    assert unconfigured.json()["error"] == "portal_grant_not_configured"


def test_every_registered_tool_has_required_annotations() -> None:
    server, _ = make_server()
    tools = server._tool_manager._tools.values()  # pylint: disable=protected-access

    for tool in tools:
        annotations = tool.annotations
        assert annotations is not None, tool.name
        assert annotations.readOnlyHint is not None, tool.name
        assert annotations.destructiveHint is not None, tool.name
        assert annotations.openWorldHint is not None, tool.name
        if annotations.readOnlyHint:
            assert annotations.idempotentHint is True, tool.name

    delete_tool = server._tool_manager.get_tool("qdrant-delete-points")
    assert delete_tool.annotations.destructiveHint is True
    find_tool = server._tool_manager.get_tool("qdrant-find")
    assert find_tool.annotations.readOnlyHint is True
    assert find_tool.annotations.destructiveHint is False

    for tool in tools:
        for property_name, property_schema in tool.parameters.get(
            "properties", {}
        ).items():
            assert property_schema.get("description"), f"{tool.name}.{property_name}"


@pytest.mark.asyncio
async def test_compact_search_default_excludes_payload_and_cache_reuses_embedding() -> (
    None
):
    server, provider = make_server()
    connector = FakeSearchConnector()
    server._default_qdrant_connector = cast(  # pylint: disable=protected-access
        Any,
        connector,
    )
    find_tool = server._tool_manager.get_tool("qdrant-find")
    ctx = SimpleNamespace(request_id="search-test")

    compact = await find_tool.fn(ctx, query="same query", top_k=3)
    result = compact["data"]["results"][0]
    assert result["id"] == "point-1"
    assert result["score"] == 0.98
    assert len(result["snippet"]) <= 240
    assert "payload" not in result
    assert "document" not in result
    assert result["metadata"] == {
        "class": "MUS327",
        "subject": "World Music",
        "module": "1",
        "week": "1",
        "status": "active",
        "material_type": "lesson",
        "title": "Module 1 Lesson 1",
        "author": "Leo Lara",
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
async def test_study_search_defaults_to_school_collection_and_compact_filters() -> None:
    server, _ = make_server()
    connector = FakeSearchConnector()
    server._default_qdrant_connector = cast(  # pylint: disable=protected-access
        Any,
        connector,
    )
    study_tool = server._tool_manager.get_tool("qdrant-study-search")

    response = await study_tool.fn(
        SimpleNamespace(request_id="study-test"),
        query="forced retrieval melody",
        class_code="MUS327",
        subject="World Music",
        module="1",
        week="1",
        status="active",
        material_type="lesson",
        author="Leo Lara",
        top_k=3,
    )

    assert connector.collection_names == ["school"]
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
    assert result["metadata"]["class"] == "MUS327"


@pytest.mark.asyncio
async def test_find_supports_metadata_fields_grouping_and_budget() -> None:
    server, _ = make_server()
    connector = FakeContextConnector(empty_filtered=False)
    server._default_qdrant_connector = cast(  # pylint: disable=protected-access
        Any,
        connector,
    )
    find_tool = server._tool_manager.get_tool("qdrant-find")

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
    context_tool = server._tool_manager.get_tool("qdrant-build-context")

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
    recommend_tool = server._tool_manager.get_tool("qdrant-recommend-memories")

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

    configuration = await server._tool_manager.get_tool(  # pylint: disable=protected-access
        "qdrant-check-configuration"
    ).fn(ctx)
    assert configuration["data"]["service"] == "qdrant-mcp"
    assert configuration["data"]["portal_grant_configured"] is True
    assert configuration["data"]["query_embedding_cache"]["size"] == 256

    capabilities = await server._tool_manager.get_tool(  # pylint: disable=protected-access
        "qdrant-list-capabilities"
    ).fn(ctx)
    assert capabilities["data"]["tool_count"] >= 1
    assert any(
        group["name"] == "navigation" for group in capabilities["data"]["groups"]
    )

    coverage = await server._tool_manager.get_tool(  # pylint: disable=protected-access
        "qdrant-get-endpoint-coverage"
    ).fn(ctx)
    assert coverage["data"]["source"] == "docs/endpoint-coverage.md"
    assert coverage["data"]["summary"]["points"] == "covered for memory workflows"

    usage = await server._tool_manager.get_tool(  # pylint: disable=protected-access
        "qdrant-get-tool-usage"
    ).fn(ctx, tool_name="qdrant-build-context")
    assert usage["data"]["name"] == "qdrant-build-context"
    assert usage["data"]["annotations"]["readOnlyHint"] is True
    assert "top_k=3-5" in usage["data"]["guidance"]


@pytest.mark.asyncio
async def test_collection_navigation_describes_schema_and_suggests_filters() -> None:
    server, _ = make_server()
    connector = FakeSchemaConnector()
    server._default_qdrant_connector = cast(  # pylint: disable=protected-access
        Any,
        connector,
    )
    ctx = SimpleNamespace(request_id="collection-navigation-test")

    describe = await server._tool_manager.get_tool(  # pylint: disable=protected-access
        "qdrant-describe-collection"
    ).fn(ctx)
    assert describe["data"]["collection_name"] == "memories"
    assert describe["data"]["points_count"] == 3
    assert "class" in describe["data"]["known_memory_filter_fields"]

    schema = await server._tool_manager.get_tool(  # pylint: disable=protected-access
        "qdrant-summarize-collection-schema"
    ).fn(ctx)
    assert schema["data"]["memory_filter_keys"]["class"] == "class_code"
    assert "metadata.status" in schema["data"]["indexed_fields"]

    suggested = await server._tool_manager.get_tool(  # pylint: disable=protected-access
        "qdrant-suggest-filters"
    ).fn(ctx, query="Need MUS327 module 1 active context")
    assert suggested["data"]["recommended_filters"] == {
        "class_code": "MUS327",
        "module": "1",
        "status": "active",
    }
    field_names = {field["field"] for field in suggested["data"]["fields"]}
    assert {"class", "module", "status"}.issubset(field_names)
    assert "text_hash" not in field_names
    assert all(
        len(field["sample_values"]) <= 4 for field in suggested["data"]["fields"]
    )


@pytest.mark.asyncio
async def test_health_check_suggests_close_collection_names() -> None:
    server, _ = make_server(collection_name="jarvis-knowledge-base")
    server._default_qdrant_connector = cast(  # pylint: disable=protected-access
        Any,
        FakeHealthConnector(),
    )
    health_tool = server._tool_manager.get_tool("qdrant-health-check")

    response = await health_tool.fn(SimpleNamespace(request_id="health-test"))

    check = response["data"]["checks"]["collection_exists"]
    assert check["ok"] is False
    assert check["collection_name"] == "jarvis-knowledge-base"
    assert "jarvis-knowledgebase" in check["suggested_collections"]


def test_env_example_and_endpoint_coverage_docs_are_present(monkeypatch) -> None:
    env_example = Path(".env.example").read_text()
    required_names = {
        "MCP_PORTAL_GRANT_TOKEN",
        "MCP_PORTAL_GRANT_HEADER",
        "QDRANT_URL",
        "QDRANT_API_KEY",
        "COLLECTION_NAME",
        "MCP_ALLOW_REQUEST_OVERRIDES",
        "MCP_REQUIRE_REQUEST_QDRANT_URL",
        "MCP_REQUIRE_REQUEST_QDRANT_API_KEY",
        "MCP_REQUIRE_REQUEST_COLLECTION",
        "MCP_DISABLE_DEFAULT_QDRANT_FALLBACK",
        "MCP_DISABLE_DEFAULT_EMBEDDING_FALLBACK",
        "MCP_QDRANT_URL_HEADER",
        "MCP_QDRANT_API_KEY_HEADER",
        "MCP_COLLECTION_NAME_HEADER",
        "MCP_EMBEDDING_PROVIDER_HEADER",
        "MCP_EMBEDDING_MODEL_HEADER",
        "MCP_OPENAI_API_KEY_HEADER",
        "MCP_QUERY_EMBEDDING_CACHE_SIZE",
        "MCP_QUERY_EMBEDDING_CACHE_TTL_SECONDS",
        "MCP_STUDY_COLLECTION",
        "FASTMCP_SERVER_HOST",
        "FASTMCP_SERVER_PORT",
        "FORWARDED_ALLOW_IPS",
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
