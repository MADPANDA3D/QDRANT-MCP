from types import SimpleNamespace

import pytest

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.hosted_server import HostedQdrantMCPServer
from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.qdrant import QdrantConnector
from mcp_server_qdrant.settings import (
    MemorySettings,
    QdrantSettings,
    RequestOverrideSettings,
    ToolSettings,
)


class DummyEmbeddingProvider(EmbeddingProvider):
    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in documents]

    async def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def get_vector_name(self) -> str:
        return "text"

    def get_vector_size(self) -> int:
        return 3


def make_qdrant_server() -> QdrantMCPServer:
    return QdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=QdrantSettings.model_validate(
            {"QDRANT_LOCAL_PATH": ":memory:"}
        ),
        request_override_settings=RequestOverrideSettings(),
        memory_settings=MemorySettings(),
        embedding_provider=DummyEmbeddingProvider(),
    )


def make_hosted_server() -> HostedQdrantMCPServer:
    server = HostedQdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=QdrantSettings.model_validate({"QDRANT_LOCAL_PATH": ":memory:"}),
        request_override_settings=RequestOverrideSettings(),
        memory_settings=MemorySettings(),
        embedding_provider=DummyEmbeddingProvider(),
    )
    server.request_override_settings.allow_request_overrides = True
    server.request_override_settings.require_request_qdrant_url = True
    server.request_override_settings.require_request_collection = False
    server.request_override_settings.require_request_qdrant_api_key = False
    server.request_override_settings.disable_default_qdrant_fallback = True
    return server


@pytest.mark.asyncio
async def test_create_collection_without_default_collection() -> None:
    connector = QdrantConnector(
        qdrant_url=":memory:",
        qdrant_api_key=None,
        collection_name=None,
        embedding_provider=DummyEmbeddingProvider(),
    )

    created = await connector.create_collection("agents-project")
    assert created["collection_name"] == "agents-project"
    assert created["created"] is True
    assert created["already_exists"] is False
    assert "agents-project" in await connector.get_collection_names()

    existing = await connector.create_collection("agents-project")
    assert existing["created"] is False
    assert existing["already_exists"] is True


@pytest.mark.asyncio
async def test_missing_collection_error_message_is_actionable() -> None:
    server = make_qdrant_server()
    find_tool = server._tool_manager.get_tool("qdrant-find")

    with pytest.raises(
        ValueError,
        match=(
            r"Tool requires a collection, provide `collection_name` arg or "
            r"`X-Collection-Name` header\."
        ),
    ):
        await find_tool.fn(SimpleNamespace(request_id="test-request"), query="hello")


def test_collection_requirement_tags_are_applied() -> None:
    server = make_qdrant_server()

    requires_collection = server._tool_manager.get_tool("qdrant-find")
    assert "[Requires collection name]" in (requires_collection.description or "")

    no_collection_required = server._tool_manager.get_tool("qdrant-list-collections")
    assert "[Requires collection name]" not in (
        no_collection_required.description or ""
    )

    create_collection = server._tool_manager.get_tool("qdrant-create-collection")
    assert "[Requires collection name]" in (create_collection.description or "")


def test_hosted_overrides_accept_no_collection_header() -> None:
    server = make_hosted_server()
    overrides = server._build_request_overrides(  # pylint: disable=protected-access
        {
            "x-qdrant-url": "https://example.qdrant.io:6333",
            "x-qdrant-api-key": "secret",
        }
    )
    assert overrides is not None
    assert overrides.collection_name is None
