from mcp_server_qdrant.hosted_server import HostedQdrantMCPServer
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    MemorySettings,
    QdrantSettings,
    RequestOverrideSettings,
    ToolSettings,
)

mcp = HostedQdrantMCPServer(
    tool_settings=ToolSettings(),
    qdrant_settings=QdrantSettings(),
    request_override_settings=RequestOverrideSettings(),
    embedding_provider_settings=EmbeddingProviderSettings(),
    memory_settings=MemorySettings(),
)
