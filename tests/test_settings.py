import pytest

from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.settings import (
    DEFAULT_TOOL_FIND_DESCRIPTION,
    DEFAULT_TOOL_STORE_DESCRIPTION,
    EmbeddingProviderSettings,
    MemorySettings,
    QdrantSettings,
    ToolSettings,
)


class TestQdrantSettings:
    def test_default_values(self):
        """Test that required fields raise errors when not provided."""

        # Should not raise error because there are no required fields
        QdrantSettings()

    def test_minimal_config(self, monkeypatch):
        """Test loading minimal configuration from environment variables."""
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("COLLECTION_NAME", "test_collection")

        settings = QdrantSettings()
        assert settings.location == "http://localhost:6333"
        assert settings.collection_name == "test_collection"
        assert settings.api_key is None
        assert settings.local_path is None

    def test_full_config(self, monkeypatch):
        """Test loading full configuration from environment variables."""
        monkeypatch.setenv("QDRANT_URL", "http://qdrant.example.com:6333")
        monkeypatch.setenv("QDRANT_API_KEY", "test_api_key")
        monkeypatch.setenv("COLLECTION_NAME", "my_memories")
        monkeypatch.setenv("QDRANT_SEARCH_LIMIT", "15")
        monkeypatch.setenv("QDRANT_READ_ONLY", "1")

        settings = QdrantSettings()
        assert settings.location == "http://qdrant.example.com:6333"
        assert settings.api_key == "test_api_key"
        assert settings.collection_name == "my_memories"
        assert settings.search_limit == 15
        assert settings.read_only is True

    def test_local_path_config(self, monkeypatch):
        """Test loading local path configuration from environment variables."""
        monkeypatch.setenv("QDRANT_LOCAL_PATH", "/path/to/local/qdrant")

        settings = QdrantSettings()
        assert settings.local_path == "/path/to/local/qdrant"

    def test_local_path_is_exclusive_with_url(self, monkeypatch):
        """Test that local path cannot be set if Qdrant URL is provided."""
        monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
        monkeypatch.setenv("QDRANT_LOCAL_PATH", "/path/to/local/qdrant")

        with pytest.raises(ValueError):
            QdrantSettings()

        monkeypatch.delenv("QDRANT_URL", raising=False)
        monkeypatch.setenv("QDRANT_API_KEY", "test_api_key")
        with pytest.raises(ValueError):
            QdrantSettings()


class TestEmbeddingProviderSettings:
    def test_default_values(self):
        """Test default values are set correctly."""
        settings = EmbeddingProviderSettings()
        assert settings.provider_type == EmbeddingProviderType.FASTEMBED
        assert settings.model_name == "sentence-transformers/all-MiniLM-L6-v2"

    def test_custom_values(self, monkeypatch):
        """Test loading custom values from environment variables."""
        monkeypatch.setenv("EMBEDDING_MODEL", "custom_model")
        settings = EmbeddingProviderSettings()
        assert settings.provider_type == EmbeddingProviderType.FASTEMBED
        assert settings.model_name == "custom_model"

    def test_embedding_version(self, monkeypatch):
        """Test loading embedding version from environment variables."""
        monkeypatch.setenv("EMBEDDING_VERSION", "v1")
        settings = EmbeddingProviderSettings()
        assert settings.version == "v1"


class TestToolSettings:
    def test_default_values(self):
        """Test that default values are set correctly when no env vars are provided."""
        settings = ToolSettings()
        assert settings.tool_store_description == DEFAULT_TOOL_STORE_DESCRIPTION
        assert settings.tool_find_description == DEFAULT_TOOL_FIND_DESCRIPTION
        assert settings.admin_tools_enabled is False
        assert settings.mutations_require_admin is False
        assert settings.max_batch_size == 500
        assert settings.max_point_ids == 500

    def test_custom_store_description(self, monkeypatch):
        """Test loading custom store description from environment variable."""
        monkeypatch.setenv("TOOL_STORE_DESCRIPTION", "Custom store description")
        settings = ToolSettings()
        assert settings.tool_store_description == "Custom store description"
        assert settings.tool_find_description == DEFAULT_TOOL_FIND_DESCRIPTION

    def test_custom_find_description(self, monkeypatch):
        """Test loading custom find description from environment variable."""
        monkeypatch.setenv("TOOL_FIND_DESCRIPTION", "Custom find description")
        settings = ToolSettings()
        assert settings.tool_store_description == DEFAULT_TOOL_STORE_DESCRIPTION
        assert settings.tool_find_description == "Custom find description"

    def test_all_custom_values(self, monkeypatch):
        """Test loading all custom values from environment variables."""
        monkeypatch.setenv("TOOL_STORE_DESCRIPTION", "Custom store description")
        monkeypatch.setenv("TOOL_FIND_DESCRIPTION", "Custom find description")
        settings = ToolSettings()
        assert settings.tool_store_description == "Custom store description"
        assert settings.tool_find_description == "Custom find description"

    def test_admin_tools_enabled(self, monkeypatch):
        """Test loading admin tools flag from environment variable."""
        monkeypatch.setenv("MCP_ADMIN_TOOLS_ENABLED", "1")
        settings = ToolSettings()
        assert settings.admin_tools_enabled is True

    def test_mutation_limits(self, monkeypatch):
        monkeypatch.setenv("MCP_MUTATIONS_REQUIRE_ADMIN", "1")
        monkeypatch.setenv("MCP_MAX_BATCH_SIZE", "250")
        monkeypatch.setenv("MCP_MAX_POINT_IDS", "100")
        settings = ToolSettings()
        assert settings.mutations_require_admin is True
        assert settings.max_batch_size == 250
        assert settings.max_point_ids == 100


class TestMemorySettings:
    def test_default_values(self):
        settings = MemorySettings()
        assert settings.strict_params is False
        assert settings.max_text_length == 8000
        assert settings.dedupe_action == "update"
        assert settings.health_check_collection is None
        assert settings.ingest_validation_mode == "allow"
        assert settings.quarantine_collection == "jarvis-quarantine"

    def test_custom_values(self, monkeypatch):
        monkeypatch.setenv("MCP_STRICT_PARAMS", "1")
        monkeypatch.setenv("MCP_MAX_TEXT_LENGTH", "2048")
        monkeypatch.setenv("MCP_DEDUPE_ACTION", "skip")
        monkeypatch.setenv("MCP_HEALTH_CHECK_COLLECTION", "jarvis-knowledge-base")
        monkeypatch.setenv("MCP_INGEST_VALIDATION_MODE", "quarantine")
        monkeypatch.setenv("MCP_QUARANTINE_COLLECTION", "jarvis-quarantine-dev")
        settings = MemorySettings()
        assert settings.strict_params is True
        assert settings.max_text_length == 2048
        assert settings.dedupe_action == "skip"
        assert settings.health_check_collection == "jarvis-knowledge-base"
        assert settings.ingest_validation_mode == "quarantine"
        assert settings.quarantine_collection == "jarvis-quarantine-dev"
