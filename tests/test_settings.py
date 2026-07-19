import pytest
from pydantic import ValidationError

from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.settings import (
    DEFAULT_TOOL_FIND_DESCRIPTION,
    DEFAULT_TOOL_STORE_DESCRIPTION,
    STOCK_FASTEMBED_REVISION,
    EmbeddingProviderSettings,
    MemorySettings,
    QdrantSettings,
    RequestOverrideSettings,
    ToolSettings,
)


@pytest.mark.parametrize(
    "settings_type",
    [
        ToolSettings,
        EmbeddingProviderSettings,
        QdrantSettings,
        RequestOverrideSettings,
        MemorySettings,
    ],
)
def test_all_settings_models_hide_raw_inputs_in_validation_errors(settings_type):
    assert settings_type.model_config["hide_input_in_errors"] is True


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

    def test_api_key_is_accessible_but_absent_from_model_representations(self):
        secret = "synthetic-qdrant-secret-for-repr"
        private_endpoint = "https://private-qdrant.internal.example"
        settings = QdrantSettings.model_validate(
            {
                "QDRANT_URL": private_endpoint,
                "QDRANT_API_KEY": secret,
                "COLLECTION_NAME": "private-collection-identity",
                "QDRANT_VECTOR_NAME": "private-vector-identity",
            }
        )

        assert settings.api_key == secret
        assert settings.location == private_endpoint
        assert settings.collection_name == "private-collection-identity"
        assert settings.vector_name == "private-vector-identity"
        for rendered in (repr(settings), str(settings)):
            assert secret not in rendered
            assert private_endpoint not in rendered
            assert "private-collection-identity" not in rendered
            assert "private-vector-identity" not in rendered

        local_path = "/private/qdrant/storage"
        local_settings = QdrantSettings.model_validate({"QDRANT_LOCAL_PATH": local_path})
        assert local_settings.local_path == local_path
        assert local_path not in repr(local_settings)
        assert local_path not in str(local_settings)

    def test_validation_error_never_echoes_qdrant_secret_fragments(self):
        secret = "qdrant-startup-secret-ALPHA-OMEGA"
        with pytest.raises(ValidationError) as captured:
            QdrantSettings.model_validate(
                {
                    "QDRANT_LOCAL_PATH": "/private/qdrant/storage",
                    "QDRANT_API_KEY": secret,
                }
            )

        rendered = f"{captured.value!s} {captured.value!r}"
        for fragment in (secret, secret[:14], secret[-14:]):
            assert fragment not in rendered


class TestEmbeddingProviderSettings:
    def test_default_values(self):
        """Test default values are set correctly."""
        settings = EmbeddingProviderSettings()
        assert settings.provider_type == EmbeddingProviderType.FASTEMBED
        assert settings.model_name == "sentence-transformers/all-MiniLM-L6-v2"
        assert settings.fastembed_model_path is None
        assert settings.fastembed_model_revision is None

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

    def test_empty_vector_size_is_none(self, monkeypatch):
        monkeypatch.setenv("EMBEDDING_VECTOR_SIZE", "")
        settings = EmbeddingProviderSettings()
        assert settings.vector_size is None

    def test_fastembed_model_path_is_optional_and_hidden(self, monkeypatch):
        model_path = "/opt/reviewed-model"
        monkeypatch.setenv("FASTEMBED_MODEL_PATH", model_path)
        monkeypatch.setenv("FASTEMBED_MODEL_REVISION", " qdrant/model@0123456789abcdef ")
        settings = EmbeddingProviderSettings()

        assert settings.fastembed_model_path == model_path
        assert settings.fastembed_model_revision == "qdrant/model@0123456789abcdef"
        assert model_path not in repr(settings)
        assert model_path not in str(settings)

        monkeypatch.setenv("FASTEMBED_MODEL_PATH", "")
        monkeypatch.setenv("FASTEMBED_MODEL_REVISION", "")
        assert EmbeddingProviderSettings().fastembed_model_path is None

    def test_fastembed_revision_requires_a_model_path(self, monkeypatch):
        monkeypatch.setenv("FASTEMBED_MODEL_REVISION", "qdrant/model@0123456789abcdef")
        with pytest.raises(ValidationError, match="FASTEMBED_MODEL_REVISION requires"):
            EmbeddingProviderSettings()

    def test_stock_fastembed_revision_rejects_a_different_fastembed_model(
        self,
        monkeypatch,
    ):
        monkeypatch.setenv("FASTEMBED_MODEL_PATH", "/opt/reviewed-model")
        monkeypatch.setenv("FASTEMBED_MODEL_REVISION", STOCK_FASTEMBED_REVISION)
        monkeypatch.setenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

        with pytest.raises(ValidationError, match="stock FastEmbed model revision requires"):
            EmbeddingProviderSettings()

    def test_openai_provider_ignores_the_stock_fastembed_identity(self, monkeypatch):
        monkeypatch.setenv("FASTEMBED_MODEL_PATH", "/opt/reviewed-model")
        monkeypatch.setenv("FASTEMBED_MODEL_REVISION", STOCK_FASTEMBED_REVISION)
        monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
        monkeypatch.setenv("EMBEDDING_MODEL", "text-embedding-3-small")
        monkeypatch.setenv("OPENAI_API_KEY", "synthetic-openai-key")

        settings = EmbeddingProviderSettings()

        assert settings.provider_type == EmbeddingProviderType.OPENAI
        assert settings.model_name == "text-embedding-3-small"

    def test_openai_key_is_accessible_but_absent_from_model_representations(self):
        secret = "synthetic-openai-secret-for-repr"
        private_endpoint = "https://private-openai.internal.example/v1"
        organization = "org-private-identity"
        project = "project-private-identity"
        settings = EmbeddingProviderSettings.model_validate(
            {
                "EMBEDDING_PROVIDER": "openai",
                "EMBEDDING_MODEL": "text-embedding-3-small",
                "OPENAI_API_KEY": secret,
                "OPENAI_BASE_URL": private_endpoint,
                "OPENAI_ORG": organization,
                "OPENAI_PROJECT": project,
            }
        )

        assert settings.openai_api_key == secret
        assert settings.openai_base_url == private_endpoint
        assert settings.openai_organization == organization
        assert settings.openai_project == project
        for rendered in (repr(settings), str(settings)):
            assert secret not in rendered
            assert private_endpoint not in rendered
            assert organization not in rendered
            assert project not in rendered

    def test_validation_error_never_echoes_openai_secret_fragments(self):
        secret = "openai-startup-secret-ALPHA-OMEGA"
        with pytest.raises(ValidationError) as captured:
            EmbeddingProviderSettings.model_validate(
                {
                    "EMBEDDING_PROVIDER": "openai",
                    "EMBEDDING_MODEL": "",
                    "OPENAI_API_KEY": secret,
                }
            )

        rendered = f"{captured.value!s} {captured.value!r}"
        for fragment in (secret, secret[:14], secret[-14:]):
            assert fragment not in rendered


class TestRequestOverrideSettings:
    def test_allowlist_accepts_comma_separated_env(self, monkeypatch):
        monkeypatch.setenv(
            "MCP_QDRANT_HOST_ALLOWLIST",
            "api.qdrant.io, example.com",
        )
        settings = RequestOverrideSettings()
        assert settings.qdrant_host_allowlist == ["api.qdrant.io", "example.com"]

    @pytest.mark.parametrize(
        "environment_name",
        ["MCP_QDRANT_HOST_ALLOWLIST", "MCP_OPENAI_HOST_ALLOWLIST"],
    )
    def test_request_host_allowlists_reject_wildcards(self, monkeypatch, environment_name):
        monkeypatch.setenv(environment_name, "*.example.com")

        with pytest.raises(ValidationError, match="exact hostnames only"):
            RequestOverrideSettings()


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
        assert settings.quarantine_collection == "quarantine"
        assert settings.short_term_collection == "short-term"
        assert settings.study_collection == "study"
        assert settings.short_term_ttl_days == 7
        assert settings.textbook_max_file_bytes == 250 * 1024 * 1024
        assert settings.file_upload_max_chunk_bytes == 512 * 1024
        assert settings.textbook_max_pages == 5000
        assert settings.textbook_max_extracted_chars == 12_000_000
        assert settings.textbook_max_chunks == 80_000
        assert settings.textbook_ocr_low_text_threshold_chars == 120
        assert settings.textbook_ocr_min_coverage_ratio == 0.85
        assert settings.textbook_ocr_max_pages == 120
        assert settings.textbook_ocr_max_page_ratio == 0.30
        assert settings.textbook_job_timeout_seconds == 7200
        assert settings.textbook_embed_batch_size == 32
        assert settings.textbook_upsert_batch_size == 64
        assert settings.textbook_max_concurrency == 2
        assert settings.textbook_job_state_dir == "/tmp/mcp-server-qdrant/jobs"
        assert settings.textbook_job_state_retention_hours == 168
        assert settings.query_embedding_cache_size == 256
        assert settings.query_embedding_cache_ttl_seconds == 3600

    def test_custom_values(self, monkeypatch):
        monkeypatch.setenv("MCP_STRICT_PARAMS", "1")
        monkeypatch.setenv("MCP_MAX_TEXT_LENGTH", "2048")
        monkeypatch.setenv("MCP_DEDUPE_ACTION", "skip")
        monkeypatch.setenv("MCP_HEALTH_CHECK_COLLECTION", "example-knowledge-base")
        monkeypatch.setenv("MCP_INGEST_VALIDATION_MODE", "quarantine")
        monkeypatch.setenv("MCP_QUARANTINE_COLLECTION", "example-quarantine-dev")
        monkeypatch.setenv("MCP_SHORT_TERM_COLLECTION", "example-short-term-dev")
        monkeypatch.setenv("MCP_STUDY_COLLECTION", "example-study-dev")
        monkeypatch.setenv("MCP_SHORT_TERM_TTL_DAYS", "3")
        monkeypatch.setenv("MCP_TEXTBOOK_MAX_FILE_BYTES", "1048576")
        monkeypatch.setenv("MCP_FILE_UPLOAD_MAX_CHUNK_BYTES", "262144")
        monkeypatch.setenv("MCP_TEXTBOOK_MAX_PAGES", "250")
        monkeypatch.setenv("MCP_TEXTBOOK_MAX_EXTRACTED_CHARS", "250000")
        monkeypatch.setenv("MCP_TEXTBOOK_MAX_CHUNKS", "1800")
        monkeypatch.setenv("MCP_TEXTBOOK_OCR_LOW_TEXT_THRESHOLD_CHARS", "90")
        monkeypatch.setenv("MCP_TEXTBOOK_OCR_MIN_COVERAGE_RATIO", "0.9")
        monkeypatch.setenv("MCP_TEXTBOOK_OCR_MAX_PAGES", "80")
        monkeypatch.setenv("MCP_TEXTBOOK_OCR_MAX_PAGE_RATIO", "0.25")
        monkeypatch.setenv("MCP_TEXTBOOK_JOB_TIMEOUT_SECONDS", "600")
        monkeypatch.setenv("MCP_TEXTBOOK_EMBED_BATCH_SIZE", "32")
        monkeypatch.setenv("MCP_TEXTBOOK_UPSERT_BATCH_SIZE", "48")
        monkeypatch.setenv("MCP_TEXTBOOK_MAX_CONCURRENCY", "2")
        monkeypatch.setenv("MCP_TEXTBOOK_JOB_STATE_DIR", "/tmp/custom-job-store")
        monkeypatch.setenv("MCP_TEXTBOOK_JOB_STATE_RETENTION_HOURS", "72")
        monkeypatch.setenv("MCP_QUERY_EMBEDDING_CACHE_SIZE", "64")
        monkeypatch.setenv("MCP_QUERY_EMBEDDING_CACHE_TTL_SECONDS", "120")
        settings = MemorySettings()
        assert settings.strict_params is True
        assert settings.max_text_length == 2048
        assert settings.dedupe_action == "skip"
        assert settings.health_check_collection == "example-knowledge-base"
        assert settings.ingest_validation_mode == "quarantine"
        assert settings.quarantine_collection == "example-quarantine-dev"
        assert settings.short_term_collection == "example-short-term-dev"
        assert settings.study_collection == "example-study-dev"
        assert settings.short_term_ttl_days == 3
        assert settings.textbook_max_file_bytes == 1_048_576
        assert settings.file_upload_max_chunk_bytes == 262_144
        assert settings.textbook_max_pages == 250
        assert settings.textbook_max_extracted_chars == 250_000
        assert settings.textbook_max_chunks == 1800
        assert settings.textbook_ocr_low_text_threshold_chars == 90
        assert settings.textbook_ocr_min_coverage_ratio == 0.9
        assert settings.textbook_ocr_max_pages == 80
        assert settings.textbook_ocr_max_page_ratio == 0.25
        assert settings.textbook_job_timeout_seconds == 600
        assert settings.textbook_embed_batch_size == 32
        assert settings.textbook_upsert_batch_size == 48
        assert settings.textbook_max_concurrency == 2
        assert settings.textbook_job_state_dir == "/tmp/custom-job-store"
        assert settings.textbook_job_state_retention_hours == 72
        assert settings.query_embedding_cache_size == 64
        assert settings.query_embedding_cache_ttl_seconds == 120

    def test_outbound_host_allowlist_rejects_wildcards(self, monkeypatch):
        monkeypatch.setenv("MCP_OUTBOUND_HOST_ALLOWLIST", "*.example.com")

        with pytest.raises(ValidationError, match="exact hostnames only"):
            MemorySettings()

    def test_invalid_textbook_limits(self, monkeypatch):
        monkeypatch.setenv("MCP_TEXTBOOK_MAX_FILE_BYTES", "0")
        with pytest.raises(ValueError):
            MemorySettings()

    def test_invalid_ocr_ratio_limits(self, monkeypatch):
        monkeypatch.setenv("MCP_TEXTBOOK_OCR_MIN_COVERAGE_RATIO", "1.1")
        with pytest.raises(ValueError):
            MemorySettings()

        monkeypatch.delenv("MCP_TEXTBOOK_OCR_MIN_COVERAGE_RATIO", raising=False)
        monkeypatch.setenv("MCP_TEXTBOOK_OCR_MAX_PAGE_RATIO", "-0.1")
        with pytest.raises(ValueError):
            MemorySettings()
