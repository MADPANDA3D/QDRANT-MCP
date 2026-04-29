import re
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode

from mcp_server_qdrant.embeddings.types import EmbeddingProviderType

DEFAULT_TOOL_STORE_DESCRIPTION = (
    "Keep the memory for later use, when you are asked to remember something."
)
DEFAULT_TOOL_FIND_DESCRIPTION = (
    "Look up memories in Qdrant. Use this tool when you need to: \n"
    " - Find memories by their content \n"
    " - Access memories for further analysis \n"
    " - Get some personal information about the user"
)

METADATA_PATH = "metadata"


class ToolSettings(BaseSettings):
    """
    Configuration for all the tools.
    """

    tool_store_description: str = Field(
        default=DEFAULT_TOOL_STORE_DESCRIPTION,
        validation_alias="TOOL_STORE_DESCRIPTION",
    )
    tool_find_description: str = Field(
        default=DEFAULT_TOOL_FIND_DESCRIPTION,
        validation_alias="TOOL_FIND_DESCRIPTION",
    )
    admin_tools_enabled: bool = Field(
        default=False,
        validation_alias="MCP_ADMIN_TOOLS_ENABLED",
    )
    mutations_require_admin: bool = Field(
        default=False,
        validation_alias="MCP_MUTATIONS_REQUIRE_ADMIN",
    )
    max_batch_size: int = Field(
        default=500,
        validation_alias="MCP_MAX_BATCH_SIZE",
    )
    max_point_ids: int = Field(
        default=500,
        validation_alias="MCP_MAX_POINT_IDS",
    )


class EmbeddingProviderSettings(BaseSettings):
    """
    Configuration for the embedding provider.
    """

    provider_type: EmbeddingProviderType = Field(
        default=EmbeddingProviderType.FASTEMBED,
        validation_alias="EMBEDDING_PROVIDER",
    )
    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        validation_alias="EMBEDDING_MODEL",
    )
    vector_size: int | None = Field(
        default=None,
        validation_alias="EMBEDDING_VECTOR_SIZE",
    )
    openai_api_key: str | None = Field(
        default=None,
        validation_alias="OPENAI_API_KEY",
    )
    openai_base_url: str | None = Field(
        default=None,
        validation_alias="OPENAI_BASE_URL",
    )
    openai_organization: str | None = Field(
        default=None,
        validation_alias="OPENAI_ORG",
    )
    openai_project: str | None = Field(
        default=None,
        validation_alias="OPENAI_PROJECT",
    )
    version: str | None = Field(
        default=None,
        validation_alias="EMBEDDING_VERSION",
    )

    @field_validator("vector_size", mode="before")
    @classmethod
    def empty_vector_size_as_none(cls, value: object) -> object:
        if value == "":
            return None
        return value

    @model_validator(mode="after")
    def check_openai_settings(self) -> "EmbeddingProviderSettings":
        if self.provider_type == EmbeddingProviderType.OPENAI:
            if not self.openai_api_key:
                raise ValueError(
                    "OPENAI_API_KEY is required when EMBEDDING_PROVIDER=openai."
                )
            if not self.model_name:
                raise ValueError(
                    "EMBEDDING_MODEL is required when EMBEDDING_PROVIDER=openai."
                )
        return self


class FilterableField(BaseModel):
    name: str = Field(description="The name of the field payload field to filter on")
    description: str = Field(
        description="A description for the field used in the tool description"
    )
    field_type: Literal["keyword", "integer", "float", "boolean"] = Field(
        description="The type of the field"
    )
    condition: Literal["==", "!=", ">", ">=", "<", "<=", "any", "except"] | None = (
        Field(
            default=None,
            description=(
                "The condition to use for the filter. If not provided, the field will be indexed, but no "
                "filter argument will be exposed to MCP tool."
            ),
        )
    )
    required: bool = Field(
        default=False,
        description="Whether the field is required for the filter.",
    )


class QdrantSettings(BaseSettings):
    """
    Configuration for the Qdrant connector.
    """

    location: str | None = Field(default=None, validation_alias="QDRANT_URL")
    api_key: str | None = Field(default=None, validation_alias="QDRANT_API_KEY")
    collection_name: str | None = Field(
        default=None, validation_alias="COLLECTION_NAME"
    )
    vector_name: str | None = Field(default=None, validation_alias="QDRANT_VECTOR_NAME")
    local_path: str | None = Field(default=None, validation_alias="QDRANT_LOCAL_PATH")
    search_limit: int = Field(default=10, validation_alias="QDRANT_SEARCH_LIMIT")
    read_only: bool = Field(default=False, validation_alias="QDRANT_READ_ONLY")

    filterable_fields: list[FilterableField] | None = Field(default=None)

    allow_arbitrary_filter: bool = Field(
        default=False, validation_alias="QDRANT_ALLOW_ARBITRARY_FILTER"
    )

    def filterable_fields_dict(self) -> dict[str, FilterableField]:
        if self.filterable_fields is None:
            return {}
        return {field.name: field for field in self.filterable_fields}

    def filterable_fields_dict_with_conditions(self) -> dict[str, FilterableField]:
        if self.filterable_fields is None:
            return {}
        return {
            field.name: field
            for field in self.filterable_fields
            if field.condition is not None
        }

    @model_validator(mode="after")
    def check_local_path_conflict(self) -> "QdrantSettings":
        if self.local_path:
            if self.location is not None or self.api_key is not None:
                raise ValueError(
                    "If 'local_path' is set, 'location' and 'api_key' must be None."
                )
        return self


class RequestOverrideSettings(BaseSettings):
    """
    Per-request overrides for hosted deployments.
    """

    allow_request_overrides: bool = Field(
        default=False,
        validation_alias="MCP_ALLOW_REQUEST_OVERRIDES",
    )
    require_request_qdrant_url: bool = Field(
        default=True,
        validation_alias="MCP_REQUIRE_REQUEST_QDRANT_URL",
    )
    require_request_collection: bool = Field(
        default=True,
        validation_alias="MCP_REQUIRE_REQUEST_COLLECTION",
    )
    require_request_qdrant_api_key: bool = Field(
        default=False,
        validation_alias="MCP_REQUIRE_REQUEST_QDRANT_API_KEY",
    )
    disable_default_qdrant_fallback: bool = Field(
        default=False,
        validation_alias="MCP_DISABLE_DEFAULT_QDRANT_FALLBACK",
    )
    disable_default_embedding_fallback: bool = Field(
        default=False,
        validation_alias="MCP_DISABLE_DEFAULT_EMBEDDING_FALLBACK",
    )
    qdrant_url_header: str = Field(
        default="x-qdrant-url",
        validation_alias="MCP_QDRANT_URL_HEADER",
    )
    qdrant_api_key_header: str = Field(
        default="x-qdrant-api-key",
        validation_alias="MCP_QDRANT_API_KEY_HEADER",
    )
    collection_name_header: str = Field(
        default="x-collection-name",
        validation_alias="MCP_COLLECTION_NAME_HEADER",
    )
    vector_name_header: str = Field(
        default="x-qdrant-vector-name",
        validation_alias="MCP_QDRANT_VECTOR_NAME_HEADER",
    )
    embedding_provider_header: str = Field(
        default="x-embedding-provider",
        validation_alias="MCP_EMBEDDING_PROVIDER_HEADER",
    )
    embedding_model_header: str = Field(
        default="x-embedding-model",
        validation_alias="MCP_EMBEDDING_MODEL_HEADER",
    )
    embedding_vector_size_header: str = Field(
        default="x-embedding-vector-size",
        validation_alias="MCP_EMBEDDING_VECTOR_SIZE_HEADER",
    )
    openai_api_key_header: str = Field(
        default="x-openai-api-key",
        validation_alias="MCP_OPENAI_API_KEY_HEADER",
    )
    openai_base_url_header: str = Field(
        default="x-openai-base-url",
        validation_alias="MCP_OPENAI_BASE_URL_HEADER",
    )
    openai_organization_header: str = Field(
        default="x-openai-org",
        validation_alias="MCP_OPENAI_ORG_HEADER",
    )
    openai_project_header: str = Field(
        default="x-openai-project",
        validation_alias="MCP_OPENAI_PROJECT_HEADER",
    )
    qdrant_host_allowlist: Annotated[list[str], NoDecode] = Field(
        default_factory=list,
        validation_alias="MCP_QDRANT_HOST_ALLOWLIST",
    )
    portal_grant_token: str | None = Field(
        default=None,
        validation_alias="MCP_PORTAL_GRANT_TOKEN",
    )
    portal_grant_header: str = Field(
        default="x-madpanda-portal-grant",
        validation_alias="MCP_PORTAL_GRANT_HEADER",
    )

    @field_validator("qdrant_host_allowlist", mode="before")
    @classmethod
    def parse_allowlist(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            parts = [item.strip().lower() for item in re.split(r"[,\s]+", value)]
            return [item for item in parts if item]
        if isinstance(value, list):
            return [str(item).strip().lower() for item in value if str(item).strip()]
        return []

    @model_validator(mode="after")
    def normalize_headers(self) -> "RequestOverrideSettings":
        self.qdrant_url_header = self.qdrant_url_header.lower()
        self.qdrant_api_key_header = self.qdrant_api_key_header.lower()
        self.collection_name_header = self.collection_name_header.lower()
        self.vector_name_header = self.vector_name_header.lower()
        self.embedding_provider_header = self.embedding_provider_header.lower()
        self.embedding_model_header = self.embedding_model_header.lower()
        self.embedding_vector_size_header = self.embedding_vector_size_header.lower()
        self.openai_api_key_header = self.openai_api_key_header.lower()
        self.openai_base_url_header = self.openai_base_url_header.lower()
        self.openai_organization_header = self.openai_organization_header.lower()
        self.openai_project_header = self.openai_project_header.lower()
        self.portal_grant_header = self.portal_grant_header.lower()
        return self


class MemorySettings(BaseSettings):
    """
    Configuration for memory contract normalization and safety guards.
    """

    strict_params: bool = Field(default=False, validation_alias="MCP_STRICT_PARAMS")
    max_text_length: int = Field(default=8000, validation_alias="MCP_MAX_TEXT_LENGTH")
    dedupe_action: Literal["update", "skip"] = Field(
        default="update",
        validation_alias="MCP_DEDUPE_ACTION",
    )
    health_check_collection: str | None = Field(
        default=None,
        validation_alias="MCP_HEALTH_CHECK_COLLECTION",
    )
    ingest_validation_mode: Literal["allow", "reject", "quarantine"] = Field(
        default="allow",
        validation_alias="MCP_INGEST_VALIDATION_MODE",
    )
    quarantine_collection: str = Field(
        default="jarvis-quarantine",
        validation_alias="MCP_QUARANTINE_COLLECTION",
    )
    short_term_collection: str = Field(
        default="jarvis-short-term",
        validation_alias="MCP_SHORT_TERM_COLLECTION",
    )
    short_term_ttl_days: int = Field(
        default=7,
        validation_alias="MCP_SHORT_TERM_TTL_DAYS",
    )
    textbook_max_file_bytes: int = Field(
        default=100 * 1024 * 1024,
        validation_alias="MCP_TEXTBOOK_MAX_FILE_BYTES",
    )
    textbook_max_pages: int = Field(
        default=1000,
        validation_alias="MCP_TEXTBOOK_MAX_PAGES",
    )
    textbook_max_extracted_chars: int = Field(
        default=3_000_000,
        validation_alias="MCP_TEXTBOOK_MAX_EXTRACTED_CHARS",
    )
    textbook_max_chunks: int = Field(
        default=20_000,
        validation_alias="MCP_TEXTBOOK_MAX_CHUNKS",
    )
    textbook_ocr_low_text_threshold_chars: int = Field(
        default=120,
        validation_alias="MCP_TEXTBOOK_OCR_LOW_TEXT_THRESHOLD_CHARS",
    )
    textbook_ocr_min_coverage_ratio: float = Field(
        default=0.85,
        validation_alias="MCP_TEXTBOOK_OCR_MIN_COVERAGE_RATIO",
    )
    textbook_ocr_max_pages: int = Field(
        default=120,
        validation_alias="MCP_TEXTBOOK_OCR_MAX_PAGES",
    )
    textbook_ocr_max_page_ratio: float = Field(
        default=0.30,
        validation_alias="MCP_TEXTBOOK_OCR_MAX_PAGE_RATIO",
    )
    textbook_job_timeout_seconds: int = Field(
        default=45 * 60,
        validation_alias="MCP_TEXTBOOK_JOB_TIMEOUT_SECONDS",
    )
    textbook_embed_batch_size: int = Field(
        default=32,
        validation_alias="MCP_TEXTBOOK_EMBED_BATCH_SIZE",
    )
    textbook_upsert_batch_size: int = Field(
        default=64,
        validation_alias="MCP_TEXTBOOK_UPSERT_BATCH_SIZE",
    )
    textbook_max_concurrency: int = Field(
        default=2,
        validation_alias="MCP_TEXTBOOK_MAX_CONCURRENCY",
    )
    textbook_job_state_dir: str = Field(
        default="/tmp/mcp-server-qdrant/jobs",
        validation_alias="MCP_TEXTBOOK_JOB_STATE_DIR",
    )
    textbook_job_state_retention_hours: int = Field(
        default=168,
        validation_alias="MCP_TEXTBOOK_JOB_STATE_RETENTION_HOURS",
    )
    query_embedding_cache_size: int = Field(
        default=256,
        validation_alias="MCP_QUERY_EMBEDDING_CACHE_SIZE",
    )
    query_embedding_cache_ttl_seconds: int = Field(
        default=3600,
        validation_alias="MCP_QUERY_EMBEDDING_CACHE_TTL_SECONDS",
    )

    @model_validator(mode="after")
    def validate_positive_limits(self) -> "MemorySettings":
        positive_fields = (
            "max_text_length",
            "short_term_ttl_days",
            "textbook_max_file_bytes",
            "textbook_max_pages",
            "textbook_max_extracted_chars",
            "textbook_max_chunks",
            "textbook_ocr_low_text_threshold_chars",
            "textbook_ocr_max_pages",
            "textbook_job_timeout_seconds",
            "textbook_embed_batch_size",
            "textbook_upsert_batch_size",
            "textbook_max_concurrency",
            "textbook_job_state_retention_hours",
        )
        for field_name in positive_fields:
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be positive.")
        if not (0 <= self.textbook_ocr_min_coverage_ratio <= 1):
            raise ValueError("textbook_ocr_min_coverage_ratio must be between 0 and 1.")
        if not (0 <= self.textbook_ocr_max_page_ratio <= 1):
            raise ValueError("textbook_ocr_max_page_ratio must be between 0 and 1.")
        if not self.textbook_job_state_dir.strip():
            raise ValueError("textbook_job_state_dir must be a non-empty path.")
        if self.query_embedding_cache_size < 0:
            raise ValueError("query_embedding_cache_size must be >= 0.")
        if self.query_embedding_cache_ttl_seconds < 0:
            raise ValueError("query_embedding_cache_ttl_seconds must be >= 0.")
        return self
