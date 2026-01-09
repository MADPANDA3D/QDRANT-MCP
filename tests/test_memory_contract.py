import pytest

from mcp_server_qdrant.memory import (
    EmbeddingInfo,
    build_memory_backfill_patch,
    build_memory_filter,
    compute_text_hash,
    normalize_memory_input,
)


def test_normalize_memory_defaults():
    embedding = EmbeddingInfo(provider="fastembed", model="model", dim=3, version="v1")
    records, warnings = normalize_memory_input(
        information="Remember this",
        metadata={"source": "user"},
        memory=None,
        embedding_info=embedding,
        strict=False,
        max_text_length=5000,
    )
    assert len(records) == 1
    record = records[0]
    assert record.text == "Remember this"
    assert record.metadata["text"] == "Remember this"
    assert record.metadata["source"] == "user"
    assert record.metadata["type"] == "note"
    assert record.metadata["scope"] == "global"
    assert record.metadata["text_hash"]
    assert record.metadata["embedding_provider"] == "fastembed"
    assert record.metadata["embedding_model"] == "model"
    assert record.metadata["embedding_dim"] == 3
    assert record.metadata["embedding_version"] == "v1"
    assert warnings


def test_normalize_memory_strict_missing_fields():
    embedding = EmbeddingInfo(provider="fastembed", model="model", dim=3, version="v1")
    with pytest.raises(ValueError):
        normalize_memory_input(
            information="Remember this",
            metadata=None,
            memory=None,
            embedding_info=embedding,
            strict=True,
            max_text_length=5000,
        )


def test_normalize_memory_chunking():
    embedding = EmbeddingInfo(provider="fastembed", model="model", dim=3, version="v1")
    text = "x" * 1200
    records, warnings = normalize_memory_input(
        information=text,
        metadata={"source": "user"},
        memory=None,
        embedding_info=embedding,
        strict=False,
        max_text_length=500,
    )
    assert len(records) > 1
    assert any("chunked" in warning for warning in warnings)
    for record in records:
        assert "chunk_index" in record.metadata
        assert "chunk_count" in record.metadata
        assert "parent_text_hash" in record.metadata


def test_normalize_memory_labels_strict():
    embedding = EmbeddingInfo(provider="fastembed", model="model", dim=3, version="v1")
    records, warnings = normalize_memory_input(
        information="Remember this",
        metadata={
            "text": "Remember this",
            "type": "note",
            "entities": [],
            "labels": ["housekeeping"],
            "source": "user",
            "created_at": "2024-01-01T00:00:00+00:00",
            "updated_at": "2024-01-01T00:00:00+00:00",
            "scope": "global",
            "confidence": 0.5,
        },
        memory=None,
        embedding_info=embedding,
        strict=True,
        max_text_length=5000,
    )
    assert records[0].metadata["labels"] == ["housekeeping"]
    assert warnings == []


def test_build_memory_filter_labels():
    warnings: list[str] = []
    filt = build_memory_filter(
        {"labels": ["alpha", "beta"]},
        strict=False,
        warnings=warnings,
    )
    assert warnings == []
    assert filt is not None
    assert filt.must[0].key.endswith(".labels")


def test_backfill_patch_adds_missing_fields():
    embedding = EmbeddingInfo(provider="openai", model="model", dim=3, version="v1")
    patch, warnings = build_memory_backfill_patch(
        text="Hello world",
        metadata={},
        embedding_info=embedding,
        strict=False,
    )
    assert patch["text"] == "Hello world"
    assert patch["type"] == "note"
    assert patch["entities"] == []
    assert patch["source"] == "user"
    assert patch["scope"] == "global"
    assert "confidence" in patch
    assert "created_at" in patch
    assert "updated_at" in patch
    assert "created_at_ts" in patch
    assert "updated_at_ts" in patch
    assert "last_seen_at" in patch
    assert "last_seen_at_ts" in patch
    assert patch["text_hash"]
    assert patch["embedding_provider"] == "openai"
    assert patch["embedding_model"] == "model"
    assert patch["embedding_dim"] == 3
    assert patch["embedding_version"] == "v1"
    assert warnings == []


def test_backfill_patch_preserves_existing_fields():
    embedding = EmbeddingInfo(provider="openai", model="model", dim=3, version="v1")
    text = "Hello world"
    metadata = {
        "text": text,
        "type": "person",
        "entities": ["example"],
        "source": "import",
        "scope": "user",
        "confidence": 0.9,
        "created_at": "2024-01-01T00:00:00+00:00",
        "updated_at": "2024-01-02T00:00:00+00:00",
        "created_at_ts": 1704067200000,
        "updated_at_ts": 1704153600000,
        "text_hash": compute_text_hash(text),
        "embedding_provider": "openai",
        "embedding_model": "legacy",
        "embedding_dim": 3,
        "embedding_version": "v0",
    }
    patch, warnings = build_memory_backfill_patch(
        text=text,
        metadata=metadata,
        embedding_info=embedding,
        strict=False,
    )
    assert "type" not in patch
    assert "entities" not in patch
    assert "source" not in patch
    assert "scope" not in patch
    assert "confidence" not in patch
    assert "created_at" not in patch
    assert "updated_at" not in patch
    assert "created_at_ts" not in patch
    assert "updated_at_ts" not in patch
    assert "text_hash" not in patch
    assert "embedding_provider" not in patch
    assert "embedding_model" not in patch
    assert "embedding_dim" not in patch
    assert "embedding_version" not in patch
    assert warnings == []
