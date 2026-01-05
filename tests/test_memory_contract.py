import pytest

from mcp_server_qdrant.memory import EmbeddingInfo, normalize_memory_input


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
