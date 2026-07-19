from pathlib import Path

import pytest

import mcp_server_qdrant.embeddings.fastembed as fastembed_module
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider
from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.settings import EmbeddingProviderSettings


class RecordingTextEmbedding:
    calls: list[tuple[str, dict[str, object]]] = []

    def __init__(self, model_name: str, **kwargs: object):
        self.model_name = model_name
        self.calls.append((model_name, kwargs))


def test_direct_python_provider_keeps_fastembed_default_behavior(monkeypatch) -> None:
    RecordingTextEmbedding.calls = []
    monkeypatch.setattr(fastembed_module, "TextEmbedding", RecordingTextEmbedding)

    provider = FastEmbedProvider("sentence-transformers/all-MiniLM-L6-v2")

    assert provider.model_path is None
    assert RecordingTextEmbedding.calls == [("sentence-transformers/all-MiniLM-L6-v2", {})]


def test_preinstalled_model_path_forces_specific_offline_loading(
    monkeypatch,
    tmp_path: Path,
) -> None:
    RecordingTextEmbedding.calls = []
    monkeypatch.setattr(fastembed_module, "TextEmbedding", RecordingTextEmbedding)
    model_path = tmp_path / "reviewed-model"
    model_path.mkdir()

    provider = FastEmbedProvider(
        "sentence-transformers/all-MiniLM-L6-v2",
        model_path=str(model_path),
    )

    assert provider.model_path == model_path.resolve()
    assert provider.version is None
    assert RecordingTextEmbedding.calls == [
        (
            "sentence-transformers/all-MiniLM-L6-v2",
            {
                "specific_model_path": str(model_path.resolve()),
                "local_files_only": True,
            },
        )
    ]


def test_preinstalled_model_path_must_be_an_existing_directory(
    monkeypatch,
    tmp_path: Path,
) -> None:
    RecordingTextEmbedding.calls = []
    monkeypatch.setattr(fastembed_module, "TextEmbedding", RecordingTextEmbedding)

    with pytest.raises(ValueError, match="existing model directory"):
        FastEmbedProvider("sentence-transformers/all-MiniLM-L6-v2", str(tmp_path / "missing"))

    model_file = tmp_path / "model.onnx"
    model_file.write_bytes(b"not a directory")
    with pytest.raises(ValueError, match="existing model directory"):
        FastEmbedProvider("sentence-transformers/all-MiniLM-L6-v2", str(model_file))

    assert RecordingTextEmbedding.calls == []


def test_factory_passes_the_configured_fastembed_model_path(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "reviewed-model"
    model_path.mkdir()
    revision = "qdrant/all-MiniLM-L6-v2-onnx@5f1b8cd78bc4fb444dd171e59b18f3a3af89a079"
    received: list[tuple[str, str | None, str | None]] = []

    class FakeFastEmbedProvider:
        def __init__(
            self,
            model_name: str,
            model_path: str | None = None,
            model_revision: str | None = None,
        ):
            received.append((model_name, model_path, model_revision))

    monkeypatch.setattr(fastembed_module, "FastEmbedProvider", FakeFastEmbedProvider)
    settings = EmbeddingProviderSettings.model_validate(
        {
            "FASTEMBED_MODEL_PATH": str(model_path),
            "FASTEMBED_MODEL_REVISION": revision,
        }
    )

    provider = create_embedding_provider(settings)

    assert isinstance(provider, FakeFastEmbedProvider)
    assert received == [("sentence-transformers/all-MiniLM-L6-v2", str(model_path), revision)]


def test_embedding_info_prefers_preinstalled_model_revision(
    monkeypatch,
    tmp_path: Path,
) -> None:
    RecordingTextEmbedding.calls = []
    monkeypatch.setattr(fastembed_module, "TextEmbedding", RecordingTextEmbedding)
    model_path = tmp_path / "reviewed-model"
    model_path.mkdir()
    revision = "qdrant/all-MiniLM-L6-v2-onnx@5f1b8cd78bc4fb444dd171e59b18f3a3af89a079"
    provider = FastEmbedProvider(
        "sentence-transformers/all-MiniLM-L6-v2",
        model_path=str(model_path),
        model_revision=revision,
    )
    provider.get_vector_size = lambda: 384  # type: ignore[method-assign]
    settings = EmbeddingProviderSettings.model_validate(
        {
            "FASTEMBED_MODEL_PATH": str(model_path),
            "FASTEMBED_MODEL_REVISION": revision,
        }
    )
    server = object.__new__(QdrantMCPServer)

    info = QdrantMCPServer._resolve_embedding_info(
        server,
        provider=provider,
        settings=settings,
    )

    assert provider.version == revision
    assert info.version == revision
