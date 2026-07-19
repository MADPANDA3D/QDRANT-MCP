import asyncio
import re
from pathlib import Path

from fastembed import TextEmbedding
from fastembed.common.model_description import DenseModelDescription

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

_MODEL_REVISION_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/@:-]{0,199}$")


class FastEmbedProvider(EmbeddingProvider):
    """
    FastEmbed implementation of the embedding provider.
    :param model_name: The name of the FastEmbed model to use.
    :param model_path: Optional preinstalled model directory. When set, FastEmbed is offline-only.
    :param model_revision: Reviewed source identity for a preinstalled model directory.
    """

    def __init__(
        self,
        model_name: str,
        model_path: str | None = None,
        model_revision: str | None = None,
    ):
        self.model_name = model_name
        self.model_path: Path | None = None
        self.version: str | None = None
        model_options: dict[str, str | bool] = {}
        if model_path:
            try:
                resolved_path = Path(model_path).expanduser().resolve(strict=True)
            except OSError:
                raise ValueError(
                    "FASTEMBED_MODEL_PATH must point to an existing model directory."
                ) from None
            if not resolved_path.is_dir():
                raise ValueError("FASTEMBED_MODEL_PATH must point to an existing model directory.")
            self.model_path = resolved_path
            if model_revision:
                normalized_revision = model_revision.strip()
                if not _MODEL_REVISION_PATTERN.fullmatch(normalized_revision):
                    raise ValueError("FASTEMBED_MODEL_REVISION has an invalid format.")
                self.version = normalized_revision
            model_options = {
                "specific_model_path": str(resolved_path),
                "local_files_only": True,
            }
        elif model_revision:
            raise ValueError("FASTEMBED_MODEL_REVISION requires FASTEMBED_MODEL_PATH.")
        self.embedding_model = TextEmbedding(model_name, **model_options)

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        """Embed a list of documents into vectors."""
        # Run in a thread pool since FastEmbed is synchronous
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self.embedding_model.passage_embed(documents))
        )
        return [embedding.tolist() for embedding in embeddings]

    async def embed_query(self, query: str) -> list[float]:
        """Embed a query into a vector."""
        # Run in a thread pool since FastEmbed is synchronous
        loop = asyncio.get_running_loop()
        embeddings = await loop.run_in_executor(
            None, lambda: list(self.embedding_model.query_embed([query]))
        )
        return embeddings[0].tolist()

    def get_vector_name(self) -> str:
        """
        Return the name of the vector for the Qdrant collection.
        Important: This is compatible with the FastEmbed logic used before 0.6.0.
        """
        model_name = self.embedding_model.model_name.split("/")[-1].lower()
        return f"fast-{model_name}"

    def get_vector_size(self) -> int:
        """Get the size of the vector for the Qdrant collection."""
        model_description: DenseModelDescription = self.embedding_model._get_model_description(
            self.model_name
        )
        return model_description.dim
