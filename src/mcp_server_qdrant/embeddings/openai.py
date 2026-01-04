import asyncio

from openai import OpenAI

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

OPENAI_MODEL_DIMS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}


class OpenAIProvider(EmbeddingProvider):
    """
    OpenAI embeddings implementation.
    """

    def __init__(
        self,
        api_key: str,
        model_name: str,
        vector_size: int | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
    ):
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for openai embeddings.")
        self.model_name = model_name
        resolved_size = vector_size or OPENAI_MODEL_DIMS.get(model_name)
        if resolved_size is None:
            raise ValueError(
                "Unknown embedding size for OpenAI model "
                f"'{model_name}'. Set EMBEDDING_VECTOR_SIZE."
            )
        self._vector_size = resolved_size
        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url or None,
            organization=organization or None,
            project=project or None,
        )

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        def _embed():
            response = self._client.embeddings.create(
                model=self.model_name,
                input=documents,
            )
            return [item.embedding for item in response.data]

        return await asyncio.to_thread(_embed)

    async def embed_query(self, query: str) -> list[float]:
        embeddings = await self.embed_documents([query])
        return embeddings[0]

    def get_vector_name(self) -> str:
        sanitized = self.model_name.replace("/", "-").replace(":", "-")
        return f"openai-{sanitized}"

    def get_vector_size(self) -> int:
        return self._vector_size
