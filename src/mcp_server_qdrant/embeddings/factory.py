from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.types import EmbeddingProviderType
from mcp_server_qdrant.settings import EmbeddingProviderSettings


def create_embedding_provider(settings: EmbeddingProviderSettings) -> EmbeddingProvider:
    """
    Create an embedding provider based on the specified type.
    :param settings: The settings for the embedding provider.
    :return: An instance of the specified embedding provider.
    """
    if settings.provider_type == EmbeddingProviderType.FASTEMBED:
        from mcp_server_qdrant.embeddings.fastembed import FastEmbedProvider

        return FastEmbedProvider(settings.model_name)
    if settings.provider_type == EmbeddingProviderType.OPENAI:
        from mcp_server_qdrant.embeddings.openai import OpenAIProvider

        return OpenAIProvider(
            api_key=settings.openai_api_key or "",
            model_name=settings.model_name,
            vector_size=settings.vector_size,
            base_url=settings.openai_base_url,
            organization=settings.openai_organization,
            project=settings.openai_project,
        )
    else:
        raise ValueError(f"Unsupported embedding provider: {settings.provider_type}")
