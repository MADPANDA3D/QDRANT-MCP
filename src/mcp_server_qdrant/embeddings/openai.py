import asyncio

import httpx
from openai import AsyncOpenAI
from pydantic import SecretStr

from mcp_server_qdrant.embeddings.base import EmbeddingProvider

OPENAI_MODEL_DIMS = {
    "text-embedding-3-large": 3072,
    "text-embedding-3-small": 1536,
    "text-embedding-ada-002": 1536,
}

OPENAI_CONNECT_TIMEOUT_SECONDS = 10.0
OPENAI_READ_TIMEOUT_SECONDS = 60.0
OPENAI_WRITE_TIMEOUT_SECONDS = 30.0
OPENAI_POOL_TIMEOUT_SECONDS = 5.0
OPENAI_MAX_CONNECTIONS = 20
OPENAI_MAX_KEEPALIVE_CONNECTIONS = 10
OPENAI_KEEPALIVE_EXPIRY_SECONDS = 30.0


class OpenAIProvider(EmbeddingProvider):
    """OpenAI embeddings with an explicitly owned, proxy-free async client."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        vector_size: int | None = None,
        base_url: str | None = None,
        organization: str | None = None,
        project: str | None = None,
    ) -> None:
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
        self._api_key = SecretStr(api_key)
        self._base_url = base_url or None
        self._organization = organization or None
        self._project = project or None
        self._http_client: httpx.AsyncClient | None = None
        self._client: AsyncOpenAI | None = None
        self._client_lock = asyncio.Lock()
        self._closed = False

    def __repr__(self) -> str:
        return f"OpenAIProvider(vector_size={self._vector_size}, closed={self._closed})"

    @property
    def closed(self) -> bool:
        """Return whether this provider has released its owned client."""

        return self._closed

    async def _get_client(self) -> AsyncOpenAI:
        if self._closed:
            raise RuntimeError("OpenAI embedding provider is closed.")
        if self._client is not None:
            return self._client

        async with self._client_lock:
            if self._closed:
                raise RuntimeError("OpenAI embedding provider is closed.")
            if self._client is not None:
                return self._client

            timeout = httpx.Timeout(
                connect=OPENAI_CONNECT_TIMEOUT_SECONDS,
                read=OPENAI_READ_TIMEOUT_SECONDS,
                write=OPENAI_WRITE_TIMEOUT_SECONDS,
                pool=OPENAI_POOL_TIMEOUT_SECONDS,
            )
            limits = httpx.Limits(
                max_connections=OPENAI_MAX_CONNECTIONS,
                max_keepalive_connections=OPENAI_MAX_KEEPALIVE_CONNECTIONS,
                keepalive_expiry=OPENAI_KEEPALIVE_EXPIRY_SECONDS,
            )
            try:
                http_client = httpx.AsyncClient(
                    timeout=timeout,
                    limits=limits,
                    trust_env=False,
                    # A validated endpoint must not redirect the credential-bearing
                    # request to a destination that did not pass outbound validation.
                    follow_redirects=False,
                )
                client = AsyncOpenAI(
                    api_key=self._api_key.get_secret_value(),
                    base_url=self._base_url,
                    organization=self._organization,
                    project=self._project,
                    timeout=timeout,
                    max_retries=2,
                    http_client=http_client,
                )
            except Exception:
                if "http_client" in locals():
                    await http_client.aclose()
                raise RuntimeError("OpenAI embedding client initialization failed.") from None

            self._http_client = http_client
            self._client = client
            return client

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        client = await self._get_client()
        try:
            response = await client.embeddings.create(
                model=self.model_name,
                input=documents,
            )
        except Exception:
            raise RuntimeError("OpenAI embedding request failed.") from None
        return [item.embedding for item in response.data]

    async def embed_query(self, query: str) -> list[float]:
        embeddings = await self.embed_documents([query])
        return embeddings[0]

    async def aclose(self) -> None:
        """Idempotently close the explicitly owned OpenAI and HTTP clients."""

        async with self._client_lock:
            if self._closed:
                return
            self._closed = True
            client = self._client
            http_client = self._http_client
            self._client = None
            self._http_client = None
            self._api_key = SecretStr("")

        close_failed = False
        if client is not None:
            try:
                await client.close()
            except Exception:
                close_failed = True
        if http_client is not None and not http_client.is_closed:
            try:
                await http_client.aclose()
            except Exception:
                close_failed = True
        if close_failed:
            raise RuntimeError("OpenAI embedding client shutdown failed.") from None

    def get_vector_name(self) -> str:
        sanitized = self.model_name.replace("/", "-").replace(":", "-")
        return f"openai-{sanitized}"

    def get_vector_size(self) -> int:
        return self._vector_size
