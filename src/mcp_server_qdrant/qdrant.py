import json
import logging
import uuid
from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel
from qdrant_client import AsyncQdrantClient, models

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.settings import METADATA_PATH

logger = logging.getLogger(__name__)

Metadata = dict[str, Any]
ArbitraryFilter = dict[str, Any]


class Entry(BaseModel):
    """
    A single entry in the Qdrant collection.
    """

    content: str
    metadata: Metadata | None = None


class QdrantConnector:
    """
    Encapsulates the connection to a Qdrant server and all the methods to interact with it.
    :param qdrant_url: The URL of the Qdrant server.
    :param qdrant_api_key: The API key to use for the Qdrant server.
    :param collection_name: The name of the default collection to use. If not provided, each tool will require
                            the collection name to be provided.
    :param embedding_provider: The embedding provider to use.
    :param qdrant_local_path: The path to the storage directory for the Qdrant client, if local mode is used.
    """

    def __init__(
        self,
        qdrant_url: str | None,
        qdrant_api_key: str | None,
        collection_name: str | None,
        embedding_provider: EmbeddingProvider,
        vector_name_override: str | None = None,
        qdrant_local_path: str | None = None,
        field_indexes: dict[str, models.PayloadSchemaType] | None = None,
    ):
        self._qdrant_url = qdrant_url.rstrip("/") if qdrant_url else None
        self._qdrant_api_key = qdrant_api_key
        self._default_collection_name = collection_name
        self._embedding_provider = embedding_provider
        self._vector_name_override_set = vector_name_override is not None
        self._vector_name_override = (
            vector_name_override.strip() if vector_name_override is not None else None
        )
        if self._vector_name_override == "":
            self._vector_name_override = None
        self._client = AsyncQdrantClient(
            location=qdrant_url, api_key=qdrant_api_key, path=qdrant_local_path
        )
        self._field_indexes = field_indexes

    async def get_collection_names(self) -> list[str]:
        """
        Get the names of all collections in the Qdrant server.
        :return: A list of collection names.
        """
        response = await self._client.get_collections()
        return [collection.name for collection in response.collections]

    async def collection_exists(self, collection_name: str) -> bool:
        return await self._client.collection_exists(collection_name)

    async def store(self, entry: Entry, *, collection_name: str | None = None):
        """
        Store some information in the Qdrant collection, along with the specified metadata.
        :param entry: The entry to store in the Qdrant collection.
        :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                the default collection is used.
        """
        collection_name = collection_name or self._default_collection_name
        assert collection_name is not None
        await self._ensure_collection_exists(collection_name)

        # Embed the document
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.
        embeddings = await self._embedding_provider.embed_documents([entry.content])

        # Add to Qdrant
        vector_name = await self._resolve_vector_name(collection_name)
        payload = {"document": entry.content, METADATA_PATH: entry.metadata}
        vector_payload: list[float] | dict[str, list[float]]
        if vector_name is None:
            vector_payload = embeddings[0]
        else:
            vector_payload = {vector_name: embeddings[0]}
        await self._client.upsert(
            collection_name=collection_name,
            points=[
                models.PointStruct(
                    id=uuid.uuid4().hex,
                    vector=vector_payload,
                    payload=payload,
                )
            ],
        )

    async def search(
        self,
        query: str,
        *,
        collection_name: str | None = None,
        limit: int = 10,
        query_filter: models.Filter | None = None,
    ) -> list[Entry]:
        """
        Find points in the Qdrant collection. If there are no entries found, an empty list is returned.
        :param query: The query to use for the search.
        :param collection_name: The name of the collection to search in, optional. If not provided,
                                the default collection is used.
        :param limit: The maximum number of entries to return.
        :param query_filter: The filter to apply to the query, if any.

        :return: A list of entries found.
        """
        collection_name = collection_name or self._default_collection_name
        if not collection_name:
            raise ValueError("collection_name is required")
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            return []

        # Embed the query
        # ToDo: instead of embedding text explicitly, use `models.Document`,
        # it should unlock usage of server-side inference.

        query_vector = await self._embedding_provider.embed_query(query)
        vector_name = await self._resolve_vector_name(collection_name)

        # Search in Qdrant
        search_kwargs: dict[str, Any] = {
            "collection_name": collection_name,
            "query": query_vector,
            "limit": limit,
            "query_filter": query_filter,
            "with_payload": True,
        }
        if vector_name is not None:
            search_kwargs["using"] = vector_name

        search_results = await self._client.query_points(**search_kwargs)

        entries: list[Entry] = []
        for result in search_results.points:
            payload = result.payload or {}
            content = payload.get("document")
            if content is None:
                content = payload.get("content") or payload.get("text")
            if content is None:
                content = (
                    json.dumps(payload, sort_keys=True)
                    if payload
                    else f"(no document payload for point {result.id})"
                )
            metadata = payload.get(METADATA_PATH) or payload.get("metadata")
            entries.append(Entry(content=content, metadata=metadata))
        return entries

    async def _ensure_collection_exists(self, collection_name: str):
        """
        Ensure that the collection exists, creating it if necessary.
        :param collection_name: The name of the collection to ensure exists.
        """
        collection_exists = await self._client.collection_exists(collection_name)
        if not collection_exists:
            # Create the collection with the appropriate vector size
            vector_size = self._embedding_provider.get_vector_size()

            # Use the vector name as defined in the embedding provider
            vector_name = await self._resolve_vector_name(collection_name)
            if vector_name is None and self._vector_name_override_set:
                vectors_config: models.VectorsConfig = models.VectorParams(
                    size=vector_size, distance=models.Distance.COSINE
                )
            else:
                if vector_name is None:
                    vector_name = self._embedding_provider.get_vector_name()
                vectors_config = {
                    vector_name: models.VectorParams(
                        size=vector_size,
                        distance=models.Distance.COSINE,
                    )
                }
            await self._client.create_collection(
                collection_name=collection_name,
                vectors_config=vectors_config,
            )

            # Create payload indexes if configured

            if self._field_indexes:
                for field_name, field_type in self._field_indexes.items():
                    await self._client.create_payload_index(
                        collection_name=collection_name,
                        field_name=field_name,
                        field_schema=field_type,
                    )

    async def get_collection_info(
        self, collection_name: str | None = None
    ) -> models.CollectionInfo:
        collection_name = collection_name or self._default_collection_name
        if not collection_name:
            raise ValueError("collection_name is required")
        return await self._client.get_collection(collection_name)

    async def list_aliases(self) -> list[models.AliasDescription]:
        response = await self._client.get_aliases()
        return response.aliases

    async def list_collection_aliases(
        self, collection_name: str
    ) -> list[models.AliasDescription]:
        response = await self._client.get_collection_aliases(collection_name)
        return response.aliases

    async def list_snapshots(
        self, collection_name: str
    ) -> list[models.SnapshotDescription]:
        return await self._client.list_snapshots(collection_name)

    async def list_full_snapshots(self) -> list[models.SnapshotDescription]:
        return await self._client.list_full_snapshots()

    async def list_shard_snapshots(
        self, collection_name: str, shard_id: int
    ) -> list[models.SnapshotDescription]:
        return await self._client.list_shard_snapshots(collection_name, shard_id)

    async def get_collection_cluster_info(
        self, collection_name: str
    ) -> models.CollectionClusterInfo:
        return await self._client.collection_cluster_info(collection_name)

    async def get_collection_vectors(
        self, collection_name: str | None = None
    ) -> dict[str, dict[str, Any]]:
        info = await self.get_collection_info(collection_name)
        vector_map = self._parse_dense_vectors(info.config.params.vectors)
        return {
            self._display_vector_name(name): self._vector_params_to_dict(params)
            for name, params in vector_map.items()
        }

    async def get_collection_payload_schema(
        self, collection_name: str | None = None
    ) -> dict[str, Any]:
        info = await self.get_collection_info(collection_name)
        schema = {}
        if info.payload_schema:
            for field_name, field_schema in info.payload_schema.items():
                schema[field_name] = str(field_schema)
        return schema

    async def get_collection_summary(
        self, collection_name: str | None = None
    ) -> dict[str, Any]:
        info = await self.get_collection_info(collection_name)
        vectors = self._parse_dense_vectors(info.config.params.vectors)
        payload_schema = {}
        if info.payload_schema:
            for field_name, field_schema in info.payload_schema.items():
                payload_schema[field_name] = str(field_schema)
        summary: dict[str, Any] = {
            "status": str(info.status),
            "optimizer_status": str(info.optimizer_status),
            "points_count": info.points_count,
            "indexed_vectors_count": info.indexed_vectors_count,
            "segments_count": info.segments_count,
            "vectors": {
                self._display_vector_name(name): self._vector_params_to_dict(params)
                for name, params in vectors.items()
            },
            "payload_schema": payload_schema,
        }
        if info.warnings:
            summary["warnings"] = [str(warning) for warning in info.warnings]
        if getattr(info.config.params, "sparse_vectors", None):
            summary["sparse_vectors"] = list(info.config.params.sparse_vectors.keys())
        return summary

    async def resolve_vector_name(self, collection_name: str) -> str | None:
        return await self._resolve_vector_name(collection_name)

    def _vector_params_to_dict(self, params: models.VectorParams) -> dict[str, Any]:
        distance = params.distance
        if hasattr(distance, "value"):
            distance = distance.value
        return {
            "size": params.size,
            "distance": distance,
            "on_disk": params.on_disk,
        }

    def _display_vector_name(self, name: str | None) -> str:
        return "(default)" if name is None else name

    def _parse_dense_vectors(
        self, vectors: models.VectorsConfig
    ) -> dict[str | None, models.VectorParams]:
        if isinstance(vectors, models.VectorParams):
            return {None: vectors}
        if isinstance(vectors, Mapping):
            return dict(vectors)
        return {}

    async def _resolve_vector_name(self, collection_name: str) -> str | None:
        if self._vector_name_override_set:
            if not collection_name or not await self._client.collection_exists(
                collection_name
            ):
                return self._vector_name_override

            info = await self._client.get_collection(collection_name)
            vector_map = self._parse_dense_vectors(info.config.params.vectors)
            embedding_size = self._embedding_provider.get_vector_size()

            if self._vector_name_override is None:
                if len(vector_map) == 1 and next(iter(vector_map.keys())) is None:
                    params = next(iter(vector_map.values()))
                    if params.size != embedding_size:
                        raise ValueError(
                            "Embedding size does not match the collection vector size. "
                            f"Embedding size={embedding_size}, collection vector size={params.size}."
                        )
                    return None
                available = ", ".join(
                    self._display_vector_name(name) for name in vector_map.keys()
                )
                raise ValueError(
                    "Collection uses named vectors, but QDRANT_VECTOR_NAME is not set. "
                    f"Available: {available}."
                )

            if self._vector_name_override not in vector_map:
                available = ", ".join(
                    self._display_vector_name(name) for name in vector_map.keys()
                )
                raise ValueError(
                    "QDRANT_VECTOR_NAME is not configured in this collection. "
                    f"Available: {available}."
                )

            params = vector_map[self._vector_name_override]
            if params.size != embedding_size:
                raise ValueError(
                    "Embedding size does not match the collection vector size. "
                    f"Embedding size={embedding_size}, collection vector size={params.size}."
                )
            return self._vector_name_override

        if not collection_name:
            return self._embedding_provider.get_vector_name()

        if not await self._client.collection_exists(collection_name):
            return self._embedding_provider.get_vector_name()

        info = await self._client.get_collection(collection_name)
        vector_map = self._parse_dense_vectors(info.config.params.vectors)
        if not vector_map:
            return self._embedding_provider.get_vector_name()

        embedding_size = self._embedding_provider.get_vector_size()

        if len(vector_map) == 1:
            name, params = next(iter(vector_map.items()))
            if params.size != embedding_size:
                raise ValueError(
                    "Embedding size does not match the collection vector size. "
                    f"Embedding size={embedding_size}, collection vector size={params.size}."
                )
            return name

        matching = [
            name for name, params in vector_map.items() if params.size == embedding_size
        ]
        if len(matching) == 1:
            return matching[0]

        if len(matching) == 0:
            available = ", ".join(
                f"{self._display_vector_name(name)}(size={params.size})"
                for name, params in vector_map.items()
            )
            raise ValueError(
                "No vectors in the collection match the embedding size. "
                f"Embedding size={embedding_size}. Available: {available}. "
                "Set QDRANT_VECTOR_NAME or adjust EMBEDDING_MODEL."
            )

        options = ", ".join(self._display_vector_name(name) for name in matching)
        raise ValueError(
            "Multiple vectors match the embedding size. "
            f"Matches: {options}. Set QDRANT_VECTOR_NAME."
        )
