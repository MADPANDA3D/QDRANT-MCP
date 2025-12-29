import json
import logging
from typing import Annotated, Any, Optional

from fastmcp import Context, FastMCP
from mcp.types import EmbeddedResource, ImageContent, TextContent
from pydantic import Field
from qdrant_client import models

from mcp_server_qdrant.common.filters import make_indexes
from mcp_server_qdrant.common.func_tools import make_partial_function
from mcp_server_qdrant.common.wrap_filters import wrap_filters
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.embeddings.factory import create_embedding_provider
from mcp_server_qdrant.qdrant import ArbitraryFilter, Entry, Metadata, QdrantConnector
from mcp_server_qdrant.settings import (
    EmbeddingProviderSettings,
    QdrantSettings,
    ToolSettings,
)

logger = logging.getLogger(__name__)


# FastMCP is an alternative interface for declaring the capabilities
# of the server. Its API is based on FastAPI.
class QdrantMCPServer(FastMCP):
    """
    A MCP server for Qdrant.
    """

    def __init__(
        self,
        tool_settings: ToolSettings,
        qdrant_settings: QdrantSettings,
        embedding_provider_settings: Optional[EmbeddingProviderSettings] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
        name: str = "mcp-server-qdrant",
        instructions: str | None = None,
        **settings: Any,
    ):
        self.tool_settings = tool_settings
        self.qdrant_settings = qdrant_settings

        if embedding_provider_settings and embedding_provider:
            raise ValueError(
                "Cannot provide both embedding_provider_settings and embedding_provider"
            )

        if not embedding_provider_settings and not embedding_provider:
            raise ValueError(
                "Must provide either embedding_provider_settings or embedding_provider"
            )

        self.embedding_provider_settings: Optional[EmbeddingProviderSettings] = None
        self.embedding_provider: Optional[EmbeddingProvider] = None

        if embedding_provider_settings:
            self.embedding_provider_settings = embedding_provider_settings
            self.embedding_provider = create_embedding_provider(
                embedding_provider_settings
            )
        else:
            self.embedding_provider_settings = None
            self.embedding_provider = embedding_provider

        assert self.embedding_provider is not None, "Embedding provider is required"

        self.qdrant_connector = QdrantConnector(
            qdrant_settings.location,
            qdrant_settings.api_key,
            qdrant_settings.collection_name,
            self.embedding_provider,
            qdrant_settings.vector_name,
            qdrant_settings.local_path,
            make_indexes(qdrant_settings.filterable_fields_dict()),
        )

        super().__init__(name=name, instructions=instructions, **settings)

        self.setup_tools()

    async def _mcp_call_tool(
        self, key: str, arguments: dict[str, Any]
    ) -> list[TextContent | ImageContent | EmbeddedResource]:
        """
        Normalize tool arguments before validation to tolerate MCP clients
        that wrap arguments inside Airtable-like records (id/createdTime/fields).
        """
        if isinstance(arguments, dict) and self._tool_manager.has_tool(key):
            tool = self._tool_manager.get_tool(key)
            allowed = set(tool.parameters.get("properties", {}).keys())
            filtered: dict[str, Any] = {}
            fields = arguments.get("fields")
            if isinstance(fields, dict):
                for arg_key, arg_value in fields.items():
                    if arg_key in allowed:
                        filtered[arg_key] = arg_value
            for arg_key, arg_value in arguments.items():
                if arg_key in allowed:
                    filtered[arg_key] = arg_value
            arguments = filtered

        return await super()._mcp_call_tool(key, arguments)

    def format_entry(self, entry: Entry) -> str:
        """
        Feel free to override this method in your subclass to customize the format of the entry.
        """
        entry_metadata = json.dumps(entry.metadata) if entry.metadata else ""
        return f"<entry><content>{entry.content}</content><metadata>{entry_metadata}</metadata></entry>"

    def setup_tools(self):
        """
        Register the tools in the server.
        """

        async def store(
            ctx: Context,
            information: Annotated[str, Field(description="Text to store")],
            collection_name: Annotated[
                str, Field(description="The collection to store the information in")
            ],
            # The `metadata` parameter is defined as non-optional, but it can be None.
            # If we set it to be optional, some of the MCP clients, like Cursor, cannot
            # handle the optional parameter correctly.
            metadata: Annotated[
                Metadata | None,
                Field(
                    description="Extra metadata stored along with memorised information. Any json is accepted."
                ),
            ] = None,
        ) -> str:
            """
            Store some information in Qdrant.
            :param ctx: The context for the request.
            :param information: The information to store.
            :param metadata: JSON metadata to store with the information, optional.
            :param collection_name: The name of the collection to store the information in, optional. If not provided,
                                    the default collection is used.
            :return: A message indicating that the information was stored.
            """
            await ctx.debug(f"Storing information {information} in Qdrant")

            entry = Entry(content=information, metadata=metadata)

            await self.qdrant_connector.store(entry, collection_name=collection_name)
            if collection_name:
                return f"Remembered: {information} in collection {collection_name}"
            return f"Remembered: {information}"

        async def find(
            ctx: Context,
            query: Annotated[str, Field(description="What to search for")],
            collection_name: Annotated[
                str, Field(description="The collection to search in")
            ],
            query_filter: ArbitraryFilter | None = None,
        ) -> list[str] | None:
            """
            Find memories in Qdrant.
            :param ctx: The context for the request.
            :param query: The query to use for the search.
            :param collection_name: The name of the collection to search in, optional. If not provided,
                                    the default collection is used.
            :param query_filter: The filter to apply to the query.
            :return: A list of entries found or None.
            """

            # Log query_filter
            await ctx.debug(f"Query filter: {query_filter}")

            query_filter = models.Filter(**query_filter) if query_filter else None

            await ctx.debug(f"Finding results for query {query}")

            entries = await self.qdrant_connector.search(
                query,
                collection_name=collection_name,
                limit=self.qdrant_settings.search_limit,
                query_filter=query_filter,
            )
            if not entries:
                return None
            content = [
                f"Results for the query '{query}'",
            ]
            for entry in entries:
                content.append(self.format_entry(entry))
            return content

        def resolve_collection_name(collection_name: str) -> str:
            name = collection_name.strip() if collection_name else ""
            if name:
                return name
            if self.qdrant_settings.collection_name:
                return self.qdrant_settings.collection_name
            raise ValueError("collection_name is required")

        def format_json(data: dict[str, Any]) -> str:
            return json.dumps(data, indent=2, sort_keys=True, default=str)

        async def list_collections(ctx: Context) -> str:
            collections = await self.qdrant_connector.get_collection_names()
            if not collections:
                return "No collections found."
            return "Collections:\n- " + "\n- ".join(collections)

        async def collection_exists(ctx: Context, collection_name: str = "") -> str:
            name = resolve_collection_name(collection_name)
            exists = await self.qdrant_connector.collection_exists(name)
            return f"Collection '{name}' exists: {exists}"

        async def collection_info(ctx: Context, collection_name: str = "") -> str:
            name = resolve_collection_name(collection_name)
            summary = await self.qdrant_connector.get_collection_summary(name)
            summary["collection_name"] = name
            return format_json(summary)

        async def collection_stats(ctx: Context, collection_name: str = "") -> str:
            name = resolve_collection_name(collection_name)
            info = await self.qdrant_connector.get_collection_info(name)
            data = {
                "collection_name": name,
                "status": str(info.status),
                "optimizer_status": str(info.optimizer_status),
                "points_count": info.points_count,
                "indexed_vectors_count": info.indexed_vectors_count,
                "segments_count": info.segments_count,
            }
            if info.warnings:
                data["warnings"] = [str(warning) for warning in info.warnings]
            return format_json(data)

        async def collection_vectors(ctx: Context, collection_name: str = "") -> str:
            name = resolve_collection_name(collection_name)
            vectors = await self.qdrant_connector.get_collection_vectors(name)
            data = {"collection_name": name, "vectors": vectors}
            return format_json(data)

        async def collection_payload_schema(ctx: Context, collection_name: str = "") -> str:
            name = resolve_collection_name(collection_name)
            schema = await self.qdrant_connector.get_collection_payload_schema(name)
            data = {"collection_name": name, "payload_schema": schema}
            return format_json(data)

        async def get_vector_name(ctx: Context, collection_name: str = "") -> str:
            name = resolve_collection_name(collection_name)
            vector_name = await self.qdrant_connector.resolve_vector_name(name)
            label = "(default)" if vector_name is None else vector_name
            return f"Vector name for '{name}': {label}"

        async def list_aliases(ctx: Context) -> str:
            aliases = await self.qdrant_connector.list_aliases()
            if not aliases:
                return "No aliases found."
            formatted = "\n".join(
                f"- {alias.alias_name} -> {alias.collection_name}"
                for alias in aliases
            )
            return f"Aliases:\n{formatted}"

        async def collection_aliases(ctx: Context, collection_name: str = "") -> str:
            name = resolve_collection_name(collection_name)
            aliases = await self.qdrant_connector.list_collection_aliases(name)
            if not aliases:
                return f"No aliases found for collection '{name}'."
            formatted = "\n".join(
                f"- {alias.alias_name} -> {alias.collection_name}"
                for alias in aliases
            )
            return f"Aliases for '{name}':\n{formatted}"

        async def collection_cluster_info(ctx: Context, collection_name: str = "") -> str:
            name = resolve_collection_name(collection_name)
            info = await self.qdrant_connector.get_collection_cluster_info(name)
            data = info.model_dump() if hasattr(info, "model_dump") else info.dict()
            data["collection_name"] = name
            return format_json(data)

        async def list_snapshots(ctx: Context, collection_name: str = "") -> str:
            name = resolve_collection_name(collection_name)
            snapshots = await self.qdrant_connector.list_snapshots(name)
            if not snapshots:
                return f"No snapshots found for collection '{name}'."
            data = [
                {
                    "name": snap.name,
                    "creation_time": str(snap.creation_time),
                    "size": snap.size,
                    "checksum": snap.checksum,
                }
                for snap in snapshots
            ]
            return format_json({"collection_name": name, "snapshots": data})

        async def list_full_snapshots(ctx: Context) -> str:
            snapshots = await self.qdrant_connector.list_full_snapshots()
            if not snapshots:
                return "No full snapshots found."
            data = [
                {
                    "name": snap.name,
                    "creation_time": str(snap.creation_time),
                    "size": snap.size,
                    "checksum": snap.checksum,
                }
                for snap in snapshots
            ]
            return format_json({"full_snapshots": data})

        async def list_shard_snapshots(
            ctx: Context, shard_id: int, collection_name: str = ""
        ) -> str:
            name = resolve_collection_name(collection_name)
            if shard_id <= 0:
                raise ValueError("shard_id must be a positive integer")
            snapshots = await self.qdrant_connector.list_shard_snapshots(name, shard_id)
            if not snapshots:
                return f"No snapshots found for collection '{name}' shard {shard_id}."
            data = [
                {
                    "name": snap.name,
                    "creation_time": str(snap.creation_time),
                    "size": snap.size,
                    "checksum": snap.checksum,
                }
                for snap in snapshots
            ]
            return format_json(
                {"collection_name": name, "shard_id": shard_id, "snapshots": data}
            )

        find_foo = find
        store_foo = store

        filterable_conditions = (
            self.qdrant_settings.filterable_fields_dict_with_conditions()
        )

        if len(filterable_conditions) > 0:
            find_foo = wrap_filters(find_foo, filterable_conditions)
        elif not self.qdrant_settings.allow_arbitrary_filter:
            find_foo = make_partial_function(find_foo, {"query_filter": None})

        if self.qdrant_settings.collection_name:
            find_foo = make_partial_function(
                find_foo, {"collection_name": self.qdrant_settings.collection_name}
            )
            store_foo = make_partial_function(
                store_foo, {"collection_name": self.qdrant_settings.collection_name}
            )

        self.tool(
            find_foo,
            name="qdrant-find",
            description=self.tool_settings.tool_find_description,
        )

        self.tool(
            list_collections,
            name="qdrant-list-collections",
            description="List all Qdrant collections.",
        )
        self.tool(
            collection_exists,
            name="qdrant-collection-exists",
            description="Check if a collection exists.",
        )
        self.tool(
            collection_info,
            name="qdrant-collection-info",
            description="Get collection details including vectors and payload schema.",
        )
        self.tool(
            collection_stats,
            name="qdrant-collection-stats",
            description="Get basic collection statistics (points, segments, status).",
        )
        self.tool(
            collection_vectors,
            name="qdrant-collection-vectors",
            description="List vector names and sizes for a collection.",
        )
        self.tool(
            collection_payload_schema,
            name="qdrant-collection-payload-schema",
            description="Get payload schema for a collection.",
        )
        self.tool(
            get_vector_name,
            name="qdrant-get-vector-name",
            description="Resolve the vector name used by this MCP server.",
        )
        self.tool(
            list_aliases,
            name="qdrant-list-aliases",
            description="List all collection aliases.",
        )
        self.tool(
            collection_aliases,
            name="qdrant-collection-aliases",
            description="List aliases for a specific collection.",
        )
        self.tool(
            collection_cluster_info,
            name="qdrant-collection-cluster-info",
            description="Get cluster info for a collection.",
        )
        self.tool(
            list_snapshots,
            name="qdrant-list-snapshots",
            description="List snapshots for a collection.",
        )
        self.tool(
            list_full_snapshots,
            name="qdrant-list-full-snapshots",
            description="List full cluster snapshots.",
        )
        self.tool(
            list_shard_snapshots,
            name="qdrant-list-shard-snapshots",
            description="List snapshots for a specific shard.",
        )

        if not self.qdrant_settings.read_only:
            # Those methods can modify the database
            self.tool(
                store_foo,
                name="qdrant-store",
                description=self.tool_settings.tool_store_description,
            )
