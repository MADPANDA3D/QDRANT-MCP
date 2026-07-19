"""Deterministic, provider-owned ToolManifest for the Qdrant MCP.

The manifest is built from the active FastMCP registry so read-only and admin
deployments publish truthful counts, while every descriptor is enriched from
provider-owned metadata rather than runtime credentials or user arguments.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any

from mcp_server_qdrant.common.telemetry import BUILD_SHA

SCHEMA_VERSION = "1.0.0"
SERVICE_ID = "qdrant"
SERVICE_ALIASES = ("qdrant-mcp", "qdrant_mcp", "vector memory")
CATALOG_VERSION = "qdrant-2026.07.19.1"
DOCUMENTATION_URL = "https://github.com/MADPANDA3D/QDRANT-MCP/blob/v2.0.0/README.md#tools"
ENDPOINT_COVERAGE_URL = (
    "https://github.com/MADPANDA3D/QDRANT-MCP/blob/main/docs/endpoint-coverage.md"
)

_SHA_PATTERN = re.compile(r"^[a-fA-F0-9]{7,64}$")
_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_CONTRACT_TIERS = {"agent_ready", "legacy", "hidden"}
_RISK_LEVELS = {"read", "write", "destructive"}


CATEGORY_TOOLS: dict[str, tuple[str, ...]] = {
    "navigation": (
        "check_configuration",
        "list_capabilities",
        "get_endpoint_coverage",
        "get_tool_usage",
        "find_tools",
        "qdrant-check-configuration",
        "qdrant-list-capabilities",
        "qdrant-get-endpoint-coverage",
        "qdrant-get-tool-usage",
        "qdrant-find-tools",
    ),
    "configuration": ("qdrant-health-check",),
    "retrieval": (
        "qdrant-build-context",
        "qdrant-study-search",
        "qdrant-find",
        "qdrant-find-short-term",
        "qdrant-recommend-memories",
        "qdrant-validate-memory",
    ),
    "points": (
        "qdrant-list-points",
        "qdrant-get-points",
        "qdrant-count-points",
    ),
    "uploads": (
        "qdrant-start-file-upload",
        "qdrant-append-file-upload",
        "qdrant-finish-file-upload",
    ),
    "ingest": (
        "qdrant-store",
        "qdrant-cache-memory",
        "qdrant-promote-short-term",
        "qdrant-ingest-with-validation",
        "qdrant-ingest-document",
        "qdrant-ingest-manifest",
        "qdrant-ingest-class-manifest",
        "qdrant-ingest-school-manifest",
        "qdrant-ingest-textbook",
    ),
    "jobs": (
        "qdrant-submit-job",
        "qdrant-get-ingest-status",
        "qdrant-cancel-ingest",
        "qdrant-job-status",
        "qdrant-job-progress",
        "qdrant-job-logs",
        "qdrant-job-result",
        "qdrant-cancel-job",
    ),
    "collections": (
        "qdrant-list-collections",
        "qdrant-create-collection",
        "qdrant-collection-exists",
        "qdrant-describe-collection",
        "qdrant-summarize-collection-schema",
        "qdrant-suggest-filters",
        "qdrant-collection-info",
        "qdrant-collection-stats",
        "qdrant-collection-vectors",
        "qdrant-collection-payload-schema",
        "qdrant-optimizer-status",
        "qdrant-metrics-snapshot",
        "qdrant-get-vector-name",
        "qdrant-list-aliases",
        "qdrant-collection-aliases",
        "qdrant-collection-cluster-info",
    ),
    "snapshots": (
        "qdrant-list-snapshots",
        "qdrant-create-snapshot",
        "qdrant-list-full-snapshots",
        "qdrant-list-shard-snapshots",
    ),
    "maintenance": (
        "qdrant-audit-memories",
        "qdrant-find-near-duplicates",
        "qdrant-ensure-payload-indexes",
        "qdrant-backfill-memory-contract",
        "qdrant-update-point",
        "qdrant-patch-payload",
        "qdrant-tag-memories",
        "qdrant-link-memories",
        "qdrant-reembed-points",
        "qdrant-migrate-collection-embedding",
        "qdrant-bulk-patch",
        "qdrant-dedupe-memories",
        "qdrant-merge-duplicates",
        "qdrant-expire-memories",
        "qdrant-expire-short-term",
        "qdrant-update-optimizer-config",
        "qdrant-delete-points",
        "qdrant-delete-by-filter",
        "qdrant-delete-document",
    ),
}

_CATEGORY_BY_TOOL: dict[str, str] = {}
for _category, _tool_names in CATEGORY_TOOLS.items():
    for _tool_name in _tool_names:
        if _tool_name in _CATEGORY_BY_TOOL:
            raise RuntimeError(f"Duplicate Qdrant category assignment: {_tool_name}")
        _CATEGORY_BY_TOOL[_tool_name] = _category


TOOL_PURPOSES: dict[str, str] = {
    "check_configuration": "check service access, Qdrant, embedding, cache, and upload readiness without returning credential values",
    "list_capabilities": "inspect capability groups, catalog counts, and optionally the complete lossless ToolManifest",
    "get_endpoint_coverage": "review covered Qdrant API areas and intentionally excluded provider operations",
    "get_tool_usage": "retrieve one complete tool descriptor by native name, canonical name, or compatibility alias",
    "find_tools": "find Qdrant tools for a multi-word task using deterministic local catalog search",
    "qdrant-check-configuration": "check service access, Qdrant, embedding, cache, and upload readiness without returning credential values",
    "qdrant-list-capabilities": "inspect capability groups, catalog counts, and optionally the complete lossless ToolManifest",
    "qdrant-get-endpoint-coverage": "review covered Qdrant API areas and intentionally excluded provider operations",
    "qdrant-get-tool-usage": "retrieve one complete tool descriptor by native name, canonical name, or compatibility alias",
    "qdrant-find-tools": "find Qdrant tools for a multi-word task using deterministic local catalog search",
    "qdrant-health-check": "verify Qdrant collection, index, embedding, and configuration health",
    "qdrant-build-context": "build a cited, token-budgeted context pack from relevant vector-memory matches",
    "qdrant-study-search": "search study material with compact results and exact academic metadata filters",
    "qdrant-find": "search vector memories by semantic meaning and optional filters",
    "qdrant-find-short-term": "search the configured short-term memory collection",
    "qdrant-recommend-memories": "retrieve memories related to positive examples while excluding negative examples",
    "qdrant-validate-memory": "validate a proposed memory contract and receive governance recommendations before ingest",
    "qdrant-list-points": "scroll point identifiers and optional payloads with bounded pagination",
    "qdrant-get-points": "retrieve known Qdrant points by stable identifier",
    "qdrant-count-points": "count points matching an optional structured memory filter",
    "qdrant-audit-memories": "audit stored memories for missing fields, invalid payloads, and duplicates",
    "qdrant-find-near-duplicates": "find semantic near-duplicate point groups without changing them",
    "qdrant-submit-job": "submit a legacy generic maintenance job when no typed direct workflow is suitable",
    "qdrant-start-file-upload": "start a bounded temporary upload session for caller-local document bytes",
    "qdrant-append-file-upload": "append one ordered base64 chunk to an active bounded upload session",
    "qdrant-finish-file-upload": "finalize an upload and return upload URI aliases for a later ingest call",
    "qdrant-ingest-textbook": "submit a bounded asynchronous textbook PDF ingest from HTTP(S) or a finalized upload URI",
    "qdrant-get-ingest-status": "read textbook-ingest progress, metrics, logs, and structured errors",
    "qdrant-cancel-ingest": "stop a running textbook-ingest job",
    "qdrant-job-status": "read the current state and safe summary of a maintenance job",
    "qdrant-job-progress": "read progress counters and phase for a maintenance job",
    "qdrant-job-logs": "read a bounded tail of safe maintenance-job logs",
    "qdrant-job-result": "retrieve the final result of a completed maintenance job",
    "qdrant-cancel-job": "stop a running maintenance job",
    "qdrant-list-collections": "list Qdrant collections visible to the configured account",
    "qdrant-create-collection": "create a collection using the active embedding vector contract and default indexes",
    "qdrant-collection-exists": "check whether one named Qdrant collection exists",
    "qdrant-describe-collection": "summarize one collection's size, vectors, indexes, and useful retrieval paths",
    "qdrant-summarize-collection-schema": "map indexed payload schema to supported memory-filter fields",
    "qdrant-suggest-filters": "sample indexed metadata values and suggest exact retrieval filters",
    "qdrant-collection-info": "retrieve a collection's configuration, vectors, and payload schema",
    "qdrant-collection-stats": "retrieve collection point, segment, and status statistics",
    "qdrant-collection-vectors": "list a collection's vector names, dimensions, and distance settings",
    "qdrant-collection-payload-schema": "list a collection's payload schema and indexed fields",
    "qdrant-optimizer-status": "inspect optimizer configuration and index coverage for a collection",
    "qdrant-metrics-snapshot": "capture bounded collection and index health metrics",
    "qdrant-get-vector-name": "resolve the active vector name used by this MCP",
    "qdrant-list-aliases": "list all collection aliases visible to the configured account",
    "qdrant-collection-aliases": "list aliases that resolve to one collection",
    "qdrant-collection-cluster-info": "inspect shard and cluster placement for one collection",
    "qdrant-list-snapshots": "list snapshots for one collection",
    "qdrant-create-snapshot": "create a new point-in-time collection snapshot",
    "qdrant-list-full-snapshots": "list full-storage snapshots on the configured Qdrant server",
    "qdrant-list-shard-snapshots": "list snapshots for one collection shard",
    "qdrant-store": "store one normalized memory and return its stable point identifier",
    "qdrant-cache-memory": "store one short-term memory with a bounded expiry",
    "qdrant-promote-short-term": "copy selected short-term memories into a long-term collection and optionally remove their source records",
    "qdrant-ingest-with-validation": "validate, govern, optionally quarantine, and store one memory",
    "qdrant-ingest-document": "extract, chunk, embed, and store a supported document source",
    "qdrant-ingest-manifest": "run the legacy generic alias for the typed school-manifest ingest workflow",
    "qdrant-ingest-class-manifest": "run the legacy class-manifest alias for the typed school-manifest ingest workflow",
    "qdrant-ingest-school-manifest": "ingest a batch of school capture items with shared metadata and search-back verification",
    "qdrant-ensure-payload-indexes": "create missing payload indexes required by the maintained memory contract",
    "qdrant-backfill-memory-contract": "preview or fill missing memory-contract and governance fields",
    "qdrant-update-point": "replace a point's content and metadata while recomputing its embedding",
    "qdrant-patch-payload": "overwrite selected payload metadata fields on a known point",
    "qdrant-tag-memories": "append or replace labels on selected memories",
    "qdrant-link-memories": "replace or extend related-memory links and associations",
    "qdrant-reembed-points": "overwrite selected vectors with embeddings from the active model",
    "qdrant-migrate-collection-embedding": "add and populate the active named vector on a legacy collection",
    "qdrant-bulk-patch": "overwrite payload fields on a bounded point set selected by IDs or filter",
    "qdrant-dedupe-memories": "find exact duplicate memories and optionally remove redundant points",
    "qdrant-merge-duplicates": "merge duplicate points into one canonical record and remove redundant records",
    "qdrant-expire-memories": "archive or delete memories whose expiry timestamp has passed",
    "qdrant-expire-short-term": "delete expired points from the short-term memory collection",
    "qdrant-update-optimizer-config": "overwrite bounded optimizer settings for one collection",
    "qdrant-delete-points": "permanently delete selected points by identifier",
    "qdrant-delete-by-filter": "permanently delete every point matching a bounded filter",
    "qdrant-delete-document": "permanently delete all chunks belonging to one document identifier",
}

if set(TOOL_PURPOSES) != set(_CATEGORY_BY_TOOL):
    raise RuntimeError(
        "Qdrant ToolManifest metadata drift: "
        f"missing purposes={sorted(set(_CATEGORY_BY_TOOL) - set(TOOL_PURPOSES))}; "
        f"stale purposes={sorted(set(TOOL_PURPOSES) - set(_CATEGORY_BY_TOOL))}."
    )


SPECIAL_ALIASES: dict[str, tuple[str, ...]] = {
    "check_configuration": ("configuration_status",),
    "list_capabilities": ("get_manifest",),
    "get_endpoint_coverage": ("endpoint_coverage",),
    "get_tool_usage": ("describe_tool",),
    "find_tools": ("search_tools", "discover_tools"),
    "qdrant-find": ("semantic_search", "vector_search"),
    "qdrant-build-context": ("build_context", "retrieve_context"),
    "qdrant-study-search": ("study_search", "search_school"),
    "qdrant-store": ("remember", "store_memory"),
    "qdrant-ingest-school-manifest": ("ingest_school_manifest",),
}

NAVIGATION_ROLES = {
    "check_configuration": "configuration",
    "list_capabilities": "catalog",
    "get_endpoint_coverage": "coverage",
    "get_tool_usage": "reference",
    "find_tools": "discovery",
}

STANDARD_NAVIGATION_TOOLS = frozenset(NAVIGATION_ROLES)
PREFIXED_NAVIGATION_TOOLS = frozenset(
    {
        "qdrant-check-configuration",
        "qdrant-list-capabilities",
        "qdrant-get-endpoint-coverage",
        "qdrant-get-tool-usage",
        "qdrant-find-tools",
    }
)
LOCAL_NAVIGATION_TOOLS = STANDARD_NAVIGATION_TOOLS | PREFIXED_NAVIGATION_TOOLS

LEGACY_TOOLS = {
    "qdrant-submit-job",
    "qdrant-ingest-manifest",
    "qdrant-ingest-class-manifest",
    *PREFIXED_NAVIGATION_TOOLS,
}

READ_ONLY_TOOLS = {
    name
    for name in _CATEGORY_BY_TOOL
    if name
    not in {
        "qdrant-submit-job",
        "qdrant-start-file-upload",
        "qdrant-append-file-upload",
        "qdrant-finish-file-upload",
        "qdrant-ingest-textbook",
        "qdrant-cancel-ingest",
        "qdrant-cancel-job",
        "qdrant-create-collection",
        "qdrant-create-snapshot",
        "qdrant-store",
        "qdrant-cache-memory",
        "qdrant-promote-short-term",
        "qdrant-ingest-with-validation",
        "qdrant-ingest-document",
        "qdrant-ingest-manifest",
        "qdrant-ingest-class-manifest",
        "qdrant-ingest-school-manifest",
        "qdrant-ensure-payload-indexes",
        "qdrant-backfill-memory-contract",
        "qdrant-update-point",
        "qdrant-patch-payload",
        "qdrant-tag-memories",
        "qdrant-link-memories",
        "qdrant-reembed-points",
        "qdrant-migrate-collection-embedding",
        "qdrant-bulk-patch",
        "qdrant-dedupe-memories",
        "qdrant-merge-duplicates",
        "qdrant-expire-memories",
        "qdrant-expire-short-term",
        "qdrant-update-optimizer-config",
        "qdrant-delete-points",
        "qdrant-delete-by-filter",
        "qdrant-delete-document",
    }
}

DESTRUCTIVE_TOOLS = {
    "qdrant-submit-job",
    "qdrant-cancel-ingest",
    "qdrant-cancel-job",
    "qdrant-promote-short-term",
    "qdrant-update-point",
    "qdrant-patch-payload",
    "qdrant-tag-memories",
    "qdrant-link-memories",
    "qdrant-reembed-points",
    "qdrant-bulk-patch",
    "qdrant-dedupe-memories",
    "qdrant-merge-duplicates",
    "qdrant-expire-memories",
    "qdrant-expire-short-term",
    "qdrant-update-optimizer-config",
    "qdrant-delete-points",
    "qdrant-delete-by-filter",
    "qdrant-delete-document",
}


ENDPOINT_COVERAGE: tuple[dict[str, Any], ...] = (
    {
        "feature": "collections",
        "status": "covered",
        "tools": [
            "qdrant-list-collections",
            "qdrant-create-collection",
            "qdrant-collection-info",
        ],
        "documentationUrl": "https://api.qdrant.tech/api-reference/collections",
        "notes": "Collection creation and bounded inspection are covered.",
    },
    {
        "feature": "points and memory workflows",
        "status": "covered",
        "tools": ["qdrant-find", "qdrant-store", "qdrant-list-points"],
        "documentationUrl": "https://api.qdrant.tech/api-reference/points",
        "notes": "Point operations are exposed through typed memory-safe workflows.",
    },
    {
        "feature": "payload indexes",
        "status": "covered",
        "tools": ["qdrant-ensure-payload-indexes", "qdrant-collection-payload-schema"],
        "documentationUrl": "https://api.qdrant.tech/api-reference/indexes",
        "notes": "Maintained memory-contract indexes can be inspected and created.",
    },
    {
        "feature": "aliases",
        "status": "read_only",
        "tools": ["qdrant-list-aliases", "qdrant-collection-aliases"],
        "documentationUrl": "https://api.qdrant.tech/api-reference/aliases/get-collections-aliases",
        "notes": "Alias reads are covered; mutation waits for a dedicated preview contract.",
    },
    {
        "feature": "snapshots",
        "status": "partial",
        "tools": [
            "qdrant-list-snapshots",
            "qdrant-create-snapshot",
            "qdrant-list-full-snapshots",
            "qdrant-list-shard-snapshots",
        ],
        "documentationUrl": "https://api.qdrant.tech/api-reference/snapshots/list-snapshots",
        "notes": "Bounded snapshot metadata and creation are covered. Restore and binary transfer are excluded from the public contract.",
    },
    {
        "feature": "cluster and service",
        "status": "partial",
        "tools": ["qdrant-collection-cluster-info", "qdrant-health-check"],
        "documentationUrl": "https://api.qdrant.tech/api-reference/service",
        "notes": "Bounded readiness and collection cluster reads are covered; infrastructure-sensitive telemetry is excluded.",
    },
)


def normalize_terms(value: Any) -> tuple[str, ...]:
    """Normalize punctuation, hyphens, underscores, camel case, and whitespace."""
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", str(value or ""))
    return tuple(_TOKEN_PATTERN.findall(text.lower()))


def canonical_json(value: Any) -> str:
    def normalize(entry: Any) -> Any:
        if isinstance(entry, dict):
            return {key: normalize(item) for key, item in entry.items()}
        if isinstance(entry, list):
            return [normalize(item) for item in entry]
        if isinstance(entry, float) and math.isfinite(entry):
            # JSON.stringify canonicalizes integral IEEE-754 numbers without a
            # decimal suffix; match the shared broker descriptor hasher exactly.
            if entry == 0:
                return 0
            if entry.is_integer():
                return int(entry)
        return entry

    return json.dumps(normalize(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def descriptor_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def get_build_sha(value: str | None = None) -> str:
    """Return a commit-shaped build identifier and never arbitrary env content."""
    candidate = str(value if value is not None else BUILD_SHA).strip()
    return candidate.lower() if _SHA_PATTERN.fullmatch(candidate) else "unknown"


def _title(tool_name: str) -> str:
    words = tool_name.removeprefix("qdrant-").replace("-", " ").replace("_", " ").split()
    return "Qdrant " + " ".join(
        word.upper() if word in {"url", "id"} else word.title() for word in words
    )


def _aliases(tool_name: str) -> list[str]:
    snake_name = tool_name.replace("-", "_")
    short_name = tool_name.removeprefix("qdrant-").replace("-", "_")
    aliases = [snake_name, *SPECIAL_ALIASES.get(tool_name, ())]
    if short_name not in STANDARD_NAVIGATION_TOOLS:
        aliases.append(short_name)
    return sorted({alias for alias in aliases if alias and alias != tool_name})


def _risk_for(tool_name: str) -> str:
    if tool_name in DESTRUCTIVE_TOOLS:
        return "destructive"
    if tool_name in READ_ONLY_TOOLS:
        return "read"
    return "write"


def annotations_for(tool_name: str) -> dict[str, bool]:
    if tool_name not in _CATEGORY_BY_TOOL:
        raise RuntimeError(f"Unknown Qdrant ToolManifest tool: {tool_name}")
    read_only = tool_name in READ_ONLY_TOOLS
    return {
        "readOnlyHint": read_only,
        "destructiveHint": tool_name in DESTRUCTIVE_TOOLS,
        "openWorldHint": tool_name not in LOCAL_NAVIGATION_TOOLS,
        "idempotentHint": read_only,
    }


def complete_description(tool_name: str) -> str:
    purpose = TOOL_PURPOSES[tool_name].rstrip(".")
    annotations = annotations_for(tool_name)
    if annotations["readOnlyHint"]:
        impact = "It does not intentionally change Qdrant or MCP state."
    elif annotations["destructiveHint"]:
        impact = (
            "It can overwrite, remove, cancel, restore, or otherwise irreversibly "
            "change state and therefore requires the manifest-defined client confirmation."
        )
    else:
        impact = "It creates or extends Qdrant or MCP-managed state without deleting existing provider data by default."
    provider = (
        "It reads the provider-owned catalog in memory and does not contact Qdrant."
        if tool_name in LOCAL_NAVIGATION_TOOLS
        else "It contacts the configured Qdrant service or uses MCP-owned job/upload state."
    )
    legacy = (
        " Prefer its typed replacement when one is available because this tool remains in the legacy tier."
        if tool_name in LEGACY_TOOLS
        else ""
    )
    return (
        f"Use this when you need to {purpose}. {impact} {provider} Returns normalized "
        "provider data in data plus safe request and trace metadata in meta. Requires an "
        f"authenticated MCP service request and any provider configuration named by the input schema.{legacy}"
    )


def runtime_registration(tool_name: str) -> dict[str, Any]:
    """Return canonical metadata for native FastMCP registration."""
    return {
        "title": _title(tool_name),
        "description": complete_description(tool_name),
        "annotations": annotations_for(tool_name),
        "meta": {
            "madpanda": {
                "serviceId": SERVICE_ID,
                "category": _CATEGORY_BY_TOOL[tool_name],
                "tier": "legacy" if tool_name in LEGACY_TOOLS else "agent_ready",
                "catalogVersion": CATALOG_VERSION,
                "navigationRole": NAVIGATION_ROLES.get(tool_name),
            }
        },
    }


def _fallback_parameter_description(tool_name: str, parameter_name: str) -> str:
    common = {
        "include_descriptors": "When true, include the complete ordered ToolManifest under manifest for client or broker catalog ingestion.",
        "query": "Natural-language task or tool terms; every punctuation-normalized token must match the descriptor.",
        "category": "Optional exact ToolManifest category filter; empty includes every category.",
        "risk": "Optional risk filter: read, write, or destructive.",
        "limit": "Maximum ranked results to return, clamped from 1 through 25.",
        "include_legacy": "When true, include legacy tools; hidden tools remain excluded.",
        "feature": "Optional endpoint-coverage feature terms; empty returns every maintained coverage area.",
        "tool_name": "Native tool name, canonical service-qualified name, or exact compatibility alias.",
        "tool": "Legacy compatibility alias for tool_name.",
        "confirm": "Native safety confirmation used only where the tool schema documents it; broker-level confirmation remains out-of-band.",
        "dry_run": "When true, validate and return a bounded preview without applying provider mutations.",
    }
    return common.get(
        parameter_name,
        f"Validated {parameter_name.replace('_', ' ')} input used by {_title(tool_name)}; follow the type, limits, and defaults in this schema.",
    )


def _enrich_schema_node(node: Any, tool_name: str) -> None:
    if isinstance(node, list):
        for item in node:
            _enrich_schema_node(item, tool_name)
        return
    if not isinstance(node, dict):
        return
    properties = node.get("properties")
    if isinstance(properties, dict):
        for parameter_name, parameter_schema in properties.items():
            if isinstance(parameter_schema, dict):
                if not str(parameter_schema.get("description") or "").strip():
                    parameter_schema["description"] = _fallback_parameter_description(
                        tool_name, str(parameter_name)
                    )
                _enrich_schema_node(parameter_schema, tool_name)
    for keyword in ("allOf", "anyOf", "oneOf", "prefixItems"):
        _enrich_schema_node(node.get(keyword), tool_name)
    _enrich_schema_node(node.get("items"), tool_name)
    for definitions_key in ("$defs", "definitions"):
        definitions = node.get(definitions_key)
        if isinstance(definitions, dict):
            for definition in definitions.values():
                _enrich_schema_node(definition, tool_name)


def enrich_input_schema(tool_name: str, schema_value: Mapping[str, Any]) -> dict[str, Any]:
    schema = copy.deepcopy(dict(schema_value))
    schema["title"] = f"{_title(tool_name)} input"
    schema["description"] = f"Validated input contract for the {tool_name} native Qdrant MCP tool."
    _enrich_schema_node(schema, tool_name)
    return schema


def _meta_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "description": "Secret-safe request, trace, size, timing, and warning metadata.",
        "properties": {
            "request_id": {
                "type": "string",
                "description": "Stable request trace identifier.",
            },
            "server_version": {
                "type": "string",
                "description": "Installed Qdrant MCP package version.",
            },
            "build_sha": {
                "type": "string",
                "description": "Deployed source revision when available, otherwise unknown.",
            },
            "server_instance_id": {
                "type": "string",
                "description": "Ephemeral server-process identifier.",
            },
            "server_uptime_ms": {
                "type": "integer",
                "description": "Server process uptime in milliseconds.",
            },
            "elapsed_ms": {
                "type": "integer",
                "description": "Tool execution time in milliseconds.",
            },
            "bytes_in": {
                "type": "integer",
                "description": "Approximate serialized input byte count.",
            },
            "bytes_out": {
                "type": "integer",
                "description": "Approximate serialized output byte count.",
            },
            "serialization_ms": {
                "type": "integer",
                "description": "Response serialization time in milliseconds.",
            },
            "warnings": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Safe recovery warnings.",
            },
        },
        "additionalProperties": True,
    }


def _manifest_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "description": "Complete versioned Qdrant provider ToolManifest.",
        "required": [
            "schemaVersion",
            "serviceId",
            "catalogVersion",
            "buildSha",
            "descriptorHash",
            "counts",
            "tools",
        ],
        "properties": {
            "schemaVersion": {"const": SCHEMA_VERSION},
            "serviceId": {"const": SERVICE_ID},
            "serviceAliases": {"type": "array", "items": {"type": "string"}},
            "catalogVersion": {"type": "string", "minLength": 1},
            "buildSha": {"type": "string", "minLength": 1},
            "descriptorHash": {"type": "string", "pattern": "^[a-f0-9]{64}$"},
            "counts": {
                "type": "object",
                "required": ["raw", "agentReady", "legacy", "hidden"],
                "properties": {
                    "raw": {"type": "integer", "minimum": 0},
                    "agentReady": {"type": "integer", "minimum": 0},
                    "legacy": {"type": "integer", "minimum": 0},
                    "hidden": {"type": "integer", "minimum": 0},
                    "documented": {"type": "integer", "minimum": 0},
                },
                "additionalProperties": False,
            },
            "tools": {
                "type": "array",
                "items": {"type": "object", "additionalProperties": True},
            },
        },
        "additionalProperties": False,
    }


def output_schema_for(tool_name: str) -> dict[str, Any]:
    data_schema: dict[str, Any] = {
        "type": "object",
        "description": f"Normalized result data for {tool_name}: {TOOL_PURPOSES[tool_name]}.",
        "additionalProperties": True,
    }
    if tool_name in {"list_capabilities", "qdrant-list-capabilities"}:
        data_schema = {
            "type": "object",
            "description": "Qdrant capability summary and optional complete ToolManifest.",
            "required": [
                "service",
                "service_id",
                "tool_count",
                "catalog_version",
                "counts",
                "groups",
                "catalog_groups",
            ],
            "properties": {
                "service": {
                    "type": "string",
                    "description": "Canonical provider service identifier.",
                },
                "service_id": {
                    "const": SERVICE_ID,
                    "description": "Canonical ToolManifest service identifier.",
                },
                "tool_count": {
                    "type": "integer",
                    "description": "Compatibility count of active native tools.",
                },
                "catalog_version": {
                    "type": "string",
                    "description": "Published Qdrant catalog version.",
                },
                "counts": {
                    "type": "object",
                    "additionalProperties": True,
                    "description": "Raw and tier-specific descriptor counts.",
                },
                "groups": {
                    "type": "array",
                    "items": {"type": "object", "additionalProperties": True},
                    "description": "Compact category and risk summaries.",
                },
                "catalog_groups": {
                    "type": "array",
                    "items": {"type": "object", "additionalProperties": True},
                    "description": "Complete category-level catalog counts.",
                },
                "manifest": _manifest_schema(),
            },
            "additionalProperties": False,
        }
    elif tool_name in {"get_tool_usage", "qdrant-get-tool-usage"}:
        data_schema = {
            "type": "object",
            "description": "One lossless Qdrant descriptor and safe usage guidance.",
            "required": ["descriptor"],
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Compatibility native tool name.",
                },
                "description": {
                    "type": "string",
                    "description": "Compatibility full tool description.",
                },
                "input_schema": {
                    "type": "object",
                    "additionalProperties": True,
                    "description": "Compatibility complete input schema.",
                },
                "annotations": {
                    "type": "object",
                    "additionalProperties": True,
                    "description": "Compatibility MCP safety annotations.",
                },
                "descriptor": {
                    "type": "object",
                    "additionalProperties": True,
                    "description": "Complete catalog descriptor.",
                },
                "guidance": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "Optional workflow-specific next-step guidance.",
                },
            },
            "additionalProperties": False,
        }
    elif tool_name in {"find_tools", "qdrant-find-tools"}:
        data_schema = {
            "type": "object",
            "description": "Deterministically ranked Qdrant tool matches.",
            "required": ["query", "count", "matches"],
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Normalized caller search task.",
                },
                "count": {
                    "type": "integer",
                    "description": "Number of returned matches.",
                },
                "matches": {
                    "type": "array",
                    "items": {"type": "object", "additionalProperties": True},
                    "description": "Ranked compact descriptor matches.",
                },
            },
            "additionalProperties": False,
        }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "description": "Qdrant MCP normalized response envelope.",
        "required": ["data", "meta"],
        "properties": {
            "data": data_schema,
            "meta": _meta_schema(),
        },
        "additionalProperties": False,
    }


def _registered_tool_mapping(registered_tools: Any) -> Mapping[str, Any]:
    if isinstance(registered_tools, Mapping):
        return registered_tools
    for attribute in ("_tools", "tools"):
        candidate = getattr(registered_tools, attribute, None)
        if isinstance(candidate, Mapping):
            return candidate
    raise TypeError("Registered tools must be a mapping or FastMCP ToolManager.")


def _input_schema_for(tool: Any) -> Mapping[str, Any]:
    if isinstance(tool, Mapping):
        candidate = tool.get("inputSchema") or tool.get("parameters") or tool
    else:
        candidate = getattr(tool, "parameters", None)
    if not isinstance(candidate, Mapping):
        raise TypeError("Registered Qdrant tool does not expose an input JSON schema.")
    return candidate


def _deprecation_for(tool_name: str) -> dict[str, Any]:
    if tool_name in {"qdrant-ingest-manifest", "qdrant-ingest-class-manifest"}:
        return {
            "deprecated": True,
            "since": "2026-07-12",
            "replacement": "qdrant-ingest-school-manifest",
            "sunsetAt": None,
            "message": "Compatibility alias retained for existing callers; use the typed school manifest tool for new workflows.",
        }
    return {
        "deprecated": False,
        "since": None,
        "replacement": None,
        "sunsetAt": None,
        "message": None,
    }


def _confirmation_for(tool_name: str) -> dict[str, Any]:
    if tool_name not in DESTRUCTIVE_TOOLS:
        return {"required": False, "parameter": None, "exactPhrase": None, "when": None}
    phrase = "CONFIRM QDRANT " + tool_name.removeprefix("qdrant-").replace("-", " ").upper()
    return {
        "required": True,
        "parameter": None,
        "exactPhrase": phrase,
        "when": "Supply this exact phrase through the authenticated client's destructive-call confirmation flow after preview; it is never a provider credential or native argument.",
    }


def build_tool_manifest(
    registered_tools: Any,
    *,
    build_sha: str | None = None,
) -> dict[str, Any]:
    """Build the active catalog and fail closed on unknown or missing navigation tools."""
    registered = _registered_tool_mapping(registered_tools)
    actual = set(registered)
    known = set(_CATEGORY_BY_TOOL)
    if actual - known:
        raise RuntimeError(
            f"Qdrant ToolManifest drift; unexpected registered tools={sorted(actual - known)}."
        )
    missing_navigation = set(NAVIGATION_ROLES) - actual
    if missing_navigation:
        raise RuntimeError(
            f"Qdrant ToolManifest drift; missing navigation tools={sorted(missing_navigation)}."
        )

    descriptors: list[dict[str, Any]] = []
    for tool_name in sorted(actual):
        tier = "legacy" if tool_name in LEGACY_TOOLS else "agent_ready"
        if tier not in _CONTRACT_TIERS:  # pragma: no cover - module invariant
            raise RuntimeError(f"Invalid Qdrant contract tier: {tier}")
        descriptor: dict[str, Any] = {
            "serviceId": SERVICE_ID,
            "nativeToolName": tool_name,
            "canonicalName": f"{SERVICE_ID}.{tool_name}",
            "aliases": _aliases(tool_name),
            "title": _title(tool_name),
            "description": complete_description(tool_name),
            "category": _CATEGORY_BY_TOOL[tool_name],
            "deprecation": _deprecation_for(tool_name),
            "inputSchema": enrich_input_schema(tool_name, _input_schema_for(registered[tool_name])),
            "outputSchema": output_schema_for(tool_name),
            "annotations": annotations_for(tool_name),
            "confirmation": _confirmation_for(tool_name),
            "documentationUrl": (
                ENDPOINT_COVERAGE_URL
                if tool_name in {"get_endpoint_coverage", "qdrant-get-endpoint-coverage"}
                else DOCUMENTATION_URL
            ),
            "navigationRole": NAVIGATION_ROLES.get(tool_name),
            "catalogVersion": CATALOG_VERSION,
            "tier": tier,
        }
        descriptor["descriptorHash"] = descriptor_hash(descriptor)
        descriptors.append(descriptor)

    identities: dict[str, str] = {}
    for descriptor in descriptors:
        identities_to_check = [
            descriptor["nativeToolName"],
            descriptor["canonicalName"],
            *descriptor["aliases"],
        ]
        for identity in identities_to_check:
            normalized = str(identity).strip().lower()
            owner = identities.get(normalized)
            if owner and owner != descriptor["nativeToolName"]:
                raise RuntimeError(
                    f"Qdrant ToolManifest identity collision: {identity!r} belongs to {owner!r} and {descriptor['nativeToolName']!r}."
                )
            identities[normalized] = descriptor["nativeToolName"]

    counts = {
        "raw": len(descriptors),
        "agentReady": sum(tool["tier"] == "agent_ready" for tool in descriptors),
        "legacy": sum(tool["tier"] == "legacy" for tool in descriptors),
        "hidden": sum(tool["tier"] == "hidden" for tool in descriptors),
        "documented": len(descriptors),
    }
    return {
        "schemaVersion": SCHEMA_VERSION,
        "serviceId": SERVICE_ID,
        "serviceAliases": list(SERVICE_ALIASES),
        "catalogVersion": CATALOG_VERSION,
        "buildSha": get_build_sha(build_sha),
        "descriptorHash": descriptor_hash(descriptors),
        "counts": counts,
        "tools": descriptors,
    }


def manifest_categories(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    categories: dict[str, dict[str, int]] = {}
    for tool in manifest.get("tools", []):
        category = str(tool["category"])
        tier = str(tool["tier"])
        bucket = categories.setdefault(
            category, {"raw": 0, "agentReady": 0, "legacy": 0, "hidden": 0}
        )
        bucket["raw"] += 1
        bucket["agentReady" if tier == "agent_ready" else tier] += 1
    return [{"category": name, **counts} for name, counts in sorted(categories.items())]


def find_tool_descriptor(manifest: Mapping[str, Any], requested_name: str) -> dict[str, Any] | None:
    target = str(requested_name or "").strip().lower()
    if target.startswith(f"{SERVICE_ID}."):
        target = target.split(".", maxsplit=1)[1]
    if not target:
        return None
    for descriptor in manifest.get("tools", []):
        identities = [
            descriptor["nativeToolName"],
            descriptor["canonicalName"],
            *descriptor.get("aliases", []),
        ]
        if target in {str(identity).lower() for identity in identities}:
            return copy.deepcopy(descriptor)
    return None


def find_manifest_tools(
    manifest: Mapping[str, Any],
    query: str,
    *,
    category: str = "",
    risk: str = "",
    limit: int = 8,
    include_legacy: bool = False,
) -> list[dict[str, Any]]:
    query_terms = normalize_terms(query)
    if not query_terms:
        raise ValueError("query must contain at least one letter or number.")
    category_filter = " ".join(normalize_terms(category))
    risk_filter = " ".join(normalize_terms(risk))
    if risk_filter and risk_filter not in _RISK_LEVELS:
        raise ValueError("risk must be read, write, or destructive.")
    bounded_limit = max(1, min(int(limit), 25))
    matches: list[tuple[int, int, dict[str, Any]]] = []
    for index, descriptor in enumerate(manifest.get("tools", [])):
        tier = descriptor["tier"]
        if tier == "hidden" or (tier == "legacy" and not include_legacy):
            continue
        if category_filter and " ".join(normalize_terms(descriptor["category"])) != category_filter:
            continue
        tool_risk = _risk_for(str(descriptor["nativeToolName"]))
        if risk_filter and tool_risk != risk_filter:
            continue
        name_terms = normalize_terms(descriptor["nativeToolName"])
        alias_terms = tuple(
            term for alias in descriptor.get("aliases", []) for term in normalize_terms(alias)
        )
        title_terms = normalize_terms(descriptor["title"])
        category_terms = normalize_terms(descriptor["category"])
        description_terms = normalize_terms(descriptor["description"])
        searchable = set(
            name_terms + alias_terms + title_terms + category_terms + description_terms
        )
        if not all(term in searchable for term in query_terms):
            continue
        exact_name = query_terms == name_terms or query_terms == name_terms[1:]
        exact_alias = any(
            query_terms == normalize_terms(alias) for alias in descriptor.get("aliases", [])
        )
        score = (
            (1000 if exact_name else 0)
            + (900 if exact_alias else 0)
            + sum(80 for term in query_terms if term in name_terms)
            + sum(60 for term in query_terms if term in alias_terms)
            + sum(30 for term in query_terms if term in title_terms)
            + sum(15 for term in query_terms if term in category_terms)
            + sum(5 for term in query_terms if term in description_terms)
        )
        matches.append(
            (
                -score,
                index,
                {
                    "serviceId": SERVICE_ID,
                    "toolName": descriptor["nativeToolName"],
                    "title": descriptor["title"],
                    "category": descriptor["category"],
                    "risk": tool_risk,
                    "tier": tier,
                    "summary": descriptor["description"],
                    "score": score,
                    "nextAction": {
                        "toolName": "get_tool_usage",
                        "arguments": {"tool_name": descriptor["nativeToolName"]},
                    },
                },
            )
        )
    matches.sort(key=lambda item: (item[0], item[1], item[2]["toolName"]))
    return [item[2] for item in matches[:bounded_limit]]


def filter_endpoint_coverage(feature: str = "") -> list[dict[str, Any]]:
    terms = normalize_terms(feature)
    if not terms:
        return copy.deepcopy(list(ENDPOINT_COVERAGE))
    matches = []
    for item in ENDPOINT_COVERAGE:
        searchable = set(
            normalize_terms(item["feature"])
            + normalize_terms(item["status"])
            + normalize_terms(item["notes"])
            + tuple(term for tool in item["tools"] for term in normalize_terms(tool))
        )
        if all(term in searchable for term in terms):
            matches.append(copy.deepcopy(item))
    return matches
