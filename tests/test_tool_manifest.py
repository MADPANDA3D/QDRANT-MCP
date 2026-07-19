import json
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.testclient import TestClient

from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.settings import (
    MemorySettings,
    QdrantSettings,
    RequestOverrideSettings,
    ToolSettings,
)
from mcp_server_qdrant.tool_manifest import (
    CATALOG_VERSION,
    SPECIAL_ALIASES,
    build_tool_manifest,
    descriptor_hash,
    find_manifest_tools,
    find_tool_descriptor,
)
from tests.test_collection_management import make_hosted_server
from tests.test_mcp_standards import CountingEmbeddingProvider, make_server

REQUIRED_DESCRIPTOR_FIELDS = {
    "serviceId",
    "nativeToolName",
    "canonicalName",
    "aliases",
    "title",
    "description",
    "category",
    "deprecation",
    "inputSchema",
    "outputSchema",
    "annotations",
    "confirmation",
    "documentationUrl",
    "navigationRole",
    "catalogVersion",
    "tier",
    "descriptorHash",
}


def _assert_described_properties(schema: Any, path: str = "inputSchema") -> None:
    if isinstance(schema, list):
        for index, item in enumerate(schema):
            _assert_described_properties(item, f"{path}[{index}]")
        return
    if not isinstance(schema, dict):
        return
    properties = schema.get("properties")
    if isinstance(properties, dict):
        for name, property_schema in properties.items():
            assert isinstance(property_schema, dict), f"{path}.properties.{name}"
            assert property_schema.get("description"), f"{path}.properties.{name}"
            _assert_described_properties(property_schema, f"{path}.properties.{name}")
    for keyword in ("allOf", "anyOf", "oneOf", "prefixItems", "items"):
        _assert_described_properties(schema.get(keyword), f"{path}.{keyword}")
    for definitions_key in ("$defs", "definitions"):
        definitions = schema.get(definitions_key)
        if isinstance(definitions, dict):
            for name, definition in definitions.items():
                _assert_described_properties(definition, f"{path}.{definitions_key}.{name}")


def _make_admin_server() -> QdrantMCPServer:
    return QdrantMCPServer(
        tool_settings=ToolSettings.model_validate({"MCP_ADMIN_TOOLS_ENABLED": True}),
        qdrant_settings=QdrantSettings.model_validate(
            {"QDRANT_LOCAL_PATH": ":memory:", "COLLECTION_NAME": "memories"}
        ),
        request_override_settings=RequestOverrideSettings(),
        memory_settings=MemorySettings(),
        embedding_provider=CountingEmbeddingProvider(),
    )


def test_manifest_is_complete_deterministic_and_secret_safe() -> None:
    server, _ = make_server()
    manifest = build_tool_manifest(
        server._registered_tools,  # pylint: disable=protected-access
        build_sha="abcdef1234567",
    )

    assert manifest["schemaVersion"] == "1.0.0"
    assert manifest["serviceId"] == "qdrant"
    assert manifest["catalogVersion"] == CATALOG_VERSION
    assert manifest["buildSha"] == "abcdef1234567"
    assert manifest["counts"] == {
        "raw": 77,
        "agentReady": 69,
        "legacy": 8,
        "hidden": 0,
        "documented": 77,
    }
    assert manifest["descriptorHash"] == descriptor_hash(manifest["tools"])
    serialized_manifest = json.dumps(manifest)
    assert "portal-secret" not in serialized_manifest

    identities: dict[str, str] = {}
    for descriptor in manifest["tools"]:
        assert REQUIRED_DESCRIPTOR_FIELDS == set(descriptor)
        assert descriptor["canonicalName"] == (f"qdrant.{descriptor['nativeToolName']}")
        assert descriptor["catalogVersion"] == CATALOG_VERSION
        hash_input = dict(descriptor)
        supplied_hash = hash_input.pop("descriptorHash")
        assert supplied_hash == descriptor_hash(hash_input)
        assert len(descriptor["description"]) > 250
        _assert_described_properties(descriptor["inputSchema"])
        output = descriptor["outputSchema"]
        assert output["required"] == ["data", "meta"]
        assert output["properties"]["data"]["description"]
        assert output["properties"]["meta"]["properties"]
        annotations = descriptor["annotations"]
        assert all(
            isinstance(annotations[field], bool)
            for field in (
                "readOnlyHint",
                "destructiveHint",
                "openWorldHint",
                "idempotentHint",
            )
        )
        if annotations["destructiveHint"]:
            assert descriptor["confirmation"]["required"] is True
            assert descriptor["confirmation"]["exactPhrase"].startswith("CONFIRM QDRANT ")
        for identity in (
            descriptor["nativeToolName"],
            descriptor["canonicalName"],
            *descriptor["aliases"],
        ):
            normalized = identity.lower()
            assert normalized not in identities, (
                identity,
                identities.get(normalized),
                descriptor["nativeToolName"],
            )
            identities[normalized] = descriptor["nativeToolName"]


def test_admin_registry_count_is_truthful() -> None:
    server = _make_admin_server()
    manifest = server.get_tool_manifest()

    assert len(server._registered_tools) == 79  # pylint: disable=protected-access
    assert "qdrant-create-snapshot" in server._registered_tools
    assert "qdrant-update-optimizer-config" in server._registered_tools
    assert "qdrant-restore-snapshot" not in server._registered_tools
    assert find_tool_descriptor(manifest, "qdrant-restore-snapshot") is None
    assert manifest["counts"] == {
        "raw": 79,
        "agentReady": 71,
        "legacy": 8,
        "hidden": 0,
        "documented": 79,
    }
    descriptor = find_tool_descriptor(manifest, "update_optimizer_config")
    assert descriptor is not None
    assert descriptor["nativeToolName"] == "qdrant-update-optimizer-config"


def test_navigation_roles_and_standard_aliases_are_lossless() -> None:
    server, _ = make_server()
    manifest = server.get_tool_manifest()
    expected = {
        "check_configuration": ("check_configuration", "configuration"),
        "list_capabilities": ("list_capabilities", "catalog"),
        "get_endpoint_coverage": ("get_endpoint_coverage", "coverage"),
        "get_tool_usage": ("get_tool_usage", "reference"),
        "find_tools": ("find_tools", "discovery"),
    }

    for alias, (native_name, navigation_role) in expected.items():
        descriptor = find_tool_descriptor(manifest, alias)
        assert descriptor is not None
        assert descriptor["nativeToolName"] == native_name
        assert descriptor["navigationRole"] == navigation_role
        assert descriptor["annotations"] == {
            "readOnlyHint": True,
            "destructiveHint": False,
            "openWorldHint": False,
            "idempotentHint": True,
        }

    legacy_names = {
        descriptor["nativeToolName"]
        for descriptor in manifest["tools"]
        if descriptor["tier"] == "legacy"
    }
    assert legacy_names == {
        "qdrant-submit-job",
        "qdrant-ingest-manifest",
        "qdrant-ingest-class-manifest",
        "qdrant-check-configuration",
        "qdrant-list-capabilities",
        "qdrant-get-endpoint-coverage",
        "qdrant-get-tool-usage",
        "qdrant-find-tools",
    }

    for prefixed_name in {
        "qdrant-check-configuration",
        "qdrant-list-capabilities",
        "qdrant-get-endpoint-coverage",
        "qdrant-get-tool-usage",
        "qdrant-find-tools",
    }:
        descriptor = find_tool_descriptor(manifest, prefixed_name)
        assert descriptor is not None
        assert descriptor["tier"] == "legacy"
        assert descriptor["navigationRole"] is None
        assert descriptor["annotations"]["openWorldHint"] is False


def test_manifest_search_handles_multi_token_alias_and_negative_queries() -> None:
    server, _ = make_server()
    manifest = server.get_tool_manifest()

    semantic_matches = find_manifest_tools(manifest, "semantic vector search")
    assert semantic_matches[0]["toolName"] == "qdrant-find"

    alias_matches = find_manifest_tools(manifest, "find_tools")
    assert alias_matches[0]["toolName"] == "find_tools"

    destructive_matches = find_manifest_tools(
        manifest,
        "delete document",
        risk="destructive",
    )
    assert destructive_matches[0]["toolName"] == "qdrant-delete-document"
    assert find_manifest_tools(manifest, "discord channel moderation") == []


def test_manifest_builder_rejects_case_insensitive_alias_collisions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server, _ = make_server()
    monkeypatch.setitem(
        SPECIAL_ALIASES,
        "qdrant-find",
        ("store",),
    )

    with pytest.raises(RuntimeError, match="identity collision"):
        server.get_tool_manifest()


@pytest.mark.asyncio
async def test_navigation_tools_publish_and_search_the_manifest() -> None:
    server, _ = make_server()
    ctx = SimpleNamespace(request_id="tool-manifest-navigation")

    capabilities = await server._registered_tools.get(  # pylint: disable=protected-access
        "list_capabilities"
    ).fn(ctx, include_descriptors=True)
    manifest = capabilities["data"]["manifest"]
    assert capabilities["data"]["counts"]["raw"] == 77
    assert len(manifest["tools"]) == 77

    prefixed_capabilities = await server._registered_tools.get(  # pylint: disable=protected-access
        "qdrant-list-capabilities"
    ).fn(ctx, include_descriptors=False)
    assert prefixed_capabilities["data"]["counts"] == capabilities["data"]["counts"]

    usage = await server._registered_tools.get(  # pylint: disable=protected-access
        "get_tool_usage"
    ).fn(ctx, tool_name="check_configuration")
    assert usage["data"]["descriptor"]["nativeToolName"] == "check_configuration"

    found = await server._registered_tools.get(  # pylint: disable=protected-access
        "find_tools"
    ).fn(ctx, query="local file upload", limit=3)
    assert found["data"]["count"] > 0
    assert found["data"]["matches"][0]["toolName"] in {
        "qdrant-start-file-upload",
        "qdrant-append-file-upload",
        "qdrant-finish-file-upload",
    }


def test_health_preserves_legacy_fields_and_reports_catalog_truth(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MCP_MODE", "standalone")
    monkeypatch.setenv(
        "MCP_ACCESS_TOKEN",
        "standalone-access-token-0000000000000001",
    )
    monkeypatch.setenv("MCP_ALLOWED_HOSTS", "testserver")
    server = make_hosted_server()
    server.request_override_settings.credential_mode = "request"
    server.request_override_settings.qdrant_host_allowlist = ["qdrant.example"]
    app = server.http_app(path="/mcp")

    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert payload["service"] == "qdrant-mcp"
    assert payload["tool_count"] == 77
    assert payload["tools"]["total"] == 77
    assert payload["raw_tool_count"] == 77
    assert payload["exposed_tool_count"] == 77
    assert payload["agent_ready_tool_count"] == 69
    assert payload["catalog_version"] == CATALOG_VERSION
    assert payload["version"] == "2.0.0"
    assert payload["status"] == "ok"
    assert payload["configuration_ready"] is True
    assert set(payload["configuration_readiness"]) == {
        "access_boundary_configured",
        "mode",
        "credential_mode",
        "default_qdrant_configured",
        "request_overrides_enabled",
        "request_qdrant_allowlist_configured",
    }
