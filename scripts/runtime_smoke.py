#!/usr/bin/env python3
"""Provider-free wire smoke for all authenticated Qdrant MCP HTTP profiles."""

from __future__ import annotations

import http.client
import json
import os
from importlib.metadata import version
from typing import Any

HOST = "127.0.0.1"
PORT = int(os.getenv("FASTMCP_SERVER_PORT", "8000"))
MODE = os.environ["MCP_MODE"]
CREDENTIAL_MODE = os.environ["QDRANT_CREDENTIAL_MODE"]
EXPECTED_TOOL_COUNT = int(os.getenv("MCP_EXPECTED_TOOL_COUNT", "77"))
EXPECTED_AGENT_READY_COUNT = int(os.getenv("MCP_EXPECTED_AGENT_READY_COUNT", "69"))
EXPECTED_BUILD_SHA = os.environ["MCP_BUILD_SHA"]
EXPECTED_SOURCE_FINGERPRINT = os.environ["MCP_SOURCE_FINGERPRINT"]
EXPECTED_IMAGE_REFERENCE = os.environ["MCP_IMAGE_REFERENCE"]
EXPECTED_IMAGE_DIGEST = os.getenv("MCP_IMAGE_DIGEST", "") or "unknown"
EXPECTED_CATALOG_VERSION = os.getenv("MCP_EXPECTED_CATALOG_VERSION", "qdrant-2026.07.19.1")
ACCESS_TOKEN = os.getenv("MCP_ACCESS_TOKEN", "")
PORTAL_GRANT = os.getenv("MCP_PORTAL_GRANT_TOKEN", "")
PORTAL_HEADER = os.getenv("MCP_PORTAL_GRANT_HEADER", "X-MADPANDA-PORTAL-GRANT")
TENANT_HEADER = os.getenv("MCP_TENANT_ID_HEADER", "X-MADPANDA-USER-ID")
PACKAGE_VERSION = version("mad-mcp-qdrant")
SYNTHETIC_TENANT = "tenant-smoke-0001"
SYNTHETIC_QDRANT_URL = "https://qdrant.example.com"
SYNTHETIC_QDRANT_KEY = "synthetic-qdrant-key-000000000000000000000000000000"
SYNTHETIC_OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")
SYNTHETIC_REQUEST_OPENAI_KEY = "synthetic-request-openai-key-000000000000000000000000"


def service_headers(*, valid: bool = True, tenant: bool = True) -> dict[str, str]:
    if MODE == "standalone":
        token = ACCESS_TOKEN if valid else "wrong-standalone-token-000000000000"
        return {"Authorization": f"Bearer {token}"}
    if MODE == "portal":
        token = PORTAL_GRANT if valid else "wrong-portal-grant-0000000000000000"
        headers = {PORTAL_HEADER: token}
        if tenant:
            headers[TENANT_HEADER] = SYNTHETIC_TENANT
        return headers
    raise AssertionError(f"unexpected MCP_MODE={MODE!r}")


def provider_headers() -> dict[str, str]:
    if CREDENTIAL_MODE != "request":
        return {}
    return {
        os.getenv("MCP_QDRANT_URL_HEADER", "X-Qdrant-Url"): SYNTHETIC_QDRANT_URL,
        os.getenv("MCP_QDRANT_API_KEY_HEADER", "X-Qdrant-Api-Key"): (SYNTHETIC_QDRANT_KEY),
        os.getenv("MCP_COLLECTION_NAME_HEADER", "X-Collection-Name"): "smoke-memory",
        os.getenv("MCP_EMBEDDING_PROVIDER_HEADER", "X-Embedding-Provider"): "openai",
        os.getenv("MCP_EMBEDDING_MODEL_HEADER", "X-Embedding-Model"): ("text-embedding-3-small"),
        os.getenv("MCP_OPENAI_API_KEY_HEADER", "X-OpenAI-Api-Key"): (SYNTHETIC_REQUEST_OPENAI_KEY),
    }


def decode_response(raw: bytes, content_type: str) -> Any:
    if not raw:
        return None
    if "text/event-stream" in content_type:
        events: list[Any] = []
        for line in raw.decode("utf-8", errors="replace").splitlines():
            if not line.startswith("data:"):
                continue
            payload = line.removeprefix("data:").strip()
            if not payload:
                continue
            events.append(json.loads(payload))
        if not events:
            return raw.decode("utf-8", errors="replace")
        return events[-1]
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return raw.decode("utf-8", errors="replace")


def request(
    method: str,
    path: str,
    *,
    payload: dict[str, Any] | bytes | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], Any]:
    body = (
        json.dumps(payload, separators=(",", ":")).encode()
        if isinstance(payload, dict)
        else payload
    )
    merged = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    if headers:
        merged.update(headers)
    connection = http.client.HTTPConnection(HOST, PORT, timeout=8)
    try:
        connection.request(method, path, body=body, headers=merged)
        response = connection.getresponse()
        raw = response.read()
        response_headers = {key.lower(): value for key, value in response.getheaders()}
    finally:
        connection.close()
    decoded = decode_response(raw, response_headers.get("content-type", ""))
    return response.status, response_headers, decoded


def rpc(method: str, request_id: int, params: dict[str, Any]) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def tool_payload(response: Any, tool_name: str) -> dict[str, Any]:
    require(isinstance(response, dict), f"{tool_name} response is not JSON")
    result = response.get("result")
    require(isinstance(result, dict), f"{tool_name} response has no result")
    require(not result.get("isError", False), f"{tool_name} returned an error: {result}")
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        return structured
    for item in result.get("content", []):
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        try:
            decoded = json.loads(item.get("text", ""))
        except (TypeError, json.JSONDecodeError):
            continue
        if isinstance(decoded, dict):
            return decoded
    raise AssertionError(f"{tool_name} returned no structured object")


def call_tool(
    headers: dict[str, str],
    request_id: int,
    name: str,
    arguments: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    status, _, response = request(
        "POST",
        "/mcp",
        payload=rpc("tools/call", request_id, {"name": name, "arguments": arguments}),
        headers=headers,
    )
    require(status == 200, f"{name} failed with HTTP {status}: {response}")
    return response, tool_payload(response, name)


def main() -> None:
    observed: list[Any] = []

    status, _, health = request("GET", "/health", headers={"Accept": "application/json"})
    observed.append(health)
    require(status == 200 and isinstance(health, dict), f"health={status} {health}")
    require(health.get("ok") is True and health.get("status") == "ok", f"health={health}")
    require(health.get("version") == PACKAGE_VERSION == "2.0.0", f"version={health}")
    require(health.get("build_sha") == EXPECTED_BUILD_SHA, f"build_sha={health}")
    require(
        health.get("source_fingerprint") == EXPECTED_SOURCE_FINGERPRINT,
        f"source_fingerprint={health}",
    )
    require(
        health.get("image_reference") == EXPECTED_IMAGE_REFERENCE,
        f"image_reference={health}",
    )
    require(health.get("image_digest") == EXPECTED_IMAGE_DIGEST, f"digest={health}")
    require(
        health.get("catalog_version") == EXPECTED_CATALOG_VERSION,
        f"catalog={health}",
    )
    require(health.get("tool_count") == EXPECTED_TOOL_COUNT, f"tool_count={health}")
    require(
        health.get("agent_ready_tool_count") == EXPECTED_AGENT_READY_COUNT,
        f"agent_ready={health}",
    )
    readiness = health.get("configuration_readiness", {})
    require(readiness.get("mode") == MODE, f"mode={health}")
    require(readiness.get("credential_mode") == CREDENTIAL_MODE, f"credential={health}")
    require(health.get("configuration_ready") is True, f"readiness={health}")

    status, _, denied = request("POST", "/mcp", payload=b"malformed-before-auth")
    observed.append(denied)
    require(status == 401, f"missing auth was not rejected first: {status} {denied}")

    status, _, denied = request(
        "POST",
        "/mcp",
        payload=b"malformed-before-auth",
        headers=service_headers(valid=False),
    )
    observed.append(denied)
    require(status == 401, f"invalid auth was not rejected first: {status} {denied}")

    if MODE == "portal":
        status, _, denied = request(
            "POST",
            "/mcp",
            payload=b"malformed-before-auth",
            headers=service_headers(tenant=False),
        )
        observed.append(denied)
        require(status == 401, f"missing tenant was accepted: {status} {denied}")

    origin_headers = service_headers()
    origin_headers["Origin"] = "https://untrusted.invalid"
    status, _, denied = request(
        "POST", "/mcp", payload=rpc("tools/list", 2, {}), headers=origin_headers
    )
    observed.append(denied)
    require(status == 403, f"browser Origin was not rejected: {status} {denied}")

    host_headers = service_headers()
    host_headers["Host"] = "untrusted.invalid"
    status, _, denied = request(
        "POST", "/mcp", payload=rpc("tools/list", 3, {}), headers=host_headers
    )
    observed.append(denied)
    require(status == 421, f"untrusted Host was not rejected: {status} {denied}")

    authenticated = service_headers()
    authenticated.update(provider_headers())
    status, response_headers, initialized = request(
        "POST",
        "/mcp",
        payload=rpc(
            "initialize",
            4,
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "qdrant-mcp-image-smoke", "version": "1"},
            },
        ),
        headers=authenticated,
    )
    observed.append(initialized)
    require(status == 200 and isinstance(initialized, dict), f"initialize={status} {initialized}")
    server_info = initialized.get("result", {}).get("serverInfo", {})
    require(server_info.get("version") == PACKAGE_VERSION, f"serverInfo={server_info}")

    session_id = response_headers.get("mcp-session-id")
    navigation_headers = service_headers()
    if session_id:
        navigation_headers["Mcp-Session-Id"] = session_id
    status, _, tools = request(
        "POST", "/mcp", payload=rpc("tools/list", 5, {}), headers=navigation_headers
    )
    observed.append(tools)
    require(status == 200 and isinstance(tools, dict), f"tools/list={status} {tools}")
    listed = tools.get("result", {}).get("tools", [])
    require(len(listed) == EXPECTED_TOOL_COUNT, f"tools/list count={len(listed)}")
    names = {tool.get("name") for tool in listed if isinstance(tool, dict)}
    for required_name in (
        "check_configuration",
        "list_capabilities",
        "qdrant-build-context",
        "qdrant-list-collections",
    ):
        require(required_name in names, f"missing tool: {required_name}")

    raw_response, capabilities = call_tool(
        navigation_headers,
        6,
        "list_capabilities",
        {"include_descriptors": False},
    )
    observed.append(raw_response)
    capability_data = capabilities.get("data", capabilities)
    require(
        capability_data.get("catalog_version") == EXPECTED_CATALOG_VERSION,
        f"capabilities={capabilities}",
    )
    require(
        capability_data.get("counts", {}).get("raw") == EXPECTED_TOOL_COUNT,
        f"capabilities={capabilities}",
    )

    raw_response, configuration = call_tool(
        navigation_headers,
        7,
        "check_configuration",
        {},
    )
    observed.append(raw_response)
    configuration_data = configuration.get("data", configuration)
    require(
        configuration_data.get("catalog_version") == EXPECTED_CATALOG_VERSION,
        f"configuration={configuration}",
    )
    require(
        configuration_data.get("catalog_counts", {}).get("raw") == EXPECTED_TOOL_COUNT,
        f"configuration={configuration}",
    )

    if CREDENTIAL_MODE == "request":
        status, _, missing_connector = request(
            "POST",
            "/mcp",
            payload=rpc(
                "tools/call",
                8,
                {"name": "qdrant-list-collections", "arguments": {}},
            ),
            headers=navigation_headers,
        )
        observed.append(missing_connector)
        require(
            status == 200 and isinstance(missing_connector, dict),
            f"missing connector response={status} {missing_connector}",
        )
        missing_result = missing_connector.get("result", {})
        require(
            missing_result.get("isError") is True,
            f"data tool did not fail closed without connector headers: {missing_connector}",
        )
        missing_text = json.dumps(missing_result, ensure_ascii=True).lower()
        require(
            "missing required header" in missing_text,
            f"connector error was not actionable: {missing_connector}",
        )

    serialized = json.dumps(observed, ensure_ascii=True)
    for secret in (
        ACCESS_TOKEN,
        PORTAL_GRANT,
        SYNTHETIC_QDRANT_KEY,
        SYNTHETIC_OPENAI_KEY,
        SYNTHETIC_REQUEST_OPENAI_KEY,
    ):
        if secret:
            require(secret not in serialized, "a synthetic credential appeared in a response")

    print(
        json.dumps(
            {
                "ok": True,
                "mode": MODE,
                "credential_mode": CREDENTIAL_MODE,
                "tool_count": len(listed),
                "catalog_version": EXPECTED_CATALOG_VERSION,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
