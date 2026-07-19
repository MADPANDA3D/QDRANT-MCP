#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 IMAGE BUILD_SHA SOURCE_FINGERPRINT" >&2
  exit 2
fi

image=$1
build_sha=$2
source_fingerprint=$3
if [[ ! "$build_sha" =~ ^[0-9a-f]{40}$ ]]; then
  echo "BUILD_SHA must be a lowercase 40-character Git commit SHA" >&2
  exit 2
fi
if [[ ! "$source_fingerprint" =~ ^[0-9a-f]{64}$ ]]; then
  echo "SOURCE_FINGERPRINT must be a lowercase 64-character SHA-256 value" >&2
  exit 2
fi
portal_grant=ci-portal-grant-000000000000000000000000000000000000000000
access_token=ci-standalone-token-000000000000000000000000000000000000000000
qdrant_key=synthetic-qdrant-key-000000000000000000000000000000
openai_key=synthetic-openai-key-000000000000000000000000000000
active_container=

cleanup() {
  if [[ -n "$active_container" ]]; then
    docker rm -f "$active_container" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

configured_env=$(docker image inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$image")
if grep -Eq '^HOME=' <<<"$configured_env"; then
  echo "image must not override the runtime user's passwd HOME" >&2
  exit 1
fi

if ! offline_smoke_output=$(
  docker run --rm \
    --init --network none --read-only --user 10001:10001 \
    --cap-drop ALL --security-opt no-new-privileges --pids-limit 256 \
    --memory 2g --cpus 2 \
    --tmpfs /tmp:rw,noexec,nosuid,nodev,size=1g,mode=1777 \
    "$image" python -c \
    "import asyncio,os,pwd; from mcp_server_qdrant.embeddings.factory import create_embedding_provider; from mcp_server_qdrant.settings import EmbeddingProviderSettings; s=EmbeddingProviderSettings(); expected='qdrant/all-MiniLM-L6-v2-onnx@5f1b8cd78bc4fb444dd171e59b18f3a3af89a079'; runtime_home=os.environ.get('HOME'); assert runtime_home == pwd.getpwuid(os.getuid()).pw_dir; assert runtime_home != '/tmp'; assert os.environ.get('HF_HUB_OFFLINE') == '1'; assert s.fastembed_model_path == '/opt/qdrant-mcp/models/all-MiniLM-L6-v2'; assert s.fastembed_model_revision == expected; p=create_embedding_provider(s); v=asyncio.run(p.embed_query('offline immutable image smoke')); assert p.version == expected; assert len(v) == 384, len(v); print('offline-fastembed-smoke: 384 dimensions; passwd HOME preserved')" \
    2>&1
); then
  printf '%s\n' "$offline_smoke_output" >&2
  exit 1
fi
printf '%s\n' "$offline_smoke_output"

for profile in standalone-server standalone-request portal; do
  active_container="qdrant-mcp-smoke-$profile"
  cleanup
  active_container="qdrant-mcp-smoke-$profile"

  case "$profile" in
    standalone-server)
      mode_env=(
        -e MCP_MODE=standalone
        -e QDRANT_CREDENTIAL_MODE=server
        -e "MCP_ACCESS_TOKEN=$access_token"
        -e QDRANT_URL=https://qdrant.example.com
        -e "QDRANT_API_KEY=$qdrant_key"
        -e MCP_ALLOW_REQUEST_OVERRIDES=false
        -e MCP_REQUIRE_REQUEST_QDRANT_URL=false
        -e MCP_DISABLE_DEFAULT_QDRANT_FALLBACK=false
        -e MCP_DISABLE_DEFAULT_EMBEDDING_FALLBACK=false
      )
      ;;
    standalone-request)
      mode_env=(
        -e MCP_MODE=standalone
        -e QDRANT_CREDENTIAL_MODE=request
        -e "MCP_ACCESS_TOKEN=$access_token"
        -e MCP_ALLOW_REQUEST_OVERRIDES=true
        -e MCP_REQUIRE_REQUEST_QDRANT_URL=true
        -e MCP_REQUIRE_REQUEST_QDRANT_API_KEY=true
        -e MCP_REQUIRE_REQUEST_COLLECTION=false
        -e MCP_DISABLE_DEFAULT_QDRANT_FALLBACK=true
        -e MCP_DISABLE_DEFAULT_EMBEDDING_FALLBACK=true
        -e MCP_QDRANT_HOST_ALLOWLIST=qdrant.example.com
        -e MCP_QDRANT_ALLOWED_PORTS=443,6333
      )
      ;;
    portal)
      mode_env=(
        -e MCP_MODE=portal
        -e QDRANT_CREDENTIAL_MODE=request
        -e "MCP_PORTAL_GRANT_TOKEN=$portal_grant"
        -e MCP_PORTAL_GRANT_HEADER=x-madpanda-portal-grant
        -e MCP_TENANT_ID_HEADER=x-madpanda-user-id
        -e MCP_ALLOW_REQUEST_OVERRIDES=true
        -e MCP_REQUIRE_REQUEST_QDRANT_URL=true
        -e MCP_REQUIRE_REQUEST_QDRANT_API_KEY=true
        -e MCP_REQUIRE_REQUEST_COLLECTION=false
        -e MCP_DISABLE_DEFAULT_QDRANT_FALLBACK=true
        -e MCP_DISABLE_DEFAULT_EMBEDDING_FALLBACK=true
        -e MCP_QDRANT_HOST_ALLOWLIST=qdrant.example.com
        -e MCP_QDRANT_ALLOWED_PORTS=443,6333
      )
      ;;
  esac

  docker run -d --rm --name "$active_container" \
    --init --network none --read-only --user 10001:10001 \
    --cap-drop ALL --security-opt no-new-privileges --pids-limit 256 \
    --memory 2g --cpus 2 \
    --tmpfs /tmp:rw,noexec,nosuid,nodev,size=1g,mode=1777 \
    "${mode_env[@]}" \
    -e EMBEDDING_PROVIDER=openai \
    -e EMBEDDING_MODEL=text-embedding-3-small \
    -e "OPENAI_API_KEY=$openai_key" \
    -e MCP_SERVER_VERSION=2.0.0 \
    -e MCP_EXPECTED_TOOL_COUNT=77 \
    -e MCP_EXPECTED_AGENT_READY_COUNT=69 \
    -e MCP_EXPECTED_CATALOG_VERSION=qdrant-2026.07.19.1 \
    -e MCP_ALLOWED_HOSTS=127.0.0.1:*,localhost:* \
    -e MCP_ALLOWED_ORIGINS= \
    -e MCP_REQUEST_BODY_MAX_BYTES=1048576 \
    -e MCP_REQUEST_BODY_TIMEOUT_SECONDS=10 \
    -e MCP_OUTBOUND_HOST_ALLOWLIST= \
    -e MCP_OUTBOUND_ALLOWED_PORTS=443 \
    -e MCP_ALLOW_INSECURE_OUTBOUND_HTTP=false \
    -e MCP_OPENAI_HOST_ALLOWLIST=api.openai.com \
    -e MCP_OPENAI_ALLOWED_PORTS=443 \
    "$image" >/dev/null

  ready=false
  for _ in {1..45}; do
    if docker exec "$active_container" python -c \
      "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).read()" \
      >/dev/null 2>&1; then
      ready=true
      break
    fi
    sleep 1
  done
  if [[ "$ready" != true ]]; then
    docker logs "$active_container" >&2 || true
    exit 1
  fi
  docker exec \
    -e "EXPECTED_BUILD_SHA=$build_sha" \
    -e "EXPECTED_SOURCE_FINGERPRINT=$source_fingerprint" \
    -e EXPECTED_IMAGE_REFERENCE=development \
    -e EXPECTED_IMAGE_DIGEST=unknown \
    "$active_container" python -c \
    "import json,os,urllib.request; o=urllib.request.build_opener(urllib.request.ProxyHandler({})); p=json.loads(o.open('http://127.0.0.1:8000/health',timeout=3).read()); assert p['build_sha']==os.environ['EXPECTED_BUILD_SHA'],p; assert p['source_fingerprint']==os.environ['EXPECTED_SOURCE_FINGERPRINT'],p; assert p['image_reference']==os.environ['EXPECTED_IMAGE_REFERENCE'],p; assert p['image_digest']==os.environ['EXPECTED_IMAGE_DIGEST'],p"
  if ! smoke_output=$(
    docker exec "$active_container" python /app/scripts/runtime_smoke.py 2>&1
  ); then
    printf '%s\n' "$smoke_output" >&2
    docker logs "$active_container" >&2 || true
    exit 1
  fi
  printf '%s\n' "$smoke_output"
  cleanup
  active_container=
done
