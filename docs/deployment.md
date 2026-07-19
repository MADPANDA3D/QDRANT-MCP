# Deployment guide

Qdrant MCP supports local `stdio` and three authenticated Streamable HTTP profiles: standalone with
server credentials, standalone with request-scoped credentials, and Portal request mode. This guide
uses placeholders only. Never commit a populated `.env` file.

## Prerequisites

- Docker Engine with Compose v2, or Python 3.12/3.13 plus `uv` for source execution;
- a reachable Qdrant endpoint and least-privilege Qdrant API key when the endpoint requires one;
- a TLS reverse proxy for any HTTP access beyond the local host;
- deliberate host allowlists for request-scoped Qdrant connectors or remote document fetching.

For the complete document-ingest surface, direct Python hosts also need `antiword`, Poppler tools,
and Tesseract. Debian/Ubuntu package names are `antiword poppler-utils tesseract-ocr`; Arch package
names are `antiword poppler tesseract`. Without `antiword`, `.doc` ingest fails. Without Poppler,
PDF fallback extraction and OCR page rendering are unavailable. Without Tesseract, OCR cannot
recover image text. Text-native PDFs may still work through `pypdf`, while scanned or damaged PDFs
can fail or return bounded degradation warnings. These native helpers are already in the stock
container.

## Local stdio

```bash
uv sync --frozen --python 3.12.13
QDRANT_URL=https://qdrant.example.com \
QDRANT_API_KEY=replace-with-your-qdrant-key \
uv run mad-mcp-qdrant --transport stdio
```

`stdio` does not expose an HTTP listener. The local process and MCP client configuration form its
trust boundary.

## Prepare Compose

```bash
cp .env.example .env
```

Leave `COMPOSE_PROFILES` empty and choose exactly one profile on the command line. The profiles share
the same loopback host port, so starting more than one is intentionally unsupported.

The source build records a commit SHA, source fingerprint, and image version as build metadata:

```bash
MCP_BUILD_SHA="$(git rev-parse HEAD)"
MCP_SOURCE_FINGERPRINT="$(git archive --format=tar HEAD | sha256sum | awk '{print $1}')"
```

Copy those values into `.env` when building from a reviewed commit.

## FastEmbed acquisition modes

The stock image contains one reviewed model and performs no model acquisition at runtime:

- model: `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions);
- source: `qdrant/all-MiniLM-L6-v2-onnx` at commit
  `5f1b8cd78bc4fb444dd171e59b18f3a3af89a079`;
- runtime path: `/opt/qdrant-mcp/models/all-MiniLM-L6-v2`;
- runtime controls: `HF_HUB_OFFLINE=1`, `specific_model_path`, and `local_files_only=True`.

Compose fixes `FASTEMBED_MODEL_PATH` and `FASTEMBED_MODEL_REVISION` to that image-owned model. Do
not mount over the path or override its identity in a stock-image deployment. A different model is
a custom-image change: update the installer pins, file hashes, attribution, dimensional smoke test,
and image digest together. Startup rejects a FastEmbed model-name override while the stock MiniLM
revision is active; OpenAI mode safely ignores the dormant FastEmbed path and revision.

A direct Python/package installation deliberately behaves differently. When
`FASTEMBED_MODEL_PATH` is unset, this project invokes FastEmbed normally; FastEmbed may use its
cache or acquire the selected model on first use. Leave `HF_HUB_OFFLINE` unset for that behavior.
To operate a direct Python process from a trusted local directory instead, set both
`FASTEMBED_MODEL_PATH` and a truthful `FASTEMBED_MODEL_REVISION`; the provider then passes
`specific_model_path` and `local_files_only=True` to FastEmbed. The repository installer can prepare
the same reviewed model:

```bash
python scripts/install_fastembed_model.py --destination /opt/qdrant-fastembed/all-MiniLM-L6-v2
export FASTEMBED_MODEL_PATH=/opt/qdrant-fastembed/all-MiniLM-L6-v2
export FASTEMBED_MODEL_REVISION=qdrant/all-MiniLM-L6-v2-onnx@5f1b8cd78bc4fb444dd171e59b18f3a3af89a079
export HF_HUB_OFFLINE=1
```

## Standalone with server-owned credentials

Set:

```dotenv
MCP_ACCESS_TOKEN=replace-with-a-generated-high-entropy-token
QDRANT_CREDENTIAL_MODE=server
QDRANT_URL=https://qdrant.example.com
QDRANT_API_KEY=replace-with-your-qdrant-key
```

Then:

```bash
docker compose --profile standalone config --quiet
docker compose --profile standalone up --detach --build
docker compose --profile standalone ps
```

The MCP client authenticates with `Authorization: Bearer …`.

## Standalone with request-scoped credentials

Set:

```dotenv
MCP_ACCESS_TOKEN=replace-with-a-generated-high-entropy-token
QDRANT_CREDENTIAL_MODE=request
QDRANT_URL=
QDRANT_API_KEY=
MCP_QDRANT_HOST_ALLOWLIST=qdrant.example.com
```

Start the dedicated request-scoped profile:

```bash
docker compose --profile standalone-request config --quiet
docker compose --profile standalone-request up --detach --build
```

Each data-tool request needs the Bearer token plus the configured Qdrant URL/API key headers.
Authenticated navigation tools can report missing connector setup. The profile fixes
`QDRANT_CREDENTIAL_MODE=request` and disables server credential fallback at startup.

The profile clears `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_ORG`, and `OPENAI_PROJECT` even when
the host environment defines them. The stock local FastEmbed default does not carry a provider
credential and remains available. Request-scoped OpenAI configuration must arrive through headers.
A direct-Python request-mode deployment that deliberately supplies a server OpenAI default must set
`MCP_DISABLE_DEFAULT_EMBEDDING_FALLBACK=true`; each embedding request must then provide its own
provider, model, and key headers.

## Portal with request-scoped credentials

Set:

```dotenv
MCP_PORTAL_GRANT_TOKEN=replace-with-the-broker-grant
MCP_PORTAL_GRANT_HEADER=x-madpanda-portal-grant
MCP_TENANT_ID_HEADER=x-madpanda-user-id
MCP_QDRANT_HOST_ALLOWLIST=qdrant.example.com
```

Then:

```bash
docker compose --profile portal config --quiet
docker compose --profile portal up --detach --build
docker compose --profile portal ps
```

Portal fixes `QDRANT_CREDENTIAL_MODE=request` and clears server Qdrant credentials plus all
server-owned OpenAI credential/endpoint fields. The credential-free local FastEmbed default remains
available. See [portal-mode.md](portal-mode.md) for the request boundary.

## Reverse proxy

The stock Compose file publishes `127.0.0.1:8000` only. A reverse proxy on the same host should:

- terminate TLS;
- forward to `http://127.0.0.1:8000`;
- preserve the `Authorization` or configured Portal/Qdrant headers;
- disable caching and access-log capture of sensitive headers;
- apply request-body and connection timeouts suitable for MCP streaming;
- avoid rewriting `/mcp` unexpectedly.

Add the reverse proxy's exact public `Host` value to `MCP_ALLOWED_HOSTS`; the default includes only
loopback and the three Compose service names. Keep `MCP_ALLOWED_ORIGINS` empty for non-browser MCP
clients. If a browser client is deliberately supported, list only its exact trusted origins.

Do not bind the container port to `0.0.0.0` merely to make routing easier.

## Immutable release image

After `v2.0.0` completes every release gate, use the exact digest from its GitHub Release:

```dotenv
MCP_RUNTIME_IMAGE=ghcr.io/madpanda3d/qdrant-mcp-server@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

Run `docker compose config` and confirm the resolved service image is the expected digest before
starting it. Mutable tags are for discovery, not deployment identity. The release gate also runs an
actual 384-dimension embedding from the stock model with container networking disabled; a healthy
HTTP endpoint alone does not satisfy this gate.

Pull and start the selected digest without invoking the source-build path:

```bash
docker compose --profile portal config --quiet
docker compose --profile portal pull
docker compose --profile portal up --detach --no-build
docker compose --profile portal ps
```

Use `standalone` or `standalone-request` in all four commands when that is the selected profile.
Never add `--build` to an immutable-digest start.

## Upload and background-job quotas

All upload and job ceilings are enforced per server process/container, not fleet-wide. An owner is
the authenticated Portal tenant in Portal mode. Standalone and local modes use the shared
process principal, so their callers share one owner budget. Replicated deployments
need an external fleet-wide admission policy if these per-process bounds are not sufficient.

| Variable | Default |
|---|---:|
| `MCP_FILE_UPLOAD_MAX_ACTIVE_PER_OWNER` | `4` |
| `MCP_FILE_UPLOAD_MAX_ACTIVE_GLOBAL` | `32` |
| `MCP_FILE_UPLOAD_MAX_AGGREGATE_BYTES_PER_OWNER` | `268435456` (256 MiB) |
| `MCP_FILE_UPLOAD_MAX_AGGREGATE_BYTES_GLOBAL` | `536870912` (512 MiB) |
| `MCP_BACKGROUND_JOB_MAX_ACTIVE_PER_OWNER` | `2` |
| `MCP_BACKGROUND_JOB_MAX_ACTIVE_GLOBAL` | `16` |
| `MCP_BACKGROUND_JOB_MAX_RETAINED_PER_OWNER` | `20` |
| `MCP_BACKGROUND_JOB_MAX_RETAINED_GLOBAL` | `100` |
| `MCP_BACKGROUND_JOB_LOG_MESSAGE_MAX_CHARS` | `512` |
| `MCP_BACKGROUND_JOB_MAX_LOGS_PER_JOB` | `100` |
| `MCP_BACKGROUND_JOB_MAX_LOG_TAIL` | `100` |
| `MCP_BACKGROUND_JOB_MAX_RESULT_BYTES` | `262144` |
| `MCP_BACKGROUND_JOB_MAX_RECORD_BYTES` | `524288` |

The stock Compose profile mounts a 1 GiB ephemeral `/tmp` tmpfs shared by upload bytes, job records,
OCR intermediates, and other temporary files. The aggregate upload defaults deliberately leave
headroom, but operators must tune all bounds together if they change `/tmp`, concurrency, file-size,
or OCR limits.

## Operations

- `/health` is the container health endpoint; do not infer provider reachability from HTTP 200 alone.
- The container root filesystem is read-only and the stock `/tmp` is an ephemeral 1 GiB tmpfs.
- Upload sessions and persisted textbook job status do not survive container replacement in the
  stock profile.
- Back up Qdrant using Qdrant-native controls before destructive maintenance.
- Pin, scan, and review any custom reverse proxy, volume, network, or resource-limit changes.
- Rotate service and provider credentials in their owning systems; never bake them into an image.
