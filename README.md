<p align="center">
  <img src="https://raw.githubusercontent.com/MADPANDA3D/QDRANT-MCP/v2.0.0/assets/brand/header.jpg" alt="MADPANDA3D Qdrant MCP" />
</p>

<p align="center">
  <a href="https://github.com/MADPANDA3D/QDRANT-MCP/actions/workflows/ci.yml"><img src="https://github.com/MADPANDA3D/QDRANT-MCP/actions/workflows/ci.yml/badge.svg" alt="Verify" /></a>
  <a href="https://github.com/MADPANDA3D/QDRANT-MCP/blob/v2.0.0/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-4c1" alt="Apache-2.0" /></a>
  <img src="https://img.shields.io/badge/Python-3.12%20%7C%203.13-3776AB?logo=python&logoColor=white" alt="Python 3.12 and 3.13" />
  <img src="https://img.shields.io/badge/MCP-Streamable%20HTTP%20%7C%20stdio-111" alt="MCP Streamable HTTP and stdio" />
</p>

```text
╭──────────────────────────────────────────────────────────────╮
│  Q D R A N T  //  M C P                                    │
│  VECTOR MEMORY · INGEST · RETRIEVE · GOVERN                 │
╰──────────────────────────────────────────────────────────────╯
```

# MADPANDA3D Qdrant MCP

A typed Model Context Protocol server for agents that need to search, store, ingest, inspect, and
maintain data in a Qdrant vector database. It combines broad Qdrant coverage with compact agent
navigation, bounded document workflows, dry-run/confirmation controls, and two authenticated HTTP
deployment modes.

> **Release integrity:** `v2.0.0` source, Python artifacts, container digest, and GitHub Release form
> one verified release only after the exact-tag workflow completes. Deploy the container by the
> digest recorded in that release; a branch, mutable tag, or HTTP 200 alone is not release proof.

## Why this server

- **Policy-derived catalog:** 77 tools by default (69 agent-ready + 8 legacy), 79 with admin tools
  enabled (71 + 8), and 45 in read-only mode (40 + 5). Health and discovery report the active set.
- **Agent navigation:** `check_configuration`, `list_capabilities`, `get_endpoint_coverage`,
  `get_tool_usage`, and `find_tools` help a client choose the right operation before spending calls.
- **Vector-memory workflows:** compact retrieval, context packs, short-term memory, promotion,
  validation, deduplication, lifecycle maintenance, and embedding migration.
- **Document ingestion:** text, base64, bounded uploads, HTTPS sources, PDF extraction, OCR, school
  manifests, and observable asynchronous textbook jobs.
- **Operational controls:** read-only mode, admin gating, dry-run previews, explicit confirmation,
  batch/output bounds, connector allowlists, and deny-by-default outbound URL fetching.
- **Portable runtime:** local `stdio`, authenticated standalone HTTP, or brokered Portal HTTP with
  request-scoped bring-your-own Qdrant credentials.

The maintained coverage matrix is in the
[endpoint coverage guide](https://github.com/MADPANDA3D/QDRANT-MCP/blob/v2.0.0/docs/endpoint-coverage.md).

## Trust model

| Runtime | Service authentication | Qdrant credentials | Intended use |
|---|---|---|---|
| Local `stdio` | Local process boundary | Environment | Desktop and local agents |
| Standalone HTTP / `server` | `Authorization: Bearer …` | Server environment | One operator-controlled connector |
| Standalone HTTP / `request` | `Authorization: Bearer …` | Per-request headers | One authenticated service serving multiple connector contexts |
| Portal HTTP | Portal grant header | Per-request headers | Brokered, tenant-scoped BYOK |

There is no unauthenticated MCP-over-HTTP mode. `/health` remains reachable for orchestration and
must return only non-secret readiness metadata. `QDRANT_CREDENTIAL_MODE` is a startup choice; it is
not remotely mutable through an MCP tool.

## Quick start: local stdio

From source:

```bash
git clone https://github.com/MADPANDA3D/QDRANT-MCP.git
cd QDRANT-MCP
uv sync --frozen --group dev --python 3.12.13

export QDRANT_URL='https://qdrant.example.com'
export QDRANT_API_KEY='replace-with-your-qdrant-key'
export COLLECTION_NAME='agent-memory'
uv run mad-mcp-qdrant --transport stdio
```

Direct Python installs need native document helpers for the complete ingest surface. On
Debian/Ubuntu install `antiword`, `poppler-utils`, and `tesseract-ocr`; on Arch install `antiword`,
`poppler`, and `tesseract`. Without `antiword`, legacy `.doc` ingest fails. Without Poppler, PDF
fallback extraction and OCR rendering are unavailable; without Tesseract, OCR produces no recovered
text. Text-native PDF extraction can still succeed, but scanned or damaged PDFs may fail or return
bounded warnings. The stock container includes all three helpers.

Example MCP client entry:

```json
{
  "mcpServers": {
    "qdrant": {
      "command": "uv",
      "args": [
        "--directory",
        "/absolute/path/to/QDRANT-MCP",
        "run",
        "mad-mcp-qdrant",
        "--transport",
        "stdio"
      ],
      "env": {
        "QDRANT_URL": "https://qdrant.example.com",
        "QDRANT_API_KEY": "replace-with-your-qdrant-key",
        "COLLECTION_NAME": "agent-memory"
      }
    }
  }
}
```

The equivalent pinned package command is:

```bash
uvx --from mad-mcp-qdrant==2.0.0 mad-mcp-qdrant --transport stdio
```

## Deploy: authenticated standalone HTTP

Copy the placeholder file and generate a high-entropy service token:

```bash
cp .env.example .env
openssl rand -hex 32
```

Set at least these values in `.env` for server-owned credentials:

```dotenv
MCP_ACCESS_TOKEN=replace-with-generated-token
QDRANT_CREDENTIAL_MODE=server
QDRANT_URL=https://qdrant.example.com
QDRANT_API_KEY=replace-with-your-qdrant-key
```

Start only the standalone profile:

```bash
docker compose --profile standalone up --detach --build
```

An HTTP client sends:

```text
Authorization: Bearer replace-with-generated-token
```

To keep connector credentials out of the service environment, stop the server-owned profile and
select the request-scoped profile with:

```dotenv
QDRANT_URL=
QDRANT_API_KEY=
MCP_QDRANT_HOST_ALLOWLIST=qdrant.example.com
MCP_QDRANT_ALLOWED_PORTS=443,6333
```

```bash
docker compose --profile standalone down
docker compose --profile standalone-request up --detach --build
```

Then send the normal Bearer token plus `X-Qdrant-Url`, `X-Qdrant-Api-Key`, and optionally
`X-Collection-Name` on each data-tool request. Authenticated navigation tools can report missing
setup without connector headers. Request mode disables server credential fallback.
The Compose profile fixes `QDRANT_CREDENTIAL_MODE=request` and the associated fail-closed override
flags at startup. It also blanks `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_ORG`, and
`OPENAI_PROJECT` so host environment values cannot become request-mode credentials. The
credential-free local FastEmbed default remains available.

## Deploy: Portal HTTP with request-scoped BYOK

Portal mode always uses `QDRANT_CREDENTIAL_MODE=request`. Set:

```dotenv
MCP_PORTAL_GRANT_TOKEN=replace-with-the-broker-grant
MCP_PORTAL_GRANT_HEADER=x-madpanda-portal-grant
MCP_TENANT_ID_HEADER=x-madpanda-user-id
MCP_QDRANT_HOST_ALLOWLIST=qdrant.example.com
MCP_QDRANT_ALLOWED_PORTS=443,6333
```

Start only the Portal profile:

```bash
docker compose --profile portal up --detach --build
```

The trusted broker forwards these headers over TLS:

```text
X-MADPANDA-PORTAL-GRANT: replace-with-the-broker-grant
X-MADPANDA-USER-ID: stable-tenant-identifier
X-QDRANT-URL: https://qdrant.example.com
X-QDRANT-API-KEY: replace-with-the-tenant-qdrant-key
X-COLLECTION-NAME: agent-memory
```

`X-MADPANDA-USER-ID` is a required tenant-routing identifier, not a credential. It is used to keep
request-scoped resources from crossing tenant boundaries. Do not treat it as authorization and do
not put real tenant identifiers in public issue reports or logs. Header names are configurable at
startup; their security meaning is not.

Portal data-tool requests without a valid grant, tenant identifier, Qdrant URL, or Qdrant API key
fail before provider work. Authenticated navigation tools remain available without connector
headers so an agent can discover the missing setup. The Portal service has no server-side Qdrant
or OpenAI credential/endpoint fallback; request-scoped OpenAI settings arrive only through the
documented headers. The credential-free local FastEmbed default remains available.

## Immutable image deployment

The release image namespace is:

```text
ghcr.io/madpanda3d/qdrant-mcp-server
```

After release, select the exact digest shown in the GitHub Release rather than a mutable tag:

```dotenv
MCP_RUNTIME_IMAGE=ghcr.io/madpanda3d/qdrant-mcp-server@sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef
```

Resolve and start that image without invoking a source build:

```bash
docker compose --profile portal config --quiet
docker compose --profile portal pull
docker compose --profile portal up --detach --no-build
```

The stock Compose profiles publish only to `127.0.0.1`, run as UID/GID `10001`, drop all Linux
capabilities, enable `no-new-privileges`, use a read-only root filesystem, bound PIDs/CPU/memory,
and mount only an ephemeral `/tmp`. Put a TLS reverse proxy in front of HTTP deployments and keep
the service port off public interfaces. Add the proxy's exact public Host to `MCP_ALLOWED_HOSTS`;
leave `MCP_ALLOWED_ORIGINS` empty unless exact browser origins are deliberately supported.

The stock image also includes
[`sentence-transformers/all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
from
[`qdrant/all-MiniLM-L6-v2-onnx@5f1b8cd78bc4fb444dd171e59b18f3a3af89a079`](https://huggingface.co/qdrant/all-MiniLM-L6-v2-onnx/tree/5f1b8cd78bc4fb444dd171e59b18f3a3af89a079).
Its archive and all
six runtime files are SHA-256 pinned during the build. Runtime uses an exact local model path with
`HF_HUB_OFFLINE=1` and `local_files_only=True`; release smoke performs a real 384-dimension embedding
under `--network none`. Startup rejects a different FastEmbed model name while this exact baked
revision is active. Direct Python installs retain normal FastEmbed cache/acquisition behavior
when `FASTEMBED_MODEL_PATH` is unset. See the
[deployment guide](https://github.com/MADPANDA3D/QDRANT-MCP/blob/v2.0.0/docs/deployment.md) and
[security model](https://github.com/MADPANDA3D/QDRANT-MCP/blob/v2.0.0/docs/security-model.md) for
the two acquisition modes and complete provenance.

## Outbound URL policy

Remote document and textbook downloads are denied by default:

```dotenv
MCP_OUTBOUND_HOST_ALLOWLIST=
MCP_OUTBOUND_ALLOWED_PORTS=443
MCP_ALLOW_INSECURE_OUTBOUND_HTTP=false
```

Allowlisting a remote host is a trust decision: the service will connect to it and process its
response. Prefer the bounded `upload://` bridge when you cannot trust a remote host. To allow an
HTTPS source, list exact hostnames, for example:

```dotenv
MCP_OUTBOUND_HOST_ALLOWLIST=docs.example.com,cdn.example.com
MCP_OUTBOUND_ALLOWED_PORTS=443
```

Plain HTTP remains blocked unless the operator explicitly sets
`MCP_ALLOW_INSECURE_OUTBOUND_HTTP=true`; that opt-in enables only standard HTTP/80. Nonstandard HTTP
ports remain forbidden. `MCP_OUTBOUND_ALLOWED_PORTS` governs trusted HTTPS ports and defaults to
443. The outbound document allowlist is separate from `MCP_QDRANT_HOST_ALLOWLIST`, which constrains
request-scoped Qdrant connector URLs.

Request-scoped custom OpenAI-compatible base URLs are also fail closed. The default policy allows
only `api.openai.com` on port 443. Add an exact trusted hostname to `MCP_OPENAI_HOST_ALLOWLIST` and
its trusted HTTPS port to `MCP_OPENAI_ALLOWED_PORTS` before accepting a custom
`X-OpenAI-Base-URL`. These allowlists never authorize private, loopback, link-local, or cloud
metadata destinations in Portal/request mode.

Request mode may reuse the server's credential-free local FastEmbed default. The stock
standalone-request and Portal Compose profiles blank all server-owned OpenAI credential and endpoint
fields. A direct-Python request-mode deployment configured with a server OpenAI default must set
`MCP_DISABLE_DEFAULT_EMBEDDING_FALLBACK=true`; each embedding request must then supply its own
provider, model, and OpenAI key headers.

## Important configuration

| Variable | Purpose | Safe default |
|---|---|---|
| `MCP_MODE` | `standalone` or `portal`; fixed by the Compose profile | `standalone` |
| `MCP_ACCESS_TOKEN` | Bearer token for standalone HTTP | Empty; HTTP startup must fail |
| `MCP_PORTAL_GRANT_TOKEN` | Shared broker-to-server Portal grant | Empty; Portal startup must fail |
| `MCP_PORTAL_GRANT_HEADER` | Portal grant header name | `x-madpanda-portal-grant` |
| `MCP_TENANT_ID_HEADER` | Required Portal tenant header name | `x-madpanda-user-id` |
| `QDRANT_CREDENTIAL_MODE` | `server` or `request`; fixed by Compose profile and restart required | `server` in standalone, `request` in standalone-request/Portal |
| `MCP_QDRANT_HOST_ALLOWLIST` | Allowed request-scoped Qdrant hostnames | Empty; request mode must fail closed |
| `MCP_QDRANT_ALLOWED_PORTS` | Allowed request-scoped Qdrant ports | `443,6333` |
| `MCP_OPENAI_HOST_ALLOWLIST` | Allowed request-scoped OpenAI-compatible hosts | `api.openai.com` |
| `MCP_OPENAI_ALLOWED_PORTS` | Allowed request-scoped OpenAI-compatible ports | `443` |
| `MCP_DISABLE_DEFAULT_EMBEDDING_FALLBACK` | Require request-scoped embedding headers; mandatory when the server default is OpenAI | `false` for local FastEmbed |
| `FASTEMBED_MODEL_PATH` | Trusted preinstalled FastEmbed directory; enables `specific_model_path` plus offline-only loading | Unset for direct Python; fixed inside the stock image |
| `FASTEMBED_MODEL_REVISION` | Source revision reported for the preinstalled model | Unset for direct Python; fixed to the reviewed commit in the stock image |
| `MCP_OUTBOUND_HOST_ALLOWLIST` | Allowed HTTPS document/textbook hosts | Empty; URL ingest denied |
| `MCP_OUTBOUND_ALLOWED_PORTS` | Allowed outbound URL ports | `443` |
| `MCP_ALLOW_INSECURE_OUTBOUND_HTTP` | Explicit plain-HTTP opt-in | `false` |
| `MCP_ALLOWED_HOSTS` | Accepted HTTP Host patterns | Loopback and all three Compose service names |
| `MCP_ALLOWED_ORIGINS` | Exact allowed browser origins | Empty |
| `MCP_REQUEST_BODY_MAX_BYTES` | Maximum authenticated HTTP request body | `1048576` |
| `MCP_REQUEST_BODY_TIMEOUT_SECONDS` | Request-body read timeout | `10` |
| `QDRANT_READ_ONLY` | Hide or block mutating tools | `false` |
| `MCP_ADMIN_TOOLS_ENABLED` | Enable the two optional admin-only tools, for 79 active tools instead of 77 | `false` |
| `QDRANT_ALLOW_ARBITRARY_FILTER` | Permit raw provider filters | `false` |

Upload and background-job quotas are enforced per server process/container, not across a replicated
fleet. In Portal mode an owner is the authenticated Portal tenant. Standalone and local modes use
the shared process principal, so all callers of that process share one owner budget.
The stock container's ephemeral `/tmp` is a 1 GiB tmpfs shared by uploads, job records, OCR
intermediates, and other temporary data.

| Quota variable | Default |
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

See the [environment template](https://github.com/MADPANDA3D/QDRANT-MCP/blob/v2.0.0/.env.example)
for the complete placeholder-only surface. Allowlist and workload
limits are deployment policy, not universal recommendations; tune them for the data and hosts you
actually trust.

## Tools

Start with the five native navigation tools:

1. `check_configuration` — validate access, connector, embedding, cache, and upload readiness
   without returning credential values.
2. `list_capabilities` — inspect capability groups and catalog counts; request full descriptors only
   when a catalog client needs them.
3. `find_tools` — rank tools for a natural-language task.
4. `get_tool_usage` — retrieve one complete descriptor before a complex or destructive call.
5. `get_endpoint_coverage` — inspect provider coverage and deliberate exclusions.

For retrieval, begin with `qdrant-build-context`, `qdrant-find`, or `qdrant-study-search` using small
`top_k` and output budgets. For large inputs, use document/manifest/textbook workflows rather than
placing whole files in a tool argument. Preview maintenance and destructive work with `dry_run=true`
where supported, inspect the returned diff, and supply the documented confirmation only after
review. The [maintenance playbooks](https://github.com/MADPANDA3D/QDRANT-MCP/blob/v2.0.0/docs/MAINTENANCE_PLAYBOOKS.md)
contain safe sequences.

## Develop and verify

```bash
uv sync --frozen --group dev --python 3.12.13
uv run python -m compileall -q src tests scripts
uv run pytest
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run python scripts/check_source_safety.py
uv build
uv run twine check dist/*
uv run python scripts/check_package_archives.py
```

Public CI uses no live provider credentials. Its authenticated HTTP smokes are provider-free and its
local FastEmbed image smoke runs with networking disabled. Tests must use unmistakably synthetic
credentials and must not contact a live Qdrant cluster, Portal, tenant, or deployment.

## Release contract

The clean release line is intentionally narrow:

Before tagging, the canonical repository's GitHub Actions identity must have write access to the
`qdrant-mcp-server` GHCR package namespace. Repository replacement or renaming does not itself
transfer an existing package's Actions access. The linked package must also be public before the
anonymous candidate and promoted-tag gates can pass. As an explicit operator cutover gate, read back
successful public **Verify** and **CodeQL** runs for the exact current default-branch SHA before
creating the tag.

1. a maintainer creates the exact **annotated** tag `v2.0.0` at the current protected `main` tip; the
   release job refetches the default branch and requires exact SHA equality;
2. locked Python tests, lint, dependency audit, secret/history scan, a successful public CodeQL job
   for the exact commit, package allowlist, deterministic double-build, and Compose validation pass;
3. the canonical source repository must already be public before any package mutation;
4. the image job publishes only the run-scoped
   `candidate-<commit-sha>-<workflow-run-id>` reference with SBOM and provenance, then verifies,
   scans, attests, anonymously pulls, and preflights every stable tag as absent or already identical
   before it starts through Compose with `--no-build` and smokes that exact digest;
5. PyPI Trusted Publishing publishes `mad-mcp-qdrant==2.0.0`, and the workflow verifies the exact
   public files and PyPI provenance;
6. only after PyPI succeeds does a separate job attach `2.0.0`, `2.0`, `2`, and `latest` to the same
   candidate digest without rebuilding; the promotion job repeats the absent-or-identical check for
   race protection, and every promoted tag is rechecked anonymously;
7. the GitHub Release is created last and records the immutable container digest.

No API token is accepted by the PyPI job. If a gate fails, correct the external condition and rerun
the failed jobs in the **same workflow run** so the run-scoped candidate remains stable. In
particular, do not start a new full run to repair a failure after image-tag promotion has begun; the
promotion job is retry-safe for absent or already-identical tags and refuses conflicting tags.

## Project policy

- [Security policy](https://github.com/MADPANDA3D/QDRANT-MCP/blob/v2.0.0/SECURITY.md)
- [Security model](https://github.com/MADPANDA3D/QDRANT-MCP/blob/v2.0.0/docs/security-model.md)
- [Deployment guide](https://github.com/MADPANDA3D/QDRANT-MCP/blob/v2.0.0/docs/deployment.md)
- [Portal and request-mode guide](https://github.com/MADPANDA3D/QDRANT-MCP/blob/v2.0.0/docs/portal-mode.md)
- [Contributing](https://github.com/MADPANDA3D/QDRANT-MCP/blob/v2.0.0/CONTRIBUTING.md)
- [Support](https://github.com/MADPANDA3D/QDRANT-MCP/blob/v2.0.0/SUPPORT.md)
- [Changelog](https://github.com/MADPANDA3D/QDRANT-MCP/blob/v2.0.0/CHANGELOG.md)
- [Notice and attribution](https://github.com/MADPANDA3D/QDRANT-MCP/blob/v2.0.0/NOTICE)

Apache-2.0 licensed. Qdrant is a trademark of Qdrant Solutions GmbH; this independent project is not
affiliated with or endorsed by Qdrant.
