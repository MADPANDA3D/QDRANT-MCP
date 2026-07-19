# Security model

This document describes the intended v2 trust boundaries. It is not a substitute for deploying
Qdrant, the reverse proxy, broker, and host securely.

## Assets

- standalone access tokens and Portal grants;
- Qdrant and embedding-provider credentials;
- tenant identity and request isolation;
- collection contents, vectors, payloads, and document inputs;
- integrity of tool metadata, package artifacts, images, and releases;
- availability of the MCP service and its upstream providers.

## Trust boundaries

### Local stdio

`stdio` trusts the local process boundary and inherits its environment. Do not expose the process's
standard streams through an untrusted multi-user relay.

### Standalone HTTP

Every MCP request needs a Bearer token. `QDRANT_CREDENTIAL_MODE=server` uses one operator-owned
connector from the process environment. `request` takes connector headers per request and disables
server fallback. Changing credential mode requires a restart.

### Portal HTTP

Every request needs a broker grant, broker-derived tenant identifier, and request-scoped connector.
The grant authenticates the broker; the tenant header partitions state but does not authenticate the
caller. A compromised or incorrectly implemented broker is inside the trusted computing base.

### Qdrant

Qdrant remains the authority for database access. Use a least-privilege Qdrant key and Qdrant-native
network controls. Tool confirmation cannot recover data that the Qdrant identity is allowed to
destroy.

## Credential handling

- Store credentials in a secret manager or protected runtime environment, never source control.
- Send HTTP credentials only over TLS.
- Compare service tokens without content-dependent timing where practical.
- Authenticate before parsing large bodies or performing provider work.
- Never include credentials in health, MCP responses, logs, traces, exception strings, image labels,
  package archives, or release notes.
- Request mode must construct provider context from the current request and dispose of it after use.
- Tenant-scoped caches and jobs require both bounded retention and tenant-qualified keys.

## Outbound requests and untrusted content

Remote document and textbook URL ingest is an SSRF and content-processing boundary. The default
empty `MCP_OUTBOUND_HOST_ALLOWLIST` denies URL fetching. Operators must list exact trusted hosts and
ports; redirects must be revalidated. HTTPS on port 443 is the default permitted transport.

Setting `MCP_ALLOW_INSECURE_OUTBOUND_HTTP=true` explicitly enables only standard HTTP/80. Nonstandard
HTTP ports remain forbidden. `MCP_OUTBOUND_ALLOWED_PORTS` controls trusted HTTPS ports and defaults
to 443. Plain HTTP permits interception and content substitution and should remain disabled.

Host allowlisting is a trust decision, not proof that returned content is safe. Prefer the bounded
`upload://` workflow when the source host is not trusted. Downloaded bytes, OCR input, document
metadata, Qdrant payloads, and embedding-provider responses remain untrusted data. Bound sizes,
timeouts, redirects, decompression, extraction, output, and concurrency.

`MCP_QDRANT_HOST_ALLOWLIST` and `MCP_QDRANT_ALLOWED_PORTS` form a separate boundary for
request-scoped Qdrant connector URLs. Custom request-scoped OpenAI-compatible base URLs must match
their own `MCP_OPENAI_HOST_ALLOWLIST` and `MCP_OPENAI_ALLOWED_PORTS`; the stock policy allows only
`api.openai.com:443`. These policies do not permit private, loopback, link-local, or cloud metadata
destinations in Portal/request mode. Never reuse a document-host setting as a provider connector
policy.

Credential-free local FastEmbed may remain the request-mode default. Stock standalone-request and
Portal Compose profiles blank every server-owned OpenAI credential and endpoint field. In a custom
direct-Python request-mode deployment, a server-owned OpenAI default must never become an implicit
tenant credential: startup therefore requires `MCP_DISABLE_DEFAULT_EMBEDDING_FALLBACK=true` in that
configuration, and the request must supply its own embedding provider, model, and OpenAI key.

## Tool risk controls

- Read-only mode should remove or reject mutating tools.
- Admin tools remain disabled unless explicitly enabled.
- Destructive tools require the catalog-defined confirmation flow.
- `dry_run` is a preview, not authorization and not a transaction guarantee.
- Raw provider filters remain disabled unless the operator accepts their expanded capability.
- Tool output, batch, upload, page, extracted-text, and job limits protect availability.

## Container boundary

The stock container is non-root, read-only, capability-free, PID/CPU/memory bounded, and publishes
only to loopback through Compose. Those controls reduce impact; they do not make a shared Docker
daemon or compromised host safe. The Docker daemon, host kernel, reverse proxy, and broker are trusted.

The stock FastEmbed model is copied into the read-only image at build time. Runtime sets
`HF_HUB_OFFLINE=1`; provider construction supplies both `specific_model_path` and
`local_files_only=True`. Direct Python installs keep normal FastEmbed cache/acquisition behavior only
when `FASTEMBED_MODEL_PATH` is unset. A configured local path is treated as operator-trusted code and
data, so protect it from untrusted writes.

Temporary uploads and textbook job state use ephemeral `/tmp`. Persisting them adds a data-retention
boundary and requires a separate permissions, encryption, backup, quota, and cleanup review.

## Supply chain and release

Dependencies are locked; Actions and base images are SHA/digest pinned; source history, packages,
filesystems, images, and exact published digests are scanned. Release artifacts and images receive
provenance/attestations when the source is public. PyPI uses Trusted Publishing without a stored API
token. A GitHub Release is created only after public package and anonymous image verification.

The image build obtains the FastEmbed archive only from
`https://storage.googleapis.com/qdrant-fastembed/sentence-transformers-all-MiniLM-L6-v2.tar.gz`,
with ambient proxies disabled, HTTPS redirects constrained to the same host on port 443, and a
96 MiB download ceiling. The archive must match SHA-256
`2735afe656e156af64ed603dbb1c96f3cae7f937286a8feb27fff7fa979f6a77`. The installer never calls
`extractall`: it copies only six exact regular files, checks each pinned size and SHA-256, rejects
links/duplicates/traversal/unexpected members, and ignores only the archive's exact bounded
AppleDouble metadata members. Those six files byte-match
`qdrant/all-MiniLM-L6-v2-onnx@5f1b8cd78bc4fb444dd171e59b18f3a3af89a079`. Release smoke then
loads that specific directory with container networking disabled and requires a 384-dimension
embedding. Model license and source attribution are in `NOTICE`.

These checks reduce accidental or known risk. They do not prove that dependencies, build runners, or
upstream registries are uncompromised.

## Explicit non-goals

- replacing Qdrant authorization, backups, encryption, or audit logging;
- securing a compromised host, Docker daemon, reverse proxy, or broker;
- treating a tenant header as authentication;
- making arbitrary internet URL ingestion safe;
- guaranteeing recovery of ephemeral upload or job state;
- guaranteeing that database content or tool output is trustworthy.
