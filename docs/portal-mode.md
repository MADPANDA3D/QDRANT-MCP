# Portal and request-scoped mode

Portal mode lets a trusted broker expose one Qdrant MCP service while each request supplies its own
connector context. It does not make arbitrary public MCP access safe.

## Required request boundary

Every Portal MCP request must arrive over TLS with:

1. the configured Portal grant header;
2. the configured tenant identity header.

Every data-tool request also needs:

1. the configured Qdrant URL header;
2. the configured Qdrant API key header;
3. optional collection, vector, and embedding headers when the requested tool needs them.

Authenticated navigation tools intentionally work without connector headers so an agent can
discover and report the missing setup without making a provider call.

Defaults:

```text
X-MADPANDA-PORTAL-GRANT
X-MADPANDA-USER-ID
X-QDRANT-URL
X-QDRANT-API-KEY
X-COLLECTION-NAME
```

Header names may be configured at startup. Header values must never be committed, logged, returned
by tools, or copied into public issue reports.

## What each field means

- **Portal grant:** authenticates the trusted broker to this MCP service. It is a service secret.
- **Tenant ID:** partitions request-scoped state and policy. It is required but is not a credential
  and does not authorize a request on its own.
- **Qdrant URL and API key:** authenticate the request to its own Qdrant endpoint.
- **Collection/vector headers:** select request context; they do not grant provider permissions.
- **Embedding headers:** optionally select request-scoped embedding configuration and credentials.

The broker must derive the tenant identifier from its authenticated session. It must not trust a
tenant value supplied directly by an untrusted caller.

## Isolation rules

- Portal always forces `QDRANT_CREDENTIAL_MODE=request`.
- Server-owned Qdrant URL and API key values are cleared.
- Server-owned `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_ORG`, and `OPENAI_PROJECT` values are
  cleared; request-scoped OpenAI configuration may arrive only through the documented headers.
- Missing or invalid service authentication is rejected before parsing or provider work.
- Missing grant or tenant identity fails closed on every request. Missing connector context fails
  closed on data tools; provider-local navigation remains available to explain the setup gap.
- Request-scoped clients, caches, uploads, textbook jobs, and provider configuration must be keyed by
  tenant and bounded.
- One request may never inherit a previous request's Qdrant or embedding credentials.
- `MCP_QDRANT_HOST_ALLOWLIST` limits request connector destinations independently of document URL
  fetching.
- `MCP_QDRANT_ALLOWED_PORTS` limits request-scoped Qdrant ports; the stock policy permits 443 and
  6333.
- custom OpenAI-compatible base URLs must match `MCP_OPENAI_HOST_ALLOWLIST` and
  `MCP_OPENAI_ALLOWED_PORTS`; the stock policy permits only `api.openai.com:443`.
- the server's credential-free local FastEmbed default may be reused.
- request mode still rejects private, loopback, link-local, and cloud metadata destinations even if
  a hostname or port is mistakenly listed.

Standalone HTTP can also use `QDRANT_CREDENTIAL_MODE=request`; it keeps Bearer service authentication
but applies the same connector-header and no-fallback rules.

## Broker checklist

- authenticate the end user before constructing MCP headers;
- derive and normalize one stable tenant identifier;
- remove any untrusted inbound copies of protected headers;
- inject the Portal grant only on the trusted hop;
- transmit all headers over TLS;
- redact service and provider credentials from logs and traces;
- bound request size, duration, concurrency, and retained state;
- pass only Qdrant hosts allowed by operator policy;
- treat Qdrant payloads and tool output as untrusted content.
