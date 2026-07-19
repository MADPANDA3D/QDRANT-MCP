# Security policy

## Supported versions

Security fixes target the latest public `2.x` release. Legacy `1.x` packages, tags, images, and
deployments do not represent the v2 security contract.

| Version | Status |
|---|---|
| Latest public `2.x` | Supported |
| Legacy `1.x` | Not supported by the v2 policy |

## Report a vulnerability

Use a private
[GitHub security advisory](https://github.com/MADPANDA3D/QDRANT-MCP/security/advisories/new).
Do not open a public issue for vulnerabilities, suspected credential exposure, tenant data, or an
uncoordinated exploit.

Include only sanitized information needed to reproduce the problem:

- affected version, commit, or immutable image digest;
- runtime and credential mode;
- the smallest provider-free reproduction available;
- expected and observed behavior;
- security impact and known mitigations.

Never send a real access token, Portal grant, Qdrant key, OpenAI key, tenant identifier, database
export, private hostname, runtime `.env`, or operator filesystem path. Use obvious synthetic values.

Maintainers will validate the report, coordinate a fix and release, and publish sanitized guidance
when users need to act. No fixed response or embargo SLA is promised.

## Security boundary

Read [the security model](docs/security-model.md) before reporting documented behavior as a
vulnerability. The server authenticates access to its HTTP MCP surface; it does not replace Qdrant
authorization, reverse-proxy TLS, host hardening, broker authentication, or tenant policy.

Important boundaries:

- local `stdio` trusts the local process boundary;
- standalone HTTP requires a Bearer access token;
- Portal HTTP requires a broker grant and a tenant identifier, but the tenant header is not itself
  an authorization credential;
- request-scoped Qdrant and embedding credentials must never fall back to another tenant's or the
  server's credentials;
- URL ingest is denied until an operator explicitly trusts destination hosts and ports;
- Qdrant content and downloaded documents are untrusted input;
- the stock container keeps upload and textbook job state in ephemeral `/tmp`.
