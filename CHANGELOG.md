# Changelog

All notable changes are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and releases use
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2.0.0] - 2026-07-19

### Added

- A clean Python 3.12/3.13 package and pinned Python 3.12.13 container line.
- Authenticated standalone and Portal HTTP modes plus local `stdio`.
- Server-owned and request-scoped Qdrant credential modes selected at startup.
- Required Portal tenant identity and request-scoped BYOK isolation.
- A deterministic policy-derived catalog: 77 default tools, 79 with admin enabled, and 45 in
  read-only mode, with agent-ready and legacy counts reported at runtime.
- Deny-by-default outbound document fetching with independent host, port, and insecure-HTTP controls.
- Non-root, read-only, loopback-only Compose profiles and immutable digest selection.
- Locked dependency, package allowlist, source/history scan, CodeQL, image scan, SBOM, provenance,
  attestation, anonymous GHCR pull, public source, and PyPI Trusted Publishing release gates.

### Migration notes

- HTTP clients must choose an authenticated mode; unauthenticated MCP-over-HTTP is not supported.
- Portal deployments must send the configured grant, tenant, Qdrant URL, and Qdrant API key headers.
- Request-scoped credentials no longer fall back to server-owned credentials.
- URL-based document and textbook ingest is disabled until destination hosts are explicitly trusted.
- The release image moves to `ghcr.io/madpanda3d/qdrant-mcp-server`.
- Legacy semantic-release, token-based PyPI publishing, internal audits, live deployment references,
  and workflow screenshots are excluded from the public release surface.

[Unreleased]: https://github.com/MADPANDA3D/QDRANT-MCP/compare/v2.0.0...HEAD
[2.0.0]: https://github.com/MADPANDA3D/QDRANT-MCP/releases/tag/v2.0.0
