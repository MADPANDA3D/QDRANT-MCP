# Contributing

Thank you for improving Qdrant MCP. Contributions must preserve the server's fail-closed access,
credential isolation, deterministic catalog, package, and public/private boundaries.

## Before opening a change

- Search existing issues and confirm the request belongs in the documented Qdrant scope.
- Use a private security advisory for vulnerabilities or possible credential exposure.
- Never attach real tokens, Qdrant/OpenAI keys, tenant data, database exports, private hostnames,
  runtime `.env` files, agent instructions, tickets, handovers, or deployment evidence.
- Keep Qdrant, embedding-provider, Portal, and external URL calls mocked in tests. Public CI must
  remain provider-free.
- Do not create release tags from feature branches. The `v2.0.0` tag is maintainer-only, annotated,
  and must point to a commit already reachable from protected `main`.

## Development setup

```bash
git clone https://github.com/MADPANDA3D/QDRANT-MCP.git
cd QDRANT-MCP
uv sync --frozen --group dev --python 3.12.13
```

Run the release-relevant local gates:

```bash
uv run python -m compileall -q src tests scripts
uv run pytest
uv run ruff check src tests scripts
uv run ruff format --check src tests scripts
uv run pip-audit
uv run python scripts/check_source_safety.py
uv build
uv run twine check dist/*
uv run python scripts/check_package_archives.py
```

Test Python 3.13 as well when behavior or dependencies change.

Plain `pytest` deliberately skips the two integrations that initialize and download a real
FastEmbed model. Run those checks explicitly only from an environment where that external download
is intended:

```bash
QDRANT_MCP_RUN_MODEL_INTEGRATION=1 \
  uv run pytest tests/test_fastembed_integration.py tests/test_qdrant_integration.py
```

These opt-in checks use an in-memory Qdrant client; the network access is for the embedding model,
not a live Qdrant deployment.

## Tool-contract changes

A tool change is incomplete until all affected surfaces agree:

1. native FastMCP registration;
2. deterministic ToolManifest descriptor;
3. input and output descriptions and schemas;
4. risk, tier, admin, and confirmation annotations;
5. endpoint coverage and public documentation;
6. provider-free tests and wire smoke;
7. catalog version and descriptor-hash expectations.

Do not add arbitrary raw Qdrant request tools. New provider operations need a stable typed contract,
bounded output, explicit policy, normalized errors, and dry-run/confirmation semantics where risk
requires them.

## Security-sensitive changes

Authentication, tenant identity, request-scoped credentials, URL fetching, file handling, output or
log redaction, state retention, allowlists, and confirmation changes need adversarial tests. Do not
weaken a gate to make CI pass. Document residual risk in [docs/security-model.md](docs/security-model.md).

## Pull requests

- Keep the change focused and describe the user-visible outcome.
- Include tests and documentation for behavior changes.
- State the Python versions and gates you ran.
- Call out compatibility, catalog, release, and security impact.
- Keep private deployment work out of public source changes.
