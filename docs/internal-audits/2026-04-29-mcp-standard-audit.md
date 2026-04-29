# Qdrant MCP Standard Audit - 2026-04-29

## Summary Verdict

**Status:** Fails current MAD MCP standard.

QDRANT-MCP is live and functional through the MAD MCP Portal, but it is not yet
portal-hardened enough to treat as complete. The main blockers are missing
portal grant-token enforcement, missing OpenAI MCP tool annotations, missing
`.env.example`, missing endpoint coverage matrix, and Docker/deployment drift.

## Classification

- **Classification:** Live MCP.
- **Repo scope:** Actual MCP repository is `mcp-server-qdrant/`; parent
  `/home/madpanda/services/qdrant-mcp` is a meta/session root.
- **Language/framework:** Python FastMCP (`fastmcp==2.7.0`).
- **Transport:** Streamable HTTP for the hosted route; Dockerfile default still
  uses SSE.
- **Package managers:** Python venv present; `uv.lock` exists, but `uv` is not
  installed in the shell environment. Node release tooling uses `pnpm`.
- **Git:** `origin` is `https://github.com/MADPANDA3D/QDRANT-MCP.git`; branch
  `main`.
- **Worktree:** Dirty before audit. Existing changes touched release workflow,
  README, source, and tests. This audit did not modify source.

## Live And Broker Status

- **Public health:** `https://qdrant-mcp.madpanda3d.com/health` returned
  `200 OK`, version `5e0a5737f6e6e1fdbec69ee33731724fe8351719`, and
  `tool_count: 58`.
- **Canonical route:** `https://qdrant-mcp.madpanda3d.com/mcp` initializes
  successfully.
- **Trailing slash route:** `https://qdrant-mcp.madpanda3d.com/mcp/` returned
  `410 Gone` with the expected migration message.
- **Portal registry:** MAD MCP Portal lists service `qdrant` / `QDRANT-MCP` as
  configured with `toolCount: 58`.
- **Portal tool listing:** Broker listed 58 Qdrant tools.
- **Broker smoke:** `qdrant-validate-memory` returned a valid structured
  validation response through the portal.
- **Docker local status:** Not verified locally. Docker daemon access from this
  shell failed with permission denied on `/var/run/docker.sock`.

## Blocking Failures

### 1. Portal grant token is not enforced

**Evidence:**
- No source or docs reference `MCP_PORTAL_GRANT_TOKEN` or
  `X-MADPANDA-PORTAL-GRANT`.
- Direct public `initialize` and `tools/list` succeeded without a portal grant
  header.
- Direct tool call without portal grant failed only after Qdrant header
  validation (`x-qdrant-url`, `x-qdrant-api-key`), not at the service gate.

**Impact:** Direct access is not fail-closed at the MCP service boundary. The
standard requires broker grant validation before provider configuration or
provider calls are processed.

**Likely files:** `src/mcp_server_qdrant/settings.py`,
`src/mcp_server_qdrant/hosted_server.py`, `README.md`, `.env.example`,
`docs/VPS_SETUP.md`, tests.

### 2. Tool annotations are missing on every tool

**Evidence:**
- Live `tools/list` returned 58 tools.
- `tools_with_annotations=0`.
- Each tool descriptor contained only `name`, `description`, and `inputSchema`.

**Impact:** The service fails OpenAI Apps SDK / MCP readiness for
`readOnlyHint`, `destructiveHint`, and `openWorldHint`; `idempotentHint` is also
missing where applicable.

**Likely files:** `src/mcp_server_qdrant/mcp_server.py`, tests for tool
registration metadata.

### 3. No `.env.example`

**Evidence:** `.env.example` does not exist in `mcp-server-qdrant/`.

**Impact:** Required runtime, hosted header, portal grant, admin, and embedding
configuration are only partially documented in README prose. This fails the MAD
MCP standard and repo standards.

**Likely files:** Add `.env.example`; sync `README.md` and `docs/VPS_SETUP.md`.

### 4. No endpoint coverage matrix

**Evidence:** `docs/endpoint-coverage.md` does not exist.

**Impact:** The MCP exposes a broad Qdrant surface, but there is no auditable map
from official Qdrant REST endpoints to MCP tools or explicit exclusions. The
service cannot be marked endpoint-complete.

**Official docs consulted:**
- https://api.qdrant.tech/master/api-reference
- https://api.qdrant.tech/api-reference/points
- https://api.qdrant.tech/api-reference/aliases/get-collections-aliases
- https://api.qdrant.tech/api-reference/snapshots/list-snapshots
- https://api.qdrant.tech/api-reference/service

**Likely files:** Add `docs/endpoint-coverage.md`; optionally add
`qdrant-get-endpoint-coverage` navigation tool.

### 5. Missing agent-navigation tools

**Evidence:** Portal and live tool list have no dedicated tools equivalent to:
- `check_configuration`
- `list_capabilities`
- `get_endpoint_coverage`
- `get_tool_usage`

**Impact:** Agents must infer setup, risk, and workflows from README/tool names.
There is no safe built-in way to inspect configuration readiness or the coverage
matrix from the MCP itself.

**Likely files:** `src/mcp_server_qdrant/mcp_server.py`, README, tests.

### 6. Docker/deployment drift

**Evidence:**
- Dockerfile default command is `uvx mcp-server-qdrant --transport sse`, while
  hosted docs use streamable HTTP.
- Dockerfile has duplicated `ARG PACKAGE_VERSION` / version `ENV` block.
- No `docker-compose.yml` exists to codify `restart: unless-stopped` and
  `mcp-network`.
- README Docker examples omit `--restart unless-stopped`; one omits
  `--network mcp-network`.
- `docs/VPS_SETUP.md` still clones `https://github.com/qdrant/mcp-server-qdrant.git`
  instead of the MADPANDA3D repo.

**Impact:** A new deployment can drift from the live hosted configuration and
fail the runtime/network standard.

**Likely files:** `Dockerfile`, add `docker-compose.yml`, `README.md`,
`docs/VPS_SETUP.md`.

## Non-Blocking Improvements

- Tool descriptions are functional but often too terse for review. Several do
  not explain side effects, external system use, or safe follow-up behavior.
- `qdrant-restore-snapshot` accepts an `api_key` parameter directly in the tool
  schema. Even if it is not returned in normal output, secrets in tool arguments
  are risky for broker logs and review; prefer a documented broker header or
  admin-only server-side config.
- The live health route includes `tool_count` and matches portal count, but
  local Docker health could not be verified from this shell.
- Local lint/type tooling is not available in the current environment:
  `.venv` lacks `ruff`, `isort`, and `mypy`, and `uv` is not installed.

## Validation Run

- `./.venv/bin/python -m pytest` passed: **55 passed in 11.43s**.
- Direct live `/health` smoke passed: **200 OK**, `tool_count: 58`.
- Direct live `/mcp/` smoke passed expected deprecation behavior: **410 Gone**.
- Direct live `/mcp` initialized without portal grant header: **security
  failure**.
- Direct live `tools/list` count: **58**.
- Direct live annotation check: **0/58 tools have annotations**.
- Portal service list: `qdrant` configured, `toolCount: 58`.
- Portal broker tool list: **58 tools**.
- Portal broker safe smoke: `qdrant-validate-memory` returned structured output.
- Local Docker status: blocked by Docker socket permission denied.
- Lint/type checks: blocked by missing local toolchain (`uv`, `ruff`, `isort`,
  `mypy` unavailable).

## Recommended Next Commit Units

1. `fix(auth): require portal grant header`
   - Add `MCP_PORTAL_GRANT_TOKEN` setting.
   - Validate `X-MADPANDA-PORTAL-GRANT` fail-closed before request override
     validation.
   - Add missing/invalid grant tests and docs.

2. `feat(tools): add mcp annotations and navigation tools`
   - Add read/write/destructive/open-world annotations for all tools.
   - Add configuration, capabilities, endpoint coverage, and usage helper tools.

3. `docs(config): add env example and endpoint coverage matrix`
   - Add `.env.example`.
   - Add `docs/endpoint-coverage.md` tied to official Qdrant API docs.

4. `fix(deploy): align docker runtime with hosted standard`
   - Switch Docker default to streamable HTTP.
   - Add compose with `mcp-network` and `restart: unless-stopped`.
   - Fix MADPANDA repo clone URL and restart/network docs.

5. `chore(dev): restore local validation toolchain`
   - Ensure `uv` is available or document bootstrap.
   - Ensure dev venv can run ruff, format check, isort, and mypy without relying
     on auto-fixing pre-commit hooks.
