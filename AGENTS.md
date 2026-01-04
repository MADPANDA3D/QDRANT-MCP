# Repository Guidelines

This repository hosts the Qdrant MCP server. Use these notes to keep contributions consistent and easy to review.

## Project Structure & Module Organization
- `src/mcp_server_qdrant/`: core server implementation and entrypoints (`main.py`, `server.py`).
- `src/mcp_server_qdrant/embeddings/`: embedding provider implementations.
- `tests/`: pytest suites; integration tests use the `test_*_integration.py` pattern.
- `README.md`, `VPS_SETUP.md`, `Dockerfile`, `.github/workflows/`: docs, ops, and CI definitions.

## Build, Test, and Development Commands
- `uv sync`: install runtime and dev dependencies from `pyproject.toml`/`uv.lock`.
- `uv run pytest`: run the test suite (CI uses this).
- `uv run pre-commit run --all-files`: run ruff, ruff-format, isort, and mypy hooks.
- `QDRANT_URL=... COLLECTION_NAME=... uvx mcp-server-qdrant`: run the server locally.
- `COLLECTION_NAME=mcp-dev fastmcp dev src/mcp_server_qdrant/server.py`: dev mode with MCP inspector.
- `docker build -t mcp-server-qdrant .`: build the container image.

## Coding Style & Naming Conventions
- Python with 4-space indentation; keep imports grouped and ordered.
- Formatting via `ruff format`; linting via `ruff` (auto-fix) and `isort --profile black`.
- Type checks run through `mypy` in pre-commit.
- Naming: modules/functions `snake_case`, classes `PascalCase`, constants `UPPER_SNAKE_CASE`.

## Testing Guidelines
- Framework: `pytest` with `pytest-asyncio` (`asyncio_mode=auto`).
- Naming: files `test_*.py`, functions `test_*` (see `pyproject.toml`).
- Add tests for behavior changes; prefer integration coverage for Qdrant/embedding flows.
- No explicit coverage threshold, but keep critical paths covered.

## Commit & Pull Request Guidelines
- Commit messages are short, imperative summaries; optional `type:` prefix (e.g., `fix:`, `deps:`) and issue/PR number in parentheses when available.
- PRs should include: what changed, why, how tested (commands), and any new/updated env vars or docs.
- If configuration changes, update the env var table in `README.md`.

## Configuration & Secrets
- Configure via environment variables; CLI args are not supported.
- Never commit secrets; use `.env` for local values.
- `QDRANT_URL` and `QDRANT_LOCAL_PATH` are mutually exclusive; set `OPENAI_API_KEY` only when using OpenAI embeddings.
