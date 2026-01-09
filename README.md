# MADPANDA3D QDRANT MCP

<p align="center">
  <img src="assets/brand/header.jpg" alt="MADPANDA3D QDRANT MCP header" />
</p>

<p align="center">
  <a href="https://github.com/MADPANDA3D/QDRANT-MCP/actions/workflows/pre-commit.yaml">
    <img src="https://img.shields.io/github/actions/workflow/status/MADPANDA3D/QDRANT-MCP/pre-commit.yaml?branch=main" alt="pre-commit status" />
  </a>
  <a href="https://github.com/MADPANDA3D/QDRANT-MCP/actions/workflows/pytest.yaml">
    <img src="https://img.shields.io/github/actions/workflow/status/MADPANDA3D/QDRANT-MCP/pytest.yaml?branch=main" alt="tests status" />
  </a>
  <a href="https://github.com/MADPANDA3D/QDRANT-MCP/releases">
    <img src="https://img.shields.io/github/v/release/MADPANDA3D/QDRANT-MCP?display_name=tag" alt="release" />
  </a>
  <a href="https://github.com/MADPANDA3D/QDRANT-MCP/blob/main/LICENSE">
    <img src="https://img.shields.io/github/license/MADPANDA3D/QDRANT-MCP" alt="license" />
  </a>
</p>

<p align="center">
  <strong>Manage your Vector Database how you see fit</strong>
</p>
<p align="center">
  MADPANDA3D QDRANT MCP is a production-ready Model Context Protocol server for Qdrant.
  It turns your vector store into a managed memory layer with structured controls for
  ingest, retrieval, validation, and ongoing cleanup.
</p>
<p align="center">
  Use it to keep memories clean, deduped, and relevance-tuned over time. The toolkit
  includes safe dry-run previews, bulk maintenance jobs with progress reporting, and
  operational guardrails so agents can manage your database at scale without chaos.
</p>

## Deploy

<p align="center">
  <a href="https://example.com/railway-deploy">
    <img src="https://railway.app/button.svg" alt="Deploy on Railway" />
  </a>
  <a href="https://example.com/vercel-deploy">
    <img src="https://vercel.com/button" alt="Deploy with Vercel" />
  </a>
  <a href="https://example.com/vps-deploy">
    <img src="https://img.shields.io/badge/Deploy_to_VPS-Hostinger-blue?style=for-the-badge&logo=linux&logoColor=white" alt="Deploy to VPS" />
  </a>
  <a href="https://example.com/donate">
    <img src="https://img.shields.io/badge/Donate_to_the_Project-Support_Development-ff69b4?style=for-the-badge&logo=heart&logoColor=white" alt="Donate to the Project" />
  </a>
</p>

## Quickstart

<details>
<summary>Run with Docker</summary>

```bash
docker build -t mcp-server-qdrant .
docker run -d --name mcp-qdrant \
  --env-file .env \
  mcp-server-qdrant mcp-server-qdrant --transport streamable-http
```

</details>

<details>
<summary>Run locally (uvx)</summary>

```bash
QDRANT_URL=... COLLECTION_NAME=... uvx mcp-server-qdrant
```

</details>

## Tools

Most mutating tools support `dry_run` + `confirm` and return a `dry_run_diff` preview for safer approvals.

<details>
<summary>Core Memory Tools</summary>

- `qdrant-store`
- `qdrant-ingest-with-validation`
- `qdrant-ingest-document`
- `qdrant-find`
- `qdrant-update-point`
- `qdrant-patch-payload`
- `qdrant-list-points`
- `qdrant-get-points`
- `qdrant-count-points`

</details>

<details>
<summary>Housekeeping + Quality</summary>

- `qdrant-audit-memories`
- `qdrant-backfill-memory-contract`
- `qdrant-bulk-patch`
- `qdrant-dedupe-memories`
- `qdrant-find-near-duplicates`
- `qdrant-merge-duplicates`
- `qdrant-reembed-points`
- `qdrant-expire-memories`
- `qdrant-delete-points`
- `qdrant-delete-by-filter`
- `qdrant-delete-document`

</details>

<details>
<summary>Jobs + Progress</summary>

- `qdrant-submit-job`
- `qdrant-job-status`
- `qdrant-job-progress`
- `qdrant-job-logs`
- `qdrant-job-result`
- `qdrant-cancel-job`

</details>

<details>
<summary>Collection + Admin</summary>

- `qdrant-health-check`
- `qdrant-metrics-snapshot`
- `qdrant-ensure-payload-indexes`
- `qdrant-optimizer-status`
- `qdrant-update-optimizer-config`
- `qdrant-list-collections`
- `qdrant-collection-exists`
- `qdrant-collection-info`
- `qdrant-collection-stats`
- `qdrant-collection-vectors`
- `qdrant-collection-payload-schema`
- `qdrant-get-vector-name`
- `qdrant-list-aliases`
- `qdrant-collection-aliases`
- `qdrant-collection-cluster-info`
- `qdrant-list-snapshots`
- `qdrant-list-full-snapshots`
- `qdrant-list-shard-snapshots`
- `qdrant-create-snapshot`
- `qdrant-restore-snapshot`

</details>

## Configuration

<details>
<summary>Environment Variables</summary>

| Name                       | Description                                                         | Default Value                                                     |
|----------------------------|---------------------------------------------------------------------|-------------------------------------------------------------------|
| `QDRANT_URL`               | URL of the Qdrant server                                            | None                                                              |
| `QDRANT_API_KEY`           | API key for the Qdrant server                                       | None                                                              |
| `COLLECTION_NAME`          | Name of the default collection to use.                              | None                                                              |
| `QDRANT_VECTOR_NAME`       | Override vector name used by the MCP server                         | None                                                              |
| `QDRANT_LOCAL_PATH`        | Path to the local Qdrant database (alternative to `QDRANT_URL`)     | None                                                              |
| `EMBEDDING_PROVIDER`       | Embedding provider to use (`fastembed` or `openai`)                  | `fastembed`                                                       |
| `EMBEDDING_MODEL`          | Name of the embedding model to use                                  | `sentence-transformers/all-MiniLM-L6-v2`                          |
| `EMBEDDING_VECTOR_SIZE`    | Vector size override (required for unknown OpenAI models)           | unset                                                             |
| `EMBEDDING_VERSION`        | Embedding version label stored with each memory                     | unset                                                             |
| `OPENAI_API_KEY`           | OpenAI API key (required for `openai` provider)                     | unset                                                             |
| `OPENAI_BASE_URL`          | OpenAI-compatible base URL (optional)                               | unset                                                             |
| `OPENAI_ORG`               | OpenAI organization ID (optional)                                   | unset                                                             |
| `OPENAI_PROJECT`           | OpenAI project ID (optional)                                        | unset                                                             |
| `TOOL_STORE_DESCRIPTION`   | Custom description for the store tool                               | See default in `src/mcp_server_qdrant/settings.py`               |
| `TOOL_FIND_DESCRIPTION`    | Custom description for the find tool                                | See default in `src/mcp_server_qdrant/settings.py`               |
| `MCP_ADMIN_TOOLS_ENABLED`  | Enable admin-only tools (optimizer updates)                         | `false`                                                           |
| `MCP_MUTATIONS_REQUIRE_ADMIN` | Require admin access for mutating tools                         | `false`                                                           |
| `MCP_MAX_BATCH_SIZE`       | Max batch size for bulk operations                                  | `500`                                                             |
| `MCP_MAX_POINT_IDS`        | Max point id list size                                              | `500`                                                             |
| `MCP_STRICT_PARAMS`        | Reject unknown keys/filters and oversized text                      | `false`                                                           |
| `MCP_MAX_TEXT_LENGTH`      | Max text length before chunking                                     | `8000`                                                            |
| `MCP_DEDUPE_ACTION`        | Dedupe behavior (`update` or `skip`)                                | `update`                                                          |
| `MCP_INGEST_VALIDATION_MODE` | Validation mode (`allow`, `reject`, `quarantine`)                 | `allow`                                                           |
| `MCP_QUARANTINE_COLLECTION` | Collection name for quarantined memories                           | `jarvis-quarantine`                                               |
| `MCP_HEALTH_CHECK_COLLECTION` | Default collection for health check                              | unset                                                             |
| `MCP_SERVER_VERSION`       | Optional git SHA for telemetry                                      | unset                                                             |

Note: You cannot provide both `QDRANT_URL` and `QDRANT_LOCAL_PATH` at the same time.

</details>

<details>
<summary>Memory Contract</summary>

Stored memories are normalized to include at least:
`text`, `type`, `entities`, `source`, `created_at`, `updated_at`, `scope`, `confidence`, and `text_hash`.

Optional fields include `expires_at` / `ttl_days`, `labels`, validation metadata
(`validation_status`, `validation_errors`), merge markers (`merged_into`, `merged_from`),
plus embedding metadata
(`embedding_model`, `embedding_dim`, `embedding_provider`, `embedding_version`).

Document ingestion stores additional fields such as `doc_id`, `doc_title`, `doc_hash`,
`source_url`, `file_name`, `file_type`, `page_start`, `page_end`, and `section_heading`.

When a duplicate `text_hash` is found in the same `scope`, the server updates
`last_seen_at` and `reinforcement_count` instead of inserting a duplicate.

</details>

<details>
<summary>Maintenance Playbooks</summary>

See `MAINTENANCE_PLAYBOOKS.md` for recommended maintenance flows.

</details>

## Release & Versioning

This repo uses conventional commits and semantic-release. Every push to `main` runs the
release workflow, and a release is created only when commit messages warrant a version bump.

## License

MIT

<p align="center">
  <img src="assets/brand/logo.jpeg" alt="MADPANDA3D logo" width="140" />
</p>
<p align="center">
  <strong>MADPANDA3D</strong>
</p>
