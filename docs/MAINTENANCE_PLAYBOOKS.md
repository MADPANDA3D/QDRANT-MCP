# Maintenance Playbooks

These sequences prioritize safe approvals, observability, and repeatability.

## Weekly Maintenance (Safe Default)

1. `qdrant-health-check` to confirm connectivity and index status.
2. `qdrant-metrics-snapshot` to capture collection size + index coverage.
3. `qdrant-audit-memories` (or `qdrant-submit-job` with `audit-memories`) for contract drift.
4. `qdrant-expire-memories` with `dry_run=true`, review `dry_run_diff`, then re-run with `confirm=true`.
5. `qdrant-dedupe-memories` with `dry_run=true`, review `dry_run_diff`, then re-run with `confirm=true`.
6. If audit shows contract gaps, run `qdrant-backfill-memory-contract` with `dry_run=true` first.

## Emergency Recovery (After Bad Cleanup)

1. Stop further MCP mutations and preserve the incident evidence.
2. Use `qdrant-list-snapshots` (or `qdrant-list-full-snapshots` if cluster-wide) to identify an
   operator-reviewed recovery point.
3. Perform restore through Qdrant's native authenticated administration path; snapshot restore is
   intentionally not exposed through this MCP server.
4. Use `qdrant-collection-info` and `qdrant-metrics-snapshot` to verify recovery before re-enabling
   agent mutations.

## Embedding Model Upgrade

1. Choose the runtime-specific upgrade path before changing collection data:
   - **Stock container / Compose FastEmbed:** do not override the baked model. Build a reviewed custom
     image that updates the model installer revision, archive/file hashes, `NOTICE`, expected vector
     dimension, offline smoke, and image identity together; scan it and deploy it by exact digest.
   - **Direct Python FastEmbed:** update `EMBEDDING_MODEL` and a truthful `EMBEDDING_VERSION`. Either
     allow normal FastEmbed acquisition or set a trusted `FASTEMBED_MODEL_PATH` plus
     `FASTEMBED_MODEL_REVISION`; restart the process after the model is locally available.
   - **Direct Python or request-scoped OpenAI-compatible provider:** update provider, model, version,
     host allowlist, and the appropriate server environment or request headers. Never move a tenant
     key into the shared server environment in request/Portal mode.
2. Run `check_configuration` and `qdrant-health-check`; record the active provider, model, version,
   vector name, and dimension.
3. Run `qdrant-metrics-snapshot` to capture pre-upgrade stats.
4. If the vector name and dimension are unchanged, preview `qdrant-reembed-points` with
   `dry_run=true` and the active `target_version`. If the vector contract changes, preview
   `qdrant-migrate-collection-embedding` instead so the active named vector can be added safely.
5. Review `dry_run_diff`, samples, storage headroom, and retrieval compatibility; adjust filters or
   batches before applying anything.
6. Re-run the selected operation with its documented confirmation. Use the bounded background-job
   path only where the selected tool supports it.
7. Monitor `qdrant-job-progress` and `qdrant-job-logs` when a job was submitted, then spot check
   relevance with `qdrant-find` or `qdrant-find-near-duplicates` before retiring an old vector.

## Safety Notes

- For destructive operations, create a snapshot first. `qdrant-create-snapshot` is available only
  when `MCP_ADMIN_TOOLS_ENABLED=true` and still requires `confirm=true`; otherwise use Qdrant's
  native authenticated snapshot administration path.
- Prefer `dry_run=true` on mutators and review `dry_run_diff` before confirming.
