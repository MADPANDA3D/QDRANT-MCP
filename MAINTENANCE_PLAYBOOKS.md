# Maintenance Playbooks

These sequences prioritize safe approvals, observability, and repeatability.

## Weekly Maintenance (Safe Default)

1. `qdrant-health-check` to confirm connectivity and index status.
2. `qdrant-metrics-snapshot` to capture collection size + index coverage.
3. `qdrant-audit-memories` (or `qdrant-submit-job` with `audit-memories`) for contract drift.
4. `qdrant-expire-memories` with `dry_run=true`, review `dry_run_diff`, then re-run with `confirm=true`.
5. `qdrant-dedupe-memories` with `dry_run=true`, review `dry_run_diff`, then re-run with `confirm=true`.
6. If audit shows contract gaps, run `qdrant-backfill-memory-contract` with `dry_run=true` first.

## Emergency Rollback (After Bad Cleanup)

1. `qdrant-list-snapshots` (or `qdrant-list-full-snapshots` if cluster-wide).
2. `qdrant-restore-snapshot` with `confirm=true` (admin-only).
3. `qdrant-collection-info` and `qdrant-metrics-snapshot` to verify recovery.

## Embedding Model Upgrade

1. Update embedding provider env vars (`EMBEDDING_PROVIDER`, `EMBEDDING_MODEL`, `EMBEDDING_VERSION`) and restart the server.
2. `qdrant-metrics-snapshot` to capture pre-upgrade stats.
3. `qdrant-reembed-points` with `dry_run=true` and `target_version` to preview the impact.
4. Review `dry_run_diff` and samples; adjust filters if needed.
5. Re-run `qdrant-reembed-points` with `dry_run=false` and `confirm=true` (use `qdrant-submit-job` for large runs).
6. Monitor `qdrant-job-progress` and `qdrant-job-logs` until complete.
7. Spot check relevance with `qdrant-find` or `qdrant-find-near-duplicates`.

## Safety Notes

- For destructive operations, create a snapshot (`qdrant-create-snapshot`) first.
- Prefer `dry_run=true` on mutators and review `dry_run_diff` before confirming.
