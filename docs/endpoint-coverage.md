# Qdrant Endpoint Coverage

Provider reference surfaces:

- https://api.qdrant.tech/master/api-reference
- https://api.qdrant.tech/api-reference/collections
- https://api.qdrant.tech/api-reference/points
- https://api.qdrant.tech/api-reference/aliases/get-collections-aliases
- https://api.qdrant.tech/api-reference/snapshots/list-snapshots
- https://api.qdrant.tech/api-reference/service

## Coverage Matrix

| Qdrant area | Representative endpoint(s) | MCP coverage | Risk | Smoke/test status |
|---|---|---|---|---|
| Service health | `GET /` | `/health`, `qdrant-health-check` | Read-only | Provider-free image health smoke |
| Collections list | `GET /collections` | `qdrant-list-collections` | Read-only | Unit/integration tests |
| Collection create | `PUT /collections/{collection_name}` | `qdrant-create-collection` | Write | Unit/integration tests |
| Collection info | `GET /collections/{collection_name}` | `qdrant-collection-info`, `qdrant-collection-stats`, `qdrant-collection-vectors`, `qdrant-collection-payload-schema`, `qdrant-describe-collection`, `qdrant-summarize-collection-schema` | Read-only | Unit/integration tests |
| Collection vector schema update | `PUT /collections/{collection_name}/vectors/{vector_name}` | `qdrant-migrate-collection-embedding` | Admin write | Dry-run/confirm guarded |
| Collection existence | SDK existence check | `qdrant-collection-exists` | Read-only | Unit tests |
| Optimizer config | collection optimizer update | `qdrant-update-optimizer-config` | Admin write | Confirm/dry-run guarded |
| Points upsert | `PUT /collections/{collection_name}/points` | `qdrant-store`, `qdrant-ingest-with-validation`, `qdrant-ingest-document`, `qdrant-ingest-school-manifest` (`qdrant-ingest-manifest`, `qdrant-ingest-class-manifest` aliases), `qdrant-ingest-textbook` | Write | Unit/integration tests; governance metadata optional on validated ingest; manifest ingest reuses document ingest per item |
| MCP upload staging | N/A - MCP server temp upload bridge | `qdrant-start-file-upload`, `qdrant-append-file-upload`, `qdrant-finish-file-upload` | Write to bounded temp storage | Unit tests; enables `upload://` inputs without arbitrary `file://` reads |
| Point vector update | `PUT /collections/{collection_name}/points/vectors` | `qdrant-migrate-collection-embedding` | Admin write | Dry-run/confirm guarded |
| Points retrieve | `POST /collections/{collection_name}/points` | `qdrant-get-points` | Read-only | Unit/integration tests |
| Points search/query | query/search endpoints | `qdrant-build-context`, `qdrant-find`, `qdrant-find-short-term`, `qdrant-recommend-memories`, `qdrant-study-search` | Read-only external call | Unit/integration tests |
| Points scroll | `POST /collections/{collection_name}/points/scroll` | `qdrant-list-points`, `qdrant-suggest-filters`, maintenance scans | Read-only | Unit/integration tests |
| Points count | `POST /collections/{collection_name}/points/count` | `qdrant-count-points` | Read-only | Unit/integration tests |
| Payload set/overwrite | payload update endpoints | `qdrant-update-point`, `qdrant-patch-payload`, `qdrant-tag-memories`, `qdrant-link-memories`, maintenance tools | Write | Unit/integration tests; `qdrant-backfill-memory-contract` can dry-run/apply governance metadata |
| Payload indexes | payload index endpoints | `qdrant-ensure-payload-indexes`, automatic document/governance index checks | Write | Unit/integration tests |
| Delete points | delete by ids/filter endpoints | `qdrant-delete-points`, `qdrant-delete-by-filter`, `qdrant-delete-document` | Destructive | Confirm/dry-run guarded |
| Aliases read | `GET /aliases`, `GET /collections/{collection_name}/aliases` | `qdrant-list-aliases`, `qdrant-collection-aliases` | Read-only | Unit and catalog tests |
| Alias mutations | `POST /collections/aliases` | Excluded | Write | Not implemented |
| Snapshots list/create | collection/full/shard snapshot endpoints | `qdrant-list-snapshots`, `qdrant-list-full-snapshots`, `qdrant-list-shard-snapshots`, `qdrant-create-snapshot` | Admin read/write | Create is confirm/admin guarded |
| Snapshot download/upload/delete | snapshot file transfer endpoints | Excluded | Binary/destructive | Not implemented |
| Cluster info | collection cluster info endpoint | `qdrant-collection-cluster-info` | Read-only | Unit and catalog tests |
| Telemetry | service/cluster telemetry endpoints | Partially covered by `/health` and `qdrant-health-check` | Read-only, token-heavy | Raw telemetry excluded |

## Intentional Exclusions

- Snapshot file download/upload endpoints are excluded because MCP responses should not carry binary snapshot files.
- Snapshot delete endpoints are excluded until a dedicated destructive confirmation flow is added.
- Snapshot restore is excluded from the MCP surface; use Qdrant's native authenticated administration
  path after an operator reviews the recovery point.
- Alias mutation endpoints are excluded until alias switch operations have dry-run previews and confirmation semantics.
- Raw service/cluster telemetry is excluded by default because it can expose infrastructure details and produce token-heavy responses.
- Qdrant Cloud management APIs are out of scope; this MCP targets Qdrant database operations through user-provided Qdrant endpoints.

## Agent Guidance

- Use `check_configuration` before provider calls when setup is uncertain.
- Use `find_tools` for punctuation-normalized multi-token discovery, then
  call `get_tool_usage` for the complete lossless descriptor before a
  complex or destructive call.
- Use `list_capabilities` to choose a workflow. Pass
  `include_descriptors=true` only when a catalog client needs the complete
  versioned ToolManifest; the default response stays compact.
- The standard names are executable native tools. Existing Qdrant-prefixed
  navigation names remain callable as legacy compatibility tools.
- For general second-brain retrieval, use `qdrant-describe-collection` or
  `qdrant-suggest-filters` first when the collection schema is unknown, then use
  `qdrant-build-context` with `top_k=3-5`, `max_output_chars`, `group_by_doc`,
  and exact `memory_filter` values when available.
- Use `qdrant-find` with `response_mode="compact"` and `top_k=3-5` for lower
  level semantic search; add `metadata_fields`, `group_by_doc`,
  `max_chunks_per_doc`, `min_score`, and `max_output_chars` to control output.
- Use `qdrant-study-search` as a convenience helper for school collections; pass
  `class_code`, `subject`, `module`, `week`, `status`, `material_type`, `title`,
  `author`, `doc_id`, or `chapter` when known.
- Use `qdrant-ingest-school-manifest` for SCHOOL class-capture batches instead
  of one `qdrant-ingest-document` call per file. It applies shared
  `default_metadata`, accepts per-item text/base64/HTTP(S)/`upload://` sources,
  and can run compact search-back verification for each stored item.
  Compatibility aliases: `qdrant-ingest-manifest` and
  `qdrant-ingest-class-manifest`.
- Use `response_mode="payload"` only when the full stored payload is needed for the next action.
- Use document ingest tools for large source text or PDFs instead of sending large content directly to `qdrant-store`.
- HTTP(S) PDF download URLs do not need a `.pdf` suffix when the downloaded bytes have a PDF signature.
- For local files that are not reachable over HTTP(S), use
  `qdrant-start-file-upload`, append chunks with `qdrant-append-file-upload`,
  finish with `qdrant-finish-file-upload`, then pass the returned `source_url`
  value to `qdrant-ingest-document`, `qdrant-ingest-school-manifest`, or
  `qdrant-ingest-textbook`. The finish response also includes `upload_uri`,
  `uri`, and legacy `uploaded_file_uri` aliases with the same `upload://...`
  value. Direct `file://` paths remain disabled because they would point at the
  remote MCP server filesystem, not the caller workspace. The
  Portal path advertises a 512 KiB decoded append chunk limit so the
  base64 JSON-RPC body stays below broker request limits.
- Use `qdrant-validate-memory` to preview required memory fields and
  governance recommendations. Use `qdrant-ingest-with-validation` with
  `apply_governance_metadata=true` for new review-aware memories, and use
  `qdrant-backfill-memory-contract` with `include_governance_metadata=true`
  to preview/apply governance fields to existing points.
- Use `qdrant-migrate-collection-embedding` when `qdrant-find` reports an
  embedding dimension mismatch on a legacy collection. Start with `dry_run=true`;
  apply only with `dry_run=false` and `confirm=true` after reviewing the target
  vector name, vector size, and scan counts. For large collections, pass
  `max_points` and resume with the returned `next_offset`. If Qdrant returns
  404 for vector schema updates, the deployment does not expose the v1.18.0+
  in-place migration endpoint and needs a replacement-collection migration.
