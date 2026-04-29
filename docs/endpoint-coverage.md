# Qdrant Endpoint Coverage

Source docs checked on 2026-04-29:

- https://api.qdrant.tech/master/api-reference
- https://api.qdrant.tech/api-reference/collections
- https://api.qdrant.tech/api-reference/points
- https://api.qdrant.tech/api-reference/aliases/get-collections-aliases
- https://api.qdrant.tech/api-reference/snapshots/list-snapshots
- https://api.qdrant.tech/api-reference/service

## Coverage Matrix

| Qdrant area | Representative endpoint(s) | MCP coverage | Risk | Smoke/test status |
|---|---|---|---|---|
| Service health | `GET /` | `/health`, `qdrant-health-check` | Read-only | Live `/health` smoke |
| Collections list | `GET /collections` | `qdrant-list-collections` | Read-only | Unit/integration tests |
| Collection create | `PUT /collections/{collection_name}` | `qdrant-create-collection` | Write | Unit/integration tests |
| Collection info | `GET /collections/{collection_name}` | `qdrant-collection-info`, `qdrant-collection-stats`, `qdrant-collection-vectors`, `qdrant-collection-payload-schema`, `qdrant-describe-collection`, `qdrant-summarize-collection-schema` | Read-only | Unit/integration tests |
| Collection existence | SDK existence check | `qdrant-collection-exists` | Read-only | Unit tests |
| Optimizer config | collection optimizer update | `qdrant-update-optimizer-config` | Admin write | Confirm/dry-run guarded |
| Points upsert | `PUT /collections/{collection_name}/points` | `qdrant-store`, `qdrant-ingest-with-validation`, `qdrant-ingest-document`, `qdrant-ingest-textbook` | Write | Unit/integration tests |
| Points retrieve | `POST /collections/{collection_name}/points` | `qdrant-get-points` | Read-only | Unit/integration tests |
| Points search/query | query/search endpoints | `qdrant-build-context`, `qdrant-find`, `qdrant-find-short-term`, `qdrant-recommend-memories`, `qdrant-study-search` | Read-only external call | Unit/integration tests |
| Points scroll | `POST /collections/{collection_name}/points/scroll` | `qdrant-list-points`, `qdrant-suggest-filters`, maintenance scans | Read-only | Unit/integration tests |
| Points count | `POST /collections/{collection_name}/points/count` | `qdrant-count-points` | Read-only | Unit/integration tests |
| Payload set/overwrite | payload update endpoints | `qdrant-update-point`, `qdrant-patch-payload`, `qdrant-tag-memories`, `qdrant-link-memories`, maintenance tools | Write | Unit/integration tests |
| Payload indexes | payload index endpoints | `qdrant-ensure-payload-indexes`, automatic document index checks | Write | Unit/integration tests |
| Delete points | delete by ids/filter endpoints | `qdrant-delete-points`, `qdrant-delete-by-filter`, `qdrant-delete-document` | Destructive | Confirm/dry-run guarded |
| Aliases read | `GET /aliases`, `GET /collections/{collection_name}/aliases` | `qdrant-list-aliases`, `qdrant-collection-aliases` | Read-only | Live/tool-list smoke |
| Alias mutations | `POST /collections/aliases` | Excluded | Write | Not implemented |
| Snapshots list/create/restore | collection/full/shard snapshot endpoints | `qdrant-list-snapshots`, `qdrant-list-full-snapshots`, `qdrant-list-shard-snapshots`, `qdrant-create-snapshot`, `qdrant-restore-snapshot` | Admin read/write | Confirm/admin guarded |
| Snapshot download/upload/delete | snapshot file transfer endpoints | Excluded | Binary/destructive | Not implemented |
| Cluster info | collection cluster info endpoint | `qdrant-collection-cluster-info` | Read-only | Live/tool-list smoke |
| Telemetry | service/cluster telemetry endpoints | Partially covered by `/health` and `qdrant-health-check` | Read-only, token-heavy | Raw telemetry excluded |

## Intentional Exclusions

- Snapshot file download/upload endpoints are excluded because MCP responses should not carry binary snapshot files.
- Snapshot delete endpoints are excluded until a dedicated destructive confirmation flow is added.
- Alias mutation endpoints are excluded until alias switch operations have dry-run previews and confirmation semantics.
- Raw service/cluster telemetry is excluded by default because it can expose infrastructure details and produce token-heavy responses.
- Qdrant Cloud management APIs are out of scope; this MCP targets Qdrant database operations through user-provided Qdrant endpoints.

## Agent Guidance

- Use `qdrant-check-configuration` before provider calls when setup is uncertain.
- Use `qdrant-list-capabilities` to choose a workflow.
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
- Use `response_mode="payload"` only when the full stored payload is needed for the next action.
- Use document ingest tools for large source text or PDFs instead of sending large content directly to `qdrant-store`.
