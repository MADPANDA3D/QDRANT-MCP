# Qdrant Tool Usability Audit - 2026-04-29

## Summary Verdict

**Status:** Live service is usable, but the deployed tool surface is too easy for
agents to misuse for school search.

The MAD MCP Portal currently exposes QDRANT-MCP with 58 tools. Basic collection
operations work, and the `school` collection is available with 3,166 points.
However, the live `qdrant-find` path returns full payloads by default and an
unfiltered school query returned 9,097 bytes for only 3 results, including noisy
cross-course matches. That is the main token-efficiency failure.

This branch fixes the local MCP surface by adding a school-first search tool,
more explicit school metadata filters, payload indexes for those filters,
collection-name suggestions in health checks, and schema descriptions for
previously ambiguous common parameters.

## Portal Evidence

- **Portal service:** `qdrant` / `QDRANT-MCP`, configured, native adapter,
  `toolCount: 58`.
- **Open portal tickets:** no Qdrant-specific tickets. The visible open tickets
  are for FFMPEG-MCP.
- **Collections:** `the-barn`, `school`, `jarvis-knowledgebase`, `memories`.
- **School collection:** green, optimizer `ok`, 3,166 points, vector
  `openai-text-embedding-3-large` with size 3,072.
- **Current health check:** reports `ok=false` because the configured default
  collection is `jarvis-knowledge-base`, while the live collection is
  `jarvis-knowledgebase`.
- **Current school search smoke:** `qdrant-find` against `school` with
  `top_k=3` returned 9,097 bytes and full payloads by default. The top results
  included music flashcards and Cengage boilerplate for a biology-style query,
  which shows why agents need compact output and first-class school filters.

## Live Tool Audit

### Navigation and Health

- `qdrant-health-check`
- `qdrant-list-collections`
- `qdrant-get-vector-name`

**Ease of use:** Health is useful, but the live output only says the configured
default collection is missing. It does not suggest close collection names, so a
single hyphen typo sends agents down a blind path.

**Fix in branch:** Missing collection health results now include
`available_collections` and `suggested_collections`.

### School and Memory Retrieval

- `qdrant-find`
- `qdrant-recommend-memories`
- `qdrant-find-short-term`
- `qdrant-get-points`
- `qdrant-list-points`
- `qdrant-count-points`
- `qdrant-find-near-duplicates`

**Ease of use:** The deployed `qdrant-find` accepts broad semantic queries but
does not lead agents toward course, subject, material, document, or chapter
filters. Full payload defaults also make one small search cost thousands of
output tokens.

**Fix in branch:** Adds `qdrant-study-search`, defaulting to
`MCP_STUDY_COLLECTION=school`, with exact filters for `class_code`, `subject`,
`material_type`, `title`, `doc_id`, and `chapter`. It wraps compact search by
default so agents get `id`, `score`, `snippet`, and selected metadata first.

### Memory and Document Writes

- `qdrant-store`
- `qdrant-cache-memory`
- `qdrant-promote-short-term`
- `qdrant-ingest-with-validation`
- `qdrant-ingest-document`
- `qdrant-ingest-textbook`
- `qdrant-update-point`
- `qdrant-patch-payload`
- `qdrant-bulk-patch`
- `qdrant-tag-memories`
- `qdrant-link-memories`
- `qdrant-backfill-memory-contract`
- `qdrant-reembed-points`

**Ease of use:** The ingest tools are the right path for school content because
they chunk documents server-side instead of forcing agents to send large text
directly to store tools. Tool guidance should steer agents there.

**Fix in branch:** README and endpoint guidance now explicitly tell agents to
use document-ingest tools for long school content and compact search for
retrieval. Search metadata now includes study fields so agents can identify the
right textbook/lesson without requesting the full payload.

### Validation and Cleanup

- `qdrant-validate-memory`
- `qdrant-audit-memories`
- `qdrant-dedupe-memories`
- `qdrant-merge-duplicates`
- `qdrant-expire-memories`
- `qdrant-expire-short-term`
- `qdrant-delete-points`
- `qdrant-delete-by-filter`
- `qdrant-delete-document`

**Ease of use:** The destructive tools are named clearly and require confirmation
where expected. The highest-risk usability issue is not deletion; it is agents
running broad retrieval, then trying to repair bad results manually.

**Fix in branch:** The study search path reduces broad retrieval pressure. The
existing annotation tests also require destructive tools to be marked
destructive for MCP clients.

### Collection Administration

- `qdrant-create-collection`
- `qdrant-collection-exists`
- `qdrant-collection-info`
- `qdrant-collection-stats`
- `qdrant-collection-vectors`
- `qdrant-collection-payload-schema`
- `qdrant-ensure-payload-indexes`
- `qdrant-metrics-snapshot`
- `qdrant-optimizer-status`
- `qdrant-update-optimizer-config`

**Ease of use:** `qdrant-collection-info` exposed the missing study indexes:
live `school` indexes include generic memory fields and `ingest_fingerprint`,
but not `metadata.class`, `metadata.subject`, `metadata.material_type`,
`metadata.title`, `metadata.chapter`, or related study fields.

**Fix in branch:** Default payload indexes now include school/search metadata so
`qdrant-ensure-payload-indexes` can bring existing collections up to the search
standard after deployment.

### Snapshots, Aliases, and Jobs

- `qdrant-list-aliases`
- `qdrant-collection-aliases`
- `qdrant-list-snapshots`
- `qdrant-list-full-snapshots`
- `qdrant-list-shard-snapshots`
- `qdrant-create-snapshot`
- `qdrant-restore-snapshot`
- `qdrant-submit-job`
- `qdrant-cancel-job`
- `qdrant-job-status`
- `qdrant-job-progress`
- `qdrant-job-logs`
- `qdrant-job-result`
- `qdrant-cancel-ingest`
- `qdrant-get-ingest-status`

**Ease of use:** These tools are mostly operational and should not be the first
choice for school workflows. Job status tools are useful for textbook ingest,
but retrieval agents need stronger defaults so they do not call admin-style
tools unnecessarily.

**Fix in branch:** Capability and usage navigation points agents toward
`qdrant-study-search`, `qdrant-find`, and ingest tools first.

## Changes Made From This Audit

- Added `qdrant-study-search` for compact school retrieval with study filters.
- Added `MCP_STUDY_COLLECTION`, defaulting to `school`.
- Added explicit memory filter fields for class/course, subject, module, status,
  year, textbook metadata, document metadata, and chapter metadata.
- Added study payload indexes to the default index set.
- Added compact metadata fields for course, subject, material type, title,
  author, edition, ISBN, publisher, and year.
- Added health-check suggestions for near-miss collection names.
- Filled missing schema descriptions for common parameters including
  `collection_name`, `query_filter`, and `shard_id`.
- Updated README and endpoint coverage guidance for school search and large
  document ingestion.

## Remaining Deployment Smoke

After this branch deploys:

1. Run `qdrant-health-check` and verify the missing default collection now
   includes close-name suggestions.
2. Run `qdrant-ensure-payload-indexes` on `school` with dry-run/review behavior
   where available, then apply only after confirming expected index additions.
3. Run `qdrant-study-search` with `class_code`, `subject`, or `material_type`
   filters and verify compact results stay small.
4. Run `qdrant-find` with `response_mode="payload"` only for a known result that
   actually needs full text.
5. Verify portal tool list shows the new navigation tools and
   `qdrant-study-search` after deployment.
