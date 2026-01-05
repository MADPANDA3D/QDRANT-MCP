from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import models

from mcp_server_qdrant.settings import METADATA_PATH

DEFAULT_MEMORY_TYPE = "note"
DEFAULT_SCOPE = "global"
DEFAULT_SOURCE = "user"
DEFAULT_CONFIDENCE = 0.5

_TEXT_NORMALIZE_RE = re.compile(r"\s+")

REQUIRED_FIELDS = {
    "text",
    "type",
    "entities",
    "source",
    "created_at",
    "updated_at",
    "scope",
    "confidence",
}

OPTIONAL_FIELDS = {
    "expires_at",
    "ttl_days",
    "last_seen_at",
    "reinforcement_count",
    "embedding_model",
    "embedding_dim",
    "embedding_provider",
    "embedding_version",
    "text_hash",
    "chunk_index",
    "chunk_count",
    "parent_text_hash",
    "created_at_ts",
    "updated_at_ts",
    "expires_at_ts",
    "last_seen_at_ts",
}

ALLOWED_MEMORY_KEYS = REQUIRED_FIELDS | OPTIONAL_FIELDS

FILTER_FIELDS = {
    "type",
    "entities",
    "scope",
    "source",
    "created_at_from",
    "created_at_to",
    "updated_at_from",
    "updated_at_to",
    "expires_at_from",
    "expires_at_to",
    "confidence_min",
    "confidence_max",
    "text_hash",
}


@dataclass(frozen=True)
class EmbeddingInfo:
    provider: str
    model: str
    dim: int
    version: str


@dataclass
class MemoryRecord:
    text: str
    metadata: dict[str, Any]


class MemoryInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    text: str | None = Field(default=None, description="Memory text content.")
    type: str | None = Field(default=None, description="Memory type.")
    entities: list[str] | None = Field(
        default=None, description="Entities mentioned in the memory."
    )
    source: str | None = Field(default=None, description="Source of the memory.")
    created_at: Any | None = Field(default=None, description="Creation timestamp.")
    updated_at: Any | None = Field(default=None, description="Last update timestamp.")
    scope: str | None = Field(default=None, description="Scope for dedupe and search.")
    confidence: float | None = Field(default=None, description="Confidence score.")
    expires_at: Any | None = Field(
        default=None, description="Optional expiration timestamp."
    )
    ttl_days: int | None = Field(
        default=None, description="Optional TTL in days (used to set expires_at)."
    )


class MemoryFilterInput(BaseModel):
    model_config = ConfigDict(extra="allow")

    type: str | None = Field(default=None, description="Filter by memory type.")
    entities: list[str] | str | None = Field(
        default=None, description="Filter by entities (any match)."
    )
    scope: str | None = Field(default=None, description="Filter by scope.")
    source: str | None = Field(default=None, description="Filter by source.")
    created_at_from: Any | None = Field(
        default=None, description="Filter by creation time (start)."
    )
    created_at_to: Any | None = Field(
        default=None, description="Filter by creation time (end)."
    )
    updated_at_from: Any | None = Field(
        default=None, description="Filter by update time (start)."
    )
    updated_at_to: Any | None = Field(
        default=None, description="Filter by update time (end)."
    )
    expires_at_from: Any | None = Field(
        default=None, description="Filter by expiration time (start)."
    )
    expires_at_to: Any | None = Field(
        default=None, description="Filter by expiration time (end)."
    )
    confidence_min: float | None = Field(
        default=None, description="Minimum confidence score."
    )
    confidence_max: float | None = Field(
        default=None, description="Maximum confidence score."
    )
    text_hash: str | None = Field(default=None, description="Filter by text hash.")


def normalize_text(text: str) -> str:
    text = text.strip().lower()
    return _TEXT_NORMALIZE_RE.sub(" ", text)


def compute_text_hash(text: str) -> str:
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def chunk_text(text: str, max_length: int) -> list[str]:
    if max_length <= 0:
        return [text]
    if len(text) <= max_length:
        return [text]

    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_length, length)
        if end < length:
            split = text.rfind(" ", start, end)
            if split <= start:
                split = end
        else:
            split = end
        chunk = text[start:split].strip()
        if chunk:
            chunks.append(chunk)
        start = split

    return chunks or [text[:max_length]]


def _coerce_text(value: Any, field_name: str, strict: bool, warnings: list[str]) -> str:
    if isinstance(value, str):
        return value
    if strict:
        raise ValueError(f"{field_name} must be a string.")
    warnings.append(f"{field_name} coerced to string.")
    return str(value)


def _coerce_float(
    value: Any, field_name: str, strict: bool, warnings: list[str]
) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        if strict:
            raise ValueError(f"{field_name} must be a number.")
        warnings.append(f"{field_name} defaulted due to invalid value.")
        return None


def _parse_datetime(
    value: Any, field_name: str, strict: bool, warnings: list[str]
) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 1_000_000_000_000:
            seconds = seconds / 1000
        parsed = datetime.fromtimestamp(seconds, tz=timezone.utc)
    elif isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            if strict:
                raise ValueError(f"{field_name} must be ISO-8601 or epoch value.")
            warnings.append(f"{field_name} ignored due to invalid value.")
            return None
    else:
        if strict:
            raise ValueError(f"{field_name} must be datetime, string, or epoch.")
        warnings.append(f"{field_name} ignored due to invalid value.")
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    else:
        parsed = parsed.astimezone(timezone.utc)
    return parsed


def _datetime_to_iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _datetime_to_ms(value: datetime) -> int:
    return int(value.timestamp() * 1000)


def _extract_raw_dict(memory: MemoryInput | dict[str, Any] | None) -> dict[str, Any]:
    if memory is None:
        return {}
    if isinstance(memory, BaseModel):
        return memory.model_dump(exclude_none=True)
    return dict(memory)


def normalize_memory_input(
    *,
    information: str | None,
    metadata: dict[str, Any] | None,
    memory: MemoryInput | dict[str, Any] | None,
    embedding_info: EmbeddingInfo | None,
    strict: bool,
    max_text_length: int,
) -> tuple[list[MemoryRecord], list[str]]:
    warnings: list[str] = []
    raw = _extract_raw_dict(memory)

    if metadata:
        if memory is not None and strict:
            raise ValueError("Provide either memory or metadata, not both.")
        if memory is not None:
            warnings.append("metadata merged into memory payload.")
        raw.update(metadata)

    if information:
        if "text" in raw and raw["text"] != information:
            if strict:
                raise ValueError("information does not match memory.text.")
            warnings.append("information overrides memory.text.")
        raw["text"] = information

    if "text" not in raw:
        for fallback in ("content", "document"):
            if fallback in raw:
                raw["text"] = raw[fallback]
                warnings.append(f"{fallback} used as memory text.")
                break

    if "text" not in raw:
        raise ValueError("text is required to store memory.")

    extras = set(raw.keys()) - ALLOWED_MEMORY_KEYS - {"content", "document"}
    if extras:
        if strict:
            raise ValueError(f"Unknown memory keys: {sorted(extras)}")
        warnings.append(f"Unknown memory keys preserved: {sorted(extras)}")

    text = _coerce_text(raw["text"], "text", strict, warnings)

    if len(text) > max_text_length:
        if strict:
            raise ValueError(
                f"text length {len(text)} exceeds max {max_text_length}."
            )
        warnings.append("text chunked due to size.")

    memory_type = raw.get("type")
    if memory_type is None:
        if strict:
            raise ValueError("type is required.")
        warnings.append("type defaulted.")
        memory_type = DEFAULT_MEMORY_TYPE
    else:
        memory_type = _coerce_text(memory_type, "type", strict, warnings)

    entities = raw.get("entities")
    if entities is None:
        if strict:
            raise ValueError("entities is required.")
        warnings.append("entities defaulted to empty list.")
        entities_list: list[str] = []
    elif isinstance(entities, str):
        entities_list = [item.strip() for item in entities.split(",") if item.strip()]
        warnings.append("entities coerced to list.")
    elif isinstance(entities, list):
        entities_list = [
            _coerce_text(item, "entities", strict, warnings) for item in entities
        ]
    else:
        if strict:
            raise ValueError("entities must be a list or string.")
        warnings.append("entities coerced to list.")
        entities_list = [_coerce_text(entities, "entities", strict, warnings)]

    source = raw.get("source")
    if source is None:
        if strict:
            raise ValueError("source is required.")
        warnings.append("source defaulted.")
        source = DEFAULT_SOURCE
    else:
        source = _coerce_text(source, "source", strict, warnings)

    scope = raw.get("scope")
    if scope is None:
        if strict:
            raise ValueError("scope is required.")
        warnings.append("scope defaulted.")
        scope = DEFAULT_SCOPE
    else:
        scope = _coerce_text(scope, "scope", strict, warnings)

    confidence_value = _coerce_float(
        raw.get("confidence"), "confidence", strict, warnings
    )
    if confidence_value is None:
        if strict:
            raise ValueError("confidence is required.")
        warnings.append("confidence defaulted.")
        confidence_value = DEFAULT_CONFIDENCE
    if confidence_value < 0 or confidence_value > 1:
        if strict:
            raise ValueError("confidence must be between 0 and 1.")
        warnings.append("confidence clamped to [0,1].")
        confidence_value = max(0.0, min(1.0, confidence_value))

    now = datetime.now(timezone.utc)

    created_at = _parse_datetime(raw.get("created_at"), "created_at", strict, warnings)
    if created_at is None:
        if strict:
            raise ValueError("created_at is required.")
        warnings.append("created_at defaulted.")
        created_at = now

    updated_at = _parse_datetime(raw.get("updated_at"), "updated_at", strict, warnings)
    if updated_at is None:
        if strict:
            raise ValueError("updated_at is required.")
        warnings.append("updated_at defaulted.")
        updated_at = now

    if updated_at < created_at:
        if strict:
            raise ValueError("updated_at cannot be before created_at.")
        warnings.append("updated_at adjusted to created_at.")
        updated_at = created_at

    expires_at = _parse_datetime(raw.get("expires_at"), "expires_at", strict, warnings)
    ttl_days = raw.get("ttl_days")
    if ttl_days is not None:
        if not isinstance(ttl_days, int):
            if strict:
                raise ValueError("ttl_days must be an integer.")
            warnings.append("ttl_days coerced to int.")
            ttl_days = int(ttl_days)
        if expires_at is None:
            expires_at = created_at + timedelta(days=ttl_days)

    last_seen_at = _parse_datetime(
        raw.get("last_seen_at"), "last_seen_at", strict, warnings
    )
    if last_seen_at is None:
        last_seen_at = now

    reinforcement_count = raw.get("reinforcement_count")
    if reinforcement_count is None:
        reinforcement_count = 1
    elif not isinstance(reinforcement_count, int):
        if strict:
            raise ValueError("reinforcement_count must be an integer.")
        warnings.append("reinforcement_count coerced to int.")
        reinforcement_count = int(reinforcement_count)

    resolved_embedding = embedding_info or EmbeddingInfo(
        provider="unknown",
        model="unknown",
        dim=0,
        version="unknown",
    )

    existing_embedding = raw.get("embedding_model")
    if existing_embedding and existing_embedding != resolved_embedding.model:
        if strict:
            raise ValueError("embedding_model does not match current settings.")
        warnings.append("embedding_model overridden to current settings.")

    existing_provider = raw.get("embedding_provider")
    if existing_provider and existing_provider != resolved_embedding.provider:
        if strict:
            raise ValueError("embedding_provider does not match current settings.")
        warnings.append("embedding_provider overridden to current settings.")

    existing_version = raw.get("embedding_version")
    if existing_version and existing_version != resolved_embedding.version:
        if strict:
            raise ValueError("embedding_version does not match current settings.")
        warnings.append("embedding_version overridden to current settings.")

    if resolved_embedding.dim and raw.get("embedding_dim") not in (
        None,
        resolved_embedding.dim,
    ):
        if strict:
            raise ValueError("embedding_dim does not match current settings.")
        warnings.append("embedding_dim overridden to current settings.")

    base_metadata = dict(raw)
    for key in ("text", "content", "document", "text_hash"):
        base_metadata.pop(key, None)

    base_metadata.update(
        {
            "text": text,
            "type": memory_type,
            "entities": entities_list,
            "source": source,
            "created_at": _datetime_to_iso(created_at),
            "updated_at": _datetime_to_iso(updated_at),
            "scope": scope,
            "confidence": confidence_value,
            "embedding_model": resolved_embedding.model,
            "embedding_dim": resolved_embedding.dim,
            "embedding_provider": resolved_embedding.provider,
            "embedding_version": resolved_embedding.version,
            "created_at_ts": _datetime_to_ms(created_at),
            "updated_at_ts": _datetime_to_ms(updated_at),
            "last_seen_at": _datetime_to_iso(last_seen_at),
            "last_seen_at_ts": _datetime_to_ms(last_seen_at),
            "reinforcement_count": reinforcement_count,
        }
    )

    if ttl_days is not None:
        base_metadata["ttl_days"] = ttl_days

    if expires_at is not None:
        base_metadata["expires_at"] = _datetime_to_iso(expires_at)
        base_metadata["expires_at_ts"] = _datetime_to_ms(expires_at)

    chunks = (
        chunk_text(text, max_text_length) if len(text) > max_text_length else [text]
    )
    parent_hash = compute_text_hash(text)
    records: list[MemoryRecord] = []

    for index, chunk in enumerate(chunks):
        metadata_copy = dict(base_metadata)
        if len(chunks) > 1:
            metadata_copy["chunk_index"] = index
            metadata_copy["chunk_count"] = len(chunks)
            metadata_copy["parent_text_hash"] = parent_hash
        metadata_copy["text"] = chunk
        metadata_copy["text_hash"] = compute_text_hash(chunk)
        records.append(MemoryRecord(text=chunk, metadata=metadata_copy))

    return records, warnings


def build_memory_filter(
    memory_filter: MemoryFilterInput | dict[str, Any] | None,
    strict: bool,
    warnings: list[str],
) -> models.Filter | None:
    if not memory_filter:
        return None

    raw = _extract_raw_dict(memory_filter)

    extras = set(raw.keys()) - FILTER_FIELDS
    if extras:
        if strict:
            raise ValueError(f"Unknown filter keys: {sorted(extras)}")
        warnings.append(f"Unknown filter keys ignored: {sorted(extras)}")

    must: list[models.FieldCondition] = []

    def add_match(field: str, value: Any):
        if value is None:
            return
        must.append(
            models.FieldCondition(
                key=f"{METADATA_PATH}.{field}",
                match=models.MatchValue(value=value),
            )
        )

    add_match("type", raw.get("type"))
    add_match("scope", raw.get("scope"))
    add_match("source", raw.get("source"))
    add_match("text_hash", raw.get("text_hash"))

    entities = raw.get("entities")
    if entities is not None:
        if isinstance(entities, str):
            entities = [item.strip() for item in entities.split(",") if item.strip()]
        if not isinstance(entities, list):
            if strict:
                raise ValueError("entities filter must be a list or string.")
            warnings.append("entities filter coerced to list.")
            entities = [entities]
        must.append(
            models.FieldCondition(
                key=f"{METADATA_PATH}.entities",
                match=models.MatchAny(any=entities),
            )
        )

    def add_range(field: str, start: Any, end: Any):
        range_kwargs: dict[str, Any] = {}
        if start is not None:
            parsed_start = _parse_datetime(start, field, strict, warnings)
            if parsed_start is not None:
                range_kwargs["gte"] = _datetime_to_ms(parsed_start)
        if end is not None:
            parsed_end = _parse_datetime(end, field, strict, warnings)
            if parsed_end is not None:
                range_kwargs["lte"] = _datetime_to_ms(parsed_end)
        if range_kwargs:
            must.append(
                models.FieldCondition(
                    key=f"{METADATA_PATH}.{field}_ts",
                    range=models.Range(**range_kwargs),
                )
            )

    add_range("created_at", raw.get("created_at_from"), raw.get("created_at_to"))
    add_range("updated_at", raw.get("updated_at_from"), raw.get("updated_at_to"))
    add_range("expires_at", raw.get("expires_at_from"), raw.get("expires_at_to"))

    confidence_min = raw.get("confidence_min")
    confidence_max = raw.get("confidence_max")
    if confidence_min is not None or confidence_max is not None:
        if confidence_min is not None and not isinstance(confidence_min, (int, float)):
            if strict:
                raise ValueError("confidence_min must be a number.")
            warnings.append("confidence_min ignored due to invalid value.")
            confidence_min = None
        if confidence_max is not None and not isinstance(confidence_max, (int, float)):
            if strict:
                raise ValueError("confidence_max must be a number.")
            warnings.append("confidence_max ignored due to invalid value.")
            confidence_max = None
        range_kwargs = {}
        if confidence_min is not None:
            range_kwargs["gte"] = float(confidence_min)
        if confidence_max is not None:
            range_kwargs["lte"] = float(confidence_max)
        if range_kwargs:
            must.append(
                models.FieldCondition(
                    key=f"{METADATA_PATH}.confidence",
                    range=models.Range(**range_kwargs),
                )
            )

    if not must:
        return None

    return models.Filter(must=must)


def default_memory_indexes() -> dict[str, models.PayloadSchemaType]:
    return {
        f"{METADATA_PATH}.text_hash": models.PayloadSchemaType.KEYWORD,
        f"{METADATA_PATH}.scope": models.PayloadSchemaType.KEYWORD,
        f"{METADATA_PATH}.type": models.PayloadSchemaType.KEYWORD,
        f"{METADATA_PATH}.source": models.PayloadSchemaType.KEYWORD,
        f"{METADATA_PATH}.entities": models.PayloadSchemaType.KEYWORD,
        f"{METADATA_PATH}.created_at_ts": models.PayloadSchemaType.INTEGER,
        f"{METADATA_PATH}.updated_at_ts": models.PayloadSchemaType.INTEGER,
        f"{METADATA_PATH}.expires_at_ts": models.PayloadSchemaType.INTEGER,
        f"{METADATA_PATH}.confidence": models.PayloadSchemaType.FLOAT,
    }


def build_memory_backfill_patch(
    *,
    text: str | None,
    metadata: dict[str, Any] | None,
    embedding_info: EmbeddingInfo,
    strict: bool,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    patch: dict[str, Any] = {}
    metadata = dict(metadata or {})

    if not text:
        if strict:
            raise ValueError("text is required to backfill memory contract.")
        warnings.append("missing text; skipping backfill for this point.")
        return patch, warnings

    now = datetime.now(timezone.utc)

    if "text" not in metadata:
        patch["text"] = text

    if "type" not in metadata or not isinstance(metadata.get("type"), str):
        patch["type"] = (
            metadata.get("type")
            if isinstance(metadata.get("type"), str)
            else DEFAULT_MEMORY_TYPE
        )

    entities = metadata.get("entities")
    if entities is None:
        patch["entities"] = []
    elif not isinstance(entities, list):
        if strict:
            raise ValueError("entities must be a list for backfill.")
        warnings.append("entities coerced to list during backfill.")
        patch["entities"] = [str(entities)]

    if "source" not in metadata or not isinstance(metadata.get("source"), str):
        patch["source"] = (
            metadata.get("source")
            if isinstance(metadata.get("source"), str)
            else DEFAULT_SOURCE
        )

    if "scope" not in metadata or not isinstance(metadata.get("scope"), str):
        patch["scope"] = (
            metadata.get("scope")
            if isinstance(metadata.get("scope"), str)
            else DEFAULT_SCOPE
        )

    confidence_raw = metadata.get("confidence")
    confidence_value = _coerce_float(confidence_raw, "confidence", strict, warnings)
    if confidence_value is None:
        patch["confidence"] = DEFAULT_CONFIDENCE
    elif not isinstance(confidence_raw, (int, float)):
        patch["confidence"] = confidence_value

    created_at_raw = metadata.get("created_at")
    created_at = _parse_datetime(created_at_raw, "created_at", strict, warnings)
    if created_at is None:
        created_at = now
        patch["created_at"] = _datetime_to_iso(created_at)
    elif "created_at" not in metadata:
        patch["created_at"] = _datetime_to_iso(created_at)

    updated_at_raw = metadata.get("updated_at")
    updated_at = _parse_datetime(updated_at_raw, "updated_at", strict, warnings)
    if updated_at is None:
        updated_at = created_at
        patch["updated_at"] = _datetime_to_iso(updated_at)
    elif "updated_at" not in metadata:
        patch["updated_at"] = _datetime_to_iso(updated_at)

    if updated_at < created_at:
        warnings.append("updated_at was before created_at; adjusted during backfill.")
        updated_at = created_at
        patch["updated_at"] = _datetime_to_iso(updated_at)

    created_at_ts = metadata.get("created_at_ts")
    if not isinstance(created_at_ts, int):
        patch["created_at_ts"] = _datetime_to_ms(created_at)

    updated_at_ts = metadata.get("updated_at_ts")
    if not isinstance(updated_at_ts, int):
        patch["updated_at_ts"] = _datetime_to_ms(updated_at)

    expires_at_raw = metadata.get("expires_at")
    expires_at = _parse_datetime(expires_at_raw, "expires_at", strict, warnings)
    if expires_at is None and isinstance(metadata.get("ttl_days"), int):
        expires_at = created_at + timedelta(days=int(metadata["ttl_days"]))
        if "expires_at" not in metadata:
            patch["expires_at"] = _datetime_to_iso(expires_at)
    if expires_at is not None:
        expires_at_ts = metadata.get("expires_at_ts")
        if not isinstance(expires_at_ts, int):
            patch["expires_at_ts"] = _datetime_to_ms(expires_at)

    last_seen_at_raw = metadata.get("last_seen_at")
    last_seen_at = _parse_datetime(last_seen_at_raw, "last_seen_at", strict, warnings)
    if last_seen_at is None:
        last_seen_at = updated_at
        patch["last_seen_at"] = _datetime_to_iso(last_seen_at)
    elif "last_seen_at" not in metadata:
        patch["last_seen_at"] = _datetime_to_iso(last_seen_at)

    last_seen_at_ts = metadata.get("last_seen_at_ts")
    if not isinstance(last_seen_at_ts, int):
        patch["last_seen_at_ts"] = _datetime_to_ms(last_seen_at)

    reinforcement_count = metadata.get("reinforcement_count")
    if not isinstance(reinforcement_count, int) or reinforcement_count < 1:
        patch["reinforcement_count"] = 1

    if not isinstance(metadata.get("text_hash"), str):
        patch["text_hash"] = compute_text_hash(text)

    if "embedding_provider" not in metadata:
        patch["embedding_provider"] = embedding_info.provider
    if "embedding_model" not in metadata:
        patch["embedding_model"] = embedding_info.model
    if "embedding_dim" not in metadata or not isinstance(
        metadata.get("embedding_dim"), int
    ):
        patch["embedding_dim"] = embedding_info.dim
    if "embedding_version" not in metadata:
        patch["embedding_version"] = embedding_info.version

    return patch, warnings
