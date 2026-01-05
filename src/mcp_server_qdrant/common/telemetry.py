from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


def _find_git_root(start: Path) -> Path | None:
    for parent in [start, *start.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def _resolve_git_sha() -> str:
    env_sha = os.getenv("MCP_SERVER_VERSION") or os.getenv("GIT_SHA")
    if env_sha:
        return env_sha

    root = _find_git_root(Path(__file__).resolve())
    if not root:
        return "unknown"

    head_path = root / ".git" / "HEAD"
    if not head_path.exists():
        return "unknown"

    head = head_path.read_text().strip()
    if head.startswith("ref: "):
        ref = head.split(" ", maxsplit=1)[1]
        ref_path = root / ".git" / ref
        if ref_path.exists():
            return ref_path.read_text().strip()
        return "unknown"
    return head


SERVER_INSTANCE_ID = uuid.uuid4().hex
SERVER_START = time.monotonic()
SERVER_VERSION = _resolve_git_sha()


@dataclass
class RequestTelemetry:
    request_id: str
    start_time: float
    inputs: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


def new_request(ctx: Any, inputs: dict[str, Any]) -> RequestTelemetry:
    request_id = getattr(ctx, "request_id", None) or uuid.uuid4().hex
    return RequestTelemetry(
        request_id=request_id,
        start_time=time.perf_counter(),
        inputs=inputs,
    )


def add_warning(state: RequestTelemetry, message: str) -> None:
    state.warnings.append(message)


def _safe_json_size(value: Any) -> int:
    try:
        return len(json.dumps(value, default=str).encode("utf-8"))
    except TypeError:
        return 0


def finish_request(
    state: RequestTelemetry,
    data: Any,
    extra_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    elapsed_ms = int((time.perf_counter() - state.start_time) * 1000)
    meta: dict[str, Any] = {
        "request_id": state.request_id,
        "server_version": SERVER_VERSION,
        "server_instance_id": SERVER_INSTANCE_ID,
        "server_uptime_ms": int((time.monotonic() - SERVER_START) * 1000),
        "elapsed_ms": elapsed_ms,
        "bytes_in": _safe_json_size(state.inputs),
        "warnings": state.warnings,
    }
    if extra_meta:
        meta.update(extra_meta)

    response = {"data": data, "meta": dict(meta)}
    serialization_start = time.perf_counter()
    serialized = json.dumps(response, default=str).encode("utf-8")
    serialization_ms = int((time.perf_counter() - serialization_start) * 1000)

    meta["serialization_ms"] = serialization_ms
    meta["bytes_out"] = len(serialized)

    return {"data": data, "meta": meta}
