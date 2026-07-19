import asyncio
import base64
import json
import stat
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
from fastmcp import FastMCP

import mcp_server_qdrant.mcp_server as mcp_module
from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.mcp_server import QdrantMCPServer, RequestQdrantOverrides
from mcp_server_qdrant.runtime_security import RequestIdentity
from mcp_server_qdrant.settings import (
    MemorySettings,
    QdrantSettings,
    RequestOverrideSettings,
    ToolSettings,
)


class CountingEmbeddingProvider(EmbeddingProvider):
    def __init__(self) -> None:
        self.query_calls = 0
        self.close_calls = 0

    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in documents]

    async def embed_query(self, query: str) -> list[float]:
        self.query_calls += 1
        return [float(self.query_calls), 0.2, 0.3]

    def get_vector_name(self) -> str:
        return "text"

    def get_vector_size(self) -> int:
        return 3

    async def aclose(self) -> None:
        self.close_calls += 1


class EmptyConnector:
    def __init__(self) -> None:
        self.close_calls = 0

    async def scroll_points_page(self, **_kwargs: Any) -> tuple[list[Any], None]:
        return [], None

    async def aclose(self) -> None:
        self.close_calls += 1


class BlockingConnector(EmptyConnector):
    def __init__(self) -> None:
        super().__init__()
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def scroll_points_page(self, **_kwargs: Any) -> tuple[list[Any], None]:
        self.started.set()
        await self.release.wait()
        return [], None


class SampleConnector(EmptyConnector):
    def __init__(self, point_id: str) -> None:
        super().__init__()
        self.point_id = point_id

    async def scroll_points_page(self, **_kwargs: Any) -> tuple[list[Any], None]:
        return [SimpleNamespace(id=self.point_id, payload={})], None


class FailingConnector(EmptyConnector):
    async def scroll_points_page(self, **_kwargs: Any) -> tuple[list[Any], None]:
        raise RuntimeError("synthetic failure")


def make_settings(state_dir: Path, **overrides: Any) -> MemorySettings:
    values: dict[str, Any] = {
        "MCP_TEXTBOOK_JOB_STATE_DIR": str(state_dir),
        "MCP_TEXTBOOK_MAX_FILE_BYTES": 64,
        "MCP_FILE_UPLOAD_MAX_CHUNK_BYTES": 32,
        "MCP_FILE_UPLOAD_MAX_ACTIVE_PER_OWNER": 2,
        "MCP_FILE_UPLOAD_MAX_ACTIVE_GLOBAL": 4,
        "MCP_FILE_UPLOAD_MAX_AGGREGATE_BYTES_PER_OWNER": 64,
        "MCP_FILE_UPLOAD_MAX_AGGREGATE_BYTES_GLOBAL": 128,
        "MCP_BACKGROUND_JOB_MAX_ACTIVE_PER_OWNER": 2,
        "MCP_BACKGROUND_JOB_MAX_ACTIVE_GLOBAL": 4,
        "MCP_BACKGROUND_JOB_MAX_RETAINED_PER_OWNER": 4,
        "MCP_BACKGROUND_JOB_MAX_RETAINED_GLOBAL": 8,
        "MCP_BACKGROUND_JOB_LOG_MESSAGE_MAX_CHARS": 128,
        "MCP_BACKGROUND_JOB_MAX_LOGS_PER_JOB": 20,
        "MCP_BACKGROUND_JOB_MAX_LOG_TAIL": 10,
        "MCP_BACKGROUND_JOB_MAX_RESULT_BYTES": 4096,
        "MCP_BACKGROUND_JOB_MAX_RECORD_BYTES": 32768,
    }
    values.update(overrides)
    return MemorySettings.model_validate(values)


def make_server(
    state_dir: Path,
    *,
    memory_settings: MemorySettings | None = None,
    provider: CountingEmbeddingProvider | None = None,
) -> QdrantMCPServer:
    return QdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=QdrantSettings.model_validate({"QDRANT_LOCAL_PATH": ":memory:"}),
        request_override_settings=RequestOverrideSettings(),
        memory_settings=memory_settings or make_settings(state_dir),
        embedding_provider=provider or CountingEmbeddingProvider(),
    )


def set_identity(
    monkeypatch: pytest.MonkeyPatch,
    identity: RequestIdentity | None,
) -> None:
    monkeypatch.setattr(mcp_module, "get_request_identity", lambda: identity)


def ctx(name: str = "tenant-state-test") -> SimpleNamespace:
    return SimpleNamespace(request_id=name)


async def submit_audit_job(server: QdrantMCPServer) -> tuple[str, asyncio.Task[Any]]:
    response = await server._registered_tools["qdrant-submit-job"].fn(  # noqa: SLF001
        ctx(),
        job_type="audit-memories",
        job_args={"collection_name": "memories"},
    )
    job_id = response["data"]["job_id"]
    task = server._job_tasks[job_id]  # noqa: SLF001
    return job_id, task


def test_typed_owner_keys_are_domain_separated(monkeypatch: pytest.MonkeyPatch) -> None:
    identities = (
        None,
        RequestIdentity(kind="standalone", subject="standalone"),
        RequestIdentity(kind="portal", subject="standalone"),
        RequestIdentity(kind="portal", subject="tenant-a"),
        RequestIdentity(kind="portal", subject="tenant-b"),
    )
    keys: list[str] = []
    for identity in identities:
        set_identity(monkeypatch, identity)
        keys.append(mcp_module._current_state_owner_key())  # noqa: SLF001

    assert len(set(keys)) == len(identities)
    assert all(len(key) == 64 for key in keys)
    assert all(
        (identity.subject if identity else "local") not in key
        for identity, key in zip(identities, keys)
    )


@pytest.mark.asyncio
async def test_uploads_are_owned_persisted_private_and_restart_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant_a = RequestIdentity(kind="portal", subject="tenant-a")
    tenant_b = RequestIdentity(kind="portal", subject="tenant-b")
    set_identity(monkeypatch, tenant_a)
    server = make_server(tmp_path)
    docx_mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

    started = await server._registered_tools["qdrant-start-file-upload"].fn(  # noqa: SLF001
        ctx(),
        file_name="report.docx",
        expected_size=3,
        mime_type=docx_mime,
    )
    upload_id = started["data"]["upload_id"]
    await server._registered_tools["qdrant-append-file-upload"].fn(  # noqa: SLF001
        ctx(),
        upload_id=upload_id,
        chunk_base64=base64.b64encode(b"doc").decode("ascii"),
        offset=0,
    )
    finished = await server._registered_tools["qdrant-finish-file-upload"].fn(  # noqa: SLF001
        ctx(), upload_id=upload_id
    )
    assert finished["data"]["uploaded_file_uri"] == f"upload://{upload_id}"

    upload_dir = tmp_path / "uploads"
    data_path = upload_dir / f"{upload_id}.upload"
    metadata_path = upload_dir / f"{upload_id}.json"
    assert stat.S_IMODE(tmp_path.stat().st_mode) == 0o700
    assert stat.S_IMODE(upload_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE(data_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(metadata_path.stat().st_mode) == 0o600
    persisted = metadata_path.read_text(encoding="utf-8")
    assert "tenant-a" not in persisted
    assert docx_mime in persisted

    config_a = await server._registered_tools["qdrant-check-configuration"].fn(ctx())  # noqa: SLF001
    assert config_a["data"]["file_uploads"]["owner_active_sessions"] == 1
    set_identity(monkeypatch, tenant_b)
    config_b = await server._registered_tools["qdrant-check-configuration"].fn(ctx())  # noqa: SLF001
    assert config_b["data"]["file_uploads"]["owner_active_sessions"] == 0
    with pytest.raises(ValueError, match="Upload was not found"):
        await server._registered_tools["qdrant-finish-file-upload"].fn(  # noqa: SLF001
            ctx(), upload_id=upload_id
        )
    with pytest.raises(ValueError, match="Upload was not found"):
        await server._registered_tools["qdrant-append-file-upload"].fn(  # noqa: SLF001
            ctx(),
            upload_id=upload_id,
            chunk_base64=base64.b64encode(b"x").decode("ascii"),
        )
    with pytest.raises(ValueError, match="Upload was not found"):
        server._claim_file_upload(f"upload://{upload_id}")  # noqa: SLF001

    restarted = make_server(tmp_path)
    with pytest.raises(ValueError, match="Upload was not found"):
        await restarted._registered_tools["qdrant-finish-file-upload"].fn(  # noqa: SLF001
            ctx(), upload_id=upload_id
        )
    set_identity(monkeypatch, tenant_a)
    hydrated = await restarted._registered_tools["qdrant-finish-file-upload"].fn(  # noqa: SLF001
        ctx(), upload_id=upload_id
    )
    assert hydrated["data"]["file_name"] == "report.docx"
    data, mime_type, file_name, size, _digest = await restarted._read_and_delete_uploaded_file(  # noqa: SLF001
        f"upload://{upload_id}"
    )
    assert (data, mime_type, file_name, size) == (b"doc", docx_mime, "report.docx", 3)
    assert not data_path.exists()
    assert not metadata_path.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "file_name",
    ["", ".", "..", "../secret", "dir/file.pdf", "dir\\file.pdf", "bad\nname", "x" * 256],
)
async def test_upload_filename_validation_rejects_paths_and_controls(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    file_name: str,
) -> None:
    set_identity(monkeypatch, RequestIdentity(kind="portal", subject="tenant-a"))
    server = make_server(tmp_path)
    with pytest.raises(ValueError, match="safe base name"):
        await server._registered_tools["qdrant-start-file-upload"].fn(  # noqa: SLF001
            ctx(), file_name=file_name
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("mime_type", ["text", "text/plain; charset=utf-8", "a/" + ("b" * 127)])
async def test_upload_mime_and_identifier_validation_is_bounded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mime_type: str,
) -> None:
    set_identity(monkeypatch, RequestIdentity(kind="portal", subject="tenant-a"))
    server = make_server(tmp_path)
    with pytest.raises(ValueError, match="valid type/subtype"):
        await server._registered_tools["qdrant-start-file-upload"].fn(  # noqa: SLF001
            ctx(), file_name="file.pdf", mime_type=mime_type
        )
    with pytest.raises(ValueError, match="Upload was not found"):
        await server._registered_tools["qdrant-finish-file-upload"].fn(  # noqa: SLF001
            ctx(), upload_id="../../attacker"
        )


@pytest.mark.asyncio
async def test_upload_active_and_aggregate_quotas_include_hydrated_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant_a = RequestIdentity(kind="portal", subject="tenant-a")
    set_identity(monkeypatch, tenant_a)
    loose = make_settings(
        tmp_path,
        MCP_TEXTBOOK_MAX_FILE_BYTES=8,
        MCP_FILE_UPLOAD_MAX_ACTIVE_PER_OWNER=3,
        MCP_FILE_UPLOAD_MAX_ACTIVE_GLOBAL=3,
        MCP_FILE_UPLOAD_MAX_AGGREGATE_BYTES_PER_OWNER=16,
        MCP_FILE_UPLOAD_MAX_AGGREGATE_BYTES_GLOBAL=24,
    )
    server = make_server(tmp_path, memory_settings=loose)
    upload_ids: list[str] = []
    for index in range(3):
        response = await server._registered_tools["qdrant-start-file-upload"].fn(  # noqa: SLF001
            ctx(), file_name=f"{index}.pdf"
        )
        upload_ids.append(response["data"]["upload_id"])

    strict = make_settings(
        tmp_path,
        MCP_TEXTBOOK_MAX_FILE_BYTES=8,
        MCP_FILE_UPLOAD_MAX_ACTIVE_PER_OWNER=1,
        MCP_FILE_UPLOAD_MAX_ACTIVE_GLOBAL=1,
        MCP_FILE_UPLOAD_MAX_AGGREGATE_BYTES_PER_OWNER=8,
        MCP_FILE_UPLOAD_MAX_AGGREGATE_BYTES_GLOBAL=8,
    )
    restarted = make_server(tmp_path, memory_settings=strict)
    assert len(restarted._file_upload_sessions) == 1  # noqa: SLF001
    assert len(list((tmp_path / "uploads").glob("*.upload"))) == 1

    remaining_id = next(iter(restarted._file_upload_sessions))  # noqa: SLF001
    await restarted._registered_tools["qdrant-append-file-upload"].fn(  # noqa: SLF001
        ctx(),
        upload_id=remaining_id,
        chunk_base64=base64.b64encode(b"12345678").decode("ascii"),
        offset=0,
    )
    with pytest.raises(ValueError, match="capacity"):
        await restarted._registered_tools["qdrant-start-file-upload"].fn(  # noqa: SLF001
            ctx(), file_name="extra.pdf"
        )


@pytest.mark.asyncio
async def test_upload_aggregate_byte_quota_rejects_inflight_growth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_identity(monkeypatch, RequestIdentity(kind="portal", subject="tenant-a"))
    settings = make_settings(
        tmp_path,
        MCP_TEXTBOOK_MAX_FILE_BYTES=8,
        MCP_FILE_UPLOAD_MAX_ACTIVE_PER_OWNER=3,
        MCP_FILE_UPLOAD_MAX_AGGREGATE_BYTES_PER_OWNER=8,
        MCP_FILE_UPLOAD_MAX_AGGREGATE_BYTES_GLOBAL=16,
    )
    server = make_server(tmp_path, memory_settings=settings)
    first = await server._registered_tools["qdrant-start-file-upload"].fn(  # noqa: SLF001
        ctx(), file_name="first.pdf"
    )
    await server._registered_tools["qdrant-append-file-upload"].fn(  # noqa: SLF001
        ctx(),
        upload_id=first["data"]["upload_id"],
        chunk_base64=base64.b64encode(b"123456").decode("ascii"),
    )
    second = await server._registered_tools["qdrant-start-file-upload"].fn(  # noqa: SLF001
        ctx(), file_name="second.pdf"
    )
    with pytest.raises(ValueError, match="Upload byte capacity"):
        await server._registered_tools["qdrant-append-file-upload"].fn(  # noqa: SLF001
            ctx(),
            upload_id=second["data"]["upload_id"],
            chunk_base64=base64.b64encode(b"789").decode("ascii"),
        )


@pytest.mark.asyncio
async def test_claimed_upload_bytes_remain_reserved_until_consumer_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_identity(monkeypatch, RequestIdentity(kind="portal", subject="tenant-a"))
    settings = make_settings(
        tmp_path,
        MCP_TEXTBOOK_MAX_FILE_BYTES=8,
        MCP_FILE_UPLOAD_MAX_ACTIVE_PER_OWNER=3,
        MCP_FILE_UPLOAD_MAX_AGGREGATE_BYTES_PER_OWNER=8,
        MCP_FILE_UPLOAD_MAX_AGGREGATE_BYTES_GLOBAL=16,
    )
    server = make_server(tmp_path, memory_settings=settings)
    started = await server._registered_tools["qdrant-start-file-upload"].fn(  # noqa: SLF001
        ctx(), file_name="claimed.pdf", expected_size=8
    )
    upload_id = started["data"]["upload_id"]
    await server._registered_tools["qdrant-append-file-upload"].fn(  # noqa: SLF001
        ctx(),
        upload_id=upload_id,
        chunk_base64=base64.b64encode(b"12345678").decode("ascii"),
    )
    await server._registered_tools["qdrant-finish-file-upload"].fn(  # noqa: SLF001
        ctx(), upload_id=upload_id
    )
    claimed = server._claim_file_upload(f"upload://{upload_id}")  # noqa: SLF001
    assert upload_id in server._claimed_file_upload_sessions  # noqa: SLF001
    config = await server._registered_tools["qdrant-check-configuration"].fn(ctx())  # noqa: SLF001
    assert config["data"]["file_uploads"]["owner_active_sessions"] == 1
    with pytest.raises(ValueError, match="Upload byte capacity"):
        await server._registered_tools["qdrant-start-file-upload"].fn(  # noqa: SLF001
            ctx(), file_name="extra.pdf", expected_size=1
        )
    server._remove_file_upload_session(claimed.upload_id, delete_file=True)  # noqa: SLF001
    assert upload_id not in server._claimed_file_upload_sessions  # noqa: SLF001


def test_ownerless_upload_records_are_rejected_and_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_identity(monkeypatch, RequestIdentity(kind="portal", subject="tenant-a"))
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir(parents=True, mode=0o700)
    upload_id = "d" * 32
    data_path = upload_dir / f"{upload_id}.upload"
    metadata_path = upload_dir / f"{upload_id}.json"
    data_path.write_bytes(b"x")
    metadata_path.write_text(
        json.dumps(
            {
                "version": 1,
                "upload_id": upload_id,
                "file_name": "legacy.pdf",
                "purpose": "document",
                "expected_size": 1,
                "expected_sha256": None,
                "mime_type": "application/pdf",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "expires_at": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
                "bytes_received": 1,
                "finalized": True,
            }
        ),
        encoding="utf-8",
    )
    data_path.chmod(0o600)
    metadata_path.chmod(0o600)

    server = make_server(tmp_path)
    assert upload_id not in server._file_upload_sessions  # noqa: SLF001
    assert not data_path.exists()
    assert not metadata_path.exists()


@pytest.mark.asyncio
async def test_job_access_and_cancel_are_owner_isolated_and_tasks_are_released(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant_a = RequestIdentity(kind="portal", subject="tenant-a")
    tenant_b = RequestIdentity(kind="portal", subject="tenant-b")
    set_identity(monkeypatch, tenant_a)
    server = make_server(tmp_path)
    connector = BlockingConnector()
    server._default_qdrant_connector = cast(Any, connector)  # noqa: SLF001
    job_id, task = await submit_audit_job(server)
    await connector.started.wait()

    set_identity(monkeypatch, tenant_b)
    for tool_name in (
        "qdrant-job-status",
        "qdrant-job-progress",
        "qdrant-job-logs",
        "qdrant-job-result",
        "qdrant-cancel-job",
        "qdrant-get-ingest-status",
        "qdrant-cancel-ingest",
    ):
        with pytest.raises(ValueError, match="^Job was not found\\.$"):
            await server._registered_tools[tool_name].fn(ctx(), job_id=job_id)  # noqa: SLF001

    set_identity(monkeypatch, tenant_a)
    cancelled = await server._registered_tools["qdrant-cancel-job"].fn(  # noqa: SLF001
        ctx(), job_id=job_id
    )
    assert cancelled["data"]["cancelled"] is True
    await task
    await asyncio.sleep(0)
    assert job_id not in server._job_tasks  # noqa: SLF001
    status = await server._registered_tools["qdrant-job-status"].fn(ctx(), job_id=job_id)  # noqa: SLF001
    assert status["data"]["status"] == "cancelled"
    assert "owner_key" not in status["data"]
    assert connector.close_calls == 0


@pytest.mark.asyncio
async def test_active_job_quota_rejects_excess_owner_work(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_identity(monkeypatch, RequestIdentity(kind="portal", subject="tenant-a"))
    settings = make_settings(
        tmp_path,
        MCP_BACKGROUND_JOB_MAX_ACTIVE_PER_OWNER=1,
        MCP_BACKGROUND_JOB_MAX_ACTIVE_GLOBAL=2,
    )
    server = make_server(tmp_path, memory_settings=settings)
    connector = BlockingConnector()
    server._default_qdrant_connector = cast(Any, connector)  # noqa: SLF001
    job_id, task = await submit_audit_job(server)
    await connector.started.wait()
    with pytest.raises(ValueError, match="Background job capacity"):
        await server._registered_tools["qdrant-submit-job"].fn(  # noqa: SLF001
            ctx(),
            job_type="audit-memories",
            job_args={"collection_name": "memories"},
        )
    await server._registered_tools["qdrant-cancel-job"].fn(ctx(), job_id=job_id)  # noqa: SLF001
    await task


@pytest.mark.asyncio
async def test_job_ownership_survives_hydration_and_ownerless_jobs_are_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant_a = RequestIdentity(kind="portal", subject="tenant-a")
    tenant_b = RequestIdentity(kind="portal", subject="tenant-b")
    set_identity(monkeypatch, tenant_a)
    server = make_server(tmp_path)
    server._default_qdrant_connector = cast(Any, EmptyConnector())  # noqa: SLF001
    job_id, task = await submit_audit_job(server)
    await task
    await asyncio.sleep(0)
    record_path = tmp_path / f"{job_id}.json"
    persisted = record_path.read_text(encoding="utf-8")
    assert stat.S_IMODE(record_path.stat().st_mode) == 0o600
    assert "tenant-a" not in persisted

    set_identity(monkeypatch, tenant_b)
    restarted = make_server(tmp_path)
    with pytest.raises(ValueError, match="^Job was not found\\.$"):
        await restarted._registered_tools["qdrant-job-status"].fn(ctx(), job_id=job_id)  # noqa: SLF001
    set_identity(monkeypatch, tenant_a)
    result = await restarted._registered_tools["qdrant-job-result"].fn(  # noqa: SLF001
        ctx(), job_id=job_id
    )
    assert result["data"]["status"] == "completed"

    legacy_id = "e" * 32
    legacy_path = tmp_path / f"{legacy_id}.json"
    legacy_path.write_text(
        json.dumps(
            {
                "job_id": legacy_id,
                "job_type": "audit-memories",
                "status": "completed",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "result": {},
            }
        ),
        encoding="utf-8",
    )
    legacy_path.chmod(0o600)
    second_restart = make_server(tmp_path)
    assert legacy_id not in second_restart._jobs  # noqa: SLF001
    assert not legacy_path.exists()


@pytest.mark.asyncio
async def test_job_identifier_tail_and_retention_limits_are_enforced(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_identity(monkeypatch, RequestIdentity(kind="portal", subject="tenant-a"))
    settings = make_settings(
        tmp_path,
        MCP_BACKGROUND_JOB_MAX_ACTIVE_PER_OWNER=1,
        MCP_BACKGROUND_JOB_MAX_ACTIVE_GLOBAL=2,
        MCP_BACKGROUND_JOB_MAX_RETAINED_PER_OWNER=2,
        MCP_BACKGROUND_JOB_MAX_RETAINED_GLOBAL=2,
    )
    server = make_server(tmp_path, memory_settings=settings)
    server._default_qdrant_connector = cast(Any, EmptyConnector())  # noqa: SLF001
    completed_ids: list[str] = []
    for _ in range(3):
        job_id, task = await submit_audit_job(server)
        completed_ids.append(job_id)
        await task
        await asyncio.sleep(0)

    assert completed_ids[0] not in server._jobs  # noqa: SLF001
    assert not (tmp_path / f"{completed_ids[0]}.json").exists()
    assert len(server._jobs) == 2  # noqa: SLF001
    current_id = completed_ids[-1]
    with pytest.raises(ValueError, match="^Job was not found\\.$"):
        await server._registered_tools["qdrant-job-status"].fn(  # noqa: SLF001
            ctx(), job_id="../../attacker"
        )
    for invalid_tail in (-1, settings.background_job_max_log_tail + 1):
        with pytest.raises(ValueError, match="configured job log tail limit"):
            await server._registered_tools["qdrant-job-logs"].fn(  # noqa: SLF001
                ctx(), job_id=current_id, tail=invalid_tail
            )
    with pytest.raises(ValueError, match="^Job was not found\\.$"):
        await server._registered_tools["qdrant-get-ingest-status"].fn(  # noqa: SLF001
            ctx(), job_id=current_id
        )


@pytest.mark.asyncio
async def test_large_results_and_secret_bearing_state_are_not_retained(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_identity(monkeypatch, RequestIdentity(kind="portal", subject="tenant-a"))
    settings = make_settings(
        tmp_path,
        MCP_BACKGROUND_JOB_MAX_RESULT_BYTES=512,
        MCP_BACKGROUND_JOB_MAX_RECORD_BYTES=8192,
        MCP_BACKGROUND_JOB_LOG_MESSAGE_MAX_CHARS=64,
    )
    server = make_server(tmp_path, memory_settings=settings)
    marker = "ATTACKER_SECRET_VALUE"
    server._default_qdrant_connector = cast(Any, SampleConnector(marker * 100))  # noqa: SLF001
    submitted = await server._registered_tools["qdrant-submit-job"].fn(  # noqa: SLF001
        ctx(),
        job_type="audit-memories",
        job_args={
            "collection_name": "memories",
            "include_samples": True,
            "sample_limit": 5,
        },
    )
    job_id = submitted["data"]["job_id"]
    task = server._job_tasks[job_id]  # noqa: SLF001
    await task
    status = await server._registered_tools["qdrant-job-status"].fn(ctx(), job_id=job_id)  # noqa: SLF001
    assert status["data"]["status"] == "failed"
    assert status["data"]["error"] == "Background job result exceeded the retention limit."
    persisted = (tmp_path / f"{job_id}.json").read_text(encoding="utf-8")
    assert marker not in persisted

    owned_key = mcp_module._current_state_owner_key()  # noqa: SLF001
    hydrated_id = "f" * 32
    hydrated_path = tmp_path / f"{hydrated_id}.json"
    secret = "SYNTHETIC_HEADER_SECRET"
    synthetic_windows_path = "\\".join(("C:", "Users", "sample-user", f"{secret}.txt"))
    hydrated_path.write_text(
        json.dumps(
            {
                "job_id": hydrated_id,
                "owner_key": owned_key,
                "job_type": "audit-memories",
                "status": "completed",
                "created_at": datetime.now(timezone.utc).isoformat(),
                "finished_at": datetime.now(timezone.utc).isoformat(),
                "result": {
                    "url": f"https://example.test/path?token={secret}",
                    "headers": {"Authorization": f"Bearer {secret}"},
                    "posix_path": f"path=/tmp/{secret}",
                    "windows_path": synthetic_windows_path,
                },
                "logs": [
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "stage": "runtime",
                        "event": "log",
                        "message": f"authorization=Bearer {secret}",
                    },
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "stage": "runtime",
                        "event": "log",
                        "message": "x" * 1000,
                    },
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "stage": "runtime",
                        "event": "log",
                        "message": f"path=/tmp/{secret}",
                    },
                    {
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "stage": "runtime",
                        "event": "log",
                        "message": synthetic_windows_path,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    hydrated_path.chmod(0o600)
    restarted = make_server(tmp_path, memory_settings=settings)
    returned = await restarted._registered_tools["qdrant-job-result"].fn(  # noqa: SLF001
        ctx(), job_id=hydrated_id
    )
    logs = await restarted._registered_tools["qdrant-job-logs"].fn(  # noqa: SLF001
        ctx(), job_id=hydrated_id, tail=10
    )
    serialized = json.dumps({"result": returned, "logs": logs}) + hydrated_path.read_text()
    assert secret not in serialized
    assert "?token=" not in serialized
    assert "exceeded the log limit" in serialized


@pytest.mark.asyncio
async def test_background_resources_transfer_until_done_and_cache_is_owner_scoped(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tenant_a = RequestIdentity(kind="portal", subject="tenant-a")
    tenant_b = RequestIdentity(kind="portal", subject="tenant-b")
    set_identity(monkeypatch, tenant_a)
    default_provider = CountingEmbeddingProvider()
    server = make_server(tmp_path, provider=default_provider)

    assert await server._embed_query_cached("same") == [1.0, 0.2, 0.3]  # noqa: SLF001
    assert await server._embed_query_cached("same") == [1.0, 0.2, 0.3]  # noqa: SLF001
    assert default_provider.query_calls == 1
    config_a = await server._registered_tools["qdrant-check-configuration"].fn(ctx())  # noqa: SLF001
    assert config_a["data"]["query_embedding_cache"]["owner_entries"] == 1

    set_identity(monkeypatch, tenant_b)
    assert await server._embed_query_cached("same") == [2.0, 0.2, 0.3]  # noqa: SLF001
    assert default_provider.query_calls == 2
    config_b = await server._registered_tools["qdrant-check-configuration"].fn(ctx())  # noqa: SLF001
    assert config_b["data"]["query_embedding_cache"]["owner_entries"] == 1

    request_provider = CountingEmbeddingProvider()
    request_connector = BlockingConnector()
    overrides = RequestQdrantOverrides(
        url="https://qdrant.example/",
        api_key="synthetic-secret",
        collection_name="memories",
        vector_name=None,
        embedding_provider=request_provider,
    )
    connector_token = server._connector_var.set(cast(Any, request_connector))  # noqa: SLF001
    provider_token = server._embedding_provider_var.set(request_provider)  # noqa: SLF001
    overrides_token = server._request_overrides_var.set(overrides)  # noqa: SLF001
    try:
        assert await server._embed_query_cached("request-query") == [1.0, 0.2, 0.3]  # noqa: SLF001
        assert await server._embed_query_cached("request-query") == [2.0, 0.2, 0.3]  # noqa: SLF001
        assert request_provider.query_calls == 2
        job_id, task = await submit_audit_job(server)
        await request_connector.started.wait()
        assert overrides.resources_claimed is True
        assert request_connector.close_calls == 0
        assert request_provider.close_calls == 0
        await server._registered_tools["qdrant-cancel-job"].fn(ctx(), job_id=job_id)  # noqa: SLF001
        await task
        await asyncio.sleep(0)
        assert request_connector.close_calls == 1
        assert request_provider.close_calls == 1
        assert job_id not in server._job_tasks  # noqa: SLF001
    finally:
        server._request_overrides_var.reset(overrides_token)  # noqa: SLF001
        server._embedding_provider_var.reset(provider_token)  # noqa: SLF001
        server._connector_var.reset(connector_token)  # noqa: SLF001


@pytest.mark.asyncio
@pytest.mark.parametrize("connector_type", [EmptyConnector, FailingConnector])
async def test_background_resources_close_after_success_and_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    connector_type: type[EmptyConnector],
) -> None:
    set_identity(monkeypatch, RequestIdentity(kind="portal", subject="tenant-a"))
    server = make_server(tmp_path)
    request_connector = connector_type()
    request_provider = CountingEmbeddingProvider()
    overrides = RequestQdrantOverrides(
        url="https://qdrant.example/",
        api_key="synthetic-secret",
        collection_name="memories",
        vector_name=None,
        embedding_provider=request_provider,
    )
    connector_token = server._connector_var.set(cast(Any, request_connector))  # noqa: SLF001
    provider_token = server._embedding_provider_var.set(request_provider)  # noqa: SLF001
    overrides_token = server._request_overrides_var.set(overrides)  # noqa: SLF001
    try:
        job_id, task = await submit_audit_job(server)
        await task
        await asyncio.sleep(0)
        assert request_connector.close_calls == 1
        assert request_provider.close_calls == 1
        assert job_id not in server._job_tasks  # noqa: SLF001
        expected_status = "failed" if connector_type is FailingConnector else "completed"
        status = await server._registered_tools["qdrant-job-status"].fn(  # noqa: SLF001
            ctx(), job_id=job_id
        )
        assert status["data"]["status"] == expected_status
    finally:
        server._request_overrides_var.reset(overrides_token)  # noqa: SLF001
        server._embedding_provider_var.reset(provider_token)  # noqa: SLF001
        server._connector_var.reset(connector_token)  # noqa: SLF001


@pytest.mark.asyncio
async def test_background_resources_close_when_task_is_cancelled_before_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_identity(monkeypatch, RequestIdentity(kind="portal", subject="tenant-a"))
    server = make_server(tmp_path)
    request_connector = EmptyConnector()
    request_provider = CountingEmbeddingProvider()
    overrides = RequestQdrantOverrides(
        url="https://qdrant.example/",
        api_key="synthetic-secret",
        collection_name="memories",
        vector_name=None,
        embedding_provider=request_provider,
    )
    connector_token = server._connector_var.set(cast(Any, request_connector))  # noqa: SLF001
    provider_token = server._embedding_provider_var.set(request_provider)  # noqa: SLF001
    overrides_token = server._request_overrides_var.set(overrides)  # noqa: SLF001
    try:
        job_id, task = await submit_audit_job(server)
        assert overrides.resources_claimed is True
        assert task.cancel() is True
        with pytest.raises(asyncio.CancelledError):
            await task
        for _ in range(4):
            await asyncio.sleep(0)
        assert request_connector.close_calls == 1
        assert request_provider.close_calls == 1
        assert job_id not in server._job_tasks  # noqa: SLF001
        assert server._jobs[job_id]["status"] == "cancelled"  # noqa: SLF001
    finally:
        server._request_overrides_var.reset(overrides_token)  # noqa: SLF001
        server._embedding_provider_var.reset(provider_token)  # noqa: SLF001
        server._connector_var.reset(connector_token)  # noqa: SLF001


def test_override_repr_hides_credentials_and_provider_objects() -> None:
    secret = "synthetic-secret-value"
    provider = CountingEmbeddingProvider()
    overrides = RequestQdrantOverrides(
        url=f"https://user:{secret}@qdrant.example/path?token={secret}",
        api_key=secret,
        collection_name="memories",
        vector_name=None,
        embedding_provider=provider,
    )

    rendered = repr(overrides)
    assert secret not in rendered
    assert overrides.api_key == secret
    assert overrides.embedding_provider is provider


def test_bare_request_overrides_fail_closed_for_ssrf_and_missing_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = make_server(tmp_path)
    settings = server.request_override_settings
    settings.allow_request_overrides = True
    settings.require_request_qdrant_url = True
    settings.require_request_qdrant_api_key = True
    settings.require_request_collection = False
    headers = {
        settings.qdrant_url_header: "https://qdrant.example:6333/",
        settings.qdrant_api_key_header: "synthetic-api-key",
    }

    with pytest.raises(ValueError, match="destination is not permitted"):
        server._build_request_overrides(headers)  # noqa: SLF001

    settings.qdrant_host_allowlist = ["127.0.0.1"]
    private_headers = dict(headers)
    private_headers[settings.qdrant_url_header] = "https://127.0.0.1:6333/"
    with pytest.raises(ValueError, match="destination is not permitted"):
        server._build_request_overrides(private_headers)  # noqa: SLF001

    secret = "URL_EMBEDDED_SECRET"
    settings.qdrant_host_allowlist = ["qdrant.example"]
    credential_headers = dict(headers)
    credential_headers[settings.qdrant_url_header] = (
        f"https://user:{secret}@qdrant.example:6333/path?token={secret}"
    )
    with pytest.raises(ValueError, match="destination is not permitted") as captured:
        server._build_request_overrides(credential_headers)  # noqa: SLF001
    assert secret not in str(captured.value)

    missing_key_headers = {settings.qdrant_url_header: "https://qdrant.example:6333/"}
    with pytest.raises(ValueError, match=settings.qdrant_api_key_header):
        server._build_request_overrides(missing_key_headers)  # noqa: SLF001

    monkeypatch.setattr(
        "mcp_server_qdrant.safe_fetch.socket.getaddrinfo",
        lambda *_args, **_kwargs: [(2, 1, 6, "", ("93.184.216.34", 6333))],
    )
    overrides = server._build_request_overrides(headers)  # noqa: SLF001
    assert overrides is not None
    assert overrides.url == "https://qdrant.example:6333/"
    assert overrides.api_key == "synthetic-api-key"


@pytest.mark.asyncio
async def test_exact_core_server_rejects_every_non_stdio_transport(
    tmp_path: Path,
) -> None:
    server = make_server(tmp_path)
    message = "QdrantMCPServer supports stdio only"
    for transport in ("http", "streamable-http", "sse", "websocket"):
        with pytest.raises(RuntimeError, match=message):
            server.http_app(transport=cast(Any, transport))
        with pytest.raises(RuntimeError, match=message):
            await server.run_async(transport=transport)
        with pytest.raises(RuntimeError, match=message):
            server.run(transport=transport)
    with pytest.raises(RuntimeError, match=message):
        await server.run_http_async()


@pytest.mark.asyncio
async def test_exact_core_server_preserves_stdio_entrypoints(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    server = make_server(tmp_path)
    async_calls: list[Any] = []
    sync_calls: list[Any] = []

    async def fake_run_async(
        _server: FastMCP,
        transport: Any = None,
        **_kwargs: Any,
    ) -> None:
        async_calls.append(transport)

    def fake_run(
        _server: FastMCP,
        transport: Any = None,
        **_kwargs: Any,
    ) -> None:
        sync_calls.append(transport)

    monkeypatch.setattr(FastMCP, "run_async", fake_run_async)
    monkeypatch.setattr(FastMCP, "run", fake_run)
    await server.run_async()
    await server.run_async(transport="stdio")
    server.run()
    server.run(transport="stdio")
    assert async_calls == ["stdio", "stdio"]
    assert sync_calls == ["stdio", "stdio"]


@pytest.mark.asyncio
async def test_bare_dispatch_closes_ordinary_connector_and_transfers_background_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    set_identity(monkeypatch, RequestIdentity(kind="portal", subject="tenant-a"))
    server = make_server(tmp_path)
    server.request_override_settings.allow_request_overrides = True
    connectors: list[BlockingConnector] = []

    def connector_factory(*_args: Any, **_kwargs: Any) -> BlockingConnector:
        connector = BlockingConnector()
        connectors.append(connector)
        return connector

    monkeypatch.setattr(mcp_module, "QdrantConnector", connector_factory)
    active_overrides: list[RequestQdrantOverrides] = []

    def build_overrides(_headers: Any) -> RequestQdrantOverrides:
        overrides = RequestQdrantOverrides(
            url="https://qdrant.example:6333/",
            api_key="synthetic-api-key",
            collection_name="memories",
            vector_name=None,
        )
        active_overrides.append(overrides)
        return overrides

    monkeypatch.setattr(server, "_build_request_overrides", build_overrides)

    async def delegate(
        _server: FastMCP,
        key: str,
        arguments: dict[str, Any],
    ) -> Any:
        if key == "qdrant-submit-job":
            return await server._registered_tools[key].fn(ctx(), **arguments)  # noqa: SLF001
        return []

    monkeypatch.setattr(FastMCP, "_call_tool_mcp", delegate)
    await server._call_tool_mcp("qdrant-list-collections", {})  # noqa: SLF001
    assert connectors[0].close_calls == 1
    assert active_overrides[0].resources_claimed is False

    submitted = await server._call_tool_mcp(  # noqa: SLF001
        "qdrant-submit-job",
        {"job_type": "audit-memories", "job_args": {"collection_name": "memories"}},
    )
    job_id = submitted["data"]["job_id"]
    task = server._job_tasks[job_id]  # noqa: SLF001
    background_connector = connectors[1]
    await background_connector.started.wait()
    assert active_overrides[1].resources_claimed is True
    assert background_connector.close_calls == 0
    await server._registered_tools["qdrant-cancel-job"].fn(ctx(), job_id=job_id)  # noqa: SLF001
    await task
    await asyncio.sleep(0)
    assert background_connector.close_calls == 1
    assert job_id not in server._job_tasks  # noqa: SLF001
