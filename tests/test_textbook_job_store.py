import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from mcp_server_qdrant.embeddings.base import EmbeddingProvider
from mcp_server_qdrant.mcp_server import QdrantMCPServer
from mcp_server_qdrant.settings import (
    MemorySettings,
    QdrantSettings,
    RequestOverrideSettings,
    ToolSettings,
)


class DummyEmbeddingProvider(EmbeddingProvider):
    async def embed_documents(self, documents: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in documents]

    async def embed_query(self, query: str) -> list[float]:
        return [0.1, 0.2, 0.3]

    def get_vector_name(self) -> str:
        return "text"

    def get_vector_size(self) -> int:
        return 3


def make_server(job_state_dir: Path, retention_hours: int = 168) -> QdrantMCPServer:
    return QdrantMCPServer(
        tool_settings=ToolSettings(),
        qdrant_settings=QdrantSettings.model_validate({"QDRANT_LOCAL_PATH": ":memory:"}),
        request_override_settings=RequestOverrideSettings(),
        memory_settings=MemorySettings.model_validate(
            {
                "MCP_TEXTBOOK_JOB_STATE_DIR": str(job_state_dir),
                "MCP_TEXTBOOK_JOB_STATE_RETENTION_HOURS": retention_hours,
            }
        ),
        embedding_provider=DummyEmbeddingProvider(),
    )


def write_job(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_hydrate_marks_running_job_failed_after_restart(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    job_id = "running-job"
    write_job(
        tmp_path / f"{job_id}.json",
        {
            "job_id": job_id,
            "job_type": "ingest-textbook",
            "status": "running",
            "created_at": (now - timedelta(minutes=2)).isoformat(),
            "started_at": (now - timedelta(minutes=1)).isoformat(),
            "finished_at": None,
            "error": None,
            "progress": {
                "phase": "extract",
                "items_done": 0,
                "items_total": None,
                "percent": None,
                "updated_at": now.isoformat(),
            },
            "metrics": {"bytes_downloaded": 1024, "pages_extracted": 0, "stages": {}},
            "logs": [],
        },
    )

    server = make_server(tmp_path)
    hydrated = server._jobs[job_id]
    assert hydrated["status"] == "failed"
    assert hydrated["structured_error"]["error_code"] == "server_restarted"
    assert hydrated["progress"]["phase"] == "failed"
    assert hydrated["finished_at"] is not None


def test_prune_removes_expired_terminal_job_files(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    expired_id = "expired-completed-job"
    expired_path = tmp_path / f"{expired_id}.json"
    write_job(
        expired_path,
        {
            "job_id": expired_id,
            "job_type": "ingest-textbook",
            "status": "completed",
            "created_at": (now - timedelta(hours=48)).isoformat(),
            "started_at": (now - timedelta(hours=47)).isoformat(),
            "finished_at": (now - timedelta(hours=46)).isoformat(),
            "result": {"status": "ingested"},
            "progress": {"phase": "completed", "items_done": 10, "items_total": 10},
            "metrics": {"chunks_upserted": 10},
            "logs": [],
        },
    )

    server = make_server(tmp_path, retention_hours=1)
    assert expired_id not in server._jobs
    assert not expired_path.exists()


def test_hydrate_keeps_recent_terminal_job(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    job_id = "recent-completed-job"
    write_job(
        tmp_path / f"{job_id}.json",
        {
            "job_id": job_id,
            "job_type": "ingest-textbook",
            "status": "completed",
            "created_at": (now - timedelta(minutes=10)).isoformat(),
            "started_at": (now - timedelta(minutes=9)).isoformat(),
            "finished_at": (now - timedelta(minutes=8)).isoformat(),
            "result": {"status": "ingested"},
            "progress": {"phase": "completed", "items_done": 3, "items_total": 3},
            "metrics": {"chunks_upserted": 3},
            "logs": [],
        },
    )

    server = make_server(tmp_path, retention_hours=24)
    hydrated = server._jobs[job_id]
    assert hydrated["status"] == "completed"
    assert hydrated["result"]["status"] == "ingested"
