from __future__ import annotations

import hashlib
import io
import stat
import tarfile
from pathlib import Path

import pytest

from scripts.install_fastembed_model import (
    MODEL_ARCHIVE_SHA256,
    MODEL_ARCHIVE_URL,
    MODEL_FILES,
    MODEL_SOURCE_COMMIT,
    ModelFile,
    ModelInstallError,
    download_archive,
    extract_verified_model,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_archive(
    path: Path,
    *,
    prefix: str,
    files: dict[str, bytes],
    extra_files: dict[str, bytes] | None = None,
    symlink_name: str | None = None,
) -> str:
    with tarfile.open(path, mode="w:gz") as archive:
        root = tarfile.TarInfo(prefix)
        root.type = tarfile.DIRTYPE
        root.mode = 0o755
        archive.addfile(root)

        for filename, content in files.items():
            member_name = f"{prefix}/{filename}"
            member = tarfile.TarInfo(member_name)
            if member_name == symlink_name:
                member.type = tarfile.SYMTYPE
                member.linkname = "/tmp/not-a-model"
                member.size = 0
                archive.addfile(member)
                continue
            member.size = len(content)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(content))

        for name, content in (extra_files or {}).items():
            member = tarfile.TarInfo(name)
            member.size = len(content)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(content))

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _specs(files: dict[str, bytes]) -> dict[str, ModelFile]:
    return {
        filename: ModelFile(size=len(content), sha256=_sha256(content))
        for filename, content in files.items()
    }


def test_release_model_constants_are_exactly_pinned() -> None:
    assert MODEL_ARCHIVE_URL == (
        "https://storage.googleapis.com/qdrant-fastembed/"
        "sentence-transformers-all-MiniLM-L6-v2.tar.gz"
    )
    assert MODEL_ARCHIVE_SHA256 == (
        "2735afe656e156af64ed603dbb1c96f3cae7f937286a8feb27fff7fa979f6a77"
    )
    assert MODEL_SOURCE_COMMIT == "5f1b8cd78bc4fb444dd171e59b18f3a3af89a079"
    assert {name: spec.sha256 for name, spec in MODEL_FILES.items()} == {
        "config.json": "1b4d8e2a3988377ed8b519a31d8d31025a25f1c5f8606998e8014111438efcd7",
        "model.onnx": "bbd7b466f6d58e646fdc2bd5fd67b2f5e93c0b687011bd4548c420f7bd46f0c5",
        "special_tokens_map.json": (
            "5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a"
        ),
        "tokenizer.json": ("da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0"),
        "tokenizer_config.json": (
            "bd2e06a5b20fd1b13ca988bedc8763d332d242381b4fbc98f8fead4524158f79"
        ),
        "vocab.txt": "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    }


def test_extract_verified_model_copies_only_expected_regular_files(tmp_path: Path) -> None:
    files = {"config.json": b"config", "model.onnx": b"model-bytes"}
    archive = tmp_path / "model.tar.gz"
    archive_sha256 = _write_archive(
        archive,
        prefix="model",
        files=files,
        extra_files={"._model": b"reviewed metadata"},
    )
    destination = tmp_path / "installed"

    extract_verified_model(
        archive,
        destination,
        expected_archive_sha256=archive_sha256,
        archive_prefix="model",
        expected_files=_specs(files),
        ignored_members=frozenset({"._model"}),
    )

    assert {path.name for path in destination.iterdir()} == set(files)
    assert {path.name: path.read_bytes() for path in destination.iterdir()} == files
    assert stat.S_IMODE(destination.stat().st_mode) == 0o555
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o444 for path in destination.iterdir())


@pytest.mark.parametrize(
    ("extra_name", "message"),
    [
        ("model/unexpected.json", "unexpected member"),
        ("../escape", "unsafe member path"),
    ],
)
def test_extract_verified_model_rejects_unreviewed_members(
    tmp_path: Path,
    extra_name: str,
    message: str,
) -> None:
    files = {"config.json": b"config"}
    archive = tmp_path / "model.tar.gz"
    archive_sha256 = _write_archive(
        archive,
        prefix="model",
        files=files,
        extra_files={extra_name: b"unexpected"},
    )
    destination = tmp_path / "installed"

    with pytest.raises(ModelInstallError, match=message):
        extract_verified_model(
            archive,
            destination,
            expected_archive_sha256=archive_sha256,
            archive_prefix="model",
            expected_files=_specs(files),
            ignored_members=frozenset(),
        )

    assert not destination.exists()


def test_extract_verified_model_rejects_link_for_expected_file(tmp_path: Path) -> None:
    files = {"config.json": b"config"}
    archive = tmp_path / "model.tar.gz"
    archive_sha256 = _write_archive(
        archive,
        prefix="model",
        files=files,
        symlink_name="model/config.json",
    )

    with pytest.raises(ModelInstallError, match="file metadata"):
        extract_verified_model(
            archive,
            tmp_path / "installed",
            expected_archive_sha256=archive_sha256,
            archive_prefix="model",
            expected_files=_specs(files),
            ignored_members=frozenset(),
        )


def test_extract_verified_model_rejects_an_unsafe_allowlist_filename(tmp_path: Path) -> None:
    files = {"config.json": b"config"}
    archive = tmp_path / "model.tar.gz"
    archive_sha256 = _write_archive(archive, prefix="model", files=files)

    with pytest.raises(ModelInstallError, match="non-flat path"):
        extract_verified_model(
            archive,
            tmp_path / "installed",
            expected_archive_sha256=archive_sha256,
            archive_prefix="model",
            expected_files={"nested/config.json": _specs(files)["config.json"]},
            ignored_members=frozenset(),
        )


def test_extract_verified_model_rejects_archive_and_file_digest_mismatch(tmp_path: Path) -> None:
    files = {"config.json": b"config"}
    archive = tmp_path / "model.tar.gz"
    archive_sha256 = _write_archive(archive, prefix="model", files=files)

    with pytest.raises(ModelInstallError, match="archive digest"):
        extract_verified_model(
            archive,
            tmp_path / "bad-archive",
            expected_archive_sha256="0" * 64,
            archive_prefix="model",
            expected_files=_specs(files),
            ignored_members=frozenset(),
        )

    bad_specs = {"config.json": ModelFile(size=6, sha256="0" * 64)}
    with pytest.raises(ModelInstallError, match="file digest"):
        extract_verified_model(
            archive,
            tmp_path / "bad-file",
            expected_archive_sha256=archive_sha256,
            archive_prefix="model",
            expected_files=bad_specs,
            ignored_members=frozenset(),
        )


def test_download_archive_rejects_non_pinned_origin_without_network(tmp_path: Path) -> None:
    rejected_urls = (
        "http://storage.googleapis.com/model.tar.gz",
        "https://example.com/model.tar.gz",
        "https://user@storage.googleapis.com/model.tar.gz",
        "https://storage.googleapis.com:444/model.tar.gz",
    )
    for url in rejected_urls:
        with pytest.raises(ModelInstallError, match="pinned HTTPS origin"):
            download_archive(url, tmp_path / "archive")


def test_download_archive_never_deletes_an_existing_destination(tmp_path: Path) -> None:
    destination = tmp_path / "archive"
    destination.write_bytes(b"operator-owned")

    with pytest.raises(ModelInstallError, match="must not already exist"):
        download_archive(MODEL_ARCHIVE_URL, destination)

    assert destination.read_bytes() == b"operator-owned"
