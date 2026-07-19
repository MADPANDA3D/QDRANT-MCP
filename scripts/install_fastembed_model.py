#!/usr/bin/env python3
"""Install the release image's pinned FastEmbed model without unsafe tar extraction."""

from __future__ import annotations

import argparse
import hashlib
import hmac
import os
import shutil
import ssl
import tarfile
import tempfile
import urllib.error
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

MODEL_ARCHIVE_URL = (
    "https://storage.googleapis.com/qdrant-fastembed/sentence-transformers-all-MiniLM-L6-v2.tar.gz"
)
MODEL_ARCHIVE_SHA256 = "2735afe656e156af64ed603dbb1c96f3cae7f937286a8feb27fff7fa979f6a77"
MODEL_SOURCE_REPOSITORY = "https://huggingface.co/qdrant/all-MiniLM-L6-v2-onnx"
MODEL_SOURCE_COMMIT = "5f1b8cd78bc4fb444dd171e59b18f3a3af89a079"
MODEL_ARCHIVE_PREFIX = "fast-all-MiniLM-L6-v2"
MODEL_ARCHIVE_HOST = "storage.googleapis.com"
MAX_ARCHIVE_BYTES = 96 * 1024 * 1024
MAX_IGNORED_METADATA_BYTES = 4096
DOWNLOAD_CHUNK_BYTES = 1024 * 1024


class ModelInstallError(RuntimeError):
    """Raised when the pinned model cannot be installed exactly as reviewed."""


@dataclass(frozen=True)
class ModelFile:
    """One immutable model file expected inside the reviewed archive."""

    size: int
    sha256: str


MODEL_FILES: Mapping[str, ModelFile] = {
    "config.json": ModelFile(
        size=650,
        sha256="1b4d8e2a3988377ed8b519a31d8d31025a25f1c5f8606998e8014111438efcd7",
    ),
    "model.onnx": ModelFile(
        size=90_387_630,
        sha256="bbd7b466f6d58e646fdc2bd5fd67b2f5e93c0b687011bd4548c420f7bd46f0c5",
    ),
    "special_tokens_map.json": ModelFile(
        size=695,
        sha256="5d5b662e421ea9fac075174bb0688ee0d9431699900b90662acd44b2a350503a",
    ),
    "tokenizer.json": ModelFile(
        size=711_661,
        sha256="da0e79933b9ed51798a3ae27893d3c5fa4a201126cef75586296df9b4d2c62a0",
    ),
    "tokenizer_config.json": ModelFile(
        size=1_433,
        sha256="bd2e06a5b20fd1b13ca988bedc8763d332d242381b4fbc98f8fead4524158f79",
    ),
    "vocab.txt": ModelFile(
        size=231_508,
        sha256="07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
    ),
}

# The reviewed upstream tar contains macOS AppleDouble metadata. These exact members are ignored;
# no metadata member is extracted into the release image.
IGNORED_ARCHIVE_MEMBERS = frozenset(
    {"._fast-all-MiniLM-L6-v2"} | {f"{MODEL_ARCHIVE_PREFIX}/._{name}" for name in MODEL_FILES}
)


def _require_sha256(value: str, *, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ModelInstallError(f"{label} must be a lowercase SHA-256 value.")


def _require_safe_member_name(name: str) -> None:
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
        or "\\" in name
    ):
        raise ModelInstallError("Model archive contains an unsafe member path.")


def _require_flat_filename(name: str) -> None:
    _require_safe_member_name(name)
    if len(PurePosixPath(name).parts) != 1:
        raise ModelInstallError("Model file allowlist contains a non-flat path.")


def _require_pinned_https_url(url: str, *, label: str) -> None:
    parsed = urlsplit(url)
    try:
        port = parsed.port
    except ValueError as exc:
        raise ModelInstallError(f"{label} has an invalid port.") from exc
    if (
        parsed.scheme != "https"
        or parsed.hostname != MODEL_ARCHIVE_HOST
        or parsed.username is not None
        or parsed.password is not None
        or port not in {None, 443}
    ):
        raise ModelInstallError(f"{label} must use the pinned HTTPS origin on port 443.")


class _StrictHTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    def redirect_request(  # type: ignore[override]
        self,
        req: urllib.request.Request,
        fp: object,
        code: int,
        msg: str,
        headers: object,
        newurl: str,
    ) -> urllib.request.Request | None:
        _require_pinned_https_url(newurl, label="Model download redirect")
        return super().redirect_request(req, fp, code, msg, headers, newurl)  # type: ignore[arg-type]


def download_archive(
    url: str,
    output_path: Path,
    *,
    max_bytes: int = MAX_ARCHIVE_BYTES,
) -> None:
    """Download an HTTPS archive with no ambient proxy and a hard byte ceiling."""

    _require_pinned_https_url(url, label="Model archive URL")
    if max_bytes <= 0:
        raise ModelInstallError("Model archive byte limit must be positive.")
    if output_path.exists():
        raise ModelInstallError("Model download destination must not already exist.")

    request = urllib.request.Request(
        url,
        headers={
            "Accept-Encoding": "identity",
            "User-Agent": "mad-mcp-qdrant-model-installer/2.0",
        },
    )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}),
        urllib.request.HTTPSHandler(context=ssl.create_default_context()),
        _StrictHTTPSRedirectHandler(),
    )

    downloaded = 0
    created_output = False
    try:
        with opener.open(request, timeout=60) as response:
            _require_pinned_https_url(response.geturl(), label="Final model download URL")
            if response.status != 200:
                raise ModelInstallError("Model download returned an unexpected HTTP status.")
            content_encoding = response.headers.get("Content-Encoding", "identity").lower()
            if content_encoding != "identity":
                raise ModelInstallError("Model download used an unexpected content encoding.")
            content_length = response.headers.get("Content-Length")
            if content_length is not None:
                try:
                    declared_size = int(content_length)
                except ValueError as exc:
                    raise ModelInstallError(
                        "Model download returned an invalid byte length."
                    ) from exc
                if declared_size < 1 or declared_size > max_bytes:
                    raise ModelInstallError("Model download exceeded its byte limit.")

            output = output_path.open("xb")
            created_output = True
            with output:
                while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                    downloaded += len(chunk)
                    if downloaded > max_bytes:
                        raise ModelInstallError("Model download exceeded its byte limit.")
                    output.write(chunk)
    except ModelInstallError:
        if created_output:
            output_path.unlink(missing_ok=True)
        raise
    except (OSError, urllib.error.URLError) as exc:
        if created_output:
            output_path.unlink(missing_ok=True)
        raise ModelInstallError("Pinned model download failed.") from exc

    if downloaded == 0:
        output_path.unlink(missing_ok=True)
        raise ModelInstallError("Pinned model download was empty.")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(DOWNLOAD_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def extract_verified_model(
    archive_path: Path,
    destination: Path,
    *,
    expected_archive_sha256: str = MODEL_ARCHIVE_SHA256,
    archive_prefix: str = MODEL_ARCHIVE_PREFIX,
    expected_files: Mapping[str, ModelFile] = MODEL_FILES,
    ignored_members: frozenset[str] = IGNORED_ARCHIVE_MEMBERS,
) -> None:
    """Verify and copy only the exact regular model files into a new directory."""

    _require_sha256(expected_archive_sha256, label="Archive digest")
    _require_flat_filename(archive_prefix)
    if not expected_files:
        raise ModelInstallError("Model file allowlist must not be empty.")
    total_expected_bytes = 0
    for filename, spec in expected_files.items():
        _require_flat_filename(filename)
        _require_sha256(spec.sha256, label="Model file digest")
        if spec.size < 0:
            raise ModelInstallError("Model file size must not be negative.")
        total_expected_bytes += spec.size
    if total_expected_bytes > MAX_ARCHIVE_BYTES:
        raise ModelInstallError("Model file allowlist exceeds its byte limit.")
    for ignored_member in ignored_members:
        _require_safe_member_name(ignored_member)

    if not archive_path.is_file():
        raise ModelInstallError("Pinned model archive is missing.")
    if not 0 < archive_path.stat().st_size <= MAX_ARCHIVE_BYTES:
        raise ModelInstallError("Model archive exceeded its byte limit.")
    if not hmac.compare_digest(_file_sha256(archive_path), expected_archive_sha256):
        raise ModelInstallError("Model archive digest did not match the reviewed release input.")
    if destination.exists():
        raise ModelInstallError("Model destination must not already exist.")

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent))
    installed = False
    try:
        expected_members = {
            f"{archive_prefix}/{filename}": (filename, spec)
            for filename, spec in expected_files.items()
        }
        discovered: dict[str, tarfile.TarInfo] = {}
        seen: set[str] = set()

        try:
            with tarfile.open(archive_path, mode="r:gz") as archive:
                for member in archive.getmembers():
                    _require_safe_member_name(member.name)
                    if member.name in seen:
                        raise ModelInstallError("Model archive contains a duplicate member.")
                    seen.add(member.name)

                    if member.name == archive_prefix:
                        if not member.isdir():
                            raise ModelInstallError("Model archive root is not a directory.")
                        continue
                    if member.name in ignored_members:
                        if (
                            not member.isfile()
                            or not 0 <= member.size <= MAX_IGNORED_METADATA_BYTES
                        ):
                            raise ModelInstallError("Model archive metadata member is invalid.")
                        continue
                    expected = expected_members.get(member.name)
                    if expected is None:
                        raise ModelInstallError("Model archive contains an unexpected member.")
                    filename, spec = expected
                    if not member.isfile() or member.size != spec.size:
                        raise ModelInstallError(
                            "Model archive file metadata did not match the pin."
                        )
                    discovered[filename] = member

                if set(discovered) != set(expected_files):
                    raise ModelInstallError("Model archive is missing a required file.")

                for filename, spec in expected_files.items():
                    source = archive.extractfile(discovered[filename])
                    if source is None:
                        raise ModelInstallError("Model archive file could not be read.")
                    output_path = staging / filename
                    digest = hashlib.sha256()
                    written = 0
                    with source, output_path.open("xb") as output:
                        while chunk := source.read(DOWNLOAD_CHUNK_BYTES):
                            written += len(chunk)
                            if written > spec.size:
                                raise ModelInstallError(
                                    "Model archive file exceeded its pinned size."
                                )
                            digest.update(chunk)
                            output.write(chunk)
                    if written != spec.size or not hmac.compare_digest(
                        digest.hexdigest(), spec.sha256
                    ):
                        raise ModelInstallError("Model file digest did not match the reviewed pin.")
                    output_path.chmod(0o444)
        except (tarfile.TarError, EOFError, OSError) as exc:
            raise ModelInstallError("Pinned model archive could not be validated.") from exc

        staging.chmod(0o555)
        if destination.exists():
            raise ModelInstallError("Model destination appeared during installation.")
        os.replace(staging, destination)
        installed = True
    finally:
        if not installed:
            shutil.rmtree(staging, ignore_errors=True)


def install_model(destination: Path) -> None:
    """Download, verify, and install the release image model."""

    with tempfile.TemporaryDirectory(prefix="qdrant-fastembed-download-") as temporary:
        archive_path = Path(temporary) / "model.tar.gz"
        download_archive(MODEL_ARCHIVE_URL, archive_path)
        extract_verified_model(archive_path, destination)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    destination = args.destination.expanduser().resolve(strict=False)
    if destination == Path(destination.anchor):
        raise SystemExit("Refusing to install a model at a filesystem root.")
    try:
        install_model(destination)
    except ModelInstallError as exc:
        raise SystemExit(f"FastEmbed model install failed: {exc}") from None
    print(f"Installed pinned FastEmbed model from commit {MODEL_SOURCE_COMMIT} at {destination}")


if __name__ == "__main__":
    main()
