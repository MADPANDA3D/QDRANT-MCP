#!/usr/bin/env python3
"""Verify that wheel and sdist contain only the intended public Python package."""

from __future__ import annotations

import stat
import tarfile
import tomllib
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / "dist"
PACKAGE_ROOT = ROOT / "src" / "mcp_server_qdrant"
SDIST_ROOT_FILES = {
    ".gitignore",
    "LICENSE",
    "NOTICE",
    "PKG-INFO",
    "README.md",
    "pyproject.toml",
}


def project_version() -> str:
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    version = project.get("version")
    if version != "2.0.0":
        raise AssertionError(f"expected static project version 2.0.0, found {version!r}")
    return version


def package_files() -> set[str]:
    files = {
        path.relative_to(ROOT / "src").as_posix()
        for path in PACKAGE_ROOT.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    if not files or "mcp_server_qdrant/__init__.py" not in files:
        raise AssertionError("source package allowlist is unexpectedly empty")
    return files


def require_safe_path(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or not path.parts or any(part in {"", ".", ".."} for part in path.parts):
        raise AssertionError(f"unsafe archive path: {name}")
    return path


def check_wheel(path: Path, version: str, sources: set[str]) -> None:
    dist_info = f"mad_mcp_qdrant-{version}.dist-info"
    expected = set(sources)
    expected.update(
        {
            f"{dist_info}/METADATA",
            f"{dist_info}/RECORD",
            f"{dist_info}/WHEEL",
            f"{dist_info}/entry_points.txt",
            f"{dist_info}/licenses/LICENSE",
            f"{dist_info}/licenses/NOTICE",
        }
    )
    with zipfile.ZipFile(path) as archive:
        files = set()
        for member in archive.infolist():
            require_safe_path(member.filename)
            if member.is_dir():
                continue
            mode = member.external_attr >> 16
            if mode and stat.S_ISLNK(mode):
                raise AssertionError(f"wheel contains symlink: {member.filename}")
            files.add(member.filename)
    if files != expected:
        raise AssertionError(
            f"wheel allowlist mismatch: missing={sorted(expected - files)} "
            f"unexpected={sorted(files - expected)}"
        )


def check_sdist(path: Path, version: str, sources: set[str]) -> None:
    prefix = f"mad_mcp_qdrant-{version}"
    expected = {f"{prefix}/{name}" for name in SDIST_ROOT_FILES}
    expected.update(f"{prefix}/src/{name}" for name in sources)
    with tarfile.open(path, mode="r:gz") as archive:
        files = set()
        for member in archive.getmembers():
            require_safe_path(member.name)
            if member.issym() or member.islnk():
                raise AssertionError(f"sdist contains link: {member.name}")
            if member.isfile():
                files.add(member.name)
    if files != expected:
        raise AssertionError(
            f"sdist allowlist mismatch: missing={sorted(expected - files)} "
            f"unexpected={sorted(files - expected)}"
        )


def main() -> None:
    version = project_version()
    sources = package_files()
    wheels = sorted(DIST.glob("*.whl"))
    sdists = sorted(DIST.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit("Expected exactly one wheel and one .tar.gz source distribution.")
    check_wheel(wheels[0], version, sources)
    check_sdist(sdists[0], version, sources)
    print(f"package archive allowlist passed for mad-mcp-qdrant {version}")


if __name__ == "__main__":
    main()
