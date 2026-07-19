#!/usr/bin/env python3
"""Fail CI when publishable source crosses the public/operator boundary."""

from __future__ import annotations

import ipaddress
import re
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PATH_PARTS = {
    ".whoami",
    "AGENTS.md",
    "BUGS.md",
    "HANDOVER.md",
    "IMPORT.md",
    "internal-audits",
    "private-archives",
    "tickets",
}
FORBIDDEN_PATHS = {
    "assets/n8n",
}
FORBIDDEN_PATH_FRAGMENTS = (
    "credential-export",
    "live-test",
    "live_workspace_report",
    "operator-evidence",
    "runtime-snapshot",
)
TEXT_RULES = {
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "private operator path": re.compile(
        r"(?:"
        r"/(?:home/[A-Za-z0-9._-]+|root)(?:/|\b)"
        r"|"
        r"(?<![A-Za-z0-9._-])[A-Za-z]:[\\/]+Users[\\/]+[A-Za-z0-9._-]+(?:[\\/]|\b)"
        r")",
        re.I,
    ),
}
SKIP_SUFFIXES = {".gif", ".ico", ".jpeg", ".jpg", ".pdf", ".png"}
ABSOLUTE_URL_PATTERN = re.compile(r"https?://[^\s<>'\"`\\]+", re.I)
APPROVED_PUBLIC_URL_HOSTS = {
    "api.openai.com",
    "api.qdrant.tech",
    "drive.google.com",
    "example.com",
    "example.qdrant.io",
    "example.test",
    "files.pythonhosted.org",
    "ghcr.io",
    "github.com",
    "huggingface.co",
    "img.shields.io",
    "json-schema.org",
    "keepachangelog.com",
    "localhost",
    "pdm.fming.dev",
    "pypi.org",
    "python-poetry.org",
    "raw.githubusercontent.com",
    "semver.org",
    "storage.googleapis.com",
    "www.apache.org",
}
APPROVED_SYNTHETIC_URL_SUFFIXES = (
    ".example",
    ".example.com",
    ".example.test",
    ".invalid",
)
IPV4_PATTERN = re.compile(r"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b")
PUBLIC_SAFE_IPV4_NETWORKS = tuple(
    ipaddress.ip_network(value)
    for value in (
        "0.0.0.0/32",
        "127.0.0.0/8",
        "192.0.2.0/24",
        "198.51.100.0/24",
        "203.0.113.0/24",
    )
)


def unapproved_url_hosts(content: str) -> set[str]:
    """Return absolute URL hosts outside the explicit public/synthetic boundary."""

    violations: set[str] = set()
    for match in ABSOLUTE_URL_PATTERN.finditer(content):
        candidate = match.group(0).rstrip("),.;]}")
        hostname = urlsplit(candidate).hostname
        if hostname is None:
            violations.add("<invalid>")
            continue
        normalized = hostname.lower().rstrip(".")
        if normalized in APPROVED_PUBLIC_URL_HOSTS or normalized.endswith(
            APPROVED_SYNTHETIC_URL_SUFFIXES
        ):
            continue
        try:
            ipaddress.ip_address(normalized)
        except ValueError:
            violations.add(normalized)
    return violations


def public_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if result.returncode == 0 and result.stdout:
        return [
            ROOT / item.decode()
            for item in result.stdout.split(b"\0")
            if item and (ROOT / item.decode()).is_file()
        ]
    return [
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and ".git" not in path.parts
        and ".venv" not in path.parts
        and "__pycache__" not in path.parts
    ]


def main() -> None:
    violations: list[str] = []
    files = public_files()
    for path in files:
        relative = path.relative_to(ROOT)
        rendered = relative.as_posix()
        if (
            set(relative.parts) & FORBIDDEN_PATH_PARTS
            or any(
                rendered == value or rendered.startswith(f"{value}/") for value in FORBIDDEN_PATHS
            )
            or any(fragment in rendered.lower() for fragment in FORBIDDEN_PATH_FRAGMENTS)
        ):
            violations.append(f"forbidden public path: {rendered}")
            continue
        if path.suffix.lower() in SKIP_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for label, pattern in TEXT_RULES.items():
            if pattern.search(content):
                violations.append(f"{label} in {rendered}")
        for hostname in unapproved_url_hosts(content):
            violations.append(f"unapproved absolute URL host {hostname!r} in {rendered}")
        if "tests" in relative.parts:
            continue
        for candidate in IPV4_PATTERN.findall(content):
            try:
                address = ipaddress.ip_address(candidate)
            except ValueError:
                continue
            if not any(address in network for network in PUBLIC_SAFE_IPV4_NETWORKS):
                violations.append(f"non-documentation IPv4 address in {rendered}")
    if violations:
        raise SystemExit(
            "Public-source safety gate failed:\n- " + "\n- ".join(sorted(set(violations)))
        )
    print(f"public-source safety gate passed ({len(files)} files)")


if __name__ == "__main__":
    main()
