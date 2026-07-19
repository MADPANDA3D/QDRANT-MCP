from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType


def load_source_safety() -> ModuleType:
    script = Path(__file__).resolve().parents[1] / "scripts" / "check_source_safety.py"
    spec = importlib.util.spec_from_file_location("check_source_safety", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_operator_path_rule_covers_unix_and_windows_user_profiles() -> None:
    source_safety = load_source_safety()
    pattern = source_safety.TEXT_RULES["private operator path"]
    windows_path = "C:" + "\\" + "Users" + "\\" + "sample-user" + "\\" + "workspace"
    unix_home_path = "/" + "home" + "/sample-user/workspace"
    unix_root_path = "/" + "root" + "/workspace"

    assert pattern.search(unix_home_path)
    assert pattern.search(unix_root_path)
    assert pattern.search(windows_path)
    assert not pattern.search("relative/sample-user/workspace")


def test_absolute_url_hosts_are_allowlisted_by_public_or_synthetic_identity() -> None:
    source_safety = load_source_safety()
    unknown_url = "https" + "://" + "private-service" + ".corp.local/resource"

    assert source_safety.unapproved_url_hosts("https://github.com/example/project") == set()
    assert source_safety.unapproved_url_hosts("https://ghcr.io/token") == set()
    assert source_safety.unapproved_url_hosts("https://files.example.test/resource") == set()
    assert source_safety.unapproved_url_hosts(unknown_url) == {"private-service.corp.local"}
