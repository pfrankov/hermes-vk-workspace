"""Admission, discovery and authorization against the pinned, unmodified Hermes.

These tests use the real user plugin directory and host registries. They never
connect a bot or construct an agent/LLM. Build output is excluded from the source
snapshot because Hermes' admission scanner also examines nested Python files.
"""

from __future__ import annotations

import inspect
import os
import re
import shutil
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path
from types import SimpleNamespace

import gateway.config
import hermes_cli.plugins
import hermes_yaml as yaml
import pytest
from agent.secret_scope import (
    reset_multiplex_context,
    reset_secret_scope,
    set_multiplex_context,
    set_secret_scope,
)
from gateway.authz_mixin import GatewayAuthorizationMixin
from gateway.config import Platform, PlatformConfig, load_gateway_config
from gateway.platform_registry import platform_registry
from gateway.platforms.base import BasePlatformAdapter
from gateway.session import build_session_key
from hermes_cli.gateway_setup_wizard import _platform_status
from hermes_cli.plugin_validate import validate_plugin_dir
from hermes_cli.plugin_validate_private_imports import private_import_findings
from hermes_cli.plugins import PluginManager, get_plugin_manager
from pm.workspace import _workspace_member

ROOT = Path(__file__).resolve().parents[1]
HERMES_REVISION = "d923a5eb487a17c0a68801bba9ef0f56c81040a8"
PLATFORM = "vk-workspace"
PLUGIN = "vk-workspace-platform"


def readme_configs() -> list[dict]:
    """Test the actual copy-and-paste examples rather than a second YAML fixture."""
    blocks = re.findall(
        r"^[ \t]*```yaml[ \t]*\n(.*?)^[ \t]*```[ \t]*$",
        (ROOT / "README.md").read_text(),
        re.S | re.M,
    )
    configs = [yaml.safe_load(textwrap.dedent(block)) for block in blocks]
    return [
        config
        for config in configs
        if isinstance(config, dict) and PLATFORM in config.get("platforms", {})
    ]


@pytest.fixture
def installed_plugin(tmp_path, monkeypatch):
    home = tmp_path / "native-hermes-home"
    plugin_dir = home / "plugins" / PLUGIN
    shutil.copytree(
        ROOT,
        plugin_dir,
        ignore=shutil.ignore_patterns(
            ".git",
            ".venv",
            ".ci",
            ".env",
            ".pytest_cache",
            ".ruff_cache",
            "__pycache__",
            "*.egg-info",
            "build",
            "dist",
        ),
    )
    bundled = tmp_path / "empty-bundled-plugins"
    bundled.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("HERMES_SKIP_DOTENV", "1")
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
    monkeypatch.setenv("HERMES_ENABLE_PROJECT_PLUGINS", "0")
    monkeypatch.delenv("HERMES_SAFE_MODE", raising=False)
    for key in tuple(os.environ):
        if key.startswith(("VK_WORKSPACE_", "GATEWAY_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("VK_WORKSPACE_BOT_TOKEN", "host-contract-test-token")

    examples = readme_configs()
    assert examples, "README must contain a complete platforms.vk-workspace example"
    documented = examples[0]
    documented["plugins"] = {"enabled": [PLUGIN]}
    (home / "config.yaml").write_text(yaml.safe_dump(documented))

    manager = get_plugin_manager()
    assert isinstance(manager, PluginManager)
    assert Path(manager.scope_key) == home
    manager.discover_and_load()
    try:
        entry = platform_registry.get(PLATFORM)
        assert entry is not None, manager.list_plugins()
        config = load_gateway_config()
        platform_config = config.platforms[Platform(PLATFORM)]
        adapter = platform_registry.create_adapter(PLATFORM, platform_config)
        assert adapter is not None, "The real Hermes factory rejected the README config"
        yield SimpleNamespace(
            home=home,
            path=plugin_dir,
            manager=manager,
            entry=entry,
            config=config,
            adapter=adapter,
        )
    finally:
        manager.unload()


def test_imports_use_the_exact_supported_hermes_checkout():
    source = Path(
        os.environ.get("HERMES_SOURCE_PATH", str(ROOT.parent / "references" / "hermes-agent"))
    ).resolve()
    assert source.is_dir(), "Set HERMES_SOURCE_PATH to the documented Hermes checkout"
    revision = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()
    assert revision == HERMES_REVISION
    for module in (gateway.config, hermes_cli.plugins):
        assert Path(module.__file__).resolve().is_relative_to(source)


def test_real_admission_probe_and_public_runtime_imports(installed_plugin):
    report = validate_plugin_dir(installed_plugin.path, probe=([sys.executable], dict(os.environ)))
    assert report.ok, report.to_dict()
    assert any("capabilit" in name for name, _, _ in report.checks), report.to_dict()
    assert private_import_findings(installed_plugin.path) == []
    assert installed_plugin.entry.check_fn() is True


def test_user_directory_discovery_registration_and_native_factory(installed_plugin):
    record = next(
        item for item in installed_plugin.manager.list_plugins() if item["name"] == PLUGIN
    )
    assert record["source"] == "user"
    assert record["enabled"] is True
    assert record["error"] is None
    entry = installed_plugin.entry
    assert entry.plugin_name == PLUGIN
    assert entry.allowed_users_env == "VK_WORKSPACE_ALLOWED_USERS"
    assert entry.allow_all_env == "VK_WORKSPACE_ALLOW_ALL_USERS"
    assert entry.trusted_inbound is False
    assert entry.max_message_length == 4096
    assert callable(entry.standalone_sender_fn)
    adapter = installed_plugin.adapter
    assert isinstance(adapter, BasePlatformAdapter)
    assert isinstance(adapter.config, PlatformConfig)
    assert adapter.platform == Platform(PLATFORM)
    assert Path(inspect.getfile(type(adapter))).resolve().is_relative_to(installed_plugin.path)
    assert adapter.settings.base_url == adapter.config.extra["base_url"]
    assert not adapter._running


def test_two_profile_installs_have_distinct_native_dependency_members(installed_plugin, tmp_path):
    """Catch the real PM failure from duplicate buildable project names."""
    secondary_path = tmp_path / "secondary-home" / "plugins" / PLUGIN
    shutil.copytree(installed_plugin.path, secondary_path)
    workspace = tmp_path / "native-pm-workspace"
    workspace.mkdir()
    members = [
        _workspace_member(path, workspace, identity=path)
        for path in (installed_plugin.path, secondary_path)
    ]
    documents = [tomllib.loads((member / "pyproject.toml").read_text()) for member in members]
    assert members[0] != members[1]
    assert documents[0]["project"]["name"] != documents[1]["project"]["name"]
    declared = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    for document in documents:
        assert document["project"]["dependencies"] == declared["dependencies"]
        assert document["project"]["requires-python"] == declared["requires-python"]


def test_readme_yaml_uses_the_native_platform_config_loader(installed_plugin):
    examples = readme_configs()
    assert len(examples) >= 2, "Keep both DM and group setup examples in README"
    for documented in examples:
        fields = documented["platforms"][PLATFORM]
        documented["plugins"] = {"enabled": [PLUGIN]}
        (installed_plugin.home / "config.yaml").write_text(yaml.safe_dump(documented))
        config = load_gateway_config().platforms[Platform(PLATFORM)]
        assert config.enabled is True
        assert config.extra["base_url"] == fields["base_url"]
        for field in ("allowed_users", "group_allow_from", "allowed_chats"):
            if field in fields:
                assert config.extra[field] == fields[field]
        assert installed_plugin.entry.validate_config(config) is True


def authorization_host(installed_plugin, adapter=None):
    """The exact mixin used by GatewayRunner, without its LLM/service startup."""
    host = GatewayAuthorizationMixin()
    host.config = installed_plugin.config
    host.adapters = {Platform(PLATFORM): adapter or installed_plugin.adapter}
    host._primary_profile_name = "default"
    host._profile_adapters = {}
    return host


def test_core_authorizes_yaml_users_and_group_override(installed_plugin):
    adapter = installed_plugin.adapter
    adapter.config.extra["allowed_users"] = ["alice@example.com"]
    adapter.config.extra["group_allow_from"] = ["bob@example.com"]
    host = authorization_host(installed_plugin)

    def allowed(user, chat_type):
        source = adapter.build_source(
            chat_id="123456@chat.agent" if chat_type == "group" else user,
            user_id=user,
            chat_type=chat_type,
        )
        return host._is_user_authorized(source)

    assert allowed("alice@example.com", "dm") is True
    assert allowed("bob@example.com", "dm") is False
    assert allowed("bob@example.com", "group") is True
    assert allowed("alice@example.com", "group") is False
    assert allowed("unknown@example.com", "group") is False
    adapter.config.extra["allow_from"] = ["carol@example.com"]
    assert allowed("carol@example.com", "dm") is True
    assert allowed("alice@example.com", "dm") is False
    assert allowed("bob@example.com", "group") is True


def test_registered_auth_environment_overrides_yaml(installed_plugin, monkeypatch):
    adapter = installed_plugin.adapter
    adapter.config.extra["allowed_users"] = ["alice@example.com"]
    host = authorization_host(installed_plugin)
    alice = adapter.build_source(chat_id="alice@example.com", user_id="alice@example.com")
    bob = adapter.build_source(chat_id="bob@example.com", user_id="bob@example.com")
    monkeypatch.setenv("VK_WORKSPACE_ALLOWED_USERS", "bob@example.com")
    assert host._is_user_authorized(alice) is False
    assert host._is_user_authorized(bob) is True
    monkeypatch.delenv("VK_WORKSPACE_ALLOWED_USERS")
    adapter.config.extra["allowed_users"] = []
    assert host._is_user_authorized(alice) is False
    monkeypatch.setenv("VK_WORKSPACE_ALLOW_ALL_USERS", "true")
    assert host._is_user_authorized(alice) is True


def test_setup_explicit_users_close_existing_allow_all_in_real_host(installed_plugin, monkeypatch):
    adapter = installed_plugin.adapter
    host = authorization_host(installed_plugin)
    alice = adapter.build_source(chat_id="alice@example.com", user_id="alice@example.com")
    outsider = adapter.build_source(chat_id="outsider@example.com", user_id="outsider@example.com")
    monkeypatch.setenv("VK_WORKSPACE_ALLOW_ALL_USERS", "true")
    assert host._is_user_authorized(outsider) is True
    setup = inspect.getmodule(installed_plugin.entry.setup_fn)
    answers = iter(["https://new.example", "new-test-token", "alice@example.com"])
    monkeypatch.setattr(setup, "_ask", lambda *_args, **_kwargs: next(answers))

    installed_plugin.entry.setup_fn()

    assert host._is_user_authorized(alice) is True
    assert host._is_user_authorized(outsider) is False
    assert _platform_status({"_registry_entry": installed_plugin.entry}) == "configured"


@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        ({"enabled": True, "base_url": "https://teams.example"}, "configured"),
        ({"enabled": False, "base_url": "https://teams.example"}, "not configured"),
        ({"enabled": True}, "not configured"),
        ({"enabled": True, "base_url": "http://teams.example"}, "not configured"),
    ],
)
def test_native_setup_status_reads_the_active_profile(installed_plugin, fields, expected):
    (installed_plugin.home / "config.yaml").write_text(
        yaml.safe_dump({"platforms": {PLATFORM: fields}})
    )
    assert _platform_status({"_registry_entry": installed_plugin.entry}) == expected
    # A substantive supplied config remains authoritative, even when the saved profile is valid.
    invalid = PlatformConfig(enabled=True, extra={"base_url": "http://invalid.example"})
    assert installed_plugin.entry.is_connected(invalid) is False


def test_native_setup_status_does_not_borrow_another_profiles_token(
    installed_plugin, monkeypatch, tmp_path
):
    secondary = tmp_path / "secondary-profile"
    secondary.mkdir()
    (secondary / "config.yaml").write_text(
        yaml.safe_dump(
            {"platforms": {PLATFORM: {"enabled": True, "base_url": "https://teams.example"}}}
        )
    )
    monkeypatch.setenv("HERMES_HOME", str(secondary))
    multiplex = set_multiplex_context(True)
    scope = set_secret_scope({})
    try:
        assert _platform_status({"_registry_entry": installed_plugin.entry}) == "not configured"
    finally:
        reset_secret_scope(scope)
        reset_multiplex_context(multiplex)


def test_real_source_profile_ownership_and_session_isolation(installed_plugin):
    primary = installed_plugin.adapter
    primary.config.extra["allowed_users"] = ["alice@example.com"]
    secondary = platform_registry.create_adapter(
        PLATFORM,
        PlatformConfig(
            enabled=True,
            extra={
                "base_url": "https://work.example.com/bot/v1",
                "allowed_users": ["bob@example.com"],
            },
        ),
    )
    assert isinstance(secondary, BasePlatformAdapter)
    secondary.set_owner_profile("work")
    host = authorization_host(installed_plugin)
    host.config.multiplex_profiles = True
    host._profile_adapters = {"work": {Platform(PLATFORM): secondary}}

    alice = primary.build_source(chat_id="same@chat.agent", user_id="alice@example.com")
    bob = secondary.build_source(chat_id="same@chat.agent", user_id="bob@example.com")
    outsider = secondary.build_source(chat_id="same@chat.agent", user_id="alice@example.com")
    assert bob.profile == "work"
    assert host._intake_adapter_for(alice) is primary
    assert host._intake_adapter_for(bob) is secondary
    assert host._is_user_authorized(alice) is True
    assert host._is_user_authorized(bob) is True
    assert host._is_user_authorized(outsider) is False
    assert build_session_key(alice, profile=alice.profile) != build_session_key(
        bob, profile=bob.profile
    )

    group_a = secondary.build_source(
        chat_id="same@chat.agent", chat_type="group", user_id="alice@example.com"
    )
    group_b = secondary.build_source(
        chat_id="same@chat.agent", chat_type="group", user_id="bob@example.com"
    )
    assert build_session_key(group_a) != build_session_key(group_b)
    thread_a = secondary.build_source(
        chat_id="child@chat.agent",
        parent_chat_id="same@chat.agent",
        thread_id="child@chat.agent",
        chat_type="group",
        user_id="alice@example.com",
    )
    thread_b = secondary.build_source(
        chat_id="child@chat.agent",
        parent_chat_id="same@chat.agent",
        thread_id="child@chat.agent",
        chat_type="group",
        user_id="bob@example.com",
    )
    assert build_session_key(thread_a) == build_session_key(thread_b)
    assert build_session_key(thread_a) != build_session_key(group_a)
