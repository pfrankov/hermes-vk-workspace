"""Native setup uses Hermes' real config merge and profile credential writers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from gateway.config import Platform
from hermes_cli.config import (
    get_env_path,
    get_env_value,
    load_env,
    read_raw_config,
    save_config,
    save_env_value,
)
from hermes_vk_workspace.config import PLATFORM, TOKEN_ENV
from hermes_vk_workspace.setup import setup_platform


def _prompts(monkeypatch, *answers):
    answers = iter(answers)
    monkeypatch.setattr("hermes_vk_workspace.setup._ask", lambda *_args, **_kwargs: next(answers))
    messages = []
    monkeypatch.setattr("hermes_vk_workspace.setup._warn", lambda value: messages.append(value))
    return messages


def test_wizard_saves_url_token_and_allowlist_without_losing_other_config(adapter, monkeypatch):
    save_config(
        {
            "agent": {"max_turns": 12},
            "platforms": {
                PLATFORM: {"require_mention": True, "allowed_chats": ["team@chat.agent"]}
            },
        }
    )
    warnings = _prompts(
        monkeypatch,
        "https://teams.example.com/proxy",
        "new-secret-token",
        " alice@example, bob@example,alice@example ",
    )
    setup_platform()
    saved = read_raw_config()
    assert saved["agent"]["max_turns"] == 12
    assert saved["platforms"][PLATFORM]["allowed_chats"] == ["team@chat.agent"]
    assert saved["platforms"][PLATFORM]["require_mention"] is True
    assert saved["platforms"][PLATFORM]["base_url"] == "https://teams.example.com/proxy/bot/v1"
    assert saved["platforms"][PLATFORM]["enabled"] is True
    assert get_env_value(TOKEN_ENV) == "new-secret-token"
    assert load_env()["VK_WORKSPACE_ALLOWED_USERS"] == "alice@example,bob@example"
    assert not warnings


@pytest.mark.parametrize(
    ("server", "token", "users"),
    [
        ("http://insecure.example.com", "token", "alice@example"),
        ("https://secret@example.com", "token", "alice@example"),
        ("https://example.com", "", "alice@example"),
        ("https://example.com", "token", "*"),
        ("https://example.com", "token", "id with spaces"),
        # Hermes strips non-ASCII from .env values instead of failing.
        ("https://example.com", "tok\u0415n", "alice@example"),
        ("https://example.com", "token", "\u0438\u0432\u0430\u043d@example"),
    ],
)
def test_wizard_invalid_inputs_do_not_write_secrets_or_yaml(
    adapter, monkeypatch, server, token, users
):
    # Invalid URL returns before the remaining prompts.
    _prompts(monkeypatch, server, token, users)
    setup_platform()
    assert get_env_value(TOKEN_ENV) is None
    assert PLATFORM not in read_raw_config().get("platforms", {})


def test_wizard_keeps_token_and_access_when_prompts_are_blank(adapter, monkeypatch):
    save_env_value(TOKEN_ENV, "existing-secret")
    save_env_value("VK_WORKSPACE_ALLOWED_USERS", "existing@example")
    save_env_value("VK_WORKSPACE_ALLOW_ALL_USERS", "true")
    _prompts(monkeypatch, "https://teams.example.com", "", "")
    setup_platform()
    assert get_env_value(TOKEN_ENV) == "existing-secret"
    assert get_env_value("VK_WORKSPACE_ALLOWED_USERS") == "existing@example"
    assert get_env_value("VK_WORKSPACE_ALLOW_ALL_USERS") == "true"


@pytest.mark.parametrize(
    "managed_key",
    [TOKEN_ENV, "VK_WORKSPACE_ALLOWED_USERS", "VK_WORKSPACE_ALLOW_ALL_USERS"],
)
def test_wizard_preflights_managed_credentials_before_any_profile_write(
    adapter, monkeypatch, tmp_path, managed_key
):
    save_config({"platforms": {PLATFORM: {"enabled": False, "base_url": "https://old.example"}}})
    save_env_value(TOKEN_ENV, "old-test-token")
    save_env_value("VK_WORKSPACE_ALLOWED_USERS", "old@example")
    save_env_value("VK_WORKSPACE_ALLOW_ALL_USERS", "true")
    before_config = read_raw_config()
    before_env = get_env_path().read_bytes()
    managed = tmp_path / "managed"
    managed.mkdir()
    (managed / ".env").write_text(f"{managed_key}=managed-test-value\n")
    monkeypatch.setenv("HERMES_MANAGED_DIR", str(managed))
    warnings = _prompts(monkeypatch, "https://new.example", "new-test-token", "new@example")

    setup_platform()

    assert read_raw_config() == before_config
    assert get_env_path().read_bytes() == before_env
    assert any(managed_key in warning for warning in warnings)


@pytest.mark.parametrize("duplicate_top_level_url", [False, True])
def test_wizard_updates_nested_url_without_leaving_a_conflicting_copy(
    adapter, monkeypatch, duplicate_top_level_url
):
    platform = {
        "enabled": False,
        "extra": {"base_url": "https://old.example/bot/v1", "require_mention": False},
    }
    if duplicate_top_level_url:
        platform["base_url"] = "https://shadowed.example/bot/v1"
    save_config({"agent": {"max_turns": 12}, "platforms": {PLATFORM: platform}})
    warnings = _prompts(monkeypatch, "https://new.example/proxy", "new-test-token", "")

    setup_platform()

    saved = read_raw_config()
    assert saved["agent"]["max_turns"] == 12
    assert saved["platforms"][PLATFORM] == {
        "enabled": True,
        "extra": {"base_url": "https://new.example/proxy/bot/v1", "require_mention": False},
    }
    assert get_env_value(TOKEN_ENV) == "new-test-token"
    assert not warnings


@pytest.mark.parametrize(
    ("overridden_url", "enabled"),
    [("https://managed.example/bot/v1", True), ("https://desired.example/bot/v1", False)],
)
def test_wizard_refuses_token_when_effective_managed_config_disagrees(
    adapter, monkeypatch, overridden_url, enabled
):
    import gateway.config as host_config

    def fake_effective():
        return SimpleNamespace(
            platforms={
                Platform(PLATFORM): SimpleNamespace(
                    extra={"base_url": overridden_url}, enabled=enabled
                )
            }
        )

    monkeypatch.setattr(host_config, "load_gateway_config", fake_effective)
    warnings = _prompts(monkeypatch, "https://desired.example", "never-save-token", "")
    setup_platform()
    assert get_env_value(TOKEN_ENV) is None
    assert any("managed or overridden" in warning for warning in warnings)


def test_wizard_is_registered_as_native_hermes_setup_hook(adapter):
    from gateway.platform_registry import platform_registry

    assert platform_registry.get(PLATFORM).setup_fn is setup_platform
