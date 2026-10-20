"""Configuration boundaries exercised against the real Hermes profile and registry APIs."""

from pathlib import Path
from types import SimpleNamespace

import hermes_vk_workspace as plugin
import pytest
from agent.secret_scope import (
    reset_multiplex_context,
    reset_secret_scope,
    set_multiplex_context,
    set_secret_scope,
)
from gateway.config import PlatformConfig
from gateway.platform_registry import PlatformEntry
from hermes_cli.plugins_manifest import parse_manifest_file, requires_hermes_error
from hermes_vk_workspace.config import (
    ConfigError,
    Settings,
    identifier,
    parse_target_ref,
    validate_target_ref,
)


def platform(**extra):
    return PlatformConfig(
        token="test-token", extra={"base_url": "https://teams.example.com", **extra}
    )


def test_defaults_are_https_profile_local_and_private_by_default():
    settings = Settings.from_platform(platform())
    assert settings.base_url == "https://teams.example.com/bot/v1"
    assert settings.media_allowed_origins == ("https://teams.example.com",)
    assert settings.poll_time == 30
    assert settings.request_timeout == 30.0
    assert settings.media_max_bytes == 20 * 1024 * 1024
    assert settings.require_mention is True
    assert settings.allowed_chats == ()
    assert "test-token" not in repr(settings)


def test_top_level_hermes_platform_keys_are_promoted_and_auth_remains_host_owned():
    config = PlatformConfig.from_dict(
        {
            "enabled": True,
            "token": "test-token",
            "base_url": "https://teams.example.com",
            "allowed_users": ["alice@example.com"],
            "allowed_chats": [123, "team@chat.agent"],
            "require_mention": False,
            "extra": {"poll_time": 60},
        }
    )
    settings = Settings.from_platform(config)
    assert settings.allowed_chats == ("123", "team@chat.agent")
    assert settings.require_mention is False
    assert settings.poll_time == 60
    assert config.extra["allowed_users"] == ["alice@example.com"]


def test_scoped_secret_wins_over_native_token_and_never_borrows_another_profile(monkeypatch):
    monkeypatch.setenv("VK_WORKSPACE_BOT_TOKEN", "default-profile-token")
    multiplex = set_multiplex_context(True)
    scope = set_secret_scope({"VK_WORKSPACE_BOT_TOKEN": "secondary-profile-token"})
    try:
        assert Settings.from_platform(platform()).token == "secondary-profile-token"
    finally:
        reset_secret_scope(scope)
        reset_multiplex_context(multiplex)

    multiplex = set_multiplex_context(True)
    scope = set_secret_scope({})
    try:
        config = platform()
        config.token = None
        with pytest.raises(ConfigError, match="active Hermes profile"):
            Settings.from_platform(config)
        config.token = "host-injected-secondary-token"
        assert Settings.from_platform(config).token == "host-injected-secondary-token"
    finally:
        reset_secret_scope(scope)
        reset_multiplex_context(multiplex)


def test_environment_token_is_used_and_blank_environment_allows_native_token(monkeypatch):
    monkeypatch.setenv("VK_WORKSPACE_BOT_TOKEN", "environment-token")
    assert Settings.from_platform(platform()).token == "environment-token"
    monkeypatch.setenv("VK_WORKSPACE_BOT_TOKEN", "")
    assert Settings.from_platform(platform()).token == "test-token"


@pytest.mark.parametrize("token", [None, "", " ", "two words", "token\nvalue", False, 123])
def test_missing_or_malformed_tokens_fail_without_echoing_value(token):
    config = platform()
    config.token = token
    with pytest.raises(ConfigError, match="Set VK_WORKSPACE_BOT_TOKEN") as caught:
        Settings.from_platform(config)
    if isinstance(token, str) and token.strip():
        assert token not in str(caught.value)


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://TEAMS.example.com:443/", "https://teams.example.com/bot/v1"),
        ("https://teams.example.com/bot/v1/", "https://teams.example.com/bot/v1"),
        ("https://teams.example.com/prefix", "https://teams.example.com/prefix/bot/v1"),
        ("https://teams.example.com/prefix/bot/v1", "https://teams.example.com/prefix/bot/v1"),
        ("https://[::1]:8443/bot/v1", "https://[::1]:8443/bot/v1"),
        ("https://пример.рф", "https://xn--e1afmkfd.xn--p1ai/bot/v1"),
        ("https://faß.example", "https://xn--fa-hia.example/bot/v1"),
    ],
)
def test_base_url_preserves_reverse_proxy_prefix_and_normalizes_origin(url, expected):
    assert Settings.from_platform(platform(base_url=url)).base_url == expected


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "teams.example.com",
        "http://teams.example.com",
        "ftp://teams.example.com",
        "//teams.example.com",
        "https://",
        " https://teams.example.com",
        "https://teams.example.com\n",
        "https://secret@teams.example.com",
        "https://user:secret@teams.example.com",
        "https://teams.example.com?token=secret",
        "https://teams.example.com?",
        "https://teams.example.com#secret",
        "https://teams.example.com#",
        "https://teams.example.com:70000",
        "https://teams.example.com:0",
        "https://teams.example.com:invalid",
        "https://[invalid]/bot/v1",
        "https://teams.example.com\\@evil.example.com",
        "https://teams.example.com/../bot/v1",
        "https://teams.example.com/%2e%2e/bot/v1",
        "https://teams.example.com/%252e%252e/bot/v1",
        "https://teams.example.com/prefix%2fbot/v1",
        "https://teams.example.com/a//bot/v1",
        "https://teams.example.com\u200b",
    ],
)
def test_unsafe_server_urls_fail_without_exposing_sensitive_url_parts(url):
    with pytest.raises(ConfigError) as caught:
        Settings.from_platform(platform(base_url=url))
    assert "secret" not in str(caught.value)


def test_http_requires_explicit_boolean_opt_in():
    settings = Settings.from_platform(
        platform(
            base_url="http://127.0.0.1:8765",
            allow_insecure_http=True,
            media_allowed_origins=["http://localhost:9876"],
        )
    )
    assert settings.media_allowed_origins == ("http://127.0.0.1:8765", "http://localhost:9876")


def test_media_origins_are_exact_canonical_origins_including_api_origin():
    settings = Settings.from_platform(
        platform(
            media_allowed_origins=[
                "https://CDN.example.com:443/",
                "https://teams.example.com",
                "https://cdn.example.com",
                "https://cdn.example.com:8443",
            ]
        )
    )
    assert settings.media_allowed_origins == (
        "https://teams.example.com",
        "https://cdn.example.com",
        "https://cdn.example.com:8443",
    )


@pytest.mark.parametrize(
    "origin",
    [
        "https://cdn.example.com/files",
        "https://*.example.com",
        "https://secret@cdn.example.com",
        "https://cdn.example.com?token=secret",
        "https://cdn.example.com#anchor",
        "http://cdn.example.com",
        "cdn.example.com",
        None,
    ],
)
def test_media_origin_allowlist_does_not_accept_urls_or_wildcards(origin):
    with pytest.raises(ConfigError):
        Settings.from_platform(platform(media_allowed_origins=[origin]))


@pytest.mark.parametrize(
    "extra",
    [
        {"allow_insecure_http": "true"},
        {"require_mention": "false"},
        {"require_mention": 0},
        {"allow_all_users": False},
        {"allow_all_users": True},
        {"poll_time": 0},
        {"poll_time": 61},
        {"poll_time": 30.0},
        {"poll_time": "30"},
        {"poll_time": True},
        {"request_timeout": 0},
        {"request_timeout": 301},
        {"request_timeout": "30"},
        {"request_timeout": False},
        {"request_timeout": float("inf")},
        {"request_timeout": float("nan")},
        {"request_timeout": 10**1000},
        {"media_max_bytes": 0},
        {"media_max_bytes": 100 * 1024 * 1024 + 1},
        {"media_max_mb": 0},
        {"media_max_mb": 101},
        {"media_max_mb": 1.5},
        {"media_max_mb": True},
        {"media_max_mb": 20, "media_max_bytes": 10},
        {"media_allowed_origins": "https://cdn.example.com"},
        {"allowed_chats": "chat@chat.agent"},
        {"allowed_chats": [True]},
        {"allowed_chats": ["chat with spaces"]},
    ],
)
def test_invalid_values_fail_instead_of_being_coerced_or_silently_ignored(extra):
    with pytest.raises(ConfigError):
        Settings.from_platform(platform(**extra))


def test_media_byte_limit_and_mebibyte_convenience_have_exact_values():
    assert Settings.from_platform(platform(media_max_bytes=1234)).media_max_bytes == 1234
    assert Settings.from_platform(platform(media_max_mb=5)).media_max_bytes == 5 * 1024 * 1024
    assert Settings.from_platform(platform(request_timeout=1.5)).request_timeout == 1.5


@pytest.mark.parametrize("value", ["room@chat.agent", "alice@example.com", "123", 123, "native:id"])
def test_native_identifiers_remain_opaque(value):
    assert identifier(value) == str(value)
    assert validate_target_ref(str(value)) is True


@pytest.mark.parametrize(
    "value",
    [
        True,
        None,
        [],
        1.5,
        "",
        "a b",
        "a\nb",
        "a\x00b",
        "a\u200bb",
        "a" * 257,
        pytest.param(10**5000, id="oversized-integer"),
    ],
)
def test_identifiers_reject_ambiguous_or_unbounded_values(value):
    with pytest.raises(ConfigError):
        identifier(value)


@pytest.mark.parametrize("value", ["team@chat.agent", "alice@example.com", "1234567890"])
def test_native_target_parser_preempts_named_chat_directory(value):
    assert parse_target_ref(value) == (value, None)


@pytest.mark.parametrize(
    "value",
    ["Engineering", "a b", "", "x@@example.com", "@example.com", "x@", "x@y/thread", "x@y:thread"],
)
def test_target_parser_leaves_directory_names_and_non_native_syntax_to_host(value):
    assert parse_target_ref(value) is None


def test_registered_contract_is_accepted_by_current_hermes_and_supports_cli_and_cron(monkeypatch):
    entries = []
    context = SimpleNamespace(
        register_platform=lambda **kwargs: entries.append(PlatformEntry(**kwargs))
    )
    plugin.register(context)
    (entry,) = entries
    assert entry.name == "vk-workspace"
    assert entry.trusted_inbound is False
    assert entry.required_env == ["VK_WORKSPACE_BOT_TOKEN"]
    assert entry.allowed_users_env == "VK_WORKSPACE_ALLOWED_USERS"
    assert entry.allow_all_env == "VK_WORKSPACE_ALLOW_ALL_USERS"
    assert entry.cron_deliver_env_var == "VK_WORKSPACE_HOME_CHANNEL"
    assert entry.max_message_length == 4096
    assert entry.check_fn() is True
    assert entry.is_connected(platform()) is True
    assert entry.validate_config(platform(base_url="http://teams.example.com")) is False
    assert entry.parse_target_ref_fn("team@chat.agent") == ("team@chat.agent", None)
    assert entry.validate_target_ref_fn("bad value") is not True
    assert entry.standalone_sender_fn is plugin.standalone_send
    monkeypatch.setenv("VK_WORKSPACE_HOME_CHANNEL", "team@chat.agent")
    monkeypatch.setenv("VK_WORKSPACE_HOME_CHANNEL_NAME", "Engineering")
    assert entry.env_enablement_fn() == {
        "home_channel": {"chat_id": "team@chat.agent", "name": "Engineering"}
    }


def test_manifest_is_valid_for_current_hermes_and_registers_no_model_tools():
    root = Path(__file__).resolve().parents[1]
    manifest = parse_manifest_file(root / "plugin.yaml", root, "user", "")
    assert manifest is not None
    assert manifest.name == "vk-workspace-platform"
    assert manifest.kind == "platform"
    assert manifest.requires_hermes == ">=0.22.2"
    assert requires_hermes_error(manifest) is None
    assert manifest.provides_tools == []
