"""VK Workspace setup via Hermes' native platform setup_fn; no network or secret logging."""

from __future__ import annotations

from collections.abc import Mapping
from getpass import getpass

from hermes_cli.config import get_env_value, read_raw_config, save_config, save_env_value

from .config import (
    PLATFORM,
    TOKEN_ENV,
    ConfigError,
    _base_url,
    _has_unsafe_characters,
    identifier,
)

USERS_ENV = "VK_WORKSPACE_ALLOWED_USERS"


def _ask(message: str, *, default: str = "", password: bool = False) -> str:
    """Use the terminal already owned by the official Hermes setup wizard."""
    if password:
        return getpass(f"{message}: ")
    suffix = f" [{default}]" if default else ""
    return input(f"{message}{suffix}: ") or default


def _warn(message: str) -> None:
    print(message)


def _effective_platform():
    """Consult the same merged YAML, env and managed overlay as the running gateway."""
    from gateway.config import Platform, load_gateway_config

    return load_gateway_config().platforms.get(Platform(PLATFORM))


def _extra(platform) -> Mapping:
    value = platform.extra if platform is not None else None
    return value if isinstance(value, Mapping) else {}


def _users(value: str) -> str | None:
    """None means preserve the existing authorization policy, never open access."""
    if not value.strip():
        return None
    entries = tuple(dict.fromkeys(identifier(v.strip()) for v in value.split(",") if v.strip()))
    if not entries or len(entries) > 256 or "*" in entries:
        raise ConfigError("Enter up to 256 explicit user IDs; open access is not configured here")
    return ",".join(entries)


def _collect() -> tuple[str, str, str | None] | None:
    try:
        existing = _effective_platform()
    except Exception:
        existing = None
    extra = _extra(existing)
    default_url = extra.get("base_url", "")
    raw_url = _ask("  VK Teams Bot API server URL", default=default_url or "")
    if not isinstance(raw_url, str) or not raw_url.strip():
        _warn("  VK Workspace setup cancelled: a Bot API URL is required.")
        return None
    try:
        url = _base_url(raw_url.strip(), extra.get("allow_insecure_http") is True)
    except ConfigError:
        _warn("  Invalid server URL. Use HTTPS without credentials or query strings.")
        return None

    raw_token = _ask("  Bot API token (leave blank to keep existing)", password=True)
    token = (raw_token or "").strip() if isinstance(raw_token, str) else ""
    token = token or (get_env_value(TOKEN_ENV) or "").strip()
    if not token or _has_unsafe_characters(token):
        _warn("  A valid Bot API token is required; no settings were changed.")
        return None
    raw_users = _ask("  Allowed user IDs (comma-separated; blank keeps current access)", default="")
    if not isinstance(raw_users, str):
        _warn("  Setup cancelled; no settings were changed.")
        return None
    try:
        allowed_users = _users(raw_users)
    except ConfigError:
        _warn("  Invalid user IDs; no settings were changed.")
        return None
    return url, token, allowed_users


def _save_server(url: str) -> None:
    """Change only our YAML platform block; saving is deliberately before token writes."""
    config = read_raw_config()
    if not isinstance(config, dict):
        raise ConfigError("Malformed configuration")
    platforms = config.setdefault("platforms", {})
    if not isinstance(platforms, dict):
        raise ConfigError("Malformed platform configuration")
    values = platforms.setdefault(PLATFORM, {})
    if not isinstance(values, dict):
        raise ConfigError("Malformed VK Workspace configuration")
    values["base_url"] = url
    values["enabled"] = True
    save_config(config)


def setup_platform() -> None:
    """Hermes platform setup_fn: validate inputs, commit URL, verify, then save secrets."""
    print("\n  VK Workspace / VK Teams setup")
    values = _collect()
    if values is None:
        return
    url, token, allowed_users = values
    try:
        _save_server(url)
        actual = _effective_platform()
        extra = _extra(actual)
        verified = _base_url(extra.get("base_url"), extra.get("allow_insecure_http") is True)
    except ConfigError, OSError, ValueError, RuntimeError:
        _warn("  Could not save or verify VK Workspace settings; token was not changed.")
        return
    if verified != url or actual is None or not actual.enabled:
        _warn(
            "  Effective VK Workspace server or enabled state is managed or overridden; "
            "token was not changed."
        )
        return
    save_env_value(TOKEN_ENV, token)
    if get_env_value(TOKEN_ENV) != token:
        _warn("  Token could not be saved in this Hermes profile.")
        return
    if allowed_users is not None:
        save_env_value(USERS_ENV, allowed_users)
        if get_env_value(USERS_ENV) != allowed_users:
            _warn("  Token saved, but allowed users could not be updated.")
            return
    print("  VK Workspace configured in the current Hermes profile.")
