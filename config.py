"""Validated VK Workspace settings; Hermes owns profiles and sender authorization."""

from __future__ import annotations

import ipaddress
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from urllib.parse import urlsplit, urlunsplit

if TYPE_CHECKING:
    from gateway.config import PlatformConfig

PLATFORM = "vk-workspace"
TOKEN_ENV = "VK_WORKSPACE_BOT_TOKEN"
HOME_ENV = "VK_WORKSPACE_HOME_CHANNEL"
MAX_IDENTIFIER_LENGTH = 256
MAX_MEDIA_BYTES = 100 * 1024 * 1024
_NATIVE_TARGET = re.compile(r"(?:[0-9]+|[^@/:?#\\]+@[^@/:?#\\]+)\Z")


class ConfigError(ValueError):
    """An invalid setting, described without exposing credentials or URL query strings."""


def _has_unsafe_characters(value: str) -> bool:
    return any(char.isspace() or unicodedata.category(char).startswith("C") for char in value)


def identifier(value: object) -> str:
    """Preserve native opaque IDs without allowing invisible delimiters or oversized input."""
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ConfigError("VK Workspace IDs must be strings or integers")
    try:
        value = str(value)
    except ValueError:
        raise ConfigError("VK Workspace IDs must contain at most 256 characters") from None
    if not value or len(value) > MAX_IDENTIFIER_LENGTH or _has_unsafe_characters(value):
        raise ConfigError("VK Workspace IDs must contain 1–256 visible characters without spaces")
    return value


def parse_target_ref(value: str) -> tuple[str, None] | None:
    """Resolve numeric and email-shaped native IDs before Hermes' named-chat directory."""
    try:
        target = identifier(value)
    except ConfigError:
        return None
    return (target, None) if _NATIVE_TARGET.fullmatch(target) else None


def validate_target_ref(value: str) -> bool | str:
    """Validate an ID after Hermes has resolved a directory name or native target."""
    try:
        identifier(value)
    except ConfigError as exc:
        return str(exc)
    return True


def _split_url(value: object, key: str, allow_http: bool):
    if not isinstance(value, str) or not value or _has_unsafe_characters(value) or "\\" in value:
        raise ConfigError(f"{key} must be an absolute HTTPS URL without spaces")
    try:
        url = urlsplit(value)
        port = url.port
        host = url.hostname
    except ValueError:
        raise ConfigError(f"{key} contains an invalid hostname or port") from None
    if url.scheme not in ({"http", "https"} if allow_http else {"https"}):
        raise ConfigError(f"{key} requires HTTPS (HTTP is available only with allow_insecure_http)")
    if (
        not host
        or url.username is not None
        or url.password is not None
        or "?" in value
        or "#" in value
    ):
        raise ConfigError(f"{key} must have a hostname and no credentials, query, or fragment")
    if "%" in url.netloc or (port is not None and not 1 <= port <= 65535):
        raise ConfigError(f"{key} contains an invalid hostname or port")
    try:
        if ":" in host:
            host = f"[{ipaddress.IPv6Address(host).compressed}]"
        else:
            host = host.encode("idna").decode("ascii").lower()
            if not re.fullmatch(r"[a-z0-9_.-]+", host):
                raise ValueError
    except UnicodeError, ValueError:
        raise ConfigError(f"{key} contains an invalid hostname") from None
    if port is not None and port != (443 if url.scheme == "https" else 80):
        host = f"{host}:{port}"
    return url._replace(netloc=host)


def _base_url(value: object, allow_http: bool) -> str:
    url = _split_url(value, "base_url", allow_http)
    path = url.path.rstrip("/")
    if "%" in path or any(segment in {".", ".."} for segment in path.split("/")) or "//" in path:
        raise ConfigError("base_url must use a clean path without traversal or encoded delimiters")
    if not path.endswith("/bot/v1"):
        path += "/bot/v1"
    return urlunsplit(url._replace(path=path))


def _origin(value: object, allow_http: bool) -> str:
    url = _split_url(value, "media_allowed_origins", allow_http)
    if url.path not in {"", "/"}:
        raise ConfigError("media_allowed_origins must contain exact origins without paths")
    return urlunsplit(url._replace(path=""))


def _boolean(extra: Mapping, key: str, default: bool) -> bool:
    value = extra.get(key, default)
    if not isinstance(value, bool):
        raise ConfigError(f"{key} must be a YAML boolean (true or false)")
    return value


def _integer(extra: Mapping, key: str, default: int, minimum: int, maximum: int) -> int:
    value = extra.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not minimum <= value <= maximum:
        raise ConfigError(f"{key} must be an integer between {minimum} and {maximum}")
    return value


def _sequence(extra: Mapping, key: str) -> list | tuple:
    value = extra.get(key, ())
    if not isinstance(value, (list, tuple)):
        raise ConfigError(f"{key} must be a YAML list")
    return value


@dataclass(frozen=True, slots=True)
class Settings:
    base_url: str
    token: str = field(repr=False)
    request_timeout: float = 30.0
    poll_time: int = 30
    media_max_bytes: int = 20 * 1024 * 1024
    media_allowed_origins: tuple[str, ...] = ()
    allow_insecure_http: bool = False
    allowed_chats: tuple[str, ...] = ()
    require_mention: bool = True

    @property
    def base_origin(self) -> str:
        return urlunsplit(urlsplit(self.base_url)._replace(path=""))

    @classmethod
    def from_platform(cls, config: PlatformConfig) -> Settings:
        from gateway.platforms._shared import get_scoped_secret

        extra = config.extra
        if not isinstance(extra, Mapping):
            raise ConfigError("VK Workspace platform settings must be a mapping")
        if "allow_all_users" in extra:
            raise ConfigError(
                "Hermes does not read YAML allow_all_users; use allowed_users: ['*'] "
                "or VK_WORKSPACE_ALLOW_ALL_USERS instead"
            )
        allow_http = _boolean(extra, "allow_insecure_http", False)
        base_url = _base_url(extra.get("base_url"), allow_http)
        # A scoped miss never borrows another profile's environment. Native config.token
        # is available to host-managed credential injection; plugin YAML uses TOKEN_ENV.
        token = get_scoped_secret(TOKEN_ENV, None)
        if token is None or (isinstance(token, str) and not token.strip()):
            token = config.token
        if not isinstance(token, str) or not token.strip() or _has_unsafe_characters(token.strip()):
            raise ConfigError(f"Set {TOKEN_ENV} in the active Hermes profile")
        token = token.strip()

        timeout = extra.get("request_timeout", 30.0)
        if (
            isinstance(timeout, bool)
            or not isinstance(timeout, (float, int))
            or not 1 <= timeout <= 300
        ):
            raise ConfigError("request_timeout must be a finite number between 1 and 300 seconds")
        if "media_max_bytes" in extra and "media_max_mb" in extra:
            raise ConfigError("Set either media_max_bytes or media_max_mb")
        if "media_max_mb" in extra:
            media_limit = _integer(extra, "media_max_mb", 20, 1, 100) * 1024 * 1024
        else:
            media_limit = _integer(extra, "media_max_bytes", 20 * 1024 * 1024, 1, MAX_MEDIA_BYTES)
        base_origin = urlunsplit(urlsplit(base_url)._replace(path=""))
        origins = dict.fromkeys(
            [
                base_origin,
                *(
                    _origin(value, allow_http)
                    for value in _sequence(extra, "media_allowed_origins")
                ),
            ]
        )
        chats = dict.fromkeys(identifier(value) for value in _sequence(extra, "allowed_chats"))
        return cls(
            base_url=base_url,
            token=token,
            request_timeout=float(timeout),
            poll_time=_integer(extra, "poll_time", 30, 1, 60),
            media_max_bytes=media_limit,
            media_allowed_origins=tuple(origins),
            allow_insecure_http=allow_http,
            allowed_chats=tuple(chats),
            require_mention=_boolean(extra, "require_mention", True),
        )
