"""Hermes platform registration. Importing this module never opens a connection."""

from __future__ import annotations

from importlib.util import find_spec
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gateway.config import PlatformConfig
    from hermes_cli.plugins import PluginContext


def check_requirements() -> bool:
    """Passive dependency probe used by Hermes' setup and status displays."""
    return find_spec("aiohttp") is not None and find_spec("markdown") is not None


def validate_config(config: PlatformConfig) -> bool:
    from .config import ConfigError, Settings

    try:
        Settings.from_platform(config)
    except ConfigError:
        return False
    return True


def create_adapter(config: PlatformConfig):
    from .adapter import VKWorkspaceAdapter

    return VKWorkspaceAdapter(config)


async def standalone_send(config, chat_id, message, **kwargs) -> dict:
    from .adapter import standalone_send as send

    return await send(config, chat_id, message, **kwargs)


def _home_channel_seed() -> dict:
    from gateway.platforms._shared import seed_extra_from_env

    from .config import HOME_ENV

    return seed_extra_from_env((), home_env=HOME_ENV)


def register(ctx: PluginContext) -> None:
    from .config import HOME_ENV, PLATFORM, TOKEN_ENV, parse_target_ref, validate_target_ref

    ctx.register_platform(
        name=PLATFORM,
        label="VK Workspace",
        adapter_factory=create_adapter,
        check_fn=check_requirements,
        validate_config=validate_config,
        is_connected=validate_config,
        required_env=[TOKEN_ENV],
        install_hint="Run `hermes plugins install pfrankov/hermes-vk-workspace --yes-deps --enable`",
        allowed_users_env="VK_WORKSPACE_ALLOWED_USERS",
        allow_all_env="VK_WORKSPACE_ALLOW_ALL_USERS",
        env_enablement_fn=_home_channel_seed,
        cron_deliver_env_var=HOME_ENV,
        parse_target_ref_fn=parse_target_ref,
        validate_target_ref_fn=validate_target_ref,
        standalone_sender_fn=standalone_send,
        max_message_length=4096,
        emoji="💬",
        display_tier="medium",
        platform_hint=(
            "You are chatting through VK Workspace (VK Teams). Use concise Markdown: "
            "bold, italic, inline code, code blocks, and links. Prefer lists to tables."
        ),
    )
