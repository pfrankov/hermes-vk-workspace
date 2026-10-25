"""Opt-in native foreground lifecycle against loopback APIs, without agent inference."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import sys
from pathlib import Path

import pytest
from aiohttp import web

ROOT = Path(__file__).resolve().parents[1]


def is_metadata_probe(method: str, path: str, body: bytes) -> bool:
    """Recognize the pinned host's local-server metadata probes, never inference."""
    if method == "GET":
        return path in {"/api/v1/models", "/api/tags", "/v1/props", "/props", "/version"}
    if method != "POST" or path != "/api/show":
        return False
    try:
        return json.loads(body) == {"name": "synthetic-model"}
    except ValueError, UnicodeError:
        return False


@pytest.mark.parametrize(
    ("method", "path", "body", "expected"),
    [
        ("GET", "/api/tags", b"", True),
        ("POST", "/api/show", b'{"name":"synthetic-model"}', True),
        ("POST", "/api/show", b'{"name":"different-model"}', False),
        ("POST", "/api/show", b"not-json", False),
        ("POST", "/v1/chat/completions", b'{"model":"synthetic-model"}', False),
        ("POST", "/v1/responses", b'{"model":"synthetic-model"}', False),
        ("POST", "/api/generate", b'{"model":"synthetic-model"}', False),
        ("POST", "/api/chat", b'{"model":"synthetic-model"}', False),
        ("POST", "/api/tags", b"", False),
        ("GET", "/unexpected", b"", False),
    ],
)
def test_metadata_allowance_does_not_allow_inference(method, path, body, expected):
    assert is_metadata_probe(method, path, body) is expected


def gateway_environment(home: Path, user: Path, bundled: Path, base: str) -> dict[str, str]:
    """Preserve transport trust/proxy configuration, not operator credentials."""
    keys = {
        "PATH",
        "LANG",
        "LC_ALL",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "REQUESTS_CA_BUNDLE",
        "CURL_CA_BUNDLE",
        "HTTPS_PROXY",
        "HTTP_PROXY",
        "ALL_PROXY",
        "NO_PROXY",
        "https_proxy",
        "http_proxy",
        "all_proxy",
        "no_proxy",
        "UV_CACHE_DIR",
        "UV_PYTHON_INSTALL_DIR",
    }
    return {
        **{key: value for key, value in os.environ.items() if key in keys},
        "HOME": str(user),
        "USERPROFILE": str(user),
        "HERMES_HOME": str(home),
        "HERMES_SKIP_DOTENV": "1",
        "HERMES_ENABLE_PROJECT_PLUGINS": "0",
        "HERMES_BUNDLED_PLUGINS": str(bundled),
        "PYTHONUNBUFFERED": "1",
        "VK_WORKSPACE_BOT_TOKEN": "synthetic-loopback-token",
        "VK_WORKSPACE_ALLOWED_USERS": "synthetic-user",
        "VK_WORKSPACE_ALLOW_ALL_USERS": "false",
        "OPENAI_API_KEY": "synthetic-loopback-key",
        "OPENAI_BASE_URL": base + "/v1",
    }


async def run_gateway_cycle(home: Path, env: dict[str, str], polls: list[str], cycle: int) -> None:
    """Wait for ten real polls, then require SIGINT to drain and release ownership."""
    log_path = home / f"gateway-smoke-{cycle}.log"
    start = len(polls)
    with log_path.open("wb") as log:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "hermes_cli.main",
            "gateway",
            "run",
            cwd=home,
            env=env,
            stdout=log,
            stderr=asyncio.subprocess.STDOUT,
        )
        try:
            async with asyncio.timeout(90):
                while len(polls) - start < 10:
                    assert process.returncode is None, log_path.read_text()
                    await asyncio.sleep(0.1)
            process.send_signal(signal.SIGINT)
            assert await asyncio.wait_for(process.wait(), 30) == 0, log_path.read_text()
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()
    assert not (home / "gateway.sock").exists(), log_path.read_text()


# Keep the real host's Unix control socket; restricted sandboxes skip only this test.
@pytest.mark.skipif(
    os.environ.get("HERMES_GATEWAY_SMOKE") != "1",
    reason="Opt in on the hosted Linux CI runner with native Unix sockets",
)
async def test_native_gateway_polls_and_restarts_cleanly(tmp_path):
    home, user, bundled = (tmp_path / name for name in ("home", "user", "bundled"))
    for path in (home, user, bundled):
        path.mkdir()
    shutil.copytree(
        ROOT,
        home / "plugins" / "vk-workspace-platform",
        ignore=shutil.ignore_patterns(
            ".git",
            ".venv",
            ".ci",
            ".hermes-test",
            ".env",
            "__pycache__",
            ".pytest_cache",
            ".ruff_cache",
            "build",
            "dist",
            "*.egg-info",
        ),
    )
    polls, unexpected = [], []

    async def peer(request):
        # Native local endpoint detection also uses POST /api/show for metadata.
        # Return 404 (not an Ollama/LM Studio server), without allowing inference.
        if is_metadata_probe(request.method, request.path, await request.read()):
            raise web.HTTPNotFound()
        if request.method != "GET":
            unexpected.append((request.method, request.path))
            raise web.HTTPForbidden()
        if request.path == "/bot/v1/self/get":
            return web.json_response({"ok": True, "userId": "synthetic-bot"})
        if request.path == "/bot/v1/events/get":
            polls.append(request.query.get("lastEventId"))
            await asyncio.sleep(0.2)
            return web.json_response({"ok": True, "events": []})
        if request.path == "/v1/models":
            return web.json_response({"object": "list", "data": [{"id": "synthetic-model"}]})
        unexpected.append((request.method, request.path))
        raise web.HTTPNotFound()

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", peer)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        base = f"http://127.0.0.1:{runner.addresses[0][1]}"
        config = {
            "plugins": {"enabled": ["vk-workspace-platform"]},
            "platforms": {
                "vk-workspace": {
                    "enabled": True,
                    "base_url": base + "/bot/v1",
                    "allow_insecure_http": True,
                    "poll_time": 1,
                }
            },
            "model": {
                "provider": "custom",
                "default": "synthetic-model",
                "base_url": base + "/v1",
                "api_key": "synthetic-loopback-key",
            },
            "gateway": {"multiplex_profiles": False},
            "terminal": {"cwd": str(home)},
            "auxiliary": {"title_generation": {"enabled": False}},
        }
        (home / "config.yaml").write_text(json.dumps(config))
        env = gateway_environment(home, user, bundled, base)
        for cycle in range(2):
            await run_gateway_cycle(home, env, polls, cycle)
        assert not unexpected, unexpected
    finally:
        await runner.cleanup()
