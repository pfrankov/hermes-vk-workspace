"""Run against real Hermes imports in an isolated, credential-free profile."""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import create_autospec

import pytest

# Set before importing Hermes: some core cache paths are initialized at import.
_home = tempfile.TemporaryDirectory(prefix="hermes-vk-test-")
os.environ["HERMES_HOME"] = _home.name
os.environ["HERMES_SKIP_DOTENV"] = "1"
for _key in tuple(os.environ):
    if _key.startswith("VK_WORKSPACE_"):
        del os.environ[_key]

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "hermes_vk_workspace", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
)
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)


@pytest.fixture(autouse=True)
def isolated_profile(tmp_path, monkeypatch):
    """Never read the developer's bot token or write to their actual profile."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "profile"))
    for key in tuple(os.environ):
        if key.startswith("VK_WORKSPACE_"):
            monkeypatch.delenv(key)
    yield


@pytest.fixture
def adapter():
    """Use the real registration and adapter; mock only the remote API contract."""
    import hermes_vk_workspace as plugin
    from gateway.config import PlatformConfig
    from hermes_cli.plugins import PluginContext, PluginManager
    from hermes_cli.plugins_manifest import parse_manifest_file
    from hermes_vk_workspace.adapter import VKWorkspaceAdapter
    from hermes_vk_workspace.api import TeamsAPI

    manager = PluginManager()
    plugin.register(
        PluginContext(parse_manifest_file(ROOT / "plugin.yaml", ROOT, "user", ""), manager)
    )
    config = PlatformConfig(token="fake-bot-token", extra={"base_url": "https://vk.example"})
    result = VKWorkspaceAdapter(config)
    result._running = True
    result.self_id = "bot@example"
    result.api = create_autospec(TeamsAPI, instance=True, spec_set=True)
    result.api.send_text.side_effect = lambda *_args, **_kwargs: str(
        result.api.send_text.call_count
    )
    result.api.send_file.return_value = "file-1"
    result.api.send_voice.return_value = "voice-1"
    result.set_authorization_check(lambda *_args, **_kwargs: True)
    yield result
    manager.unload()
