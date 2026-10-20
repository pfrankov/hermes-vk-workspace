"""Run against real Hermes imports in an isolated, credential-free profile."""

import importlib.util
import os
import sys
import tempfile
from pathlib import Path

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
