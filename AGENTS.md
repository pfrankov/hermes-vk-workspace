# Repository guide

## Scope and architecture

This is the `vk-workspace` platform plugin for Hermes Agent and the VK Workspace
(VK Teams) Bot API. It does not integrate with the VK social network. User, chat
and message identifiers are opaque strings, not numeric VK peer IDs.

- `plugin.yaml` and `__init__.py`: manifest and lazy platform registration. Import
  and dependency checks must not open connections or read bot credentials.
- `config.py`: validated settings, endpoint trust and target references.
- `adapter.py`: Hermes lifecycle, authorization, inbound dispatch and outbound delivery.
- `api.py`: bounded asynchronous Bot API transport and sanitized errors.
- `events.py` and `state.py`: event normalization, cursor ownership and persistence.
- `interactions.py`: requester-bound, one-use native model and choice pickers.
- `media.py` and `formatting.py`: bounded file handling and VK-compatible rendering.
- `tests/`: local HTTP fixtures, boundary regressions and integration with the real host.

Use the current stable Hermes extension contracts. The supported host is pinned in
CI; update that immutable revision, compatibility documentation and integration
tests together when upgrading. Do not add legacy imports, version-dependent
fallbacks, monkey patches or a vendored copy of Hermes.

Keep production dependencies small and explicit. Reuse the standard library and
the host's extension contracts; do not add framework layers for isolated helpers.
Keep `pyproject.toml`, `uv.lock`, `plugin.yaml`, README and CHANGELOG consistent.
Never commit bot tokens, private server URLs, downloaded attachments, runtime state,
virtual environments or built distributions.

## Code Review Rules

### Simplicity and responsibility

Keep McCabe cyclomatic complexity at or below 10 per function; Ruff C901 enforces
the limit in production code and tests. Do not hide branches in arbitrary lookup
tables or suppress the rule to make the metric pass.

Extract helpers for code reused by at least two production callers, or for a clear
responsibility that needs its own boundary: a host hook, callback, parser, validation
policy, worker or resource lifetime. A single call is valid for those boundaries;
one-use glue should stay at its call site. Test calls do not establish production
reuse. Remove dead parameters and duplicate state before adding abstractions.

Use shared fixtures for repeated setup and spec-constrained mocks at external
boundaries. Test observable behavior and failure boundaries, not every private
helper separately. Do not derive expected results by calling the implementation
being tested. Keep distinct regression cases, but combine equivalent cases with
parametrization when their setup and assertion are the same.

### Authorization and source isolation

Authorize the native sender and chat before downloading attachments or starting
agent work. Unknown DMs may reach Hermes' pairing gate without media downloads.
Preserve the originating platform,
bot, chat and user when constructing sources, session keys and replies. Network
callback data must not grant tool permission or become a privileged command.

### Transport and file boundaries

The Bot API authenticates through a `token` query parameter. Never expose tokens,
signed URLs, raw provider responses or secret-bearing exception causes in logs,
model context or persisted errors. Check logical errors even on HTTP 200. Bound
response sizes, downloads, redirects and timeouts; validate every media redirect
against configured trust using the same URL parser as the HTTP client, including
IDNA and IPv6 normalization. Preserve local-file checks against traversal, symlink
escapes and blocking special files. Do not infer retry safety from HTTP GET:
VK message sends also use GET. Never automatically retry an ambiguous send.

### Host integration and delivery

Review against the actual pinned stable host, including plugin discovery and
platform registration. Permissive mocks are not proof of SDK compatibility. Keep
polling, cursor ownership, in-flight dispatch and shutdown consistent; cancellation
must not acknowledge unfinished work or allow a late reply after shutdown. Surface
delivery failures honestly and document recovery semantics without claiming
exactly-once delivery.

## Validation

Set `HERMES_SOURCE_PATH` to a checkout of the exact revision in CI. CI supplies it
as `.ci/hermes-agent`. Use the same Python version and locked development tools:

```sh
uv sync --locked --group dev
uv pip install --python .venv/bin/python --editable "$HERMES_SOURCE_PATH"
uv pip check --python .venv/bin/python
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync pytest
```

Do not run an exact dependency sync between host installation and tests: it removes
the separately installed Hermes package. Add focused regression coverage for
authorization, serialization, file boundaries, cancellation and delivery fixes.
Assert that rejected input cannot reach downstream effects. Keep tests independent
of operator profiles and credentials. Local HTTP fixtures verify protocol behavior;
do not describe them as a live VK Workspace deployment test.

Keep CI read-only, pin GitHub Actions by full commit SHA and disable persisted Git
credentials. Do not expose secrets to pull requests. Keep mechanical checks in CI;
review rules should describe consequential behavior. Codex automatic review is a
repository setting outside Git: neither this file nor a passing CI run enables it.
