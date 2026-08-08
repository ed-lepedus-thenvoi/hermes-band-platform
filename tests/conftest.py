"""Shared test fixtures for the Band platform plugin.

The band SDK is NOT assumed installed in the test environment. A minimal but
*faithful* stub is injected into ``sys.modules`` at collection time — BEFORE
``hermes_band_platform.adapter`` (or ``.tools``) is imported — so the adapter's
top-level ``try: from band ...`` binds the stub and ``BAND_AVAILABLE`` stays
True, and so ``tools.py`` constructs real request-type objects (not auto-attr
MagicMocks).

If the real ``band-sdk`` is installed, we leave it in place.
"""

from __future__ import annotations

import sys
from enum import StrEnum
from unittest.mock import AsyncMock, MagicMock

import pytest


def _install_band_mock() -> MagicMock:
    """Register stub band sub-packages into sys.modules (idempotent).

    Returns the root ``band`` mock so callers can attach extra attrs.
    """
    if "band" in sys.modules:
        # Already present (real SDK or a prior install) — reuse it.
        return sys.modules["band"]

    # A class that, when instantiated, returns an AsyncMock-backed link object.
    class _FakeLinkClass:
        def __init__(self, agent_id, api_key, ws_url, rest_url):
            self._agent_id = agent_id
            self._api_key = api_key
            self._ws_url = ws_url
            self._rest_url = rest_url
            self.connect = AsyncMock()
            self.disconnect = AsyncMock()
            self.subscribe_agent_rooms = AsyncMock()
            self.subscribe_room = AsyncMock()
            self.unsubscribe_room = AsyncMock()
            self.rest = MagicMock()
            self._events = []

        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

    # SDK request types used by the adapter *and* the tools module. The stub
    # must be a faithful stand-in for every request type either module
    # constructs, with the real keyword signatures.
    class _FakeChatMessageRequest:
        def __init__(self, content, mentions):
            self.content = content
            self.mentions = mentions

    class _FakeChatMessageRequestMentionsItem:
        def __init__(self, id, handle=None, name=None):
            self.id = id
            self.handle = handle
            self.name = name

    class _FakeParticipantRequest:
        def __init__(self, participant_id, role=None):
            self.participant_id = participant_id
            self.role = role

    class _FakeChatRoomRequest:
        def __init__(self, task_id=None):
            self.task_id = task_id

    class _FakeChatEventRequest:
        def __init__(self, content, message_type, metadata=None):
            self.content = content
            self.message_type = message_type
            self.metadata = metadata

    # band.core.types — the canonical event taxonomy the execution-event
    # emitter keys its payloads by. Real StrEnums, because the SDK relies on
    # members *being* their string values (a payload keyed by ToolEventKey
    # json.dumps'es to plain "name"/"args"/… keys).
    class _FakeMessageType(StrEnum):
        TEXT = "text"
        TOOL_CALL = "tool_call"
        TOOL_RESULT = "tool_result"
        THOUGHT = "thought"
        ERROR = "error"
        TASK = "task"
        USAGE = "usage"

    class _FakeToolEventKey(StrEnum):
        NAME = "name"
        ARGS = "args"
        OUTPUT = "output"
        TOOL_CALL_ID = "tool_call_id"
        IS_ERROR = "is_error"

    # band.runtime.formatters — pure helper the adapter reuses. Faithful
    # stand-in for replace_uuid_mentions so the adapter's independent import
    # binds the stub rather than its passthrough fallback.
    def _fake_replace_uuid_mentions(content, participants):
        for p in participants or []:
            pid, handle = p.get("id"), p.get("handle")
            if pid and handle:
                content = content.replace(f"@[[{pid}]]", f"@{handle}")
        return content

    band_mod = MagicMock()
    band_platform_mod = MagicMock()
    band_platform_link_mod = MagicMock()
    band_platform_link_mod.BandLink = _FakeLinkClass
    band_platform_event_mod = MagicMock()
    band_client_mod = MagicMock()
    band_client_rest_mod = MagicMock()
    band_client_rest_mod.ChatMessageRequest = _FakeChatMessageRequest
    band_client_rest_mod.ChatMessageRequestMentionsItem = _FakeChatMessageRequestMentionsItem
    band_client_rest_mod.ParticipantRequest = _FakeParticipantRequest
    band_client_rest_mod.ChatRoomRequest = _FakeChatRoomRequest
    band_client_rest_mod.ChatEventRequest = _FakeChatEventRequest
    band_client_rest_mod.DEFAULT_REQUEST_OPTIONS = {"max_retries": 3}
    band_runtime_mod = MagicMock()
    band_runtime_formatters_mod = MagicMock()
    band_runtime_formatters_mod.replace_uuid_mentions = _fake_replace_uuid_mentions
    band_core_mod = MagicMock()
    band_core_types_mod = MagicMock()
    band_core_types_mod.MessageType = _FakeMessageType
    band_core_types_mod.ToolEventKey = _FakeToolEventKey

    sys.modules["band"] = band_mod
    sys.modules["band.platform"] = band_platform_mod
    sys.modules["band.platform.link"] = band_platform_link_mod
    sys.modules["band.platform.event"] = band_platform_event_mod
    sys.modules["band.client"] = band_client_mod
    sys.modules["band.client.rest"] = band_client_rest_mod
    sys.modules["band.core"] = band_core_mod
    sys.modules["band.core.types"] = band_core_types_mod
    sys.modules["band.runtime"] = band_runtime_mod
    sys.modules["band.runtime.formatters"] = band_runtime_formatters_mod

    return band_mod


# Install the stub at collection time, before any test module imports the
# adapter / tools package.
_install_band_mock()


@pytest.fixture(scope="session", autouse=True)
def _register_band_platform():
    """Register the ``band`` platform in the host registry before tests run.

    The host's ``gateway.config.Platform`` is a *strict* enum: its ``_missing_``
    hook only mints a pseudo-member (so ``Platform("band")`` resolves) once the
    platform is present in ``platform_registry`` — which the gateway does at
    plugin-load time, before any adapter is constructed. The unit tests build
    ``BandAdapter`` directly, bypassing that load, so without this fixture
    ``Platform("band")`` raises ``ValueError`` during construction.

    We mirror the gateway's own ``register_platform`` → ``PlatformEntry`` path
    by driving the plugin's real ``register()`` with a registry-forwarding
    context. No-op if the host isn't importable (the adapter import would have
    already failed in that case).
    """
    try:
        from gateway.platform_registry import PlatformEntry, platform_registry
    except Exception:
        yield
        return

    if not platform_registry.is_registered("band"):
        import hermes_band_platform

        class _RegistryCtx:
            """Forwards register_platform into platform_registry; ignores the rest."""

            def register_platform(
                self,
                name,
                label=None,
                adapter_factory=None,
                check_fn=None,
                validate_config=None,
                required_env=None,
                install_hint=None,
                **extra,
            ):
                extra.setdefault("plugin_name", "band")
                platform_registry.register(
                    PlatformEntry(
                        name=name,
                        label=label,
                        adapter_factory=adapter_factory,
                        check_fn=check_fn,
                        validate_config=validate_config,
                        required_env=required_env or [],
                        install_hint=install_hint,
                        source="plugin",
                        **extra,
                    )
                )

            def register_tool(self, **kwargs):
                pass

            def register_skill(self, *args, **kwargs):
                pass

            def register_hook(self, hook_name, callback):
                pass

        hermes_band_platform.register(_RegistryCtx())

    yield
