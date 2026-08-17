
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
from enum import Enum
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
            # Activity/presence group — the working indicator posts here. The
            # real client is
            # ``report_agent_chat_activity(chat_id, *, working, request_options)``.
            self.rest.agent_api_activity.report_agent_chat_activity = AsyncMock()
            self._events = []

        async def report_activity(self, room_id, working, *, timeout_seconds=2):
            """Faithful stand-in for ``BandLink.report_activity``.

            Same contract as the real helper: delegates to the activity REST
            group with a per-POST deadline and retries disabled, swallows any
            failure, and reports success as a bool.
            """
            try:
                await self.rest.agent_api_activity.report_agent_chat_activity(
                    chat_id=room_id,
                    working=working,
                    request_options={
                        "timeout_in_seconds": timeout_seconds,
                        "max_retries": 0,
                    },
                )
            except Exception:
                return False
            return True

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
        """Stand-in for the Fern ``ChatEventRequest``.

        Keyword-only in the real SDK and carries NO ``mentions`` field — events
        are exempt from Band's @mention requirement. Keeping the stub's
        signature identical is what makes ``test_error_events`` a real check
        that the adapter never attaches mentions to an event.
        """

        def __init__(self, content, message_type, metadata=None):
            self.content = content
            self.message_type = message_type
            self.metadata = metadata

    # band.core.types.ToolEventKey — the canonical payload keys the
    # execution-event emitter builds its content dict from. Same (str, Enum)
    # shape as _FakeMessageType above: the SDK relies on members *being* their
    # string values, so a payload keyed by ToolEventKey json.dumps'es to plain
    # "name"/"args"/... keys.
    class _FakeToolEventKey(str, Enum):
        NAME = "name"
        ARGS = "args"
        OUTPUT = "output"
        TOOL_CALL_ID = "tool_call_id"
        IS_ERROR = "is_error"

        def __str__(self):
            return self.value
    # band.core.types — the SDK's usage contract, mirrored faithfully because
    # usage_events.py builds its per-turn accumulator on TurnUsage's arithmetic
    # and serialization, and posts under the two constants. See the note in the
    # real module: usage rides an accepted ``task`` event today because the
    # backend's message_type whitelist rejects ``usage``.
    def _as_int(value):
        return value if isinstance(value, int) else 0

    class _FakeTurnUsage:
        def __init__(
            self,
            input_tokens=0,
            output_tokens=0,
            cache_read_tokens=0,
            cache_write_tokens=0,
        ):
            self.input_tokens = input_tokens
            self.output_tokens = output_tokens
            self.cache_read_tokens = cache_read_tokens
            self.cache_write_tokens = cache_write_tokens

        def __add__(self, other):
            return _FakeTurnUsage(
                input_tokens=self.input_tokens + other.input_tokens,
                output_tokens=self.output_tokens + other.output_tokens,
                cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
                cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            )

        @property
        def total_tokens(self):
            return self.input_tokens + self.output_tokens

        @property
        def is_empty(self):
            return not (
                self.input_tokens
                or self.output_tokens
                or self.cache_read_tokens
                or self.cache_write_tokens
            )

        def to_dict(self):
            return {
                "input_tokens": self.input_tokens,
                "output_tokens": self.output_tokens,
                "cache_read_tokens": self.cache_read_tokens,
                "cache_write_tokens": self.cache_write_tokens,
            }

        @classmethod
        def from_mapping(
            cls,
            data,
            *,
            input,
            output,
            cache_read=None,
            cache_write=None,
            reasoning=None,
        ):
            if not isinstance(data, dict):
                return cls()
            out = _as_int(data.get(output, 0))
            if reasoning:
                out += _as_int(data.get(reasoning, 0))
            return cls(
                input_tokens=_as_int(data.get(input, 0)),
                output_tokens=out,
                cache_read_tokens=_as_int(data.get(cache_read, 0)) if cache_read else 0,
                cache_write_tokens=_as_int(data.get(cache_write, 0)) if cache_write else 0,
            )

    # band.runtime.formatters — pure helper the adapter reuses. Faithful
    # stand-in for replace_uuid_mentions so the adapter's independent import
    # binds the stub rather than its passthrough fallback.
    def _fake_replace_uuid_mentions(content, participants):
        for p in participants or []:
            pid, handle = p.get("id"), p.get("handle")
            if pid and handle:
                content = content.replace(f"@[[{pid}]]", f"@{handle}")
        return content

    # band.core.types.MessageType — the canonical message_type taxonomy. A real
    # StrEnum, because the SDK's is one and the value is what goes on the wire.
    class _FakeMessageType(str, Enum):
        TEXT = "text"
        TOOL_CALL = "tool_call"
        TOOL_RESULT = "tool_result"
        THOUGHT = "thought"
        ERROR = "error"
        TASK = "task"
        USAGE = "usage"

        def __str__(self):
            return self.value

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
    band_core_mod = MagicMock()
    # MERGE NOTE: build each stub module EXACTLY ONCE. More than one slice needs
    # ``band.core.types``; a keep-both merge that leaves two
    # ``band_core_types_mod = MagicMock()`` lines silently discards whatever the
    # first one had attached — and that surfaces far from the cause, as a
    # MagicMock rendered into event content rather than as an import error.
    # Attach to the module below; never rebuild it.
    band_core_types_mod = MagicMock()
    band_core_types_mod.MessageType = _FakeMessageType
    band_core_types_mod.ToolEventKey = _FakeToolEventKey
    band_core_types_mod.TurnUsage = _FakeTurnUsage
    band_core_types_mod.USAGE_EVENT_TYPE = "task"
    band_core_types_mod.USAGE_METADATA_KEY = "band_usage"
    band_runtime_mod = MagicMock()
    band_runtime_formatters_mod = MagicMock()
    band_runtime_formatters_mod.replace_uuid_mentions = _fake_replace_uuid_mentions

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


@pytest.fixture(autouse=True)
def _clean_turn_state():
    """Turn state is module-level, so it must not leak between tests.

    Whether a send is a model reply or out-of-turn traffic depends on whether a
    turn is open for that room. One test opening a turn would otherwise change
    what the next test's send does — which is exactly how this fixture came to
    exist.
    """
    from hermes_band_platform.adapter import reset_turn_state

    reset_turn_state()
    yield
    reset_turn_state()
