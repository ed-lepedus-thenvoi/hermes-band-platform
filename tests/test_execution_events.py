"""Execution events — tool_call / tool_result emission into a Band room.

Covers the emission shape, the redaction + size guards (both safety-critical:
a Band event cannot be deleted once written), the best-effort failure contract,
and the two properties the design leans on — events never carry mentions, and
they never re-enter the agent's own context.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from hermes_band_platform import execution_events as ee


ROOM = "11111111-1111-1111-1111-111111111111"
SESSION = "sess-abc"
SESSION_KEY = "agent:main:band:group:" + ROOM


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_room_cache():
    ee._SESSION_ROOM_CACHE.clear()
    yield
    ee._SESSION_ROOM_CACHE.clear()


def _make_adapter(*, rooms=(ROOM,), session_id=SESSION):
    """A stand-in adapter exposing only what the emitter reads off it."""
    link = MagicMock()
    link.rest.agent_api_events.create_agent_chat_event = AsyncMock()

    store = MagicMock()
    store._ensure_loaded = MagicMock()
    store._entries = {SESSION_KEY: SimpleNamespace(session_id=session_id)}

    adapter = SimpleNamespace(
        _link=link,
        _link_loop=None,
        _session_store=store,
        _known_rooms=set(rooms),
        _hub_room_id=None,
        _session_key_for=lambda room_id: "agent:main:band:group:" + room_id,
    )
    return adapter


def _sent_event(adapter):
    """The single ChatEventRequest handed to the events endpoint."""
    call = adapter._link.rest.agent_api_events.create_agent_chat_event.call_args
    assert call is not None, "no execution event was posted"
    return call


# ---------------------------------------------------------------------------
# Emission shape
# ---------------------------------------------------------------------------

class TestEmissionShape:
    async def test_tool_call_emits_message_type_and_tool_event_keys(self):
        adapter = _make_adapter()
        ok = await ee.emit_event(
            adapter,
            ee._TOOL_CALL,
            SESSION,
            {
                ee._K_NAME: "terminal",
                ee._K_ARGS: {"command": "ls -la"},
                ee._K_TOOL_CALL_ID: "call_1",
            },
        )
        assert ok is True

        call = _sent_event(adapter)
        assert call.kwargs["chat_id"] == ROOM
        event = call.kwargs["event"]
        assert event.message_type == "tool_call"

        payload = json.loads(event.content)
        assert payload == {
            "name": "terminal",
            "args": {"command": "ls -la"},
            "tool_call_id": "call_1",
        }
        # Bounded correlation fields ride alongside; the bulk stays in content
        # under the single documented size cap.
        assert event.metadata == {"name": "terminal", "tool_call_id": "call_1"}

    async def test_tool_result_carries_output_and_is_error(self):
        adapter = _make_adapter()
        await ee.emit_event(
            adapter,
            ee._TOOL_RESULT,
            SESSION,
            {
                ee._K_NAME: "read_file",
                ee._K_OUTPUT: "boom",
                ee._K_TOOL_CALL_ID: "call_2",
                ee._K_IS_ERROR: True,
            },
        )
        event = _sent_event(adapter).kwargs["event"]
        assert event.message_type == "tool_result"

        payload = json.loads(event.content)
        assert payload["output"] == "boom"
        assert payload["is_error"] is True
        assert event.metadata["is_error"] is True

    async def test_request_options_are_passed(self):
        adapter = _make_adapter()
        await ee.emit_event(adapter, ee._TOOL_CALL, SESSION, {ee._K_NAME: "x"})
        assert "request_options" in _sent_event(adapter).kwargs


# ---------------------------------------------------------------------------
# No mentions, ever
# ---------------------------------------------------------------------------

class TestNoMentions:
    async def test_event_request_carries_no_mentions_and_no_message_is_sent(self):
        adapter = _make_adapter()
        await ee.emit_event(adapter, ee._TOOL_CALL, SESSION, {ee._K_NAME: "web_search"})

        event = _sent_event(adapter).kwargs["event"]
        # Messages require >=1 mention; events are exempt and must not carry
        # any, so emitting many can never ping a room participant.
        assert not hasattr(event, "mentions")
        assert getattr(event, "mentions", None) is None
        adapter._link.rest.agent_api_messages.create_agent_chat_message.assert_not_called()


# ---------------------------------------------------------------------------
# Size and blank content
# ---------------------------------------------------------------------------

class TestSizeAndBlankContent:
    def test_truncate_is_a_noop_under_the_cap(self):
        content = "x" * (ee._EVENT_CONTENT_MAX_LENGTH - 1)
        assert ee._truncate_event_content(content) == content

    def test_truncate_keeps_head_and_tail_around_the_marker(self):
        content = "H" * 20000 + "TAIL"
        out = ee._truncate_event_content(content)
        assert len(out) == ee._EVENT_CONTENT_MAX_LENGTH
        assert ee._EVENT_TRUNCATION_MARKER in out
        assert out.startswith("H")
        assert out.endswith("TAIL")

    async def test_oversized_payload_is_truncated_before_emit(self):
        adapter = _make_adapter()
        await ee.emit_event(
            adapter,
            ee._TOOL_RESULT,
            SESSION,
            {ee._K_NAME: "terminal", ee._K_OUTPUT: "y" * 100_000},
        )
        content = _sent_event(adapter).kwargs["event"].content
        assert len(content) == ee._EVENT_CONTENT_MAX_LENGTH
        assert ee._EVENT_TRUNCATION_MARKER in content

    def test_blank_tool_output_becomes_the_placeholder(self):
        assert ee._output_text("") == ee._EVENT_EMPTY_CONTENT_PLACEHOLDER
        assert ee._output_text(None) == ee._EVENT_EMPTY_CONTENT_PLACEHOLDER

    async def test_blank_result_emits_the_placeholder(self):
        adapter = _make_adapter()
        ee.on_post_tool_call(
            tool_name="noop", result="", session_id=SESSION, tool_call_id="c",
        )
        # on_post_tool_call schedules; drive the coroutine directly for the
        # payload assertion.
        await ee.emit_event(
            adapter,
            ee._TOOL_RESULT,
            SESSION,
            {ee._K_NAME: "noop", ee._K_OUTPUT: ee._output_text("")},
        )
        payload = json.loads(_sent_event(adapter).kwargs["event"].content)
        assert payload["output"] == ee._EVENT_EMPTY_CONTENT_PLACEHOLDER

    def test_blank_serialized_content_falls_back_to_the_placeholder(self, monkeypatch):
        """The SDK's own last-resort guard, reproduced: never post empty content."""
        monkeypatch.setattr(ee, "json", SimpleNamespace(dumps=lambda *a, **kw: ""))
        assert ee._event_content({"a": 1}) == ee._EVENT_EMPTY_CONTENT_PLACEHOLDER


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------

class TestRedaction:
    async def test_payload_is_run_through_the_host_redactor(self, monkeypatch):
        seen = {}

        def _fake_redact(text, force=False):
            seen["text"] = text
            seen["force"] = force
            return text.replace("hunter2", "[REDACTED]")

        import agent.redact

        monkeypatch.setattr(agent.redact, "redact_sensitive_text", _fake_redact)

        adapter = _make_adapter()
        await ee.emit_event(
            adapter,
            ee._TOOL_CALL,
            SESSION,
            {ee._K_NAME: "terminal", ee._K_ARGS: {"command": "login hunter2"}},
        )
        # force=True: the redactor runs even when security.redact_secrets is off.
        assert seen["force"] is True
        content = _sent_event(adapter).kwargs["event"].content
        assert "hunter2" not in content
        assert "[REDACTED]" in content

    async def test_redacted_content_is_still_valid_json(self):
        """Regression: redacting the serialized JSON broke it.

        The host redactor masks a matched token *and its trailing punctuation*,
        so scrubbing after ``json.dumps`` ate the escaped quote closing an
        embedded string and left content Band's own parser could not read.
        Values are scrubbed before serialization instead.
        """
        adapter = _make_adapter()
        secret = "sk-proj-AbCdEf0123456789AbCdEf0123456789"
        await ee.emit_event(
            adapter,
            ee._TOOL_CALL,
            SESSION,
            {
                ee._K_NAME: "terminal",
                ee._K_ARGS: {
                    "command": f'curl -H "Authorization: Bearer {secret}" https://x'
                },
            },
        )
        content = _sent_event(adapter).kwargs["event"].content
        payload = json.loads(content)  # must still parse
        assert secret not in content
        assert payload["name"] == "terminal"
        assert payload["args"]["command"].startswith("curl -H ")

    async def test_non_json_values_are_stringified_and_scrubbed(self, monkeypatch):
        """``json.dumps(default=str)`` would render these after the redactor."""
        import agent.redact

        monkeypatch.setattr(
            agent.redact,
            "redact_sensitive_text",
            lambda text, force=False: text.replace("hunter2", "[REDACTED]"),
        )

        class _Exotic:
            def __str__(self):
                return "password=hunter2"

        adapter = _make_adapter()
        await ee.emit_event(
            adapter,
            ee._TOOL_CALL,
            SESSION,
            {ee._K_NAME: "t", ee._K_ARGS: {"blob": _Exotic()}},
        )
        content = _sent_event(adapter).kwargs["event"].content
        assert "hunter2" not in content
        assert "[REDACTED]" in content

    async def test_event_is_dropped_when_the_redactor_is_unavailable(self, monkeypatch):
        def _boom(text, force=False):
            raise RuntimeError("redactor gone")

        import agent.redact

        monkeypatch.setattr(agent.redact, "redact_sensitive_text", _boom)

        adapter = _make_adapter()
        ok = await ee.emit_event(
            adapter, ee._TOOL_CALL, SESSION, {ee._K_NAME: "terminal"}
        )
        # Fail closed: a Band event cannot be deleted once written.
        assert ok is False
        adapter._link.rest.agent_api_events.create_agent_chat_event.assert_not_called()


# ---------------------------------------------------------------------------
# Best-effort failure contract
# ---------------------------------------------------------------------------

class TestFailuresAreSwallowed:
    async def test_raising_api_call_is_swallowed(self):
        adapter = _make_adapter()
        adapter._link.rest.agent_api_events.create_agent_chat_event = AsyncMock(
            side_effect=RuntimeError("502 Bad Gateway")
        )
        assert await ee.emit_event(
            adapter, ee._TOOL_CALL, SESSION, {ee._K_NAME: "terminal"}
        ) is False

    async def test_missing_link_is_swallowed(self):
        adapter = _make_adapter()
        adapter._link = None
        assert await ee.emit_event(
            adapter, ee._TOOL_CALL, SESSION, {ee._K_NAME: "terminal"}
        ) is False

    async def test_broken_session_store_is_swallowed(self):
        adapter = _make_adapter()
        adapter._session_store._ensure_loaded.side_effect = RuntimeError("db locked")
        assert await ee.emit_event(
            adapter, ee._TOOL_CALL, SESSION, {ee._K_NAME: "terminal"}
        ) is False

    def test_hook_callbacks_never_raise_without_a_gateway(self):
        # No live gateway runner: both observers must no-op silently and return
        # None (a non-None dict would be read as a block/approve directive).
        assert ee.on_pre_tool_call(
            tool_name="terminal", args={"command": "ls"},
            session_id=SESSION, tool_call_id="c1",
        ) is None
        assert ee.on_post_tool_call(
            tool_name="terminal", result="ok",
            session_id=SESSION, tool_call_id="c1", status="ok",
        ) is None

    def test_scheduler_no_ops_when_the_link_loop_is_not_running(self, monkeypatch):
        adapter = _make_adapter()
        adapter._link_loop = SimpleNamespace(is_running=lambda: False)
        monkeypatch.setattr(ee, "_live_adapter", lambda: adapter)
        ee._schedule(ee._TOOL_CALL, SESSION, {ee._K_NAME: "terminal"})
        adapter._link.rest.agent_api_events.create_agent_chat_event.assert_not_called()


# ---------------------------------------------------------------------------
# Room resolution — the filter that keeps other platforms' turns out
# ---------------------------------------------------------------------------

class TestRoomResolution:
    def test_resolves_a_known_room_from_its_session_id(self):
        adapter = _make_adapter()
        assert ee._room_for_session(adapter, SESSION) == ROOM

    def test_unknown_session_resolves_to_none(self):
        adapter = _make_adapter()
        assert ee._room_for_session(adapter, "sess-from-telegram") is None

    def test_blank_session_resolves_to_none(self):
        assert ee._room_for_session(_make_adapter(), "") is None

    def test_hub_room_is_searched_even_when_not_in_known_rooms(self):
        adapter = _make_adapter(rooms=())
        adapter._hub_room_id = ROOM
        assert ee._room_for_session(adapter, SESSION) == ROOM

    def test_resolution_is_memoized(self):
        adapter = _make_adapter()
        ee._room_for_session(adapter, SESSION)
        adapter._session_store._ensure_loaded.reset_mock()
        assert ee._room_for_session(adapter, SESSION) == ROOM
        adapter._session_store._ensure_loaded.assert_not_called()

    async def test_a_turn_from_another_platform_emits_nothing(self):
        adapter = _make_adapter()
        ok = await ee.emit_event(
            adapter, ee._TOOL_CALL, "sess-from-telegram", {ee._K_NAME: "terminal"}
        )
        assert ok is False
        adapter._link.rest.agent_api_events.create_agent_chat_event.assert_not_called()


# ---------------------------------------------------------------------------
# Hook payload mapping
# ---------------------------------------------------------------------------

class TestHookPayloads:
    def _capture(self, monkeypatch):
        calls = []
        monkeypatch.setattr(
            ee,
            "_schedule",
            lambda message_type, session_id, payload: calls.append(
                (message_type, session_id, payload)
            ),
        )
        return calls

    def test_pre_hook_maps_onto_tool_call(self, monkeypatch):
        calls = self._capture(monkeypatch)
        ee.on_pre_tool_call(
            tool_name="terminal",
            args={"command": "ls"},
            session_id=SESSION,
            tool_call_id="c1",
            # invoke_hook always adds this, plus other observer fields.
            telemetry_schema_version="hermes.observer.v1",
            task_id="",
            turn_id="",
            api_request_id="",
            middleware_trace=[],
        )
        assert calls == [
            (
                "tool_call",
                SESSION,
                {"name": "terminal", "args": {"command": "ls"}, "tool_call_id": "c1"},
            )
        ]

    def test_pre_hook_coerces_non_dict_args(self, monkeypatch):
        calls = self._capture(monkeypatch)
        ee.on_pre_tool_call(tool_name="t", args=None, session_id=SESSION)
        assert calls[0][2]["args"] == {}

    @pytest.mark.parametrize(
        "status,expected",
        [("ok", False), ("error", True), ("blocked", True), ("cancelled", True),
         (None, False)],
    )
    def test_post_hook_maps_status_onto_is_error(self, monkeypatch, status, expected):
        calls = self._capture(monkeypatch)
        ee.on_post_tool_call(
            tool_name="terminal",
            result="done",
            session_id=SESSION,
            tool_call_id="c1",
            status=status,
            telemetry_schema_version="hermes.observer.v1",
        )
        assert calls[0][0] == "tool_result"
        assert calls[0][2]["is_error"] is expected

    def test_post_hook_serializes_non_string_results(self, monkeypatch):
        calls = self._capture(monkeypatch)
        ee.on_post_tool_call(
            tool_name="t", result={"rows": 3}, session_id=SESSION, status="ok",
        )
        assert json.loads(calls[0][2]["output"]) == {"rows": 3}

    def test_nameless_tool_emits_nothing(self, monkeypatch):
        calls = self._capture(monkeypatch)
        ee.on_pre_tool_call(tool_name="", session_id=SESSION)
        ee.on_post_tool_call(tool_name="", session_id=SESSION)
        assert calls == []


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register_wires_both_tool_hooks(self):
        import hermes_band_platform

        ctx = MagicMock()
        hermes_band_platform.register(ctx)
        registered = {c.args[0] for c in ctx.register_hook.call_args_list}
        assert {"pre_tool_call", "post_tool_call"} <= registered

    def test_registration_survives_a_host_without_register_hook(self):
        class _OldHostCtx:
            pass

        # Must not raise: the platform still registers on an older host.
        ee.register_hooks(_OldHostCtx())


# ---------------------------------------------------------------------------
# Events never re-enter the agent's own context
# ---------------------------------------------------------------------------

class TestEventsStayOutOfContext:
    def test_rehydration_parser_drops_non_text_context_items(self):
        from hermes_band_platform.adapter import _seedable_text

        for message_type in ("tool_call", "tool_result"):
            item = SimpleNamespace(
                message_type=message_type,
                content=json.dumps({"name": "terminal", "args": {}}),
                sender_type="Agent",
                sender_id="a",
                sender_name="agent",
            )
            assert _seedable_text(item, []) is None

        text_item = SimpleNamespace(
            message_type="text",
            content="hello",
            sender_type="User",
            sender_id="u",
            sender_name="Alice",
        )
        assert _seedable_text(text_item, []) is not None

    async def test_inbound_dispatch_skips_non_text_messages(self):
        from hermes_band_platform.adapter import BandAdapter, _Inbound

        adapter = BandAdapter.__new__(BandAdapter)
        adapter._agent_id = "agent-1"
        adapter._sent_ids = set()
        adapter._seen_inbound_ids = set()
        adapter._normalize_inbound = lambda event: _Inbound(
            room_id=ROOM,
            msg_id="m1",
            content=json.dumps({"name": "terminal"}),
            sender_id="agent-1",
            sender_type="Agent",
            sender_name="agent",
            message_type="tool_call",
            payload=SimpleNamespace(),
        )
        # An event authored by another agent still must not wake this one.
        adapter._agent_id = "someone-else"
        assert await adapter._handle_message_created(SimpleNamespace()) is False
