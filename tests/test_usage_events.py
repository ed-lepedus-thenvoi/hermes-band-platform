"""Tests for per-turn token-usage events.

The band SDK stub (including ``band.core.types``' usage contract) is installed
by ``tests/conftest.py`` at collection time, BEFORE this module imports
``usage_events`` — so its lazy SDK binding sees the stub and
``USAGE_SDK_AVAILABLE`` stays True.
"""

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import hermes_band_platform.adapter as _band_mod
import hermes_band_platform.usage_events as ue


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clean_module_state(monkeypatch):
    """Isolate the module globals — both are process-wide by design.

    Leaves BAND_EMIT_USAGE UNSET, so every test starts from the shipped default
    (off). Tests that exercise emission opt in via the ``emit_all`` fixture,
    which keeps their dependence on non-default config visible.
    """
    ue._PENDING.clear()
    ue._ACTIVE_ADAPTERS.clear()
    ue._STARTUP_SCOPE = None
    monkeypatch.delenv("BAND_EMIT_USAGE", raising=False)
    yield
    ue._PENDING.clear()
    ue._ACTIVE_ADAPTERS.clear()
    ue._STARTUP_SCOPE = None


@pytest.fixture
def emit_all(monkeypatch):
    """Opt into emission. Required by anything asserting an event is produced."""
    monkeypatch.setenv("BAND_EMIT_USAGE", "all")


def _usage(inp=0, out=0, cache_read=0, cache_write=0, reasoning=0):
    """A CanonicalUsage summary in the exact shape the host's hook passes.

    Mirrors ``AIAgent._usage_summary_for_api_request_hook``: the dataclass
    fields minus ``raw_usage``, plus the two derived totals.
    """
    return {
        "input_tokens": inp,
        "output_tokens": out,
        "cache_read_tokens": cache_read,
        "cache_write_tokens": cache_write,
        "reasoning_tokens": reasoning,
        "request_count": 1,
        "prompt_tokens": inp + cache_read + cache_write,
        "total_tokens": inp + cache_read + cache_write + out,
    }


def _api_call(session_id="sid-1", turn_id="turn-1", platform="band", **kwargs):
    """Kwargs as the host fires them for ``post_api_request``."""
    payload = {
        "session_id": session_id,
        "turn_id": turn_id,
        "platform": platform,
        "model": "claude-sonnet-5",
        "provider": "anthropic",
        "usage": _usage(inp=100, out=20),
    }
    payload.update(kwargs)
    return payload


def _turn_end(session_id="sid-1", turn_id="turn-1", platform="band"):
    """Kwargs as the host fires them for ``post_llm_call``."""
    return {
        "session_id": session_id,
        "turn_id": turn_id,
        "platform": platform,
        "model": "claude-sonnet-5",
        "assistant_response": "done",
    }


_DEFAULT_LOOP = object()


class _FakeAdapter:
    """Stand-in exposing only what the hooks reach for on a live adapter."""

    def __init__(
        self, rooms=(), session_map=None, hub_room_id=None, loop=_DEFAULT_LOOP
    ):
        self._known_rooms = set(rooms)
        self._hub_room_id = hub_room_id
        self._link = MagicMock()
        self._link.rest.agent_api_events.create_agent_chat_event = AsyncMock()
        self._link_loop = (
            SimpleNamespace(is_running=lambda: True) if loop is _DEFAULT_LOOP else loop
        )
        entries = {
            f"agent:main:band:group:{room}": SimpleNamespace(session_id=sid)
            for room, sid in (session_map or {}).items()
        }
        self._session_store = SimpleNamespace(
            _ensure_loaded=lambda: None, _entries=entries
        )

    def _session_key_for(self, room_id):
        return f"agent:main:band:group:{room_id}"

    @property
    def sent(self):
        return self._link.rest.agent_api_events.create_agent_chat_event


def _scheduled(monkeypatch):
    """Capture ``_schedule_emit`` calls instead of touching an event loop."""
    calls = []
    monkeypatch.setattr(
        ue,
        "_schedule_emit",
        lambda sid, room, bucket: calls.append(
            (ue._current_adapter(sid, room), room, bucket)
        )
        or True,
    )
    return calls


# ---------------------------------------------------------------------------
# 1. Accumulation across a turn
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("emit_all")
class TestAccumulation:

    def test_calls_are_summed_into_one_turn_event(self, monkeypatch):
        adapter = _FakeAdapter(rooms=["room-a"], session_map={"room-a": "sid-1"})
        ue.track_adapter(adapter)
        calls = _scheduled(monkeypatch)

        ue.on_post_api_request(**_api_call(usage=_usage(inp=100, out=20, cache_read=5)))
        ue.on_post_api_request(**_api_call(usage=_usage(inp=200, out=30, cache_write=7)))
        ue.on_post_api_request(**_api_call(usage=_usage(inp=50, out=1)))
        assert calls == []  # nothing emitted mid-turn

        ue.on_post_llm_call(**_turn_end())

        assert len(calls) == 1
        _, room, bucket = calls[0]
        assert room == "room-a"
        assert bucket.api_calls == 3
        assert bucket.usage.input_tokens == 350
        assert bucket.usage.output_tokens == 51
        assert bucket.usage.cache_read_tokens == 5
        assert bucket.usage.cache_write_tokens == 7

    def test_turn_is_flushed_exactly_once(self, monkeypatch):
        adapter = _FakeAdapter(rooms=["room-a"], session_map={"room-a": "sid-1"})
        ue.track_adapter(adapter)
        calls = _scheduled(monkeypatch)

        ue.on_post_api_request(**_api_call())
        ue.on_post_llm_call(**_turn_end())
        ue.on_post_llm_call(**_turn_end())  # replayed / duplicate hook

        assert len(calls) == 1
        assert ue._PENDING == {}

    def test_concurrent_turns_do_not_bleed(self, monkeypatch):
        adapter = _FakeAdapter(
            rooms=["room-a", "room-b"],
            session_map={"room-a": "sid-a", "room-b": "sid-b"},
        )
        ue.track_adapter(adapter)
        calls = _scheduled(monkeypatch)

        ue.on_post_api_request(**_api_call(session_id="sid-a", usage=_usage(inp=10, out=1)))
        ue.on_post_api_request(**_api_call(session_id="sid-b", usage=_usage(inp=999, out=9)))
        ue.on_post_llm_call(**_turn_end(session_id="sid-a"))

        assert len(calls) == 1
        _, room, bucket = calls[0]
        assert room == "room-a"
        assert bucket.usage.input_tokens == 10

    def test_successive_turns_in_one_session_are_separate(self, monkeypatch):
        adapter = _FakeAdapter(rooms=["room-a"], session_map={"room-a": "sid-1"})
        ue.track_adapter(adapter)
        calls = _scheduled(monkeypatch)

        ue.on_post_api_request(**_api_call(turn_id="t1", usage=_usage(inp=10, out=1)))
        ue.on_post_llm_call(**_turn_end(turn_id="t1"))
        ue.on_post_api_request(**_api_call(turn_id="t2", usage=_usage(inp=20, out=2)))
        ue.on_post_llm_call(**_turn_end(turn_id="t2"))

        assert [b.usage.input_tokens for _, _, b in calls] == [10, 20]

    def test_reasoning_tokens_are_not_added_to_output(self, monkeypatch):
        """The host reports reasoning INSIDE output_tokens (see normalize_usage);
        folding it again would double-count on every reasoning model."""
        adapter = _FakeAdapter(rooms=["room-a"], session_map={"room-a": "sid-1"})
        ue.track_adapter(adapter)
        calls = _scheduled(monkeypatch)

        ue.on_post_api_request(**_api_call(usage=_usage(inp=10, out=500, reasoning=450)))
        ue.on_post_llm_call(**_turn_end())

        assert calls[0][2].usage.output_tokens == 500


# ---------------------------------------------------------------------------
# 2. What is skipped
# ---------------------------------------------------------------------------

@pytest.mark.usefixtures("emit_all")
class TestSkipped:
    """Emission is ON for these — so a missing event pins the stated reason,
    not the default."""

    def test_other_platforms_are_ignored(self, monkeypatch):
        adapter = _FakeAdapter(rooms=["room-a"], session_map={"room-a": "sid-1"})
        ue.track_adapter(adapter)
        calls = _scheduled(monkeypatch)

        ue.on_post_api_request(**_api_call(platform="telegram"))
        ue.on_post_llm_call(**_turn_end(platform="telegram"))

        assert ue._PENDING == {}
        assert calls == []

    def test_platform_enum_is_accepted(self, monkeypatch):
        adapter = _FakeAdapter(rooms=["room-a"], session_map={"room-a": "sid-1"})
        ue.track_adapter(adapter)
        calls = _scheduled(monkeypatch)

        ue.on_post_api_request(**_api_call(platform=SimpleNamespace(value="band")))
        ue.on_post_llm_call(**_turn_end())

        assert len(calls) == 1

    def test_call_without_usage_is_not_counted(self, monkeypatch):
        adapter = _FakeAdapter(rooms=["room-a"], session_map={"room-a": "sid-1"})
        ue.track_adapter(adapter)
        calls = _scheduled(monkeypatch)

        ue.on_post_api_request(**_api_call(usage=None))
        ue.on_post_api_request(**_api_call(usage=_usage(inp=10, out=1)))
        ue.on_post_llm_call(**_turn_end())

        assert calls[0][2].api_calls == 1

    def test_all_zero_usage_is_not_emitted(self, monkeypatch):
        """A zero-only record would read as a real measurement of zero spend."""
        adapter = _FakeAdapter(rooms=["room-a"], session_map={"room-a": "sid-1"})
        ue.track_adapter(adapter)
        calls = _scheduled(monkeypatch)

        ue.on_post_api_request(**_api_call(usage=_usage()))
        ue.on_post_llm_call(**_turn_end())

        assert calls == []

    def test_turn_without_calls_emits_nothing(self, monkeypatch):
        adapter = _FakeAdapter(rooms=["room-a"], session_map={"room-a": "sid-1"})
        ue.track_adapter(adapter)
        calls = _scheduled(monkeypatch)

        ue.on_post_llm_call(**_turn_end())

        assert calls == []

    def test_missing_session_id_is_skipped(self, monkeypatch):
        adapter = _FakeAdapter(rooms=["room-a"], session_map={"room-a": "sid-1"})
        ue.track_adapter(adapter)
        _scheduled(monkeypatch)

        ue.on_post_api_request(**_api_call(session_id=""))

        assert ue._PENDING == {}

    def test_unresolvable_room_emits_nothing(self, monkeypatch):
        adapter = _FakeAdapter(rooms=["room-a"], session_map={"room-a": "other-sid"})
        ue.track_adapter(adapter)
        calls = _scheduled(monkeypatch)

        ue.on_post_api_request(**_api_call())
        ue.on_post_llm_call(**_turn_end())

        assert calls == []


# ---------------------------------------------------------------------------
# 3. The unflushed-turn leak is bounded
# ---------------------------------------------------------------------------

class TestPendingBounded:
    """Caps backstop crashes and hosts without processing-complete hooks."""

    def test_pending_is_capped(self):
        for i in range(ue._PENDING_MAX + 25):
            ue.on_post_api_request(**_api_call(session_id=f"sid-{i}"))
        assert len(ue._PENDING) <= ue._PENDING_MAX

    def test_expired_buckets_are_dropped(self, monkeypatch):
        ue.on_post_api_request(**_api_call(session_id="old"))
        for bucket in ue._PENDING.values():
            bucket.created_at -= ue._PENDING_TTL_SECONDS + 1

        ue.on_post_api_request(**_api_call(session_id="fresh"))

        assert ("old", "turn-1") not in ue._PENDING
        assert ("fresh", "turn-1") in ue._PENDING


@pytest.mark.usefixtures("emit_all")
class TestTerminalLifecycle:
    """Success and incomplete terminal paths share exactly-once bucket pops."""

    def _adapter(self):
        adapter = _FakeAdapter(
            rooms=["room-a"], session_map={"room-a": "sid-1"}
        )
        ue.track_adapter(adapter)
        return adapter

    def test_failure_flushes_accumulated_usage(self, monkeypatch):
        adapter = self._adapter()
        calls = _scheduled(monkeypatch)
        ue.on_post_api_request(**_api_call(usage=_usage(inp=12, out=3)))

        ue.flush_incomplete_turn(adapter, "sid-1", "room-a")

        assert len(calls) == 1
        assert calls[0][2].usage.input_tokens == 12

    def test_interruption_flushes_accumulated_usage(self, monkeypatch):
        adapter = self._adapter()
        calls = _scheduled(monkeypatch)
        ue.on_post_api_request(**_api_call(usage=_usage(inp=8, out=2)))

        ue.flush_incomplete_turn(adapter, "sid-1", "room-a")

        assert len(calls) == 1

    def test_success_then_processing_completion_does_not_double_emit(self, monkeypatch):
        adapter = self._adapter()
        calls = _scheduled(monkeypatch)
        ue.on_post_api_request(**_api_call())

        ue.on_post_llm_call(**_turn_end())
        ue.flush_incomplete_turn(adapter, "sid-1", "room-a")

        assert len(calls) == 1

    def test_no_usage_finalization_emits_nothing(self, monkeypatch):
        adapter = self._adapter()
        calls = _scheduled(monkeypatch)

        ue.flush_incomplete_turn(adapter, "sid-1", "room-a")

        assert calls == []

    def test_duplicate_incomplete_finalization_emits_once(self, monkeypatch):
        adapter = self._adapter()
        calls = _scheduled(monkeypatch)
        ue.on_post_api_request(**_api_call())

        ue.flush_incomplete_turn(adapter, "sid-1", "room-a")
        ue.flush_incomplete_turn(adapter, "sid-1", "room-a")

        assert len(calls) == 1

    def test_per_user_sessions_in_one_room_are_not_merged(self, monkeypatch):
        adapter = _FakeAdapter(rooms=["room-a"])
        adapter._session_store._entries = {
            "agent:main:band:group:room-a:user-1": SimpleNamespace(
                session_id="sid-1"
            ),
            "agent:main:band:group:room-a:user-2": SimpleNamespace(
                session_id="sid-2"
            ),
        }
        ue.track_adapter(adapter)
        calls = _scheduled(monkeypatch)
        ue.on_post_api_request(
            **_api_call(session_id="sid-1", usage=_usage(inp=10, out=1))
        )
        ue.on_post_api_request(
            **_api_call(session_id="sid-2", usage=_usage(inp=20, out=2))
        )

        ue.flush_incomplete_turn(adapter, "sid-1", "room-a")

        assert [call[2].usage.input_tokens for call in calls] == [10]
        assert ("sid-2", "turn-1") in ue._PENDING


# ---------------------------------------------------------------------------
# 4. Room resolution
# ---------------------------------------------------------------------------

class TestRoomResolution:

    def test_resolves_against_the_real_session_store(self, monkeypatch, tmp_path):
        """The reverse lookup must agree with the store's own key derivation —
        the hooks carry session_id, only the session KEY holds the room."""
        from gateway.config import GatewayConfig, PlatformConfig
        from gateway.session import SessionSource, SessionStore
        from hermes_state import SessionDB

        monkeypatch.setenv("BAND_AGENT_ID", "a")
        monkeypatch.setenv("BAND_API_KEY", "k")
        adapter = _band_mod.BandAdapter(PlatformConfig(enabled=True, extra={}))

        store = SessionStore(tmp_path, GatewayConfig())
        store._db = SessionDB(db_path=tmp_path / "s.db")
        adapter._session_store = store
        adapter._known_rooms = {"room-x", "room-y"}

        entry = store.get_or_create_session(
            SessionSource(platform=adapter.platform, chat_id="room-x", chat_type="group")
        )

        assert ue._room_for_session(adapter, entry.session_id) == "room-x"
        assert ue._room_for_session(adapter, "no-such-session") is None

    def test_per_user_session_key_still_resolves_the_room(self):
        """With BAND_GROUP_SESSIONS_PER_USER the real key carries a trailing
        :<user_id> that _session_key_for (room-only) never produces."""
        adapter = _FakeAdapter(rooms=["room-a"])
        adapter._session_store._entries = {
            "agent:main:band:group:room-a:user-77": SimpleNamespace(session_id="sid-1")
        }

        assert ue._room_for_session(adapter, "sid-1") == "room-a"

    def test_missing_store_resolves_to_none(self):
        adapter = _FakeAdapter(rooms=["room-a"])
        adapter._session_store = None
        assert ue._room_for_session(adapter, "sid-1") is None


# ---------------------------------------------------------------------------
# 5. Scope policy
# ---------------------------------------------------------------------------

class TestScope:

    def test_default_is_off(self, monkeypatch):
        """Shipped default: nothing is emitted anywhere. Off until a consumer
        for band_usage exists — see _scope()."""
        assert ue._scope() == ue.SCOPE_OFF

    def test_default_emits_nothing_for_a_complete_turn(self, monkeypatch):
        """The whole pipeline runs (band platform, resolvable room, real
        tokens) and still produces no event, purely because of the default."""
        adapter = _FakeAdapter(
            rooms=["room-a"], session_map={"room-a": "sid-1"}, hub_room_id="hub-1"
        )
        ue.track_adapter(adapter)
        calls = _scheduled(monkeypatch)

        ue.on_post_api_request(**_api_call(usage=_usage(inp=100, out=20)))
        ue.on_post_llm_call(**_turn_end())

        assert calls == []

    @pytest.mark.asyncio
    async def test_default_posts_nothing_to_band(self):
        """End to end, with no _schedule_emit stub in the way: the Band REST
        events endpoint is never called."""
        loop = asyncio.get_running_loop()
        adapter = _FakeAdapter(
            rooms=["room-a"], session_map={"room-a": "sid-1"}, loop=loop
        )
        ue.track_adapter(adapter)

        ue.on_post_api_request(**_api_call(usage=_usage(inp=100, out=20)))
        ue.on_post_llm_call(**_turn_end())
        await asyncio.sleep(0)

        adapter.sent.assert_not_awaited()

    def test_all_scope_emits_in_the_room_the_turn_ran_in(self, monkeypatch, emit_all):
        adapter = _FakeAdapter(
            rooms=["room-a"], session_map={"room-a": "sid-1"}, hub_room_id="hub-1"
        )
        ue.track_adapter(adapter)
        calls = _scheduled(monkeypatch)

        ue.on_post_api_request(**_api_call())
        ue.on_post_llm_call(**_turn_end())

        assert [c[1] for c in calls] == ["room-a"]

    def test_hub_scope_suppresses_group_rooms(self, monkeypatch):
        monkeypatch.setenv("BAND_EMIT_USAGE", "hub")
        adapter = _FakeAdapter(
            rooms=["room-a"], session_map={"room-a": "sid-1"}, hub_room_id="hub-1"
        )
        ue.track_adapter(adapter)
        calls = _scheduled(monkeypatch)

        ue.on_post_api_request(**_api_call())
        ue.on_post_llm_call(**_turn_end())

        assert calls == []

    def test_hub_scope_still_emits_in_the_hub(self, monkeypatch):
        monkeypatch.setenv("BAND_EMIT_USAGE", "hub")
        adapter = _FakeAdapter(
            rooms=["hub-1"], session_map={"hub-1": "sid-1"}, hub_room_id="hub-1"
        )
        ue.track_adapter(adapter)
        calls = _scheduled(monkeypatch)

        ue.on_post_api_request(**_api_call())
        ue.on_post_llm_call(**_turn_end())

        assert [c[1] for c in calls] == ["hub-1"]

    @pytest.mark.parametrize("value", ["off", "false", "0", "no"])
    def test_off_suppresses_everything(self, monkeypatch, value):
        monkeypatch.setenv("BAND_EMIT_USAGE", value)
        adapter = _FakeAdapter(rooms=["room-a"], session_map={"room-a": "sid-1"})
        ue.track_adapter(adapter)
        calls = _scheduled(monkeypatch)

        ue.on_post_api_request(**_api_call())
        ue.on_post_llm_call(**_turn_end())

        assert calls == []

    def test_unknown_value_falls_back_to_off(self, monkeypatch):
        """A typo must not silently start emitting."""
        monkeypatch.setenv("BAND_EMIT_USAGE", "banana")
        assert ue._scope() == ue.SCOPE_OFF

    def test_empty_value_is_off(self, monkeypatch):
        monkeypatch.setenv("BAND_EMIT_USAGE", "   ")
        assert ue._scope() == ue.SCOPE_OFF

    @pytest.mark.parametrize("value", ["all", "ALL", "true", "1", "yes", "on"])
    def test_truthy_spellings_opt_in(self, monkeypatch, value):
        monkeypatch.setenv("BAND_EMIT_USAGE", value)
        assert ue._scope() == ue.SCOPE_ALL


# ---------------------------------------------------------------------------
# 6. The event on the wire
# ---------------------------------------------------------------------------

class TestEmit:

    def _bucket(self, **kwargs):
        from band.core.types import TurnUsage

        bucket = ue._Bucket(**kwargs)
        if "usage" not in kwargs:
            bucket.usage = TurnUsage(
                input_tokens=1200, output_tokens=340, cache_read_tokens=50,
                cache_write_tokens=10,
            )
        return bucket

    @pytest.mark.asyncio
    async def test_posts_a_usage_event_with_no_mentions(self):
        adapter = _FakeAdapter(
            rooms=["room-a"], session_map={"room-a": "sid-1"},
            loop=asyncio.get_running_loop(),
        )
        ue.track_adapter(adapter)
        bucket = self._bucket(api_calls=13, model="claude-sonnet-5", provider="anthropic")

        await ue._emit("sid-1", "room-a", bucket)

        adapter.sent.assert_awaited_once()
        kwargs = adapter.sent.await_args.kwargs
        assert kwargs["chat_id"] == "room-a"
        # Events are exempt from Band's >=1-mention rule; never attach any.
        assert "mentions" not in kwargs
        assert not hasattr(kwargs["event"], "mentions")

    @pytest.mark.asyncio
    async def test_rides_the_sdk_usage_event_type_and_metadata_key(self):
        """Not MessageType.USAGE: the backend's whitelist rejects it, so the
        SDK routes usage through an accepted event under a discriminator key."""
        from band.core.types import USAGE_EVENT_TYPE, USAGE_METADATA_KEY

        adapter = _FakeAdapter(
            rooms=["room-a"], session_map={"room-a": "sid-1"},
            loop=asyncio.get_running_loop(),
        )
        ue.track_adapter(adapter)
        await ue._emit("sid-1", "room-a", self._bucket(api_calls=2, model="m"))

        event = adapter.sent.await_args.kwargs["event"]
        assert event.message_type == USAGE_EVENT_TYPE
        assert event.message_type != "usage"
        assert event.metadata[USAGE_METADATA_KEY] == {
            "input_tokens": 1200,
            "output_tokens": 340,
            "cache_read_tokens": 50,
            "cache_write_tokens": 10,
        }
        assert event.metadata["api_calls"] == 2
        assert event.metadata["model"] == "m"

    @pytest.mark.asyncio
    async def test_content_carries_the_numbers(self):
        adapter = _FakeAdapter(
            rooms=["room-a"], session_map={"room-a": "sid-1"},
            loop=asyncio.get_running_loop(),
        )
        ue.track_adapter(adapter)
        await ue._emit(
            "sid-1", "room-a", self._bucket(api_calls=13, model="claude-sonnet-5")
        )

        content = adapter.sent.await_args.kwargs["event"].content
        assert content.startswith("Token usage: input=1200 output=340")
        assert "cache_read=50" in content
        assert "13 API calls" in content
        assert "claude-sonnet-5" in content

    def test_content_is_truncated_and_never_blank(self):
        bucket = self._bucket(api_calls=1, model="x" * (ue._EVENT_CONTENT_MAX_LENGTH * 2))
        content = ue.build_content(bucket)
        assert len(content) == ue._EVENT_CONTENT_MAX_LENGTH
        assert ue._EVENT_TRUNCATION_MARKER in content

    def test_metadata_omits_absent_model_and_provider(self):
        assert "model" not in ue.build_metadata(self._bucket(api_calls=1))
        assert "provider" not in ue.build_metadata(self._bucket(api_calls=1))

    @pytest.mark.asyncio
    async def test_a_failing_send_never_raises(self):
        adapter = _FakeAdapter(
            rooms=["room-a"], session_map={"room-a": "sid-1"},
            loop=asyncio.get_running_loop(),
        )
        ue.track_adapter(adapter)
        adapter.sent.side_effect = RuntimeError("Band is down")

        await ue._emit("sid-1", "room-a", self._bucket(api_calls=1))  # must not raise

    @pytest.mark.asyncio
    async def test_disconnected_adapter_is_a_noop(self):
        adapter = _FakeAdapter()
        sent = adapter.sent
        adapter._link = None

        ue.track_adapter(adapter)
        await ue._emit("sid-1", "room-a", self._bucket(api_calls=1))

        sent.assert_not_awaited()


class TestScheduling:

    def test_no_link_loop_means_no_emit(self):
        adapter = _FakeAdapter(
            rooms=["room-a"], session_map={"room-a": "sid-1"}, loop=None
        )
        ue.track_adapter(adapter)
        assert ue._schedule_emit("sid-1", "room-a", ue._Bucket()) is False

    @pytest.mark.asyncio
    async def test_emit_runs_on_the_links_own_loop(self):
        """The hooks fire on the agent's thread; the link's primitives are bound
        to the loop connect() ran on."""
        from band.core.types import TurnUsage

        loop = asyncio.get_running_loop()
        adapter = _FakeAdapter(
            rooms=["room-a"], session_map={"room-a": "sid-1"}, loop=loop
        )
        ue.track_adapter(adapter)
        bucket = ue._Bucket(usage=TurnUsage(input_tokens=5, output_tokens=1), api_calls=1)

        assert ue._schedule_emit("sid-1", "room-a", bucket) is True
        await asyncio.sleep(0)  # let the scheduled task run
        adapter.sent.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_reconnect_emits_only_through_current_live_adapter(self, emit_all):
        loop = asyncio.get_running_loop()
        old = _FakeAdapter(
            rooms=["room-a"], session_map={"room-a": "sid-1"}, loop=loop
        )
        new = _FakeAdapter(
            rooms=["room-a"], session_map={"room-a": "sid-1"}, loop=loop
        )
        ue.track_adapter(old)
        old_sent = old.sent
        old._link = None
        ue.untrack_adapter(old)
        ue.track_adapter(new)

        ue.on_post_api_request(**_api_call())
        ue.on_post_llm_call(**_turn_end())
        await asyncio.sleep(0)

        old_sent.assert_not_awaited()
        new.sent.assert_awaited_once()


# ---------------------------------------------------------------------------
# 7. Wiring
# ---------------------------------------------------------------------------

class TestRegistration:

    def test_registers_both_hooks_when_opted_in(self, emit_all):
        registered = {}
        ctx = SimpleNamespace(register_hook=lambda name, cb: registered.__setitem__(name, cb))

        assert ue.register_hooks(ctx) is True
        assert registered["post_api_request"] is ue.on_post_api_request
        assert registered["post_llm_call"] is ue.on_post_llm_call

    def test_default_registers_nothing(self):
        """The property that matters most now that off is the default:
        registering post_api_request makes the host build a sanitized response
        payload for EVERY API call, and nobody should pay that for a feature
        they did not opt into."""
        registered = {}
        ctx = SimpleNamespace(register_hook=lambda name, cb: registered.__setitem__(name, cb))

        assert ue.register_hooks(ctx) is False
        assert registered == {}

    def test_explicit_off_registers_nothing(self, monkeypatch):
        monkeypatch.setenv("BAND_EMIT_USAGE", "off")
        registered = {}
        ctx = SimpleNamespace(register_hook=lambda name, cb: registered.__setitem__(name, cb))

        assert ue.register_hooks(ctx) is False
        assert registered == {}

    def test_enabling_after_startup_requires_new_registration(self, monkeypatch):
        registered = {}
        ctx = SimpleNamespace(
            register_hook=lambda name, cb: registered.__setitem__(name, cb)
        )
        assert ue.register_hooks(ctx) is False

        monkeypatch.setenv("BAND_EMIT_USAGE", "all")

        # The already-started plugin has no hooks to observe the env change,
        # and another discovery pass in the same process retains startup scope.
        assert registered == {}
        restarted = {}
        new_ctx = SimpleNamespace(
            register_hook=lambda name, cb: restarted.__setitem__(name, cb)
        )
        assert ue.register_hooks(new_ctx) is False
        assert restarted == {}

        # A gateway restart creates fresh module state and captures the value.
        ue._STARTUP_SCOPE = None
        assert ue.register_hooks(new_ctx) is True
        assert set(restarted) == {"post_api_request", "post_llm_call"}

    def test_plugin_registration_wires_no_usage_hooks_by_default(self, monkeypatch):
        """Through the real plugin entry point, not just register_hooks().

        Asserts about the USAGE hooks specifically, not that the plugin registers
        no hooks at all: execution events register ``pre_tool_call`` /
        ``post_tool_call`` unconditionally, so a blanket ``assert_not_called``
        passes on this module's own branch and fails the moment the two slices
        are composed. Assert only about your own slice's effects.
        """
        monkeypatch.setenv("BAND_AGENT_ID", "a")
        monkeypatch.setenv("BAND_API_KEY", "k")
        ctx = MagicMock()

        _band_mod.register(ctx)

        wired = {call.args[0] for call in ctx.register_hook.call_args_list if call.args}
        assert "post_api_request" not in wired
        assert "post_llm_call" not in wired

    def test_older_host_without_register_hook(self, emit_all):
        assert ue.register_hooks(SimpleNamespace()) is False

    def test_sdk_without_the_usage_contract(self, monkeypatch, emit_all):
        monkeypatch.setattr(ue, "_ensure_sdk_bindings", lambda: False)
        ctx = SimpleNamespace(register_hook=MagicMock())

        assert ue.register_hooks(ctx) is False
        ctx.register_hook.assert_not_called()

    def test_sdk_binding_retries_after_early_discovery_miss(self, monkeypatch, emit_all):
        monkeypatch.setattr(ue, "USAGE_SDK_AVAILABLE", False)
        monkeypatch.setattr(ue, "TurnUsage", None)
        ctx = SimpleNamespace(register_hook=MagicMock())

        assert ue.register_hooks(ctx) is True
        assert ue.USAGE_SDK_AVAILABLE is True
        assert ue.TurnUsage is not None

    def test_adapter_construction_is_not_tracked_until_connected(self, monkeypatch):
        monkeypatch.setenv("BAND_AGENT_ID", "a")
        monkeypatch.setenv("BAND_API_KEY", "k")
        from gateway.config import PlatformConfig

        adapter = _band_mod.BandAdapter(PlatformConfig(enabled=True, extra={}))

        assert adapter not in ue._adapters()


class TestNoSelfFeedback:
    """Emitted events must never come back to the agent as its own input."""

    def test_rehydration_drops_non_text_context_items(self):
        from band.core.types import USAGE_EVENT_TYPE

        usage_item = SimpleNamespace(
            message_type=USAGE_EVENT_TYPE,
            content="Token usage: input=1200 output=340",
            sender_type="Agent",
            sender_id="agent-1",
            sender_name="bot",
        )
        text_item = SimpleNamespace(
            message_type="text",
            content="hello",
            sender_type="User",
            sender_id="user-1",
            sender_name="Alice",
        )

        assert _band_mod._seedable_text(usage_item, []) is None
        assert _band_mod._seedable_text(text_item, []) is not None


# ---------------------------------------------------------------------------
# 9. Observability
#
# Usage is the quietest of the three event paths: it fires once per turn and
# has a dozen legitimate reasons to produce nothing. Each of them has to be
# distinguishable from a broken one, or "no usage events in this room" is
# unanswerable.
# ---------------------------------------------------------------------------

_UE_LOGGER = "hermes_band_platform.usage_events"


class TestUsageLogging:

    def _bucket(self, **kwargs):
        from band.core.types import TurnUsage

        bucket = ue._Bucket(**kwargs)
        bucket.usage = TurnUsage(input_tokens=1200, output_tokens=340)
        return bucket

    @pytest.mark.asyncio
    async def test_emission_is_traceable_at_debug(self, caplog):
        adapter = _FakeAdapter(
            rooms=["room-a"], session_map={"room-a": "sid-1"},
            loop=asyncio.get_running_loop(),
        )
        ue.track_adapter(adapter)
        with caplog.at_level(logging.DEBUG, logger=_UE_LOGGER):
            await ue._emit("sid-1", "room-a", self._bucket(api_calls=13))

        emitted = [r for r in caplog.records if "Emitted usage event" in r.getMessage()]
        assert len(emitted) == 1
        assert emitted[0].levelno == logging.DEBUG
        message = emitted[0].getMessage()
        assert "room-a" in message
        assert "13 API call(s)" in message

    @pytest.mark.asyncio
    async def test_failed_post_warns_with_the_room(self, caplog):
        adapter = _FakeAdapter(
            rooms=["room-a"], session_map={"room-a": "sid-1"},
            loop=asyncio.get_running_loop(),
        )
        ue.track_adapter(adapter)
        adapter._link.rest.agent_api_events.create_agent_chat_event = AsyncMock(
            side_effect=RuntimeError("502 Bad Gateway")
        )
        with caplog.at_level(logging.DEBUG, logger=_UE_LOGGER):
            await ue._emit("sid-1", "room-a", self._bucket(api_calls=1))

        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "room-a" in warnings[0].getMessage()
        assert "502 Bad Gateway" in warnings[0].getMessage()

    def test_non_band_turn_is_logged_as_filtered(self, caplog, monkeypatch, emit_all):
        """No adapter owns the session: a CLI turn, not a broken resolution.

        Needs ``emit_all``: emission is off by default now, and the off path
        short-circuits before the room resolution this asserts on.
        """
        _scheduled(monkeypatch)
        ue.on_post_api_request(**_api_call())
        with caplog.at_level(logging.DEBUG, logger=_UE_LOGGER):
            ue.on_post_llm_call(**_turn_end())

        assert any(
            "not a Band turn" in r.getMessage() and r.levelno == logging.DEBUG
            for r in caplog.records
        )

    def test_hub_scope_suppression_says_it_was_a_policy_choice(
        self, caplog, monkeypatch
    ):
        monkeypatch.setenv("BAND_EMIT_USAGE", "hub")
        adapter = _FakeAdapter(
            rooms=("room-group",), session_map={"room-group": "sid-1"},
            hub_room_id="room-hub",
        )
        ue.track_adapter(adapter)
        calls = _scheduled(monkeypatch)

        ue.on_post_api_request(**_api_call())
        with caplog.at_level(logging.DEBUG, logger=_UE_LOGGER):
            ue.on_post_llm_call(**_turn_end())

        assert calls == []
        assert any("BAND_EMIT_USAGE=hub" in r.getMessage() for r in caplog.records)

    def test_a_turn_that_reported_no_tokens_says_so(self, caplog, monkeypatch):
        _scheduled(monkeypatch)
        ue.on_post_api_request(**_api_call(usage=None))
        with caplog.at_level(logging.DEBUG, logger=_UE_LOGGER):
            ue.on_post_llm_call(**_turn_end())

        assert any(
            "no accumulated usage" in r.getMessage() for r in caplog.records
        )

    def test_the_per_call_platform_filter_stays_silent(self, caplog):
        """It fires on every API call of every platform — a tight loop."""
        with caplog.at_level(logging.DEBUG, logger=_UE_LOGGER):
            for _ in range(20):
                ue.on_post_api_request(**_api_call(platform="telegram"))
        assert caplog.records == []

    @pytest.mark.asyncio
    async def test_no_link_is_logged_before_the_event_is_dropped(self, caplog):
        adapter = _FakeAdapter()
        adapter._link = None
        ue.track_adapter(adapter)
        with caplog.at_level(logging.DEBUG, logger=_UE_LOGGER):
            await ue._emit("sid-1", "room-a", self._bucket(api_calls=1))

        assert any("no live adapter" in r.getMessage() for r in caplog.records)
