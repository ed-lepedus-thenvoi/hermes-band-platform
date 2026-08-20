"""Tests for the Band action tools (``hermes_band_platform/tools.py``).

These cover the tools pass (Build Plan Step 6): the seven ``band_*`` action
tools, the room-resolution matrix, and the owner gate.

Strategy
--------
The band SDK *is* importable here (the request types tools.py constructs —
``ChatRoomRequest`` / ``ParticipantRequest`` / ``ChatMessageRequest`` /
``ChatMessageRequestMentionsItem`` — bind at module import).  Each tool's REST
calls are mocked by patching ``tools._rest`` to return a fake async REST client
(``_make_rest()``), so no network / live gateway is involved.

Session context (``HERMES_SESSION_PLATFORM`` / ``HERMES_SESSION_CHAT_ID`` /
``HERMES_SESSION_USER_ID``) is driven with the real
``gateway.session_context.set_session_vars`` / ``clear_session_vars`` helpers —
the same mechanism the gateway uses — so the room-resolution and owner-gate
tests exercise the production read path (``get_session_env``).
"""

from __future__ import annotations

import asyncio
import json
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.session_context import set_session_vars, clear_session_vars
from hermes_band_platform import tools as band_tools


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

def _make_rest() -> MagicMock:
    """Build a fake async REST client mirroring the resources tools.py drives.

    Every SDK method the tools call is an ``AsyncMock`` with a sensible default
    return value; individual tests override ``return_value`` / ``side_effect``
    as needed.
    """
    rest = MagicMock()

    # agent_api_chats.create_agent_chat -> .data.id
    rest.agent_api_chats.create_agent_chat = AsyncMock(
        return_value=SimpleNamespace(data=SimpleNamespace(id="room-new-001"))
    )
    # agent_api_chats.list_agent_chats -> .data[] + .metadata.total_pages
    rest.agent_api_chats.list_agent_chats = AsyncMock(
        return_value=SimpleNamespace(data=[], metadata=SimpleNamespace(total_pages=1))
    )

    # agent_api_participants
    rest.agent_api_participants.add_agent_chat_participant = AsyncMock(
        return_value=SimpleNamespace(data=SimpleNamespace(id="part-001"))
    )
    rest.agent_api_participants.remove_agent_chat_participant = AsyncMock(
        return_value=SimpleNamespace(data=None)
    )
    rest.agent_api_participants.list_agent_chat_participants = AsyncMock(
        return_value=SimpleNamespace(data=[])
    )

    # agent_api_messages.create_agent_chat_message -> .data.id
    rest.agent_api_messages.create_agent_chat_message = AsyncMock(
        return_value=SimpleNamespace(data=SimpleNamespace(id="msg-001"))
    )

    # agent_api_peers.list_agent_peers -> .data[] + .metadata.total_pages
    rest.agent_api_peers.list_agent_peers = AsyncMock(
        return_value=SimpleNamespace(data=[], metadata=SimpleNamespace(total_pages=1))
    )
    return rest


def _peer(pid, handle=None, name=None, ptype="User"):
    return SimpleNamespace(id=pid, handle=handle, name=name, type=ptype)


def _patch_rest(rest):
    """Patch ``tools._rest`` to return the given fake rest client."""
    return patch.object(band_tools, "_rest", AsyncMock(return_value=rest))


def _patch_agent_id(agent_id="agent-self"):
    """Patch ``tools._agent_id_or_none`` so mention-building self-excludes."""
    return patch.object(band_tools, "_agent_id_or_none", AsyncMock(return_value=agent_id))


@pytest.fixture
def owner_session():
    """A Band session whose user is on the owner allowlist.

    Sets HERMES_SESSION_PLATFORM=band, CHAT_ID=room-current, USER_ID=u-owner
    and BAND_TOOL_OWNERS=band:u-owner. Cleans up the context on teardown.
    """
    tokens = set_session_vars(
        platform="band",
        chat_id="room-current",
        user_id="u-owner",
    )
    try:
        yield
    finally:
        clear_session_vars(tokens)


@pytest.fixture(autouse=True)
def _owner_env(monkeypatch):
    """Default owner allowlist for the Band-session owner fixture."""
    monkeypatch.setenv("BAND_TOOL_OWNERS", "band:u-owner")
    yield


def _parse(result: str) -> dict:
    assert isinstance(result, str)
    return json.loads(result)


@pytest.fixture
def other_loop():
    """A second event loop, really running, in another thread.

    Stands in for the gateway's loop while the test body runs on the agent's.
    A mismatch cannot be faked with a non-running loop object: the condition
    under test is two live loops in one process, which is exactly the shape a
    gateway-hosted tool call has.
    """
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def _run():
        asyncio.set_event_loop(loop)
        loop.call_soon(ready.set)
        loop.run_forever()

    thread = threading.Thread(target=_run, daemon=True, name="other-loop")
    thread.start()
    assert ready.wait(timeout=5), "second loop never started"
    try:
        yield loop
    finally:
        loop.call_soon_threadsafe(loop.stop)
        thread.join(timeout=5)
        loop.close()


# ---------------------------------------------------------------------------
# 1. band_create_room — bare / +person / +person+message
# ---------------------------------------------------------------------------

class TestCreateRoom:

    @pytest.mark.asyncio
    async def test_bare_create_only_creates_room(self, owner_session):
        rest = _make_rest()
        with _patch_rest(rest):
            out = _parse(await band_tools._handle_create_room({}))
        assert out["success"] is True
        assert out["room_id"] == "room-new-001"
        assert out["added"] == []
        assert "sent" not in out
        rest.agent_api_chats.create_agent_chat.assert_awaited_once()
        rest.agent_api_participants.add_agent_chat_participant.assert_not_called()
        rest.agent_api_messages.create_agent_chat_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_with_person_resolves_and_adds(self, owner_session):
        rest = _make_rest()
        rest.agent_api_peers.list_agent_peers = AsyncMock(
            return_value=SimpleNamespace(
                data=[_peer("sarah-uuid", handle="sarah", name="Sarah")],
                metadata=SimpleNamespace(total_pages=1),
            )
        )
        with _patch_rest(rest):
            out = _parse(await band_tools._handle_create_room({"person": "Sarah"}))
        assert out["success"] is True
        assert out["room_id"] == "room-new-001"
        assert out["added"] == [{"id": "sarah-uuid", "handle": "sarah", "name": "Sarah"}]
        assert "sent" not in out
        # add called on the freshly-created room with a ParticipantRequest
        rest.agent_api_participants.add_agent_chat_participant.assert_awaited_once()
        call = rest.agent_api_participants.add_agent_chat_participant.await_args
        assert call.args[0] == "room-new-001"
        assert call.kwargs["participant"].participant_id == "sarah-uuid"
        rest.agent_api_messages.create_agent_chat_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_create_with_person_and_message_sends(self, owner_session):
        rest = _make_rest()
        rest.agent_api_peers.list_agent_peers = AsyncMock(
            return_value=SimpleNamespace(
                data=[_peer("sarah-uuid", handle="sarah", name="Sarah")],
                metadata=SimpleNamespace(total_pages=1),
            )
        )
        with _patch_rest(rest):
            out = _parse(
                await band_tools._handle_create_room(
                    {"person": "Sarah", "message": "the report's ready"}
                )
            )
        assert out["success"] is True
        assert out["sent"] == "msg-001"
        # message sent into the created room, mentioning the resolved person
        rest.agent_api_messages.create_agent_chat_message.assert_awaited_once()
        call = rest.agent_api_messages.create_agent_chat_message.await_args
        assert call.args[0] == "room-new-001"
        msg = call.kwargs["message"]
        assert msg.content == "the report's ready"
        assert len(msg.mentions) == 1
        assert msg.mentions[0].id == "sarah-uuid"

    @pytest.mark.asyncio
    async def test_create_with_unmatched_person_warns_but_creates(self, owner_session):
        rest = _make_rest()  # peers list empty -> no match
        with _patch_rest(rest):
            out = _parse(await band_tools._handle_create_room({"person": "Nobody"}))
        assert out["success"] is True
        assert out["room_id"] == "room-new-001"
        assert "warning" in out
        rest.agent_api_participants.add_agent_chat_participant.assert_not_called()

    @pytest.mark.asyncio
    async def test_message_without_person_errors(self, owner_session):
        rest = _make_rest()
        with _patch_rest(rest):
            out = _parse(await band_tools._handle_create_room({"message": "hi"}))
        assert "error" in out
        rest.agent_api_messages.create_agent_chat_message.assert_not_called()


# ---------------------------------------------------------------------------
# 2. band_find_room (read-only)
# ---------------------------------------------------------------------------

class TestFindRoom:

    @pytest.mark.asyncio
    async def test_matches_on_title(self, owner_session):
        rest = _make_rest()
        rest.agent_api_chats.list_agent_chats = AsyncMock(
            return_value=SimpleNamespace(
                data=[
                    SimpleNamespace(id="room-a", title="Design Team"),
                    SimpleNamespace(id="room-b", title="Random"),
                ],
                metadata=SimpleNamespace(total_pages=1),
            )
        )
        with _patch_rest(rest):
            out = _parse(await band_tools._handle_find_room({"query": "design"}))
        assert out["success"] is True
        assert out["rooms"] == [{"room_id": "room-a", "title": "Design Team"}]
        assert out["truncated"] is False

    @pytest.mark.asyncio
    async def test_empty_query_lists_all(self, owner_session):
        rest = _make_rest()
        rest.agent_api_chats.list_agent_chats = AsyncMock(
            return_value=SimpleNamespace(
                data=[SimpleNamespace(id="room-a", title="A")],
                metadata=SimpleNamespace(total_pages=1),
            )
        )
        with _patch_rest(rest):
            out = _parse(await band_tools._handle_find_room({}))
        assert out["success"] is True
        assert {r["room_id"] for r in out["rooms"]} == {"room-a"}


# ---------------------------------------------------------------------------
# 3. band_find_contact (read-only)
# ---------------------------------------------------------------------------

class TestFindContact:

    @pytest.mark.asyncio
    async def test_matches_on_handle_and_name(self, owner_session):
        rest = _make_rest()
        rest.agent_api_peers.list_agent_peers = AsyncMock(
            return_value=SimpleNamespace(
                data=[
                    _peer("u-1", handle="sarahj", name="Sarah Jones"),
                    _peer("u-2", handle="bob", name="Bob"),
                ],
                metadata=SimpleNamespace(total_pages=1),
            )
        )
        with _patch_rest(rest):
            out = _parse(await band_tools._handle_find_contact({"query": "sarah"}))
        assert out["success"] is True
        assert out["contacts"] == [
            {"id": "u-1", "handle": "sarahj", "name": "Sarah Jones"}
        ]

    @pytest.mark.asyncio
    async def test_missing_query_errors(self, owner_session):
        rest = _make_rest()
        with _patch_rest(rest):
            out = _parse(await band_tools._handle_find_contact({}))
        assert "error" in out


# ---------------------------------------------------------------------------
# 4. band_send_message — mentions built, >=1 enforced, chunking
# ---------------------------------------------------------------------------

class TestSendMessage:

    @pytest.mark.asyncio
    async def test_builds_mentions_from_participants(self, owner_session):
        rest = _make_rest()
        rest.agent_api_participants.list_agent_chat_participants = AsyncMock(
            return_value=SimpleNamespace(
                data=[
                    _peer("agent-self", handle="bot", ptype="Agent"),
                    _peer("u-human", handle="alice", name="Alice"),
                ]
            )
        )
        with _patch_rest(rest), _patch_agent_id("agent-self"):
            out = _parse(
                await band_tools._handle_send_message({"content": "hello"})
            )
        assert out["success"] is True
        assert out["room_id"] == "room-current"  # from band session context
        assert out["message_id"] == "msg-001"
        call = rest.agent_api_messages.create_agent_chat_message.await_args
        mentions = call.kwargs["message"].mentions
        # The agent is self-excluded; only the human is mentioned.
        assert [m.id for m in mentions] == ["u-human"]

    @pytest.mark.asyncio
    async def test_explicit_mention_ids_used(self, owner_session):
        rest = _make_rest()
        rest.agent_api_participants.list_agent_chat_participants = AsyncMock(
            return_value=SimpleNamespace(
                data=[_peer("u-x", handle="x"), _peer("u-y", handle="y")]
            )
        )
        with _patch_rest(rest), _patch_agent_id("agent-self"):
            out = _parse(
                await band_tools._handle_send_message(
                    {"content": "hi", "mention_ids": ["u-y"]}
                )
            )
        assert out["success"] is True
        call = rest.agent_api_messages.create_agent_chat_message.await_args
        mentions = call.kwargs["message"].mentions
        assert [m.id for m in mentions] == ["u-y"]
        # handle resolved from the participant list
        assert mentions[0].handle == "y"

    @pytest.mark.asyncio
    async def test_self_in_mention_ids_sends_as_a_reference(self, owner_session):
        """A model that names its own id must not lose the whole message.

        Band answers a self @mention with 422 cannot_mention_self and drops the
        entire message, so the self entry is demoted to a reference and the real
        recipient still receives it.
        """
        rest = _make_rest()
        rest.agent_api_participants.list_agent_chat_participants = AsyncMock(
            return_value=SimpleNamespace(
                data=[
                    _peer("agent-self", handle="bot", ptype="Agent"),
                    _peer("u-y", handle="y"),
                ]
            )
        )
        with _patch_rest(rest), _patch_agent_id("agent-self"):
            out = _parse(
                await band_tools._handle_send_message(
                    {"content": "hi", "mention_ids": ["u-y", "agent-self"]}
                )
            )
        assert out["success"] is True
        call = rest.agent_api_messages.create_agent_chat_message.await_args
        mentions = call.kwargs["message"].mentions
        assert [m.id for m in mentions] == ["u-y", "agent-self"]
        assert getattr(mentions[0], "kind", None) is None   # normal recipient
        assert mentions[1].kind == "reference"               # us, demoted

    @pytest.mark.asyncio
    async def test_only_self_in_mention_ids_errors_before_sending(
        self, owner_session
    ):
        """Self alone leaves nobody to deliver to — say so, don't 422."""
        rest = _make_rest()
        rest.agent_api_participants.list_agent_chat_participants = AsyncMock(
            return_value=SimpleNamespace(
                data=[
                    _peer("agent-self", handle="bot", ptype="Agent"),
                    _peer("u-y", handle="y"),
                ]
            )
        )
        with _patch_rest(rest), _patch_agent_id("agent-self"):
            out = _parse(
                await band_tools._handle_send_message(
                    {"content": "hi", "mention_ids": ["agent-self"]}
                )
            )
        assert "error" in out
        assert "cannot @mention itself" in out["error"]
        rest.agent_api_messages.create_agent_chat_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_mentionable_recipient_errors(self, owner_session):
        rest = _make_rest()
        # Only the agent in the room -> nothing to mention after self-exclusion.
        rest.agent_api_participants.list_agent_chat_participants = AsyncMock(
            return_value=SimpleNamespace(
                data=[_peer("agent-self", handle="bot", ptype="Agent")]
            )
        )
        with _patch_rest(rest), _patch_agent_id("agent-self"):
            out = _parse(await band_tools._handle_send_message({"content": "hello"}))
        assert "error" in out
        assert "mention" in out["error"].lower()
        rest.agent_api_messages.create_agent_chat_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_empty_content_errors(self, owner_session):
        rest = _make_rest()
        with _patch_rest(rest):
            out = _parse(await band_tools._handle_send_message({"content": "   "}))
        assert "error" in out
        rest.agent_api_messages.create_agent_chat_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_long_content_is_chunked(self, owner_session):
        rest = _make_rest()
        rest.agent_api_participants.list_agent_chat_participants = AsyncMock(
            return_value=SimpleNamespace(data=[_peer("u-human", handle="alice")])
        )
        with _patch_rest(rest), _patch_agent_id("agent-self"):
            out = _parse(
                await band_tools._handle_send_message({"content": "x" * 9000})
            )
        assert out["success"] is True
        # >4000 chars -> at least two create_agent_chat_message calls.
        assert rest.agent_api_messages.create_agent_chat_message.await_count >= 2
        # Every chunk repeats the mandatory mention.
        for call in rest.agent_api_messages.create_agent_chat_message.await_args_list:
            assert len(call.kwargs["message"].mentions) >= 1


# ---------------------------------------------------------------------------
# 5. band_add_participant / band_remove_participant / band_get_participants
# ---------------------------------------------------------------------------

class TestParticipantTools:

    @pytest.mark.asyncio
    async def test_add_participant(self, owner_session):
        rest = _make_rest()
        with _patch_rest(rest):
            out = _parse(
                await band_tools._handle_add_participant(
                    {"participant_id": "u-new", "role": "admin"}
                )
            )
        assert out["success"] is True
        assert out["room_id"] == "room-current"
        assert out["participant_id"] == "u-new"
        assert out["role"] == "admin"
        call = rest.agent_api_participants.add_agent_chat_participant.await_args
        assert call.args[0] == "room-current"
        assert call.kwargs["participant"].participant_id == "u-new"
        assert call.kwargs["participant"].role == "admin"

    @pytest.mark.asyncio
    async def test_add_participant_requires_id(self, owner_session):
        rest = _make_rest()
        with _patch_rest(rest):
            out = _parse(await band_tools._handle_add_participant({}))
        assert "error" in out
        rest.agent_api_participants.add_agent_chat_participant.assert_not_called()

    @pytest.mark.asyncio
    async def test_remove_participant_positional_id(self, owner_session):
        rest = _make_rest()
        with _patch_rest(rest):
            out = _parse(
                await band_tools._handle_remove_participant({"participant_id": "u-bye"})
            )
        assert out["success"] is True
        assert out["participant_id"] == "u-bye"
        call = rest.agent_api_participants.remove_agent_chat_participant.await_args
        # remove takes (chat_id, participant_id) positionally
        assert call.args[0] == "room-current"
        assert call.args[1] == "u-bye"

    @pytest.mark.asyncio
    async def test_get_participants_lists_and_is_readonly(self, owner_session):
        rest = _make_rest()
        rest.agent_api_participants.list_agent_chat_participants = AsyncMock(
            return_value=SimpleNamespace(
                data=[
                    _peer("agent-self", handle="bot", name="Bot", ptype="Agent"),
                    _peer("u-1", handle="alice", name="Alice"),
                ]
            )
        )
        with _patch_rest(rest):
            out = _parse(await band_tools._handle_get_participants({}))
        assert out["success"] is True
        assert out["room_id"] == "room-current"
        assert {p["id"] for p in out["participants"]} == {"agent-self", "u-1"}
        assert out["truncated"] is False


# ---------------------------------------------------------------------------
# 6. Room-resolution matrix
# ---------------------------------------------------------------------------

class TestRoomResolution:

    @pytest.mark.asyncio
    async def test_band_session_defaults_to_chat_id(self, monkeypatch):
        monkeypatch.setenv("BAND_TOOL_OWNERS", "band:u1")
        tokens = set_session_vars(platform="band", chat_id="band-room-7", user_id="u1")
        rest = _make_rest()
        try:
            with _patch_rest(rest):
                out = _parse(
                    await band_tools._handle_add_participant({"participant_id": "p"})
                )
        finally:
            clear_session_vars(tokens)
        assert out["success"] is True
        assert out["room_id"] == "band-room-7"

    @pytest.mark.asyncio
    async def test_telegram_session_requires_explicit_room_id(self, monkeypatch):
        monkeypatch.setenv("BAND_TOOL_OWNERS", "telegram:u1")
        tokens = set_session_vars(platform="telegram", chat_id="tg-chat-1", user_id="u1")
        rest = _make_rest()
        try:
            with _patch_rest(rest):
                out = _parse(
                    await band_tools._handle_add_participant({"participant_id": "p"})
                )
        finally:
            clear_session_vars(tokens)
        # No band room in this conversation -> error, no add attempted.
        assert "error" in out
        assert "room_id required" in out["error"]
        rest.agent_api_participants.add_agent_chat_participant.assert_not_called()

    @pytest.mark.asyncio
    async def test_telegram_session_with_explicit_room_id_works(self, monkeypatch):
        monkeypatch.setenv("BAND_TOOL_OWNERS", "telegram:u1")
        tokens = set_session_vars(platform="telegram", chat_id="tg-chat-1", user_id="u1")
        rest = _make_rest()
        try:
            with _patch_rest(rest):
                out = _parse(
                    await band_tools._handle_add_participant(
                        {"participant_id": "p", "room_id": "explicit-room"}
                    )
                )
        finally:
            clear_session_vars(tokens)
        assert out["success"] is True
        assert out["room_id"] == "explicit-room"

    @pytest.mark.asyncio
    async def test_explicit_room_id_overrides_band_session(self, monkeypatch):
        monkeypatch.setenv("BAND_TOOL_OWNERS", "band:u1")
        tokens = set_session_vars(platform="band", chat_id="session-room", user_id="u1")
        rest = _make_rest()
        try:
            with _patch_rest(rest):
                out = _parse(
                    await band_tools._handle_add_participant(
                        {"participant_id": "p", "room_id": "override-room"}
                    )
                )
        finally:
            clear_session_vars(tokens)
        assert out["success"] is True
        assert out["room_id"] == "override-room"

    def test_resolve_room_no_context_raises(self, monkeypatch):
        # Outside any session, with no explicit room_id -> _ToolError.
        tokens = set_session_vars(platform="", chat_id="", user_id="")
        try:
            with pytest.raises(band_tools._ToolError):
                band_tools._resolve_room({})
        finally:
            clear_session_vars(tokens)


class TestSendToOwnerFallback:
    """`band_send_message` reaches the owner's hub when no room is in context.

    This is what makes "send a message to me" work from a non-Band session
    (CLI / web / another platform), where `_resolve_room` finds no current room.
    """

    @pytest.mark.asyncio
    async def test_non_band_session_falls_back_to_home_and_mentions_owner(self, monkeypatch):
        monkeypatch.setenv("BAND_TOOL_OWNERS", "telegram:u-owner")
        monkeypatch.setenv("BAND_HOME_ROOM", "hub-home")
        monkeypatch.setenv("BAND_OWNER_ID", "owner-uuid")
        tokens = set_session_vars(platform="telegram", chat_id="tg-1", user_id="u-owner")
        rest = _make_rest()
        try:
            with _patch_rest(rest), _patch_agent_id("agent-self"):
                out = _parse(await band_tools._handle_send_message({"content": "ping"}))
        finally:
            clear_session_vars(tokens)
        assert out["success"] is True
        assert out["room_id"] == "hub-home"
        call = rest.agent_api_messages.create_agent_chat_message.await_args
        mentions = call.kwargs["message"].mentions
        assert [m.id for m in mentions] == ["owner-uuid"]  # owner pinged by default

    @pytest.mark.asyncio
    async def test_falls_back_to_hub_room_when_no_home(self, monkeypatch):
        # Only BAND_HUB_ROOM set (no BAND_HOME_ROOM) -> still resolves.
        monkeypatch.setenv("BAND_TOOL_OWNERS", "telegram:u-owner")
        monkeypatch.delenv("BAND_HOME_ROOM", raising=False)
        monkeypatch.setenv("BAND_HUB_ROOM", "hub-1")
        monkeypatch.setenv("BAND_OWNER_ID", "owner-uuid")
        tokens = set_session_vars(platform="telegram", chat_id="tg-1", user_id="u-owner")
        rest = _make_rest()
        try:
            with _patch_rest(rest), _patch_agent_id("agent-self"):
                out = _parse(await band_tools._handle_send_message({"content": "ping"}))
        finally:
            clear_session_vars(tokens)
        assert out["success"] is True
        assert out["room_id"] == "hub-1"

    @pytest.mark.asyncio
    async def test_no_home_configured_errors(self, monkeypatch):
        monkeypatch.setenv("BAND_TOOL_OWNERS", "telegram:u-owner")
        monkeypatch.delenv("BAND_HOME_ROOM", raising=False)
        monkeypatch.delenv("BAND_HUB_ROOM", raising=False)
        tokens = set_session_vars(platform="telegram", chat_id="tg-1", user_id="u-owner")
        rest = _make_rest()
        try:
            with _patch_rest(rest), _patch_agent_id("agent-self"):
                out = _parse(await band_tools._handle_send_message({"content": "ping"}))
        finally:
            clear_session_vars(tokens)
        assert "error" in out
        assert "no target room" in out["error"]
        rest.agent_api_messages.create_agent_chat_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_band_session_does_not_fall_back(self, owner_session, monkeypatch):
        # In a Band room the current room wins — no home fallback.
        monkeypatch.setenv("BAND_HOME_ROOM", "hub-home")
        rest = _make_rest()
        rest.agent_api_participants.list_agent_chat_participants = AsyncMock(
            return_value=SimpleNamespace(data=[_peer("u-human", handle="alice")])
        )
        with _patch_rest(rest), _patch_agent_id("agent-self"):
            out = _parse(await band_tools._handle_send_message({"content": "hi"}))
        assert out["success"] is True
        assert out["room_id"] == "room-current"  # not hub-home


# ---------------------------------------------------------------------------
# 7. Owner gate
# ---------------------------------------------------------------------------

class TestOwnerGate:

    @pytest.mark.asyncio
    async def test_unset_owners_allows_any_caller(self, monkeypatch):
        # Loose by default: with BAND_TOOL_OWNERS unset, the mutating tools are
        # allowed for any caller — Band owns access control and enforces roles
        # server-side, so Hermes does not re-gate the agent's outbound actions.
        monkeypatch.delenv("BAND_TOOL_OWNERS", raising=False)
        tokens = set_session_vars(platform="band", chat_id="room-1", user_id="u1")
        rest = _make_rest()
        try:
            with _patch_rest(rest):
                for handler, args in (
                    (band_tools._handle_create_room, {}),
                    (band_tools._handle_send_message, {"content": "hi", "mention_ids": ["p-1"]}),
                    (band_tools._handle_add_participant, {"participant_id": "p"}),
                    (band_tools._handle_remove_participant, {"participant_id": "p"}),
                ):
                    out = _parse(await handler(args))
                    assert out.get("success") is True, out
        finally:
            clear_session_vars(tokens)

    @pytest.mark.asyncio
    async def test_unset_owners_allows_offplatform_caller(self, monkeypatch):
        # Loose applies to off-Band callers too (Telegram / CLI / cron); no
        # platform allowlist is consulted.
        monkeypatch.delenv("BAND_TOOL_OWNERS", raising=False)
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
        tokens = set_session_vars(platform="telegram", chat_id="c", user_id="rando")
        rest = _make_rest()
        try:
            with _patch_rest(rest):
                out = _parse(
                    await band_tools._handle_add_participant(
                        {"participant_id": "p", "room_id": "r"}
                    )
                )
        finally:
            clear_session_vars(tokens)
        assert out["success"] is True
        rest.agent_api_participants.add_agent_chat_participant.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_wrong_platform_user_refused(self, monkeypatch):
        monkeypatch.setenv("BAND_TOOL_OWNERS", "telegram:owner-id")
        # Right user id, wrong platform -> "telegram:other" not on the list.
        tokens = set_session_vars(platform="telegram", chat_id="c", user_id="other")
        rest = _make_rest()
        try:
            with _patch_rest(rest):
                out = _parse(
                    await band_tools._handle_add_participant(
                        {"participant_id": "p", "room_id": "r"}
                    )
                )
        finally:
            clear_session_vars(tokens)
        assert "error" in out
        assert "not authorized" in out["error"]
        rest.agent_api_participants.add_agent_chat_participant.assert_not_called()

    @pytest.mark.asyncio
    async def test_listed_owner_allowed(self, monkeypatch):
        monkeypatch.setenv("BAND_TOOL_OWNERS", "telegram:owner-id,band:owner-uuid")
        tokens = set_session_vars(platform="telegram", chat_id="c", user_id="owner-id")
        rest = _make_rest()
        try:
            with _patch_rest(rest):
                out = _parse(
                    await band_tools._handle_add_participant(
                        {"participant_id": "p", "room_id": "r"}
                    )
                )
        finally:
            clear_session_vars(tokens)
        assert out["success"] is True
        rest.agent_api_participants.add_agent_chat_participant.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_readonly_tools_never_gated(self, monkeypatch):
        # Read-only tools work even when BAND_TOOL_OWNERS restricts and the
        # caller is not on the allowlist.
        monkeypatch.setenv("BAND_TOOL_OWNERS", "band:someone-else")
        tokens = set_session_vars(platform="telegram", chat_id="c", user_id="rando")
        rest = _make_rest()
        rest.agent_api_participants.list_agent_chat_participants = AsyncMock(
            return_value=SimpleNamespace(data=[_peer("u-1", handle="alice")])
        )
        try:
            with _patch_rest(rest):
                get_out = _parse(
                    await band_tools._handle_get_participants({"room_id": "r"})
                )
                find_room_out = _parse(await band_tools._handle_find_room({}))
                find_contact_out = _parse(
                    await band_tools._handle_find_contact({"query": "alice"})
                )
        finally:
            clear_session_vars(tokens)
        assert get_out["success"] is True
        assert find_room_out["success"] is True
        assert find_contact_out["success"] is True


# ---------------------------------------------------------------------------
# 8. check_fn / availability
# ---------------------------------------------------------------------------

class TestCheckBandToolsAvailable:

    def test_true_when_sdk_and_key_present(self, monkeypatch):
        monkeypatch.setenv("BAND_API_KEY", "k")
        assert band_tools._check_band_tools_available() is True

    def test_false_when_no_api_key(self, monkeypatch):
        monkeypatch.delenv("BAND_API_KEY", raising=False)
        assert band_tools._check_band_tools_available() is False


# ---------------------------------------------------------------------------
# 9. Registry tuple shape
# ---------------------------------------------------------------------------

class TestMentionItems:
    """The shared mention builder used by both the adapter send and the tool."""

    @staticmethod
    def _ids(items):
        return [i.id for i in items]

    def test_explicit_ids_resolve_handle_from_participants(self):
        from hermes_band_platform.adapter import _mention_items

        parts = [{"id": "u1", "handle": "alice", "name": "Alice"}]
        items = _mention_items(parts, agent_id="me", explicit_ids=["u1", "u2"])
        assert self._ids(items) == ["u1", "u2"]
        assert items[0].handle == "alice"   # resolved
        assert items[1].handle is None      # unknown id → bare mention

    def test_explicit_ids_carry_no_kind_so_the_server_defaults_them(self):
        """Only a self entry is ever sent with an explicit ``kind``.

        Every other recipient omits the field, leaving the server's ``"mention"``
        default to apply — so the wire payload for a normal send is unchanged.
        """
        from hermes_band_platform.adapter import _mention_items

        items = _mention_items(
            [{"id": "u1", "handle": "alice"}], agent_id="me", explicit_ids=["u1"]
        )
        assert getattr(items[0], "kind", None) is None

    def test_explicit_self_id_is_demoted_to_a_reference(self):
        """Band 422s a self @mention and rejects the *whole* message with it.

        The self entry survives as a narrative reference (so an ``@self`` already
        in the content still renders) while the real recipient stays a delivery
        mention, which is what makes the message send at all.
        """
        from hermes_band_platform.adapter import (
            _is_delivery_mention_item,
            _mention_items,
        )

        parts = [
            {"id": "me", "handle": "bot", "type": "Agent"},
            {"id": "u1", "handle": "alice", "type": "User"},
        ]
        items = _mention_items(parts, agent_id="me", explicit_ids=["u1", "me"])
        assert self._ids(items) == ["u1", "me"]          # order preserved
        assert items[1].kind == "reference"
        assert items[1].handle == "bot"                  # still resolved
        assert _is_delivery_mention_item(items[0]) is True
        assert _is_delivery_mention_item(items[1]) is False

    def test_self_only_explicit_list_has_no_delivery_recipient(self):
        """The demotion does not conjure a recipient — the caller must error."""
        from hermes_band_platform.adapter import (
            _is_delivery_mention_item,
            _mention_items,
        )

        items = _mention_items([], agent_id="me", explicit_ids=["me"])
        assert self._ids(items) == ["me"]
        assert not any(_is_delivery_mention_item(i) for i in items)

    def test_preferred_wins_over_fallback(self):
        from hermes_band_platform.adapter import _mention_items

        items = _mention_items(
            [{"id": "x", "handle": "x"}],  # would be the fallback
            agent_id="me",
            preferred={"id": "human", "handle": "bob"},
        )
        assert self._ids(items) == ["human"]

    def test_fallback_excludes_agent_and_other_agents(self):
        from hermes_band_platform.adapter import _mention_items

        parts = [
            {"id": "me", "handle": "bot", "type": "Agent"},      # self
            {"id": "a2", "handle": "peer", "type": "Agent"},     # other agent
            {"id": "h1", "handle": "alice", "type": "User"},     # human
        ]
        items = _mention_items(parts, agent_id="me")
        assert self._ids(items) == ["h1"]


class TestBandToolsTuple:

    def test_all_tools_present(self):
        names = {t[0] for t in band_tools.BAND_TOOLS}
        assert names == {
            "band_create_room",
            "band_find_room",
            "band_find_contact",
            "band_send_message",
            "band_add_participant",
            "band_remove_participant",
            "band_get_participants",
        }

    def test_tuple_entries_are_well_formed(self):
        for name, schema, handler, emoji in band_tools.BAND_TOOLS:
            assert schema["name"] == name
            assert callable(handler)
            assert isinstance(emoji, str) and emoji


# ---------------------------------------------------------------------------
# 9. Owner-implies-authority (the Band owner bypasses BAND_TOOL_OWNERS from
#    any Band room)
# ---------------------------------------------------------------------------

class TestOwnerAuthority:

    @pytest.fixture(autouse=True)
    def _restrict_to_allowlist(self, monkeypatch):
        """Restrict Band tools to an allowlist that excludes the test callers,
        so only the owner bypass can authorize. (With BAND_TOOL_OWNERS unset the
        tools are loose and the owner bypass would be moot.)"""
        monkeypatch.setenv("BAND_TOOL_OWNERS", "band:not-a-test-user")
        for var in ("BAND_ALLOWED_USERS", "BAND_ALLOW_ALL", "BAND_ALLOW_ALL_USERS"):
            monkeypatch.delenv(var, raising=False)
        yield

    def test_owner_identity_env_fallback(self, monkeypatch):
        monkeypatch.setenv("BAND_OWNER_ID", "owner-1")
        assert band_tools._owner_identity() == "owner-1"

    def test_owner_identity_unresolved(self, monkeypatch):
        monkeypatch.delenv("BAND_OWNER_ID", raising=False)
        assert band_tools._owner_identity() is None

    @pytest.mark.asyncio
    async def test_owner_in_hub_authorized_despite_allowlist(self, monkeypatch):
        monkeypatch.setenv("BAND_OWNER_ID", "owner-1")
        tokens = set_session_vars(platform="band", chat_id="hub-1", user_id="owner-1")
        rest = _make_rest()
        try:
            with _patch_rest(rest):
                out = _parse(
                    await band_tools._handle_add_participant({"participant_id": "p"})
                )
        finally:
            clear_session_vars(tokens)
        assert out["success"] is True
        rest.agent_api_participants.add_agent_chat_participant.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_owner_outside_hub_authorized(self, monkeypatch):
        """Owner implies authority from ANY Band room, not just the hub."""
        monkeypatch.setenv("BAND_OWNER_ID", "owner-1")
        tokens = set_session_vars(
            platform="band", chat_id="other-room", user_id="owner-1"
        )
        rest = _make_rest()
        try:
            with _patch_rest(rest):
                out = _parse(
                    await band_tools._handle_add_participant({"participant_id": "p"})
                )
        finally:
            clear_session_vars(tokens)
        assert out["success"] is True
        rest.agent_api_participants.add_agent_chat_participant.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_owner_not_in_allowlist_refused(self, monkeypatch):
        # A non-owner who isn't on the restricting allowlist is refused.
        monkeypatch.setenv("BAND_OWNER_ID", "owner-1")
        tokens = set_session_vars(platform="band", chat_id="hub-1", user_id="someone")
        rest = _make_rest()
        try:
            with _patch_rest(rest):
                out = _parse(
                    await band_tools._handle_add_participant({"participant_id": "p"})
                )
        finally:
            clear_session_vars(tokens)
        assert "error" in out
        assert "not authorized" in out["error"]
        rest.agent_api_participants.add_agent_chat_participant.assert_not_called()

    @pytest.mark.asyncio
    async def test_unresolved_owner_not_in_allowlist_refused(self, monkeypatch):
        # With an allowlist set and the owner unresolved, the caller has no
        # bypass and is not listed -> refused.
        monkeypatch.delenv("BAND_OWNER_ID", raising=False)
        tokens = set_session_vars(platform="band", chat_id="hub-1", user_id="owner-1")
        rest = _make_rest()
        try:
            with _patch_rest(rest):
                out = _parse(
                    await band_tools._handle_add_participant({"participant_id": "p"})
                )
        finally:
            clear_session_vars(tokens)
        assert "error" in out
        assert "not authorized" in out["error"]
        rest.agent_api_participants.add_agent_chat_participant.assert_not_called()

    @pytest.mark.asyncio
    async def test_owner_bypass_band_only(self, monkeypatch):
        """The bypass is for Band sessions; other platforms still need a gate."""
        monkeypatch.setenv("BAND_OWNER_ID", "owner-1")
        monkeypatch.delenv("TELEGRAM_ALLOWED_USERS", raising=False)
        monkeypatch.delenv("TELEGRAM_ALLOW_ALL", raising=False)
        tokens = set_session_vars(
            platform="telegram", chat_id="tg-chat", user_id="owner-1"
        )
        rest = _make_rest()
        try:
            with _patch_rest(rest):
                out = _parse(
                    await band_tools._handle_add_participant({"participant_id": "p"})
                )
        finally:
            clear_session_vars(tokens)
        assert "error" in out
        rest.agent_api_participants.add_agent_chat_participant.assert_not_called()

    @pytest.mark.asyncio
    async def test_live_adapter_identity_preferred_over_env(self, monkeypatch):
        """The connected adapter's resolved owner wins over stale env."""
        monkeypatch.setenv("BAND_OWNER_ID", "stale-owner")
        fake_adapter = SimpleNamespace(_owner_uuid="live-owner")
        fake_runner = SimpleNamespace(adapters={band_tools.Platform("band"): fake_adapter})
        import gateway.run as gateway_run
        monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: fake_runner)
        assert band_tools._owner_identity() == "live-owner"


# ---------------------------------------------------------------------------
# 10. _rest() loop affinity — the live link's REST client is loop-bound
# ---------------------------------------------------------------------------

class TestRestClientLoopAffinity:
    """``_rest`` must not hand a gateway-loop client to an agent-loop caller.

    ``BandLink`` builds one long-lived ``AsyncRestClient`` in its constructor, so
    its pool binds to the loop that issues the first request. Awaiting it from
    another loop raises ``<asyncio.locks.Event ...> is bound to a different event
    loop`` — observed on a live gateway as every ``band_*`` tool failing in
    ~0.02s without reaching the network.

    Not reachable through the mocked-``_rest`` strategy the rest of this file
    uses: the bug is in ``_rest`` itself and needs two real running loops.
    """

    @staticmethod
    def _fake_runner(adapter):
        return SimpleNamespace(adapters={band_tools.Platform("band"): adapter})

    @pytest.mark.asyncio
    async def test_same_loop_reuses_the_live_link(self, monkeypatch):
        """The no-second-connection optimisation must survive the fix."""
        link = SimpleNamespace(rest=MagicMock(name="link-rest"))
        adapter = SimpleNamespace(_link=link, _link_loop=asyncio.get_running_loop())
        import gateway.run as gateway_run
        monkeypatch.setattr(
            gateway_run, "_gateway_runner_ref", lambda: self._fake_runner(adapter)
        )
        assert await band_tools._rest() is link.rest

    @pytest.mark.asyncio
    async def test_unknown_link_loop_keeps_previous_behaviour(self, monkeypatch):
        """Only a *proven* mismatch declines; an unrecorded loop does not."""
        link = SimpleNamespace(rest=MagicMock(name="link-rest"))
        adapter = SimpleNamespace(_link=link, _link_loop=None)
        import gateway.run as gateway_run
        monkeypatch.setattr(
            gateway_run, "_gateway_runner_ref", lambda: self._fake_runner(adapter)
        )
        assert await band_tools._rest() is link.rest

    @pytest.mark.asyncio
    async def test_cross_loop_declines_the_shared_client(self, monkeypatch, other_loop):
        """The regression: a different running loop must not get link.rest."""
        monkeypatch.setenv("BAND_API_KEY", "k-test")
        monkeypatch.setenv("BAND_BASE_URL", "")
        link = SimpleNamespace(rest=MagicMock(name="link-rest"))
        adapter = SimpleNamespace(_link=link, _link_loop=other_loop)
        import gateway.run as gateway_run
        monkeypatch.setattr(
            gateway_run, "_gateway_runner_ref", lambda: self._fake_runner(adapter)
        )

        fresh = MagicMock(name="fresh-per-call-client")
        with patch("band.client.rest.AsyncRestClient", return_value=fresh) as ctor:
            got = await band_tools._rest()

        assert got is not link.rest, (
            "returned the gateway loop's client to an agent-loop caller — "
            "awaiting it raises 'bound to a different event loop'"
        )
        assert got is fresh, "did not fall back to a fresh per-call client"
        assert ctor.call_args.kwargs["api_key"] == "k-test"

    @pytest.mark.asyncio
    async def test_cross_loop_without_credentials_reports_clearly(
        self, monkeypatch, other_loop
    ):
        """Declining must not degrade into a cryptic failure."""
        monkeypatch.delenv("BAND_API_KEY", raising=False)
        link = SimpleNamespace(rest=MagicMock(name="link-rest"))
        adapter = SimpleNamespace(_link=link, _link_loop=other_loop)
        import gateway.run as gateway_run
        monkeypatch.setattr(
            gateway_run, "_gateway_runner_ref", lambda: self._fake_runner(adapter)
        )
        with pytest.raises(band_tools._ToolUnavailable):
            await band_tools._rest()


class TestToolExcLogging:
    """An unexpected tool exception must leave frames behind, not one bare line."""

    def test_unexpected_exception_logs_traceback(self, caplog):
        try:
            raise RuntimeError("<Event> is bound to a different event loop")
        except RuntimeError as exc:
            with caplog.at_level("DEBUG", logger=band_tools.logger.name):
                out = _parse(band_tools._tool_exc(exc))
        assert "error" in out
        assert any(r.exc_info for r in caplog.records), "no traceback captured"
        assert "RuntimeError" in caplog.text

    def test_expected_tool_error_stays_quiet(self, caplog):
        with caplog.at_level("DEBUG", logger=band_tools.logger.name):
            out = _parse(band_tools._tool_exc(band_tools._ToolError("no room_id")))
        assert "error" in out
        assert not any(r.exc_info for r in caplog.records)
