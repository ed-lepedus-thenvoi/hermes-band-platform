"""
Band action tools for the Hermes agent.

This module is the **tools pass** companion to ``adapter.py`` (the messaging
adapter).  Where the adapter relays inbound/outbound *messages*, these tools let
the agent *act on Band* — create rooms, add/remove participants, send messages,
look up contacts — from any conversation it is in (a Band room, Telegram, the
CLI, …).

Design anchors (see ``Drafts/hermes-band-tools-events-buildplan.md``):

  * **Native REST.** Handlers drive the Band ``AsyncRestClient`` directly,
    reusing the *live* :class:`BandAdapter`'s authenticated link when the
    gateway is in-process, and falling back to a fresh client from env creds for
    out-of-process callers (cron, tests).
  * **Session-bound room.** Room-context tools take an *optional* ``room_id``;
    when absent they default to the current Band room resolved from session
    context (``HERMES_SESSION_PLATFORM`` / ``HERMES_SESSION_CHAT_ID``).  An
    explicit id always wins (cross-room / off-platform / freshly-created rooms).
  * **Loose outbound, Band owns access.** Creating rooms and
    adding/removing/messaging are the agent's own outbound actions; Band itself
    is the ACL (it admits participants and enforces member/admin/owner roles
    server-side), so Hermes does not re-gate them — they are loose by default.
    The one owner-only surface is Hermes slash commands, gated in ``adapter.py``.
    Optional tightening: set ``BAND_TOOL_OWNERS`` (``platform:user_id`` list) to
    restrict these tools to specific callers (the resolved Band owner always
    passes).  Read-only tools (find/get) are never gated.

Conventions mirrored from ``adapter.py``: lazy SDK import (the module imports
cleanly when ``band-sdk`` is absent), ``_short_id`` for low-cardinality logs,
and API keys are **never** logged.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, List, Optional

from gateway.config import Platform
from gateway.platforms.base import BasePlatformAdapter
from gateway.session_context import get_session_env
from tools.registry import tool_error, tool_result

# Reuse the adapter's helpers + the lazy SDK-availability check. Importing the
# adapter module is safe even when the SDK is absent (it guards its own import).
from .adapter import (
    DEFAULT_REQUEST_OPTIONS,
    note_deliberate_send,
    sender_of,
    _derive_urls,
    _mention_items,
    _short_id,
    check_band_requirements,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy SDK import guard.
#
# The request *types* we construct (ChatRoomRequest / ParticipantRequest) are
# imported lazily — like the adapter's module-top guard — so this module loads
# even when ``band-sdk`` isn't installed (plugin discovery runs before deps
# are guaranteed present). ``check_band_requirements()`` (re)binds the names on
# demand; ``_load_sdk()`` below re-imports inside handlers so a late install is
# picked up without a process restart.
# ---------------------------------------------------------------------------
try:
    from band.client.rest import (  # noqa: F401
        ChatMessageRequest,
        ChatMessageRequestMentionsItem,
        ChatRoomRequest,
        ParticipantRequest,
    )
except ImportError:  # SDK not present yet — bind to None, rebind in _load_sdk().
    ChatMessageRequest = None
    ChatMessageRequestMentionsItem = None
    ChatRoomRequest = None
    ParticipantRequest = None


# Cap for read-only listings (find_room / find_contact / get_participants) so a
# huge org directory can't blow up a tool result; we log when truncated.
_MAX_RESULTS = 50
# Hard cap on pages we'll paginate through for lookups (defensive bound).
_MAX_LOOKUP_PAGES = 20

# Same conservative per-message content cap the adapter uses (no confirmed Band
# hard limit) — mirrors ``BandAdapter.MAX_MESSAGE_LENGTH``. Reused for chunking
# long sends via ``BasePlatformAdapter.truncate_message``.
_MAX_MESSAGE_LENGTH = 4000


class _ToolError(Exception):
    """A user-facing tool failure (bad args, auth, no room) → ``tool_error``."""


class _ToolUnavailable(Exception):
    """Band isn't configured/available in this process → ``tool_error``."""


def _load_sdk() -> bool:
    """(Re)bind the SDK request types this module constructs.

    Returns True when the symbols are bound. Mirrors the adapter's
    ``check_band_requirements()`` lazy-rebind so a late ``pip install`` is picked
    up without restarting the process.
    """
    global ChatMessageRequest, ChatMessageRequestMentionsItem
    global ChatRoomRequest, ParticipantRequest

    if ChatRoomRequest is not None:
        return True
    # Mirror check_band_requirements: re-apply the band-libs shim so an SDK
    # resolved into $HERMES_HOME/band-libs after module import is found.
    try:
        from ._band_libs import prepend_band_libs

        prepend_band_libs()
    except Exception:
        pass
    try:
        from band.client.rest import (
            ChatMessageRequest as _ChatMessageRequest,
            ChatMessageRequestMentionsItem as _ChatMessageRequestMentionsItem,
            ChatRoomRequest as _ChatRoomRequest,
            ParticipantRequest as _ParticipantRequest,
        )
    except ImportError:
        return False

    ChatMessageRequest = _ChatMessageRequest
    ChatMessageRequestMentionsItem = _ChatMessageRequestMentionsItem
    ChatRoomRequest = _ChatRoomRequest
    ParticipantRequest = _ParticipantRequest
    return True


def _check_band_tools_available() -> bool:
    """``check_fn`` for every Band tool.

    True only when the SDK is importable (``check_band_requirements``) AND a
    ``BAND_API_KEY`` is present — so the tools disappear cleanly when Band is
    unconfigured rather than failing at call time.
    """
    try:
        if not check_band_requirements():
            return False
    except Exception:
        return False
    return bool(os.getenv("BAND_API_KEY", "").strip())


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _link_rest_usable_here(adapter: Any) -> bool:
    """True when the live link's REST client can be awaited on *this* loop.

    :class:`BandLink` builds one long-lived ``AsyncRestClient`` in its
    constructor (``band/platform/link.py``), so its connection pool binds to the
    loop that issues the first request — the gateway's, at connect. Awaiting it
    from another running loop, such as the agent's tool executor, raises
    ``<asyncio.locks.Event ...> is bound to a different event loop``.

    This is the same failure :meth:`BandAdapter.send` already guards against by
    marshalling onto ``_link_loop`` (INT-899). A tool cannot use that remedy:
    ``_rest`` hands a client back and the *caller* awaits it, so there is no
    coroutine here to marshal. The mismatch is resolved instead by declining the
    shared client, which drops the caller onto the fresh per-call client that
    ``_rest``'s fallback branch already documents as loop-safe.

    Only a **proven** mismatch declines — both loops known and different. An
    unknown loop keeps the previous behaviour, since a live adapter always sets
    ``_link_loop`` alongside ``_link``, so "unknown" means "not connected" and
    the caller is not in the live-link branch anyway.
    """
    link_loop = getattr(adapter, "_link_loop", None)
    if link_loop is None:
        return True
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        return True
    if running is link_loop:
        return True
    logger.debug(
        "[band.tools] live link is bound to another event loop; using a fresh "
        "per-call REST client instead (link_loop=%#x running=%#x)",
        id(link_loop),
        id(running),
    )
    return False


async def _rest() -> Any:
    """Return an authenticated async REST client.

    Prefers the *live* :class:`BandAdapter`'s link (no second connection) — but
    only when that link's loop is the one running here, see
    :func:`_link_rest_usable_here`. Falls back to a fresh ``AsyncRestClient``
    from env creds for out-of-process callers (cron / no live gateway) and for a
    cross-loop caller. The fallback client is short-lived — created per call — so
    it never leaks a pooled connection across a loop.
    """
    try:
        from gateway.run import _gateway_runner_ref

        runner = _gateway_runner_ref()
        adapter = runner.adapters.get(Platform("band")) if runner else None
        link = getattr(adapter, "_link", None) if adapter is not None else None
        if link is not None and _link_rest_usable_here(adapter):
            return link.rest
    except Exception:
        # Fall through to the env-creds fallback.
        pass

    if not _load_sdk():
        from ._band_libs import sdk_install_command

        raise _ToolUnavailable(
            "Band not available (band-sdk not resolvable by the gateway Python). "
            "Directory plugin installs do not install dependencies; run: "
            f"{sdk_install_command()}"
        )
    from band.client.rest import AsyncRestClient

    api_key = os.getenv("BAND_API_KEY", "").strip()
    if not api_key:
        raise _ToolUnavailable("Band not configured (BAND_API_KEY)")
    _, rest_url = _derive_urls(os.getenv("BAND_BASE_URL", "").strip())
    return AsyncRestClient(api_key=api_key, base_url=rest_url)


def _resolve_room(args: Dict[str, Any]) -> str:
    """Resolve the target room id for a room-context tool.

    Precedence: explicit ``room_id`` arg → else the current Band room from
    session context (only when this conversation *is* a Band session) → else
    error.  An explicit id always wins, so cross-room / off-platform /
    freshly-created rooms work the same way.
    """
    rid = str((args or {}).get("room_id") or "").strip()
    if rid:
        return rid
    if get_session_env("HERMES_SESSION_PLATFORM") == "band":
        rid = (get_session_env("HERMES_SESSION_CHAT_ID") or "").strip()
        if rid:
            return rid
    raise _ToolError("room_id required (no current Band room in this conversation)")


def _resolve_room_for_send(args: Dict[str, Any]) -> tuple[str, bool]:
    """Resolve the send target, allowing a fallback to the owner's hub/home.

    Returns ``(room_id, fell_back_to_home)``. Precedence matches
    ``_resolve_room`` (explicit ``room_id`` → current Band room) but, instead of
    erroring when neither is present, falls back to the owner's hub/home room so
    "send a message to me" works from a non-Band session. Raises only when no
    room is in context AND no home room is configured yet (hub not bootstrapped).
    """
    try:
        return _resolve_room(args), False
    except _ToolError:
        home = _home_room()
        if home:
            return home, True
        raise _ToolError(
            "no target room: pass room_id, send from a Band room, or connect the "
            "gateway so the owner hub (home channel) is created"
        )


def _owner_identity() -> Optional[str]:
    """Resolve the Band owner UUID — live adapter first, env fallback.

    Mirrors ``_agent_id_or_none``: prefer the connected adapter's resolved
    state (set on connect from the agent identity); fall back to
    ``BAND_OWNER_ID`` for out-of-process callers. Never raises.
    """
    owner: Optional[str] = None
    try:
        from gateway.run import _gateway_runner_ref

        runner = _gateway_runner_ref()
        adapter = runner.adapters.get(Platform("band")) if runner else None
        if adapter is not None:
            owner = getattr(adapter, "_owner_uuid", None)
    except Exception:
        pass
    return owner or (os.getenv("BAND_OWNER_ID", "").strip() or None)


def _home_room() -> Optional[str]:
    """Resolve the owner's hub / home room id — live adapter first, env fallback.

    This is where the agent reaches its owner: the private owner↔agent hub, also
    wired as the Band home (main) channel. Prefer the connected adapter's
    ``_hub_room_id``; fall back to ``BAND_HOME_ROOM`` then ``BAND_HUB_ROOM`` for
    out-of-process callers. Never raises.
    """
    try:
        from gateway.run import _gateway_runner_ref

        runner = _gateway_runner_ref()
        adapter = runner.adapters.get(Platform("band")) if runner else None
        hub = getattr(adapter, "_hub_room_id", None) if adapter is not None else None
        if hub:
            return str(hub)
    except Exception:
        pass
    return (
        os.getenv("BAND_HOME_ROOM", "").strip()
        or os.getenv("BAND_HUB_ROOM", "").strip()
        or None
    )


def _authorize_band_action() -> None:
    """Authorize a mutating Band action.

    Policy: outbound Band actions are **loose** by default. Band itself owns
    access control — it decides who is admitted to a room and enforces role
    permissions (member/admin/owner) server-side — so Hermes does not re-gate
    the agent's own outbound actions. The one owner-only surface is Hermes
    slash commands, gated separately in ``adapter.py``.

    Optional tightening: set ``BAND_TOOL_OWNERS`` (comma-separated
    ``platform:user_id`` identities) to restrict Band actions to specific
    callers. When it is set, the resolved Band owner is always allowed (owner
    implies authority) and everyone else must appear in the allowlist; when it
    is unset (the default), any caller passes and Band enforces the real
    permissions.
    """
    owners = {o.strip() for o in os.getenv("BAND_TOOL_OWNERS", "").split(",") if o.strip()}
    if not owners:
        return  # loose default — Band's own ACL/roles are the authority

    platform = (get_session_env("HERMES_SESSION_PLATFORM") or "").strip()
    user_id = (get_session_env("HERMES_SESSION_USER_ID") or "").strip()

    # Owner-implies-authority: the resolved Band owner is always allowed.
    if platform == "band" and user_id and user_id == _owner_identity():
        return

    who = f"{platform}:{user_id}"
    if who not in owners:
        raise _ToolError(
            f"Band actions are restricted by BAND_TOOL_OWNERS; {who} is not authorized"
        )


def _peer_to_dict(p: Any) -> Dict[str, Any]:
    """Normalize an SDK peer/participant object into a plain dict."""
    return {
        "id": getattr(p, "id", None),
        "handle": getattr(p, "handle", None),
        "name": getattr(p, "name", None),
        "type": getattr(p, "type", None),
    }


async def _list_peers(rest: Any) -> List[Dict[str, Any]]:
    """Paginate ``list_agent_peers`` (and contacts, if present) into dicts.

    Best-effort: peers are the primary directory; contacts are merged in when
    the resource exists. Bounded by ``_MAX_LOOKUP_PAGES``.
    """
    out: List[Dict[str, Any]] = []
    seen: set = set()

    page = 1
    while page <= _MAX_LOOKUP_PAGES:
        resp = await rest.agent_api_peers.list_agent_peers(
            page=page,
            page_size=_MAX_RESULTS,
            request_options=DEFAULT_REQUEST_OPTIONS,
        )
        rows = getattr(resp, "data", None) or []
        for row in rows:
            d = _peer_to_dict(row)
            if d["id"] and d["id"] not in seen:
                seen.add(d["id"])
                out.append(d)
        meta = getattr(resp, "metadata", None)
        total_pages = getattr(meta, "total_pages", None)
        if total_pages is None or page >= total_pages:
            break
        page += 1

    # Merge in contacts when the resource exists (older SDKs may omit it).
    contacts_api = getattr(rest, "agent_api_contacts", None)
    list_contacts = getattr(contacts_api, "list_agent_contacts", None)
    if callable(list_contacts):
        try:
            resp = await list_contacts(request_options=DEFAULT_REQUEST_OPTIONS)
            for row in getattr(resp, "data", None) or []:
                d = _peer_to_dict(row)
                if d["id"] and d["id"] not in seen:
                    seen.add(d["id"])
                    out.append(d)
        except Exception as e:
            logger.debug("[band.tools] contacts lookup skipped: %s", e)

    return out


async def _find_participant(rest: Any, query: str) -> Optional[Dict[str, Any]]:
    """Resolve a free-text ``query`` to a single peer/contact.

    Case-insensitive match on handle / name / id. Prefers an exact id or handle
    match, then an exact name, then a substring match. Returns ``{id, handle,
    name}`` or None.
    """
    q = (query or "").strip()
    if not q:
        return None
    ql = q.lower()

    peers = await _list_peers(rest)
    substring: Optional[Dict[str, Any]] = None
    for p in peers:
        pid = (p.get("id") or "")
        handle = (p.get("handle") or "")
        name = (p.get("name") or "")
        if pid == q or handle.lower() == ql:
            return p
        if name.lower() == ql:
            return p
        if substring is None and (ql in handle.lower() or ql in name.lower()):
            substring = p
    return substring


async def _mentions_for(
    rest: Any, room_id: str, mention_ids: Optional[List[str]]
) -> List[Any]:
    """Build the mandatory mention list for a send (Band requires ≥1).

    If ``mention_ids`` is given, build one mention per id (handle resolved from
    the room participants when cheap). Otherwise mention every non-agent
    participant in the room. Raises ``_ToolError`` if the result is empty.
    """
    if not _load_sdk():
        raise _ToolUnavailable("Band not available (band-sdk not installed)")

    # Fetch participants once for handle resolution / fallback mentions, then
    # delegate to the shared builder (same semantics as the adapter's send).
    participants = await _list_participants(rest, room_id)
    # Resolve the running agent's id so the fallback never @mentions ourselves
    # (irrelevant when explicit mention_ids are given).
    agent_id = None if mention_ids else await _agent_id_or_none(rest)
    items = _mention_items(participants, agent_id=agent_id, explicit_ids=mention_ids)

    if not items:
        raise _ToolError(
            "Band requires at least one @mention; no mentionable recipient was found "
            "(pass mention_ids or add a participant to the room first)"
        )
    return items


async def _list_participants(rest: Any, room_id: str) -> List[Dict[str, Any]]:
    """Fetch a room's participants as plain dicts (best-effort)."""
    resp = await rest.agent_api_participants.list_agent_chat_participants(
        chat_id=room_id,
        request_options=DEFAULT_REQUEST_OPTIONS,
    )
    return [_peer_to_dict(p) for p in (getattr(resp, "data", None) or [])]


async def _agent_id_or_none(rest: Any) -> Optional[str]:
    """Best-effort: the running adapter's resolved agent id (for self-exclusion).

    Prefers the live adapter's ``_agent_id``; falls back to ``BAND_AGENT_ID``.
    Never raises.
    """
    try:
        from gateway.run import _gateway_runner_ref

        runner = _gateway_runner_ref()
        adapter = runner.adapters.get(Platform("band")) if runner else None
        aid = getattr(adapter, "_agent_id", None) if adapter is not None else None
        if aid:
            return aid
    except Exception:
        pass
    return os.getenv("BAND_AGENT_ID", "").strip() or None


def _tool_exc(exc: Exception) -> str:
    """Convert an exception into a ``tool_error`` JSON string.

    ``_ToolError`` / ``_ToolUnavailable`` carry user-facing messages; anything
    else is wrapped with its type so the model gets a useful (but bounded) hint.

    An unexpected exception also logs its traceback at ``debug``. The bounded
    string is all the model should see, but it is not enough to debug from: a
    cross-loop ``RuntimeError`` reached a log as one bare line with no frames,
    which is why the failing primitive had to be found by reading the source
    instead of the log. Nothing new is disclosed — ``str(exc)`` is already in the
    returned message — and frames name code locations, not payloads.
    """
    if isinstance(exc, (_ToolError, _ToolUnavailable)):
        return tool_error(str(exc))
    logger.debug("[band.tools] tool failed: %s", type(exc).__name__, exc_info=exc)
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return tool_error(f"Band tool failed: {exc}", status_code=status)
    return tool_error(f"Band tool failed: {type(exc).__name__}: {exc}")


# ---------------------------------------------------------------------------
# Tier-A tools — platform-level (no room context)
# ---------------------------------------------------------------------------

async def _handle_create_room(args: dict, **kwargs) -> str:
    """Create a Band room; optionally add a person and send them a message.

    Composite: with ``person`` it resolves → creates → adds → (if ``message``)
    sends in one call. No ``title`` arg — the server derives the title from the
    first message.
    """
    try:
        _authorize_band_action()
        if not _load_sdk():
            raise _ToolUnavailable("Band not available (band-sdk not installed)")
        rest = await _rest()

        created = await rest.agent_api_chats.create_agent_chat(
            chat=ChatRoomRequest(),
            request_options=DEFAULT_REQUEST_OPTIONS,
        )
        room_id = getattr(getattr(created, "data", None), "id", None)
        if not room_id:
            raise _ToolError("Band did not return a room id for the created room")

        result: Dict[str, Any] = {"success": True, "room_id": room_id, "added": []}

        person = str(args.get("person") or "").strip()
        role = str(args.get("role") or "").strip() or "member"
        resolved: Optional[Dict[str, Any]] = None
        if person:
            resolved = await _find_participant(rest, person)
            if resolved is None:
                # Room exists; report the partial result rather than silently dropping it.
                result["warning"] = f"Room created but no contact matched '{person}'"
                logger.info(
                    "[band.tools] create_room %s: no contact matched query",
                    _short_id(room_id),
                )
                return tool_result(result)
            await rest.agent_api_participants.add_agent_chat_participant(
                room_id,
                participant=ParticipantRequest(participant_id=resolved["id"], role=role),
                request_options=DEFAULT_REQUEST_OPTIONS,
            )
            result["added"] = [
                {"id": resolved["id"], "handle": resolved.get("handle"), "name": resolved.get("name")}
            ]

        message = str(args.get("message") or "").strip()
        if message:
            if resolved is None:
                raise _ToolError("'message' requires 'person' so the message has a recipient to @mention")
            mentions = [
                ChatMessageRequestMentionsItem(
                    id=resolved["id"], handle=resolved.get("handle"), name=resolved.get("name")
                )
            ]
            chunks = BasePlatformAdapter.truncate_message(message, _MAX_MESSAGE_LENGTH)
            sent_id: Optional[str] = None
            for chunk in chunks:
                resp = await rest.agent_api_messages.create_agent_chat_message(
                    room_id,
                    message=ChatMessageRequest(content=chunk, mentions=mentions),
                    request_options=DEFAULT_REQUEST_OPTIONS,
                )
                sent_id = getattr(getattr(resp, "data", None), "id", None) or sent_id
            result["sent"] = sent_id

        logger.info(
            "[band.tools] Created room %s (added=%d, sent=%s)",
            _short_id(room_id),
            len(result["added"]),
            bool(result.get("sent")),
        )
        return tool_result(result)
    except Exception as exc:
        return _tool_exc(exc)


async def _handle_find_room(args: dict, **kwargs) -> str:
    """Read-only: find existing rooms by free-text ``query`` on title/id."""
    try:
        rest = await _rest()
        q = str(args.get("query") or "").strip()
        ql = q.lower()

        matches: List[Dict[str, Any]] = []
        truncated = False
        page = 1
        while page <= _MAX_LOOKUP_PAGES:
            resp = await rest.agent_api_chats.list_agent_chats(
                page=page,
                request_options=DEFAULT_REQUEST_OPTIONS,
            )
            for room in getattr(resp, "data", None) or []:
                room_id = getattr(room, "id", None)
                title = getattr(room, "title", None) or ""
                if not room_id:
                    continue
                if not q or ql in title.lower() or ql == room_id.lower():
                    if len(matches) >= _MAX_RESULTS:
                        truncated = True
                        break
                    matches.append({"room_id": room_id, "title": title or None})
            if truncated:
                break
            meta = getattr(resp, "metadata", None)
            total_pages = getattr(meta, "total_pages", None)
            if total_pages is None or page >= total_pages:
                break
            page += 1

        if truncated:
            logger.info("[band.tools] find_room truncated at %d results", _MAX_RESULTS)
        return tool_result({"success": True, "rooms": matches, "truncated": truncated})
    except Exception as exc:
        return _tool_exc(exc)


async def _handle_find_contact(args: dict, **kwargs) -> str:
    """Read-only: resolve a free-text ``query`` to matching contacts/peers."""
    try:
        rest = await _rest()
        q = str(args.get("query") or "").strip()
        if not q:
            return tool_error("query is required")
        ql = q.lower()

        peers = await _list_peers(rest)
        matches: List[Dict[str, Any]] = []
        for p in peers:
            handle = (p.get("handle") or "").lower()
            name = (p.get("name") or "").lower()
            pid = (p.get("id") or "")
            if pid == q or ql in handle or ql in name:
                matches.append(
                    {"id": p.get("id"), "handle": p.get("handle"), "name": p.get("name")}
                )
                if len(matches) >= _MAX_RESULTS:
                    break
        return tool_result({"success": True, "contacts": matches})
    except Exception as exc:
        return _tool_exc(exc)


# ---------------------------------------------------------------------------
# Tier-B tools — room-context (room from session or explicit room_id)
# ---------------------------------------------------------------------------

async def _handle_send_message(args: dict, **kwargs) -> str:
    """Send a message into a room (mandatory @mention; chunked at the cap).

    Targeting precedence: explicit ``room_id`` → current Band room → the owner's
    hub/home room. The last fallback is what makes "send a message to me"
    (the owner) work from a non-Band session (CLI / web / another platform),
    where there is no current Band room: the message lands in the owner↔agent
    hub and @mentions the owner by default.
    """
    try:
        _authorize_band_action()
        if not _load_sdk():
            raise _ToolUnavailable("Band not available (band-sdk not installed)")
        rest = await _rest()
        # Send may fall back to the owner's hub when no room is in context, so
        # the agent can reach its owner from anywhere.
        room_id, fell_back_to_home = _resolve_room_for_send(args)

        content = str(args.get("content") or "")
        if not content.strip():
            return tool_error("content is required")

        mention_ids = args.get("mention_ids")
        if mention_ids is not None and not isinstance(mention_ids, list):
            mention_ids = [mention_ids]

        # reply_to is the primary way to address a reply: the model names the
        # MESSAGE it is answering and we resolve its author. That keeps the
        # choice deliberate while sparing the model a roster lookup — and a
        # failed roster lookup falling back to the owner is exactly how replies
        # ended up addressed to bystanders.
        reply_to = args.get("reply_to")
        if reply_to and mention_ids is None:
            author = sender_of(str(reply_to))
            if not author or not author.get("id"):
                # Hard error, never a fallback. An unresolvable recipient is
                # something the model can act on; a guess is something nobody
                # can see.
                return tool_error(
                    f"reply_to {reply_to!s} is not a message this agent has seen "
                    "recently — pass mention_ids explicitly instead"
                )
            mention_ids = [author["id"]]

        # When reaching the owner via the hub fallback with no explicit mentions,
        # @mention the owner specifically so "message me" always pings the owner.
        if mention_ids is None and fell_back_to_home:
            owner = _owner_identity()
            if owner:
                mention_ids = [owner]
        mentions = await _mentions_for(rest, room_id, mention_ids)

        chunks = BasePlatformAdapter.truncate_message(content, _MAX_MESSAGE_LENGTH)
        last_id: Optional[str] = None
        for chunk in chunks:
            resp = await rest.agent_api_messages.create_agent_chat_message(
                room_id,
                message=ChatMessageRequest(content=chunk, mentions=mentions),
                request_options=DEFAULT_REQUEST_OPTIONS,
            )
            last_id = getattr(getattr(resp, "data", None), "id", None) or last_id

        # This is the deliberate send. Recording it tells the adapter the model
        # has addressed this room itself, so the host's copy of the final text is
        # dropped rather than posted again (see the deliberate-send registry).
        note_deliberate_send(room_id)
        await _record_send_outcome(room_id, ok=True)

        logger.info(
            "[band.tools] Sent message to room %s (chunks=%d)", _short_id(room_id), len(chunks)
        )
        return tool_result({"success": True, "room_id": room_id, "message_id": last_id})
    except Exception as exc:
        return _tool_exc(exc)


async def _record_send_outcome(room_id: str, *, ok: bool, error: Any = None) -> None:
    """Hand a deliberate send's outcome back to the live adapter, if there is one.

    Hub failover keys off consecutive main-channel send failures. Those sends
    used to pass through the adapter; now the tools pass owns them, so the
    outcome has to travel back or the failover threshold would never be reached
    and the main channel would lose its protection silently.

    Best-effort by construction: out-of-process callers (cron) have no adapter,
    and a failure to record must never fail the send that succeeded.
    """
    try:
        from gateway.run import _gateway_runner_ref

        runner = _gateway_runner_ref()
        adapter = runner.adapters.get(Platform("band")) if runner else None
        if adapter is None:
            return
        await adapter.record_addressed_send(room_id, ok=ok, error=error)
    except Exception as exc:  # pragma: no cover - diagnostics only
        logger.debug("[band.tools] Could not record send outcome: %s", exc)


async def _handle_add_participant(args: dict, **kwargs) -> str:
    """Add a participant (by UUID) to a room."""
    try:
        _authorize_band_action()
        if not _load_sdk():
            raise _ToolUnavailable("Band not available (band-sdk not installed)")
        rest = await _rest()
        room_id = _resolve_room(args)

        participant_id = str(args.get("participant_id") or "").strip()
        if not participant_id:
            return tool_error("participant_id is required")
        role = str(args.get("role") or "").strip() or "member"

        await rest.agent_api_participants.add_agent_chat_participant(
            room_id,
            participant=ParticipantRequest(participant_id=participant_id, role=role),
            request_options=DEFAULT_REQUEST_OPTIONS,
        )
        logger.info(
            "[band.tools] Added participant %s to room %s",
            _short_id(participant_id),
            _short_id(room_id),
        )
        return tool_result(
            {"success": True, "room_id": room_id, "participant_id": participant_id, "role": role}
        )
    except Exception as exc:
        return _tool_exc(exc)


async def _handle_remove_participant(args: dict, **kwargs) -> str:
    """Remove a participant (by UUID) from a room."""
    try:
        _authorize_band_action()
        if not _load_sdk():
            raise _ToolUnavailable("Band not available (band-sdk not installed)")
        rest = await _rest()
        room_id = _resolve_room(args)

        participant_id = str(args.get("participant_id") or "").strip()
        if not participant_id:
            return tool_error("participant_id is required")

        await rest.agent_api_participants.remove_agent_chat_participant(
            room_id,
            participant_id,
            request_options=DEFAULT_REQUEST_OPTIONS,
        )
        logger.info(
            "[band.tools] Removed participant %s from room %s",
            _short_id(participant_id),
            _short_id(room_id),
        )
        return tool_result(
            {"success": True, "room_id": room_id, "participant_id": participant_id}
        )
    except Exception as exc:
        return _tool_exc(exc)


async def _handle_get_participants(args: dict, **kwargs) -> str:
    """Read-only: list a room's participants (no owner gate)."""
    try:
        rest = await _rest()
        room_id = _resolve_room(args)
        participants = await _list_participants(rest, room_id)
        truncated = len(participants) > _MAX_RESULTS
        if truncated:
            participants = participants[:_MAX_RESULTS]
            logger.info(
                "[band.tools] get_participants truncated at %d for room %s",
                _MAX_RESULTS,
                _short_id(room_id),
            )
        return tool_result(
            {
                "success": True,
                "room_id": room_id,
                "participants": participants,
                "truncated": truncated,
            }
        )
    except Exception as exc:
        return _tool_exc(exc)


# ---------------------------------------------------------------------------
# JSON schemas
# ---------------------------------------------------------------------------

_STRING = {"type": "string"}
_ROOM_ID_PROP = {
    "type": "string",
    "description": "Target Band room id. Defaults to the current Band room; pass room_id to target another.",
}

BAND_CREATE_ROOM_SCHEMA = {
    "name": "band_create_room",
    "description": (
        "Create a new Band room. Use `person` + `message` to spin up a room and message "
        "someone in one step (the room title is derived by the server from the first "
        "message — there is no title argument)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "person": {
                "type": "string",
                "description": "Optional contact to add — a handle, name, or UUID (resolved via the directory).",
            },
            "message": {
                "type": "string",
                "description": "Optional first message to send (requires `person` to @mention).",
            },
            "role": {
                "type": "string",
                "enum": ["owner", "admin", "member"],
                "description": "Role for the added person (default: member).",
            },
        },
        "required": [],
    },
}

BAND_FIND_ROOM_SCHEMA = {
    "name": "band_find_room",
    "description": (
        "Find existing Band rooms by a free-text query matched against room title or id. "
        "Use this to get a room_id before acting on an existing room from another platform. "
        "Read-only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Text to match against room titles/ids. Omit to list rooms.",
            },
        },
        "required": [],
    },
}

BAND_FIND_CONTACT_SCHEMA = {
    "name": "band_find_contact",
    "description": (
        "Resolve a person to their Band participant UUID by a free-text query (handle, name, "
        "or id) over your peers and contacts. Use this before adding someone to a room or "
        "@mentioning them. Read-only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Handle, display name, or UUID to look up."},
        },
        "required": ["query"],
    },
}

BAND_SEND_MESSAGE_SCHEMA = {
    "name": "band_send_message",
    "description": (
        "Send a message to a Band room. Band requires at least one @mention per message: pass "
        "`mention_ids` to choose recipients, otherwise all non-agent participants are mentioned. "
        "Targets the current Band room by default; pass `room_id` to target another. To message "
        "your owner ('me' / 'the owner') from anywhere — including a non-Band session — omit "
        "`room_id`: with no current Band room the message goes to your owner's hub (home channel) "
        "and @mentions the owner."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "Message text to send."},
            "reply_to": {
                "type": "string",
                "description": (
                    "Id of the message you are answering. Preferred way to reply: "
                    "its author is addressed for you, with no lookup. Errors if "
                    "that message is not in the recent window."
                ),
            },
            "mention_ids": {
                "type": "array",
                "items": _STRING,
                "description": (
                    "Participant UUIDs to @mention. Use for fan-out, or to address "
                    "someone who has not spoken. Takes precedence over reply_to. "
                    "If both are omitted, every non-agent participant is mentioned."
                ),
            },
            "room_id": _ROOM_ID_PROP,
        },
        "required": ["content"],
    },
}

BAND_ADD_PARTICIPANT_SCHEMA = {
    "name": "band_add_participant",
    "description": (
        "Add a participant (by UUID) to a Band room. Defaults to the current Band room; pass "
        "`room_id` to target another."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "participant_id": {"type": "string", "description": "User/agent UUID to add."},
            "role": {
                "type": "string",
                "enum": ["owner", "admin", "member"],
                "description": "Role for the participant (default: member).",
            },
            "room_id": _ROOM_ID_PROP,
        },
        "required": ["participant_id"],
    },
}

BAND_REMOVE_PARTICIPANT_SCHEMA = {
    "name": "band_remove_participant",
    "description": (
        "Remove a participant (by UUID) from a Band room. Defaults to the current Band room; "
        "pass `room_id` to target another."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "participant_id": {"type": "string", "description": "User/agent UUID to remove."},
            "room_id": _ROOM_ID_PROP,
        },
        "required": ["participant_id"],
    },
}

BAND_GET_PARTICIPANTS_SCHEMA = {
    "name": "band_get_participants",
    "description": (
        "List the participants of a Band room. Defaults to the current Band room; pass `room_id` "
        "to target another. Read-only."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "room_id": _ROOM_ID_PROP,
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Registry tuple — consumed by adapter.register():
#   for name, schema, handler, emoji in BAND_TOOLS: ctx.register_tool(...)
# ---------------------------------------------------------------------------

BAND_TOOLS = (
    ("band_create_room", BAND_CREATE_ROOM_SCHEMA, _handle_create_room, "🏠"),
    ("band_find_room", BAND_FIND_ROOM_SCHEMA, _handle_find_room, "🔎"),
    ("band_find_contact", BAND_FIND_CONTACT_SCHEMA, _handle_find_contact, "👤"),
    ("band_send_message", BAND_SEND_MESSAGE_SCHEMA, _handle_send_message, "💬"),
    ("band_add_participant", BAND_ADD_PARTICIPANT_SCHEMA, _handle_add_participant, "➕"),
    ("band_remove_participant", BAND_REMOVE_PARTICIPANT_SCHEMA, _handle_remove_participant, "➖"),
    ("band_get_participants", BAND_GET_PARTICIPANTS_SCHEMA, _handle_get_participants, "👥"),
)
