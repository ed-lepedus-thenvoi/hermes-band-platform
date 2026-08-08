"""
Execution events — mirror the agent's tool calls into the Band room.

Where ``adapter.py`` relays *messages* and ``tools.py`` lets the agent *act on
Band*, this module makes the agent's work visible: a ``tool_call`` event when a
tool starts and a ``tool_result`` event when it finishes, posted to the room's
context stream.  Band's UI has an Event-type filter and a Work/Chat split, so a
user watching a long turn sees what the agent is doing instead of silence.

Why plugin hooks and not an adapter method
------------------------------------------
The host has no *adapter*-facing tool hook that a gateway-driven platform can
use:

  * ``BasePlatformAdapter.format_tool_event`` (``gateway/platforms/base.py``)
    is documented as the seam, but its only caller —
    ``gateway.stream_dispatch.GatewayEventDispatcher`` — is never instantiated
    in hermes-agent 0.19.0.  Overriding it is a no-op today.
  * ``agent.tool_start_callback`` is re-assigned per turn by
    ``gateway/run.py`` (to the Discord voice ack, else ``None``), and
    ``agent.tool_progress_callback`` to the gateway's own progress closure, so
    an adapter cannot own either.  ``agent.tool_complete_callback`` is only
    wired by ``gateway/platforms/api_server.py``, which runs the agent itself
    rather than through the gateway loop.
  * The only adapter call inside the gateway's tool lifecycle is
    ``set_status_text`` — a rendered phrase, with no args, output or call id.

The wired contract that *does* carry the full shape is the plugin hook API:
``pre_tool_call`` / ``post_tool_call`` (``hermes_cli/plugins.py`` VALID_HOOKS),
fired once per tool call by ``agent/tool_executor.py`` and
``model_tools.handle_function_call`` with ``tool_name``, ``args``, ``result``,
``tool_call_id``, ``session_id`` and ``status``.  We are the plugin, so we
register there and resolve the Band room from ``session_id``.

Those hooks are global — they fire for every tool call in the gateway,
including turns that came from Telegram or the CLI.  ``_room_for_session``
is therefore the load-bearing filter: no Band room for the session means no
event.

Safety
------
Tool args and output can carry credentials and can be enormous, and a Band
event cannot be deleted once written.  Every payload is run through the host's
own ``agent.redact.redact_sensitive_text`` (``force=True`` — the same
Tirith-grade redactor the gateway applies before chat text leaves it) and is
*dropped* rather than emitted if that redactor is unavailable, then capped with
the SDK's own head-and-tail truncation.  Emission is best-effort throughout: a
failing REST call must never break the tool or the reply.

Events carry no mentions (the API exempts them from the >=1 mention rule that
messages are held to), so however many are emitted, nobody is pinged.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, Optional

from gateway.config import Platform

# Reuse the adapter's REST defaults. Importing the adapter module is safe even
# when the SDK is absent (it guards its own import).
from .adapter import DEFAULT_REQUEST_OPTIONS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lazy SDK import guard.
#
# ``ChatEventRequest`` is the only request type this module constructs. Bound
# lazily — like the adapter's module-top guard — so the module imports cleanly
# when ``band-sdk`` isn't installed; ``_load_sdk()`` rebinds on demand so a late
# install is picked up without a process restart.
# ---------------------------------------------------------------------------
try:
    from band.client.rest import ChatEventRequest  # noqa: F401
except ImportError:
    ChatEventRequest = None

# Canonical taxonomy, imported *independently* of the guard above so an older
# band-sdk without ``band.core.types`` degrades to the same literals rather than
# disabling execution events entirely. These are the values the SDK's own
# adapters emit and its ``band.converters.parsing`` reads back.
try:
    from band.core.types import MessageType as _BandMessageType, ToolEventKey

    _TOOL_CALL = str(_BandMessageType.TOOL_CALL)
    _TOOL_RESULT = str(_BandMessageType.TOOL_RESULT)
    _K_NAME = str(ToolEventKey.NAME)
    _K_ARGS = str(ToolEventKey.ARGS)
    _K_OUTPUT = str(ToolEventKey.OUTPUT)
    _K_TOOL_CALL_ID = str(ToolEventKey.TOOL_CALL_ID)
    _K_IS_ERROR = str(ToolEventKey.IS_ERROR)
except ImportError:  # pragma: no cover - exercised only on an older SDK
    _TOOL_CALL = "tool_call"
    _TOOL_RESULT = "tool_result"
    _K_NAME = "name"
    _K_ARGS = "args"
    _K_OUTPUT = "output"
    _K_TOOL_CALL_ID = "tool_call_id"
    _K_IS_ERROR = "is_error"


# Event-content limits copied from the band SDK's ``band/runtime/tools.py``
# (``_EVENT_CONTENT_MAX_LENGTH`` / ``_EVENT_TRUNCATION_MARKER`` /
# ``_EVENT_EMPTY_CONTENT_PLACEHOLDER``). They are private to the SDK, so the
# behaviour is reproduced here rather than imported; keep the numbers in step if
# the SDK ever publishes them.
_EVENT_CONTENT_MAX_LENGTH = 16384
_EVENT_TRUNCATION_MARKER = "... [truncated] ..."
_EVENT_EMPTY_CONTENT_PLACEHOLDER = "(no content)"

# session_id → room_id memo. A Hermes session id is stable for the life of the
# session, so this resolves once per room instead of walking the session store
# on every tool call. Cleared wholesale when it grows past the cap — a miss is
# just a re-resolve, and a stale entry can only mis-address an event to a room
# the agent has since left (where the post fails harmlessly).
_SESSION_ROOM_CACHE: Dict[str, str] = {}
_SESSION_ROOM_CACHE_MAX = 512


def _load_sdk() -> bool:
    """(Re)bind ``ChatEventRequest``. Mirrors ``tools.py::_load_sdk``."""
    global ChatEventRequest

    if ChatEventRequest is not None:
        return True
    try:
        from ._band_libs import prepend_band_libs

        prepend_band_libs()
    except Exception:
        pass
    try:
        from band.client.rest import ChatEventRequest as _ChatEventRequest
    except ImportError:
        return False
    ChatEventRequest = _ChatEventRequest
    return True


def _truncate_event_content(content: str) -> str:
    """Cap *content*, keeping its head and tail around a marker.

    Byte-for-byte the band SDK's ``_truncate_event_content``: both ends are
    preserved because the tail of a truncated payload is often the informative
    part (the last lines of an error dump), which a head-only cut would drop.
    No-op when *content* already fits, so callers run it unconditionally.
    """
    if len(content) <= _EVENT_CONTENT_MAX_LENGTH:
        return content
    budget = _EVENT_CONTENT_MAX_LENGTH - len(_EVENT_TRUNCATION_MARKER)
    head_len = budget // 2
    tail_len = budget - head_len
    return content[:head_len] + _EVENT_TRUNCATION_MARKER + content[-tail_len:]


def _redact_tree(value: Any, scrub) -> Any:
    """Apply *scrub* to every string reachable in *value*.

    Redaction runs over the payload's values, NOT over the serialized JSON:
    the redactor masks a matched token and whatever trails it, so a credential
    sitting next to an escaped quote (``... Bearer sk-…\\" ...``) would take the
    escape with it and leave content that no longer parses as JSON — which is
    exactly what ``band.converters.parsing`` reads these events back as.

    Anything that isn't a JSON primitive is stringified and scrubbed here rather
    than left to ``json.dumps(default=str)``, which would render it *after* the
    redactor had run and so smuggle a raw value past it.
    """
    if isinstance(value, str):
        return scrub(value)
    if isinstance(value, dict):
        return {str(k): _redact_tree(v, scrub) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_redact_tree(v, scrub) for v in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return scrub(str(value))


def _redact_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Scrub credentials from an event payload, or None if that cannot be done.

    Delegates to ``agent.redact.redact_sensitive_text`` — the host's own
    redactor, the same one ``gateway/run.py`` runs over chat text before it
    leaves the gateway. ``force=True`` honors redaction even when
    ``security.redact_secrets`` is off, matching the gateway's reasoning for
    boundaries that must never emit raw credentials.

    The host does NOT scrub this path for us: ``redact_tool_args_for_display``
    (``agent/display.py``) only masks ``browser_type``'s ``text`` argument, and
    tool *output* is scrubbed per-tool at the tool boundary (``terminal``,
    ``process``) rather than universally. So this pass is load-bearing, not
    belt-and-braces — hence fail-*closed*: a Band event cannot be deleted once
    written, so an unavailable redactor drops the event instead of emitting raw.
    """
    try:
        from agent.redact import redact_sensitive_text

        return _redact_tree(
            payload, lambda text: redact_sensitive_text(text, force=True)
        )
    except Exception:
        logger.warning(
            "[band] execution event dropped — secret redaction unavailable"
        )
        return None


def _event_content(payload: Dict[str, Any]) -> Optional[str]:
    """Serialize an already-redacted payload, placeholder it, and truncate."""
    try:
        content = json.dumps(payload, ensure_ascii=False, default=str)
    except Exception:
        logger.debug("[band] execution event payload not serializable")
        return None
    if not content:
        content = _EVENT_EMPTY_CONTENT_PLACEHOLDER
    return _truncate_event_content(content)


def _output_text(result: Any) -> str:
    """Render a tool result as event text, placeholdering blank output.

    ``None`` is handled before serialization: ``json.dumps(None)`` is the string
    ``"null"``, which would read as real output rather than as nothing.
    """
    if result is None:
        return _EVENT_EMPTY_CONTENT_PLACEHOLDER
    if not isinstance(result, str):
        try:
            result = json.dumps(result, ensure_ascii=False, default=str)
        except Exception:
            result = str(result)
    return result or _EVENT_EMPTY_CONTENT_PLACEHOLDER


def _live_adapter() -> Optional[Any]:
    """The connected :class:`BandAdapter`, or None. Mirrors ``tools.py``."""
    try:
        from gateway.run import _gateway_runner_ref

        runner = _gateway_runner_ref()
        return runner.adapters.get(Platform("band")) if runner else None
    except Exception:
        return None


def _room_for_session(adapter: Any, session_id: str) -> Optional[str]:
    """The Band room whose Hermes session is *session_id*, or None.

    The tool hooks are global, so this is the filter that keeps a Telegram or
    CLI turn from writing events into a Band room: the session store maps each
    room's session key to a session id, and a session id we don't recognise
    isn't ours. Read-only and never raises.
    """
    if not session_id:
        return None
    cached = _SESSION_ROOM_CACHE.get(session_id)
    if cached is not None:
        return cached

    store = getattr(adapter, "_session_store", None)
    session_key_for = getattr(adapter, "_session_key_for", None)
    if store is None or session_key_for is None:
        return None
    try:
        store._ensure_loaded()
        entries = store._entries
    except Exception:
        return None

    rooms = set(getattr(adapter, "_known_rooms", None) or ())
    hub_room = getattr(adapter, "_hub_room_id", None)
    if hub_room:
        rooms.add(hub_room)

    for room_id in rooms:
        try:
            key = session_key_for(room_id)
            entry = entries.get(key) if key else None
        except Exception:
            continue
        if entry is not None and getattr(entry, "session_id", None) == session_id:
            if len(_SESSION_ROOM_CACHE) >= _SESSION_ROOM_CACHE_MAX:
                _SESSION_ROOM_CACHE.clear()
            _SESSION_ROOM_CACHE[session_id] = room_id
            return room_id
    return None


async def emit_event(
    adapter: Any, message_type: str, session_id: str, payload: Dict[str, Any]
) -> bool:
    """Post one execution event. Best-effort — never raises.

    Returns True when an event was posted, False when it was skipped or the
    post failed. ``metadata`` carries only the bounded correlation fields; the
    full ToolEventKey payload lives in ``content`` under the single documented
    size cap, which is the shape the band SDK's own adapters emit and its
    ``band.converters.parsing`` reads back.
    """
    try:
        room_id = _room_for_session(adapter, session_id)
        if room_id is None:
            return False
        safe = _redact_payload(payload)
        if safe is None:
            return False
        content = _event_content(safe)
        if content is None:
            return False
        link = getattr(adapter, "_link", None)
        if link is None or not _load_sdk():
            return False

        # Built from the redacted payload, so metadata can never carry a value
        # that content wouldn't.
        metadata = {_K_NAME: safe.get(_K_NAME, "")}
        if safe.get(_K_TOOL_CALL_ID):
            metadata[_K_TOOL_CALL_ID] = safe[_K_TOOL_CALL_ID]
        if _K_IS_ERROR in safe:
            metadata[_K_IS_ERROR] = safe[_K_IS_ERROR]

        # No mentions: the events endpoint is exempt from the >=1 mention rule
        # that create_agent_chat_message enforces, so emitting these can never
        # ping a participant however many tools a turn runs.
        await link.rest.agent_api_events.create_agent_chat_event(
            chat_id=room_id,
            event=ChatEventRequest(
                content=content,
                message_type=message_type,
                metadata=metadata,
            ),
            request_options=DEFAULT_REQUEST_OPTIONS,
        )
        return True
    except Exception as e:
        logger.debug(
            "[band] execution event (%s) dropped: %s", message_type, e
        )
        return False


def _schedule(message_type: str, session_id: str, payload: Dict[str, Any]) -> None:
    """Hand an event to the link's own loop from the agent's worker thread.

    The plugin hooks are synchronous and fire on the agent's tool-execution
    thread, while the REST client is bound to the loop ``connect()`` ran on —
    the same cross-loop problem ``BandAdapter.send`` solves. Fire-and-forget: we
    never wait on the future, so a slow or failing event cannot stall the tool.
    """
    try:
        adapter = _live_adapter()
        if adapter is None:
            return
        loop = getattr(adapter, "_link_loop", None)
        if loop is None or not loop.is_running():
            return
        asyncio.run_coroutine_threadsafe(
            emit_event(adapter, message_type, session_id, payload), loop
        )
    except Exception as e:
        logger.debug("[band] execution event (%s) not scheduled: %s", message_type, e)


# ---------------------------------------------------------------------------
# Plugin hook callbacks
#
# ``invoke_hook`` calls these with keyword arguments only, and always adds
# ``telemetry_schema_version`` — hence ``**_kw``. Both are pure observers: they
# return None, so ``_get_pre_tool_call_directive_details`` never reads a
# block/approve directive out of them.
# ---------------------------------------------------------------------------

def on_pre_tool_call(
    *,
    tool_name: str = "",
    args: Optional[Dict[str, Any]] = None,
    session_id: str = "",
    tool_call_id: str = "",
    **_kw: Any,
) -> None:
    """``pre_tool_call`` observer — emit a ``tool_call`` event."""
    if not tool_name:
        return None
    _schedule(
        _TOOL_CALL,
        session_id,
        {
            _K_NAME: tool_name,
            _K_ARGS: args if isinstance(args, dict) else {},
            _K_TOOL_CALL_ID: tool_call_id,
        },
    )
    return None


def on_post_tool_call(
    *,
    tool_name: str = "",
    result: Any = None,
    session_id: str = "",
    tool_call_id: str = "",
    status: Optional[str] = None,
    **_kw: Any,
) -> None:
    """``post_tool_call`` observer — emit a ``tool_result`` event.

    ``status`` is the host's own classification ("ok" / "error" / "blocked" /
    "cancelled"); anything but "ok" is an error. Recording it matters — without
    it the event replays as a success on the next rehydration, telling the model
    a failed operation worked.
    """
    if not tool_name:
        return None
    _schedule(
        _TOOL_RESULT,
        session_id,
        {
            _K_NAME: tool_name,
            _K_OUTPUT: _output_text(result),
            _K_TOOL_CALL_ID: tool_call_id,
            _K_IS_ERROR: str(status or "ok") != "ok",
        },
    )
    return None


def register_hooks(ctx) -> None:
    """Wire the tool observers. Best-effort — never breaks plugin load."""
    try:
        ctx.register_hook("pre_tool_call", on_pre_tool_call)
        ctx.register_hook("post_tool_call", on_post_tool_call)
    except AttributeError:
        # Older host without ctx.register_hook — no execution events, but the
        # platform still registers and works.
        logger.debug("[band] host has no ctx.register_hook; execution events off")
    except Exception as e:
        logger.debug("[band] execution-event hook registration skipped: %s", e)


__all__ = [
    "emit_event",
    "on_post_tool_call",
    "on_pre_tool_call",
    "register_hooks",
]
