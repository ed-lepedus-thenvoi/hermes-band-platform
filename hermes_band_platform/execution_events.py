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
*dropped* rather than emitted if that redactor is unavailable. Oversized data is
trimmed inside payload fields before final serialization, so the SDK's history
parser always receives valid JSON. Emission is best-effort throughout: a failing
REST call must never break the tool or the reply.

Execution-event publication is globally disabled by default. The explicit
``BAND_EMIT_EXECUTION`` gate accepts ``off``, ``hub`` (only turns originating in
the private owner hub), or ``all`` (all originating Band rooms).

Events carry no mentions (the API exempts them from the >=1 mention rule that
messages are held to), so however many are emitted, nobody is pinged.

Logging
-------
Every path out of this module says so: a successful post at ``debug`` (the
grep that answers "did that emit"), a failed post or a dropped event at
``warning`` (an operator would otherwise never learn the room lost data), and
the routine "this turn isn't a Band turn" filter at ``debug``, worded so it is
distinguishable from a *broken* room resolution.

NOTHING logs payload content. Tool args and tool output are exactly where
credentials live — the reason redaction here is fail-closed — so log sites
carry ids, types, counts and lengths only. On the pre-redaction path
(``_redact_payload``) even the exception *message* is withheld, since the
redactor is the one thing holding raw payload at the moment it raises.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
from typing import Any, Dict, Optional

from gateway.config import Platform

# Reuse the adapter's REST defaults and its id-shortening log helper. Importing
# the adapter module is safe even when the SDK is absent (it guards its own
# import).
from .adapter import DEFAULT_REQUEST_OPTIONS, _short_id

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
_EVENT_REQUIRED_FIELD_MAX_LENGTH = 1024
_ARGS_TRUNCATION_KEY = "_band_truncated"

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
    except Exception as e:
        logger.debug("[band] execution events: band-libs shim unusable: %s", e)
    try:
        from band.client.rest import ChatEventRequest as _ChatEventRequest
    except ImportError as e:
        logger.debug("[band] execution events off: band-sdk not importable: %s", e)
        return False
    ChatEventRequest = _ChatEventRequest
    return True


def _truncate_text(text: str, limit: int) -> str:
    """Head/tail truncate one JSON field to *limit* characters."""
    if len(text) <= limit:
        return text
    if limit <= len(_EVENT_TRUNCATION_MARKER):
        return _EVENT_TRUNCATION_MARKER[:limit]
    budget = limit - len(_EVENT_TRUNCATION_MARKER)
    head_len = budget // 2
    tail_len = budget - head_len
    return text[:head_len] + _EVENT_TRUNCATION_MARKER + text[-tail_len:]


def _serialize(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, default=str)


def _unique_truncation_key(mapping: Dict[str, Any]) -> str:
    """Return a deterministic marker key that cannot overwrite user args."""
    key = _ARGS_TRUNCATION_KEY
    suffix = 2
    while key in mapping:
        key = f"{_ARGS_TRUNCATION_KEY}_{suffix}"
        suffix += 1
    return key


def _fit_string_field(payload: Dict[str, Any], key: str) -> None:
    """Shrink one string field just enough for the serialized payload to fit."""
    value = payload.get(key)
    if not isinstance(value, str) or len(_serialize(payload)) <= _EVENT_CONTENT_MAX_LENGTH:
        return
    low, high = 0, len(value)
    best = ""
    while low <= high:
        mid = (low + high) // 2
        candidate = _truncate_text(value, mid)
        payload[key] = candidate
        if len(_serialize(payload)) <= _EVENT_CONTENT_MAX_LENGTH:
            best = candidate
            low = mid + 1
        else:
            high = mid - 1
    payload[key] = best


def _bounded_event_payload(payload: Dict[str, Any]) -> str:
    """Serialize a tool payload without ever cutting the resulting JSON.

    Payloads that already fit are byte-for-byte unchanged. For oversized tool
    results, ``output`` is head/tail truncated to the largest value that fits.
    For oversized tool calls, correlation strings are first bounded, then args
    are retained in insertion order while reserving a deterministic omission
    marker; the first entry that cannot fit and every following entry are
    omitted. This keeps ``args`` an object accepted by Band's history parsers.
    """
    content = _serialize(payload)
    if len(content) <= _EVENT_CONTENT_MAX_LENGTH:
        return content

    bounded = dict(payload)
    for key in (_K_NAME, _K_TOOL_CALL_ID):
        value = bounded.get(key)
        if isinstance(value, str):
            bounded[key] = _truncate_text(value, _EVENT_REQUIRED_FIELD_MAX_LENGTH)

    if isinstance(bounded.get(_K_ARGS), dict):
        original_args = bounded[_K_ARGS]
        kept: Dict[str, Any] = {}
        marker_key = _unique_truncation_key(original_args)
        items = list(original_args.items())
        bounded[_K_ARGS] = kept
        for index, (key, value) in enumerate(items):
            remaining = len(items) - index - 1
            kept[key] = value
            if remaining:
                kept[marker_key] = f"{remaining} argument(s) omitted"
            if len(_serialize(bounded)) <= _EVENT_CONTENT_MAX_LENGTH:
                if not remaining:
                    kept.pop(marker_key, None)
                continue
            kept.pop(key, None)
            kept[marker_key] = f"{remaining + 1} argument(s) omitted"
            break
    elif _K_OUTPUT in bounded:
        if not isinstance(bounded[_K_OUTPUT], str):
            bounded[_K_OUTPUT] = str(bounded[_K_OUTPUT])
        _fit_string_field(bounded, _K_OUTPUT)

    content = _serialize(bounded)
    if len(content) <= _EVENT_CONTENT_MAX_LENGTH:
        return content

    # Pathological required fields or a marker key can still consume the cap.
    # Shrink correlation fields against the real serialized size, then drop the
    # optional args marker as the final fail-safe. Required parser fields remain.
    for key in (_K_NAME, _K_TOOL_CALL_ID):
        _fit_string_field(bounded, key)
    content = _serialize(bounded)
    if len(content) > _EVENT_CONTENT_MAX_LENGTH and isinstance(
        bounded.get(_K_ARGS), dict
    ):
        bounded[_K_ARGS] = {}
        content = _serialize(bounded)
        for key in (_K_NAME, _K_TOOL_CALL_ID):
            _fit_string_field(bounded, key)
        content = _serialize(bounded)
    return content


def _redact_tree(value: Any, scrub) -> Any:
    """Apply *scrub* to every string and mapping key reachable in *value*.

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
        redacted: Dict[str, Any] = {}
        for original_key, item in value.items():
            base = str(scrub(str(original_key)))
            key = base
            suffix = 2
            while key in redacted:
                key = f"{base}#{suffix}"
                suffix += 1
            redacted[key] = _redact_tree(item, scrub)
        return redacted
    if isinstance(value, (list, tuple)):
        return [_redact_tree(v, scrub) for v in value]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return scrub(str(value))


def _redact_payload(
    payload: Dict[str, Any], *, message_type: str, room_id: str
) -> Optional[Dict[str, Any]]:
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

    The drop is logged at ``warning``: a silently discarded event is data the
    room never gets and nobody would otherwise discover. Only the exception
    *type* is logged, never its message — this is the one call site holding
    unredacted tool args / output, so a raised exception is the most plausible
    way raw payload could reach a log file.
    """
    try:
        from agent.redact import redact_sensitive_text

        return _redact_tree(
            payload, lambda text: redact_sensitive_text(text, force=True)
        )
    except Exception as e:
        logger.warning(
            "[band] Dropping %s event for room %s — secret redaction unavailable "
            "(%s); fail-closed, nothing was emitted",
            message_type,
            _short_id(room_id),
            type(e).__name__,
        )
        return None


def _event_content(
    payload: Dict[str, Any], *, message_type: str, room_id: str
) -> Optional[str]:
    """Serialize an already-redacted payload under the cap as valid JSON."""
    try:
        content = _bounded_event_payload(payload)
    except Exception as e:
        # ``default=str`` makes this near-unreachable, so reaching it means the
        # payload shape is wrong, not that a value was awkward — hence warning.
        # The payload is already redacted here and json.dumps errors name types,
        # not values, so the exception is safe to log.
        logger.warning(
            "[band] Dropping %s event for room %s — payload not serializable: %s",
            message_type,
            _short_id(room_id),
            e,
        )
        return None
    if not content:
        content = _EVENT_EMPTY_CONTENT_PLACEHOLDER
    return content


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
        except Exception as e:
            # A degradation, not a failure — the output still goes out, just
            # rendered by ``str``. Types only: this is raw, pre-redaction tool
            # output, so nothing about its *value* may be recorded.
            logger.debug(
                "[band] tool result of type %s is not JSON-serializable (%s); "
                "falling back to str()",
                type(result).__name__,
                type(e).__name__,
            )
            result = str(result)
    return result or _EVENT_EMPTY_CONTENT_PLACEHOLDER


def _live_adapter() -> Optional[Any]:
    """The connected :class:`BandAdapter`, or None. Mirrors ``tools.py``."""
    try:
        from gateway.run import _gateway_runner_ref

        runner = _gateway_runner_ref()
        if runner is None:
            logger.debug("[band] execution event skipped — no live gateway runner")
            return None
        return runner.adapters.get(Platform("band"))
    except Exception as e:
        logger.debug("[band] execution event skipped — no live Band adapter: %s", e)
        return None


def _room_for_session(adapter: Any, session_id: str) -> Optional[str]:
    """The Band room whose Hermes session is *session_id*, or None.

    The tool hooks are global, so this is the filter that keeps a Telegram or
    CLI turn from writing events into a Band room: the session store maps each
    room's session key to a session id, and a session id we don't recognise
    isn't ours. Read-only and never raises.

    Every ``None`` says *why* at ``debug``, because "no room" is both the
    normal outcome for a non-Band turn and what a broken resolution looks like,
    and an operator staring at a room with no events needs to tell those apart.
    Debug, not info: this runs on every tool call of every platform.
    """
    if not session_id:
        logger.debug("[band] no execution event: hook fired without a session id")
        return None
    cached = _SESSION_ROOM_CACHE.get(session_id)
    if cached is not None:
        return cached

    store = getattr(adapter, "_session_store", None)
    session_key_for = getattr(adapter, "_session_key_for", None)
    if store is None or session_key_for is None:
        # Structurally wrong rather than filtered — a connected BandAdapter
        # always has both. Debug all the same: it repeats per tool call.
        logger.debug(
            "[band] no execution event: adapter exposes no session store/key "
            "(store=%s, key_fn=%s)",
            store is not None,
            session_key_for is not None,
        )
        return None
    try:
        store._ensure_loaded()
        entries = store._entries
    except Exception as e:
        logger.debug("[band] no execution event: session store unreadable: %s", e)
        return None

    rooms = set(getattr(adapter, "_known_rooms", None) or ())
    hub_room = getattr(adapter, "_hub_room_id", None)
    if hub_room:
        rooms.add(hub_room)

    for room_id in rooms:
        try:
            key = session_key_for(room_id)
            entry = entries.get(key) if key else None
        except Exception as e:
            logger.debug(
                "[band] execution event: session key lookup failed for room %s: %s",
                _short_id(room_id),
                e,
            )
            continue
        if entry is not None and getattr(entry, "session_id", None) == session_id:
            if len(_SESSION_ROOM_CACHE) >= _SESSION_ROOM_CACHE_MAX:
                _SESSION_ROOM_CACHE.clear()
            _SESSION_ROOM_CACHE[session_id] = room_id
            return room_id
    logger.debug(
        "[band] no execution event: session %s maps to no Band room — a CLI or "
        "Telegram turn (searched %d room(s))",
        _short_id(session_id),
        len(rooms),
    )
    return None


def _scope_allows(adapter: Any, room_id: Optional[str] = None) -> bool:
    """Apply the process-wide execution-event privacy gate."""
    scope = str(getattr(adapter, "_execution_scope", "off") or "off").lower()
    if scope == "all":
        return True
    if scope == "hub":
        return bool(room_id and room_id == getattr(adapter, "_hub_room_id", None))
    return False


async def emit_event(
    adapter: Any, message_type: str, session_id: str, payload: Dict[str, Any]
) -> bool:
    """Post one execution event. Best-effort — never raises.

    Returns True when an event was posted, False when it was skipped or the
    post failed. ``metadata`` carries only the bounded correlation fields; the
    full ToolEventKey payload lives in ``content`` under the single documented
    size cap, which is the shape the band SDK's own adapters emit and its
    ``band.converters.parsing`` reads back.

    Every return is logged: the skips by the helper that decided them, the post
    itself at ``debug`` (message_type, room, resulting event id, content size)
    and a failed post at ``warning``. Sizes and ids only — never content.
    """
    room_id: Optional[str] = None
    try:
        room_id = _room_for_session(adapter, session_id)
        if room_id is None:
            # _room_for_session has already logged which case this was.
            return False
        if not _scope_allows(adapter, room_id):
            logger.debug(
                "[band] %s event for room %s not emitted — "
                "BAND_EMIT_EXECUTION scope does not allow this room",
                message_type,
                _short_id(room_id),
            )
            return False
        safe = _redact_payload(payload, message_type=message_type, room_id=room_id)
        if safe is None:
            return False
        content = _event_content(safe, message_type=message_type, room_id=room_id)
        if content is None:
            return False
        link = getattr(adapter, "_link", None)
        if link is None:
            logger.debug(
                "[band] %s event for room %s not emitted — adapter has no live link",
                message_type,
                _short_id(room_id),
            )
            return False
        if not _load_sdk():
            logger.debug(
                "[band] %s event for room %s not emitted — band-sdk unavailable",
                message_type,
                _short_id(room_id),
            )
            return False

        # Derive metadata from the exact bounded representation sent as content.
        # Correlation fields may themselves be pathological, so taking them from
        # ``safe`` would let them bypass the event-size cap applied above.
        bounded = json.loads(content)
        metadata = {_K_NAME: bounded.get(_K_NAME, "")}
        if bounded.get(_K_TOOL_CALL_ID):
            metadata[_K_TOOL_CALL_ID] = bounded[_K_TOOL_CALL_ID]
        if _K_IS_ERROR in bounded:
            metadata[_K_IS_ERROR] = bounded[_K_IS_ERROR]

        # No mentions: the events endpoint is exempt from the >=1 mention rule
        # that create_agent_chat_message enforces, so emitting these can never
        # ping a participant however many tools a turn runs.
        resp = await link.rest.agent_api_events.create_agent_chat_event(
            chat_id=room_id,
            event=ChatEventRequest(
                content=content,
                message_type=message_type,
                metadata=metadata,
            ),
            request_options=DEFAULT_REQUEST_OPTIONS,
        )
        # The one line that answers "did that emit?". Content is reported as a
        # length; the tool name is a registry identifier, not payload.
        logger.debug(
            "[band] Emitted %s event for tool %r to room %s (event id %s, %d chars)",
            message_type,
            metadata.get(_K_NAME) or "<unnamed>",
            _short_id(room_id),
            _short_id(getattr(getattr(resp, "data", None), "id", None)),
            len(content),
        )
        return True
    except Exception as e:
        # Warning, not debug: the room silently loses this event, and the POST
        # runs on already-redacted content, so the error text cannot carry raw
        # tool args or output.
        logger.warning(
            "[band] Failed to emit %s event to room %s: %s",
            message_type,
            _short_id(room_id),
            e,
        )
        return False


def _schedule(message_type: str, session_id: str, payload: Dict[str, Any]) -> None:
    """Hand an event to the link's own loop from the agent's worker thread.

    The plugin hooks are synchronous and fire on the agent's tool-execution
    thread, while the REST client is bound to the loop ``connect()`` ran on —
    the same cross-loop problem ``BandAdapter.send`` solves. Submissions are
    retained in the adapter's bounded pending set and never waited on here.
    """
    coroutine = None
    try:
        adapter = _live_adapter()
        if adapter is None:
            # _live_adapter has already logged why.
            return
        if str(getattr(adapter, "_execution_scope", "off") or "off").lower() == "off":
            return
        loop = getattr(adapter, "_link_loop", None)
        if loop is None or not loop.is_running():
            logger.debug(
                "[band] %s event not scheduled — link loop absent or stopped",
                message_type,
            )
            return
        pending = adapter._execution_pending
        lock = adapter._execution_pending_lock
        with lock:
            if not adapter._execution_accepting:
                return
            if len(pending) >= adapter._execution_pending_max:
                logger.warning(
                    "[band] Dropping %s event — pending emission cap (%d) reached",
                    message_type,
                    adapter._execution_pending_max,
                )
                return
            coroutine = emit_event(adapter, message_type, session_id, payload)
            try:
                future = asyncio.run_coroutine_threadsafe(coroutine, loop)
            except Exception:
                coroutine.close()
                coroutine = None
                raise
            pending.add(future)
        future.add_done_callback(
            lambda done: _emission_done(adapter, done, message_type)
        )
    except Exception as e:
        logger.debug("[band] execution event (%s) not scheduled: %s", message_type, e)


def _emission_done(adapter: Any, future: Any, message_type: str) -> None:
    """Release one retained submission and consume any unexpected exception."""
    try:
        with adapter._execution_pending_lock:
            adapter._execution_pending.discard(future)
        future.result()
    except concurrent.futures.CancelledError:
        logger.debug("[band] Pending %s event cancelled", message_type)
    except Exception as e:
        # emit_event is itself best-effort, but consume/log a future exception
        # if a later regression lets one escape. Type-only keeps this callback
        # fail-closed even if that future failed before payload redaction.
        logger.warning(
            "[band] Pending %s event failed (%s)", message_type, type(e).__name__
        )


async def cancel_pending_emissions(adapter: Any) -> None:
    """Stop submissions and cancel/drain retained futures without waiting on I/O."""
    adapter._execution_accepting = False
    lock = getattr(adapter, "_execution_pending_lock", None)
    pending = getattr(adapter, "_execution_pending", None)
    if lock is None or pending is None:
        return
    with lock:
        futures = list(pending)
    for future in futures:
        future.cancel()
    if futures:
        # Cancellation is queued onto the link loop. Yield once when disconnect
        # runs there so coroutine finally blocks can execute; never wait for a
        # slow REST operation or make shutdown depend on network progress.
        await asyncio.sleep(0)
    with lock:
        pending.difference_update(futures)


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
        logger.debug("[band] pre_tool_call hook carried no tool name — no event")
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
        logger.debug("[band] post_tool_call hook carried no tool name — no event")
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
        logger.debug(
            "[band] execution-event hooks registered; emission scope is per adapter"
        )
    except AttributeError:
        # Older host without ctx.register_hook — no execution events, but the
        # platform still registers and works. Expected, so debug.
        logger.debug("[band] host has no ctx.register_hook; execution events off")
    except Exception as e:
        # Anything else means the hooks are off on a host that should support
        # them: no tool_call/tool_result will EVER appear, for the whole
        # process. Once per plugin load, so warning costs nothing.
        logger.warning(
            "[band] Execution events off — hook registration failed: %s", e
        )


__all__ = [
    "cancel_pending_emissions",
    "emit_event",
    "on_post_tool_call",
    "on_pre_tool_call",
    "register_hooks",
]
