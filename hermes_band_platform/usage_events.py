"""Per-turn token-usage events for Band rooms.

Hermes computes token usage for every API call but never surfaces it to the
platform.  This module bridges the two: it accumulates the host's per-call
usage across a turn's tool loop and posts ONE aggregated event into the Band
room the turn ran in.

Where the numbers come from
--------------------------
The host fires two relevant plugin hooks (``hermes_cli/plugins.py::VALID_HOOKS``):

* ``post_api_request`` — once per *API call*, and the ONLY hook carrying token
  counts.  Its ``usage`` kwarg is the ``CanonicalUsage`` summary built by
  ``run_agent.py::AIAgent._usage_summary_for_api_request_hook``; the same
  numbers the host logs as ``API call #N: ... in=/out=/total=``
  (``agent/conversation_loop.py``).
* ``post_llm_call`` — once per successful, non-interrupted turn with a final
  response (``agent/turn_finalizer.py``).  Carries no usage at all.
* ``BandAdapter.on_processing_complete`` — the terminal path that also sees
  failed and cancelled turns, plus their originating room event.

So neither hook alone is enough: the per-turn hook has no numbers and the
hook with numbers fires per call.  We therefore accumulate on
``post_api_request`` keyed by ``(session_id, turn_id)`` — both hooks carry the
same pair — and atomically pop once from either terminal path.  A single real turn was
measured making 13 API calls, so per-call emission would be far too noisy for
a chat room; one event per turn is the unit a room participant can read.

Usage rides a ``task`` event, not a ``usage`` one
-------------------------------------------------
``MessageType.USAGE`` exists in the SDK but the backend's ``message_type``
whitelist rejects it today — ``ChatEventRequest.message_type`` is typed
``Literal['tool_call', 'tool_result', 'thought', 'error', 'task']``.  The SDK's
own answer (``band/core/types.py``) is to carry the counts in an accepted
``task`` event's structured metadata under a discriminator key, and it exposes
both halves as constants so the switch is a one-line flip on the SDK side when
the platform gains the type.  We import those constants rather than hardcoding
``MessageType.USAGE``, so this plugin follows the platform automatically:

    USAGE_EVENT_TYPE   -> MessageType.TASK  (today)
    USAGE_METADATA_KEY -> "band_usage"      (what a reader would filter on)

Emission is OFF by default, though — the SDK justifies riding a ``task`` event
on the grounds that the read side filters on ``band_usage``, and there is no
read side yet.  See ``_scope()``.

Everything here is best-effort: a failed emit, an unresolvable room or a
missing SDK must never affect the turn.  Events are exempt from Band's
@mention requirement, so no mentions are ever attached.  Emitted events cannot
feed the agent its own telemetry — ``adapter._seedable_text`` drops every
context item whose ``message_type`` is not ``text`` before rehydration.

Logging
-------
Every reason a turn produces no usage event is stated at ``debug`` — "no usage
reported", "not a Band room", "suppressed by BAND_EMIT_USAGE" and a broken
resolution all look identical from the outside otherwise.  A failed POST is
``warning``.  The one path that stays silent on purpose is the platform filter
in ``on_post_api_request``: it fires on every API call of every platform, and
a log there would be pure noise.

Nothing logged here is payload — token counts, call counts, model names, room
ids.  The hooks never see message or tool content in the first place.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
import time
import weakref
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy SDK import guard — same contract as adapter.py's: the module must import
# cleanly with no band-sdk present (the gateway discovers plugins before deps
# are guaranteed), so a miss degrades to "never register the hooks" rather than
# breaking plugin load.  ``USAGE_EVENT_TYPE``/``USAGE_METADATA_KEY`` were added
# alongside ``TurnUsage``; an older SDK simply has no usage contract to follow.
# ---------------------------------------------------------------------------
ChatEventRequest = None
DEFAULT_REQUEST_OPTIONS = None
USAGE_EVENT_TYPE = None
USAGE_METADATA_KEY = None
TurnUsage = None
USAGE_SDK_AVAILABLE = False


def _ensure_sdk_bindings() -> bool:
    """Bind the SDK usage contract lazily, retrying after an early miss.

    Directory plugins can be discovered before ``band-libs`` is installed or
    prepended to ``sys.path``.  A one-shot module import would permanently
    disable hook registration in that process even after dependencies become
    available, so every registration attempt may retry the binding.
    """
    global ChatEventRequest, DEFAULT_REQUEST_OPTIONS
    global USAGE_EVENT_TYPE, USAGE_METADATA_KEY, TurnUsage, USAGE_SDK_AVAILABLE

    if USAGE_SDK_AVAILABLE:
        return True
    try:
        from band.client.rest import (
            ChatEventRequest as _ChatEventRequest,
            DEFAULT_REQUEST_OPTIONS as _DEFAULT_REQUEST_OPTIONS,
        )
        from band.core.types import (
            USAGE_EVENT_TYPE as _USAGE_EVENT_TYPE,
            USAGE_METADATA_KEY as _USAGE_METADATA_KEY,
            TurnUsage as _TurnUsage,
        )
    except Exception:
        return False
    ChatEventRequest = _ChatEventRequest
    DEFAULT_REQUEST_OPTIONS = _DEFAULT_REQUEST_OPTIONS
    USAGE_EVENT_TYPE = _USAGE_EVENT_TYPE
    USAGE_METADATA_KEY = _USAGE_METADATA_KEY
    TurnUsage = _TurnUsage
    USAGE_SDK_AVAILABLE = True
    return True


_ensure_sdk_bindings()


PLATFORM_NAME = "band"

# Scope values for BAND_EMIT_USAGE.  Band has no per-room event-visibility
# control and the host has no per-room plugin config, so scoping can only be
# expressed as a global policy — see _scope().
SCOPE_ALL = "all"
SCOPE_HUB = "hub"
SCOPE_OFF = "off"
_SCOPES = (SCOPE_ALL, SCOPE_HUB, SCOPE_OFF)
_TRUE_ALIASES = {"1", "true", "yes", "on"}
_FALSE_ALIASES = {"0", "false", "no"}
_STARTUP_SCOPE: Optional[str] = None

# Mirrors the SDK's own event-content cap (band/runtime/tools.py).  Our content
# is a short generated line, so truncation is a backstop, not an expected path;
# it exists so this module cannot become the one event source that posts an
# oversized payload if the format ever grows.
_EVENT_CONTENT_MAX_LENGTH = 16384
_EVENT_TRUNCATION_MARKER = "... [truncated] ..."
_EVENT_EMPTY_CONTENT_PLACEHOLDER = "(no content)"

# Caps remain a defensive backstop for process crashes and hosts that do not
# provide the processing-complete lifecycle hook.
_PENDING_MAX = 64
_PENDING_TTL_SECONDS = 3600.0


# ---------------------------------------------------------------------------
# Live adapters
# ---------------------------------------------------------------------------
# Weak so a discarded adapter (reconnect churn, tests) drops out on its own and
# this module never keeps a dead gateway alive.
_ACTIVE_ADAPTERS: "weakref.WeakSet[Any]" = weakref.WeakSet()
_ADAPTERS_LOCK = threading.Lock()


def track_adapter(adapter: Any) -> None:
    """Register a live ``BandAdapter`` as a possible emit target."""
    with _ADAPTERS_LOCK:
        _ACTIVE_ADAPTERS.add(adapter)


def untrack_adapter(adapter: Any) -> None:
    """Remove an adapter immediately when its connection is no longer live."""
    with _ADAPTERS_LOCK:
        _ACTIVE_ADAPTERS.discard(adapter)


def _adapters() -> list:
    with _ADAPTERS_LOCK:
        return list(_ACTIVE_ADAPTERS)


def _adapter_is_live(adapter: Any) -> bool:
    loop = getattr(adapter, "_link_loop", None)
    return bool(
        getattr(adapter, "_link", None) is not None
        and loop is not None
        and loop.is_running()
    )


def _live_adapters() -> list:
    return [adapter for adapter in _adapters() if _adapter_is_live(adapter)]


# ---------------------------------------------------------------------------
# Per-turn accumulation
# ---------------------------------------------------------------------------


@dataclass
class _Bucket:
    """Running totals for one in-flight turn."""

    usage: Any = field(default_factory=lambda: TurnUsage())
    api_calls: int = 0
    model: str = ""
    provider: str = ""
    created_at: float = field(default_factory=time.time)


_PENDING: Dict[Tuple[str, str], _Bucket] = {}
_PENDING_LOCK = threading.Lock()


def _scope() -> str:
    """Which rooms usage events go to.  ``BAND_EMIT_USAGE``, default ``off``.

    This is startup configuration: ``off`` skips hook registration entirely,
    so changing the value requires a gateway restart.

      off (default) — never emit; the hooks are not registered at all.
      all           — emit into the room the turn ran in.
      hub           — emit only in the owner's private control room, for
                      operators who do not want spend shown to the other
                      participants of a shared group room.

    WHY OFF BY DEFAULT.  Not because the event shape is wrong: it follows the
    SDK's usage contract exactly and stays forward-compatible with it.  But the
    SDK justifies riding a ``task`` event on the grounds that "the read side
    filters on that key" — and as of 2026-08 no read side exists.  ``band_usage``
    appears nowhere in the Jam client; the per-agent meters that look like usage
    are provider ACCOUNT rate-limit windows owned by the Jam daemon, unrelated
    to per-turn spend.  So the event's only observable effect today is a "Task"
    chip in the room transcript reading ``Token usage: input=… output=…``, and
    ``task`` is meant for work items and coordination, not telemetry.  Opt-in
    until that changes.

    WHEN TO FLIP IT BACK.  Once a consumer lands, ``all`` is the right default —
    per-room visibility is the point of the feature, and a usage event is
    strictly less revealing than the tool_call/tool_result events already posted
    to every room.  The upstream signal is ``USAGE_EVENT_TYPE`` becoming
    ``MessageType.USAGE``: ``band/core/types.py`` documents that as a one-line
    flip, and it can only happen once the backend accepts the type, which is
    itself the point at which a first-class reader is plausible.

    Band exposes no per-room switch for event visibility (the SDK's own gate,
    ``Emit.USAGE``, is a single adapter-wide flag) and the Hermes plugin API has
    no per-room config, so this can only ever be one global policy.
    """
    raw = (os.getenv("BAND_EMIT_USAGE") or "").strip().lower()
    if raw in _SCOPES:
        return raw
    # Accept the usual boolean spellings so "true"/"false" do the obvious thing.
    if raw in _TRUE_ALIASES:
        return SCOPE_ALL
    if raw in _FALSE_ALIASES:
        return SCOPE_OFF
    # Anything else — unset, or a typo that must not silently start emitting.
    return SCOPE_OFF


def _effective_scope() -> str:
    """Use the scope captured during hook registration, if registration ran."""
    return _STARTUP_SCOPE if _STARTUP_SCOPE is not None else _scope()


def _platform_value(value: Any) -> str:
    """Normalize a hook's ``platform`` kwarg (str or Platform enum) to a str."""
    return str(getattr(value, "value", value) or "").strip().lower()


def _turn_key(kwargs: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    """Correlation key shared by ``post_api_request`` and ``post_llm_call``.

    ``turn_id`` alone is not enough (it is per-agent-run, not globally unique)
    and ``session_id`` alone would merge consecutive turns, so both are used.
    Without a session id there is nothing to correlate against and nothing to
    resolve a room from, so the turn is skipped.
    """
    session_id = str(kwargs.get("session_id") or "").strip()
    if not session_id:
        return None
    return (session_id, str(kwargs.get("turn_id") or "").strip())


def _prune_locked(now: float) -> None:
    """Drop expired then oldest buckets. Caller holds ``_PENDING_LOCK``."""
    for key in [k for k, b in _PENDING.items() if now - b.created_at > _PENDING_TTL_SECONDS]:
        _PENDING.pop(key, None)
    while len(_PENDING) > _PENDING_MAX:
        oldest = min(_PENDING, key=lambda k: _PENDING[k].created_at)
        _PENDING.pop(oldest, None)


def on_post_api_request(**kwargs: Any) -> None:
    """Accumulate one API call's tokens into its turn's bucket."""
    if not _ensure_sdk_bindings():
        return
    if _platform_value(kwargs.get("platform")) != PLATFORM_NAME:
        # Deliberately silent: this fires on every API call of every platform
        # the gateway runs, and is the definition of a tight loop.
        return
    key = _turn_key(kwargs)
    if key is None:
        logger.debug("[band] usage: API call carried no session id — not counted")
        return
    usage = kwargs.get("usage")
    if not isinstance(usage, dict):
        # No usage reported (streaming provider that omits it, an error path).
        # Recording a zero call would understate nothing but inflate the call
        # count, so skip it entirely.
        logger.debug(
            "[band] usage: API call reported no usage (%s) — not counted",
            type(usage).__name__,
        )
        return

    # CanonicalUsage already excludes cache tokens from ``input_tokens`` on
    # every provider path, matching TurnUsage's convention.  ``reasoning`` is
    # deliberately NOT passed: the host reads it from
    # ``output_tokens_details``/``completion_tokens_details``, where the
    # provider counts it INSIDE ``output_tokens`` — folding it again would
    # double-count on every reasoning model.
    call = TurnUsage.from_mapping(
        usage,
        input="input_tokens",
        output="output_tokens",
        cache_read="cache_read_tokens",
        cache_write="cache_write_tokens",
    )

    now = time.time()
    with _PENDING_LOCK:
        bucket = _PENDING.get(key)
        if bucket is None:
            bucket = _Bucket(created_at=now)
            _PENDING[key] = bucket
            _prune_locked(now)
        bucket.usage = bucket.usage + call
        bucket.api_calls += 1
        bucket.model = str(kwargs.get("model") or bucket.model or "")
        bucket.provider = str(kwargs.get("provider") or bucket.provider or "")


def on_post_llm_call(**kwargs: Any) -> None:
    """Flush a completed turn's accumulated usage as one Band event."""
    key = _turn_key(kwargs)
    if key is None:
        logger.debug("[band] usage: turn end carried no session id — nothing to flush")
        return
    _flush_key(key)


def flush_incomplete_turn(adapter: Any, session_id: str, room_id: str) -> None:
    """Flush one failed/cancelled turn from its originating Band event.

    Hermes serializes turns per resolved session.  Select the newest bucket for
    this exact session without combining buckets; room and adapter resolution
    are repeated at schedule and emit time so reconnect churn cannot target a
    disconnected instance.
    """
    if not session_id or not room_id:
        return
    with _PENDING_LOCK:
        candidates = [key for key in _PENDING if key[0] == session_id]
        key = max(candidates, key=lambda item: _PENDING[item].created_at, default=None)
    if key is None:
        logger.debug("[band] usage: no accumulated usage for this turn — no event")
        return
    _flush_key(key, originating_adapter=adapter, originating_room=room_id)


def _flush_key(
    key: Tuple[str, str],
    *,
    originating_adapter: Any = None,
    originating_room: Optional[str] = None,
) -> None:
    """Atomically pop and emit one bucket; all terminal paths converge here."""
    with _PENDING_LOCK:
        bucket = _PENDING.pop(key, None)
    if bucket is None:
        # Routine for a non-Band turn (nothing was ever accumulated) and for a
        # turn whose bucket was pruned; both are normal, hence debug.
        logger.debug("[band] usage: no accumulated usage for this turn — no event")
        return
    # An all-zero total means nothing was actually reported; emitting it would
    # look like a real measurement of zero spend (the SDK skips it too).
    if bucket.usage.is_empty:
        logger.debug(
            "[band] usage: turn totalled zero tokens over %d API call(s) — no event",
            bucket.api_calls,
        )
        return

    scope = _effective_scope()
    if scope == SCOPE_OFF:
        logger.debug("[band] usage: suppressed by BAND_EMIT_USAGE=off")
        return

    session_id = key[0]
    adapters = _live_adapters()
    # Prefer a still-live originating adapter, but never retain a disconnected
    # one across reconnect.  The room id remains authoritative and is not
    # rerouted to another room.
    if originating_adapter in adapters:
        adapters.remove(originating_adapter)
        adapters.insert(0, originating_adapter)
    for adapter in adapters:
        room_id = originating_room or _room_for_session(adapter, session_id)
        if not room_id:
            continue
        if originating_room and _room_for_session(adapter, session_id) != room_id:
            continue
        if scope == SCOPE_HUB and room_id != getattr(adapter, "_hub_room_id", None):
            logger.debug(
                "[band] usage: room %s is not the hub — suppressed by "
                "BAND_EMIT_USAGE=hub",
                room_id,
            )
            return
        _schedule_emit(session_id, room_id, bucket)
        return
    # Correctly filtered, not broken: the turn ran on the CLI or another
    # platform, so no Band adapter owns its session.
    logger.debug(
        "[band] usage: session maps to no Band room across %d adapter(s) — "
        "not a Band turn",
        len(adapters),
    )


# ---------------------------------------------------------------------------
# Room resolution
# ---------------------------------------------------------------------------


def _room_for_session(adapter: Any, session_id: str) -> Optional[str]:
    """The Band room a Hermes ``session_id`` belongs to, or None.

    The hooks carry the agent's ``session_id`` (``SessionEntry.session_id``),
    not the session *key* — and only the key holds the room.  The store maps
    key -> entry, so we find the entry with this session id and then match its
    key against each known room's derived key.

    Prefix-matching rather than equality is deliberate: with
    ``BAND_GROUP_SESSIONS_PER_USER=true`` the real key carries a trailing
    ``:<user_id>`` that ``_session_key_for`` (which knows only the room) never
    produces.  Room ids are UUIDs, so a false prefix match is not a practical
    concern.
    """
    store = getattr(adapter, "_session_store", None)
    if not store or not session_id:
        logger.debug(
            "[band] usage: cannot resolve a room (session store present=%s, "
            "session id present=%s)",
            bool(store),
            bool(session_id),
        )
        return None
    try:
        ensure = getattr(store, "_ensure_loaded", None)
        if callable(ensure):
            ensure()
        entries = getattr(store, "_entries", None) or {}
        matched_key = next(
            (k for k, e in entries.items() if getattr(e, "session_id", None) == session_id),
            None,
        )
        if not matched_key:
            return None
        for room_id in list(getattr(adapter, "_known_rooms", None) or ()):
            base = adapter._session_key_for(room_id)
            if base and (matched_key == base or matched_key.startswith(base + ":")):
                return room_id
    except Exception as e:
        logger.debug("[band] usage: could not resolve room for session: %s", e)
    return None


# ---------------------------------------------------------------------------
# Emission
# ---------------------------------------------------------------------------


def _truncate_event_content(content: str) -> str:
    """Cap *content*, keeping head and tail around a marker (SDK behaviour)."""
    if len(content) <= _EVENT_CONTENT_MAX_LENGTH:
        return content
    budget = _EVENT_CONTENT_MAX_LENGTH - len(_EVENT_TRUNCATION_MARKER)
    head_len = budget // 2
    return content[:head_len] + _EVENT_TRUNCATION_MARKER + content[-(budget - head_len):]


def build_content(bucket: _Bucket) -> str:
    """Human-readable one-liner for the event body.

    Opens with the SDK's own ``Token usage: input=... output=...`` prefix so a
    reader written against ``SimpleAdapter.emit_usage`` sees the same shape,
    then adds the cache split and which model spent it.
    """
    usage = bucket.usage
    parts = [
        f"Token usage: input={usage.input_tokens}",
        f"output={usage.output_tokens}",
        f"cache_read={usage.cache_read_tokens}",
        f"cache_write={usage.cache_write_tokens}",
    ]
    suffix = [f"{bucket.api_calls} API call{'' if bucket.api_calls == 1 else 's'}"]
    if bucket.model:
        suffix.append(bucket.model)
    content = " ".join(parts) + " (" + ", ".join(suffix) + ")"
    if not content:  # pragma: no cover - the format above is never empty
        content = _EVENT_EMPTY_CONTENT_PLACEHOLDER
    return _truncate_event_content(content)


def build_metadata(bucket: _Bucket) -> Dict[str, Any]:
    """Structured payload; ``USAGE_METADATA_KEY`` is what a reader filters on."""
    metadata: Dict[str, Any] = {
        USAGE_METADATA_KEY: bucket.usage.to_dict(),
        "api_calls": bucket.api_calls,
    }
    if bucket.model:
        metadata["model"] = bucket.model
    if bucket.provider:
        metadata["provider"] = bucket.provider
    return metadata


def _current_adapter(session_id: str, room_id: str) -> Any:
    """Return the live adapter that currently owns this session and room."""
    for adapter in _live_adapters():
        if _room_for_session(adapter, session_id) == room_id:
            return adapter
    return None


def _schedule_emit(session_id: str, room_id: str, bucket: _Bucket) -> bool:
    """Resolve the current adapter and fire onto its own link loop; never wait.

    The hooks run synchronously on the agent's thread, while the link's asyncio
    primitives are bound to the loop ``connect()`` ran on (the same constraint
    ``BandAdapter.send`` handles).  We hand the coroutine to that loop and
    return immediately, so a slow or failing Band API call cannot add latency
    to — or raise into — the turn that produced it.
    """
    adapter = _current_adapter(session_id, room_id)
    if adapter is None:
        logger.debug(
            "[band] usage: dropping event for room %s — no live adapter owns it",
            room_id,
        )
        return False
    link_loop = adapter._link_loop
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        # Not an error and not swallowed: "no running loop" is the answer this
        # asks for — the hooks fire on the agent's thread. Nothing to report.
        running = None
    coro = _emit(session_id, room_id, bucket)
    if running is link_loop:
        link_loop.create_task(coro)
    else:
        asyncio.run_coroutine_threadsafe(coro, link_loop)
    return True


async def _emit(session_id: str, room_id: str, bucket: _Bucket) -> None:
    """Re-resolve the current adapter, then post. Swallows every failure."""
    adapter = _current_adapter(session_id, room_id)
    if adapter is None:
        logger.debug(
            "[band] usage: dropping event for room %s — no live adapter owns it",
            room_id,
        )
        return
    link_loop = adapter._link_loop
    if asyncio.get_running_loop() is not link_loop:
        asyncio.run_coroutine_threadsafe(_emit(session_id, room_id, bucket), link_loop)
        return
    link = adapter._link
    try:
        resp = await link.rest.agent_api_events.create_agent_chat_event(
            chat_id=room_id,
            event=ChatEventRequest(
                content=build_content(bucket),
                # Events carry no mentions — the API exempts them from the
                # >=1-mention rule that ordinary messages must satisfy.
                message_type=USAGE_EVENT_TYPE,
                metadata=build_metadata(bucket),
            ),
            request_options=DEFAULT_REQUEST_OPTIONS,
        )
        # Counts only — the hooks never carry message or tool content.
        logger.debug(
            "[band] Emitted usage event to room %s (event id %s, %d API call(s), "
            "%d in / %d out)",
            room_id,
            getattr(getattr(resp, "data", None), "id", None),
            bucket.api_calls,
            bucket.usage.input_tokens,
            bucket.usage.output_tokens,
        )
    except Exception as e:
        logger.warning("[band] Usage event failed for room %s: %s", room_id, e)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_hooks(ctx: Any) -> bool:
    """Wire the usage hooks. Returns whether they were registered.

    Skipped when the SDK predates the usage contract, when the host is too old
    to have ``register_hook``, or when the scope is ``off`` — which is the
    DEFAULT (see ``_scope()``), so the normal state of this plugin is to
    register nothing.  That matters: ``post_api_request`` makes the host build
    a sanitized response payload for every API call it would otherwise skip, and
    no one should pay that for a feature they did not opt into.
    """
    global _STARTUP_SCOPE

    if not _ensure_sdk_bindings():
        logger.debug("[band] usage events disabled: SDK has no usage contract")
        return False
    if _STARTUP_SCOPE is None:
        _STARTUP_SCOPE = _scope()
    if _STARTUP_SCOPE == SCOPE_OFF:
        logger.debug(
            "[band] usage events off (default); set BAND_EMIT_USAGE=all to enable"
        )
        return False
    register = getattr(ctx, "register_hook", None)
    if not callable(register):
        logger.debug("[band] usage events off: host has no ctx.register_hook")
        return False
    register("post_api_request", on_post_api_request)
    register("post_llm_call", on_post_llm_call)
    logger.debug("[band] usage events on (post_api_request/post_llm_call)")
    return True
