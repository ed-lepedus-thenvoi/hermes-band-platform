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
* ``post_llm_call`` — once per *turn*, after the tool-calling loop finishes
  (``agent/turn_finalizer.py``).  Carries no usage at all.

So neither hook alone is enough: the per-turn hook has no numbers and the
hook with numbers fires per call.  We therefore accumulate on
``post_api_request`` keyed by ``(session_id, turn_id)`` — both hooks carry the
same pair — and flush once on ``post_llm_call``.  A single real turn was
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
    USAGE_METADATA_KEY -> "band_usage"      (what a reader filters on)

Everything here is best-effort: a failed emit, an unresolvable room or a
missing SDK must never affect the turn.  Events are exempt from Band's
@mention requirement, so no mentions are ever attached.  Emitted events cannot
feed the agent its own telemetry — ``adapter._seedable_text`` drops every
context item whose ``message_type`` is not ``text`` before rehydration.
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
try:
    from band.client.rest import ChatEventRequest, DEFAULT_REQUEST_OPTIONS
    from band.core.types import (
        USAGE_EVENT_TYPE,
        USAGE_METADATA_KEY,
        TurnUsage,
    )

    USAGE_SDK_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only without the SDK
    ChatEventRequest = None  # type: ignore[assignment]
    DEFAULT_REQUEST_OPTIONS = None  # type: ignore[assignment]
    USAGE_EVENT_TYPE = None  # type: ignore[assignment]
    USAGE_METADATA_KEY = None  # type: ignore[assignment]
    TurnUsage = None  # type: ignore[assignment]
    USAGE_SDK_AVAILABLE = False


PLATFORM_NAME = "band"

# Scope values for BAND_EMIT_USAGE.  Band has no per-room event-visibility
# control and the host has no per-room plugin config, so scoping can only be
# expressed as a global policy — see _scope().
SCOPE_ALL = "all"
SCOPE_HUB = "hub"
SCOPE_OFF = "off"
_SCOPES = (SCOPE_ALL, SCOPE_HUB, SCOPE_OFF)

# Mirrors the SDK's own event-content cap (band/runtime/tools.py).  Our content
# is a short generated line, so truncation is a backstop, not an expected path;
# it exists so this module cannot become the one event source that posts an
# oversized payload if the format ever grows.
_EVENT_CONTENT_MAX_LENGTH = 16384
_EVENT_TRUNCATION_MARKER = "... [truncated] ..."
_EVENT_EMPTY_CONTENT_PLACEHOLDER = "(no content)"

# A turn that is interrupted, or fails before producing a final response, never
# reaches post_llm_call — its bucket is never flushed.  Both caps bound that
# leak: whichever trips first, stale buckets are dropped, not accumulated.
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


def _adapters() -> list:
    with _ADAPTERS_LOCK:
        return list(_ACTIVE_ADAPTERS)


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
    """Which rooms usage events go to.

    Read per call so a change takes effect without a gateway restart, and so
    the policy is testable without re-registering hooks.

    Band exposes no per-room switch for event visibility (the SDK's own gate,
    ``Emit.USAGE``, is a single adapter-wide flag), and the Hermes plugin API
    has no per-room config.  So the honest choice is a global default plus an
    escape hatch:

      all  (default) — emit into the room the turn ran in.  "Token spend per
                       room" is the point of the feature, and a usage event is
                       strictly less revealing than the tool_call/tool_result
                       events the plugin already posts to every room.
      hub            — emit only in the owner's private control room, for
                       operators who do not want spend shown to the other
                       participants of a shared group room.
      off            — never emit; the hooks are not registered at all.
    """
    raw = (os.getenv("BAND_EMIT_USAGE") or "").strip().lower()
    if raw in _SCOPES:
        return raw
    # Accept the usual boolean spellings so "false" does the obvious thing.
    if raw in {"0", "false", "no", "none"}:
        return SCOPE_OFF
    return SCOPE_ALL


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
    if _platform_value(kwargs.get("platform")) != PLATFORM_NAME:
        return
    key = _turn_key(kwargs)
    if key is None:
        return
    usage = kwargs.get("usage")
    if not isinstance(usage, dict):
        # No usage reported (streaming provider that omits it, an error path).
        # Recording a zero call would understate nothing but inflate the call
        # count, so skip it entirely.
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
        return
    with _PENDING_LOCK:
        bucket = _PENDING.pop(key, None)
    if bucket is None:
        return
    # An all-zero total means nothing was actually reported; emitting it would
    # look like a real measurement of zero spend (the SDK skips it too).
    if bucket.usage.is_empty:
        return

    scope = _scope()
    if scope == SCOPE_OFF:
        return

    session_id = key[0]
    for adapter in _adapters():
        room_id = _room_for_session(adapter, session_id)
        if not room_id:
            continue
        if scope == SCOPE_HUB and room_id != getattr(adapter, "_hub_room_id", None):
            return
        _schedule_emit(adapter, room_id, bucket)
        return


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


def _schedule_emit(adapter: Any, room_id: str, bucket: _Bucket) -> bool:
    """Fire the emit onto the adapter's own link loop; never wait for it.

    The hooks run synchronously on the agent's thread, while the link's asyncio
    primitives are bound to the loop ``connect()`` ran on (the same constraint
    ``BandAdapter.send`` handles).  We hand the coroutine to that loop and
    return immediately, so a slow or failing Band API call cannot add latency
    to — or raise into — the turn that produced it.
    """
    link_loop = getattr(adapter, "_link_loop", None)
    if link_loop is None or not link_loop.is_running():
        return False
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None
    coro = _emit(adapter, room_id, bucket)
    if running is link_loop:
        link_loop.create_task(coro)
    else:
        asyncio.run_coroutine_threadsafe(coro, link_loop)
    return True


async def _emit(adapter: Any, room_id: str, bucket: _Bucket) -> None:
    """Post the usage event. Swallows every failure by design."""
    link = getattr(adapter, "_link", None)
    if link is None:
        return
    try:
        await link.rest.agent_api_events.create_agent_chat_event(
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
    except Exception as e:
        logger.debug("[band] usage event failed for room %s: %s", room_id, e)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def register_hooks(ctx: Any) -> bool:
    """Wire the usage hooks. Returns whether they were registered.

    Skipped when the SDK predates the usage contract, when the host is too old
    to have ``register_hook``, or when the operator turned usage off — the last
    one matters because registering ``post_api_request`` makes the host build a
    sanitized response payload for every API call it would otherwise skip.
    """
    if not USAGE_SDK_AVAILABLE:
        logger.debug("[band] usage events disabled: SDK has no usage contract")
        return False
    if _scope() == SCOPE_OFF:
        logger.debug("[band] usage events disabled by BAND_EMIT_USAGE")
        return False
    register = getattr(ctx, "register_hook", None)
    if not callable(register):
        return False
    register("post_api_request", on_post_api_request)
    register("post_llm_call", on_post_llm_call)
    return True
