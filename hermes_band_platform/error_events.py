"""Band ``error`` events for failed turns.

When a turn fails, the room is the only place the user is looking. The host
does surface *some* failures as ordinary replies (``gateway.run``'s
``_normalize_empty_agent_response`` turns an agent failure into "The request
failed: …", and ``BasePlatformAdapter._process_message_background`` sends a
"Sorry, I encountered an error" message when the handler raises) — but every
one of those is a *message*, and a message needs at least one @mention to be
accepted by Band. When the send itself is what failed, the user is left with
silence that is indistinguishable from the agent still thinking, and the only
record is in the gateway log on the host.

A Band ``error`` **event** is exempt from the @mention requirement, so it is
the one thing that still lands when the message path is the broken part. This
module builds and posts it.

Design constraints, in the order they matter:

  * **Best-effort, never masking.** Emission is wrapped end-to-end; a failing
    event post is logged and swallowed. The original failure — and the
    adapter's own ``mark_failed`` ack — must survive it untouched.
  * **Fail-closed redaction.** A Band event cannot be deleted once written and
    the host does not scrub this path, so every payload goes through
    ``agent.redact.redact_sensitive_text(force=True)``. If that redactor is
    unavailable we DROP the event rather than emit unredacted text — an error
    string can easily carry a credential, a token in a URL, or a traceback
    full of environment values.
  * **The host's classification, not ours.** Where ``gateway.run`` already
    knows how to describe a provider failure (auth / policy / rate-limit), we
    reuse its wording so chat surfaces stay consistent.
  * **Never fed back to the agent.** Events are not text, and the plugin's own
    rehydration skips non-text context items (``adapter._seedable_text``), as
    does inbound dispatch (``adapter._handle_message_created``). So emitting
    cannot loop back into the agent's own transcript.

Sizing (``_EVENT_CONTENT_MAX_LENGTH``, head-and-tail truncation, blank-content
placeholder) mirrors the SDK's private ``band/runtime/tools.py`` helpers: the
platform rejects content over 16384 chars and blank content with a 422 before
it ever reaches the room.

Logging
-------
A failed turn that also fails to *report* itself is the worst case here — the
user sees nothing and the operator has nothing to grep — so every way out of
this module says so. Dropping the event (redaction unavailable, no room, no
link) is ``warning``; a successful post is ``debug``; expected degradations
(an older host's classifier, a non-weakref-able adapter in tests) are ``debug``.

The failure *reason* is payload: it is error text, and error text is where a
credential in a URL or an environment dump shows up. It is never logged — not
before redaction and not after. Log sites carry the room id, the error code and
the content **length** only.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from typing import Any, Dict, Optional, Tuple
from weakref import WeakKeyDictionary

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy SDK bindings. The gateway may discover this module before dependencies
# are installed, so a failed import must not become permanent process state.
# ``_load_error_event_sdk`` retries until the imports work, then caches them.
# ---------------------------------------------------------------------------
ChatEventRequest = None
DEFAULT_REQUEST_OPTIONS = {"max_retries": 3}
BandMessageType = None


def _load_error_event_sdk() -> bool:
    """Bind the event SDK types on demand, retrying after missing imports."""
    global ChatEventRequest, DEFAULT_REQUEST_OPTIONS, BandMessageType

    if ChatEventRequest is not None and BandMessageType is not None:
        return True
    try:
        from band.client.rest import (
            ChatEventRequest as SDKChatEventRequest,
            DEFAULT_REQUEST_OPTIONS as sdk_request_options,
        )
        from band.core.types import MessageType as SDKMessageType
    except ImportError:
        return False
    except Exception as e:
        # A partially installed/broken SDK is also retryable. Keep emission
        # best-effort and log only the exception type; import errors can carry
        # environment paths that do not belong in this failure surface.
        logger.debug(
            "[band] error event: band-sdk bindings unavailable (%s)",
            type(e).__name__,
        )
        return False

    # Publish only a complete set of bindings. If discovery caught an
    # incomplete install, the next emission gets a clean retry.
    ChatEventRequest = SDKChatEventRequest
    DEFAULT_REQUEST_OPTIONS = sdk_request_options
    BandMessageType = SDKMessageType
    return True


# Platform limits for event content (thenvoi-platform ``events_controller.ex``):
# over-long content and blank content are both rejected with a 422 before the
# event reaches the room. Copied from the SDK's private ``band/runtime/tools.py``
# rather than imported, so an older band-sdk without those helpers still works.
_EVENT_CONTENT_MAX_LENGTH = 16384
_EVENT_TRUNCATION_MARKER = "... [truncated] ..."
_EVENT_EMPTY_CONTENT_PLACEHOLDER = "(no content)"

# How much of a raw failure reason to show in the room. The event carries a
# glance-able explanation for a human, not a log line — the full text is in the
# gateway log. Head AND tail are kept, for the same reason the SDK keeps both:
# the last line of a traceback is usually the one that says what broke.
_REASON_MAX_LENGTH = 300
_REASON_TRUNCATION_MARKER = " … "

# Margin kept either side of each cut when bounding a reason *before* redaction
# (see ``_bound_for_redaction``).
#
# WHY BOUND AT ALL: the host's redactor is superlinear in input length once
# ``redact_url_credentials=True`` is on. Measured in the gateway venv —
# 16KB 0.23s, 50KB 2.15s, 100KB 8.54s, i.e. O(n²) — and it is entirely
# ``agent.redact._STRICT_URL_USERINFO_RE`` (redact.py:301). Its optional scheme
# prefix ``(?:[A-Za-z][A-Za-z0-9+.-]*:)?`` is unanchored, so on any long run of
# ``[A-Za-z0-9+.-]`` the greedy ``*`` scans to end-of-string and backtracks
# looking for ``:`` at *every* start offset. A stack dump or a base64 blob is
# exactly such a run. Eight seconds on the turn-failure path is the last thing
# a struggling gateway needs, so we cap what the redactor ever sees.
#
# WHY THIS SIZE: only text that can still reach the room needs scrubbing, and
# the margin exists so a credential straddling a cut is still whole enough for
# the redactor to recognize. It must therefore exceed the longest secret
# ``agent.redact`` can match — a PEM block via ``_PRIVATE_KEY_RE`` (redact.py:221),
# ~3.2KB for RSA-4096 and ~6.3KB for RSA-8192. 8192 clears both. That caps the
# redactor's input at ~16.7KB (one event's worth) and its cost at ~0.23s
# regardless of how large the original reason was.
_REDACTION_INPUT_MARGIN = 8192

_FAILURE_HEADLINE = "⚠️ I couldn't finish that turn."
_NO_DETAIL_LINE = "No details reached this room — check the gateway logs."

# Band's ``error`` event metadata shape is ``{error_code, details}``.
_ERROR_CODE_TURN_FAILED = "turn_failed"
_ERROR_CODE_DELIVERY_FAILED = "delivery_failed"

# Most recent failed send per adapter and room. Populated by
# ``note_send_failure`` from the adapter's send path and consumed once by
# ``report_turn_failure`` — a failed delivery is the single most common way a
# Band turn goes silent, and it is the only failure reason that exists
# in-process at the time ``on_processing_complete`` fires (the hook itself
# carries no reason, only an outcome).
#
# Weak keys ensure a discarded adapter is not pinned. Each adapter retains at
# most this many rooms. New failures are most-recently-used; once full, the
# least-recently recorded room is evicted. A success or pop touches only its
# own room, so activity in one room cannot erase another room's failure.
_SEND_FAILURE_ROOMS_MAX = 256
_LAST_SEND_FAILURE: "WeakKeyDictionary[Any, OrderedDict[str, str]]" = (
    WeakKeyDictionary()
)


def _short(value: Any) -> str:
    """Truncate an id to ``first8…`` for low-cardinality logs.

    Duplicated from ``adapter._short_id`` on purpose: ``adapter`` imports this
    module at its top level, so importing back would be circular.
    """
    if not value:
        return "<none>"
    text = str(value)
    if len(text) <= 8:
        return text
    return f"{text[:8]}…"


def _head_and_tail(text: str, limit: int, marker: str) -> str:
    """Cut *text* to *limit* chars, keeping its head and tail around *marker*.

    Both ends are preserved because the tail of a truncated failure dump is
    often the informative part — the last line of a traceback, a trailing
    status — which a head-only cut would silently drop. A no-op when *text*
    already fits, so callers can run it unconditionally.
    """
    if len(text) <= limit:
        return text
    budget = limit - len(marker)
    if budget <= 0:  # pathological limit — fall back to a plain head cut
        return text[:limit]
    head_len = budget // 2
    tail_len = budget - head_len
    return text[:head_len] + marker + text[-tail_len:]


def _truncate_event_content(content: str) -> str:
    """Cap *content* at the platform's 16384-char limit (a 422 above it)."""
    return _head_and_tail(content, _EVENT_CONTENT_MAX_LENGTH, _EVENT_TRUNCATION_MARKER)


def _bound_for_redaction(text: str) -> str:
    """Cut *text* down to what could still reach the room, plus a safety margin.

    Redaction is the expensive step and its cost grows with the square of the
    input (see ``_REDACTION_INPUT_MARGIN``), so it must not be handed a 100KB
    traceback. Everything outside the head and tail this keeps is dropped by
    ``describe_reason``'s own cap anyway — the discarded middle never reaches
    Band, so it never needed scrubbing.

    Order matters and is the whole point: bound → redact → cut to the final
    limit. Cutting *after* redaction is what keeps the guarantee intact — every
    character emitted came out of the redactor — while the margin here means a
    credential straddling that final cut was still whole when the redactor saw
    it, rather than a fragment it could not recognize.
    """
    return _head_and_tail(
        text,
        _REASON_MAX_LENGTH + 2 * _REDACTION_INPUT_MARGIN,
        _REASON_TRUNCATION_MARKER,
    )


def redact_event_text(text: Any) -> Optional[str]:
    """Redact *text* for a Band event, or ``None`` when it cannot be redacted.

    **Fail-closed.** A Band event is permanent — the agent surface has no
    delete, and ``supersede`` only de-lists — so an unredacted credential
    written here cannot be taken back. ``None`` therefore means "drop the
    event", never "emit as-is". ``force=True`` redacts even when
    ``security.redact_secrets`` is off, matching every other safety boundary in
    the host (``_redact_approval_command``, ``_redact_gateway_user_facing_secrets``).

    ``redact_url_credentials=True`` goes one step beyond what the host's own
    chat-egress redactor asks for, and deliberately: it is what masks
    ``https://user:pass@host`` userinfo and credential-named query parameters.
    The redactor leaves those alone by default so that actionable OAuth
    callback / magic-link / pre-signed URLs survive ordinary tool flows — but a
    permanent failure notice in a shared room is a *non-navigation egress
    boundary*, nobody is going to click a URL out of it, and a connection error
    is one of the likeliest places for a credentialed URL to show up.

    Each ``None`` records its cause at ``debug``; the caller, which knows the
    room, is what logs the resulting drop at ``warning``. Only exception
    *types* are recorded — this function holds the raw, unredacted failure
    text, so its exceptions are the one place that text could leak into a log.
    """
    try:
        from agent.redact import redact_sensitive_text
    except Exception as e:
        logger.debug(
            "[band] error event: agent.redact is not importable (%s)",
            type(e).__name__,
        )
        return None
    try:
        return redact_sensitive_text(
            str(text or ""), force=True, redact_url_credentials=True
        )
    except TypeError:
        # Older agent.redact without the URL-credential switch — still redact.
        logger.debug(
            "[band] error event: agent.redact predates redact_url_credentials; "
            "redacting without it"
        )
        try:
            return redact_sensitive_text(str(text or ""), force=True)
        except Exception as e:
            logger.debug(
                "[band] error event: redactor raised (%s)", type(e).__name__
            )
            return None
    except Exception as e:
        logger.debug("[band] error event: redactor raised (%s)", type(e).__name__)
        return None


def describe_reason(text: str) -> Optional[str]:
    """Render an already-redacted failure reason as a human-readable line.

    Delegates to the host's own provider-error classifier so a quota / rate
    limit reads as one ("⏱️ The model provider is rate-limiting requests…")
    instead of as a raw envelope, and so chat wording stays identical to every
    other Hermes surface. The rate-limit regex is applied on its own as well:
    ``_looks_like_gateway_provider_error`` only fires when the marker leads the
    string, and a 429 usually arrives wrapped in other text.

    Falls back to the reason itself (trimmed) when the host classifier is
    unavailable or recognizes nothing — a raw-but-redacted reason in the room
    still beats silence.
    """
    reason = str(text or "").strip()
    if not reason:
        return None
    try:
        from gateway.run import (
            _GATEWAY_RATE_LIMIT_RE,
            _gateway_provider_error_reply,
            _looks_like_gateway_provider_error,
        )

        if _looks_like_gateway_provider_error(reason) or _GATEWAY_RATE_LIMIT_RE.search(
            reason
        ):
            return _gateway_provider_error_reply(reason)
    except Exception as e:
        # Expected on an older host: fall back to the redacted reason itself.
        # The reason is never logged — only that classification was skipped.
        logger.debug(
            "[band] error event: host provider-error classifier unavailable (%s); "
            "using the raw reason",
            type(e).__name__,
        )
    return _head_and_tail(reason, _REASON_MAX_LENGTH, _REASON_TRUNCATION_MARKER)


def note_send_failure(adapter: Any, room_id: str, error: Any) -> None:
    """Record (``error``) or clear (``error is None``) the last failed send.

    Called from the adapter's send path. The clear is what keeps attribution
    honest: without it, a send that failed and then recovered would still be
    reported as the reason for an unrelated turn failure minutes later.
    """
    try:
        room_key = str(room_id)
        if error is None:
            failures = _LAST_SEND_FAILURE.get(adapter)
            if failures is not None:
                failures.pop(room_key, None)
            if failures is not None and not failures:
                _LAST_SEND_FAILURE.pop(adapter, None)
            return
        failures = _LAST_SEND_FAILURE.get(adapter)
        if failures is None:
            failures = OrderedDict()
            _LAST_SEND_FAILURE[adapter] = failures
        failures[room_key] = str(error)
        failures.move_to_end(room_key)
        while len(failures) > _SEND_FAILURE_ROOMS_MAX:
            failures.popitem(last=False)
    except TypeError:
        # Non-weakref-able stand-in (test doubles); losing the reason only
        # costs detail in the event, never correctness. The error text itself
        # is payload and stays out of the log.
        logger.debug(
            "[band] error event: send failure for room %s not recorded — "
            "adapter is not weak-referenceable",
            _short(room_id),
        )


def pop_send_failure(adapter: Any, room_id: str) -> Optional[str]:
    """Consume only ``room_id``'s recorded send failure, when present."""
    try:
        failures = _LAST_SEND_FAILURE.get(adapter)
    except TypeError:
        logger.debug(
            "[band] error event: no recorded send failure for room %s — "
            "adapter is not weak-referenceable",
            _short(room_id),
        )
        return None
    if failures is None:
        return None
    error = failures.pop(str(room_id), None)
    if not failures:
        _LAST_SEND_FAILURE.pop(adapter, None)
    return error


def _fatal_error_reason(adapter: Any) -> Optional[str]:
    """The adapter's standing fatal error, when one is set.

    A dead consumer loop or a failed connect leaves the link marked fatal while
    the REST client still works, so this is often the real story behind a turn
    that produced nothing.
    """
    try:
        if getattr(adapter, "has_fatal_error", False):
            return getattr(adapter, "fatal_error_message", None)
    except Exception as e:
        logger.debug(
            "[band] error event: could not read the adapter's fatal-error state "
            "(%s); reporting without a reason",
            type(e).__name__,
        )
    return None


def build_failure_event(
    raw_reason: Optional[str] = None,
    *,
    message_id: Optional[str] = None,
    error_code: str = _ERROR_CODE_TURN_FAILED,
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Build ``(content, metadata)`` for a turn-failure event.

    Returns ``(None, None)`` when the payload cannot be redacted — the caller
    must then drop the event (see :func:`redact_event_text`).
    """
    reason: Optional[str] = None
    if raw_reason:
        # Bound BEFORE redacting: the redactor is quadratic in input length and
        # everything outside these regions is dropped by describe_reason anyway.
        safe = redact_event_text(_bound_for_redaction(str(raw_reason)))
        if safe is None:
            return None, None
        # Classify (and cut to the final length) only after redaction, so a
        # secret can never reach the host's regexes, be echoed back inside its
        # reply, or survive as an unrecognized fragment of a cut token.
        reason = describe_reason(safe)

    content = redact_event_text(f"{_FAILURE_HEADLINE}\n{reason or _NO_DETAIL_LINE}")
    if content is None:
        return None, None
    content = _truncate_event_content(content.strip()) or _EVENT_EMPTY_CONTENT_PLACEHOLDER

    metadata: Dict[str, Any] = {"error_code": error_code}
    if reason:
        metadata["details"] = reason
    if message_id:
        # Band's own id for the message whose turn failed, in this same room —
        # useful for correlation and not a secret the room doesn't already hold.
        metadata["message_id"] = str(message_id)
    return content, metadata


async def emit_error_event(
    adapter: Any,
    room_id: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Post one ``error`` event to a room. Never raises.

    Deliberately does NOT go through ``adapter.send``: events carry no mentions
    (Band rejects a mention-less *message* with a 422, but accepts a
    mention-less *event*), which is exactly why an event still lands when the
    message path is the thing that failed.
    """
    link = getattr(adapter, "_link", None)
    sdk_available = _load_error_event_sdk()
    if link is None or not sdk_available:
        # The user is left with silence on a failed turn — loud by definition,
        # and it fires at most once per failure, so warning costs nothing.
        logger.warning(
            "[band] Dropping error event for room %s — %s",
            _short(room_id),
            "adapter has no live link" if link is None else "band-sdk unavailable",
        )
        return False
    try:
        await link.rest.agent_api_events.create_agent_chat_event(
            chat_id=room_id,
            event=ChatEventRequest(
                content=content,
                message_type=BandMessageType.ERROR,
                metadata=metadata,
            ),
            request_options=DEFAULT_REQUEST_OPTIONS,
        )
        # Length, not text: the content is a redacted failure notice, but it is
        # still the failure text and has no business in a log line.
        logger.debug(
            "[band] Emitted error event to room %s (code %s, %d chars)",
            _short(room_id),
            (metadata or {}).get("error_code", "<none>"),
            len(content),
        )
        return True
    except Exception as e:
        logger.warning(
            "[band] Could not emit error event to room %s: %s", _short(room_id), e
        )
        return False


async def emit_thought_event(adapter: Any, room_id: str, content: str) -> bool:
    """Post the turn's final text as a ``thought``. Never raises.

    A thought is content the agent produced without addressing anyone, which is
    exactly what an unaddressed final reply is — Band's own model treats it that
    way. And like every event a thought is exempt from the mention requirement,
    so these words can land without the adapter inventing a recipient for them.

    Shares the event seam with :func:`emit_error_event` rather than
    ``adapter.send``, since that message path is the one being replaced.
    """
    link = getattr(adapter, "_link", None)
    sdk_available = _load_error_event_sdk()
    if link is None or not sdk_available:
        logger.warning(
            "[band] Dropping unaddressed final text for room %s — %s",
            _short(room_id),
            "adapter has no live link" if link is None else "band-sdk unavailable",
        )
        return False
    body = _truncate_event_content(content or "") or _EVENT_EMPTY_CONTENT_PLACEHOLDER
    try:
        await link.rest.agent_api_events.create_agent_chat_event(
            chat_id=room_id,
            event=ChatEventRequest(
                content=body,
                message_type=BandMessageType.THOUGHT,
                metadata=None,
            ),
            request_options=DEFAULT_REQUEST_OPTIONS,
        )
        # Length only. This is the model's own prose and has no business in a log.
        logger.debug(
            "[band] Emitted thought event to room %s (%d chars)",
            _short(room_id),
            len(body),
        )
        return True
    except Exception as e:
        logger.warning(
            "[band] Could not emit thought event to room %s: %s", _short(room_id), e
        )
        return False


async def report_turn_failure(adapter: Any, event: Any, outcome: Any) -> None:
    """Surface a failed turn in its room as a Band ``error`` event.

    Wired from ``BandAdapter.on_processing_complete``, which is the only
    adapter-facing seam the host drives on *every* turn failure — the handler
    raising, an unexpected cancellation, and (the silent one) a response that
    was produced but could not be delivered all funnel into
    ``ProcessingOutcome.FAILURE`` there.

    Only ``FAILURE`` emits. ``CANCELLED`` is a ``/stop`` or a superseded turn —
    its silence is intentional and announcing it would be noise.

    Never raises: a failure here must not mask the failure it is reporting, nor
    the caller's own server-side ack.
    """
    try:
        outcome_name = str(getattr(outcome, "value", outcome))
        if outcome_name != "failure":
            # One line per turn, at debug: this is the log that says a turn
            # finished cleanly, which is exactly what was missing when a reply
            # appeared not to arrive.
            logger.debug(
                "[band] Turn completed with outcome %s — no error event",
                outcome_name,
            )
            return
        room_id = getattr(getattr(event, "source", None), "chat_id", None)
        if not room_id:
            logger.warning(
                "[band] Turn failed but carried no room id — no error event "
                "(message %s)",
                _short(getattr(event, "message_id", None)),
            )
            return

        send_error = pop_send_failure(adapter, room_id)
        content, metadata = build_failure_event(
            send_error or _fatal_error_reason(adapter),
            message_id=getattr(event, "message_id", None),
            error_code=(
                _ERROR_CODE_DELIVERY_FAILED if send_error else _ERROR_CODE_TURN_FAILED
            ),
        )
        if content is None:
            logger.warning(
                "[band] Dropping error event for room %s — redaction unavailable "
                "(fail-closed: the room is told nothing rather than told too much)",
                _short(room_id),
            )
            return
        # The error code already says where the reason came from
        # (``delivery_failed`` = the send, ``turn_failed`` = adapter state), so
        # the reason itself never has to appear here.
        logger.debug(
            "[band] Turn failed in room %s — emitting error event (code %s)",
            _short(room_id),
            (metadata or {}).get("error_code", "<none>"),
        )
        await emit_error_event(adapter, room_id, content, metadata)
    except Exception as e:
        # The report of a failure failed. Warning, not debug: nothing else in
        # the system records that the room was never told. Only the exception
        # type — this frame has the (unredacted) failure reason in scope.
        logger.warning(
            "[band] Error-event reporting failed (%s) — the room was not told "
            "the turn failed",
            type(e).__name__,
        )
