"""Tests for the Band platform adapter.

The band SDK stub is installed by ``tests/conftest.py`` at collection time,
BEFORE this module imports the adapter — so the adapter's top-level
``try: from band ...`` binds the stub and ``BAND_AVAILABLE`` stays True.
"""

import time
import concurrent.futures
import asyncio
import logging
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import hermes_band_platform.adapter as _band_mod

BandAdapter = _band_mod.BandAdapter
check_band_requirements = _band_mod.check_band_requirements
register = _band_mod.register
_env_enablement = _band_mod._env_enablement
_is_connected = _band_mod._is_connected
validate_config = _band_mod.validate_config
_derive_urls = _band_mod._derive_urls
_short_id = _band_mod._short_id
ProcessingOutcome = _band_mod.ProcessingOutcome



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_config(extra=None):
    from gateway.config import PlatformConfig
    return PlatformConfig(enabled=True, extra=extra or {})


def _make_adapter(monkeypatch, agent_id="agent-uuid-1234", api_key="secret-key", base_url=""):
    """Build a BandAdapter with credentials in env vars (env wins over extra)."""
    monkeypatch.setenv("BAND_AGENT_ID", agent_id)
    monkeypatch.setenv("BAND_API_KEY", api_key)
    if base_url:
        monkeypatch.setenv("BAND_BASE_URL", base_url)
    else:
        monkeypatch.delenv("BAND_BASE_URL", raising=False)
    monkeypatch.delenv("BAND_OWNER_ID", raising=False)
    cfg = _make_config()
    return BandAdapter(cfg)


# ---------------------------------------------------------------------------
# 1. Init / construction
# ---------------------------------------------------------------------------

class TestBandAdapterInit:

    def test_init_reads_agent_id_from_env(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, agent_id="env-agent-id", api_key="env-api-key")
        assert adapter._cfg_agent_id == "env-agent-id"
        assert adapter._api_key == "env-api-key"

    def test_execution_emission_defaults_off(self, monkeypatch):
        monkeypatch.delenv("BAND_EMIT_EXECUTION", raising=False)
        assert _make_adapter(monkeypatch)._execution_scope == "off"

    def test_blank_execution_emission_scope_is_off(self, monkeypatch):
        monkeypatch.setenv("BAND_EMIT_EXECUTION", "")
        assert _make_adapter(monkeypatch)._execution_scope == "off"

    @pytest.mark.parametrize("scope", ["off", "all", "hub"])
    def test_execution_emission_accepts_documented_scopes(self, monkeypatch, scope):
        monkeypatch.setenv("BAND_EMIT_EXECUTION", scope)
        assert _make_adapter(monkeypatch)._execution_scope == scope

    def test_invalid_execution_emission_scope_fails_closed(self, monkeypatch):
        """A typo must not publish tool args — anything unknown means off."""
        monkeypatch.setenv("BAND_EMIT_EXECUTION", "yes")
        assert _make_adapter(monkeypatch)._execution_scope == "off"

    def test_init_reads_credentials_from_config_extra(self, monkeypatch):
        for key in ("BAND_AGENT_ID", "BAND_API_KEY", "BAND_BASE_URL", "BAND_OWNER_ID"):
            monkeypatch.delenv(key, raising=False)
        cfg = _make_config(extra={"agent_id": "extra-agent", "api_key": "extra-key"})
        adapter = BandAdapter(cfg)
        assert adapter._cfg_agent_id == "extra-agent"
        assert adapter._api_key == "extra-key"

    def test_env_overrides_config_extra(self, monkeypatch):
        monkeypatch.setenv("BAND_AGENT_ID", "env-wins")
        monkeypatch.setenv("BAND_API_KEY", "env-key")
        monkeypatch.delenv("BAND_BASE_URL", raising=False)
        monkeypatch.delenv("BAND_OWNER_ID", raising=False)
        cfg = _make_config(extra={"agent_id": "extra-loses", "api_key": "extra-key"})
        adapter = BandAdapter(cfg)
        assert adapter._cfg_agent_id == "env-wins"

    def test_init_sets_band_platform_identity(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        # platform is a Platform enum with value "band"
        assert adapter.platform.value == "band"

    def test_init_base_url_from_env(self, monkeypatch):
        adapter = _make_adapter(monkeypatch, base_url="https://custom.host")
        assert adapter._base_url == "https://custom.host"

    def test_init_base_url_from_extra(self, monkeypatch):
        for key in ("BAND_AGENT_ID", "BAND_API_KEY", "BAND_BASE_URL", "BAND_OWNER_ID"):
            monkeypatch.delenv(key, raising=False)
        cfg = _make_config(extra={
            "agent_id": "a",
            "api_key": "b",
            "base_url": "https://extra.host",
        })
        adapter = BandAdapter(cfg)
        assert adapter._base_url == "https://extra.host"

    def test_init_runtime_state_is_empty(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert adapter._link is None
        assert adapter._consumer_task is None
        assert adapter._sent_ids == set()
        assert adapter._participants_cache == {}
        assert adapter._last_human_sender == {}

    def test_name_property(self, monkeypatch):
        assert _make_adapter(monkeypatch).name == "Band"


class TestBandAccessPolicy:
    """The gateway authorizes inbound Band traffic via the adapter's own policy.

    ``gateway.authz_mixin._is_user_authorized`` only trusts an own-policy
    adapter's intake when BOTH ``enforces_own_access_policy`` is True AND the
    effective policy for the chat type is ``"allowlist"`` (the #34515 fail-open
    fix). Band has no DMs, so every source is ``chat_type="group"`` and the host
    reads ``_group_policy``. If these drift, a fresh install default-denies every
    sender — including the owner — and the agent replies "not an authorized
    user". These tests pin the contract so a host/plugin coupling regression
    fails here instead of in production.
    """

    def test_enforces_own_access_policy_is_true(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert adapter.enforces_own_access_policy is True

    def test_group_policy_is_allowlist(self, monkeypatch):
        # Band traffic is all chat_type="group"; the host reads _group_policy.
        adapter = _make_adapter(monkeypatch)
        assert adapter._group_policy == "allowlist"

    def test_dm_policy_is_allowlist(self, monkeypatch):
        # Band has no DMs, but pin _dm_policy too for forward-compat with any
        # host path that reads it.
        adapter = _make_adapter(monkeypatch)
        assert adapter._dm_policy == "allowlist"

    def test_host_authorizes_band_traffic_without_env_allowlist(self, monkeypatch):
        """End-to-end: the real host ``_is_user_authorized`` admits a Band user.

        Drives ``gateway.authz_mixin.GatewayAuthorizationMixin._is_user_authorized``
        — the exact gate that produced "not an authorized user" — with no env
        allowlist and no pairing, proving the adapter's policy attrs carry the
        intake decision. Skips if the host mixin isn't importable (e.g. tests run
        without a host on the path).
        """
        authz = pytest.importorskip("gateway.authz_mixin")
        from gateway.session import SessionSource

        # Strip every env opt-in so only the adapter's own policy can authorize.
        for var in (
            "BAND_ALLOWED_USERS",
            "BAND_ALLOW_ALL",
            "GATEWAY_ALLOWED_USERS",
            "GATEWAY_ALLOW_ALL_USERS",
        ):
            monkeypatch.delenv(var, raising=False)

        adapter = _make_adapter(monkeypatch)
        band = adapter.platform

        host = authz.GatewayAuthorizationMixin()
        host.adapters = {band: adapter}
        host.pairing_store = SimpleNamespace(is_approved=lambda *_: False)
        host.config = SimpleNamespace(platforms={band: adapter.config})

        src = SessionSource(
            platform=band,
            user_id="some-band-user",
            chat_id="room-123",
            chat_type="group",
        )
        assert host._is_user_authorized(src) is True

        # Negative control: drop the policy attrs and the same gate default-denies
        # — i.e. the bug this fix closes.
        del adapter._group_policy
        del adapter._dm_policy
        assert host._is_user_authorized(src) is False


# ---------------------------------------------------------------------------
# 2. check_band_requirements
# ---------------------------------------------------------------------------

class TestCheckBandRequirements:

    def test_returns_true_when_band_available(self):
        # The adapter was loaded with the band stub installed, so
        # BAND_AVAILABLE was set True at import time.
        assert _band_mod.BAND_AVAILABLE is True
        assert check_band_requirements() is True

    def test_returns_false_when_band_unavailable(self, monkeypatch):
        # Simulate the SDK being absent by patching the module global.
        monkeypatch.setattr(_band_mod, "BAND_AVAILABLE", False)
        # Make the lazy import path fail. Popping the stubs isn't enough when
        # the real ``band`` SDK is installed on disk (it would just be
        # re-imported), so we *block* the import by parking ``None`` at each
        # ``band.*`` key — Python raises ImportError on a ``None`` entry.
        saved = {}
        for key in list(sys.modules):
            if key == "band" or key.startswith("band."):
                saved[key] = sys.modules.pop(key)
        for key in saved:
            sys.modules[key] = None
        try:
            result = check_band_requirements()
            assert result is False
        finally:
            # Restore stubs and BAND_AVAILABLE so subsequent tests are unaffected.
            sys.modules.update(saved)
            monkeypatch.setattr(_band_mod, "BAND_AVAILABLE", True)

    def test_returns_true_immediately_when_already_available(self, monkeypatch):
        monkeypatch.setattr(_band_mod, "BAND_AVAILABLE", True)
        assert check_band_requirements() is True


# ---------------------------------------------------------------------------
# 3. register()
# ---------------------------------------------------------------------------

class TestBandPluginRegistration:

    def test_register_calls_ctx_register_platform(self):
        from gateway.platform_registry import platform_registry
        platform_registry.unregister("band")

        ctx = MagicMock()
        register(ctx)
        ctx.register_platform.assert_called_once()

    def test_register_passes_correct_name_and_label(self):
        ctx = MagicMock()
        register(ctx)
        kwargs = ctx.register_platform.call_args[1]
        assert kwargs["name"] == "band"
        assert kwargs["label"] == "Band"

    def test_register_passes_required_env(self):
        ctx = MagicMock()
        register(ctx)
        kwargs = ctx.register_platform.call_args[1]
        assert "BAND_AGENT_ID" in kwargs["required_env"]
        assert "BAND_API_KEY" in kwargs["required_env"]

    def test_register_passes_allowed_users_env(self):
        ctx = MagicMock()
        register(ctx)
        kwargs = ctx.register_platform.call_args[1]
        assert kwargs["allowed_users_env"] == "BAND_ALLOWED_USERS"

    def test_register_passes_allow_all_env(self):
        ctx = MagicMock()
        register(ctx)
        kwargs = ctx.register_platform.call_args[1]
        assert kwargs["allow_all_env"] == "BAND_ALLOW_ALL"

    def test_register_passes_max_message_length(self):
        ctx = MagicMock()
        register(ctx)
        kwargs = ctx.register_platform.call_args[1]
        assert kwargs["max_message_length"] == BandAdapter.MAX_MESSAGE_LENGTH

    def test_register_passes_env_enablement_fn(self):
        ctx = MagicMock()
        register(ctx)
        kwargs = ctx.register_platform.call_args[1]
        assert callable(kwargs["env_enablement_fn"])

    def _hint(self):
        ctx = MagicMock()
        register(ctx)
        return ctx.register_platform.call_args[1]["platform_hint"]

    def test_platform_hint_tells_the_model_to_send_its_own_reply(self):
        """The prompt and the code must agree, or one of two bugs follows.

        If the hint says replies are delivered automatically while the code no
        longer delivers them, every answer is silently demoted to a thought. If
        it said the reverse while the code still delivered, every answer would
        post twice — which is the bug the hint was previously written to fix.
        Both halves therefore change together, and this pins the hint half.
        """
        hint = self._hint()
        assert "YOU MUST SEND YOUR REPLY YOURSELF" in hint
        assert "band_send_message" in hint
        assert "reply_to" in hint

    def test_platform_hint_does_not_promise_automatic_delivery(self):
        hint = self._hint()
        assert "delivered to the room automatically" not in hint
        assert "@mentioned for you" not in hint
        # The old guard against double-posting is now wrong advice: calling the
        # tool is the only way to reach anyone.
        assert "Do NOT call band_send_message to reply" not in hint

    def test_platform_hint_explains_what_unsent_text_becomes(self):
        """A model that forgets to send should be able to recognise the symptom."""
        hint = self._hint()
        assert "thought" in hint
        assert "notifies nobody" in hint

    def test_platform_hint_keeps_owner_no_room_id_guidance(self):
        """Still-correct guidance the fix must not drop: reaching the owner from
        a non-Band session or another room does need an explicit tool call."""
        hint = self._hint()
        assert "call band_send_message with no room_id" in hint
        assert "owner's hub" in hint

    def test_conversation_skill_agrees_that_final_text_is_auto_delivered(self):
        skill = Path(_band_mod.__file__).parent / "skills" / "band-conversations" / "SKILL.md"
        guidance = skill.read_text()
        assert "final assistant text is delivered" in guidance
        assert "Plain assistant text is **not** delivered" not in guidance
        assert "Do **not** call" in guidance
        assert "`band_send_message` for that routine reply" in guidance


# ---------------------------------------------------------------------------
# 4. _env_enablement
# ---------------------------------------------------------------------------

class TestEnvEnablement:

    def test_returns_none_when_no_credentials(self, monkeypatch):
        monkeypatch.delenv("BAND_AGENT_ID", raising=False)
        monkeypatch.delenv("BAND_API_KEY", raising=False)
        assert _env_enablement() is None

    def test_returns_none_when_only_agent_id(self, monkeypatch):
        monkeypatch.setenv("BAND_AGENT_ID", "some-id")
        monkeypatch.delenv("BAND_API_KEY", raising=False)
        assert _env_enablement() is None

    def test_returns_none_when_only_api_key(self, monkeypatch):
        monkeypatch.delenv("BAND_AGENT_ID", raising=False)
        monkeypatch.setenv("BAND_API_KEY", "some-key")
        assert _env_enablement() is None

    def test_returns_dict_with_both_credentials(self, monkeypatch):
        monkeypatch.setenv("BAND_AGENT_ID", "test-agent")
        monkeypatch.setenv("BAND_API_KEY", "test-key")
        monkeypatch.delenv("BAND_BASE_URL", raising=False)
        monkeypatch.delenv("BAND_OWNER_ID", raising=False)
        result = _env_enablement()
        assert result is not None
        assert result["agent_id"] == "test-agent"
        assert result["api_key"] == "test-key"

    def test_includes_base_url_when_set(self, monkeypatch):
        monkeypatch.setenv("BAND_AGENT_ID", "test-agent")
        monkeypatch.setenv("BAND_API_KEY", "test-key")
        monkeypatch.setenv("BAND_BASE_URL", "https://custom.example.com")
        monkeypatch.delenv("BAND_OWNER_ID", raising=False)
        result = _env_enablement()
        assert result["base_url"] == "https://custom.example.com"

    def test_omits_base_url_when_not_set(self, monkeypatch):
        monkeypatch.setenv("BAND_AGENT_ID", "test-agent")
        monkeypatch.setenv("BAND_API_KEY", "test-key")
        monkeypatch.delenv("BAND_BASE_URL", raising=False)
        monkeypatch.delenv("BAND_OWNER_ID", raising=False)
        result = _env_enablement()
        assert "base_url" not in result

    def test_includes_owner_id_when_set(self, monkeypatch):
        monkeypatch.setenv("BAND_AGENT_ID", "test-agent")
        monkeypatch.setenv("BAND_API_KEY", "test-key")
        monkeypatch.delenv("BAND_BASE_URL", raising=False)
        monkeypatch.setenv("BAND_OWNER_ID", "owner-uuid")
        result = _env_enablement()
        assert result["owner_id"] == "owner-uuid"


# ---------------------------------------------------------------------------
# 5. _derive_urls
# ---------------------------------------------------------------------------

class TestDeriveUrls:

    def test_default_host_when_no_base_url(self):
        ws, rest = _derive_urls("")
        assert ws == "wss://app.band.ai/api/v1/socket/websocket"
        assert rest == "https://app.band.ai"

    def test_default_host_when_base_url_is_none(self):
        ws, rest = _derive_urls(None)
        assert ws == "wss://app.band.ai/api/v1/socket/websocket"
        assert rest == "https://app.band.ai"

    def test_custom_host_from_full_https_url(self):
        ws, rest = _derive_urls("https://myband.example.com")
        assert ws == "wss://myband.example.com/api/v1/socket/websocket"
        assert rest == "https://myband.example.com"

    def test_custom_host_without_scheme(self):
        ws, rest = _derive_urls("myband.example.com")
        assert ws == "wss://myband.example.com/api/v1/socket/websocket"
        assert rest == "https://myband.example.com"

    def test_custom_host_with_port(self):
        ws, rest = _derive_urls("https://myband.example.com:8443")
        assert ws == "wss://myband.example.com:8443/api/v1/socket/websocket"
        assert rest == "https://myband.example.com:8443"

    def test_whitespace_stripped_from_base_url(self):
        ws, rest = _derive_urls("  https://trimmed.example.com  ")
        assert ws == "wss://trimmed.example.com/api/v1/socket/websocket"
        assert rest == "https://trimmed.example.com"


# ---------------------------------------------------------------------------
# 6. _short_id redaction
# ---------------------------------------------------------------------------

class TestShortId:

    def test_full_uuid_is_truncated_to_first_8_chars(self):
        uuid = "12345678-abcd-efgh-ijkl-000000000000"
        result = _short_id(uuid)
        assert result == "12345678…"
        assert len(result) == 9  # 8 chars + ellipsis

    def test_short_value_returned_as_is(self):
        assert _short_id("abc") == "abc"

    def test_exactly_8_chars_returned_as_is(self):
        assert _short_id("12345678") == "12345678"

    def test_none_returns_none_marker(self):
        assert _short_id(None) == "<none>"

    def test_empty_string_returns_none_marker(self):
        assert _short_id("") == "<none>"

    def test_api_key_never_appears_in_truncated_output(self):
        api_key = "very-secret-api-key-value"
        truncated = _short_id(api_key)
        # Only the first 8 chars + ellipsis should be present
        assert api_key not in truncated
        assert truncated == "very-sec…"


# ---------------------------------------------------------------------------
# 7. validate_config / _is_connected
# ---------------------------------------------------------------------------

class TestValidateConfig:

    def test_validate_config_true_with_env_credentials(self, monkeypatch):
        monkeypatch.setenv("BAND_AGENT_ID", "agent")
        monkeypatch.setenv("BAND_API_KEY", "key")
        assert validate_config(_make_config()) is True

    def test_validate_config_true_with_extra_credentials(self, monkeypatch):
        monkeypatch.delenv("BAND_AGENT_ID", raising=False)
        monkeypatch.delenv("BAND_API_KEY", raising=False)
        cfg = _make_config(extra={"agent_id": "a", "api_key": "b"})
        assert validate_config(cfg) is True

    def test_validate_config_false_when_no_credentials(self, monkeypatch):
        monkeypatch.delenv("BAND_AGENT_ID", raising=False)
        monkeypatch.delenv("BAND_API_KEY", raising=False)
        assert validate_config(_make_config()) is False

    def test_is_connected_true_with_env(self, monkeypatch):
        monkeypatch.setenv("BAND_AGENT_ID", "a")
        monkeypatch.setenv("BAND_API_KEY", "b")
        assert _is_connected(_make_config()) is True

    def test_is_connected_false_missing_agent_id(self, monkeypatch):
        monkeypatch.delenv("BAND_AGENT_ID", raising=False)
        monkeypatch.setenv("BAND_API_KEY", "b")
        assert _is_connected(_make_config()) is False


# ---------------------------------------------------------------------------
# 8. _chat_type_label — hub vs group (Band has no DMs)
# ---------------------------------------------------------------------------

class TestChatTypeLabel:

    def test_hub_room_labelled_hub(self, monkeypatch):
        a = _make_adapter(monkeypatch)
        a._hub_room_id = "hub-room"
        assert a._chat_type_label("hub-room") == "hub"

    def test_regular_room_labelled_group(self, monkeypatch):
        # Never "dm" — every non-hub Band room is a group chat.
        a = _make_adapter(monkeypatch)
        a._hub_room_id = "hub-room"
        assert a._chat_type_label("some-other-room") == "group"

    def test_group_when_no_hub_resolved(self, monkeypatch):
        a = _make_adapter(monkeypatch)
        a._hub_room_id = None
        assert a._chat_type_label("any-room") == "group"


# ---------------------------------------------------------------------------
# 9. send() - success and failure paths
# ---------------------------------------------------------------------------

class TestBandAdapterSend:

    @pytest.fixture
    def adapter(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        return adapter

    @pytest.mark.asyncio
    async def test_send_returns_failure_when_not_connected(self, adapter):
        # _link is None by default after construction
        result = await adapter.send("room-123", "hello")
        assert result.success is False
        assert result.error is not None

    @pytest.mark.asyncio
    async def test_send_success_returns_message_id(self, adapter):
        # Wire up a fake link with REST API
        mock_link = MagicMock()
        resp_data = SimpleNamespace(id="sent-msg-id-001")
        resp = SimpleNamespace(data=resp_data)
        mock_link.rest.agent_api_messages.create_agent_chat_message = AsyncMock(
            return_value=resp
        )
        adapter._link = mock_link

        # Seed last human sender so build_mentions has something to work with
        # Out-of-turn delivery addresses the owner (the hub pattern), so the
        # owner is what has to be resolvable — not a "last sender".
        adapter._owner_uuid = "owner-uuid"
        adapter._participants_cache["room-123"] = [
            {"id": "owner-uuid", "type": "User", "name": "Owner", "handle": "owner"},
        ]

        result = await adapter.send("room-123", "hello world")
        assert result.success is True
        assert result.message_id == "sent-msg-id-001"

    @pytest.mark.asyncio
    async def test_send_records_message_id_in_sent_ids(self, adapter):
        mock_link = MagicMock()
        resp = SimpleNamespace(data=SimpleNamespace(id="tracked-id"))
        mock_link.rest.agent_api_messages.create_agent_chat_message = AsyncMock(
            return_value=resp
        )
        adapter._link = mock_link
        # Out-of-turn delivery addresses the owner (the hub pattern), so the
        # owner is what has to be resolvable — not a "last sender".
        adapter._owner_uuid = "owner-uuid"
        adapter._participants_cache["room-99"] = [
            {"id": "owner-uuid", "type": "User", "name": "Owner", "handle": "owner"},
        ]

        await adapter.send("room-99", "test message")
        assert "tracked-id" in adapter._sent_ids

    @pytest.mark.asyncio
    async def test_send_chunks_long_content(self, adapter):
        mock_link = MagicMock()
        call_count = 0

        async def _fake_send(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return SimpleNamespace(data=SimpleNamespace(id=f"chunk-{call_count}"))

        mock_link.rest.agent_api_messages.create_agent_chat_message = _fake_send
        adapter._link = mock_link
        # Out-of-turn delivery addresses the owner (the hub pattern), so the
        # owner is what has to be resolvable — not a "last sender".
        adapter._owner_uuid = "owner-uuid"
        adapter._participants_cache["room-big"] = [
            {"id": "owner-uuid", "type": "User", "name": "Owner", "handle": "owner"},
        ]

        # Create content longer than MAX_MESSAGE_LENGTH (4000)
        long_content = "x" * 5000
        result = await adapter.send("room-big", long_content)
        assert result.success is True
        # Should have been called at least twice (chunked)
        assert call_count >= 2

    @pytest.mark.asyncio
    async def test_send_failure_returns_error_result(self, adapter):
        mock_link = MagicMock()
        mock_link.rest.agent_api_messages.create_agent_chat_message = AsyncMock(
            side_effect=RuntimeError("network failure")
        )
        adapter._link = mock_link
        # Out-of-turn delivery addresses the owner (the hub pattern), so the
        # owner is what has to be resolvable — not a "last sender".
        adapter._owner_uuid = "owner-uuid"
        adapter._participants_cache["room-fail"] = [
            {"id": "owner-uuid", "type": "User", "name": "Owner", "handle": "owner"},
        ]

        result = await adapter.send("room-fail", "hi")
        assert result.success is False
        assert "network failure" in result.error

    @pytest.mark.asyncio
    async def test_send_no_mention_returns_failure_without_sending(self, adapter):
        mock_link = MagicMock()
        mock_link.rest.agent_api_messages.create_agent_chat_message = AsyncMock()
        mock_link.rest.agent_api_participants.list_agent_chat_participants = AsyncMock(
            return_value=SimpleNamespace(data=[])
        )
        adapter._link = mock_link

        # No last human sender and no participants cached
        result = await adapter.send("room-empty", "hello")
        assert result.success is False
        mock_link.rest.agent_api_messages.create_agent_chat_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_out_of_turn_send_addresses_the_owner(self, adapter):
        mock_link = MagicMock()
        resp = SimpleNamespace(data=SimpleNamespace(id="msg-x"))
        mock_link.rest.agent_api_messages.create_agent_chat_message = AsyncMock(
            return_value=resp
        )
        adapter._link = mock_link
        adapter._agent_id = "agent-id-xxx"
        adapter._owner_uuid = "owner-uuid"

        # Seed participants cache directly (skipping REST fetch)
        adapter._participants_cache["room-p"] = [
            {"id": "agent-id-xxx", "type": "Agent", "name": "Bot", "handle": "bot"},
            {"id": "owner-uuid", "type": "User", "name": "Owner", "handle": "owner"},
            {"id": "human-id", "type": "User", "name": "Alice", "handle": "alice"},
        ]

        result = await adapter.send("room-p", "cron output, no turn open")
        assert result.success is True
        call_kwargs = mock_link.rest.agent_api_messages.create_agent_chat_message.call_args[1]
        mentions = call_kwargs["message"].mentions
        # Exactly the owner. Not every human in the room — that is the guess this
        # design removes, and Alice never asked for this.
        assert [getattr(m, "id", None) for m in mentions] == ["owner-uuid"]

    @pytest.mark.asyncio
    async def test_send_marshals_to_link_loop_when_called_from_another_loop(self, adapter):
        """Cross-loop send is routed back onto the link's loop (INT-899).

        The Band link's phoenix primitives bind to the loop ``connect()`` ran
        on; a send() from a *different* running loop (the gateway startup-restore
        replay path) must not raise "<Event> is bound to a different event loop".
        """
        import threading

        mock_link = MagicMock()
        resp = SimpleNamespace(data=SimpleNamespace(id="cross-loop-id"))
        send_loops: list = []

        async def _create(*args, **kwargs):
            send_loops.append(asyncio.get_running_loop())
            return resp

        mock_link.rest.agent_api_messages.create_agent_chat_message = _create
        adapter._link = mock_link
        # Out-of-turn delivery addresses the owner (the hub pattern), so the
        # owner is what has to be resolvable — not a "last sender".
        adapter._owner_uuid = "owner-uuid"
        adapter._participants_cache["room-x"] = [
            {"id": "owner-uuid", "type": "User", "name": "Owner", "handle": "owner"},
        ]

        # Run a dedicated "link" loop in its own thread and pin it on the adapter,
        # exactly as connect() would.
        link_loop = asyncio.new_event_loop()
        ready = threading.Event()

        def _run_loop():
            asyncio.set_event_loop(link_loop)
            link_loop.call_soon(ready.set)
            link_loop.run_forever()

        t = threading.Thread(target=_run_loop, daemon=True)
        t.start()
        ready.wait(5)
        adapter._link_loop = link_loop
        try:
            # We're on the default test loop, which is NOT link_loop.
            assert asyncio.get_running_loop() is not link_loop
            result = await adapter.send("room-x", "hello from restore")
        finally:
            link_loop.call_soon_threadsafe(link_loop.stop)
            t.join(5)
            link_loop.close()

        assert result.success is True
        assert result.message_id == "cross-loop-id"
        # The REST call actually executed on the link's loop, not the caller's.
        assert send_loops == [link_loop]


# ---------------------------------------------------------------------------
# 10. Inbound self-filter — _handle_message_created
# ---------------------------------------------------------------------------

class TestInboundSelfFilter:
    """Tests for message filtering logic in _handle_message_created."""

    @pytest.fixture
    def adapter(self, monkeypatch):
        a = _make_adapter(monkeypatch, agent_id="agent-self-id")
        a._agent_id = "agent-self-id"
        a.handle_message = AsyncMock()
        # Provide empty participant cache to avoid REST fetch
        a._participants_cache["room-abc"] = [
            {"id": "agent-self-id", "type": "Agent", "name": "Bot", "handle": "bot"},
            {"id": "human-sender", "type": "User", "name": "Alice", "handle": "alice"},
        ]
        return a

    def _make_event(self, sender_id, sender_type, msg_id="msg-001", content="hello",
                    message_type="text", room_id="room-abc", mentioned=True):
        # Default to mentioning the agent: every Band room is mention-gated, so
        # the dispatch-mechanics tests need an addressed message to get through.
        mentions = (
            [SimpleNamespace(id="agent-self-id", handle=None)] if mentioned else []
        )
        payload = SimpleNamespace(
            id=msg_id,
            content=content,
            message_type=message_type,
            sender_id=sender_id,
            sender_type=sender_type,
            sender_name="TestUser",
            chat_room_id=room_id,
            metadata=SimpleNamespace(mentions=mentions),
        )
        return SimpleNamespace(
            type="message_created",
            room_id=room_id,
            payload=payload,
        )

    @pytest.mark.asyncio
    async def test_drops_own_agent_message(self, adapter):
        event = self._make_event(
            sender_id="agent-self-id",
            sender_type="Agent",
        )
        await adapter._handle_message_created(event)
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_drops_message_in_sent_ids(self, adapter):
        adapter._sent_ids.add("echo-msg-id")
        event = self._make_event(
            sender_id="human-sender",
            sender_type="User",
            msg_id="echo-msg-id",
        )
        await adapter._handle_message_created(event)
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatches_genuine_user_message(self, adapter):
        event = self._make_event(
            sender_id="human-sender",
            sender_type="User",
            msg_id="fresh-msg-id",
        )
        await adapter._handle_message_created(event)
        adapter.handle_message.assert_called_once()

    @pytest.mark.asyncio
    async def test_dispatched_event_has_no_thread_id(self, adapter):
        event = self._make_event(
            sender_id="human-sender",
            sender_type="User",
        )
        await adapter._handle_message_created(event)
        call_args = adapter.handle_message.call_args[0][0]
        assert call_args.source.thread_id is None

    @pytest.mark.asyncio
    async def test_dispatched_event_has_correct_chat_id(self, adapter):
        event = self._make_event(
            sender_id="human-sender",
            sender_type="User",
            room_id="room-abc",
        )
        await adapter._handle_message_created(event)
        call_args = adapter.handle_message.call_args[0][0]
        assert call_args.source.chat_id == "room-abc"

    @pytest.mark.asyncio
    async def test_drops_non_text_message_type(self, adapter):
        event = self._make_event(
            sender_id="human-sender",
            sender_type="User",
            message_type="tool_call",
        )
        await adapter._handle_message_created(event)
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_group_message_without_mention_is_ignored(self, adapter):
        # Multi-party room, no mention → ignored (mention is the only gate).
        adapter._participants_cache["group-room"] = [
            {"id": "agent-self-id", "type": "Agent", "name": "Bot", "handle": "bot"},
            {"id": "user1", "type": "User", "name": "Alice", "handle": "alice"},
            {"id": "user2", "type": "User", "name": "Bob", "handle": "bob"},
        ]
        payload = SimpleNamespace(
            id="grp-msg-001",
            content="just chatting",
            message_type="text",
            sender_id="user1",
            sender_type="User",
            sender_name="Alice",
            chat_room_id="group-room",
            metadata=SimpleNamespace(mentions=[]),  # No agent mention
        )
        event = SimpleNamespace(
            type="message_created",
            room_id="group-room",
            payload=payload,
        )
        await adapter._handle_message_created(event)
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_two_participant_room_without_mention_is_ignored(self, adapter):
        # room-abc is a 2-participant room. Band has no DMs, so it is
        # mention-gated like any other room — no mention → ignored. Locks the
        # over-respond bug fix.
        event = self._make_event(
            sender_id="human-sender",
            sender_type="User",
            mentioned=False,
        )
        await adapter._handle_message_created(event)
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_dispatched_source_chat_type_is_constant(self, adapter):
        # Even for a 2-participant room, the source chat_type is the constant —
        # never "dm" — so the session key stays anchored on room_id alone.
        event = self._make_event(sender_id="human-sender", sender_type="User")
        await adapter._handle_message_created(event)
        evt = adapter.handle_message.call_args[0][0]
        assert evt.source.chat_type == "group"

    @pytest.mark.asyncio
    async def test_active_session_without_mention_is_ignored(self, adapter):
        # An active session does NOT bypass the mention gate: the platform routes
        # by mention, so Hermes mirrors that — no active-session stickiness.
        active_key = "agent:main:band:group:room-abc"
        adapter._session_store = _FakeSessionStore(active_keys=[active_key])
        event = self._make_event(
            sender_id="human-sender",
            sender_type="User",
            mentioned=False,
        )
        await adapter._handle_message_created(event)
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_owner_command_dispatches_without_mention(self, adapter):
        # A validated owner slash command is answered even with no @mention.
        adapter._owner_uuid = "human-sender"
        event = self._make_event(
            sender_id="human-sender",
            sender_type="User",
            content="/status",
            mentioned=False,
        )
        await adapter._handle_message_created(event)
        adapter.handle_message.assert_called_once()


# ---------------------------------------------------------------------------
# 11. _handle_event routing
# ---------------------------------------------------------------------------

class TestHandleEvent:

    @pytest.fixture
    def adapter(self, monkeypatch):
        a = _make_adapter(monkeypatch)
        a._link = MagicMock()
        a._link.subscribe_room = AsyncMock()
        a._link.unsubscribe_room = AsyncMock()
        return a

    @pytest.mark.asyncio
    async def test_room_added_subscribes_to_room(self, adapter):
        event = SimpleNamespace(type="room_added", room_id="new-room-id")
        await adapter._handle_event(event)
        adapter._link.subscribe_room.assert_called_once_with("new-room-id")

    @pytest.mark.asyncio
    async def test_room_removed_unsubscribes(self, adapter):
        adapter._participants_cache["old-room"] = [{"id": "x"}]
        event = SimpleNamespace(type="room_removed", room_id="old-room")
        await adapter._handle_event(event)
        adapter._link.unsubscribe_room.assert_called_once_with("old-room")
        assert "old-room" not in adapter._participants_cache

    @pytest.mark.asyncio
    async def test_room_deleted_also_unsubscribes(self, adapter):
        event = SimpleNamespace(type="room_deleted", room_id="del-room")
        await adapter._handle_event(event)
        adapter._link.unsubscribe_room.assert_called_once_with("del-room")

    @pytest.mark.asyncio
    async def test_unhandled_event_type_is_ignored(self, adapter):
        # An event the router has no branch for (e.g. a future contact_* event)
        # falls through silently: no raise, no room subscribe/unsubscribe.
        event = SimpleNamespace(type="contact_request_received", room_id="some-room")
        await adapter._handle_event(event)
        adapter._link.subscribe_room.assert_not_called()
        adapter._link.unsubscribe_room.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("event_type", ["room_removed", "room_deleted"])
    async def test_room_departure_forgets_working_state(self, adapter, event_type):
        adapter._working_reported["gone-room"] = time.monotonic()
        event = SimpleNamespace(type=event_type, room_id="gone-room")

        await adapter._handle_event(event)

        assert "gone-room" not in adapter._working_reported


# ---------------------------------------------------------------------------
# 12. Room & participant lifecycle events — participant changes, leave-reset, re-join
# ---------------------------------------------------------------------------

class _FakeSessionStore:
    """Minimal SessionStore stand-in for the tools-pass event tests.

    Mirrors the two surfaces the adapter touches:
      * ``reset_session(key)`` — records the keys it was asked to reset.
      * ``config.group_sessions_per_user`` + ``_ensure_loaded`` + ``_entries`` —
        read by ``_has_active_session`` to decide whether a room is active.
    Band is one shared channel (locked decision #5), so the store reports
    ``group_sessions_per_user=False`` by default.
    """

    def __init__(self, active_keys=None, group_sessions_per_user=False):
        self.config = SimpleNamespace(group_sessions_per_user=group_sessions_per_user)
        self._entries = {k: object() for k in (active_keys or [])}
        self.reset_calls = []

    def _ensure_loaded(self):
        pass

    def reset_session(self, session_key, display_name=None):
        self.reset_calls.append(session_key)
        self._entries.pop(session_key, None)
        return None


def _history_store(transcripts):
    """Session-store stand-in whose 'warm' verdict is history-based.

    ``transcripts`` maps session_key -> transcript list. Unlike ``_FakeSessionStore``
    (which has no ``load_transcript`` so ``_has_active_session`` falls back to the
    legacy entry-presence branch), this exposes ``load_transcript`` and entries
    carrying ``session_id`` — so a test exercises the real history-based rule
    (empty transcript == cold), the behavior this change introduced.
    """
    entries = {key: SimpleNamespace(session_id=f"sid::{key}") for key in transcripts}
    by_sid = {f"sid::{key}": list(t) for key, t in transcripts.items()}

    class _Store:
        config = SimpleNamespace(group_sessions_per_user=False)

        def __init__(self):
            self._entries = entries

        def _ensure_loaded(self):
            pass

        def load_transcript(self, session_id):
            return list(by_sid.get(session_id, []))

    return _Store()


class TestParticipantChangeEvents:
    """participant_added / participant_removed surfacing (decision #3)."""

    @pytest.fixture
    def adapter(self, monkeypatch):
        a = _make_adapter(monkeypatch, agent_id="agent-self-id")
        a._agent_id = "agent-self-id"
        a._link = MagicMock()
        a._link.subscribe_room = AsyncMock()
        a._link.unsubscribe_room = AsyncMock()
        a.handle_message = AsyncMock()
        return a

    @pytest.mark.asyncio
    async def test_cold_room_change_invalidates_cache_no_wake(self, adapter):
        # No session store -> the room can never be "active".
        adapter._participants_cache["cold-room"] = [{"id": "stale"}]
        payload = SimpleNamespace(id="p-1", name="Carol")
        event = SimpleNamespace(
            type="participant_added", room_id="cold-room", payload=payload
        )
        await adapter._handle_event(event)
        # Cache popped (forces refetch) but the agent is NOT woken.
        assert "cold-room" not in adapter._participants_cache
        adapter.handle_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_active_room_change_emits_synthetic_internal_event(self, adapter):
        # Group session key for the room must exist in the store to be "active".
        active_key = "agent:main:band:group:active-room"
        adapter._session_store = _FakeSessionStore(active_keys=[active_key])
        adapter._participants_cache["active-room"] = [{"id": "stale"}]

        payload = SimpleNamespace(id="p-2", name="Dave")
        event = SimpleNamespace(
            type="participant_added", room_id="active-room", payload=payload
        )
        await adapter._handle_event(event)

        # Cache still invalidated.
        assert "active-room" not in adapter._participants_cache
        # Synthetic internal event routed through handle_message.
        adapter.handle_message.assert_called_once()
        evt = adapter.handle_message.call_args[0][0]
        assert evt.internal is True
        assert evt.source.chat_id == "active-room"
        assert evt.source.chat_type == "group"
        assert "Dave" in evt.text
        assert "joined" in evt.text

    @pytest.mark.asyncio
    async def test_active_room_participant_removed_says_left(self, adapter):
        active_key = "agent:main:band:group:active-room"
        adapter._session_store = _FakeSessionStore(active_keys=[active_key])
        payload = SimpleNamespace(id="p-3", name="Eve")
        event = SimpleNamespace(
            type="participant_removed", room_id="active-room", payload=payload
        )
        await adapter._handle_event(event)
        adapter.handle_message.assert_called_once()
        evt = adapter.handle_message.call_args[0][0]
        assert evt.internal is True
        assert "left" in evt.text

    @pytest.mark.asyncio
    async def test_participant_change_without_room_id_is_noop(self, adapter):
        event = SimpleNamespace(type="participant_added", room_id=None, payload=None)
        await adapter._handle_event(event)
        adapter.handle_message.assert_not_called()


class TestRoomRemovedResetsSession:
    """room_removed / room_deleted -> reset_session for the room's keys (#4)."""

    @pytest.fixture
    def adapter(self, monkeypatch):
        a = _make_adapter(monkeypatch)
        a._link = MagicMock()
        a._link.unsubscribe_room = AsyncMock()
        a._link.subscribe_room = AsyncMock()
        return a

    @pytest.mark.asyncio
    async def test_room_removed_resets_the_single_session_key(self, adapter):
        adapter._session_store = _FakeSessionStore()
        event = SimpleNamespace(type="room_removed", room_id="gone-room")
        await adapter._handle_event(event)
        # One shared session per room (constant chat_type) → exactly one key.
        assert adapter._session_store.reset_calls == ["agent:main:band:group:gone-room"]

    @pytest.mark.asyncio
    async def test_room_deleted_also_resets_session(self, adapter):
        adapter._session_store = _FakeSessionStore()
        event = SimpleNamespace(type="room_deleted", room_id="dead-room")
        await adapter._handle_event(event)
        assert adapter._session_store.reset_calls == ["agent:main:band:group:dead-room"]

    @pytest.mark.asyncio
    async def test_room_removed_without_session_store_is_safe(self, adapter):
        # No store wired -> _reset_room_session is a no-op, must not raise.
        adapter._participants_cache["r"] = [{"id": "x"}]
        event = SimpleNamespace(type="room_removed", room_id="r")
        await adapter._handle_event(event)
        adapter._link.unsubscribe_room.assert_called_once_with("r")


class TestRoomAddedRejoinFlagsRehydration:
    """room_added for a *known* room -> flag for rehydration (#4)."""

    @pytest.fixture
    def adapter(self, monkeypatch):
        a = _make_adapter(monkeypatch)
        a._link = MagicMock()
        a._link.subscribe_room = AsyncMock()
        return a

    @pytest.mark.asyncio
    async def test_new_room_is_tracked_not_flagged(self, adapter):
        event = SimpleNamespace(type="room_added", room_id="fresh-room")
        await adapter._handle_event(event)
        adapter._link.subscribe_room.assert_called_once_with("fresh-room")
        assert "fresh-room" in adapter._known_rooms
        assert "fresh-room" not in adapter._rehydrate_rooms

    @pytest.mark.asyncio
    async def test_rejoin_known_room_flags_rehydration(self, adapter):
        adapter._known_rooms.add("known-room")
        event = SimpleNamespace(type="room_added", room_id="known-room")
        await adapter._handle_event(event)
        adapter._link.subscribe_room.assert_called_once_with("known-room")
        # Re-join of a cold room: flagged for rehydration, no synthetic wake.
        assert "known-room" in adapter._rehydrate_rooms

    @pytest.mark.asyncio
    async def test_rejoin_room_with_live_session_not_flagged(self, adapter):
        # A room_added for a room that already has a live session (e.g. a
        # server-side re-subscribe replay) must NOT re-inject stale history.
        # Uses a load_transcript-bearing store so this exercises the real
        # history-based _has_active_session, not the entry-presence fallback.
        adapter._known_rooms.add("warm-room")
        adapter._session_store = _history_store(
            {"agent:main:band:group:warm-room": [{"role": "user", "content": "prior"}]}
        )
        event = SimpleNamespace(type="room_added", room_id="warm-room")
        await adapter._handle_event(event)
        assert "warm-room" not in adapter._rehydrate_rooms

    @pytest.mark.asyncio
    async def test_rejoin_room_with_empty_session_is_flagged(self, adapter):
        # The new history-based rule: a bare session entry with an EMPTY
        # transcript counts as cold, so a re-join still flags it for rehydration.
        # (Under the old entry-presence logic this room would be seen as warm and
        # NOT flagged — so this test fails if the history-based change is reverted.)
        adapter._known_rooms.add("empty-room")
        adapter._session_store = _history_store(
            {"agent:main:band:group:empty-room": []}
        )
        event = SimpleNamespace(type="room_added", room_id="empty-room")
        await adapter._handle_event(event)
        assert "empty-room" in adapter._rehydrate_rooms

    @pytest.mark.asyncio
    async def test_rejoin_cold_room_actually_drains_backlog(self, adapter):
        # A live re-join must drain the room so the backlog the seed excludes
        # actually gets answered. Drive the real _catch_up_room -> _drain_room
        # against a mocked link and assert a /next pull really happened.
        adapter._link.get_stale_processing_messages = AsyncMock(return_value=[])
        adapter._link.get_next_message = AsyncMock(return_value=None)  # empty backlog
        adapter._known_rooms.add("known-room")
        await adapter._handle_event(
            SimpleNamespace(type="room_added", room_id="known-room")
        )
        await asyncio.sleep(0)  # let the scheduled drain task run
        adapter._link.get_next_message.assert_awaited_with("known-room")

    @pytest.mark.asyncio
    async def test_rejoin_during_running_catchup_still_drains(self, adapter):
        # The seed↔drain invariant: a re-join always drains, even while an
        # all-rooms catch-up is in flight. That drain iterates a start-of-run
        # snapshot of _known_rooms, so a room re-joined mid-drain would never be
        # reached by it — skipping here would leave the excluded mention backlog
        # neither seeded nor answered until the next reconnect. Concurrent /
        # duplicate same-room drains are safe (mark_processing claim + inbound
        # dedup), so the re-join just drains.
        adapter._link.get_stale_processing_messages = AsyncMock(return_value=[])
        adapter._link.get_next_message = AsyncMock(return_value=None)
        adapter._known_rooms.add("known-room")
        adapter._catch_up_task = asyncio.create_task(asyncio.sleep(0.05))  # in flight
        await adapter._handle_event(
            SimpleNamespace(type="room_added", room_id="known-room")
        )
        await asyncio.sleep(0)  # let the scheduled per-room drain run
        adapter._link.get_next_message.assert_awaited_with("known-room")
        adapter._catch_up_task.cancel()


# ---------------------------------------------------------------------------
# 13. Cold-room rehydration & durable transcript seeding (Band as source of truth)
# ---------------------------------------------------------------------------

class TestHasActiveSession:
    """'Warm' means a session with real history, not a bare (empty) entry."""

    def _store(self, transcript):
        key = "agent:main:band:group:room-x"
        entry = SimpleNamespace(session_id="sid-x")

        class _Store:
            config = SimpleNamespace(group_sessions_per_user=False)
            _entries = {key: entry}

            def _ensure_loaded(self):
                pass

            def load_transcript(self, sid):
                return list(transcript)

        return _Store()

    def test_empty_transcript_is_not_active(self, monkeypatch):
        a = _make_adapter(monkeypatch)
        a._session_store = self._store([])
        assert a._has_active_session("room-x") is False

    def test_nonempty_transcript_is_active(self, monkeypatch):
        a = _make_adapter(monkeypatch)
        a._session_store = self._store([{"role": "user", "content": "hi"}])
        assert a._has_active_session("room-x") is True

    def test_missing_entry_is_not_active(self, monkeypatch):
        a = _make_adapter(monkeypatch)
        a._session_store = self._store([{"role": "user", "content": "hi"}])
        assert a._has_active_session("other-room") is False


class TestRehydrationOnNextMessage:
    """A flagged room's next message pulls Band context into channel_context."""

    @pytest.fixture
    def adapter(self, monkeypatch):
        a = _make_adapter(monkeypatch, agent_id="agent-self-id")
        a._agent_id = "agent-self-id"
        a.handle_message = AsyncMock()
        a._link = MagicMock()
        a._participants_cache["rejoined-room"] = [
            {"id": "agent-self-id", "type": "Agent", "name": "Bot", "handle": "bot"},
            {"id": "human-1", "type": "User", "name": "Alice", "handle": "alice"},
        ]
        return a

    def _make_event(self, room_id="rejoined-room"):
        payload = SimpleNamespace(
            id="msg-r1",
            content="hey, you back?",
            message_type="text",
            sender_id="human-1",
            sender_type="User",
            sender_name="Alice",
            chat_room_id=room_id,
            # Agent is @mentioned so the message clears the gate (every Band
            # room is mention-gated) and reaches the rehydration path.
            metadata=SimpleNamespace(
                mentions=[SimpleNamespace(id="agent-self-id", handle=None)]
            ),
        )
        return SimpleNamespace(type="message_created", room_id=room_id, payload=payload)

    @pytest.mark.asyncio
    async def test_flagged_room_hydrates_channel_context_and_clears_flag(self, adapter):
        adapter._rehydrate_rooms.add("rejoined-room")
        # Band context endpoint returns prior agent-relevant messages.
        ctx = SimpleNamespace(
            data=[
                SimpleNamespace(
                    message_type="text",
                    content="earlier note",
                    sender_name="Alice",
                ),
                SimpleNamespace(
                    message_type="tool_call",  # filtered out
                    content="ignored",
                    sender_name="Bot",
                ),
            ]
        )
        adapter._link.rest.agent_api_context.get_agent_chat_context = AsyncMock(
            return_value=ctx
        )

        await adapter._handle_message_created(self._make_event())

        adapter._link.rest.agent_api_context.get_agent_chat_context.assert_awaited_once()
        adapter.handle_message.assert_called_once()
        evt = adapter.handle_message.call_args[0][0]
        assert evt.channel_context is not None
        assert "earlier note" in evt.channel_context
        assert "ignored" not in evt.channel_context
        # Flag consumed -> not refetched next time.
        assert "rejoined-room" not in adapter._rehydrate_rooms

    @pytest.mark.asyncio
    async def test_unflagged_room_does_not_hydrate(self, adapter):
        adapter._link.rest.agent_api_context.get_agent_chat_context = AsyncMock()
        await adapter._handle_message_created(self._make_event())
        adapter._link.rest.agent_api_context.get_agent_chat_context.assert_not_called()
        evt = adapter.handle_message.call_args[0][0]
        assert evt.channel_context is None

    @pytest.mark.asyncio
    async def test_hydration_failure_still_delivers_message(self, adapter):
        adapter._rehydrate_rooms.add("rejoined-room")
        adapter._link.rest.agent_api_context.get_agent_chat_context = AsyncMock(
            side_effect=RuntimeError("context fetch boom")
        )
        await adapter._handle_message_created(self._make_event())
        # Message still delivered; channel_context simply None; flag cleared.
        adapter.handle_message.assert_called_once()
        evt = adapter.handle_message.call_args[0][0]
        assert evt.channel_context is None
        assert "rejoined-room" not in adapter._rehydrate_rooms


class _SeedingSessionStore(_FakeSessionStore):
    """Session store stand-in exposing the gateway-native atomic seed primitive.

    Records the rows passed to ``seed_transcript_if_empty`` in ``atomic_seeded``
    so tests can assert what was seeded. ``load_transcript`` starts empty (cold)
    unless constructed with ``existing_transcript`` (warm). An optional
    ``on_seed`` hook lets a test simulate a store failure during the write.
    """

    def __init__(
        self,
        *,
        existing_transcript=None,
        session_id="sess-1",
        on_seed=None,
        **kw,
    ):
        super().__init__(**kw)
        self._session_id = session_id
        self._transcripts = {session_id: list(existing_transcript or [])}
        self.atomic_seeded = None  # rows passed to seed_transcript_if_empty
        self._on_seed = on_seed

    def get_or_create_session(self, source):
        # Mirror the real store side effect: materialize an (empty) entry.
        self._entries.setdefault(self._session_id, object())
        return SimpleNamespace(session_id=self._session_id, session_key="k")

    def load_transcript(self, session_id):
        return list(self._transcripts.get(session_id, []))

    def seed_transcript_if_empty(self, session_id, messages):
        # Single-step count-then-insert: never clobbers a concurrently-filled
        # transcript (returns False when non-empty).
        if self._on_seed is not None:
            self._on_seed()
        if self._transcripts.get(session_id):
            return False
        self._transcripts[session_id] = list(messages)
        self.atomic_seeded = list(messages)
        return True


class TestDurableSeedRehydration:
    """Cold rooms seed the gateway transcript from Band (Band as source of truth)."""

    @pytest.fixture
    def adapter(self, monkeypatch):
        a = _make_adapter(monkeypatch, agent_id="agent-self-id")
        a._agent_id = "agent-self-id"
        a.handle_message = AsyncMock()
        a._link = MagicMock()
        # No actionable backlog by default.
        a._link.rest.agent_api_messages.list_agent_messages = AsyncMock(
            return_value=SimpleNamespace(
                data=[], metadata=SimpleNamespace(next_cursor=None, has_more=False)
            )
        )
        return a

    def _event(self, room_id="rejoined-room", msg_id="msg-r1"):
        payload = SimpleNamespace(
            id=msg_id,
            content="hey, you back?",
            message_type="text",
            sender_id="human-1",
            sender_type="User",
            sender_name="Alice",
            chat_room_id=room_id,
            metadata=SimpleNamespace(
                mentions=[SimpleNamespace(id="agent-self-id", handle=None)]
            ),
        )
        return SimpleNamespace(type="message_created", room_id=room_id, payload=payload)

    def _ctx(self, items):
        return AsyncMock(
            return_value=SimpleNamespace(
                data=items, metadata=SimpleNamespace(next_cursor=None, has_more=False)
            )
        )

    @pytest.mark.asyncio
    async def test_cold_room_seeds_transcript_with_roles_and_no_channel_context(
        self, adapter
    ):
        adapter._session_store = _SeedingSessionStore()  # cold (empty transcript)
        adapter._rehydrate_rooms.add("rejoined-room")
        adapter._link.rest.agent_api_context.get_agent_chat_context = self._ctx(
            [
                SimpleNamespace(
                    id="h1", message_type="text", content="what's the weather?",
                    sender_id="human-1", sender_type="User", sender_name="Alice",
                    inserted_at=None,
                ),
                SimpleNamespace(
                    id="h2", message_type="text", content="It's sunny.",
                    sender_id="agent-self-id", sender_type="Agent", sender_name="Bot",
                    inserted_at=None,
                ),
            ]
        )

        await adapter._handle_message_created(self._event())

        store = adapter._session_store
        # Durable path used → message carries no one-shot channel_context.
        evt = adapter.handle_message.call_args[0][0]
        assert evt.channel_context is None
        # Single atomic write; own reply as assistant (prevents re-answering),
        # peer as user with a [name] prefix.
        assert [(r["role"], r["content"]) for r in store.atomic_seeded] == [
            ("user", "[Alice] what's the weather?"),
            ("assistant", "It's sunny."),
        ]
        assert "rejoined-room" not in adapter._rehydrate_rooms

    @pytest.mark.asyncio
    async def test_cold_room_seeds_real_session_store_roundtrip(self, adapter, tmp_path):
        # Wire the REAL gateway SessionStore (temp SQLite) and mock ONLY the
        # Band REST endpoints (the genuine external service). This validates the
        # actual persistence contract — row-dict keys accepted, roles round-trip
        # through load_transcript — instead of a hand-rolled fake.
        from gateway.config import GatewayConfig
        from gateway.session import SessionStore
        from hermes_state import SessionDB

        store = SessionStore(tmp_path, GatewayConfig())
        store._db = SessionDB(db_path=tmp_path / "state.db")  # isolate from ~/.hermes
        adapter._session_store = store

        adapter._rehydrate_rooms.add("rejoined-room")
        adapter._link.rest.agent_api_context.get_agent_chat_context = self._ctx(
            [
                SimpleNamespace(id="h1", message_type="text", content="what's the weather?",
                                sender_id="human-1", sender_type="User",
                                sender_name="Alice", inserted_at=None),
                SimpleNamespace(id="h2", message_type="text", content="It's sunny.",
                                sender_id="agent-self-id", sender_type="Agent",
                                sender_name="Bot", inserted_at=None),
            ]
        )

        await adapter._handle_message_created(self._event())

        # Exactly one session exists — the seeded room — and the REAL
        # load_transcript reads back the user/assistant rows the seed wrote.
        assert len(store._entries) == 1
        entry = next(iter(store._entries.values()))
        assert "band:group:rejoined-room" in entry.session_key
        rows = store.load_transcript(entry.session_id)
        assert [(r["role"], r["content"]) for r in rows] == [
            ("user", "[Alice] what's the weather?"),
            ("assistant", "It's sunny."),
        ]
        # Durable path → no one-shot channel_context on the delivered message.
        evt = adapter.handle_message.call_args[0][0]
        assert evt.channel_context is None

    @pytest.mark.asyncio
    async def test_warm_session_is_not_reseeded_and_left_intact(self, adapter):
        store = _SeedingSessionStore(
            existing_transcript=[{"role": "user", "content": "prior"}]
        )
        adapter._session_store = store
        adapter._rehydrate_rooms.add("rejoined-room")
        adapter._link.rest.agent_api_context.get_agent_chat_context = self._ctx([])

        await adapter._handle_message_created(self._event())

        # Warm → no seed write, no clobber, no channel_context.
        assert store.atomic_seeded is None
        assert store.load_transcript("sess-1") == [{"role": "user", "content": "prior"}]
        evt = adapter.handle_message.call_args[0][0]
        assert evt.channel_context is None

    @pytest.mark.asyncio
    async def test_trigger_and_backlog_excluded_from_seed(self, adapter):
        adapter._session_store = _SeedingSessionStore()
        adapter._rehydrate_rooms.add("rejoined-room")
        # Actionable backlog: an unprocessed message that @mentions the agent.
        adapter._link.rest.agent_api_messages.list_agent_messages = AsyncMock(
            return_value=SimpleNamespace(
                data=[
                    SimpleNamespace(
                        id="backlog-1",
                        metadata={"mentions": [{"id": "agent-self-id"}]},
                    )
                ],
                metadata=SimpleNamespace(next_cursor=None, has_more=False),
            )
        )
        # Context includes the trigger (msg-r1), the backlog (backlog-1), and
        # one genuine history item (h1). Only h1 should be seeded.
        adapter._link.rest.agent_api_context.get_agent_chat_context = self._ctx(
            [
                SimpleNamespace(id="h1", message_type="text", content="older context",
                                sender_id="human-1", sender_type="User",
                                sender_name="Alice", inserted_at=None),
                SimpleNamespace(id="backlog-1", message_type="text", content="unanswered Q",
                                sender_id="human-1", sender_type="User",
                                sender_name="Alice", inserted_at=None),
                SimpleNamespace(id="msg-r1", message_type="text", content="hey, you back?",
                                sender_id="human-1", sender_type="User",
                                sender_name="Alice", inserted_at=None),
            ]
        )

        await adapter._handle_message_created(self._event())

        seeded = [r["content"] for r in adapter._session_store.atomic_seeded]
        assert seeded == ["[Alice] older context"]

    @pytest.mark.asyncio
    async def test_no_seed_api_falls_back_to_channel_context(self, adapter):
        # Store lacks the seed API → legacy one-shot channel_context blob.
        adapter._session_store = _FakeSessionStore()
        adapter._rehydrate_rooms.add("rejoined-room")
        adapter._link.rest.agent_api_context.get_agent_chat_context = self._ctx(
            [
                SimpleNamespace(id="h1", message_type="text", content="earlier note",
                                sender_id="human-1", sender_type="User",
                                sender_name="Alice", inserted_at=None),
            ]
        )

        await adapter._handle_message_created(self._event())

        evt = adapter.handle_message.call_args[0][0]
        assert evt.channel_context is not None
        assert "earlier note" in evt.channel_context

    @pytest.mark.asyncio
    async def test_context_fetch_failure_still_delivers_message(self, adapter):
        adapter._session_store = _SeedingSessionStore()
        adapter._rehydrate_rooms.add("rejoined-room")
        adapter._link.rest.agent_api_context.get_agent_chat_context = AsyncMock(
            side_effect=RuntimeError("context boom")
        )

        await adapter._handle_message_created(self._event())

        # Best-effort: message delivered; nothing seeded; the blob fallback also
        # has no context to offer (same failing endpoint) → channel_context None.
        adapter.handle_message.assert_called_once()
        assert adapter._session_store.atomic_seeded is None
        evt = adapter.handle_message.call_args[0][0]
        assert evt.channel_context is None

    @pytest.mark.asyncio
    async def test_seed_write_failure_falls_back_to_blob(self, adapter, caplog):
        # The seed write itself fails (transient store error) but context fetch
        # works → return False → caller produces the channel_context blob, so
        # the room is never silently left with no recovered context. The failure
        # is surfaced at WARNING (not debug): a raised seed is unexpected and
        # would otherwise silently disable durable rehydration.
        adapter._session_store = _SeedingSessionStore(
            on_seed=lambda: (_ for _ in ()).throw(RuntimeError("db locked"))
        )
        adapter._rehydrate_rooms.add("rejoined-room")
        adapter._link.rest.agent_api_context.get_agent_chat_context = self._ctx(
            [
                SimpleNamespace(id="h1", message_type="text", content="earlier note",
                                sender_id="human-1", sender_type="User",
                                sender_name="Alice", inserted_at=None),
            ]
        )

        with caplog.at_level(logging.WARNING, logger="hermes_band_platform.adapter"):
            await adapter._handle_message_created(self._event())

        evt = adapter.handle_message.call_args[0][0]
        assert evt.channel_context is not None
        assert "earlier note" in evt.channel_context
        # The seed failure is loud, not swallowed at debug.
        assert any(
            r.levelno == logging.WARNING and "seed failed" in r.message.lower()
            for r in caplog.records
        )

    def test_build_seed_rows_timestamps_are_monotonic(self, adapter):
        # The gateway replays a transcript ORDER BY (timestamp, id). A context
        # item with a missing inserted_at gets _seed_epoch's "now" fallback,
        # which would sort it AFTER genuine past history and scramble chronology
        # (an assistant reply landing before the user turn it answered). The
        # clamp keeps timestamps non-decreasing in list order so replay order ==
        # the server's oldest-first order regardless of timestamp gaps.
        import datetime as dt

        adapter._agent_id = "agent-self-id"
        t1 = dt.datetime(2024, 1, 1, tzinfo=dt.timezone.utc)
        t2 = dt.datetime(2024, 1, 2, tzinfo=dt.timezone.utc)
        items = [
            SimpleNamespace(id="a", message_type="text", content="first",
                            sender_id="human-1", sender_type="User",
                            sender_name="Alice", inserted_at=t1),
            SimpleNamespace(id="b", message_type="text", content="second (no ts)",
                            sender_id="human-1", sender_type="User",
                            sender_name="Alice", inserted_at=None),
            SimpleNamespace(id="c", message_type="text", content="third",
                            sender_id="human-1", sender_type="User",
                            sender_name="Alice", inserted_at=t2),
        ]
        rows = adapter._build_seed_rows(items, set(), [])
        # List (chronological) order is preserved...
        assert [r["content"] for r in rows] == [
            "[Alice] first",
            "[Alice] second (no ts)",
            "[Alice] third",
        ]
        # ...and timestamps never decrease, so ORDER BY (timestamp, id) cannot
        # reorder the now-stamped middle row ahead of the real t2 row that
        # follows it. (Without the clamp this list would be [t1, now, t2] —
        # unsorted — and replay would put "third" before "second (no ts)".)
        ts = [r["timestamp"] for r in rows]
        assert ts == sorted(ts)

    @pytest.mark.asyncio
    async def test_peer_agent_message_seeded_as_user(self, adapter):
        # A message from a DIFFERENT agent is a peer from our POV → seeded as a
        # `user` turn with a [name] prefix, NOT `assistant`. (Only our own
        # replies become assistant; this is the non-obvious INT-509-adjacent case.)
        store = _SeedingSessionStore()
        adapter._session_store = store
        adapter._rehydrate_rooms.add("rejoined-room")
        adapter._link.rest.agent_api_context.get_agent_chat_context = self._ctx(
            [
                SimpleNamespace(id="p1", message_type="text", content="from a peer bot",
                                sender_id="other-agent", sender_type="Agent",
                                sender_name="Helper", inserted_at=None),
                SimpleNamespace(id="o1", message_type="text", content="my reply",
                                sender_id="agent-self-id", sender_type="Agent",
                                sender_name="Me", inserted_at=None),
            ]
        )

        await adapter._handle_message_created(self._event())

        assert [(r["role"], r["content"]) for r in store.atomic_seeded] == [
            ("user", "[Helper] from a peer bot"),   # peer agent → user
            ("assistant", "my reply"),               # our own → assistant
        ]

    @pytest.mark.asyncio
    async def test_unaddressed_actionable_message_is_kept_in_seed(self, adapter):
        # Only *mentions* are excluded from the seed (they get answered). An
        # unprocessed message that does NOT mention the agent is context, not an
        # answer — it must stay in the seeded history, not be over-excluded.
        store = _SeedingSessionStore()
        adapter._session_store = store
        adapter._rehydrate_rooms.add("rejoined-room")
        adapter._link.rest.agent_api_messages.list_agent_messages = AsyncMock(
            return_value=SimpleNamespace(
                data=[
                    SimpleNamespace(id="mention-1",
                                    metadata={"mentions": [{"id": "agent-self-id"}]}),
                    SimpleNamespace(id="chatter-1", metadata={"mentions": []}),
                ],
                metadata=SimpleNamespace(next_cursor=None, has_more=False),
            )
        )
        adapter._link.rest.agent_api_context.get_agent_chat_context = self._ctx(
            [
                SimpleNamespace(id="chatter-1", message_type="text",
                                content="background chatter", sender_id="human-2",
                                sender_type="User", sender_name="Bob", inserted_at=None),
                SimpleNamespace(id="mention-1", message_type="text",
                                content="answer me", sender_id="human-1",
                                sender_type="User", sender_name="Alice", inserted_at=None),
            ]
        )

        await adapter._handle_message_created(self._event())

        # The mention is excluded (it'll be answered); the un-addressed chatter is kept.
        assert [r["content"] for r in store.atomic_seeded] == ["[Bob] background chatter"]

    @pytest.mark.asyncio
    async def test_fetch_room_context_follows_pagination(self, adapter):
        # The context fetch must follow next_cursor across pages (and terminate).
        store = _SeedingSessionStore()
        adapter._session_store = store
        adapter._rehydrate_rooms.add("rejoined-room")

        page1 = SimpleNamespace(
            data=[SimpleNamespace(id="h1", message_type="text", content="page one",
                                  sender_id="human-1", sender_type="User",
                                  sender_name="Alice", inserted_at=None)],
            metadata=SimpleNamespace(next_cursor="cursor-2", has_more=True),
        )
        page2 = SimpleNamespace(
            data=[SimpleNamespace(id="h2", message_type="text", content="page two",
                                  sender_id="human-1", sender_type="User",
                                  sender_name="Alice", inserted_at=None)],
            metadata=SimpleNamespace(next_cursor=None, has_more=False),
        )
        adapter._link.rest.agent_api_context.get_agent_chat_context = AsyncMock(
            side_effect=[page1, page2]
        )

        await adapter._handle_message_created(self._event())

        assert [r["content"] for r in store.atomic_seeded] == [
            "[Alice] page one",
            "[Alice] page two",
        ]

    @pytest.mark.asyncio
    async def test_seed_renders_uuid_mentions_with_participants(self, adapter):
        # @[[uuid]] mentions are rendered to @handle using the room's participants
        # cache so the seeded transcript is human-readable.
        store = _SeedingSessionStore()
        adapter._session_store = store
        adapter._participants_cache["rejoined-room"] = [
            {"id": "human-1", "handle": "alice"}
        ]
        adapter._rehydrate_rooms.add("rejoined-room")
        adapter._link.rest.agent_api_context.get_agent_chat_context = self._ctx(
            [
                SimpleNamespace(id="h1", message_type="text",
                                content="ping @[[human-1]] now", sender_id="human-2",
                                sender_type="User", sender_name="Bob", inserted_at=None),
            ]
        )

        await adapter._handle_message_created(self._event())

        assert store.atomic_seeded[0]["content"] == "[Bob] ping @alice now"


class TestSessionKeyParity:
    """_reset_room_session and _has_active_session derive the SAME key as the store."""

    @pytest.mark.asyncio
    async def test_reset_targets_the_real_session_key(self, monkeypatch, tmp_path):
        from gateway.config import GatewayConfig
        from gateway.session import SessionSource, SessionStore
        from hermes_state import SessionDB

        a = _make_adapter(monkeypatch)
        store = SessionStore(tmp_path, GatewayConfig())
        store._db = SessionDB(db_path=tmp_path / "s.db")
        a._session_store = store

        # _session_key_for must match the store's own derivation (single source).
        src = SessionSource(platform=a.platform, chat_id="room-x", chat_type="group")
        assert a._session_key_for("room-x") == store._generate_session_key(src)

        # Create + warm the room's session, then reset via the adapter (leave
        # path) and confirm it actually targeted that session (now cold).
        entry = store.get_or_create_session(src)
        store.rewrite_transcript(
            entry.session_id, [{"role": "user", "content": "hi", "timestamp": 1.0}]
        )
        assert a._has_active_session("room-x") is True
        a._reset_room_session("room-x")
        assert a._has_active_session("room-x") is False


class TestAtomicSeedTranscript:
    """_atomic_seed_transcript against the REAL SessionDB (the race-safe write)."""

    def _store(self, tmp_path):
        from gateway.config import GatewayConfig
        from gateway.session import SessionSource, SessionStore
        from hermes_state import SessionDB

        store = SessionStore(tmp_path, GatewayConfig())
        store._db = SessionDB(db_path=tmp_path / "s.db")
        return store, SessionSource

    def test_seeds_empty_session_and_counts(self, tmp_path):
        store, Src = self._store(tmp_path)
        from gateway.config import Platform
        e = store.get_or_create_session(
            Src(platform=Platform("telegram"), chat_id="r1", chat_type="group")
        )
        rows = [
            {"role": "user", "content": "[Al] hi", "timestamp": 1.0},
            {"role": "assistant", "content": "hello", "timestamp": 2.0},
        ]
        wrote = _band_mod._atomic_seed_transcript(store, e.session_id, rows)
        assert wrote is True
        got = store.load_transcript(e.session_id)
        assert [(m["role"], m["content"]) for m in got] == [
            ("user", "[Al] hi"),
            ("assistant", "hello"),
        ]
        assert store._db.message_count(e.session_id) == 2

    def test_no_op_on_nonempty_session(self, tmp_path):
        store, Src = self._store(tmp_path)
        from gateway.config import Platform
        e = store.get_or_create_session(
            Src(platform=Platform("telegram"), chat_id="r2", chat_type="group")
        )
        store._db.append_message(e.session_id, role="user", content="LIVE", timestamp=1.0)
        wrote = _band_mod._atomic_seed_transcript(
            store, e.session_id, [{"role": "assistant", "content": "SEED", "timestamp": 2.0}]
        )
        assert wrote is False
        contents = [m["content"] for m in store.load_transcript(e.session_id)]
        assert contents == ["LIVE"]  # not clobbered, seed not added

    def test_idempotent(self, tmp_path):
        store, Src = self._store(tmp_path)
        from gateway.config import Platform
        e = store.get_or_create_session(
            Src(platform=Platform("telegram"), chat_id="r3", chat_type="group")
        )
        rows = [{"role": "user", "content": "x", "timestamp": 1.0}]
        assert _band_mod._atomic_seed_transcript(store, e.session_id, rows) is True
        assert _band_mod._atomic_seed_transcript(store, e.session_id, rows) is False
        assert store._db.message_count(e.session_id) == 1

    def test_seeds_session_with_only_soft_deleted_rows(self, tmp_path):
        # A rewind/undo soft-deletes rows (active=0). load_transcript filters
        # active=1, so the gateway replays the session as empty/cold and the room
        # is flagged for rehydration. The atomic guard must agree — count only
        # ACTIVE rows — or it sees the inactive rows, no-ops (returns False), the
        # caller treats that as "handled" and skips the blob, and the room answers
        # cold with nothing seeded.
        store, Src = self._store(tmp_path)
        from gateway.config import Platform
        e = store.get_or_create_session(
            Src(platform=Platform("telegram"), chat_id="r6", chat_type="group")
        )
        store._db.append_message(e.session_id, role="user", content="OLD", timestamp=1.0)
        # Soft-delete every row, as a full rewind would.
        store._db._execute_write(
            lambda conn: conn.execute(
                "UPDATE messages SET active = 0 WHERE session_id = ?", (e.session_id,)
            )
        )
        assert store.load_transcript(e.session_id) == []  # cold to the gateway
        wrote = _band_mod._atomic_seed_transcript(
            store, e.session_id, [{"role": "user", "content": "[Al] hi", "timestamp": 2.0}]
        )
        assert wrote is True  # seeded, not mistaken for non-empty
        assert [m["content"] for m in store.load_transcript(e.session_id)] == ["[Al] hi"]

    def test_returns_none_without_db(self):
        # A store lacking _db/_execute_write can't do an atomic write.
        assert _band_mod._atomic_seed_transcript(
            SimpleNamespace(), "sid", [{"role": "user", "content": "x", "timestamp": 1.0}]
        ) is None

    def test_content_with_special_chars_roundtrips(self, tmp_path):
        store, Src = self._store(tmp_path)
        from gateway.config import Platform
        e = store.get_or_create_session(
            Src(platform=Platform("telegram"), chat_id="r4", chat_type="group")
        )
        weird = 'líne1\n"quoted" & <tag> 表情'
        _band_mod._atomic_seed_transcript(
            store, e.session_id, [{"role": "user", "content": weird, "timestamp": 1.0}]
        )
        assert store.load_transcript(e.session_id)[0]["content"] == weird

    def test_no_clobber_under_thread_contention(self, tmp_path):
        # The core safety property: a concurrent append is never lost, and the
        # seed never corrupts the transcript, across many fresh sessions.
        import threading

        store, Src = self._store(tmp_path)
        from gateway.config import Platform

        for i in range(40):
            e = store.get_or_create_session(
                Src(platform=Platform("telegram"), chat_id=f"c{i}", chat_type="group")
            )
            sid = e.session_id
            errs = []

            def _append():
                try:
                    store._db.append_message(sid, role="user", content="LIVE", timestamp=1.0)
                except Exception as ex:  # noqa: BLE001
                    errs.append(ex)

            def _seed():
                try:
                    _band_mod._atomic_seed_transcript(
                        store, sid, [{"role": "assistant", "content": "SEED", "timestamp": 2.0}]
                    )
                except Exception as ex:  # noqa: BLE001
                    errs.append(ex)

            ta = threading.Thread(target=_append)
            tb = threading.Thread(target=_seed)
            tb.start()
            ta.start()
            tb.join()
            ta.join()
            assert not errs, errs
            contents = [m["content"] for m in store.load_transcript(sid)]
            assert "LIVE" in contents  # the append is never clobbered


class TestSeedHelpers:
    """Pure helpers behind the durable seed."""

    def test_seedable_text_skips_non_text_and_empty(self):
        text_item = SimpleNamespace(
            message_type="text", content=" hi ", sender_type="User",
            sender_id="u1", sender_name="Al",
        )
        # Strips whitespace and returns the (type, id, name, content) tuple.
        assert _band_mod._seedable_text(text_item, []) == ("User", "u1", "Al", "hi")
        tool_item = SimpleNamespace(message_type="tool_call", content="x")
        assert _band_mod._seedable_text(tool_item, []) is None
        empty_item = SimpleNamespace(message_type="text", content="   ")
        assert _band_mod._seedable_text(empty_item, []) is None

    def test_next_page_terminates_on_falsey_or_missing_metadata(self):
        # Guards the pagination loops against spinning: only a real string cursor
        # AND has_more is True continues; everything else ends the loop.
        nxt = _band_mod.BandAdapter._next_page
        assert nxt(SimpleNamespace(
            metadata=SimpleNamespace(next_cursor="c", has_more=True))) == ("c", True)
        assert nxt(SimpleNamespace(
            metadata=SimpleNamespace(next_cursor="c", has_more=False))) == (None, False)
        assert nxt(SimpleNamespace(
            metadata=SimpleNamespace(next_cursor=None, has_more=True))) == (None, False)
        assert nxt(SimpleNamespace()) == (None, False)              # no metadata
        assert nxt(SimpleNamespace(metadata=MagicMock())) == (None, False)  # truthy mock

    def test_seed_epoch_coerces_or_falls_back(self):
        import datetime as _dt

        dt = _dt.datetime(2020, 1, 1, tzinfo=_dt.timezone.utc)
        assert _band_mod._seed_epoch(dt) == dt.timestamp()
        # None / unparseable → a real epoch, never raises (keeps ordering sane).
        assert isinstance(_band_mod._seed_epoch(None), float)
        assert isinstance(_band_mod._seed_epoch("not-a-time"), float)


# ---------------------------------------------------------------------------
# 14. Route A — server-cursor ack lifecycle + /next missed-message catch-up
# ---------------------------------------------------------------------------

def _ack_link():
    """A MagicMock link with the ack/catch-up helpers wired as AsyncMocks."""
    link = MagicMock()
    link.mark_processing = AsyncMock()
    link.mark_processed = AsyncMock()
    link.mark_failed = AsyncMock()
    link.get_next_message = AsyncMock(return_value=None)
    link.get_stale_processing_messages = AsyncMock(return_value=[])
    return link


def _platform_msg(msg_id, *, content="hi @bot", sender_id="human-1",
                  sender_type="User", message_type="text", mentions_agent=True):
    """A /next-style PlatformMessage: dict metadata, mentions as dicts."""
    metadata = {"mentions": [{"id": "agent-self-id", "handle": None}]} if mentions_agent else {}
    return SimpleNamespace(
        id=msg_id,
        content=content,
        message_type=message_type,
        sender_id=sender_id,
        sender_type=sender_type,
        sender_name="Alice",
        chat_room_id="room-abc",
        metadata=metadata,
    )


class TestProcessingAckHooks:
    """on_processing_start/complete drive the server-side read cursor."""

    @pytest.fixture
    def adapter(self, monkeypatch):
        a = _make_adapter(monkeypatch, agent_id="agent-self-id")
        a._agent_id = "agent-self-id"
        a._link = _ack_link()
        return a

    def _evt(self, msg_id="m1", room_id="room-abc", internal=False):
        return SimpleNamespace(
            message_id=msg_id,
            internal=internal,
            source=SimpleNamespace(chat_id=room_id),
        )

    @pytest.mark.asyncio
    async def test_start_marks_processing(self, adapter):
        await adapter.on_processing_start(self._evt())
        adapter._link.mark_processing.assert_awaited_once_with("room-abc", "m1")

    @pytest.mark.asyncio
    async def test_success_marks_processed(self, adapter):
        await adapter.on_processing_complete(self._evt(), ProcessingOutcome.SUCCESS)
        adapter._link.mark_processed.assert_awaited_once_with("room-abc", "m1")
        adapter._link.mark_failed.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancelled_marks_processed_not_failed(self, adapter):
        # A cancelled turn was superseded or intentionally stopped — consumed,
        # so don't re-deliver it.
        await adapter.on_processing_complete(self._evt(), ProcessingOutcome.CANCELLED)
        adapter._link.mark_processed.assert_awaited_once_with("room-abc", "m1")
        adapter._link.mark_failed.assert_not_called()

    @pytest.mark.asyncio
    async def test_failure_marks_failed(self, adapter):
        await adapter.on_processing_complete(self._evt(), ProcessingOutcome.FAILURE)
        adapter._link.mark_failed.assert_awaited_once()
        adapter._link.mark_processed.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "outcome", [ProcessingOutcome.FAILURE, ProcessingOutcome.CANCELLED]
    )
    async def test_incomplete_outcomes_flush_originating_usage(
        self, adapter, monkeypatch, outcome
    ):
        flush = MagicMock()
        monkeypatch.setattr(_band_mod.usage_events, "flush_incomplete_turn", flush)
        monkeypatch.setattr(adapter, "_session_id_for_event", lambda event: "sid-1")
        event = self._evt(room_id="room-abc")

        await adapter.on_processing_complete(event, outcome)

        flush.assert_called_once_with(adapter, "sid-1", "room-abc")

    @pytest.mark.asyncio
    async def test_success_does_not_use_incomplete_flush(self, adapter, monkeypatch):
        flush = MagicMock()
        monkeypatch.setattr(_band_mod.usage_events, "flush_incomplete_turn", flush)

        await adapter.on_processing_complete(self._evt(), ProcessingOutcome.SUCCESS)

        flush.assert_not_called()

    @pytest.mark.asyncio
    async def test_duplicate_incomplete_finalization_flushes_once(
        self, adapter, monkeypatch
    ):
        usage = _band_mod.usage_events
        usage._PENDING.clear()
        monkeypatch.setenv("BAND_EMIT_USAGE", "all")
        # The scope is captured once at hook registration and never re-read, so
        # setting the env var alone cannot enable emission here: the ``adapter``
        # fixture has already registered hooks and pinned ``_STARTUP_SCOPE`` to
        # "off". Clear it so ``_effective_scope()`` falls through to the env.
        # ``test_usage_events.py`` does the same for its whole module via an
        # autouse fixture.
        monkeypatch.setattr(usage, "_STARTUP_SCOPE", None)
        emitted = MagicMock(return_value=True)
        monkeypatch.setattr(usage, "_schedule_emit", emitted)
        flush = MagicMock(wraps=usage.flush_incomplete_turn)
        monkeypatch.setattr(usage, "flush_incomplete_turn", flush)
        monkeypatch.setattr(adapter, "_session_id_for_event", lambda event: "sid-1")
        monkeypatch.setattr(usage, "_live_adapters", lambda: [adapter])
        monkeypatch.setattr(usage, "_room_for_session", lambda a, sid: "room-abc")
        usage.on_post_api_request(
            session_id="sid-1",
            turn_id="turn-1",
            platform="band",
            model="test-model",
            provider="test-provider",
            usage={
                "input_tokens": 10,
                "output_tokens": 2,
                "cache_read_tokens": 0,
                "cache_write_tokens": 0,
            },
        )
        source = adapter.build_source(chat_id="room-abc", chat_type="group")
        first = _band_mod.MessageEvent(
            text="hello", source=source, message_id="m1"
        )
        duplicate = _band_mod.MessageEvent(
            text="hello", source=source, message_id="m1"
        )

        try:
            await adapter.on_processing_complete(first, ProcessingOutcome.FAILURE)
            await adapter.on_processing_complete(duplicate, ProcessingOutcome.FAILURE)
        finally:
            usage._PENDING.clear()

        assert flush.call_count == 2
        emitted.assert_called_once()
        assert not hasattr(first, "_band_usage_finalized")
        assert not hasattr(duplicate, "_band_usage_finalized")

    def test_usage_session_resolution_uses_store_profile_key(self, adapter):
        adapter.config.extra["group_sessions_per_user"] = True
        source = adapter.build_source(
            chat_id="room-abc", chat_type="group", user_id="user-7"
        )
        key = "agent:profile:review-profile:band:group:room-abc"
        generate_key = MagicMock(return_value=key)
        adapter._session_store = SimpleNamespace(
            config=SimpleNamespace(group_sessions_per_user=False),
            _generate_session_key=generate_key,
            _ensure_loaded=lambda: None,
            _entries={key: SimpleNamespace(session_id="sid-user-7")},
        )

        event = SimpleNamespace(source=source)

        assert adapter._session_id_for_event(event) == "sid-user-7"
        generate_key.assert_called_once_with(source)

    def test_usage_session_resolution_fallback_uses_store_config(self, adapter):
        adapter.config.extra["group_sessions_per_user"] = True
        source = adapter.build_source(
            chat_id="room-abc", chat_type="group", user_id="user-7"
        )
        key = _band_mod.build_session_key(
            source, group_sessions_per_user=False, thread_sessions_per_user=False
        )
        adapter._session_store = SimpleNamespace(
            config=SimpleNamespace(
                group_sessions_per_user=False,
                thread_sessions_per_user=False,
            ),
            _ensure_loaded=lambda: None,
            _entries={key: SimpleNamespace(session_id="sid-shared")},
        )

        assert adapter._session_id_for_event(SimpleNamespace(source=source)) == "sid-shared"

    @pytest.mark.asyncio
    async def test_internal_event_is_not_acked(self, adapter):
        # Participant notices carry no Band id — never touch the cursor.
        await adapter.on_processing_start(self._evt(internal=True))
        await adapter.on_processing_complete(self._evt(internal=True), ProcessingOutcome.SUCCESS)
        adapter._link.mark_processing.assert_not_called()
        adapter._link.mark_processed.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_message_id_is_not_acked(self, adapter):
        await adapter.on_processing_complete(self._evt(msg_id=None), ProcessingOutcome.SUCCESS)
        adapter._link.mark_processed.assert_not_called()

    @pytest.mark.asyncio
    async def test_ack_swallows_link_errors(self, adapter):
        adapter._link.mark_processed.side_effect = RuntimeError("boom")
        # Must not raise — acking is best-effort.
        await adapter.on_processing_complete(self._evt(), ProcessingOutcome.SUCCESS)


class TestBlockedCommandIsAcked:
    """A dropped (non-owner) command is command-shaped @mention text, which
    /next would re-offer — so it must be made terminal when dropped."""

    @pytest.fixture
    def adapter(self, monkeypatch):
        a = _make_adapter(monkeypatch, agent_id="agent-self-id")
        a._agent_id = "agent-self-id"
        a._owner_uuid = "owner-1"
        a._link = _ack_link()
        a.handle_message = AsyncMock()
        a._notify_command_blocked = AsyncMock()
        a._participants_cache["room-abc"] = [
            {"id": "agent-self-id", "type": "Agent", "name": "Bot", "handle": "bot"},
            {"id": "human-1", "type": "User", "name": "Alice", "handle": "alice"},
        ]
        return a

    @pytest.mark.asyncio
    async def test_blocked_command_marks_processed_and_drops(self, adapter):
        payload = SimpleNamespace(
            id="cmd-1", content="/reset", message_type="text",
            sender_id="human-1", sender_type="User", sender_name="Alice",
            chat_room_id="room-abc",
            metadata=SimpleNamespace(mentions=[SimpleNamespace(id="agent-self-id", handle=None)]),
        )
        event = SimpleNamespace(type="message_created", room_id="room-abc", payload=payload)
        forwarded = await adapter._handle_message_created(event)
        assert forwarded is False
        adapter.handle_message.assert_not_called()
        adapter._link.mark_processed.assert_awaited_once_with("room-abc", "cmd-1")


class TestCatchUpDrain:
    """/next drain + stale-processing sweep on (re)connect."""

    @pytest.fixture
    def adapter(self, monkeypatch):
        a = _make_adapter(monkeypatch, agent_id="agent-self-id")
        a._agent_id = "agent-self-id"
        a._message_handler = AsyncMock()  # gateway handler is wired before connect
        a._link = _ack_link()
        a._participants_cache["room-abc"] = [
            {"id": "agent-self-id", "type": "Agent", "name": "Bot", "handle": "bot"},
            {"id": "human-1", "type": "User", "name": "Alice", "handle": "alice"},
        ]
        # Capture forwarded events instead of running the gateway.
        a.handle_message = AsyncMock()
        return a

    @pytest.mark.asyncio
    async def test_drain_dispatches_until_next_empty(self, adapter):
        msgs = [_platform_msg("c1"), _platform_msg("c2"), None]
        adapter._link.get_next_message = AsyncMock(side_effect=msgs)
        await adapter._drain_room("room-abc")
        # Both backlog messages forwarded to the gateway.
        assert adapter.handle_message.await_count == 2
        forwarded_ids = {
            c.args[0].message_id for c in adapter.handle_message.await_args_list
        }
        assert forwarded_ids == {"c1", "c2"}
        # Each was claimed (processing) before the next /next call.
        assert adapter._link.mark_processing.await_count == 2

    @pytest.mark.asyncio
    async def test_drain_forwarded_message_not_terminal_acked_by_drain(self, adapter):
        # A forwarded message is settled by the completion hook later, NOT by the
        # drain — the drain only marks it processing.
        adapter._link.get_next_message = AsyncMock(side_effect=[_platform_msg("c1"), None])
        await adapter._drain_room("room-abc")
        adapter._link.mark_processed.assert_not_called()

    @pytest.mark.asyncio
    async def test_drain_dropped_message_is_made_terminal(self, adapter):
        # A non-mention message is dropped by gating; /next would re-offer it,
        # so the drain marks it processed itself.
        dropped = _platform_msg("d1", mentions_agent=False)
        adapter._link.get_next_message = AsyncMock(side_effect=[dropped, None])
        await adapter._drain_room("room-abc")
        adapter.handle_message.assert_not_called()
        adapter._link.mark_processed.assert_awaited_with("room-abc", "d1")

    @pytest.mark.asyncio
    async def test_drain_reoffered_id_is_force_acked_not_respun(self, adapter):
        # Server pathologically re-offers the same id; the seen-set backstop
        # force-acks it so the loop terminates.
        m = _platform_msg("c1")
        adapter._link.get_next_message = AsyncMock(side_effect=[m, m, None])
        await adapter._drain_room("room-abc")
        assert adapter.handle_message.await_count == 1  # dispatched once
        adapter._link.mark_processed.assert_awaited_with("room-abc", "c1")  # force-acked on re-offer

    @pytest.mark.asyncio
    async def test_catch_up_sweeps_stale_then_drains_each_room(self, adapter):
        adapter._known_rooms = {"room-abc"}
        stale = [_platform_msg("s1")]
        adapter._link.get_stale_processing_messages = AsyncMock(return_value=stale)
        adapter._link.get_next_message = AsyncMock(side_effect=[_platform_msg("c1"), None])
        await adapter._catch_up_all_rooms()
        adapter._link.get_stale_processing_messages.assert_awaited_once_with("room-abc")
        # Stale (s1) + backlog (c1) both dispatched.
        forwarded_ids = {
            c.args[0].message_id for c in adapter.handle_message.await_args_list
        }
        assert forwarded_ids == {"s1", "c1"}

    @pytest.mark.asyncio
    async def test_catch_up_noop_without_message_handler(self, adapter):
        adapter._message_handler = None
        adapter._known_rooms = {"room-abc"}
        await adapter._catch_up_all_rooms()
        adapter._link.get_next_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_drain_swallows_next_errors(self, adapter):
        adapter._link.get_next_message = AsyncMock(side_effect=RuntimeError("net down"))
        # Must not raise.
        await adapter._drain_room("room-abc")


class TestCatchUpRehydratesColdRoom:
    """Agent-lifecycle rehydration: a (re)connect drain flags rooms with no
    local session so the backlog is answered with recovered Band history, not
    cold. Decision A — detect via session-store emptiness."""

    @pytest.fixture
    def adapter(self, monkeypatch):
        a = _make_adapter(monkeypatch, agent_id="agent-self-id")
        a._agent_id = "agent-self-id"
        a._message_handler = AsyncMock()
        a._link = _ack_link()
        a._known_rooms = {"room-abc"}
        a._participants_cache["room-abc"] = [
            {"id": "agent-self-id", "type": "Agent", "name": "Bot", "handle": "bot"},
            {"id": "human-1", "type": "User", "name": "Alice", "handle": "alice"},
        ]
        a.handle_message = AsyncMock()
        return a

    @staticmethod
    def _ctx():
        return SimpleNamespace(
            data=[SimpleNamespace(message_type="text", content="earlier note", sender_name="Alice")]
        )

    @pytest.mark.asyncio
    async def test_cold_room_backlog_is_rehydrated(self, adapter):
        # No local session for the room (fresh deploy / lost DB) -> flagged ->
        # the first caught-up message carries recovered history.
        adapter._session_store = _FakeSessionStore()  # empty: no active session
        adapter._link.get_next_message = AsyncMock(side_effect=[_platform_msg("c1"), None])
        adapter._link.rest.agent_api_context.get_agent_chat_context = AsyncMock(
            return_value=self._ctx()
        )

        await adapter._catch_up_all_rooms()

        adapter._link.rest.agent_api_context.get_agent_chat_context.assert_awaited_once()
        evt = adapter.handle_message.call_args[0][0]
        assert evt.message_id == "c1"
        assert evt.channel_context is not None
        assert "earlier note" in evt.channel_context

    @pytest.mark.asyncio
    async def test_active_session_room_is_not_rehydrated(self, adapter):
        # Local session intact -> history is in the store -> no platform refetch.
        adapter._session_store = _FakeSessionStore(
            active_keys=["agent:main:band:group:room-abc"]
        )
        adapter._link.get_next_message = AsyncMock(side_effect=[_platform_msg("c1"), None])
        adapter._link.rest.agent_api_context.get_agent_chat_context = AsyncMock()

        await adapter._catch_up_all_rooms()

        adapter._link.rest.agent_api_context.get_agent_chat_context.assert_not_called()
        evt = adapter.handle_message.call_args[0][0]
        assert evt.channel_context is None
        assert "room-abc" not in adapter._rehydrate_rooms

    @pytest.mark.asyncio
    async def test_cold_room_without_backlog_stays_flagged(self, adapter):
        # No backlog to consume the flag now -> it persists so a later live
        # message in the room still rehydrates.
        adapter._session_store = _FakeSessionStore()  # empty
        # _ack_link defaults: get_next_message -> None, no stale messages.
        await adapter._catch_up_all_rooms()
        assert "room-abc" in adapter._rehydrate_rooms

    @pytest.mark.asyncio
    async def test_warns_once_when_gateway_isolates_sessions_per_user(
        self, adapter, caplog
    ):
        # Band rooms are shared channels; a gateway with
        # group_sessions_per_user=True fragments them per user, so per-room
        # session reset/rehydration target a key the per-user sessions don't
        # use. We can't change the store config, so surface it loudly — exactly
        # once, not on every reconnect.
        adapter._session_store = _FakeSessionStore(group_sessions_per_user=True)
        with caplog.at_level(logging.WARNING, logger="hermes_band_platform.adapter"):
            await adapter._catch_up_all_rooms()
            await adapter._catch_up_all_rooms()  # second connect — must not re-warn
        warnings = [
            r for r in caplog.records
            if r.levelno == logging.WARNING
            and "group_sessions_per_user" in r.message
        ]
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_no_isolation_warning_in_shared_session_mode(self, adapter, caplog):
        # The supported config (group_sessions_per_user=False) is silent.
        adapter._session_store = _FakeSessionStore(group_sessions_per_user=False)
        with caplog.at_level(logging.WARNING, logger="hermes_band_platform.adapter"):
            await adapter._catch_up_all_rooms()
        assert not [
            r for r in caplog.records if "group_sessions_per_user" in r.message
        ]


class TestReconnectSchedulesCatchUp:

    @pytest.mark.asyncio
    async def test_reconnected_event_schedules_catch_up(self, monkeypatch):
        a = _make_adapter(monkeypatch)
        a._link = _ack_link()
        scheduled = {"n": 0}
        monkeypatch.setattr(a, "_schedule_catch_up", lambda: scheduled.__setitem__("n", scheduled["n"] + 1))
        await a._handle_event(SimpleNamespace(type="reconnected"))
        assert scheduled["n"] == 1

    @pytest.mark.asyncio
    async def test_schedule_catch_up_is_idempotent_while_running(self, monkeypatch):
        a = _make_adapter(monkeypatch)
        a._link = _ack_link()
        a._message_handler = AsyncMock()
        a._known_rooms = set()
        a._schedule_catch_up()
        first = a._catch_up_task
        a._schedule_catch_up()  # one already in flight → no new task
        assert a._catch_up_task is first
        await first  # let it finish cleanly


class TestInboundDedup:
    """A message already dispatched this lifetime isn't double-processed."""

    @pytest.fixture
    def adapter(self, monkeypatch):
        a = _make_adapter(monkeypatch, agent_id="agent-self-id")
        a._agent_id = "agent-self-id"
        a._link = _ack_link()
        a.handle_message = AsyncMock()
        a._participants_cache["room-abc"] = [
            {"id": "agent-self-id", "type": "Agent", "name": "Bot", "handle": "bot"},
            {"id": "human-1", "type": "User", "name": "Alice", "handle": "alice"},
        ]
        return a

    @pytest.mark.asyncio
    async def test_second_dispatch_of_same_id_is_skipped(self, adapter):
        msg = _platform_msg("dup-1")
        # First dispatch forwards.
        assert await adapter._dispatch_caught_up("room-abc", msg) is True
        # Second (e.g. live arriving after the drain) is treated as forwarded
        # but does NOT re-invoke the gateway.
        assert await adapter._dispatch_caught_up("room-abc", msg) is True
        assert adapter.handle_message.await_count == 1


class TestMentionParsingDictMetadata:
    """_is_agent_mentioned must read caught-up dict metadata, not just objects."""

    def test_dict_metadata_mention_by_id(self, monkeypatch):
        a = _make_adapter(monkeypatch, agent_id="agent-self-id")
        a._agent_id = "agent-self-id"
        payload = SimpleNamespace(metadata={"mentions": [{"id": "agent-self-id"}]})
        assert a._is_agent_mentioned(payload) is True

    def test_dict_metadata_mention_by_handle(self, monkeypatch):
        a = _make_adapter(monkeypatch, agent_id="agent-self-id")
        a._handle = "bot"
        payload = SimpleNamespace(metadata={"mentions": [{"handle": "bot"}]})
        assert a._is_agent_mentioned(payload) is True

    def test_dict_metadata_no_mention(self, monkeypatch):
        a = _make_adapter(monkeypatch, agent_id="agent-self-id")
        a._agent_id = "agent-self-id"
        payload = SimpleNamespace(metadata={"mentions": [{"id": "someone-else"}]})
        assert a._is_agent_mentioned(payload) is False


# ---------------------------------------------------------------------------
# 15. connect/disconnect lifecycle
# ---------------------------------------------------------------------------

class TestConnectDisconnect:

    def test_connect_accepts_is_reconnect_kwarg(self):
        """gateway/run.py calls every adapter as connect(is_reconnect=...).

        hermes-agent pins this contract for its in-tree adapters with an
        AST-based test (test_adapter_connect_is_reconnect_contract.py), which
        cannot see an external plugin like this one — so the same contract is
        pinned here. Without the kwarg the gateway cannot start the adapter
        at all (TypeError before any connection attempt).
        """
        import inspect

        param = inspect.signature(BandAdapter.connect).parameters.get(
            "is_reconnect"
        )
        assert param is not None, "connect() must accept is_reconnect"
        assert param.kind is inspect.Parameter.KEYWORD_ONLY
        assert param.default is False

    @pytest.mark.asyncio
    async def test_connect_returns_false_when_band_unavailable(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        monkeypatch.setattr(_band_mod, "BAND_AVAILABLE", False)
        # Called with the kwarg exactly as gateway/run.py does.
        result = await adapter.connect(is_reconnect=False)
        assert result is False
        monkeypatch.setattr(_band_mod, "BAND_AVAILABLE", True)

    @pytest.mark.asyncio
    async def test_connect_returns_false_when_no_credentials(self, monkeypatch):
        monkeypatch.delenv("BAND_AGENT_ID", raising=False)
        monkeypatch.delenv("BAND_API_KEY", raising=False)
        monkeypatch.delenv("BAND_BASE_URL", raising=False)
        monkeypatch.delenv("BAND_OWNER_ID", raising=False)
        adapter = BandAdapter(_make_config())
        result = await adapter.connect()
        assert result is False

    @pytest.mark.asyncio
    async def test_connect_returns_true_on_success(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)

        # Patch out scoped lock so the test doesn't interact with global state
        monkeypatch.setattr(
            "gateway.status.acquire_scoped_lock",
            lambda scope, identity, metadata=None: (True, None),
        )
        monkeypatch.setattr(
            "gateway.status.release_scoped_lock",
            lambda scope, identity: None,
        )

        # Build a fake BandLink instance
        fake_link = MagicMock()
        fake_link.connect = AsyncMock()
        fake_link.subscribe_agent_rooms = AsyncMock()
        fake_link.subscribe_room = AsyncMock()

        # Identity response
        fake_me = SimpleNamespace(
            data=SimpleNamespace(
                id="resolved-agent-id",
                handle="bot-handle",
                owner_uuid="owner-uuid-abc",
            )
        )
        fake_link.rest.agent_api_identity.get_agent_me = AsyncMock(return_value=fake_me)

        # list_agent_chats returns empty list (no rooms to pre-subscribe)
        fake_link.rest.agent_api_chats.list_agent_chats = AsyncMock(
            return_value=SimpleNamespace(data=[], metadata=SimpleNamespace(total_pages=1))
        )

        # Async iterator that stops immediately
        fake_link.__aiter__ = lambda self: self
        fake_link.__anext__ = AsyncMock(side_effect=StopAsyncIteration)

        # Patch BandLink class in the band module
        monkeypatch.setattr(_band_mod, "BandLink", lambda *a, **kw: fake_link)

        result = await adapter.connect()
        assert result is True
        assert adapter._agent_id == "resolved-agent-id"
        assert adapter._handle == "bot-handle"
        assert adapter._consumer_task is not None
        assert adapter in _band_mod.usage_events._adapters()

        # Cleanup
        await adapter.disconnect()
        assert adapter not in _band_mod.usage_events._adapters()

    @pytest.mark.asyncio
    async def test_connect_returns_false_on_link_exception(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        _band_mod.usage_events.track_adapter(adapter)

        monkeypatch.setattr(
            "gateway.status.acquire_scoped_lock",
            lambda scope, identity, metadata=None: (True, None),
        )
        monkeypatch.setattr(
            "gateway.status.release_scoped_lock",
            lambda scope, identity: None,
        )

        def _bad_link(*a, **kw):
            raise ConnectionError("refused")

        monkeypatch.setattr(_band_mod, "BandLink", _bad_link)
        result = await adapter.connect()
        assert result is False
        assert adapter not in _band_mod.usage_events._adapters()

    @pytest.mark.asyncio
    async def test_disconnect_cancels_consumer_task(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)

        # Simulate a connected state with a long-running consumer task
        async def _forever():
            await asyncio.sleep(999)

        task = asyncio.create_task(_forever())
        adapter._consumer_task = task

        fake_link = MagicMock()
        fake_link.disconnect = AsyncMock()
        adapter._link = fake_link

        await adapter.disconnect()

        assert task.cancelled() or task.done()
        assert adapter._link is None
        assert adapter._consumer_task is None

    @pytest.mark.asyncio
    async def test_disconnect_calls_link_disconnect(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        fake_link = MagicMock()
        fake_link.disconnect = AsyncMock()
        adapter._link = fake_link

        await adapter.disconnect()
        fake_link.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_cancels_room_catch_up_tasks(self, monkeypatch):
        # Per-room re-join drains must not outlive the link — otherwise a stale
        # drain can resume against a freshly reconnected link.
        adapter = _make_adapter(monkeypatch)

        async def _forever():
            await asyncio.sleep(999)

        task = asyncio.create_task(_forever())
        adapter._room_catch_up_tasks.add(task)
        task.add_done_callback(adapter._room_catch_up_tasks.discard)

        fake_link = MagicMock()
        fake_link.disconnect = AsyncMock()
        adapter._link = fake_link

        await adapter.disconnect()

        assert task.cancelled() or task.done()
        assert adapter._room_catch_up_tasks == set()

    @pytest.mark.asyncio
    async def test_disconnect_forgets_working_state_for_reconnect(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        old_link = MagicMock()
        old_link.disconnect = AsyncMock()
        adapter._link = old_link
        adapter._working_reported["room-1"] = time.monotonic()

        await adapter.disconnect()

        assert adapter._working_reported == {}
        new_link = TestWorkingIndicator._link()
        adapter._link = new_link
        await adapter.send_typing("room-1")
        assert new_link.calls == [("room-1", True)]
    async def test_disconnect_cancels_pending_execution_emissions(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        pending = concurrent.futures.Future()
        adapter._execution_pending.add(pending)
        adapter._execution_accepting = True

        fake_link = MagicMock()
        fake_link.disconnect = AsyncMock()
        adapter._link = fake_link

        await adapter.disconnect()

        assert pending.cancelled()
        assert adapter._execution_pending == set()
        assert adapter._execution_accepting is False


# ---------------------------------------------------------------------------
# 16. _record_sent_id eviction
# ---------------------------------------------------------------------------

class TestRecordSentId:

    def test_evicts_half_when_over_cap(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        # Fill up to the cap
        for i in range(_band_mod._SENT_IDS_MAX):
            adapter._sent_ids.add(f"id-{i}")
        # Adding one more should trigger eviction
        adapter._record_sent_id("overflow-id")
        # After eviction: should be around SENT_IDS_MAX // 2 + 1 entries
        assert len(adapter._sent_ids) <= (_band_mod._SENT_IDS_MAX // 2) + 2
        assert "overflow-id" in adapter._sent_ids


# ---------------------------------------------------------------------------
# 17. get_chat_info
# ---------------------------------------------------------------------------

class TestGetChatInfo:

    @pytest.mark.asyncio
    async def test_returns_group_for_two_participant_room(self, monkeypatch):
        # Band has no DMs — a 2-participant room is a regular group chat.
        adapter = _make_adapter(monkeypatch)
        adapter._participants_cache["chat-room"] = [
            {"id": "agent", "type": "Agent", "name": "Bot", "handle": "bot"},
            {"id": "user1", "type": "User", "name": "Alice", "handle": "alice"},
        ]
        info = await adapter.get_chat_info("chat-room")
        assert info["type"] == "group"

    @pytest.mark.asyncio
    async def test_returns_hub_for_hub_room(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._hub_room_id = "hub-room"
        adapter._participants_cache["hub-room"] = [
            {"id": "agent", "type": "Agent"},
            {"id": "owner", "type": "User"},
        ]
        info = await adapter.get_chat_info("hub-room")
        assert info["type"] == "hub"

    @pytest.mark.asyncio
    async def test_returns_group_for_three_participants(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        adapter._participants_cache["grp-room"] = [
            {"id": "agent", "type": "Agent"},
            {"id": "user1", "type": "User"},
            {"id": "user2", "type": "User"},
        ]
        info = await adapter.get_chat_info("grp-room")
        assert info["type"] == "group"

    @pytest.mark.asyncio
    async def test_returns_group_when_fetch_fails(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        # Wire link with a failing REST call
        mock_link = MagicMock()
        mock_link.rest.agent_api_participants.list_agent_chat_participants = AsyncMock(
            side_effect=RuntimeError("fetch failed")
        )
        adapter._link = mock_link
        # Band has no DMs — the fall-through type is "group", never "unknown".
        info = await adapter.get_chat_info("unknown-room")
        assert info["type"] == "group"


# ---------------------------------------------------------------------------
# 18. Hub bootstrap (_ensure_hub) + main-channel wiring
# ---------------------------------------------------------------------------

def _make_hub_link(rooms=None, created_room_id="hub-new-1"):
    """Fake link whose REST surface supports every hub-bootstrap call.

    ``rooms`` is a list of room ids returned by list_agent_chats; participant
    lookups are expected to be pre-seeded into the adapter's cache by tests.
    """
    link = MagicMock()
    link.subscribe_room = AsyncMock()
    link.rest.agent_api_chats.list_agent_chats = AsyncMock(
        return_value=SimpleNamespace(
            data=[SimpleNamespace(id=rid) for rid in (rooms or [])],
            metadata=SimpleNamespace(total_pages=1),
        )
    )
    link.rest.agent_api_chats.create_agent_chat = AsyncMock(
        return_value=SimpleNamespace(data=SimpleNamespace(id=created_room_id))
    )
    link.rest.agent_api_participants.add_agent_chat_participant = AsyncMock()
    link.rest.agent_api_messages.create_agent_chat_message = AsyncMock(
        return_value=SimpleNamespace(data=SimpleNamespace(id="greet-msg-1"))
    )
    return link


class TestHubBootstrap:

    @pytest.fixture
    def adapter(self, monkeypatch):
        monkeypatch.delenv("BAND_HUB_ROOM", raising=False)
        monkeypatch.delenv("BAND_HOME_ROOM", raising=False)
        a = _make_adapter(monkeypatch, agent_id="agent-1")
        a._agent_id = "agent-1"
        a._owner_uuid = "owner-1"
        # Record .env persistence instead of writing the operator's real file.
        saved = {}
        import hermes_cli.config as _hcfg
        monkeypatch.setattr(_hcfg, "save_env_value", lambda k, v: saved.__setitem__(k, v))
        a._test_saved_env = saved
        return a

    @pytest.mark.asyncio
    async def test_creates_hub_when_none_exists(self, adapter):
        link = _make_hub_link(rooms=[])
        adapter._link = link

        await adapter._ensure_hub()

        link.rest.agent_api_chats.create_agent_chat.assert_awaited_once()
        # Owner added to the new room.
        args, kwargs = link.rest.agent_api_participants.add_agent_chat_participant.await_args
        assert args[0] == "hub-new-1"
        assert kwargs["participant"].participant_id == "owner-1"
        # Titling greeting sent, @mentioning the owner.
        _, mkwargs = link.rest.agent_api_messages.create_agent_chat_message.await_args
        assert mkwargs["chat_id"] == "hub-new-1"
        assert mkwargs["message"].content.startswith("Hermes Agent Hub")
        assert mkwargs["message"].mentions[0].id == "owner-1"
        # Greeting echo suppressed via the sent-id backstop.
        assert "greet-msg-1" in adapter._sent_ids
        # Hub recorded + subscribed + persisted + wired as the main channel.
        assert adapter._hub_room_id == "hub-new-1"
        link.subscribe_room.assert_awaited_with("hub-new-1")
        assert adapter._test_saved_env.get("BAND_HUB_ROOM") == "hub-new-1"
        assert adapter.config.extra.get("hub_room") == "hub-new-1"
        assert adapter.config.home_channel is not None
        assert adapter.config.home_channel.chat_id == "hub-new-1"
        assert adapter.config.home_channel.name == "Hermes Hub"
        # Home is persisted (not just in-memory) so every config reader sees it.
        assert adapter._test_saved_env.get("BAND_HOME_ROOM") == "hub-new-1"

    @pytest.mark.asyncio
    async def test_never_adopts_existing_owner_room(self, adapter):
        # An existing {agent, owner} room is NOT adopted: with no pinned id the
        # adapter always creates its own dedicated hub instead.
        link = _make_hub_link(rooms=["room-pair"])
        adapter._link = link
        adapter._participants_cache["room-pair"] = [
            {"id": "agent-1", "type": "Agent", "name": "Bot", "handle": "bot"},
            {"id": "owner-1", "type": "User", "name": "Owner", "handle": "owner"},
        ]

        await adapter._ensure_hub()

        link.rest.agent_api_chats.create_agent_chat.assert_awaited_once()
        assert adapter._hub_room_id == "hub-new-1"
        assert adapter._test_saved_env.get("BAND_HUB_ROOM") == "hub-new-1"
        assert adapter.config.home_channel.chat_id == "hub-new-1"

    @pytest.mark.asyncio
    async def test_pinned_hub_skips_create(self, adapter):
        adapter._hub_room_id = "pinned-1"
        link = _make_hub_link()
        adapter._link = link

        await adapter._ensure_hub()

        assert adapter._hub_room_id == "pinned-1"
        link.rest.agent_api_chats.create_agent_chat.assert_not_called()
        # A pinned reconnect posts nothing — no scan, no greeting.
        link.rest.agent_api_chats.list_agent_chats.assert_not_called()
        link.rest.agent_api_messages.create_agent_chat_message.assert_not_called()
        link.subscribe_room.assert_awaited_with("pinned-1")
        assert adapter.config.home_channel.chat_id == "pinned-1"

    @pytest.mark.asyncio
    async def test_no_owner_disables_hub(self, adapter):
        adapter._owner_uuid = None
        link = _make_hub_link(rooms=["any"])
        adapter._link = link

        await adapter._ensure_hub()

        assert adapter._hub_room_id is None
        link.rest.agent_api_chats.list_agent_chats.assert_not_called()
        link.rest.agent_api_chats.create_agent_chat.assert_not_called()
        assert adapter.config.home_channel is None

    @pytest.mark.asyncio
    async def test_explicit_home_room_not_clobbered(self, adapter, monkeypatch):
        monkeypatch.setenv("BAND_HOME_ROOM", "elsewhere-1")
        link = _make_hub_link(rooms=[])
        adapter._link = link

        await adapter._ensure_hub()

        # Hub still bootstrapped + persisted, but the operator's main-channel
        # override is respected — and NOT overwritten with the hub id.
        assert adapter._hub_room_id == "hub-new-1"
        assert adapter._test_saved_env.get("BAND_HUB_ROOM") == "hub-new-1"
        assert adapter.config.home_channel is None
        assert "BAND_HOME_ROOM" not in adapter._test_saved_env

    @pytest.mark.asyncio
    async def test_hub_create_failure_is_non_fatal(self, adapter):
        link = _make_hub_link(rooms=[])
        link.rest.agent_api_chats.create_agent_chat = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        adapter._link = link

        await adapter._ensure_hub()  # must not raise

        assert adapter._hub_room_id is None
        assert adapter.config.home_channel is None
        assert "BAND_HUB_ROOM" not in adapter._test_saved_env


class TestPersistOwner:
    """The owner resolved from /me is persisted so it survives the process."""

    @pytest.fixture
    def adapter(self, monkeypatch):
        monkeypatch.delenv("BAND_OWNER_ID", raising=False)
        a = _make_adapter(monkeypatch)
        saved = {}
        import hermes_cli.config as _hcfg

        monkeypatch.setattr(_hcfg, "save_env_value", lambda k, v: saved.__setitem__(k, v))
        a._test_saved_env = saved
        return a

    def test_persists_owner_to_env_and_extra(self, adapter, monkeypatch):
        monkeypatch.delenv("BAND_OWNER_ID", raising=False)
        adapter._persist_owner("owner-uuid")
        assert adapter._test_saved_env.get("BAND_OWNER_ID") == "owner-uuid"
        assert adapter.config.extra.get("owner_id") == "owner-uuid"

    def test_persist_owner_idempotent_when_env_matches(self, adapter, monkeypatch):
        monkeypatch.setenv("BAND_OWNER_ID", "owner-uuid")
        adapter._persist_owner("owner-uuid")
        # Already in env → no redundant .env write (extra still mirrored).
        assert "BAND_OWNER_ID" not in adapter._test_saved_env
        assert adapter.config.extra.get("owner_id") == "owner-uuid"


class TestWireHomeChannel:
    """Direct coverage of the hub→home wiring + persistence rules."""

    @pytest.fixture
    def adapter(self, monkeypatch):
        monkeypatch.delenv("BAND_HOME_ROOM", raising=False)
        a = _make_adapter(monkeypatch)
        saved = {}
        import hermes_cli.config as _hcfg

        monkeypatch.setattr(_hcfg, "save_env_value", lambda k, v: saved.__setitem__(k, v))
        a._test_saved_env = saved
        return a

    def test_wires_and_persists_when_no_override(self, adapter, monkeypatch):
        monkeypatch.delenv("BAND_HOME_ROOM", raising=False)
        adapter._wire_home_channel("hub-1")
        assert adapter.config.home_channel.chat_id == "hub-1"
        assert adapter._test_saved_env.get("BAND_HOME_ROOM") == "hub-1"

    def test_operator_override_respected_and_not_persisted(self, adapter, monkeypatch):
        monkeypatch.setenv("BAND_HOME_ROOM", "operator-room")
        adapter._wire_home_channel("hub-1")
        # A home pointing at a non-hub room is an operator choice — untouched.
        assert adapter.config.home_channel is None
        assert "BAND_HOME_ROOM" not in adapter._test_saved_env

    def test_previous_hub_is_not_an_override(self, adapter, monkeypatch):
        # Failover: BAND_HOME_ROOM still points at the old hub (our own auto-home),
        # so re-home to the new hub rather than treating it as an override.
        monkeypatch.setenv("BAND_HOME_ROOM", "old-hub")
        adapter._wire_home_channel("new-hub", previous_hub="old-hub")
        assert adapter.config.home_channel.chat_id == "new-hub"
        assert adapter._test_saved_env.get("BAND_HOME_ROOM") == "new-hub"

    def test_operator_override_survives_failover(self, adapter, monkeypatch):
        # Operator pinned a non-hub home; a failover must not steal it.
        monkeypatch.setenv("BAND_HOME_ROOM", "operator-room")
        adapter._wire_home_channel("new-hub", previous_hub="old-hub")
        assert adapter.config.home_channel is None
        assert "BAND_HOME_ROOM" not in adapter._test_saved_env

    def test_idempotent_when_home_already_hub(self, adapter, monkeypatch):
        monkeypatch.setenv("BAND_HOME_ROOM", "hub-1")
        adapter._wire_home_channel("hub-1")
        assert adapter.config.home_channel.chat_id == "hub-1"
        # Already equal → no redundant write.
        assert "BAND_HOME_ROOM" not in adapter._test_saved_env


# ---------------------------------------------------------------------------
# 19. Owner slash-command gate
# ---------------------------------------------------------------------------

class TestIsCommandText:

    @pytest.mark.parametrize(
        "text,expected",
        [
            ("/help", True),
            ("/help me out", True),
            ("/help@bot", True),
            ("/new", True),
            ("/usr/bin/ls", False),       # path-like — stays plain chat
            ("//weird", False),
            ("/", False),
            ("hello", False),
            ("", False),
            ("say /help", False),
        ],
    )
    def test_command_shapes(self, text, expected):
        assert BandAdapter._is_command_text(text) is expected


class TestStripSelfMention:

    @pytest.fixture
    def adapter(self, monkeypatch):
        a = _make_adapter(monkeypatch, agent_id="agent-self-id")
        a._agent_id = "agent-self-id"
        a._handle = "nir/hermes-pa-agent"
        return a

    def test_strips_leading_id_mention(self, adapter):
        assert adapter._strip_self_mention("@[[agent-self-id]] /help") == "/help"

    def test_strips_leading_handle_mention(self, adapter):
        assert adapter._strip_self_mention("@[[nir/hermes-pa-agent]] /new x") == "/new x"

    def test_strips_repeated_leading_mentions(self, adapter):
        assert (
            adapter._strip_self_mention("@[[agent-self-id]] @[[agent-self-id]]  hi")
            == "hi"
        )

    def test_leaves_other_mentions(self, adapter):
        # A non-self leading mention is left in place (not our token).
        assert adapter._strip_self_mention("@[[someone-else]] hi") == "@[[someone-else]] hi"

    def test_leaves_plain_text(self, adapter):
        assert adapter._strip_self_mention("just chatting") == "just chatting"

    def test_empty(self, adapter):
        assert adapter._strip_self_mention("") == ""

    @pytest.mark.asyncio
    async def test_addressed_command_dispatches(self, monkeypatch):
        """An @mentioned slash command from the owner reaches the gateway."""
        monkeypatch.delenv("BAND_HUB_ROOM", raising=False)
        a = _make_adapter(monkeypatch, agent_id="agent-self-id")
        a._agent_id = "agent-self-id"
        a._owner_uuid = "owner-1"
        a.handle_message = AsyncMock()
        a.send = AsyncMock()
        a._participants_cache["chat-room"] = [
            {"id": "agent-self-id", "type": "Agent", "name": "Bot", "handle": "bot"},
            {"id": "owner-1", "type": "User", "name": "Owner", "handle": "owner"},
        ]
        payload = SimpleNamespace(
            id="m1",
            content="@[[agent-self-id]] /help",
            message_type="text",
            sender_id="owner-1",
            sender_type="User",
            sender_name="Owner",
            chat_room_id="chat-room",
            metadata=SimpleNamespace(mentions=[]),
        )
        await a._handle_message_created(
            SimpleNamespace(type="message_created", room_id="chat-room", payload=payload)
        )
        a.handle_message.assert_called_once()
        # The relayed text is the bare command, so the gateway dispatches it.
        assert a.handle_message.call_args.args[0].text == "/help"


class TestOwnerCommandGate:

    @pytest.fixture
    def adapter(self, monkeypatch):
        monkeypatch.delenv("BAND_HUB_ROOM", raising=False)
        a = _make_adapter(monkeypatch, agent_id="agent-self-id")
        a._agent_id = "agent-self-id"
        a._owner_uuid = "owner-1"
        a._hub_room_id = "hub-room"
        a.handle_message = AsyncMock()
        a.send = AsyncMock()
        a._participants_cache["hub-room"] = [
            {"id": "agent-self-id", "type": "Agent", "name": "Bot", "handle": "bot"},
            {"id": "owner-1", "type": "User", "name": "Owner", "handle": "owner"},
        ]
        a._participants_cache["chat-room"] = [
            {"id": "agent-self-id", "type": "Agent", "name": "Bot", "handle": "bot"},
            {"id": "human-2", "type": "User", "name": "Bob", "handle": "bob"},
        ]
        return a

    @staticmethod
    def _event(room_id, sender_id, content, msg_id="msg-gate-1", sender_type="User",
               mentioned=False):
        mentions = (
            [SimpleNamespace(id="agent-self-id", handle=None)] if mentioned else []
        )
        payload = SimpleNamespace(
            id=msg_id,
            content=content,
            message_type="text",
            sender_id=sender_id,
            sender_type=sender_type,
            sender_name="Someone",
            chat_room_id=room_id,
            metadata=SimpleNamespace(mentions=mentions),
        )
        return SimpleNamespace(type="message_created", room_id=room_id, payload=payload)

    @pytest.mark.asyncio
    async def test_owner_command_in_hub_relayed(self, adapter):
        await adapter._handle_message_created(self._event("hub-room", "owner-1", "/help"))
        adapter.handle_message.assert_called_once()
        adapter.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_owner_command_outside_hub_relayed(self, adapter):
        await adapter._handle_message_created(self._event("chat-room", "owner-1", "/help"))
        adapter.handle_message.assert_called_once()
        adapter.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_owner_command_works_without_hub(self, adapter):
        adapter._hub_room_id = None
        await adapter._handle_message_created(self._event("chat-room", "owner-1", "/help"))
        adapter.handle_message.assert_called_once()
        adapter.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_non_owner_command_dropped_with_notice(self, adapter):
        await adapter._handle_message_created(self._event("chat-room", "human-2", "/help"))
        adapter.handle_message.assert_not_called()
        adapter.send.assert_awaited_once()
        args = adapter.send.await_args.args
        assert args[0] == "chat-room"
        assert "owner" in args[1]

    @pytest.mark.asyncio
    async def test_notice_sent_only_once_per_room(self, adapter):
        await adapter._handle_message_created(self._event("chat-room", "human-2", "/help"))
        await adapter._handle_message_created(
            self._event("chat-room", "human-2", "/new", msg_id="msg-gate-2")
        )
        adapter.handle_message.assert_not_called()
        assert adapter.send.await_count == 1

    @pytest.mark.asyncio
    async def test_non_owner_command_in_hub_dropped(self, adapter):
        await adapter._handle_message_created(self._event("hub-room", "intruder-9", "/new"))
        adapter.handle_message.assert_not_called()
        adapter.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_agent_sender_command_dropped_silently(self, adapter):
        await adapter._handle_message_created(
            self._event("chat-room", "other-agent-7", "/help", sender_type="Agent")
        )
        adapter.handle_message.assert_not_called()
        adapter.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_path_like_text_relayed_as_chat(self, adapter):
        # Path-like text isn't a command, so it's never dropped by the command
        # gate. @mentioned here so it also clears the normal mention gate and
        # reaches the agent as plain chat.
        await adapter._handle_message_created(
            self._event("chat-room", "human-2", "/usr/bin/ls is missing", mentioned=True)
        )
        adapter.handle_message.assert_called_once()
        adapter.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_fail_closed_without_owner(self, adapter):
        adapter._owner_uuid = None
        await adapter._handle_message_created(self._event("hub-room", "owner-1", "/help"))
        adapter.handle_message.assert_not_called()
        adapter.send.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_plain_chat_in_hub_requires_mention(self, adapter):
        # The hub is no longer a gating exception: un-mentioned plain chat is
        # ignored there like in any other room (the owner must @mention to talk).
        await adapter._handle_message_created(self._event("hub-room", "owner-1", "hello"))
        adapter.handle_message.assert_not_called()
        adapter.send.assert_not_called()

    @pytest.mark.asyncio
    async def test_mentioned_chat_in_hub_relayed(self, adapter):
        # With an @mention, plain chat in the hub reaches the agent.
        await adapter._handle_message_created(
            self._event("hub-room", "owner-1", "hello", mentioned=True)
        )
        adapter.handle_message.assert_called_once()
        adapter.send.assert_not_called()


# ---------------------------------------------------------------------------
# 20. _env_enablement — hub / home-channel seeding
# ---------------------------------------------------------------------------

class TestEnvEnablementHubSeeds:

    @pytest.fixture(autouse=True)
    def _creds(self, monkeypatch):
        monkeypatch.setenv("BAND_AGENT_ID", "test-agent")
        monkeypatch.setenv("BAND_API_KEY", "test-key")
        monkeypatch.delenv("BAND_BASE_URL", raising=False)
        monkeypatch.delenv("BAND_OWNER_ID", raising=False)

    def test_seeds_hub_room_and_home_channel(self, monkeypatch):
        monkeypatch.setenv("BAND_HUB_ROOM", "hub-1")
        monkeypatch.delenv("BAND_HOME_ROOM", raising=False)
        result = _env_enablement()
        assert result["hub_room"] == "hub-1"
        assert result["home_channel"] == {"chat_id": "hub-1", "name": "Hermes Hub"}

    def test_home_room_overrides_hub_seed(self, monkeypatch):
        monkeypatch.setenv("BAND_HUB_ROOM", "hub-1")
        monkeypatch.setenv("BAND_HOME_ROOM", "home-2")
        result = _env_enablement()
        assert result["hub_room"] == "hub-1"
        assert result["home_channel"] == {"chat_id": "home-2", "name": "Band Home"}

    def test_no_hub_no_home_channel_seed(self, monkeypatch):
        monkeypatch.delenv("BAND_HUB_ROOM", raising=False)
        monkeypatch.delenv("BAND_HOME_ROOM", raising=False)
        result = _env_enablement()
        assert "hub_room" not in result
        assert "home_channel" not in result

    def test_register_passes_cron_deliver_env_var(self):
        ctx = MagicMock()
        register(ctx)
        kwargs = ctx.register_platform.call_args[1]
        assert kwargs["cron_deliver_env_var"] == "BAND_HOME_ROOM"


# ---------------------------------------------------------------------------
# 21. Hub installation announcement (first designation must be owner-visible)
# ---------------------------------------------------------------------------

class TestHubAnnouncement:

    @pytest.fixture
    def adapter(self, monkeypatch):
        monkeypatch.delenv("BAND_HUB_ROOM", raising=False)
        monkeypatch.delenv("BAND_HOME_ROOM", raising=False)
        a = _make_adapter(monkeypatch, agent_id="agent-1")
        a._agent_id = "agent-1"
        a._owner_uuid = "owner-1"
        saved = {}
        import hermes_cli.config as _hcfg
        monkeypatch.setattr(_hcfg, "save_env_value", lambda k, v: saved.__setitem__(k, v))
        return a

    @pytest.mark.asyncio
    async def test_pinned_hub_posts_nothing(self, adapter):
        adapter._hub_room_id = "pinned-1"
        link = _make_hub_link()
        adapter._link = link

        await adapter._ensure_hub()

        link.rest.agent_api_messages.create_agent_chat_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_created_hub_posts_only_titling_greeting(self, adapter):
        link = _make_hub_link(rooms=[])
        adapter._link = link

        await adapter._ensure_hub()

        # Exactly one message — the titling greeting. Adoption (and its separate
        # announcement) no longer exists; a created hub self-announces here.
        link.rest.agent_api_messages.create_agent_chat_message.assert_awaited_once()
        _, mkwargs = link.rest.agent_api_messages.create_agent_chat_message.await_args
        assert mkwargs["message"].content.startswith("Hermes Agent Hub")

    @pytest.mark.asyncio
    async def test_greeting_post_failure_is_non_fatal(self, adapter):
        # Room creation succeeds but the titling greeting post fails: the whole
        # create is treated as failed (returns None), so no hub is wired — and
        # _ensure_hub must not raise.
        link = _make_hub_link(rooms=[])
        link.rest.agent_api_messages.create_agent_chat_message = AsyncMock(
            side_effect=RuntimeError("post failed")
        )
        adapter._link = link

        await adapter._ensure_hub()  # must not raise

        assert adapter._hub_room_id is None
        assert adapter.config.home_channel is None


# ---------------------------------------------------------------------------
# 22. Hub greeting builder (clean title + owner-facing body)
# ---------------------------------------------------------------------------

class TestHubGreeting:

    @pytest.fixture
    def adapter(self, monkeypatch):
        a = _make_adapter(monkeypatch, agent_id="agent-1")
        a._agent_id = "agent-1"
        a._owner_uuid = "owner-1"
        a._handle = "nir/hermes-pa-agent"
        return a

    def test_greeting_titles_room_with_hub_name(self, adapter):
        greeting = adapter._build_hub_greeting()
        # First line becomes the room title (server derives it from message 1).
        assert greeting.startswith("Hermes Agent Hub\n")
        assert "this is your Hermes Agent, nir/hermes-pa-agent" in greeting
        assert "set as the Band main channel" in greeting

    def test_greeting_uses_owner_label_from_participant_cache(self, adapter):
        adapter._participants_cache["some-room"] = [
            {"id": "agent-1", "type": "Agent", "name": "Bot", "handle": "bot"},
            {"id": "owner-1", "type": "User", "name": "Nir", "handle": "nir"},
        ]
        assert "Hi @Nir," in adapter._build_hub_greeting()

    def test_greeting_falls_back_to_there_when_owner_unknown(self, adapter):
        # Owner not present in any cache -> readable fallback, no dangling "@".
        assert "Hi there," in adapter._build_hub_greeting()
        assert "@" not in adapter._build_hub_greeting().split("\n\n", 1)[1].split(",")[0]

    def test_greeting_uses_default_agent_name_when_handle_missing(self, adapter):
        adapter._handle = ""
        assert "this is your Hermes Agent, Hermes." in adapter._build_hub_greeting()

    def test_greeting_body_has_no_title_line(self, adapter):
        body = adapter._hub_greeting_body()
        assert not body.startswith("Hermes Agent Hub\n")
        assert "This chat is the 'Hermes Agent Hub'" in body


# ---------------------------------------------------------------------------
# 23. Hub failover — repeated hub send failures create + re-wire a fresh hub
# ---------------------------------------------------------------------------

class TestHubFailover:

    @pytest.fixture
    def adapter(self, monkeypatch):
        monkeypatch.delenv("BAND_HUB_ROOM", raising=False)
        monkeypatch.delenv("BAND_HOME_ROOM", raising=False)
        a = _make_adapter(monkeypatch, agent_id="agent-1")
        a._agent_id = "agent-1"
        a._owner_uuid = "owner-1"
        a._hub_room_id = "old-hub"
        a._hub_failover_threshold = 3
        a._hub_failover_max_per_connect = 5
        # Last human sender so send()'s mention build always succeeds.
        # Out-of-turn delivery addresses the owner (the hub pattern), so the
        # owner is what has to be resolvable — not a "last sender".
        a._owner_uuid = "owner-uuid"
        a._participants_cache["old-hub"] = [
            {"id": "owner-uuid", "type": "User", "name": "Owner", "handle": "owner"},
        ]
        # Record .env persistence instead of writing the operator's real file.
        saved = {}
        import hermes_cli.config as _hcfg
        monkeypatch.setattr(_hcfg, "save_env_value", lambda k, v: saved.__setitem__(k, v))
        a._test_saved_env = saved
        return a

    @staticmethod
    def _hub_link(created_ids=("new-hub-1",)):
        """Link where regular sends FAIL but a new room's titling greeting succeeds.

        The hub send in ``send()`` and the greeting in ``_create_hub_room`` both
        call ``create_agent_chat_message``; they're distinguished by content —
        the greeting leads with the hub title, so it always succeeds (the new
        room accepts it), while ordinary sends raise (the broken/full hub).
        ``create_agent_chat`` hands out ``created_ids`` in order.
        """
        link = MagicMock()
        link.subscribe_room = AsyncMock()
        link.rest.agent_api_participants.add_agent_chat_participant = AsyncMock()
        # Any room resolves the owner as a mentionable participant, so a send to
        # a freshly created hub still reaches the API (and fails) rather than
        # short-circuiting on "no mentionable recipient".
        link.rest.agent_api_participants.list_agent_chat_participants = AsyncMock(
            return_value=SimpleNamespace(
                data=[SimpleNamespace(id="owner-1", name="Nir", handle="nir", type="User")]
            )
        )
        ids = iter(created_ids)
        link.rest.agent_api_chats.create_agent_chat = AsyncMock(
            side_effect=lambda *a, **k: SimpleNamespace(
                data=SimpleNamespace(id=next(ids))
            )
        )

        async def _send_msg(*args, chat_id=None, message=None, **kwargs):
            content = getattr(message, "content", "") or ""
            if content.startswith("Hermes Agent Hub"):
                return SimpleNamespace(data=SimpleNamespace(id="greet-msg"))
            raise RuntimeError("room is full")

        link.rest.agent_api_messages.create_agent_chat_message = AsyncMock(
            side_effect=_send_msg
        )
        return link

    @pytest.mark.asyncio
    async def test_failure_counter_increments_then_resets_on_success(self, adapter):
        link = MagicMock()
        # Send fails once, then succeeds.
        link.rest.agent_api_messages.create_agent_chat_message = AsyncMock(
            side_effect=[
                RuntimeError("boom"),
                SimpleNamespace(data=SimpleNamespace(id="ok-1")),
            ]
        )
        adapter._link = link

        r1 = await adapter.send("old-hub", "hi")
        assert r1.success is False
        assert adapter._hub_send_failures == 1

        r2 = await adapter.send("old-hub", "hi again")
        assert r2.success is True
        assert adapter._hub_send_failures == 0

    @pytest.mark.asyncio
    async def test_non_hub_failures_do_not_count(self, adapter):
        link = MagicMock()
        link.rest.agent_api_messages.create_agent_chat_message = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        adapter._link = link
        # Out-of-turn delivery addresses the owner (the hub pattern), so the
        # owner is what has to be resolvable — not a "last sender".
        adapter._owner_uuid = "owner-uuid"
        adapter._participants_cache["other-room"] = [
            {"id": "owner-uuid", "type": "User", "name": "Owner", "handle": "owner"},
        ]

        await adapter.send("other-room", "hi")
        assert adapter._hub_send_failures == 0

    @pytest.mark.asyncio
    async def test_failover_after_threshold_rewires_fresh_hub(self, adapter):
        link = self._hub_link(created_ids=["new-hub-1"])
        adapter._link = link

        # Three consecutive failed hub sends -> failover on the third.
        for _ in range(3):
            res = await adapter.send("old-hub", "deliver me")
            assert res.success is False

        # Fresh room created (never adopted/scanned) + wired as new main channel.
        link.rest.agent_api_chats.create_agent_chat.assert_awaited_once()
        link.rest.agent_api_chats.list_agent_chats.assert_not_called()
        assert adapter._hub_room_id == "new-hub-1"
        link.subscribe_room.assert_any_await("new-hub-1")
        assert adapter._test_saved_env.get("BAND_HUB_ROOM") == "new-hub-1"
        assert adapter.config.home_channel is not None
        assert adapter.config.home_channel.chat_id == "new-hub-1"
        # Home re-persisted to follow the new hub.
        assert adapter._test_saved_env.get("BAND_HOME_ROOM") == "new-hub-1"
        # Counter reset; one failover counted.
        assert adapter._hub_send_failures == 0
        assert adapter._hub_failovers_done == 1

    @pytest.mark.asyncio
    async def test_no_failover_before_threshold(self, adapter):
        link = self._hub_link()
        adapter._link = link

        for _ in range(2):
            await adapter.send("old-hub", "x")

        link.rest.agent_api_chats.create_agent_chat.assert_not_called()
        assert adapter._hub_room_id == "old-hub"
        assert adapter._hub_send_failures == 2

    @pytest.mark.asyncio
    async def test_no_failover_without_owner(self, adapter):
        adapter._owner_uuid = None
        link = self._hub_link()
        adapter._link = link

        for _ in range(4):
            await adapter.send("old-hub", "x")

        link.rest.agent_api_chats.create_agent_chat.assert_not_called()
        assert adapter._hub_room_id == "old-hub"

    @pytest.mark.asyncio
    async def test_failed_create_keeps_old_hub_and_resets_counter(self, adapter):
        link = self._hub_link()
        # create_agent_chat returns no id -> _create_hub_room returns None.
        link.rest.agent_api_chats.create_agent_chat = AsyncMock(
            return_value=SimpleNamespace(data=None)
        )
        adapter._link = link

        for _ in range(3):
            await adapter.send("old-hub", "x")

        # Hub unchanged, no room counted, counter re-armed (reset to 0).
        assert adapter._hub_room_id == "old-hub"
        assert adapter._hub_failovers_done == 0
        assert adapter._hub_send_failures == 0

    @pytest.mark.asyncio
    async def test_per_connect_cap_halts_runaway_failovers(self, adapter):
        adapter._hub_failover_threshold = 1
        adapter._hub_failover_max_per_connect = 2
        # Each created room accepts its greeting but then rejects ordinary
        # sends, so every new hub keeps failing and re-triggering failover.
        link = self._hub_link(created_ids=["hub-a", "hub-b", "hub-c", "hub-d"])
        adapter._link = link

        # Many failing sends, but successful failovers stop at the cap.
        for _ in range(10):
            # Target whatever the current hub is so each failure counts.
            await adapter.send(adapter._hub_room_id, "x")

        assert adapter._hub_failovers_done == 2
        assert adapter._hub_room_id == "hub-b"

    @pytest.mark.asyncio
    async def test_reentrancy_guard_blocks_nested_failover(self, adapter):
        # With a failover already in progress, a fresh trigger is a no-op.
        adapter._failover_in_progress = True
        adapter._hub_send_failures = adapter._hub_failover_threshold
        link = self._hub_link()
        adapter._link = link

        await adapter._record_hub_send("old-hub", ok=False)

        link.rest.agent_api_chats.create_agent_chat.assert_not_called()


# ---------------------------------------------------------------------------
# 24. Renderer capability flags (what the host may assume Band can render)
# ---------------------------------------------------------------------------

class TestRendererCapabilities:
    """The host reads capability flags off the adapter to pick a rendering.

    Every flag the adapter does not declare falls back to the conservative
    plain-text default on ``BasePlatformAdapter``, so silence here is not
    neutral — it downgrades output. These tests pin the two flags Band claims
    and the host-side names they hook into, so an upstream rename turns our
    declaration into dead code *here* instead of quietly in production.
    """

    def test_declares_supports_code_blocks(self, monkeypatch):
        # Band clients render markdown fences; gateway/run.py's tool-progress
        # renderer gates the fenced-code-block rendering on this flag.
        adapter = _make_adapter(monkeypatch)
        assert adapter.supports_code_blocks is True

    def test_declares_splits_long_messages(self, monkeypatch):
        # send() chunks via truncate_message(), so the host must not pre-trim.
        adapter = _make_adapter(monkeypatch)
        assert adapter.splits_long_messages is True

    def test_flags_are_real_host_capabilities_and_default_off(self):
        """Both flags exist on the host base and default False there.

        If either name disappears or flips its default upstream, the adapter's
        declaration stops meaning what its comment says — catch that here.
        """
        base = pytest.importorskip("gateway.platforms.base")
        assert base.BasePlatformAdapter.supports_code_blocks is False
        assert base.BasePlatformAdapter.splits_long_messages is False

    @pytest.mark.asyncio
    async def test_host_delivery_router_sends_full_payload_unchunked_by_gateway(
        self, monkeypatch, tmp_path
    ):
        """End-to-end: the real cron delivery router hands Band the whole payload.

        Drives ``gateway.delivery.DeliveryRouter._deliver_to_platform`` — the
        only host consumer of ``splits_long_messages``. With the flag set, an
        oversized cron output reaches ``send()`` intact and Band splits it into
        several messages; without it the host truncates at
        ``MAX_PLATFORM_OUTPUT`` and appends a "full output saved to <path>"
        footer pointing at a file on the gateway host that a Band reader cannot
        open. Skips if the host module isn't importable.
        """
        delivery = pytest.importorskip("gateway.delivery")

        adapter = _make_adapter(monkeypatch)
        # Out-of-turn delivery addresses the owner (the hub pattern), so the
        # owner is what has to be resolvable — not a "last sender".
        adapter._owner_uuid = "owner-uuid"
        adapter._participants_cache["room-cron"] = [
            {"id": "owner-uuid", "type": "User", "name": "Owner", "handle": "owner"},
        ]
        posted: list = []

        async def _create(*args, **kwargs):
            posted.append(kwargs["message"].content)
            return SimpleNamespace(data=SimpleNamespace(id=f"cron-{len(posted)}"))

        mock_link = MagicMock()
        mock_link.rest.agent_api_messages.create_agent_chat_message = _create
        adapter._link = mock_link

        router = delivery.DeliveryRouter(
            SimpleNamespace(platforms={}), {adapter.platform: adapter}
        )
        # The router audit-saves oversized output to the real hermes home; keep
        # the test off the filesystem while leaving the branch under test intact.
        router._save_full_output = lambda *_a, **_k: tmp_path / "full-output.txt"
        target = delivery.DeliveryTarget(
            platform=adapter.platform, chat_id="room-cron", is_explicit=True,
        )

        long_output = "band-cron-line\n" * 800  # ~12000 chars, well over the cap
        assert len(long_output) > delivery.MAX_PLATFORM_OUTPUT

        result = await router._deliver_to_platform(target, long_output, {"job_id": "job-1"})

        assert result.success is True
        # Chunked by the adapter, not truncated by the host: every line survived
        # across the chunks and no footer was substituted for the tail.
        assert len(posted) > 1
        assert sum(chunk.count("band-cron-line") for chunk in posted) == 800
        assert not any("full output saved to" in chunk for chunk in posted)

        # Negative control: clear the flag and the same host path truncates.
        posted.clear()
        adapter.splits_long_messages = False
        await router._deliver_to_platform(target, long_output, {"job_id": "job-1"})
        assert len(posted) == 1
        assert "full output saved to" in posted[0]


# ---------------------------------------------------------------------------
# 25. Shared send primitives — _fetch_participants / _post_chunks
# ---------------------------------------------------------------------------

_fetch_participants = _band_mod._fetch_participants
_post_chunks = _band_mod._post_chunks


def _rest_stub(participants=()):
    """Fake REST client exposing only what the shared send primitives drive.

    Mirrors the fake links used elsewhere in this file: the participant listing
    and the message create, both AsyncMocks so individual tests can override
    ``side_effect``. Posted messages get sequential ``std-msg-<n>`` ids so a
    chunked send is traceable back to its last chunk.
    """
    rest = MagicMock()
    rest.agent_api_participants.list_agent_chat_participants = AsyncMock(
        return_value=SimpleNamespace(data=list(participants))
    )
    posted = 0

    async def _create(*args, **kwargs):
        nonlocal posted
        posted += 1
        return SimpleNamespace(data=SimpleNamespace(id=f"std-msg-{posted}"))

    rest.agent_api_messages.create_agent_chat_message = AsyncMock(side_effect=_create)
    return rest


def _participant(pid, name=None, handle=None, ptype="User"):
    return SimpleNamespace(id=pid, name=name, handle=handle, type=ptype)


def _mention_tuples(create_mock):
    """``[(id, handle, name), ...]`` per posted chunk, in call order."""
    return [
        [
            (getattr(m, "id", None), getattr(m, "handle", None), getattr(m, "name", None))
            for m in call.kwargs["message"].mentions
        ]
        for call in create_mock.await_args_list
    ]


class TestFetchParticipants:

    @pytest.mark.asyncio
    async def test_normalizes_sdk_objects_to_mention_item_shape(self):
        rest = _rest_stub(
            [
                _participant("agent-1", "Bot", "bot", ptype="Agent"),
                _participant("human-1", "Alice", "alice"),
            ]
        )

        participants = await _fetch_participants(rest, "room-1")

        assert participants == [
            {"id": "agent-1", "name": "Bot", "handle": "bot", "type": "Agent"},
            {"id": "human-1", "name": "Alice", "handle": "alice", "type": "User"},
        ]
        # The room is passed as chat_id, with the retry-enabled request options.
        kwargs = rest.agent_api_participants.list_agent_chat_participants.await_args.kwargs
        assert kwargs["chat_id"] == "room-1"
        assert kwargs["request_options"] is _band_mod.DEFAULT_REQUEST_OPTIONS

    @pytest.mark.asyncio
    async def test_missing_attributes_become_none(self):
        rest = _rest_stub([SimpleNamespace(id="bare-1")])

        assert await _fetch_participants(rest, "room-1") == [
            {"id": "bare-1", "name": None, "handle": None, "type": None}
        ]

    @pytest.mark.asyncio
    async def test_absent_or_empty_data_yields_empty_list(self):
        for payload in (SimpleNamespace(data=None), SimpleNamespace(data=[]), SimpleNamespace()):
            rest = MagicMock()
            rest.agent_api_participants.list_agent_chat_participants = AsyncMock(
                return_value=payload
            )
            assert await _fetch_participants(rest, "room-1") == []

    @pytest.mark.asyncio
    async def test_errors_propagate_to_the_caller(self):
        rest = MagicMock()
        rest.agent_api_participants.list_agent_chat_participants = AsyncMock(
            side_effect=RuntimeError("participants 503")
        )

        with pytest.raises(RuntimeError, match="participants 503"):
            await _fetch_participants(rest, "room-1")

    @pytest.mark.asyncio
    async def test_adapter_cache_swallows_the_error_and_caches_empty(self, monkeypatch):
        """_get_participants still owns the swallow-and-warn policy."""
        adapter = _make_adapter(monkeypatch)
        link = MagicMock()
        link.rest.agent_api_participants.list_agent_chat_participants = AsyncMock(
            side_effect=RuntimeError("boom")
        )
        adapter._link = link

        assert await adapter._get_participants("room-1") == []
        assert adapter._participants_cache["room-1"] == []


class TestPostChunks:

    @staticmethod
    def _mentions(*ids):
        return [
            _band_mod.ChatMessageRequestMentionsItem(id=i, handle=None, name=None)
            for i in ids
        ]

    @pytest.mark.asyncio
    async def test_short_content_posts_once_with_no_continuation(self):
        rest = _rest_stub()
        mentions = self._mentions("human-1")

        last_id, continuation, last_resp = await _post_chunks(
            rest, "room-1", "hello", mentions, 4000
        )

        assert (last_id, continuation) == ("std-msg-1", [])
        assert last_resp.data.id == "std-msg-1"
        create = rest.agent_api_messages.create_agent_chat_message
        create.assert_awaited_once()
        assert create.await_args.kwargs["chat_id"] == "room-1"
        assert create.await_args.kwargs["message"].content == "hello"
        assert create.await_args.kwargs["message"].mentions is mentions

    @pytest.mark.asyncio
    async def test_long_content_is_split_exactly_as_truncate_message(self):
        rest = _rest_stub()
        content = "x" * 9000
        expected = BandAdapter.truncate_message(content, 4000)
        assert len(expected) >= 2  # guard: the fixture must actually chunk

        last_id, continuation, _ = await _post_chunks(
            rest, "room-1", content, self._mentions("human-1"), 4000
        )

        create = rest.agent_api_messages.create_agent_chat_message
        assert [c.kwargs["message"].content for c in create.await_args_list] == expected
        # Last id is returned; every earlier id is a continuation.
        assert last_id == f"std-msg-{len(expected)}"
        assert continuation == [f"std-msg-{n}" for n in range(1, len(expected))]

    @pytest.mark.asyncio
    async def test_mentions_are_repeated_on_every_chunk(self):
        rest = _rest_stub()
        content = "y" * 9000
        expected = BandAdapter.truncate_message(content, 4000)

        await _post_chunks(
            rest, "room-1", content, self._mentions("human-1", "human-2"), 4000
        )

        # Band mandates >=1 mention per message, so continuations keep them.
        create = rest.agent_api_messages.create_agent_chat_message
        assert _mention_tuples(create) == [
            [("human-1", None, None), ("human-2", None, None)]
        ] * len(expected)

    @pytest.mark.asyncio
    async def test_on_sent_receives_every_returned_id(self):
        rest = _rest_stub()
        seen: list = []
        content = "z" * 9000
        expected = BandAdapter.truncate_message(content, 4000)

        await _post_chunks(
            rest,
            "room-1",
            content,
            self._mentions("human-1"),
            4000,
            on_sent=seen.append,
        )

        assert seen == [f"std-msg-{n}" for n in range(1, len(expected) + 1)]

    @pytest.mark.asyncio
    async def test_id_less_responses_do_not_advance_the_cursor(self):
        rest = _rest_stub()
        rest.agent_api_messages.create_agent_chat_message = AsyncMock(
            return_value=SimpleNamespace(data=None)
        )
        seen: list = []

        last_id, continuation, last_resp = await _post_chunks(
            rest, "room-1", "hello", self._mentions("human-1"), 4000, on_sent=seen.append
        )

        assert (last_id, continuation, seen) == (None, [], [])
        assert last_resp is not None  # the response is still handed back

    @pytest.mark.asyncio
    async def test_errors_propagate_so_the_caller_owns_the_failure_shape(self):
        rest = _rest_stub()
        rest.agent_api_messages.create_agent_chat_message = AsyncMock(
            side_effect=RuntimeError("network failure")
        )

        with pytest.raises(RuntimeError, match="network failure"):
            await _post_chunks(rest, "room-1", "hello", self._mentions("human-1"), 4000)

    @pytest.mark.asyncio
    async def test_send_on_link_still_records_its_own_sent_ids(self, monkeypatch):
        """The live path's echo backstop rides on the on_sent hook."""
        adapter = _make_adapter(monkeypatch)
        link = MagicMock()
        link.rest = _rest_stub()
        adapter._link = link
        # Out-of-turn delivery addresses the owner (the hub pattern), so the
        # owner is what has to be resolvable — not a "last sender".
        adapter._owner_uuid = "owner-uuid"
        adapter._participants_cache["room-1"] = [
            {"id": "owner-uuid", "type": "User", "name": "Owner", "handle": "owner"},
        ]

        result = await adapter._send_on_link("room-1", "x" * 9000)

        assert result.success is True
        assert result.message_id in adapter._sent_ids
        # Continuation ids are surfaced on the SendResult and also recorded.
        assert result.continuation_message_ids
        assert set(result.continuation_message_ids) <= adapter._sent_ids


# ---------------------------------------------------------------------------
# 25. Standalone (out-of-process) sender — deliver=band with no live gateway
#
# Drives the shared primitives from section 24 with no adapter instance, so it
# reuses that section's _rest_stub / _participant / _mention_tuples helpers.
# ---------------------------------------------------------------------------

_standalone_send = _band_mod._standalone_send
_standalone_room = _band_mod._standalone_room


class TestStandaloneSenderRegistration:
    """The hook must be reachable exactly the way _send_via_adapter reaches it."""

    def test_register_passes_standalone_sender_fn(self):
        ctx = MagicMock()
        register(ctx)
        kwargs = ctx.register_platform.call_args[1]
        assert kwargs["standalone_sender_fn"] is _standalone_send

    def test_platform_entry_accepts_standalone_sender_fn(self):
        """PlatformEntry really carries the field (no version guard needed)."""
        from gateway.platform_registry import PlatformEntry

        ctx = MagicMock()
        register(ctx)
        entry = PlatformEntry(**ctx.register_platform.call_args[1])
        assert entry.standalone_sender_fn is _standalone_send
        # The home-channel var the sender falls back to is the one cron reads.
        assert entry.cron_deliver_env_var == "BAND_HOME_ROOM"


class TestStandaloneSend:

    @pytest.fixture
    def env(self, monkeypatch):
        """Minimal out-of-process environment: credentials + owner, no home room.

        The owner is part of the minimum now. Out-of-process delivery addresses
        the owner — the hub pattern — so ``BAND_OWNER_ID`` is what makes a cron
        send possible at all. It is persisted to the Hermes env on first connect
        precisely so a process with no adapter in memory can read it.
        """
        monkeypatch.setenv("BAND_AGENT_ID", "agent-self")
        monkeypatch.setenv("BAND_API_KEY", "secret-key")
        monkeypatch.setenv("BAND_OWNER_ID", "owner-uuid")
        for var in ("BAND_BASE_URL", "BAND_HOME_ROOM", "BAND_HUB_ROOM"):
            monkeypatch.delenv(var, raising=False)
        return monkeypatch

    @staticmethod
    def _patch_rest(monkeypatch, rest):
        """Route the send at ``rest`` and expose its owned HTTP transport."""
        seen = {}

        def _factory(_api_key, _base_url, httpx_client):
            seen["httpx_client"] = httpx_client
            return rest

        monkeypatch.setattr(_band_mod, "_standalone_rest", _factory)
        return seen

    # ── happy path + room resolution ──────────────────────────────────────

    @pytest.mark.asyncio
    async def test_posts_to_explicit_chat_id(self, env):
        rest = _rest_stub([
            _participant("owner-uuid", "Owner", "owner"),
            _participant("human-1", "Alice", "alice"),
        ])
        seen = self._patch_rest(env, rest)

        result = await _standalone_send(_make_config(), "room-explicit", "cron output")

        assert result == {
            "success": True,
            "platform": "band",
            "chat_id": "room-explicit",
            "message_id": "std-msg-1",
        }
        create = rest.agent_api_messages.create_agent_chat_message
        create.assert_awaited_once()
        assert create.await_args.kwargs["chat_id"] == "room-explicit"
        assert create.await_args.kwargs["message"].content == "cron output"
        # Out-of-turn delivery addresses the OWNER, not everyone in the room.
        assert _mention_tuples(create) == [[("owner-uuid", "owner", "Owner")]]
        assert seen["httpx_client"].is_closed

    @pytest.mark.asyncio
    async def test_falls_back_to_home_room_when_no_chat_id(self, env):
        env.setenv("BAND_HOME_ROOM", "room-home")
        rest = _rest_stub([
            _participant("owner-uuid", "Owner", "owner"),
            _participant("human-1", "Alice", "alice"),
        ])
        self._patch_rest(env, rest)

        result = await _standalone_send(_make_config(), "", "hello")

        assert result["success"] is True
        assert result["chat_id"] == "room-home"
        assert (
            rest.agent_api_participants.list_agent_chat_participants.await_args.kwargs[
                "chat_id"
            ]
            == "room-home"
        )

    def test_room_resolution_precedence(self, env):
        """Explicit id > BAND_HOME_ROOM > BAND_HUB_ROOM > config extra."""
        extra = {"home_channel": {"chat_id": "room-extra-home"}, "hub_room": "room-extra-hub"}
        env.setenv("BAND_HOME_ROOM", "room-home")
        env.setenv("BAND_HUB_ROOM", "room-hub")
        assert _standalone_room("room-explicit", extra) == "room-explicit"
        assert _standalone_room(None, extra) == "room-home"

        env.delenv("BAND_HOME_ROOM")
        assert _standalone_room(None, extra) == "room-hub"

        env.delenv("BAND_HUB_ROOM")
        assert _standalone_room(None, extra) == "room-extra-home"
        assert _standalone_room(None, {"hub_room": "room-extra-hub"}) == "room-extra-hub"
        assert _standalone_room(None, {}) is None
        # Whitespace-only is not a room.
        assert _standalone_room("   ", {}) is None

    @pytest.mark.asyncio
    async def test_credentials_and_host_read_from_config_extra_when_env_absent(self, env):
        """PlatformConfig.extra is the secondary source, as in BandAdapter.__init__."""
        env.delenv("BAND_AGENT_ID")
        env.delenv("BAND_API_KEY")
        rest = _rest_stub([
            _participant("owner-uuid", "Owner", "owner"),
            _participant("human-1", "Alice", "alice"),
        ])
        seen: dict = {}

        def _factory(api_key, base_url, httpx_client):
            seen["api_key"] = api_key
            seen["base_url"] = base_url
            seen["httpx_client"] = httpx_client
            return rest

        env.setattr(_band_mod, "_standalone_rest", _factory)
        cfg = _make_config(
            {"agent_id": "agent-extra", "api_key": "key-extra", "base_url": "band.internal"}
        )

        result = await _standalone_send(cfg, "room-1", "hi")

        assert result["success"] is True
        assert seen["api_key"] == "key-extra"
        assert seen["base_url"] == "band.internal"
        assert seen["httpx_client"].is_closed

    # ── parity with the live _send_on_link path ───────────────────────────

    @pytest.mark.asyncio
    async def test_agent_itself_is_never_mentioned(self, env):
        rest = _rest_stub(
            [
                _participant("agent-self", "Bot", "bot", ptype="Agent"),
                _participant("owner-uuid", "Owner", "owner"),
                _participant("human-1", "Alice", "alice"),
            ]
        )
        self._patch_rest(env, rest)

        result = await _standalone_send(_make_config(), "room-1", "hi")

        assert result["success"] is True
        # The owner, and only the owner: not the agent itself and not bystanders.
        assert _mention_tuples(rest.agent_api_messages.create_agent_chat_message) == [
            [("owner-uuid", "owner", "Owner")]
        ]

    @pytest.mark.asyncio
    async def test_chunks_long_message_and_repeats_mentions_on_every_chunk(self, env):
        rest = _rest_stub([
            _participant("owner-uuid", "Owner", "owner"),
            _participant("human-1", "Alice", "alice"),
        ])
        self._patch_rest(env, rest)
        long_content = "x" * 9000
        expected_chunks = BandAdapter.truncate_message(
            long_content, BandAdapter.MAX_MESSAGE_LENGTH
        )
        assert len(expected_chunks) >= 2  # guard: the fixture must actually chunk

        result = await _standalone_send(_make_config(), "room-1", long_content)

        create = rest.agent_api_messages.create_agent_chat_message
        assert [c.kwargs["message"].content for c in create.await_args_list] == expected_chunks
        # Band mandates >=1 mention per message, so continuations keep them.
        # Out-of-turn delivery addresses the OWNER, not everyone in the room.
        assert _mention_tuples(create) == [[("owner-uuid", "owner", "Owner")]] * len(
            expected_chunks
        )
        # message_id is the LAST chunk's id, as in _send_on_link.
        assert result["message_id"] == f"std-msg-{len(expected_chunks)}"

    @pytest.mark.asyncio
    async def test_matches_the_live_out_of_turn_path(self, env):
        """Parity with the live adapter, restated for the current design.

        The old parity was against a recipient list both paths computed. Neither
        computes one now: out-of-turn delivery addresses the owner, in-process or
        not, and that is the invariant worth holding — a cron job must reach the
        same person whether or not a gateway happens to be running.
        """
        participants = [
            _participant("agent-self", "Bot", "bot", ptype="Agent"),
            _participant("owner-uuid", "Owner", "owner"),
            _participant("human-2", "Bob", "bob"),
        ]

        standalone_rest = _rest_stub(participants)
        self._patch_rest(env, standalone_rest)
        await _standalone_send(_make_config(), "room-parity", "same text")

        adapter = _make_adapter(env, agent_id="agent-self")
        adapter._agent_id = "agent-self"
        adapter._owner_uuid = "owner-uuid"
        adapter._participants_cache["room-parity"] = [
            {"id": p.id, "type": p.type, "name": p.name, "handle": p.handle}
            for p in participants
        ]
        live_link = MagicMock()
        live_link.rest = _rest_stub(participants)
        adapter._link = live_link
        await adapter.send("room-parity", "same text")

        standalone_mentions = _mention_tuples(
            standalone_rest.agent_api_messages.create_agent_chat_message
        )
        live_mentions = _mention_tuples(
            live_link.rest.agent_api_messages.create_agent_chat_message
        )
        assert standalone_mentions == live_mentions == [[("owner-uuid", "owner", "Owner")]]

    @pytest.mark.asyncio
    async def test_missing_api_key_names_the_variable(self, env):
        env.delenv("BAND_API_KEY")
        rest = _rest_stub([_participant("human-1")])
        self._patch_rest(env, rest)

        result = await _standalone_send(_make_config(), "room-1", "hi")

        assert "BAND_API_KEY" in result["error"]
        assert "BAND_AGENT_ID" not in result["error"]
        rest.agent_api_messages.create_agent_chat_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_agent_id_names_the_variable(self, env):
        env.delenv("BAND_AGENT_ID")
        rest = _rest_stub([_participant("human-1")])
        self._patch_rest(env, rest)

        result = await _standalone_send(_make_config(), "room-1", "hi")

        assert "BAND_AGENT_ID" in result["error"]
        rest.agent_api_messages.create_agent_chat_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_missing_both_credentials_names_both(self, env):
        env.delenv("BAND_AGENT_ID")
        env.delenv("BAND_API_KEY")
        self._patch_rest(env, _rest_stub([_participant("human-1")]))

        result = await _standalone_send(_make_config(), "room-1", "hi")

        assert "BAND_AGENT_ID" in result["error"]
        assert "BAND_API_KEY" in result["error"]

    @pytest.mark.asyncio
    async def test_no_target_room_points_at_band_home_room(self, env):
        rest = _rest_stub([_participant("human-1")])
        self._patch_rest(env, rest)

        result = await _standalone_send(_make_config(), "", "hi")

        assert "BAND_HOME_ROOM" in result["error"]
        rest.agent_api_messages.create_agent_chat_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_unresolvable_owner_fails_without_posting(self, env):
        # The recipient is the owner now, so the reachable failure is not having
        # one. It must say so and post nothing, rather than mentioning whoever
        # happens to be in the room.
        env.delenv("BAND_OWNER_ID", raising=False)
        rest = _rest_stub([_participant("human-1", "Alice", "alice")])
        self._patch_rest(env, rest)

        result = await _standalone_send(_make_config(), "room-1", "hi")

        assert "BAND_OWNER_ID" in result["error"]
        assert "success" not in result
        rest.agent_api_messages.create_agent_chat_message.assert_not_called()

    @pytest.mark.asyncio
    async def test_participant_fetch_failure_reports_the_real_cause(self, env):
        rest = _rest_stub()
        rest.agent_api_participants.list_agent_chat_participants = AsyncMock(
            side_effect=RuntimeError("participants 503")
        )
        seen = self._patch_rest(env, rest)

        result = await _standalone_send(_make_config(), "room-1", "hi")

        assert "participants 503" in result["error"]
        rest.agent_api_messages.create_agent_chat_message.assert_not_called()
        assert seen["httpx_client"].is_closed

    @pytest.mark.asyncio
    async def test_post_failure_returns_error_dict_instead_of_raising(self, env):
        rest = _rest_stub([
            _participant("owner-uuid", "Owner", "owner"),
            _participant("human-1", "Alice", "alice"),
        ])
        rest.agent_api_messages.create_agent_chat_message = AsyncMock(
            side_effect=RuntimeError("network failure")
        )
        seen = self._patch_rest(env, rest)

        result = await _standalone_send(_make_config(), "room-1", "hi")

        assert "network failure" in result["error"]
        assert "success" not in result
        assert seen["httpx_client"].is_closed

    @pytest.mark.asyncio
    async def test_client_build_failure_is_reported(self, env):
        def _boom(api_key, base_url, httpx_client):
            raise RuntimeError("no client")

        env.setattr(_band_mod, "_standalone_rest", _boom)

        result = await _standalone_send(_make_config(), "room-1", "hi")

        assert "no client" in result["error"]

    @pytest.mark.asyncio
    async def test_missing_sdk_returns_the_install_command(self, env):
        env.setattr(_band_mod, "check_band_requirements", lambda: False)

        result = await _standalone_send(_make_config(), "room-1", "hi")

        assert "band-sdk" in result["error"]
        assert _band_mod._band_libs.sdk_install_command() in result["error"]

    # ── contract shape ────────────────────────────────────────────────────

    @pytest.mark.asyncio
    async def test_accepts_the_full_sender_signature(self, env):
        """thread_id / media_files / force_document are parity-only, never fatal."""
        rest = _rest_stub([
            _participant("owner-uuid", "Owner", "owner"),
            _participant("human-1", "Alice", "alice"),
        ])
        self._patch_rest(env, rest)

        result = await _standalone_send(
            _make_config(),
            "room-1",
            "hi",
            thread_id="ignored",
            media_files=["/tmp/nope.png"],
            force_document=True,
        )

        assert result["success"] is True
        # Band has rooms, not threads — nothing thread-shaped reaches the API.
        assert set(rest.agent_api_messages.create_agent_chat_message.await_args.kwargs) == {
            "chat_id",
            "message",
            "request_options",
        }


class TestStandaloneRestClient:
    """``_standalone_rest`` injects the caller-owned HTTP transport."""

    def test_builds_async_rest_client_with_derived_url(self, monkeypatch):
        built: dict = {}

        class _FakeAsyncRestClient:
            def __init__(self, api_key, base_url, httpx_client):
                built["api_key"] = api_key
                built["base_url"] = base_url
                built["httpx_client"] = httpx_client

        monkeypatch.setattr(
            sys.modules["band.client.rest"], "AsyncRestClient", _FakeAsyncRestClient
        )

        httpx_client = MagicMock()
        client = _band_mod._standalone_rest(
            "secret-key", "band.internal:8443", httpx_client
        )

        assert isinstance(client, _FakeAsyncRestClient)
        assert built == {
            "api_key": "secret-key",
            "base_url": "https://band.internal:8443",
            "httpx_client": httpx_client,
        }

    def test_default_host_when_no_base_url(self, monkeypatch):
        built: dict = {}

        class _FakeAsyncRestClient:
            def __init__(self, api_key, base_url, httpx_client):
                built["base_url"] = base_url

        monkeypatch.setattr(
            sys.modules["band.client.rest"], "AsyncRestClient", _FakeAsyncRestClient
        )

        _band_mod._standalone_rest("secret-key", "", MagicMock())

        assert built["base_url"] == "https://app.band.ai"


# A distinctive body used to prove message content never reaches a log.
SECRET_BODY = "correct-horse-battery-staple-9d2f"

# Logger the adapter emits under; asserted against via caplog.
_ADAPTER_LOGGER = "hermes_band_platform.adapter"


class TestSendLogging:

    @pytest.fixture
    def adapter(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        link = MagicMock()
        link.rest = _rest_stub()
        adapter._link = link
        # Out-of-turn delivery addresses the owner (the hub pattern), so the
        # owner is what has to be resolvable — not a "last sender".
        adapter._owner_uuid = "owner-uuid"
        adapter._participants_cache["room-1"] = [
            {"id": "owner-uuid", "type": "User", "name": "Owner", "handle": "owner"},
        ]
        return adapter

    @staticmethod
    def _posted_line(caplog):
        lines = [r for r in caplog.records if "Posted" in r.getMessage()]
        assert len(lines) == 1, f"expected one post line, got {len(lines)}"
        return lines[0]

    @pytest.mark.asyncio
    async def test_successful_send_reports_room_size_and_message_id(
        self, adapter, caplog
    ):
        with caplog.at_level(logging.DEBUG, logger=_ADAPTER_LOGGER):
            result = await adapter._send_on_link("room-1", "hello there")

        assert result.success is True
        record = self._posted_line(caplog)
        # Routine success is debug — this runs on every turn.
        assert record.levelno == logging.DEBUG
        message = record.getMessage()
        assert "1 chunk(s)" in message
        assert "11 chars" in message
        assert _short_id("room-1") in message
        assert _short_id("std-msg-1") in message

    @pytest.mark.asyncio
    async def test_chunked_send_reports_how_many_chunks_went_out(
        self, adapter, caplog
    ):
        content = "x" * 9000
        expected = BandAdapter.truncate_message(content, BandAdapter.MAX_MESSAGE_LENGTH)
        assert len(expected) >= 2  # guard: this must actually chunk

        with caplog.at_level(logging.DEBUG, logger=_ADAPTER_LOGGER):
            await adapter._send_on_link("room-1", content)

        assert f"{len(expected)} chunk(s)" in self._posted_line(caplog).getMessage()

    @pytest.mark.asyncio
    async def test_failed_post_logs_the_room_at_error(self, adapter, caplog):
        adapter._link.rest.agent_api_messages.create_agent_chat_message = AsyncMock(
            side_effect=RuntimeError("network failure")
        )
        with caplog.at_level(logging.DEBUG, logger=_ADAPTER_LOGGER):
            result = await adapter._send_on_link("room-1", "hi")

        assert result.success is False
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert _short_id("room-1") in errors[0].getMessage()
        assert "network failure" in errors[0].getMessage()

    @pytest.mark.asyncio
    async def test_link_lost_mid_flight_warns_that_a_reply_was_dropped(
        self, adapter, caplog
    ):
        # send() marshals onto the link loop, so the link can go away between
        # the caller's check and this one. That drop is otherwise invisible.
        adapter._link = None
        with caplog.at_level(logging.DEBUG, logger=_ADAPTER_LOGGER):
            result = await adapter._send_on_link("room-1", "hi")

        assert result.success is False
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "Dropping send" in warnings[0].getMessage()
        assert _short_id("room-1") in warnings[0].getMessage()

    @pytest.mark.asyncio
    async def test_id_less_response_warns_that_delivery_is_unconfirmed(
        self, adapter, caplog
    ):
        adapter._link.rest.agent_api_messages.create_agent_chat_message = AsyncMock(
            return_value=SimpleNamespace(data=None)
        )
        with caplog.at_level(logging.DEBUG, logger=_ADAPTER_LOGGER):
            result = await adapter._send_on_link("room-1", "hi")

        # Band accepted it but named no message: not an error, not a clean
        # success either, and everything downstream keys off that id.
        assert result.success is True and result.message_id is None
        assert any(
            r.levelno == logging.WARNING and "unconfirmed" in r.getMessage()
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_participant_fetch_logs_a_count_not_a_roster(self, caplog):
        rest = _rest_stub([
            _participant("owner-uuid", "Owner", "owner"),
            _participant("human-1", "Alice", "alice"),
        ])
        with caplog.at_level(logging.DEBUG, logger=_ADAPTER_LOGGER):
            await _fetch_participants(rest, "room-1")

        counted = [r for r in caplog.records if "participant(s)" in r.getMessage()]
        assert len(counted) == 1
        assert "2 participant(s)" in counted[0].getMessage()
        # Names and handles are room data, not diagnostics.
        assert "Alice" not in caplog.text
        assert "alice" not in caplog.text

    @pytest.mark.asyncio
    async def test_message_body_never_reaches_the_log(self, adapter, caplog):
        """The reply is the user's content; only its size may be logged."""
        body = f"here it is: {SECRET_BODY}"
        with caplog.at_level(logging.DEBUG, logger=_ADAPTER_LOGGER):
            await adapter._send_on_link("room-1", body)

        assert caplog.records  # not vacuously true
        assert SECRET_BODY not in caplog.text
        assert f"{len(body)} chars" in caplog.text

    @pytest.mark.asyncio
    async def test_message_body_never_reaches_the_log_on_failure(
        self, adapter, caplog
    ):
        adapter._link.rest.agent_api_messages.create_agent_chat_message = AsyncMock(
            side_effect=RuntimeError("network failure")
        )
        with caplog.at_level(logging.DEBUG, logger=_ADAPTER_LOGGER):
            await adapter._send_on_link("room-1", f"here it is: {SECRET_BODY}")

        assert caplog.records
        assert SECRET_BODY not in caplog.text


class TestStandaloneSendLogging:
    """Out-of-process delivery: the error is returned AND logged.

    The returned text lands in the cron job's result; an operator reading the
    gateway log would otherwise see a delivery that never happened and never
    explained itself.
    """

    @pytest.mark.asyncio
    async def test_missing_credentials_are_logged(self, monkeypatch, caplog):
        for var in ("BAND_AGENT_ID", "BAND_API_KEY", "BAND_HOME_ROOM", "BAND_HUB_ROOM"):
            monkeypatch.delenv(var, raising=False)
        with caplog.at_level(logging.DEBUG, logger=_ADAPTER_LOGGER):
            result = await _standalone_send(_make_config(), "room-1", "cron output")

        assert "error" in result
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "BAND_AGENT_ID and BAND_API_KEY" in errors[0].getMessage()

    @pytest.mark.asyncio
    async def test_missing_target_room_is_logged(self, monkeypatch, caplog):
        monkeypatch.setenv("BAND_AGENT_ID", "agent-self")
        monkeypatch.setenv("BAND_API_KEY", "secret-key")
        for var in ("BAND_HOME_ROOM", "BAND_HUB_ROOM"):
            monkeypatch.delenv(var, raising=False)
        with caplog.at_level(logging.DEBUG, logger=_ADAPTER_LOGGER):
            result = await _standalone_send(_make_config(), "", "cron output")

        assert "error" in result
        assert any(
            r.levelno == logging.ERROR and "no target room" in r.getMessage()
            for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_unresolvable_owner_is_logged_at_error(
        self, monkeypatch, caplog
    ):
        monkeypatch.setenv("BAND_AGENT_ID", "agent-self")
        monkeypatch.setenv("BAND_API_KEY", "secret-key")
        # No owner, so out-of-turn delivery has nobody to address.
        monkeypatch.delenv("BAND_OWNER_ID", raising=False)
        rest = _rest_stub([_participant("agent-self", "Bot", "bot", ptype="Agent")])
        monkeypatch.setattr(_band_mod, "_standalone_rest", lambda *a, **k: rest)

        with caplog.at_level(logging.DEBUG, logger=_ADAPTER_LOGGER):
            result = await _standalone_send(_make_config(), "room-1", "cron output")

        assert "error" in result
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        # Names the missing thing, so an operator can fix it without reading code.
        assert "BAND_OWNER_ID" in errors[0].getMessage()

    @pytest.mark.asyncio
    async def test_delivery_reports_the_message_id(self, monkeypatch, caplog):
        monkeypatch.setenv("BAND_AGENT_ID", "agent-self")
        monkeypatch.setenv("BAND_API_KEY", "secret-key")
        rest = _rest_stub([
            _participant("owner-uuid", "Owner", "owner"),
            _participant("human-1", "Alice", "alice"),
        ])
        monkeypatch.setattr(_band_mod, "_standalone_rest", lambda *a, **k: rest)

        with caplog.at_level(logging.DEBUG, logger=_ADAPTER_LOGGER):
            result = await _standalone_send(
                _make_config(), "room-1", f"cron output: {SECRET_BODY}"
            )

        assert result["success"] is True
        delivered = [r for r in caplog.records if "delivered" in r.getMessage()]
        assert len(delivered) == 1
        assert _short_id("std-msg-1") in delivered[0].getMessage()
        assert SECRET_BODY not in caplog.text


# ---------------------------------------------------------------------------
# 26. Working indicator (Band activity API)
# ---------------------------------------------------------------------------

class TestWorkingIndicator:
    """send_typing / stop_typing drive Band's boolean activity surface.

    The indicator is live state with a ~10s platform TTL, so these tests pin
    both halves of the guard: repeats inside the refresh floor are thinned, and
    a repeat *past* the floor still re-asserts (otherwise the bubble would die
    mid-turn).
    """

    @pytest.fixture
    def adapter(self, monkeypatch):
        return _make_adapter(monkeypatch)

    @staticmethod
    def _link(result=True, raises=None):
        """A link whose report_activity records calls as (room_id, working)."""
        link = MagicMock()
        calls = []

        async def _report(room_id, working, *, timeout_seconds=2):
            calls.append((room_id, working))
            if raises is not None:
                raise raises
            return result

        link.report_activity = _report
        link.calls = calls
        return link

    @staticmethod
    def _age(adapter, chat_id, seconds):
        """Backdate a room's last report so the refresh floor has elapsed."""
        adapter._working_reported[chat_id] = (
            adapter._working_reported[chat_id] - seconds
        )

    # ── Reporting ──

    @pytest.mark.asyncio
    async def test_send_typing_reports_working_true(self, adapter):
        adapter._link = self._link()
        await adapter.send_typing("room-1")
        assert adapter._link.calls == [("room-1", True)]

    @pytest.mark.asyncio
    async def test_stop_typing_reports_working_false(self, adapter):
        adapter._link = self._link()
        await adapter.send_typing("room-1")
        await adapter.stop_typing("room-1")
        assert adapter._link.calls == [("room-1", True), ("room-1", False)]

    @pytest.mark.asyncio
    async def test_reports_are_per_chat(self, adapter):
        adapter._link = self._link()
        await adapter.send_typing("room-a")
        await adapter.send_typing("room-b")
        assert adapter._link.calls == [("room-a", True), ("room-b", True)]

    @pytest.mark.asyncio
    async def test_send_typing_accepts_metadata(self, adapter):
        # The host passes thread metadata positionally/by keyword on every tick;
        # Band ignores it (rooms aren't threaded) but must still accept it.
        adapter._link = self._link()
        await adapter.send_typing("room-1", metadata={"thread_id": "t-1"})
        assert adapter._link.calls == [("room-1", True)]

    # ── The refresh floor ──

    @pytest.mark.asyncio
    async def test_repeat_within_floor_does_not_hit_the_network(self, adapter):
        adapter._link = self._link()
        for _ in range(5):
            await adapter.send_typing("room-1")
        assert adapter._link.calls == [("room-1", True)]

    @pytest.mark.asyncio
    async def test_repeat_past_floor_reasserts(self, adapter):
        # Critical: the platform expires the indicator ~10s after the last
        # report, so the guard must thin refreshes, never stop them.
        adapter._link = self._link()
        await adapter.send_typing("room-1")
        self._age(adapter, "room-1", _band_mod._WORKING_REFRESH_SECONDS + 1)
        await adapter.send_typing("room-1")
        assert adapter._link.calls == [("room-1", True), ("room-1", True)]

    @pytest.mark.asyncio
    async def test_floor_is_within_the_platform_ttl_headroom(self, adapter):
        # The SDK's rule is cadence < TTL/2. The host refreshes every ~2s, so
        # the effective cadence is floor rounded up to the next tick (4s here).
        assert 0 < _band_mod._WORKING_REFRESH_SECONDS < 5.0

    @pytest.mark.asyncio
    async def test_new_turn_reasserts_after_stop(self, adapter):
        adapter._link = self._link()
        await adapter.send_typing("room-1")
        await adapter.stop_typing("room-1")
        # A fresh turn must light the indicator immediately — the floor from the
        # previous turn must not suppress it.
        await adapter.send_typing("room-1")
        assert adapter._link.calls == [
            ("room-1", True),
            ("room-1", False),
            ("room-1", True),
        ]

    # ── Repeated / spurious stops ──

    @pytest.mark.asyncio
    async def test_stop_without_start_is_a_no_op(self, adapter):
        adapter._link = self._link()
        await adapter.stop_typing("never-started")
        assert adapter._link.calls == []

    @pytest.mark.asyncio
    async def test_repeated_stops_hit_the_network_once(self, adapter):
        # The host stops several times per turn (_stop_typing_refresh retries
        # twice, plus _keep_typing's finally).
        adapter._link = self._link()
        await adapter.send_typing("room-1")
        for _ in range(3):
            await adapter.stop_typing("room-1")
        assert adapter._link.calls == [("room-1", True), ("room-1", False)]

    @pytest.mark.asyncio
    async def test_failed_stop_is_retried_by_the_next_stop(self, adapter):
        # A clear that didn't land keeps the room asserted, so the host's own
        # repeat becomes the retry.
        adapter._link = self._link(result=False)
        await adapter.send_typing("room-1")
        await adapter.stop_typing("room-1")
        await adapter.stop_typing("room-1")
        assert adapter._link.calls == [
            ("room-1", True),
            ("room-1", False),
            ("room-1", False),
        ]

    # ── Robustness ──

    @pytest.mark.asyncio
    async def test_raising_report_does_not_propagate_from_send_typing(self, adapter):
        adapter._link = self._link(raises=RuntimeError("activity endpoint down"))
        await adapter.send_typing("room-1")  # must not raise

    @pytest.mark.asyncio
    async def test_raising_report_does_not_propagate_from_stop_typing(self, adapter):
        adapter._link = self._link()
        await adapter.send_typing("room-1")
        adapter._link.report_activity = AsyncMock(side_effect=RuntimeError("boom"))
        await adapter.stop_typing("room-1")  # must not raise

    @pytest.mark.asyncio
    async def test_failing_report_still_respects_the_floor(self, adapter):
        # The attempt is stamped, not the success — a dead endpoint must not be
        # hammered on every host tick.
        adapter._link = self._link(raises=RuntimeError("down"))
        for _ in range(5):
            await adapter.send_typing("room-1")
        assert adapter._link.calls == [("room-1", True)]

    @pytest.mark.asyncio
    async def test_old_sdk_without_report_activity_is_a_no_op(self, adapter):
        link = MagicMock()
        del link.report_activity
        adapter._link = link
        await adapter.send_typing("room-1")  # must not raise

    @pytest.mark.asyncio
    async def test_cache_is_capped(self, adapter):
        adapter._link = self._link()
        for i in range(_band_mod._ROOM_CACHE_MAX + 10):
            await adapter.send_typing(f"room-{i}")
        assert len(adapter._working_reported) <= _band_mod._ROOM_CACHE_MAX

    # ── No live link (standalone / out-of-process) ──

    @pytest.mark.asyncio
    async def test_send_typing_without_link_is_a_no_op(self, adapter):
        assert adapter._link is None  # as constructed
        await adapter.send_typing("room-1")
        # No state recorded either — nothing to clear later.
        assert adapter._working_reported == {}

    @pytest.mark.asyncio
    async def test_stop_typing_without_link_is_a_no_op(self, adapter):
        adapter._link = self._link()
        await adapter.send_typing("room-1")
        adapter._link = None
        await adapter.stop_typing("room-1")  # must not raise

    # ── Capability flag: the deliberate non-declaration ──

    def test_does_not_claim_status_text_support(self, adapter):
        # The activity API carries a bare boolean and the platform renders its
        # own "Reasoning…" label, so there is nowhere to put a status phrase.
        # Declaring support would make the host compute phrases we discard.
        assert adapter.supports_status_text is False

    # ── Wire level, through the SDK stub ──

    @pytest.mark.asyncio
    async def test_reaches_the_activity_rest_group(self, adapter):
        from band.platform.link import BandLink

        link = BandLink("agent", "key", "wss://h/ws", "https://h")
        if not isinstance(link.rest, MagicMock):
            pytest.skip("real band-sdk installed; stub-only wire assertion")
        adapter._link = link

        await adapter.send_typing("room-1")
        await adapter.stop_typing("room-1")

        reported = link.rest.agent_api_activity.report_agent_chat_activity
        assert [c.kwargs["working"] for c in reported.await_args_list] == [True, False]
        assert reported.await_args_list[0].kwargs["chat_id"] == "room-1"


# Logger the adapter emits under; asserted against via caplog.
_ADAPTER_LOGGER = "hermes_band_platform.adapter"


class TestWorkingIndicatorLogging:
    """The indicator refreshes every ~2s per turn — restraint is the point."""

    @pytest.fixture
    def adapter(self, monkeypatch):
        return _make_adapter(monkeypatch)

    @pytest.mark.asyncio
    async def test_assertion_is_logged_once_per_turn_not_per_tick(
        self, adapter, caplog
    ):
        adapter._link = TestWorkingIndicator._link()
        with caplog.at_level(logging.DEBUG, logger=_ADAPTER_LOGGER):
            await adapter.send_typing("room-1")   # first assertion
            await adapter.send_typing("room-1")   # inside the floor — thinned
            TestWorkingIndicator._age(adapter, "room-1", 10)
            await adapter.send_typing("room-1")   # past the floor — re-asserted

        # Two POSTs, one line: the refreshes are deliberately silent.
        assert adapter._link.calls == [("room-1", True), ("room-1", True)]
        asserted = [r for r in caplog.records if "asserted" in r.getMessage()]
        assert len(asserted) == 1
        assert asserted[0].levelno == logging.DEBUG

    @pytest.mark.asyncio
    async def test_clearing_the_indicator_is_logged(self, adapter, caplog):
        adapter._link = TestWorkingIndicator._link()
        await adapter.send_typing("room-1")
        with caplog.at_level(logging.DEBUG, logger=_ADAPTER_LOGGER):
            await adapter.stop_typing("room-1")

        assert any("cleared" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_a_failed_clear_says_the_indicator_is_still_set(
        self, adapter, caplog
    ):
        adapter._link = TestWorkingIndicator._link(result=False)
        await adapter.send_typing("room-1")
        with caplog.at_level(logging.DEBUG, logger=_ADAPTER_LOGGER):
            await adapter.stop_typing("room-1")

        assert any("still set" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_a_no_op_stop_stays_silent(self, adapter, caplog):
        # The host calls stop several times per turn; nothing was asserted for
        # this room, so there is nothing to say.
        adapter._link = TestWorkingIndicator._link()
        with caplog.at_level(logging.DEBUG, logger=_ADAPTER_LOGGER):
            await adapter.stop_typing("room-never-typed")

        assert caplog.records == []


# ---------------------------------------------------------------------------
# splits_long_messages under the deliberate-send design
# ---------------------------------------------------------------------------

class TestSplitsLongMessagesStaysTrue:
    """The capability is still true, and it is now also load-bearing.

    It tells the host's delivery router "this adapter chunks in send(), do not
    truncate for me". Two things had to be checked before leaving it True, since
    ``send()`` no longer posts the model's replies:

    1. The paths the flag actually governs — cron and other out-of-turn delivery
       — still chunk, so the claim remains honest.
    2. A False value would make the router chunk and call ``send()`` once per
       piece. Only the first call would see the open turn; the rest would look
       like out-of-turn traffic and be delivered to the owner separately. The
       flag is what guarantees one send per delivery.
    """

    def test_capability_is_declared(self, monkeypatch):
        adapter = _make_adapter(monkeypatch)
        assert adapter.splits_long_messages is True

    @pytest.mark.asyncio
    async def test_out_of_turn_delivery_chunks_and_repeats_the_owner_mention(
        self, monkeypatch
    ):
        adapter = _make_adapter(monkeypatch)
        adapter._agent_id = "agent-self"
        adapter._owner_uuid = "owner-uuid"
        adapter._participants_cache["room-cron-long"] = [
            {"id": "owner-uuid", "type": "User", "name": "Owner", "handle": "owner"},
        ]
        link = MagicMock()
        link.rest.agent_api_messages.create_agent_chat_message = AsyncMock(
            return_value=SimpleNamespace(data=SimpleNamespace(id="m"))
        )
        adapter._link = link

        long_content = "x" * (adapter.MAX_MESSAGE_LENGTH * 2 + 10)
        result = await adapter.send("room-cron-long", long_content)

        create = link.rest.agent_api_messages.create_agent_chat_message
        assert result.success is True
        assert create.await_count >= 3, "long cron output must be chunked, not truncated"
        # Band needs a mention per message, so every chunk carries the owner.
        for call in create.await_args_list:
            mentions = call.kwargs["message"].mentions
            assert [getattr(m, "id", None) for m in mentions] == ["owner-uuid"]

    @pytest.mark.asyncio
    async def test_a_second_send_in_one_turn_is_not_treated_as_cron(self, monkeypatch):
        """Why the flag must not become False.

        If the router chunked, ``send()`` would be called repeatedly for one
        reply. The first call closes the turn; a second would find no open turn
        and be delivered to the owner as if it were scheduled output. Pinning the
        behaviour makes that consequence visible rather than surprising.
        """
        adapter = _make_adapter(monkeypatch)
        adapter._agent_id = "agent-self"
        adapter._owner_uuid = "owner-uuid"
        adapter._participants_cache["room-two"] = [
            {"id": "owner-uuid", "type": "User", "name": "Owner", "handle": "owner"},
        ]
        link = MagicMock()
        link.rest.agent_api_messages.create_agent_chat_message = AsyncMock(
            return_value=SimpleNamespace(data=SimpleNamespace(id="m"))
        )
        link.rest.agent_api_events.create_agent_chat_event = AsyncMock()
        adapter._link = link

        _band_mod.begin_turn("room-two")
        await adapter.send("room-two", "first half")   # turn open -> thought
        await adapter.send("room-two", "second half")  # turn closed -> owner

        assert link.rest.agent_api_events.create_agent_chat_event.await_count == 1
        assert link.rest.agent_api_messages.create_agent_chat_message.await_count >= 1


class TestEmptyFinalTextIsNotPosted:
    """Band rejects a blank message with 422 "content can't be blank".

    The host can hand over empty final text — an interrupted turn, or one that
    only made tool calls — so both branches have to treat that as nothing to do
    rather than posting an empty body. Observed live on the twins before this
    guard existed.
    """

    def _adapter(self, monkeypatch):
        a = _make_adapter(monkeypatch)
        a._agent_id = "agent-self"
        a._owner_uuid = "owner-uuid"
        a._participants_cache["room-e"] = [
            {"id": "owner-uuid", "type": "User", "name": "Owner", "handle": "owner"},
        ]
        link = MagicMock()
        link.rest.agent_api_messages.create_agent_chat_message = AsyncMock()
        link.rest.agent_api_events.create_agent_chat_event = AsyncMock()
        a._link = link
        return a, link

    @pytest.mark.asyncio
    @pytest.mark.parametrize("blank", ["", "   ", "\n\t "])
    async def test_out_of_turn_blank_posts_nothing(self, monkeypatch, blank):
        adapter, link = self._adapter(monkeypatch)

        result = await adapter.send("room-e", blank)

        assert result.success is True
        link.rest.agent_api_messages.create_agent_chat_message.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_in_turn_blank_posts_no_thought(self, monkeypatch):
        adapter, link = self._adapter(monkeypatch)
        _band_mod.begin_turn("room-e")

        result = await adapter.send("room-e", "   ")

        assert result.success is True
        link.rest.agent_api_events.create_agent_chat_event.assert_not_awaited()
