---
name: add-band
description: "Connect Hermes to Band end-to-end."
version: 1.0.0
platforms: [linux, macos, windows]
metadata:
  hermes:
    tags: [band, messaging, onboarding, setup, integration]
    related_skills: [webhook-subscriptions]
---

# Add Band Skill

Connect a Hermes agent to Band from install through verification. This skill sets up the Hermes side and can optionally mint a Band external agent when the user provides a temporary user API key, but it does not keep user-level credentials after registration.

Band rooms are mention-gated and Band owns access control. The plugin creates a private Hermes Hub room on first gateway connect and stores only agent-scoped credentials in Hermes.

## When to Use

- The user asks to connect Hermes to Band or install the Band platform.
- A fresh Hermes gateway needs Band credentials, plugin enablement, gateway restart, and hub verification.
- The user wants the optional Band action toolset after chat already works.

Do not use this skill for ordinary Band chat troubleshooting after setup; inspect gateway logs and the adapter state directly instead.

## Prerequisites

- Hermes is installed and its gateway runs in a Python **3.11–3.13** environment (the Band SDK has no 3.14 wheels yet).
- You can run shell commands as the user who owns the Hermes install.
- The user can provide either:
  - Band agent credentials from `app.band.ai/agents/new`: `BAND_AGENT_ID` and `BAND_API_KEY`, or
  - a short-lived user API key in `BAND_USER_API_KEY` for automated Enterprise registration.

This skill installs the plugin and the Band SDK as part of the procedure — they need not be present beforehand.

Never ask the user to paste a Band user API key into a command line, and never read it yourself. Until the SDK registration CLI is published, the key is consumed by the bundled `scripts/register_agent.py` helper (a small Python helper that reads it from the environment) — ideally by the bootstrapper *before* this skill runs, so it never enters the agent's environment. Only the resulting agent-scoped `BAND_AGENT_ID` + `BAND_API_KEY` are stored; the user key is never printed or persisted. Once `band.cli.register_agent` ships in `band-sdk`, replace this temporary helper with the SDK CLI, but preserve the helper's browser-like registration headers (`User-Agent`, `Accept`, `Accept-Language`) because sparse script fingerprints can trigger Cloudflare 1010 on the registration endpoint. Have the user remove `BAND_USER_API_KEY` after the one registration step.

## How to Run

Use the `terminal` tool for commands and the helper scripts shipped with this skill. Run scripts with the same Python interpreter that runs Hermes.

Skill helper paths are relative to this skill directory:

- `scripts/gateway_python.py` — resolve + validate the gateway interpreter
- `scripts/register_agent.py` — temporary registration helper; replace with `band.cli.register_agent` after the SDK CLI is published
- `scripts/register-agent.sh` — curl-only registration (no SDK, no Python package); what bootstrappers use to mint the agent *before* anything is installed, so the user key never reaches the LLM
- `scripts/ensure_access_policy.py` — set Band's access policy to `allowlist` so the gateway trusts Band's ACL (idempotent; safe to run anytime)
- `scripts/ensure_home_channel.py` — set the hub as the home (main) channel by persisting `BAND_HOME_ROOM` (idempotent; safe to run anytime after the hub exists)
- `scripts/verify_install.py`
- `scripts/verify_gateway.py`
- `scripts/verify_roundtrip.py` — prove the agent can post to its owner

Registration temporarily ships as `scripts/register_agent.py` because the SDK CLI is not published yet. The helper saves `BAND_AGENT_ID` + `BAND_API_KEY` through Hermes's env writer and never prints the user key. It also sends a browser-like request fingerprint to avoid Cloudflare 1010 blocks on `app.band.ai`; the future `band-register-agent` / `band.cli.register_agent` path must keep equivalent headers before the bundled helper is removed. The setup scripts emit JSON so you can inspect success, missing checks, and next actions without exposing secrets.

This skill is **resumable**: run `verify_install.py` first and act *only* on what it reports as `blocking[]`, so re-running after a partial failure never double-installs or re-registers.

## Quick Reference

Identify the gateway interpreter first — every install and script call uses it. Use the
resolver rather than hand-rolling it; it fails loud on the wrong/unsupported Python
instead of leaving a silent wrong-venv install:

```bash
HERMES_PY="$(scripts/gateway_python.py --print)" || { scripts/gateway_python.py; exit 1; }
```

If the user already knows the interpreter, pre-set `HERMES_PY` (or `HERMES_PYTHON`) and the
resolver validates it instead of detecting — an override that can't import `hermes_cli.config`
is a hard error, never silently replaced by a guess.

Canonical install: the repo's **installer**, which ships the plugin as a **directory plugin** with zero site-packages writes — it works even when the gateway venv (e.g. `/opt/hermes/.venv` on hosted runtimes) is root-owned and read-only. It stages the plugin into `$HERMES_HOME/plugins/band/`, resolves `band-sdk>=1.1.0,<2.0.0` with the gateway interpreter into the user-writable `$HERMES_HOME/band-libs/` (the plugin prepends it to `sys.path` at load), verifies `import band`, and enables the plugin. Idempotent — safe to re-run:

```bash
# From the repo clone (this skill lives at <repo>/hermes_band_platform/skills/add-band):
HERMES_PY="$HERMES_PY" ./install.sh    # repo root; honors HERMES_HOME, HERMES_PY
```

Package install into the gateway's site-packages instead (**only** when the gateway venv is writable — self-managed installs; auto-installs `band-sdk`):

```bash
BAND_HERMES_REF="${BAND_HERMES_REF:-main}"
uv pip install --python "$HERMES_PY" "hermes-band @ git+https://github.com/band-ai/hermes-band-platform.git@${BAND_HERMES_REF}"
hermes plugins enable band 2>/dev/null && hermes plugins list | grep -w band >/dev/null \
  || "$HERMES_PY" -c "from hermes_cli import plugins_cmd as C; s=C._get_enabled_set(); s.add('band'); C._save_enabled_set(s); print('enabled band via config')"
```

Register (temporary bundled helper — needs `BAND_USER_API_KEY`), ensure the access policy, then verify install / gateway / prove a round-trip (always with the gateway interpreter):

```bash
"$HERMES_PY" scripts/register_agent.py        # mints the agent and saves BAND_AGENT_ID + BAND_API_KEY
"$HERMES_PY" scripts/ensure_access_policy.py   # gateway trusts Band's ACL (idempotent; safe to re-run)
"$HERMES_PY" scripts/verify_install.py
"$HERMES_PY" scripts/verify_gateway.py         # hub created + home (main) channel set
"$HERMES_PY" scripts/ensure_home_channel.py    # pin the hub as home if not already (idempotent)
"$HERMES_PY" scripts/verify_roundtrip.py  # connected != working: prove an outbound hub send
```

## Procedure

The flow is **state-driven**: resolve the interpreter once, take stock of what's already
done, then fix only the gaps. This makes a re-run after a partial failure safe — it never
re-installs or re-registers what's already in place.

1. Resolve the gateway interpreter — once, up front; every later step uses `$HERMES_PY`.
   - The plugin must be installed into the *same* interpreter that runs `hermes`, or it
     stays undiscoverable even though `import hermes_band_platform` succeeds from the repo
     directory. Use the resolver, which validates `import hermes_cli` and a supported
     version (3.11–3.13; `band-sdk` has no 3.14 wheels) and **fails loud** rather than
     leaving a silent wrong-venv install:
     ```bash
     HERMES_PY="$(scripts/gateway_python.py --print)" || { scripts/gateway_python.py; exit 1; }
     ```
   - On failure, run `scripts/gateway_python.py` (no `--print`) for the JSON reason
     (wrong version, `hermes_cli.config` not importable — with the underlying
     `ModuleNotFoundError` — and every candidate tried) and resolve that first.
   - `HERMES_PY` / `HERMES_PYTHON`, if already set, is used and validated instead of
     detection. Prefer that over guessing when the user knows their layout.
   - A `warning:` line on stderr never means failure — exit status is still 0 and the
     path is still valid. Two kinds, both a prompt to confirm with the user rather than
     to stop:
     - `looks like a system Python` (`is_venv: false`) — a system-wide Hermes is a real
       install, so this is a nudge, not a verdict.
     - `a running gateway reports <other path>` — this shell's `PATH` and the live
       gateway process disagree, which is how a plugin gets installed into an
       interpreter the gateway never loads. The warning names the path to use: prefer
       `HERMES_PY=<that path>` unless the user says otherwise.
   - `method` in the JSON says what the answer rests on, strongest first: `env` (you set
     it) → `pid-file` (Hermes's own record of the running gateway, read from
     `$HERMES_HOME/gateway.pid`) → `launcher-sibling` / `version-banner` /
     `launcher-shebang` (inferred from `hermes` on this `PATH`) → `running-process` (a
     `pgrep` scan) → `self`. Anything below `pid-file` describes an environment that
     *may* be the gateway's; set `HERMES_HOME` first if the user runs a named profile,
     so the resolver reads that profile's gateway record.

2. Take stock — run `scripts/verify_install.py` with the gateway interpreter and read
   `blocking[]`. Do **only** the steps it lists; skip the rest. (Read `blocking[]`, not
   `missing[]`: a correct directory-plugin install has no importable package and no entry
   point by design, so those two always appear in `missing[]` there. `blocking[]` is
   `missing[]` minus what the *installed* tree already satisfies — `installed_plugin_root`
   in the JSON names it, and is `null` when you are looking at a checkout rather than an
   install, in which case nothing is excused.)
   ```bash
   "$HERMES_PY" scripts/verify_install.py
   ```
   - `package_importable` / `entry_point` blocking → install (step 3).
   - `sdk_importable` false → install `band-sdk` (step 3 note).
   - `plugin_enabled` false → enable (step 4).
   - `band_agent_id_present` / `band_api_key_present` false → credentials (step 5).
   - `access_policy_allowlist` false → set the access policy (step 6).
   - All true → jump to restart + verification (steps 7–9).

3. Install the plugin (skip if `directory_manifest` or `package_importable` + `entry_point` are already true).
   - **Canonical: the repo installer** — directory plugin under `$HERMES_HOME`, zero writes to the gateway's site-packages, so it works on hosted runtimes where the gateway venv is read-only. It stages `$HERMES_HOME/plugins/band/`, resolves `band-sdk` into `$HERMES_HOME/band-libs/` with the gateway interpreter, verifies `import band`, and enables the plugin (idempotent):
     ```bash
     HERMES_PY="$HERMES_PY" ./install.sh   # at the repo root, three dirs above this skill
     ```
   - Package install into site-packages instead — **only when the gateway venv is writable** (self-managed installs; auto-installs `band-sdk`):
     ```bash
     BAND_HERMES_REF="${BAND_HERMES_REF:-main}"
     uv pip install --python "$HERMES_PY" "hermes-band @ git+https://github.com/band-ai/hermes-band-platform.git@${BAND_HERMES_REF}"
     hermes plugins enable band
     ```
   - For Nix installs, ensure the package and `band-sdk` are in the gateway Python environment and `plugins.enabled` contains `band`.
   - **Assert the install landed where the gateway looks:** re-run `"$HERMES_PY" scripts/verify_install.py` and confirm `directory_manifest` (or `entry_point`) plus `sdk_importable` and `band_libs_on_sys_path` are now true. This is exactly where the wrong-interpreter trap surfaces — catch it here, not in a silent runtime miss.

4. Enable the plugin (skip if you used `hermes plugins install … --enable`).
   - Try the CLI; if the build does not list entry-point plugins, write the `plugins.enabled` config directly. The runtime loader honors `plugins.enabled` on every Hermes version. **Never patch Hermes's own source to work around this.**
     ```bash
     hermes plugins enable band 2>/dev/null && hermes plugins list | grep -qw band \
       || "$HERMES_PY" -c "from hermes_cli import plugins_cmd as C; s=C._get_enabled_set(); s.add('band'); C._save_enabled_set(s); print('enabled band via config')"
     ```

5. Ensure agent credentials are present (`BAND_AGENT_ID` + `BAND_API_KEY` in Hermes's `.env`). `scripts/verify_install.py` reports whether they are.
   - Already saved (e.g. the bootstrapper registered the agent before handing off): continue.
   - Pre-created agent: have the user create one at `app.band.ai/agents/new` and save `BAND_AGENT_ID` + `BAND_API_KEY` with Hermes's env writer.
   - Auto-register from a user key — a **script step, not an LLM step**. Until the SDK CLI is published, use the bundled `scripts/register_agent.py` helper. With `BAND_USER_API_KEY` set in a plain shell, the helper reads it from the environment, mints the agent, and saves agent-scoped creds through Hermes's env writer; it never prints or persists the *user* key:
     ```bash
     "$HERMES_PY" scripts/register_agent.py   # saves BAND_AGENT_ID + BAND_API_KEY through Hermes env writer
     unset BAND_USER_API_KEY
     ```
     This is **idempotent at the flow level**: step 2 only routes here when `verify_install` reports the credentials missing, so a re-run never mints a second agent. **Do not put `BAND_USER_API_KEY` into the agent's own environment or read it yourself** — let the bootstrapper or a plain shell consume it before/outside the agent, so the user key never reaches the LLM. After `band.cli.register_agent` is published in `band-sdk`, replace this helper call with `"$HERMES_PY" -m band.cli.register_agent` only after confirming the SDK CLI sends the same Cloudflare-safe registration headers. Have the user remove `BAND_USER_API_KEY` afterward.

6. Ensure the access policy (skip if `access_policy_allowlist` is already true). Band owns access control, but the gateway only trusts an own-policy adapter's intake when its effective policy is `allowlist` — otherwise it default-denies every sender and the agent replies "not an authorized user". The current plugin sets this on the live adapter in code; this step also records it in config so it holds regardless of plugin version and **can be re-run anytime to repair an already-deployed agent** without a plugin reinstall:
   ```bash
   "$HERMES_PY" scripts/ensure_access_policy.py   # writes platforms.band.extra.{group,dm}_policy=allowlist; idempotent
   ```
   - Restart the gateway after a change for it to take effect.
   - Quick alternative for an immediate unblock without editing config: `hermes config set BAND_ALLOW_ALL true` (broader — trusts every sender Band delivers; the `allowlist` policy is the precise equivalent of Band's ACL).

7. Restart the gateway.
   - Use the user's normal Hermes gateway restart command.
   - On first connect the adapter resolves the owner, creates the Hermes Hub room, writes `BAND_HUB_ROOM`, and wires the hub as the home channel.

8. Verify the gateway with `scripts/verify_gateway.py`.
   - Success means the hub exists, gateway logs show Band connection signals, and no known hub failure signal appears in recent logs.
   - **Confirm the agent knows its owner** (`band_owner_present`). The adapter reads the owner from the `/me` endpoint on connect and persists `BAND_OWNER_ID`, so the agent can fulfill "send a message to me" from any session. If it is unresolved (empty, usually with an "Owner unresolved" failure signal), have the user set `BAND_OWNER_ID` and restart.
   - If no Band log signals appear, re-run install verification and confirm the gateway process is using the expected Python environment.
   - **Ensure the hub is the home (main) channel** — cron and agent notifications deliver to the home channel, and an agent with no home complains it has nowhere to deliver. The adapter wires this on connect, but make it durable / repair an older build with:
     ```bash
     "$HERMES_PY" scripts/ensure_home_channel.py   # persists BAND_HOME_ROOM = hub if unset; idempotent
     ```
     If it reports no hub yet (`BAND_HUB_ROOM` unset), the gateway hasn't created the hub — fix owner resolution / connection first (above), then re-run. Restart the gateway after a change.

9. **Prove the round-trip** with `scripts/verify_roundtrip.py` — *connected is not working*. `connect()` only opens the socket, and hub bootstrap runs in a `try/except` that never blocks connect, so confirm the agent can actually post to its owner:
   ```bash
   "$HERMES_PY" scripts/verify_roundtrip.py                 # outbound proof (default)
   "$HERMES_PY" scripts/verify_roundtrip.py --await-reply   # full duplex: also wait for the owner's @mention
   ```
   - A successful send exercises auth + room + the exact REST path real replies use. If this fails after `verify_gateway.py` passed, the problem is mentions/room state, not the socket.
   - `--await-reply` additionally posts the check, then waits for the owner to @mention back — proving inbound delivery too. Use it when you want the user to confirm live.
   - This is also the proof that **"send a message to me" works**: the agent reaches its owner by calling `band_send_message` with no `room_id` (it falls back to the owner's hub/home and @mentions the owner), which is the same path the round-trip exercises.

10. **Close the loop — offer the first real room.** The hub is a private owner↔agent control room; the user's actual goal is usually a room with other people. Don't stop at the hub:
   - Offer: *"You're live in the Hub. Want me to create your first room? Tell me who to add."*
   - On yes, use `band_create_room(person=<handle/name>, message=<intro>)` — it resolves → creates → adds → messages in one call and returns `{room_id, added, sent}`.
   - The Band tools are owner-gated and **fail-closed**. From this setup session the caller may not be the resolved Band owner, so a mutation can be refused: if so, either have the user run the request from the Hub in Band (where the owner bypass applies) or add the calling `platform:user_id` to `BAND_TOOL_OWNERS`. On decline, point them at `band_create_room` / the Band UI for later.
   - Remind them Band has no DMs; an unmentioned message is ignored by design.

11. Optionally enable Band action tools (if not already needed for step 10).
   - Add the `band` toolset to each platform that should act on Band in `platform_toolsets` (`~/.hermes/config.yaml`) — **including the `band` platform itself**, or Band sessions get the messaging channel but none of the action tools.
   - Keep `hermes-band` for the messaging channel and add `band` only where action tools are needed.
   - Mutating tools remain owner-gated by the plugin.

## Pitfalls

- Installing into a different Python than the gateway's: `import hermes_band_platform` and `hermes plugins list` can look fine from the repo directory (cwd is on `sys.path`), yet the running gateway never discovers it. Always resolve with `--python "$HERMES_PY"` and confirm from a neutral directory.
- Installing `band-sdk` into the gateway's site-packages on a hosted runtime: the venv (e.g. `/opt/hermes/.venv`) is root-owned and read-only, so `uv pip install --python "$HERMES_PY" band-sdk` dies with `Permission denied`. Use `--target "${HERMES_HOME:-$HOME/.hermes}/band-libs"` (what the installer does) — the plugin prepends that dir to the gateway's `sys.path` at load. Never reach for `sudo`.
- Hand-rolling interpreter detection, or trusting `python -c "import hermes_cli"` as proof: cwd is on `sys.path` for `-c` and `PYTHONPATH` is inherited, so from a Hermes source tree *any* interpreter passes that check — and then fails later with `ModuleNotFoundError: yaml` under a Python that was never the gateway's. Use `scripts/gateway_python.py`, which probes with `-E` from an empty directory and performs the real import.
- Patching Hermes's own source to make `hermes plugins enable`/`list` show an entry-point plugin: unnecessary and fragile (it breaks on the next Hermes upgrade and is absent on pip/Docker installs). Use the `plugins.enabled` config fallback instead — the runtime loader honors it regardless.
- Installing the package but forgetting `hermes plugins enable band` leaves the entry point discovered but inactive.
- Installing a directory plugin without `band-sdk` resolvable by the gateway (in `$HERMES_HOME/band-libs` or site-packages) fails the plugin at load with one actionable error in the gateway log naming the exact `uv pip install --target` fix.
- Saving `BAND_AGENT_ID` with generic config-setting commands can route it to config YAML instead of Hermes `.env`; use the setup wizard or Hermes env writer.
- Treating a successful WebSocket connect as full setup is incomplete; the hub must also be created and persisted as `BAND_HUB_ROOM`, **and** the agent must be able to post to it — prove the latter with `scripts/verify_roundtrip.py`, since hub bootstrap runs in a `try/except` that never blocks connect.
- The agent rejecting senders with "not an authorized user" (owner included): the gateway needs Band's effective access policy to be `allowlist`, not just the `enforces_own_access_policy` flag. Run `scripts/ensure_access_policy.py` (records `platforms.band.extra.group_policy=allowlist`) and restart — it's idempotent and version-independent, so it repairs an already-deployed agent without a plugin reinstall.
- The agent complaining it has "no home" / nowhere to deliver cron or notifications: the hub must be set as the home (main) channel. The adapter wires it on connect and persists `BAND_HOME_ROOM`, but an older build only set it in-memory, so readers that loaded config fresh saw no home. Run `scripts/ensure_home_channel.py` (persists `BAND_HOME_ROOM = BAND_HUB_ROOM`) and restart. If `BAND_HUB_ROOM` is also unset the hub was never created — resolve owner/connection first.
- Leaving `BAND_USER_API_KEY` in the environment after registration unnecessarily keeps a broad user credential live.
- Exposing the Band *user* key to the LLM: registration is a script step — the temporary bundled `scripts/register_agent.py` helper reads `BAND_USER_API_KEY` from the environment and saves only agent-scoped credentials. Don't set the user key in the *agent's* own environment or echo it; run registration in the bootstrapper or a plain shell before the agent takes over. Replace the helper with the SDK CLI once `band.cli.register_agent` is published.
- Dropping the registration headers when moving to the SDK CLI: the registration endpoint can Cloudflare-1010 sparse script clients. Preserve the helper's browser-like `User-Agent`, `Accept`, and `Accept-Language` headers in `band.cli.register_agent`.
- Setting `BAND_HOME_ROOM` to another room means cron and notifications use that room instead of the hub.

## Verification

- `scripts/gateway_python.py` resolves and version-gates the gateway interpreter (run before anything else).
- `scripts/verify_install.py` reports package or directory plugin presence, SDK importability (after applying the same `band-libs` shim the gateway applies — `band_libs_on_sys_path` asserts the shim took effect), plugin enablement, required credential presence, and whether the access policy authorizes Band traffic (`access_policy_allowlist`). Run it as `$HERMES_PY` so its checks reflect the gateway interpreter, not whatever shell python you happen to be in.
- `scripts/ensure_access_policy.py` sets Band's access policy to `allowlist` so the gateway trusts Band's ACL. Idempotent and safe to run anytime — use it to repair an agent that rejects its owner with "not an authorized user".
- `scripts/ensure_home_channel.py` persists the hub as the home (main) channel (`BAND_HOME_ROOM`). Idempotent and safe to run anytime after the hub exists — use it to repair an agent that complains it has "no home".
- `scripts/verify_gateway.py` reports `BAND_HUB_ROOM`, recent Band gateway success signals, and known failure signals.
- `scripts/verify_roundtrip.py` proves the agent can actually post to the hub (and, with `--await-reply`, that the owner's @mention reaches it) — the step that turns "connected" into "working".
- The user confirms the Hermes Agent Hub room exists in Band and an @mention test message round-trips.
