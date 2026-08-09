# Cross-machine agent install prompt

> **Most installs should use the Band web app's "Add to Hermes" flow** (generated
> from the bootstrapper source at
> [band-ai/add-band](https://github.com/band-ai/add-band)) — on the gateway host it
> installs the `add-band` skill from this repo and hands off to `hermes chat -s add-band`.

Use the prompt below only when you **can't run that on the target host** — e.g.
you're driving setup from a **different machine or a non-Hermes agent** (Claude
Code, etc.), or Hermes isn't yet configured as a working agent.
Hand it to any shell-capable agent with access to the gateway host. It fetches the
official `add-band` setup skill from GitHub and follows it end to end — install,
enable, credentials, restart, and verification all live in the skill, so this
prompt stays thin and never goes stale.

The agent pauses at two points that need you: providing Band credentials, and the
final in-Band @mention test.

---

```
You're connecting this machine's Hermes install to Band for me. Work in the shell.

1. Clone the repo and open the setup skill — it is the source of truth. Read it fully, then
   follow its Procedure end to end:
     git clone --depth 1 https://github.com/band-ai/hermes-band-platform /tmp/hbp
     cat /tmp/hbp/hermes_band_platform/skills/add-band/SKILL.md
   Run the skill's helper scripts (gateway_python.py, verify_install.py, verify_gateway.py,
   verify_roundtrip.py) from /tmp/hbp/hermes_band_platform/skills/add-band with the gateway's
   own Python (the skill shows how to derive it as HERMES_PY). Registration temporarily
   uses the bundled `scripts/register_agent.py` helper; switch to the SDK
   `band-register-agent` / `band.cli.register_agent` command after it is published,
   but only after verifying the SDK CLI preserves the helper's browser-like
   registration headers (`User-Agent`, `Accept`, `Accept-Language`) to avoid
   Cloudflare 1010 from sparse script fingerprints.

2. Ground rules — honor these even where a step is ambiguous:
   • Install with the repo's installer: `/tmp/hbp/install.sh`. It ships the plugin as a
     DIRECTORY plugin under $HERMES_HOME (default ~/.hermes) and resolves
     `band-sdk>=1.3.0,<2.0.0` into $HERMES_HOME/band-libs with the gateway's interpreter
     (Python 3.11–3.13) — ZERO writes to the gateway's site-packages, so it works on hosted
     runtimes where the gateway venv (e.g. /opt/hermes/.venv) is root-owned and read-only.
   • Never `sudo`, never write to the gateway venv, never edit the gateway's launch env
     (PYTHONPATH etc.) — the hosting platform resets such changes on redeploy. Everything
     must land under $HERMES_HOME.
   • Dependency resolution always uses the gateway interpreter (`--python "$HERMES_PY"`),
     even though files land in --target $HERMES_HOME/band-libs; wheels must match the
     gateway's Python. If `import band` still fails after the installer, surface the
     gateway log's one-line fix verbatim — do not improvise another install route.
   • Never edit or patch Hermes's own source. `hermes plugins enable band` works natively
     for directory plugins; the plugins.enabled config fallback is for legacy package
     installs only.
   • Never make me paste a Band *user* API key into a command — I'll set BAND_USER_API_KEY
     for the one registration step, then remove it.

3. Stop and ask me at the two human gates:
   • Credentials — I either create the Band agent at app.band.ai/agents/new and give you
     BAND_AGENT_ID + BAND_API_KEY, or I set BAND_USER_API_KEY for the bundled
     `scripts/register_agent.py` helper.
   • The live test — I @mention the agent in the "Hermes Agent Hub" room and confirm a reply.

4. When done, report: plugin version, how you enabled it (CLI vs config), the BAND_HUB_ROOM id,
   and the verify_gateway.py result.
```

---

## Notes

- **Reachability:** the agent must be able to clone
  `https://github.com/band-ai/hermes-band-platform` and let `uv` fetch `band-sdk`
  wheels from PyPI. No package install of `hermes-band` itself is needed — the
  installer ships it as a directory plugin.
- **Already installed?** Once the plugin is installed and the gateway restarted,
  the skill is also available natively as `hermes chat -s band:add-band` — the
  clone step is only needed for the first install on a fresh box.
- **Read-only gateway venv (hosted runtimes — the common case):** the installer is
  the only supported path; a `pip`/`uv pip` install into the gateway's Python dies
  with `Permission denied` by design. Package installs into site-packages remain an
  option solely for self-managed boxes with a writable venv; the skill covers both.
