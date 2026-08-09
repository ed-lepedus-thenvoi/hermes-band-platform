#!/usr/bin/env python3
"""Verify that the Band Hermes plugin is installed, enabled, and configured."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

# The plugin root that ships this skill: scripts/ -> add-band/ -> skills/ -> root.
# Repo layout: <repo>/hermes_band_platform; installed: $HERMES_HOME/plugins/band.
_PLUGIN_ROOT = Path(__file__).resolve().parents[3]


def _load_band_libs_shim() -> Any:
    """Import the plugin's ``_band_libs`` shim by path (no package import)."""
    path = _PLUGIN_ROOT / "_band_libs.py"
    if not path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location("_band_libs_verify", path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception:
        return None


def _apply_band_libs_shim() -> dict[str, Any]:
    """Mirror the gateway's startup shim: prepend ``$HERMES_HOME/band-libs``.

    Returns ``{present, on_sys_path, dir}`` — run this *before* probing
    ``import band`` so an SDK that lives only in ``band-libs`` (the read-only
    site-packages install path) counts as importable, exactly as it does in the
    gateway process.
    """
    shim = _load_band_libs_shim()
    if shim is None:
        home = Path(os.environ.get("HERMES_HOME", "").strip() or Path.home() / ".hermes")
        libs = home / "band-libs"
        if libs.is_dir() and str(libs) not in sys.path:
            sys.path.insert(0, str(libs))
    else:
        libs = shim.band_libs_dir()
        shim.prepend_band_libs()
    return {
        "present": libs.is_dir(),
        "on_sys_path": str(libs) in sys.path,
        "dir": str(libs),
    }


def _entry_points_for_group() -> list[Any]:
    eps = importlib.metadata.entry_points()
    if hasattr(eps, "select"):
        return list(eps.select(group="hermes_agent.plugins"))
    if isinstance(eps, dict):
        return list(eps.get("hermes_agent.plugins", []))
    return [ep for ep in eps if ep.group == "hermes_agent.plugins"]


def _has_band_entry_point() -> bool:
    try:
        return any(ep.name == "band" for ep in _entry_points_for_group())
    except Exception:
        return False


def _manifest_roots() -> list[Path]:
    """Trees that could carry a directory-plugin manifest.

    The plugin root that ships this skill ($HERMES_HOME/plugins/band when
    installed, <repo>/hermes_band_platform in a checkout), plus the legacy
    git-clone root one level up, which carries a generated shim.
    """
    roots = [_PLUGIN_ROOT]
    try:
        roots.append(Path(__file__).resolve().parents[4])
    except IndexError:
        pass
    return roots


def _has_manifest(root: Path) -> bool:
    return (root / "plugin.yaml").exists() and (root / "__init__.py").exists()


def _has_directory_manifest() -> bool:
    """Whether a directory-plugin manifest sits at either candidate root.

    Markers only: this says the tree *could* be loaded as a directory plugin, not
    that the gateway loads it — see ``_installed_plugin_root()``.
    """
    return any(_has_manifest(root) for root in _manifest_roots())


def _hermes_home() -> Path:
    """The Hermes home whose ``plugins/`` the gateway loads directory plugins from.

    Ask the host first: Hermes has a real profile concept
    (``/opt/data/profiles/<name>``) that a bare ``~/.hermes`` default cannot
    reproduce. This script runs under the gateway interpreter, so ``hermes_cli`` is
    importable by construction; env-or-default is only the fallback.
    """
    try:
        from hermes_cli.config import get_hermes_home

        home = get_hermes_home()
        if home:
            return Path(str(home)).expanduser()
    except Exception:
        pass
    explicit = os.environ.get("HERMES_HOME", "").strip()
    return Path(explicit).expanduser() if explicit else Path.home() / ".hermes"


def _installed_plugin_root() -> Optional[Path]:
    """The tree shipping this skill, *if the gateway actually loads it* — markers
    present **and** the root under ``$HERMES_HOME/plugins``, which is the only
    place the host loads directory plugins from.

    Markers alone are not evidence of an install. This repo ships ``plugin.yaml`` +
    ``__init__.py`` inside ``hermes_band_platform/`` *and*, generated, at the repo
    root, so every checkout carries them. Excusing ``package_importable`` /
    ``entry_point`` on that basis let a clone with nothing staged report those as
    non-blocking — so the "take stock" step skipped the install that was actually
    needed. Covers both installed sub-layouts, since the flattened
    (``install.sh``) and nested (``hermes plugins install``) trees both land under
    ``plugins/band``.
    """
    plugins_dir = (_hermes_home() / "plugins").resolve()
    for root in _manifest_roots():
        if not _has_manifest(root):
            continue
        try:
            if root.is_relative_to(plugins_dir):
                return root
        except (OSError, ValueError):
            continue
    return None


def _plugin_enabled() -> bool:
    try:
        from hermes_cli.config import load_config

        config = load_config()
    except Exception:
        return False
    plugins_cfg = config.get("plugins", {}) if isinstance(config, dict) else {}
    enabled = plugins_cfg.get("enabled", []) if isinstance(plugins_cfg, dict) else []
    return isinstance(enabled, list) and "band" in enabled


def _env_value(name: str) -> str:
    try:
        from hermes_cli.config import get_env_value

        return str(get_env_value(name) or "")
    except Exception:
        return os.getenv(name, "")


def _access_policy_allowlist() -> bool:
    """Whether Band's access policy authorizes Band traffic at the gateway.

    The gateway only trusts Band's own ACL when the effective policy for the
    chat type is ``"allowlist"`` (Band has no DMs, so traffic is group). True if
    the config records ``platforms.band.extra.group_policy = "allowlist"`` (the
    version-independent record written by ``ensure_access_policy.py``) or
    ``BAND_ALLOW_ALL`` is set. False (→ default-deny, "not an authorized user")
    when neither is present.
    """
    if _env_value("BAND_ALLOW_ALL").strip().lower() in {"true", "1", "yes"}:
        return True
    try:
        from hermes_cli.config import load_config

        config = load_config()
    except Exception:
        return False
    platforms = config.get("platforms", {}) if isinstance(config, dict) else {}
    band = platforms.get("band", {}) if isinstance(platforms, dict) else {}
    extra = band.get("extra", {}) if isinstance(band, dict) else {}
    return isinstance(extra, dict) and str(extra.get("group_policy", "")).strip().lower() == "allowlist"


def _conversations_skill_present() -> bool:
    """Whether the bundled ``band-conversations`` runtime skill ships with the
    install.

    This is the ``SKILL.md`` that ``adapter.register()`` registers as
    ``band:band-conversations`` — the multi-participant / delegation playbook the
    agent loads on demand from the Band platform hint. If it is missing, the
    agent still connects and chats but has no conversation playbook (an older
    build predating the skill). Checked the same two ways the install can ship:
    via the importable package (wheel) or relative to this script (directory
    manifest / editable).
    """
    try:
        spec = importlib.util.find_spec("hermes_band_platform")
        if spec is not None and spec.origin:
            pkg = Path(spec.origin).parent / "skills" / "band-conversations" / "SKILL.md"
            if pkg.is_file():
                return True
    except Exception:
        pass
    # Fallback: scripts/ -> add-band/ -> skills/ -> hermes_band_platform/
    pkg_dir = Path(__file__).resolve().parents[3]
    return (pkg_dir / "skills" / "band-conversations" / "SKILL.md").is_file()


def verify_install() -> dict[str, Any]:
    # Apply the gateway's band-libs shim first so ``sdk_importable`` reflects
    # what the gateway process actually resolves (band-libs is prepended to
    # sys.path at plugin load; site-packages stays the fallback).
    band_libs = _apply_band_libs_shim()
    package_importable = importlib.util.find_spec("hermes_band_platform") is not None
    sdk_importable = importlib.util.find_spec("band") is not None
    entry_point = _has_band_entry_point()
    directory_manifest = _has_directory_manifest()
    # Routing (`blocking`, `actions`, `success`) turns on the *installed* tree, not
    # on marker files that any checkout carries.
    installed_root = _installed_plugin_root()
    installed_directory_plugin = installed_root is not None
    enabled = _plugin_enabled()
    agent_id_present = bool(_env_value("BAND_AGENT_ID").strip())
    api_key_present = bool(_env_value("BAND_API_KEY").strip())
    access_policy = _access_policy_allowlist()
    conversations_skill = _conversations_skill_present()
    # ``band-libs`` must be on the gateway's sys.path whenever the directory
    # install owns the SDK. When the SDK resolves from site-packages instead
    # (wheel/self-managed install), the check is satisfied vacuously.
    band_libs_on_sys_path = band_libs["on_sys_path"] or (
        sdk_importable and not band_libs["present"]
    )
    checks = {
        "package_importable": package_importable,
        "sdk_importable": sdk_importable,
        "band_libs_on_sys_path": band_libs_on_sys_path,
        "entry_point": entry_point,
        "directory_manifest": directory_manifest,
        "plugin_enabled": enabled,
        "band_agent_id_present": agent_id_present,
        "band_api_key_present": api_key_present,
        "access_policy_allowlist": access_policy,
        "conversations_skill_present": conversations_skill,
    }
    missing = [name for name, ok in checks.items() if not ok]
    # `missing` is the raw check list, so on a *correct* directory-plugin install
    # it still lists package_importable/entry_point — that layout has neither, by
    # design. Routing off it tells the setup agent to re-install something that is
    # already installed, so publish the checks that actually gate success. The
    # exemption is scoped to the installed tree: a checkout carries the same
    # manifest markers while nothing has been staged, and excusing them there hides
    # the install that is genuinely missing.
    satisfied_by_directory = (
        {"package_importable", "entry_point"} if installed_directory_plugin else set()
    )
    blocking = [name for name in missing if name not in satisfied_by_directory]
    actions: list[str] = []
    if not installed_directory_plugin and (
        "package_importable" in missing or "entry_point" in missing
    ):
        actions.append(
            "Install the plugin. Canonical (works on read-only gateway venvs, "
            "no sudo): clone https://github.com/band-ai/hermes-band-platform "
            "and run ./install.sh — it stages $HERMES_HOME/plugins/band, "
            "resolves band-sdk into $HERMES_HOME/band-libs, and enables the "
            "plugin. Package alternative (writable gateway venv only): "
            "uv pip install --python \"$HERMES_PY\" "
            "\"hermes-band @ git+https://github.com/band-ai/hermes-band-platform.git@${BAND_HERMES_REF:-main}\""
        )
    if "sdk_importable" in missing or "band_libs_on_sys_path" in missing:
        actions.append(
            "band-sdk is missing. Directory plugin installs do not install Python "
            "dependencies, and the gateway's site-packages may be read-only; "
            "resolve it into the user-writable band-libs dir (needs no sudo and "
            "no site-packages write): "
            'uv pip install --python "$HERMES_PY" --target '
            f"\"{band_libs['dir']}\" 'band-sdk>=1.1.0,<2.0.0' "
            "— then restart the gateway."
        )
    if "plugin_enabled" in missing:
        actions.append(
            "Enable the plugin with `hermes plugins enable band`; if the CLI does "
            "not list entry-point plugins, add `band` to plugins.enabled in the "
            "Hermes config."
        )
    if "band_agent_id_present" in missing or "band_api_key_present" in missing:
        actions.append(
            "Save agent-scoped credentials in Hermes env: BAND_AGENT_ID and "
            "BAND_API_KEY. Use scripts/register_agent.py with BAND_USER_API_KEY, "
            "or paste credentials from a pre-created Band external agent."
        )
    if "access_policy_allowlist" in missing:
        actions.append(
            "Configure Band's access policy so the gateway trusts Band's ACL "
            "(otherwise the agent rejects senders with 'not an authorized user'). "
            "Run: \"$HERMES_PY\" scripts/ensure_access_policy.py, then restart the gateway."
        )
    if "conversations_skill_present" in missing:
        actions.append(
            "The band-conversations runtime skill is missing (older build); the "
            "agent will connect but lack the multi-participant/delegation playbook. "
            "Refresh the install (re-run ./install.sh from a fresh clone, or for "
            "package installs: uv pip install --python \"$HERMES_PY\" --upgrade "
            "\"hermes-band @ "
            "git+https://github.com/band-ai/hermes-band-platform.git@${BAND_HERMES_REF:-main}\"), "
            "then restart the gateway."
        )
    return {
        "success": (
            sdk_importable
            and band_libs_on_sys_path
            and (package_importable or installed_directory_plugin)
            and (entry_point or installed_directory_plugin)
            and enabled
            and agent_id_present
            and api_key_present
            and access_policy
            and conversations_skill
        ),
        "checks": checks,
        "band_libs_dir": band_libs["dir"],
        # Which tree the gateway loads, or null when this one is only a checkout.
        # `checks.directory_manifest` reports markers; this reports the install.
        "installed_plugin_root": str(installed_root) if installed_root else None,
        "missing": missing,
        "blocking": blocking,
        "actions": actions,
    }


def main() -> int:
    result = verify_install()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
