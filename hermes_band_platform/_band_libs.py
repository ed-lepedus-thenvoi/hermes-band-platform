"""Resolve the Band SDK from ``$HERMES_HOME/band-libs`` (directory installs).

Directory plugins ship files only — Hermes does not install their Python
dependencies (see ``docs/hermes-plugin-dependency-metadata-issue.md``), and on
hosted runtimes the gateway's own site-packages is read-only, so ``band-sdk``
cannot live there. The installer instead resolves it into a user-writable
target dir::

    uv pip install --python "<gateway_python>" \
        --target "$HERMES_HOME/band-libs" "band-sdk>=1.3.0,<2.0.0"

and this shim prepends that dir to ``sys.path`` before anything imports
``band``. Prepending (not appending) makes ``band-libs`` authoritative when
both it and a site-packages copy exist, so upgrading the target dir always
takes effect.

Two entry points, one shared probe:

- ``bootstrap()`` — called from the package ``__init__`` (before ``adapter.py``'s
  module-top SDK import guard binds). NEVER raises: when the SDK is missing it
  logs ONE clear, actionable error and lets the adapter degrade gracefully
  (``BAND_AVAILABLE = False``), so Band stays registered and enumerable as an
  unconfigured channel with an install hint — raising here would abort plugin
  load and silently drop Band from every channel surface.
- ``ensure_band_importable()`` — the hard variant for install/verify time
  (``install.sh``): raises that same error, because an installer SHOULD fail
  loud.
"""

from __future__ import annotations

import importlib.util
import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

BAND_SDK_SPEC = "band-sdk>=1.3.0,<2.0.0"


def hermes_home() -> Path:
    """The Hermes home dir — ``$HERMES_HOME`` if set, else ``~/.hermes``."""
    override = os.environ.get("HERMES_HOME", "").strip()
    return Path(override).expanduser() if override else Path.home() / ".hermes"


def band_libs_dir() -> Path:
    """Where the installer targets the Band SDK: ``$HERMES_HOME/band-libs``."""
    return hermes_home() / "band-libs"


def prepend_band_libs() -> Path | None:
    """Prepend ``band-libs`` to ``sys.path`` if it exists. Returns the path used."""
    libs = band_libs_dir()
    if not libs.is_dir():
        return None
    entry = str(libs)
    if entry in sys.path:
        return libs
    sys.path.insert(0, entry)
    return libs


def sdk_install_command() -> str:
    """The exact no-sudo, no-site-packages-write command that fixes a missing SDK."""
    return (
        f'uv pip install --python "{sys.executable}" '
        f'--target "{band_libs_dir()}" "{BAND_SDK_SPEC}"'
    )


def _resolve() -> str | None:
    """Shared probe: make ``import band`` resolvable; return an error message
    (the ONE clear fix) when it can't be, ``None`` when it can.

    Path order: an already-imported/stubbed ``band`` (tests) wins; otherwise
    ``band-libs`` is prepended so a directory install resolves its own SDK;
    site-packages remains the natural fallback for wheel installs.
    """
    if "band" in sys.modules:  # real SDK or a test stub — nothing to do
        return None
    libs = prepend_band_libs()
    try:
        found = importlib.util.find_spec("band") is not None
    except (ImportError, ValueError):
        found = False
    if found:
        return None
    checked = str(libs) if libs is not None else f"{band_libs_dir()} (not present)"
    return (
        "The Band SDK is not importable by the Hermes gateway "
        f"(checked {checked} and the interpreter's own packages). "
        "Fix it with:\n"
        f"  {sdk_install_command()}\n"
        "then restart the gateway. No sudo and no write to the gateway's "
        "site-packages is required."
    )


def bootstrap() -> str | None:
    """Plugin-load-time shim: never raises.

    Prepends ``band-libs`` and, when the SDK still can't resolve, logs the one
    actionable error and returns it — the adapter's own import guard then
    degrades to ``BAND_AVAILABLE = False`` so Band remains a registered,
    enumerable channel (with an install hint) instead of vanishing.
    """
    error = _resolve()
    if error is not None:
        logger.error("[band] %s", error)
    return error


def ensure_band_importable() -> None:
    """Install/verify-time shim: raise the one clear, actionable error."""
    error = _resolve()
    if error is not None:
        raise ImportError(error)
