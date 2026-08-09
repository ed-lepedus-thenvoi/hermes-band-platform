#!/usr/bin/env bash
#
# Install the Band plugin as a Hermes DIRECTORY plugin — no sudo, and no writes
# to the gateway's site-packages. Built for hosted runtimes where the gateway
# venv (e.g. /opt/hermes/.venv) is root-owned and read-only: everything this
# script writes lands under $HERMES_HOME (default ~/.hermes), which the gateway
# user owns.
#
#   1. Stage the plugin directory into  $HERMES_HOME/plugins/band/
#   2. Resolve band-sdk with the GATEWAY interpreter (correct wheels for its
#      Python/platform) into the user-writable  $HERMES_HOME/band-libs/
#      (the plugin's _band_libs shim prepends that dir to sys.path at load)
#   3. hermes plugins enable band
#
# Idempotent: re-running refreshes the plugin dir, re-resolves the SDK pin, and
# leaves an already-enabled plugin enabled.
#
# Env knobs:
#   HERMES_HOME    Hermes home (default ~/.hermes)
#   HERMES_PY      gateway interpreter override (skips auto-resolution)
#   BAND_SDK_SPEC  band-sdk requirement (default: band-sdk>=1.1.0,<2.0.0)
set -euo pipefail

die() { printf 'install.sh: %s\n' "$*" >&2; exit 1; }

command -v uv >/dev/null || die "install uv first: https://docs.astral.sh/uv/"
command -v hermes >/dev/null || die "install hermes first (the hermes CLI must be on PATH)"

HERMES_HOME="${HERMES_HOME:-$HOME/.hermes}"
BAND_SDK_SPEC="${BAND_SDK_SPEC:-band-sdk>=1.1.0,<2.0.0}"

# Locate the plugin source next to this script. Two layouts are supported:
# a repo checkout (hermes_band_platform/) and an extracted release bundle
# (band/ beside install.sh, or install.sh inside the plugin dir itself).
src_root="$(cd "$(dirname "$0")" && pwd)"
plugin_src=""
for candidate in "$src_root/hermes_band_platform" "$src_root/band" "$src_root"; do
  if [ -f "$candidate/plugin.yaml" ] && [ -f "$candidate/adapter.py" ]; then
    plugin_src="$candidate"
    break
  fi
done
[ -n "$plugin_src" ] || die "could not find the plugin source (plugin.yaml + adapter.py) near $src_root"

# --- Resolve the gateway interpreter -----------------------------------------
# band-sdk wheels must match the *gateway's* Python (3.11–3.13), so resolution
# always uses that interpreter — but nothing is ever installed into its venv.
resolver="$plugin_src/skills/add-band/scripts/gateway_python.py"
if [ -z "${HERMES_PY:-}" ]; then
  for py in python3 python; do
    command -v "$py" >/dev/null || continue
    if HERMES_PY="$("$py" "$resolver" --print)"; then
      break
    fi
    HERMES_PY=""
  done
  [ -n "${HERMES_PY:-}" ] || die "could not resolve the gateway Python; set HERMES_PY and re-run (diagnose with: python3 $resolver)"
else
  # Explicit override: still refuse an interpreter that isn't the gateway's or
  # can't run band-sdk (3.11–3.13; no 3.14 wheels yet).
  "$HERMES_PY" -c 'import sys, importlib.util
ok_ver = (3, 11) <= sys.version_info[:2] <= (3, 13)
ok_cli = importlib.util.find_spec("hermes_cli") is not None
sys.exit(0 if (ok_ver and ok_cli) else 1)' \
    || die "HERMES_PY=$HERMES_PY is not a gateway interpreter (needs hermes_cli importable and Python 3.11–3.13)"
fi
echo "gateway python: $HERMES_PY"

# --- 1. Stage the CANDIDATE plugin directory ----------------------------------
# Staged next to the destination (same filesystem) but NOT swapped in yet:
# every fallible step below (SDK resolve, import verification, shadow check)
# runs first, so a failure can never leave a previously-working install
# replaced by unverified files.
plugins_dir="$HERMES_HOME/plugins"
dest="$plugins_dir/band"
mkdir -p "$plugins_dir"
tmp="$(mktemp -d "$plugins_dir/.band.staging.XXXXXX")"
trap 'rm -rf "$tmp"' EXIT
cp -R "$plugin_src/." "$tmp/"
find "$tmp" -name '__pycache__' -type d -prune -exec rm -rf {} +
find "$tmp" -name '*.pyc' -delete

# --- 2. Resolve band-sdk into the user-writable band-libs dir ----------------
band_libs="$HERMES_HOME/band-libs"
uv pip install --python "$HERMES_PY" --target "$band_libs" --upgrade "$BAND_SDK_SPEC"

# Prove the gateway will resolve the SDK exactly the way the plugin loads it:
# through the CANDIDATE's _band_libs shim, not a hand-rolled sys.path edit.
HERMES_HOME="$HERMES_HOME" "$HERMES_PY" -I - "$tmp" <<'PY'
import importlib.util, sys

shim_path = sys.argv[1] + "/_band_libs.py"
spec = importlib.util.spec_from_file_location("_band_libs_install", shim_path)
shim = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shim)
shim.ensure_band_importable()  # raises with the exact fix if band is missing
import band  # noqa: F401

origin = getattr(band, "__file__", None) or "(namespace package)"
print(f"band-sdk OK:    {origin}")
PY

# --- 3. Refuse a pip shadow ----------------------------------------------------
# Entry-point plugins override directory plugins on name collision in the host
# loader, so a leftover pip install in the gateway venv would silently keep the
# OLD code running while this script reports success. Fail until it is removed
# (BAND_UNINSTALL_PIP=1 removes it here — a venv that holds a pip copy is by
# definition writable). With no pip dist present, the directory copy is provably
# the one the loader uses. Runs after verification so a later failure can't
# leave the box with neither install.
# Two names are checked: `hermes-band` (current) and `hermes-band-platform`
# (pre-rename — gateways set up from the old docs still carry it).
# -I (isolated): consult only the gateway venv's own site-packages — a plain
# -c run from a repo checkout would see its hermes_band_platform.egg-info via
# cwd on sys.path and report a phantom pip install.
# Reports EVERY match, not just the first: a gateway can carry both (installed
# from the old docs, then re-installed from the new ones), and removing one while
# the other still shadows would leave exactly the silent old-code state this
# guard exists to prevent.
pip_shadows="$("$HERMES_PY" -I -c 'import importlib.metadata as m
for name in ("hermes-band", "hermes-band-platform"):
    try:
        m.distribution(name)
    except m.PackageNotFoundError:
        continue
    print(name)' 2>/dev/null | tr "\n" " " | sed "s/ *$//" || true)"
if [ -n "$pip_shadows" ]; then
  if [ "${BAND_UNINSTALL_PIP:-}" = "1" ]; then
    echo "removing pip-installed $pip_shadows from the gateway venv (BAND_UNINSTALL_PIP=1)"
    # Unquoted on purpose: each name must arrive as its own argument.
    # shellcheck disable=SC2086
    uv pip uninstall --python "$HERMES_PY" $pip_shadows \
      || die "could not uninstall the pip copy; remove it manually and re-run"
  else
    # $pip_shadows holds one OR two names, so the wording stays count-neutral:
    # the names are a labelled field, never the sentence's subject.
    # Two substrings are load-bearing and must survive any rewording: uppercase
    # OVERRIDE and the literal `uv pip uninstall` — test_installer_refuses_pip_shadow
    # asserts both against the installer's combined output, because they are what
    # proves this refusal (and not some later failure) stopped the run.
    die "the gateway venv still has the Band plugin pip-installed, and an
entry-point install would OVERRIDE a directory install; the OLD code would keep
running. Distribution names: $pip_shadows
Remove them first:
  uv pip uninstall --python \"$HERMES_PY\" $pip_shadows
or re-run with BAND_UNINSTALL_PIP=1 to let the installer remove them."
  fi
fi

# --- 4. Swap the verified candidate into place ---------------------------------
# The only destructive moment: same-filesystem rm+mv after everything fallible
# has already succeeded.
rm -rf "$dest"
mv "$tmp" "$dest"
trap - EXIT
echo "plugin dir:     $dest"

# --- 5. Enable the plugin -------------------------------------------------------
# --no-allow-tool-override keeps enable non-interactive (band does not replace
# built-in tools); re-enabling an enabled plugin is a no-op with rc 0.
hermes plugins enable --no-allow-tool-override band || die "hermes plugins enable band failed"
# grep reads to EOF (no -q) so `set -o pipefail` never sees a SIGPIPE'd hermes.
hermes plugins list 2>/dev/null | grep -w band >/dev/null \
  || die "plugin enabled but not listed by 'hermes plugins list'"

echo
echo "Band directory plugin installed (no site-packages writes)."
echo "Next: restart the gateway, then run:  hermes chat -s band:add-band"
