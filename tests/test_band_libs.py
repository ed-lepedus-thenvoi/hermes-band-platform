"""Tests for the ``$HERMES_HOME/band-libs`` dependency shim."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from hermes_band_platform import _band_libs


@pytest.fixture()
def clean_sys_path(monkeypatch):
    monkeypatch.setattr(sys, "path", list(sys.path))


def test_hermes_home_defaults_to_dot_hermes(monkeypatch):
    monkeypatch.delenv("HERMES_HOME", raising=False)
    assert _band_libs.hermes_home() == Path.home() / ".hermes"


def test_hermes_home_honors_env_override(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "custom-home"))
    assert _band_libs.hermes_home() == tmp_path / "custom-home"
    assert _band_libs.band_libs_dir() == tmp_path / "custom-home" / "band-libs"


def test_prepend_skips_missing_dir(monkeypatch, tmp_path, clean_sys_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert _band_libs.prepend_band_libs() is None
    assert str(tmp_path / "band-libs") not in sys.path


def test_prepend_puts_band_libs_first_once(monkeypatch, tmp_path, clean_sys_path):
    libs = tmp_path / "band-libs"
    libs.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert _band_libs.prepend_band_libs() == libs
    assert sys.path[0] == str(libs)
    before = list(sys.path)
    assert _band_libs.prepend_band_libs() == libs  # idempotent
    assert sys.path == before


def test_ensure_noops_when_band_already_loaded(monkeypatch, tmp_path, clean_sys_path):
    # conftest guarantees a ``band`` stub is in sys.modules; the shim must not
    # touch sys.path or raise in that case.
    assert "band" in sys.modules
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "band-libs").mkdir()
    before = list(sys.path)
    _band_libs.ensure_band_importable()
    assert sys.path == before


def test_ensure_resolves_band_from_band_libs(monkeypatch, tmp_path, clean_sys_path):
    libs = tmp_path / "band-libs"
    (libs / "band").mkdir(parents=True)
    (libs / "band" / "__init__.py").write_text("BAND_LIBS_MARKER = True\n")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delitem(sys.modules, "band", raising=False)  # auto-restored

    _band_libs.ensure_band_importable()

    spec = importlib.util.find_spec("band")
    assert spec is not None and spec.origin is not None
    assert Path(spec.origin).is_relative_to(libs)


def test_no_unsafe_sdk_install_guidance_in_package():
    """Every band-sdk remediation the plugin can emit must be the
    read-only-venv-safe ``--target`` form (sourced from
    ``_band_libs.sdk_install_command``). A bare install into the gateway
    Python reproduces the hosted-runtime ``Permission denied`` failure."""
    pkg = Path(_band_libs.__file__).parent
    offenders = []
    for py in pkg.rglob("*.py"):
        for lineno, line in enumerate(py.read_text().splitlines(), 1):
            if (
                "pip install" in line
                and "band-sdk" in line.lower()
                and "--target" not in line
            ):
                offenders.append(f"{py.relative_to(pkg)}:{lineno}: {line.strip()}")
    assert not offenders, "unsafe band-sdk install guidance:\n" + "\n".join(offenders)


def test_sdk_install_command_is_target_form():
    assert _band_libs.BAND_SDK_SPEC == "band-sdk>=1.3.0,<2.0.0"
    cmd = _band_libs.sdk_install_command()
    assert cmd.startswith("uv pip install --python")
    assert f'--target "{_band_libs.band_libs_dir()}"' in cmd
    assert _band_libs.BAND_SDK_SPEC in cmd


def test_bootstrap_never_raises_and_logs_the_fix(
    monkeypatch, tmp_path, clean_sys_path, caplog
):
    """Plugin-load-time shim: a missing SDK must NOT abort plugin import —
    that would drop Band from the platform registry and every channel
    surface. It logs the one actionable fix instead."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delitem(sys.modules, "band", raising=False)  # auto-restored
    monkeypatch.setattr(sys, "path", [str(tmp_path / "empty")])

    with caplog.at_level("ERROR", logger="hermes_band_platform._band_libs"):
        error = _band_libs.bootstrap()  # must not raise

    assert error is not None
    assert "uv pip install --python" in error
    assert f'--target "{tmp_path / "band-libs"}"' in error
    assert any("uv pip install" in r.message for r in caplog.records)


def test_bootstrap_returns_none_when_band_resolvable(monkeypatch, tmp_path, clean_sys_path):
    assert "band" in sys.modules  # conftest stub
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert _band_libs.bootstrap() is None


def test_ensure_raises_one_actionable_error(monkeypatch, tmp_path, clean_sys_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.delitem(sys.modules, "band", raising=False)  # auto-restored
    # Point sys.path at an empty tree so ``band`` cannot resolve anywhere.
    monkeypatch.setattr(sys, "path", [str(tmp_path / "empty")])

    with pytest.raises(ImportError) as exc:
        _band_libs.ensure_band_importable()

    msg = str(exc.value)
    # The error must name the exact fix: uv pip --target into band-libs.
    assert "uv pip install --python" in msg
    assert f'--target "{tmp_path / "band-libs"}"' in msg
    assert _band_libs.BAND_SDK_SPEC in msg
    assert "sudo" in msg  # says sudo is NOT needed
