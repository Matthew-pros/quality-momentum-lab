from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / ".github" / "scripts" / "public_app_smoke.py"
SPEC = importlib.util.spec_from_file_location("public_app_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_deployment_smoke_accepts_public_streamlit_https_url():
    assert MODULE.validate_app_url("https://example.streamlit.app/") == "https://example.streamlit.app"


@pytest.mark.parametrize(
    "url",
    [
        "http://example.streamlit.app",
        "https://streamlit.app",
        "https://example.com",
        "https://user:pass@example.streamlit.app",
        "https://example.streamlit.app/?token=secret",
        "https://example.streamlit.app/#fragment",
        "",
    ],
)
def test_deployment_smoke_rejects_unsafe_or_malformed_urls(url):
    with pytest.raises(ValueError):
        MODULE.validate_app_url(url)


def test_public_smoke_expected_build_matches_app_version(monkeypatch):
    repo_root = Path(__file__).resolve().parents[1]
    monkeypatch.chdir(repo_root)
    app_source = (repo_root / "app.py").read_text(encoding="utf-8")
    version_line = next(
        (line.strip() for line in app_source.splitlines() if line.strip().startswith("APP_VERSION = ")),
        None,
    )
    assert version_line is not None
    version = version_line.split("=", 1)[1].strip().strip('"').strip("'")
    assert MODULE.expected_build_marker() == f"build {version}"


def test_public_smoke_marker_polling_script_has_bounded_timeout():
    import inspect

    signature = inspect.signature(MODULE.run_smoke)
    assert signature.parameters["timeout_ms"].default >= 180_000
    assert signature.parameters["poll_interval_ms"].default >= 1_000
