"""Public Streamlit deployment smoke check with deployment-aware polling and diagnostics.

The check is intentionally separate from unit tests: it verifies the public hosted UI,
but does not submit forms, upload private data, or authenticate to any provider.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from urllib.parse import urlparse


EXPECTED_MARKER = "QUALITY MOMENTUM LAB"
EXPECTED_SCREENER_HEADING = "Screener a podkladové metriky"
# Do not spend the full six-minute deployment timeout when the host repeatedly
# serves its generic shell but never executes the app script.
HOST_SHELL_MAX_CONSECUTIVE_ATTEMPTS = 6


def validate_app_url(value: str) -> str:
    url = value.strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme != "https" or not parsed.hostname:
        raise ValueError("App URL must be a complete HTTPS URL.")
    if not parsed.hostname.endswith(".streamlit.app"):
        raise ValueError("For safety, public deployment smoke checks only allow *.streamlit.app hosts.")
    if parsed.username or parsed.password:
        raise ValueError("Credentials must not be embedded in the app URL.")
    if parsed.query or parsed.fragment:
        raise ValueError("App URL must not contain query parameters or fragments.")
    return url


def expected_build_marker() -> str:
    """Read the build marker from the same checked-out app that CI is validating."""
    app_file = Path("app.py")
    if app_file.exists():
        source = app_file.read_text(encoding="utf-8")
        match = re.search(r'^APP_VERSION\s*=\s*["\']([^"\']+)["\']', source, re.MULTILINE)
        if match:
            return f"build {match.group(1)}"
    raise RuntimeError("Could not determine APP_VERSION from the checked-out app.py.")


def host_shell_retry_exhausted(consecutive_attempts: int) -> bool:
    """Return whether repeated generic host-shell responses should fail the smoke check."""
    return consecutive_attempts >= HOST_SHELL_MAX_CONSECUTIVE_ATTEMPTS


def classify_page_state(body: str, build_marker: str) -> str:
    """Classify visible app text so a generic HTTP-200 shell is not mistaken for a healthy app."""
    normalized = " ".join(str(body or "").split())
    upper = normalized.upper()
    if not normalized:
        return "empty"
    if "THIS APP HAS ENCOUNTERED AN ERROR" in upper:
        return "application_error"
    if EXPECTED_MARKER in upper and build_marker.lower() in normalized.lower():
        return "current_build"
    if EXPECTED_MARKER in upper:
        return "stale_build"
    if "HOSTED WITH STREAMLIT" in upper and "CREATED BY" in upper:
        return "host_shell_only"
    return "unknown"


def run_smoke(url: str, timeout_ms: int = 360_000, poll_interval_ms: int = 15_000) -> None:
    base_url = validate_app_url(url)
    build_marker = expected_build_marker()
    if timeout_ms < 30_000 or poll_interval_ms < 1_000:
        raise ValueError("Use timeout_ms >= 30000 and poll_interval_ms >= 1000.")
    try:
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise RuntimeError(
            "Playwright is required for the hosted browser smoke check; install it with 'pip install playwright' "
            "and run 'python -m playwright install chromium'."
        ) from exc

    diagnostics: dict[str, list[str]] = {"console_errors": [], "page_errors": [], "failed_requests": []}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        page = None
        response_status = None
        last_body_text = ""
        screenshot_path = Path("artifacts/public-app-smoke-failure.png")
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.set_default_timeout(min(30_000, timeout_ms))
            page.on("console", lambda message: diagnostics["console_errors"].append(message.text) if message.type == "error" else None)
            page.on("pageerror", lambda error: diagnostics["page_errors"].append(str(error)))
            page.on("requestfailed", lambda request: diagnostics["failed_requests"].append(
                f"{request.method} {request.url} :: {request.failure}"
            ))

            deadline = time.monotonic() + timeout_ms / 1000
            attempts = 0
            rendered_current_build = False
            consecutive_host_shell_only = 0
            # Streamlit Community Cloud can take time to rebuild after a Git push.
            # A single-page wait would stay on an old build forever even after the host
            # becomes current, so actively reload and poll until the expected version appears.
            while time.monotonic() < deadline:
                attempts += 1
                remaining_ms = max(1_000, int((deadline - time.monotonic()) * 1000))
                try:
                    response = page.goto(
                        base_url,
                        wait_until="domcontentloaded",
                        timeout=min(45_000, remaining_ms),
                    )
                    if response is None:
                        raise RuntimeError("The hosted app returned no top-level HTTP response.")
                    response_status = response.status
                    if response.status >= 500:
                        last_body_text = f"Host returned HTTP {response.status}."
                    elif response.status >= 400:
                        raise RuntimeError(f"The hosted app returned HTTP {response.status}.")
                    else:
                        # Streamlit may still be waking/restarting. Capture current body
                        # text on each navigation and continue polling if the build is old.
                        try:
                            last_body_text = page.locator("body").inner_text(timeout=min(10_000, remaining_ms))
                        except Exception as body_exc:
                            last_body_text = f"<body text unavailable: {type(body_exc).__name__}: {body_exc}>"
                        state = classify_page_state(last_body_text, build_marker)
                        visible_error = page.get_by_text("This app has encountered an error", exact=False)
                        if visible_error.count() > 0 and visible_error.first.is_visible():
                            # Let a later retry recover from a rebuild transition, but keep
                            # the visible error in the final diagnostics if it persists.
                            last_body_text = "STREAMLIT APPLICATION ERROR\n" + last_body_text
                            consecutive_host_shell_only = 0
                        elif state == "current_build":
                            rendered_current_build = True
                            break
                        elif state == "host_shell_only":
                            consecutive_host_shell_only += 1
                            if host_shell_retry_exhausted(consecutive_host_shell_only):
                                raise RuntimeError(
                                    f"Public host returned HTTP {response_status} but served only the generic "
                                    f"Streamlit shell for {consecutive_host_shell_only} consecutive reload attempts. "
                                    "The app content did not render; inspect Streamlit Community Cloud build/runtime "
                                    "logs and restart or redeploy the app. HTTP 200 alone is not a healthy deployment."
                                )
                        else:
                            consecutive_host_shell_only = 0
                except PlaywrightTimeoutError as nav_exc:
                    diagnostics["failed_requests"].append(f"navigation attempt {attempts}: {nav_exc}")
                if time.monotonic() >= deadline:
                    break
                page.wait_for_timeout(min(poll_interval_ms, max(1_000, int((deadline - time.monotonic()) * 1000))))

            if not rendered_current_build:
                raise PlaywrightTimeoutError(
                    f"Expected public build marker {build_marker!r} was not rendered after {attempts} reload attempts."
                )

            page.set_default_timeout(min(45_000, max(10_000, int((deadline - time.monotonic()) * 1000))))
            page.get_by_role("tab", name="Screener").click()
            page.get_by_text(EXPECTED_SCREENER_HEADING, exact=True).wait_for(state="visible", timeout=min(45_000, max(10_000, int((deadline - time.monotonic()) * 1000))))

            visible_error = page.get_by_text("This app has encountered an error", exact=False)
            if visible_error.count() > 0 and visible_error.first.is_visible():
                raise RuntimeError("Streamlit displayed its application-error message.")

            print(f"PASS: HTTP {response_status}; {build_marker}; public app shell and Screener tab rendered at {base_url}; reload attempts={attempts}")
        except Exception as exc:
            if page is not None:
                screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                try:
                    page.screenshot(path=str(screenshot_path), full_page=True, timeout=20_000)
                    print(f"Failure screenshot: {screenshot_path}", file=sys.stderr)
                except Exception as screenshot_exc:
                    print(f"Could not save screenshot: {type(screenshot_exc).__name__}: {screenshot_exc}", file=sys.stderr)
                try:
                    body_text = page.locator("body").inner_text(timeout=10_000)
                except Exception as body_exc:
                    body_text = f"<body text unavailable: {type(body_exc).__name__}: {body_exc}>"
                print(f"Expected build marker: {build_marker}", file=sys.stderr)
                print(f"URL at failure: {page.url}", file=sys.stderr)
                print(f"Page title: {page.title()!r}", file=sys.stderr)
                print(f"Top-level HTTP status: {response_status}", file=sys.stderr)
                print("Visible page text (first 12,000 chars):", file=sys.stderr)
                print((body_text or last_body_text)[:12_000], file=sys.stderr)
                for key, values in diagnostics.items():
                    print(f"{key} ({len(values)}):", file=sys.stderr)
                    for value in values[-20:]:
                        print(f"  {value[:2000]}", file=sys.stderr)
            raise
        finally:
            browser.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Public Streamlit app URL")
    parser.add_argument("--timeout-ms", type=int, default=360_000)
    parser.add_argument("--poll-interval-ms", type=int, default=15_000)
    args = parser.parse_args()
    try:
        run_smoke(args.url, timeout_ms=args.timeout_ms, poll_interval_ms=args.poll_interval_ms)
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
