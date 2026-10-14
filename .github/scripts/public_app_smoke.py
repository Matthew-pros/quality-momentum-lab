"""Public Streamlit deployment smoke check with useful failure diagnostics.

The check is intentionally separate from unit tests: it verifies the public hosted UI,
but does not submit forms, upload private data, or authenticate to any provider.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.parse import urlparse


EXPECTED_MARKER = "QUALITY MOMENTUM LAB"
EXPECTED_BUILD_MARKER = "build 2.7.0"
EXPECTED_SCREENER_HEADING = "Screener a podkladové metriky"


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


def run_smoke(url: str, timeout_ms: int = 180_000) -> None:
    base_url = validate_app_url(url)
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
        screenshot_path = Path("artifacts/public-app-smoke-failure.png")
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.set_default_timeout(timeout_ms)
            page.on("console", lambda message: diagnostics["console_errors"].append(message.text) if message.type == "error" else None)
            page.on("pageerror", lambda error: diagnostics["page_errors"].append(str(error)))
            page.on("requestfailed", lambda request: diagnostics["failed_requests"].append(
                f"{request.method} {request.url} :: {request.failure}"
            ))
            response = page.goto(base_url, wait_until="domcontentloaded", timeout=timeout_ms)
            if response is None:
                raise RuntimeError("The hosted app returned no top-level HTTP response.")
            response_status = response.status
            if response.status >= 400:
                raise RuntimeError(f"The hosted app returned HTTP {response.status}.")

            page.get_by_text(EXPECTED_MARKER, exact=False).first.wait_for(state="visible", timeout=timeout_ms)
            page.get_by_text(EXPECTED_BUILD_MARKER, exact=False).first.wait_for(state="visible", timeout=timeout_ms)
            page.get_by_role("tab", name="Screener").click()
            page.get_by_text(EXPECTED_SCREENER_HEADING, exact=True).wait_for(state="visible", timeout=timeout_ms)

            visible_error = page.get_by_text("This app has encountered an error", exact=False)
            if visible_error.count() > 0 and visible_error.first.is_visible():
                raise RuntimeError("Streamlit displayed its application-error message.")

            print(f"PASS: HTTP {response.status}; {EXPECTED_BUILD_MARKER}; public app shell and Screener tab rendered at {base_url}")
        except Exception as exc:
            # Preserve exact browser diagnostics rather than returning only a generic timeout.
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
                print(f"URL at failure: {page.url}", file=sys.stderr)
                print(f"Page title: {page.title()!r}", file=sys.stderr)
                print(f"Top-level HTTP status: {response_status}", file=sys.stderr)
                print("Visible page text (first 12,000 chars):", file=sys.stderr)
                print(body_text[:12_000], file=sys.stderr)
                for key, values in diagnostics.items():
                    print(f"{key} ({len(values)}):", file=sys.stderr)
                    for value in values[-20:]:
                        print(f"  {value[:2000]}", file=sys.stderr)
            if isinstance(exc, PlaywrightTimeoutError):
                raise RuntimeError(
                    f"Public deployment did not render the expected build/tab within {timeout_ms} ms. "
                    "See page text, browser errors, failed requests and the uploaded screenshot artifact."
                ) from exc
            raise
        finally:
            browser.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Public Streamlit app URL")
    parser.add_argument("--timeout-ms", type=int, default=180_000)
    args = parser.parse_args()
    try:
        run_smoke(args.url, timeout_ms=args.timeout_ms)
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
