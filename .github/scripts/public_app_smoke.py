"""Public Streamlit deployment smoke check.

This is intentionally separate from pytest: CI unit tests use deterministic synthetic
market data, while this script checks that the hosted public UI is reachable and
renders the expected initial tabs. It does not submit forms, upload private data,
or authenticate to any provider.
"""
from __future__ import annotations

import argparse
import sys
from urllib.parse import urlparse

from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from playwright.sync_api import sync_playwright


EXPECTED_MARKER = "QUALITY MOMENTUM LAB"
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


def run_smoke(url: str, timeout_ms: int = 120_000) -> None:
    base_url = validate_app_url(url)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        try:
            page = browser.new_page(viewport={"width": 1440, "height": 1000})
            page.set_default_timeout(timeout_ms)
            response = page.goto(base_url, wait_until="domcontentloaded", timeout=timeout_ms)
            if response is None:
                raise RuntimeError("The hosted app returned no top-level HTTP response.")
            if response.status >= 400:
                raise RuntimeError(f"The hosted app returned HTTP {response.status}.")

            page.get_by_text(EXPECTED_MARKER, exact=False).first.wait_for(state="visible", timeout=timeout_ms)
            page.get_by_role("tab", name="Screener").click()
            page.get_by_text(EXPECTED_SCREENER_HEADING, exact=True).wait_for(state="visible", timeout=timeout_ms)

            visible_error = page.get_by_text("This app has encountered an error", exact=False)
            if visible_error.count() > 0 and visible_error.first.is_visible():
                raise RuntimeError("Streamlit displayed its application-error message.")

            print(f"PASS: HTTP {response.status}; public app shell and Screener tab rendered at {base_url}")
        except PlaywrightTimeoutError as exc:
            raise RuntimeError(
                "The deployed app did not render the expected marker/tab before the timeout. "
                "Check the Streamlit Community Cloud logs and data-source startup path."
            ) from exc
        finally:
            browser.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Public Streamlit app URL")
    parser.add_argument("--timeout-ms", type=int, default=120_000)
    args = parser.parse_args()
    try:
        run_smoke(args.url, timeout_ms=args.timeout_ms)
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
