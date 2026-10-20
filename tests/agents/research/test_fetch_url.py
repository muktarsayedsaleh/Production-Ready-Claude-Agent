"""fetch_url tests against a local HTTP server on 127.0.0.1, so no network access is needed."""

import asyncio
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, ClassVar
from urllib.error import HTTPError

import pytest

from agents.research.tools import FetchUrlTool, html_to_text

PAGE = """<!doctype html>
<html><head><title>Backoff</title><style>body { color: red }</style>
<script>alert("hidden")</script></head>
<body><h1>Exponential   backoff</h1><p>Delays double &amp; are capped.</p>
<ul><li>1s</li><li>2s</li></ul></body></html>"""


class Handler(BaseHTTPRequestHandler):
    routes: ClassVar[dict[str, tuple[int, str, bytes]]] = {}

    def do_GET(self) -> None:  # http.server API
        if self.path == "/slow":
            time.sleep(1)
        status, content_type, body = self.routes.get(self.path, (404, "text/plain", b"missing"))
        self.send_response(status)
        if status in (301, 302):
            self.send_header("Location", body.decode())
            body = b""
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args: Any) -> None:
        pass


@pytest.fixture(scope="module")
def base_url() -> Iterator[str]:
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    url = f"http://127.0.0.1:{server.server_address[1]}"
    Handler.routes = {
        "/page": (200, "text/html; charset=utf-8", PAGE.encode()),
        "/plain": (200, "text/plain", b"x" * 100),
        "/slow": (200, "text/plain", b"late"),
        "/binary": (200, "application/pdf", b"%PDF"),
        "/redirect-ok": (302, "text/plain", f"{url}/plain".encode()),
        "/redirect-away": (302, "text/plain", b"http://evil.example.net/steal"),
    }
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield url
    server.shutdown()


def fetch(url: str, **options: Any) -> str:
    tool = FetchUrlTool(allowed_domains=["127.0.0.1"], **options)
    return asyncio.run(tool.run({"url": url}))


def test_strips_html_to_text(base_url: str) -> None:
    assert fetch(f"{base_url}/page") == (
        f"URL: {base_url}/page\n\nBackoff\nExponential backoff\nDelays double & are capped.\n1s\n2s"
    )


def test_truncates_to_max_chars(base_url: str) -> None:
    assert fetch(f"{base_url}/plain", max_chars=10) == (
        f"URL: {base_url}/plain\n\n{'x' * 10}\n\n[truncated to 10 characters]"
    )


def test_follows_redirects_within_the_allowlist(base_url: str) -> None:
    assert fetch(f"{base_url}/redirect-ok").startswith(f"URL: {base_url}/plain\n\nxxx")


def test_refuses_redirects_off_the_allowlist(base_url: str) -> None:
    with pytest.raises(PermissionError, match="evil.example.net"):
        fetch(f"{base_url}/redirect-away")


def test_refuses_domains_not_on_the_allowlist() -> None:
    with pytest.raises(PermissionError, match="not on the allowed domains"):
        fetch("https://example.com/")


@pytest.mark.parametrize("url", ["file:///etc/passwd", "ftp://127.0.0.1/x", "127.0.0.1/page"])
def test_refuses_non_http_urls(url: str) -> None:
    with pytest.raises(PermissionError):
        fetch(url)


def test_times_out(base_url: str) -> None:
    with pytest.raises(TimeoutError, match="timed out after 0.2s"):
        fetch(f"{base_url}/slow", timeout_seconds=0.2)


def test_rejects_non_text_content(base_url: str) -> None:
    with pytest.raises(ValueError, match="Unsupported content type 'application/pdf'"):
        fetch(f"{base_url}/binary")


def test_http_errors_propagate(base_url: str) -> None:
    with pytest.raises(HTTPError, match="404"):
        fetch(f"{base_url}/nope")


def test_html_to_text_skips_scripts_and_styles() -> None:
    assert html_to_text("<p>a</p><script>b</script><style>c</style><div>d<br>e</div>") == (
        "a\nd\ne"
    )
