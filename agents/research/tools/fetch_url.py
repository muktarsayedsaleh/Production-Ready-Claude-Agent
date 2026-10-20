import asyncio
import urllib.request
from collections.abc import Sequence
from email.message import Message
from html.parser import HTMLParser
from http.client import HTTPMessage
from typing import IO, Any, ClassVar
from urllib.error import URLError

from core.permissions import url_allowed

USER_AGENT = "production-ready-claude-agent/0.1 (+fetch_url tool)"
TEXT_TYPES = frozenset({"text/html", "application/xhtml+xml", "text/plain", "text/markdown"})
HTML_TYPES = frozenset({"text/html", "application/xhtml+xml"})


class FetchUrlTool:
    """GET a public http(s) page and return its text, restricted to an allowlist of domains.

    The policy checks the requested URL; this tool also checks every redirect, so an allowed
    page cannot bounce the request to a domain that isn't.
    """

    name = "fetch_url"
    description = (
        "Fetch a web page over http(s) and return its readable text (HTML is stripped). "
        "Only some domains are allowed; long pages are truncated."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {"url": {"type": "string", "description": "The http(s) URL to fetch."}},
        "required": ["url"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        allowed_domains: Sequence[str],
        max_chars: int = 8000,
        timeout_seconds: float = 10.0,
        max_bytes: int = 2_000_000,
    ) -> None:
        self._allowed_domains = tuple(allowed_domains)
        self._max_chars = max_chars
        self._timeout = timeout_seconds
        self._max_bytes = max_bytes
        self._opener = urllib.request.build_opener(_AllowlistRedirectHandler(self._check_url))

    async def run(self, args: dict[str, Any]) -> str:
        url = args.get("url")
        if not isinstance(url, str) or not url.strip():
            raise ValueError("'url' must be a non-empty string")
        return await asyncio.to_thread(self._fetch, url.strip())

    def _check_url(self, url: str) -> None:
        if not url_allowed(url, self._allowed_domains):
            raise PermissionError(f"URL {url!r} is not on the allowed domains")

    def _fetch(self, url: str) -> str:
        self._check_url(url)
        request = urllib.request.Request(
            url, headers={"User-Agent": USER_AGENT, "Accept": ", ".join(sorted(TEXT_TYPES))}
        )
        try:
            with self._opener.open(request, timeout=self._timeout) as response:
                headers: Message = response.headers
                content_type = headers.get_content_type()
                if content_type not in TEXT_TYPES:
                    raise ValueError(f"Unsupported content type {content_type!r} from {url}")
                charset = headers.get_content_charset() or "utf-8"
                raw = response.read(self._max_bytes)
                final_url = response.geturl()
        except TimeoutError:
            raise TimeoutError(f"Fetching {url} timed out after {self._timeout}s") from None
        except URLError as exc:
            if isinstance(exc.reason, TimeoutError):
                raise TimeoutError(f"Fetching {url} timed out after {self._timeout}s") from None
            raise

        text = raw.decode(charset, errors="replace")
        if content_type in HTML_TYPES:
            text = html_to_text(text)
        if len(text) > self._max_chars:
            text = text[: self._max_chars] + f"\n\n[truncated to {self._max_chars} characters]"
        return f"URL: {final_url}\n\n{text}"


class _AllowlistRedirectHandler(urllib.request.HTTPRedirectHandler):
    def __init__(self, check_url: Any) -> None:
        self._check_url = check_url

    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: IO[bytes],
        code: int,
        msg: str,
        headers: HTTPMessage,
        newurl: str,
    ) -> urllib.request.Request | None:
        self._check_url(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_SKIP_TAGS = frozenset({"script", "style", "noscript", "template", "svg", "iframe"})
_BLOCK_TAGS = frozenset(
    {
        "address", "article", "aside", "blockquote", "br", "dd", "div", "dl", "dt",
        "figcaption", "footer", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li",
        "main", "nav", "ol", "p", "pre", "section", "table", "td", "th", "title", "tr", "ul",
    }
)  # fmt: skip


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in _SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    """Visible text of an HTML document, one block per line, whitespace collapsed."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
    return "\n".join(line for line in lines if line)
