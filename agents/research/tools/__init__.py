from pathlib import Path

from agents.research.tools.calculator import CalculatorTool
from agents.research.tools.fetch_url import FetchUrlTool, html_to_text
from agents.research.tools.file_search import DEFAULT_DOCS_DIR, FileSearchTool
from core.permissions import Policy
from core.runtime.registry import ToolRegistry


def build_registry(
    policy: Policy,
    docs_dir: str | Path = DEFAULT_DOCS_DIR,
    fetch_max_chars: int = 8000,
    fetch_timeout_seconds: float = 10.0,
) -> ToolRegistry:
    """The research agent's tools. fetch_url's domain allowlist comes from the policy."""
    fetch_rule = policy.tools.get(FetchUrlTool.name)
    return ToolRegistry(
        [
            FileSearchTool(docs_dir),
            CalculatorTool(),
            FetchUrlTool(
                allowed_domains=fetch_rule.allowed_domains if fetch_rule else (),
                max_chars=fetch_max_chars,
                timeout_seconds=fetch_timeout_seconds,
            ),
        ]
    )


__all__ = [
    "DEFAULT_DOCS_DIR",
    "CalculatorTool",
    "FetchUrlTool",
    "FileSearchTool",
    "build_registry",
    "html_to_text",
]
