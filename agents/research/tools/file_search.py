import asyncio
import json
from pathlib import Path
from typing import Any, ClassVar

DEFAULT_DOCS_DIR = Path(__file__).resolve().parents[1] / "sample_docs"
SEARCHABLE_SUFFIXES = frozenset({".md", ".markdown", ".txt", ".rst"})


class FileSearchTool:
    """Grep-like, case-insensitive literal search over text files in a local docs directory."""

    name = "file_search"
    description = (
        "Search the local documentation for a word or phrase. Returns matching lines as "
        "JSON objects with the file path, 1-based line number and a snippet of the line."
    )
    input_schema: ClassVar[dict[str, Any]] = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Text to look for (case-insensitive)."},
            "max_results": {
                "type": "integer",
                "minimum": 1,
                "maximum": 100,
                "description": "Maximum number of matches to return (default 20).",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        docs_dir: str | Path = DEFAULT_DOCS_DIR,
        snippet_chars: int = 200,
        max_file_bytes: int = 1_000_000,
    ) -> None:
        self._docs_dir = Path(docs_dir).resolve()
        self._snippet_chars = snippet_chars
        self._max_file_bytes = max_file_bytes

    async def run(self, args: dict[str, Any]) -> str:
        query = args.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("'query' must be a non-empty string")
        max_results = args.get("max_results", 20)
        if not isinstance(max_results, int) or not 1 <= max_results <= 100:
            raise ValueError("'max_results' must be an integer between 1 and 100")

        matches = await asyncio.to_thread(self._search, query.strip().lower(), max_results)
        return json.dumps(matches)

    def _search(self, needle: str, max_results: int) -> list[dict[str, Any]]:
        if not self._docs_dir.is_dir():
            raise FileNotFoundError(f"Docs directory {self._docs_dir} does not exist")

        matches: list[dict[str, Any]] = []
        for path in sorted(self._docs_dir.rglob("*")):
            if not self._searchable(path):
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for number, line in enumerate(text.splitlines(), start=1):
                if needle in line.lower():
                    matches.append(
                        {
                            "file": path.relative_to(self._docs_dir).as_posix(),
                            "line": number,
                            "snippet": line.strip()[: self._snippet_chars],
                        }
                    )
                    if len(matches) >= max_results:
                        return matches
        return matches

    def _searchable(self, path: Path) -> bool:
        return (
            path.suffix.lower() in SEARCHABLE_SUFFIXES
            and path.is_file()
            # Skip symlinks that point outside the docs directory.
            and path.resolve().is_relative_to(self._docs_dir)
            and path.stat().st_size <= self._max_file_bytes
        )
