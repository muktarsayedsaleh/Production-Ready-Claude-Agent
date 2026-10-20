import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from agents.research.tools import DEFAULT_DOCS_DIR, FileSearchTool


def search(tool: FileSearchTool, **args: Any) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = json.loads(asyncio.run(tool.run(args)))
    return result


@pytest.fixture
def docs(tmp_path: Path) -> Path:
    (tmp_path / "guide.md").write_text("# Guide\nRetries use exponential BACKOFF.\nDone.\n")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "notes.txt").write_text("first line\nbackoff again\n")
    (tmp_path / "image.png").write_bytes(b"backoff")  # not a text suffix, ignored
    return tmp_path


def test_returns_file_line_and_snippet(docs: Path) -> None:
    assert search(FileSearchTool(docs), query="backoff") == [
        {"file": "guide.md", "line": 2, "snippet": "Retries use exponential BACKOFF."},
        {"file": "nested/notes.txt", "line": 2, "snippet": "backoff again"},
    ]


def test_max_results_and_no_matches(docs: Path) -> None:
    tool = FileSearchTool(docs)

    assert len(search(tool, query="backoff", max_results=1)) == 1
    assert search(tool, query="nothing matches this") == []


def test_snippets_are_truncated(tmp_path: Path) -> None:
    (tmp_path / "long.md").write_text("needle " + "x" * 500)

    [match] = search(FileSearchTool(tmp_path, snippet_chars=20), query="needle")

    assert match["snippet"] == "needle " + "x" * 13


def test_symlinks_outside_the_docs_dir_are_skipped(tmp_path: Path) -> None:
    outside = tmp_path / "secret.md"
    outside.write_text("password: hunter2\n")
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "link.md").symlink_to(outside)

    assert search(FileSearchTool(docs), query="password") == []


def test_default_docs_dir_has_the_sample_docs() -> None:
    matches = search(FileSearchTool(), query="exponential backoff")

    assert DEFAULT_DOCS_DIR.name == "sample_docs"
    assert {m["file"] for m in matches} == {"retries-and-backoff.md"}


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"query": "  "}, "non-empty string"),
        ({"query": 3}, "non-empty string"),
        ({"query": "x", "max_results": 0}, "between 1 and 100"),
    ],
)
def test_rejects_bad_arguments(docs: Path, args: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        asyncio.run(FileSearchTool(docs).run(args))


def test_missing_docs_dir_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        asyncio.run(FileSearchTool(tmp_path / "missing").run({"query": "x"}))
