from pathlib import Path

import pytest

from evals.cases import Fact, load_case, load_cases
from evals.run import main

CASE = """
question: What is the maximum delay?
expected_facts:
  - 30 seconds
  - regex: "doubl(es|ing)"
forbidden_tools: [fetch_url]
max_cost_usd: 0.05
fake_llm:
  - tool_calls:
      - name: file_search
        input: {query: backoff}
    cost_usd: 0.01
  - answer: {answer: "The delay doubles up to 30 seconds.", sources: [retries-and-backoff.md]}
    cost_usd: 0.01
"""


def table_rows(out: str) -> dict[str, list[str]]:
    """Table rows by case name, as [result, steps, cost]."""
    lines = out.split("\n\n")[0].splitlines()[2:]
    return {cells[0]: cells[1:] for cells in (line.split() for line in lines)}


def write_case(directory: Path, name: str, text: str) -> Path:
    directory.mkdir(exist_ok=True)
    path = directory / f"{name}.yaml"
    path.write_text(text)
    return path


def test_shipped_cases_load_and_pass_with_the_fake_llm(capsys: pytest.CaptureFixture[str]) -> None:
    assert len(load_cases()) == 3

    assert main(["--llm", "fake"]) == 0

    out = capsys.readouterr().out
    assert "3/3 passed" in out
    assert table_rows(out)["call_cost"] == ["PASS", "3", "$0.0110"]


def test_passing_case(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    write_case(tmp_path, "ok", CASE)

    assert main(["--llm", "fake", "--cases", str(tmp_path)]) == 0
    assert table_rows(capsys.readouterr().out) == {"ok": ["PASS", "2", "$0.0200"]}


@pytest.mark.parametrize(
    ("change", "failure"),
    [
        (("  - 30 seconds\n", "  - 45 seconds\n"), "missing fact '45 seconds'"),
        (("name: file_search", "name: fetch_url"), "used forbidden tool 'fetch_url'"),
        (("max_cost_usd: 0.05", "max_cost_usd: 0.015"), "CostCapExceeded"),
        (("  - answer:", "  - tool_calls: []\n  - answer_typo:"), "fake_llm turn 2 needs"),
    ],
)
def test_failing_cases(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], change: tuple[str, str], failure: str
) -> None:
    write_case(tmp_path, "bad", CASE.replace(*change))

    assert main(["--llm", "fake", "--cases", str(tmp_path)]) == 1
    out = capsys.readouterr().out
    assert table_rows(out)["bad"][0] == "FAIL"
    assert failure in out
    assert "0/1 passed" in out


def test_case_without_fake_script_fails_in_fake_mode(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_case(tmp_path, "nofake", CASE.split("fake_llm:")[0])

    assert main(["--llm", "fake", "--cases", str(tmp_path)]) == 1
    assert "no fake_llm script in this case" in capsys.readouterr().out


def test_filter_and_empty_or_invalid_cases(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_case(tmp_path, "ok", CASE)

    assert main(["--llm", "fake", "--cases", str(tmp_path), "-k", "nomatch"]) == 2
    write_case(tmp_path, "broken", "question: x\nexpected_facts: []\nmax_cost_usd: 1\n")
    assert main(["--llm", "fake", "--cases", str(tmp_path)]) == 2
    assert "Invalid eval case" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("expected_facts: [x]\nmax_cost_usd: 1\n", "'question'"),
        ("question: q\nexpected_facts: [x]\nmax_cost_usd: 0\n", "positive number"),
        ("question: q\nexpected_facts: [{regex: '('}]\nmax_cost_usd: 1\n", "bad regex"),
        ("question: q\nexpected_facts: [x]\nmax_cost_usd: 1\nextra: 1\n", "unknown key"),
    ],
)
def test_invalid_case_files(tmp_path: Path, text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_case(write_case(tmp_path, "case", text))


def test_fact_matching() -> None:
    assert Fact("Thirty Seconds").matches("waits thirty seconds")
    assert not Fact("30s").matches("30 s")
    assert Fact(r"\b30 ?s\b", regex=True).matches("up to 30 s.")
