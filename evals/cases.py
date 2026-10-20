import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

CASES_DIR = Path(__file__).with_name("cases")

_CASE_KEYS = {
    "description",
    "question",
    "expected_facts",
    "forbidden_tools",
    "max_cost_usd",
    "fake_llm",
}


@dataclass(frozen=True)
class Fact:
    """Text that must appear in the answer: a case-insensitive substring or regex."""

    pattern: str
    regex: bool = False

    def matches(self, text: str) -> bool:
        if self.regex:
            return re.search(self.pattern, text, re.IGNORECASE) is not None
        return self.pattern.lower() in text.lower()

    def __str__(self) -> str:
        return f"/{self.pattern}/" if self.regex else repr(self.pattern)


@dataclass(frozen=True)
class EvalCase:
    name: str
    question: str
    facts: tuple[Fact, ...]
    forbidden_tools: tuple[str, ...]
    max_cost_usd: float
    # Scripted model turns for `--llm fake`; see evals/fake_llm.py.
    fake_llm: tuple[Mapping[str, Any], ...] | None = None
    description: str = ""


def load_cases(directory: Path = CASES_DIR) -> list[EvalCase]:
    return [load_case(path) for path in sorted(directory.glob("*.yaml"))]


def load_case(path: Path) -> EvalCase:
    with open(path) as f:
        data = yaml.safe_load(f)
    try:
        return _parse(path.stem, data)
    except ValueError as exc:
        raise ValueError(f"Invalid eval case {path}: {exc}") from None


def _parse(name: str, data: Any) -> EvalCase:
    if not isinstance(data, Mapping):
        raise ValueError("expected a mapping at the top level")  # noqa: TRY004 - config errors are ValueError
    unknown = set(data) - _CASE_KEYS
    if unknown:
        raise ValueError(f"unknown key(s) {', '.join(sorted(unknown))}")

    question = data.get("question")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("'question' must be a non-empty string")

    facts = data.get("expected_facts")
    if not isinstance(facts, list) or not facts:
        raise ValueError("'expected_facts' must be a non-empty list")

    forbidden = data.get("forbidden_tools", [])
    if not isinstance(forbidden, list) or not all(isinstance(t, str) for t in forbidden):
        raise ValueError("'forbidden_tools' must be a list of tool names")

    max_cost = data.get("max_cost_usd")
    if not isinstance(max_cost, int | float) or isinstance(max_cost, bool) or max_cost <= 0:
        raise ValueError("'max_cost_usd' must be a positive number")

    fake = data.get("fake_llm")
    if fake is not None and (
        not isinstance(fake, list) or not all(isinstance(turn, Mapping) for turn in fake)
    ):
        raise ValueError("'fake_llm' must be a list of turns")

    return EvalCase(
        name=name,
        question=question.strip(),
        facts=tuple(_parse_fact(fact) for fact in facts),
        forbidden_tools=tuple(forbidden),
        max_cost_usd=float(max_cost),
        fake_llm=tuple(fake) if fake is not None else None,
        description=str(data.get("description", "")).strip(),
    )


def _parse_fact(raw: Any) -> Fact:
    if isinstance(raw, str) and raw:
        return Fact(raw)
    if isinstance(raw, Mapping) and set(raw) == {"regex"} and isinstance(raw["regex"], str):
        try:
            re.compile(raw["regex"])
        except re.error as exc:
            raise ValueError(f"bad regex {raw['regex']!r}: {exc}") from None
        return Fact(raw["regex"], regex=True)
    raise ValueError(f"each expected fact is a string or {{regex: ...}}, got {raw!r}")
