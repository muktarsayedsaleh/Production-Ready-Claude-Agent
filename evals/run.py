"""Run the research agent against every eval case and report pass/fail.

    python -m evals.run                 # real model (needs ANTHROPIC_API_KEY; costs money)
    python -m evals.run --llm fake      # scripted model, for CI
    python -m evals.run -k backoff      # only cases whose name contains "backoff"

Exits 1 if any case fails and 2 if the cases can't be loaded.
"""

import argparse
import asyncio
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from agents.research.agent import ResearchAgent, ResearchReport, load_policy
from core.llm import ClaudeClient, LLMClient
from core.permissions import Policy
from core.runtime import InMemoryCheckpointStore, TaskResult
from evals.cases import CASES_DIR, EvalCase, load_cases
from evals.fake_llm import ScriptedLLMClient


@dataclass
class CaseResult:
    case: EvalCase
    failures: list[str] = field(default_factory=list)
    llm_calls: int | None = None
    cost_usd: float | None = None
    report: ResearchReport | None = None

    @property
    def passed(self) -> bool:
        return not self.failures


def check(case: EvalCase, report: ResearchReport, result: TaskResult) -> list[str]:
    failures = [f"missing fact {fact}" for fact in case.facts if not fact.matches(report.answer)]
    used = {use.name for use in result.tool_calls}
    failures += [f"used forbidden tool {name!r}" for name in case.forbidden_tools if name in used]
    if report.cost_usd > case.max_cost_usd:
        failures.append(f"cost ${report.cost_usd:.4f} is over the max ${case.max_cost_usd:.4f}")
    return failures


async def run_case(case: EvalCase, llm: LLMClient, policy: Policy) -> CaseResult:
    # The case's max cost is also the runtime's cost cap, so a runaway case stops early.
    agent = ResearchAgent(
        llm,
        policy=policy,
        checkpoints=InMemoryCheckpointStore(),
        cost_cap_usd=case.max_cost_usd,
    )
    try:
        result = await agent.runtime.run(case.question)
    except Exception as exc:  # noqa: BLE001 - any error fails the case, not the run
        return CaseResult(
            case,
            failures=[f"{type(exc).__name__}: {exc}"],
            cost_usd=getattr(exc, "cost_usd", None),
        )
    report = ResearchReport.from_result(case.question, result)
    return CaseResult(
        case,
        failures=check(case, report, result),
        llm_calls=result.steps,
        cost_usd=result.cost_usd,
        report=report,
    )


async def run_cases(
    cases: list[EvalCase], make_llm: Callable[[EvalCase], LLMClient], policy: Policy
) -> list[CaseResult]:
    results = []
    for case in cases:
        try:
            llm = make_llm(case)
        except ValueError as exc:
            results.append(CaseResult(case, failures=[str(exc)]))
            continue
        results.append(await run_case(case, llm, policy))
    return results


def fake_llm(case: EvalCase) -> LLMClient:
    if case.fake_llm is None:
        raise ValueError("no fake_llm script in this case")
    return ScriptedLLMClient(case.fake_llm)


def format_table(results: list[CaseResult]) -> str:
    rows = [("case", "result", "steps", "cost")]
    for r in results:
        rows.append(
            (
                r.case.name,
                "PASS" if r.passed else "FAIL",
                "-" if r.llm_calls is None else str(r.llm_calls),
                "-" if r.cost_usd is None else f"${r.cost_usd:.4f}",
            )
        )
    widths = [max(len(row[i]) for row in rows) for i in range(4)]
    lines = [
        "  ".join(
            cell.rjust(width) if i >= 2 else cell.ljust(width)
            for i, (cell, width) in enumerate(zip(row, widths, strict=True))
        ).rstrip()
        for row in rows
    ]
    lines.insert(1, "  ".join("-" * width for width in widths))

    failed = [r for r in results if not r.passed]
    for r in failed:
        lines += ["", f"{r.case.name}:"] + [f"  - {failure}" for failure in r.failures]
    lines += ["", f"{len(results) - len(failed)}/{len(results)} passed"]
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the research agent eval cases.")
    parser.add_argument("--llm", choices=["real", "fake"], default="real")
    parser.add_argument("--cases", type=Path, default=CASES_DIR, help="directory of *.yaml cases")
    parser.add_argument("-k", dest="name_filter", help="only cases whose name contains this")
    parser.add_argument("-v", "--verbose", action="store_true", help="show agent logs")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.CRITICAL)
    try:
        cases = load_cases(args.cases)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    if args.name_filter:
        cases = [case for case in cases if args.name_filter in case.name]
    if not cases:
        print(f"No eval cases found in {args.cases}", file=sys.stderr)
        return 2

    if args.llm == "fake":
        make_llm = fake_llm
    else:
        client = ClaudeClient()
        make_llm = lambda case: client

    results = asyncio.run(run_cases(cases, make_llm, load_policy()))
    print(format_table(results))
    return 0 if all(r.passed for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
