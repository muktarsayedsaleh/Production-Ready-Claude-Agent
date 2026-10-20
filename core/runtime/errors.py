from core.llm import ToolCall


class AgentError(Exception):
    """Base class for errors that abort an agent task."""


class StepLimitExceeded(AgentError):
    def __init__(self, max_steps: int, cost_usd: float) -> None:
        self.max_steps = max_steps
        self.cost_usd = cost_usd
        super().__init__(f"Task did not finish within {max_steps} steps (spent ${cost_usd:.4f})")


class CostCapExceeded(AgentError):
    def __init__(self, cost_usd: float, cap_usd: float, steps: int) -> None:
        self.cost_usd = cost_usd
        self.cap_usd = cap_usd
        self.steps = steps
        super().__init__(
            f"Task exceeded its cost cap: spent ${cost_usd:.4f} against a cap of "
            f"${cap_usd:.4f} after {steps} LLM call(s)"
        )


class ApprovalRequired(Exception):
    """The task paused because tool calls need a reviewer's decision. Not a failure."""

    def __init__(self, idempotency_key: str, pending: list[ToolCall]) -> None:
        self.idempotency_key = idempotency_key
        self.pending = pending
        calls = ", ".join(f"{call.name} ({call.id})" for call in pending)
        super().__init__(f"Task {idempotency_key} is waiting for approval of: {calls}")
