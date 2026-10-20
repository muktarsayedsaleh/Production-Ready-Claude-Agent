from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class JobStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


@dataclass(frozen=True)
class Job:
    id: str
    payload: dict[str, Any]
    status: JobStatus
    attempts: int = 0
    last_error: str | None = None


class JobPaused(Exception):
    """Raised by a handler to park the job as waiting_approval instead of failing it."""
