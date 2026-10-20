import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from core.llm import TransientLLMError

logger = logging.getLogger(__name__)

Sleep = Callable[[float], Awaitable[None]]


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 4
    base_delay: float = 1.0
    max_delay: float = 30.0
    jitter: bool = True

    def delay(self, attempt: int) -> float:
        """Delay after the given failed attempt (1-based): exponential, capped, full jitter."""
        delay = min(self.max_delay, self.base_delay * 2 ** (attempt - 1))
        return random.uniform(0, delay) if self.jitter else delay


async def with_retries[T](
    call: Callable[[], Awaitable[T]],
    policy: RetryPolicy,
    sleep: Sleep = asyncio.sleep,
) -> T:
    """Run `call`, retrying only TransientLLMError. Re-raises the last error when exhausted."""
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return await call()
        except TransientLLMError as exc:
            if attempt == policy.max_attempts:
                raise
            delay = policy.delay(attempt)
            logger.warning(
                "Transient LLM error (attempt %d/%d), retrying in %.2fs: %s",
                attempt,
                policy.max_attempts,
                delay,
                exc,
            )
            await sleep(delay)
    raise AssertionError("unreachable")
