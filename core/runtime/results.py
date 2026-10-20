import json
from dataclasses import asdict, dataclass
from typing import Protocol

from redis.asyncio import Redis


@dataclass(frozen=True)
class TaskResult:
    answer: str
    steps: int
    cost_usd: float
    input_tokens: int
    output_tokens: int


class ResultStore(Protocol):
    """Completed results keyed by idempotency key, so a re-run doesn't pay twice."""

    async def get(self, key: str) -> TaskResult | None: ...

    async def save(self, key: str, result: TaskResult) -> None: ...


class RedisResultStore:
    def __init__(self, redis: Redis, ttl_seconds: int = 7 * 24 * 3600) -> None:
        self._redis = redis
        self._ttl = ttl_seconds

    async def get(self, key: str) -> TaskResult | None:
        raw = await self._redis.get(self._key(key))
        return None if raw is None else TaskResult(**json.loads(raw))

    async def save(self, key: str, result: TaskResult) -> None:
        await self._redis.set(self._key(key), json.dumps(asdict(result)), ex=self._ttl)

    @staticmethod
    def _key(key: str) -> str:
        return f"agent:result:{key}"
