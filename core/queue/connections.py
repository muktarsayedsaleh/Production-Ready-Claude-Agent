import os

from redis.asyncio import Redis

DEFAULT_REDIS_URL = "redis://localhost:6379/0"

# JobQueue.pop blocks in BRPOP for up to this long waiting for a job.
POP_TIMEOUT_SECONDS = 5.0
# Must stay well above the BRPOP timeout: redis-py (8.x defaults to 5s) raises TimeoutError
# when a read takes longer than socket_timeout, even though an idle BRPOP is not an error.
# It is still finite, so a silently dropped connection is noticed instead of hanging forever.
SOCKET_TIMEOUT_SECONDS = POP_TIMEOUT_SECONDS * 3


def create_redis(url: str | None = None) -> Redis:
    """A Redis client whose timeouts are safe for blocking queue reads."""
    return Redis.from_url(
        url or os.environ.get("REDIS_URL", DEFAULT_REDIS_URL),
        socket_timeout=SOCKET_TIMEOUT_SECONDS,
        socket_connect_timeout=5,
        health_check_interval=30,
    )
