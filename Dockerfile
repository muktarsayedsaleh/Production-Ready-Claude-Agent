FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /usr/local/bin/uv

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_SYSTEM_PYTHON=1 \
    UV_NO_CACHE=1

WORKDIR /app

# Dependencies first, so code changes don't reinstall them.
COPY pyproject.toml README.md LICENSE ./
RUN uv pip install -r pyproject.toml

COPY alembic.ini ./
COPY migrations ./migrations
COPY core ./core
COPY agents ./agents
COPY policies ./policies
COPY evals ./evals

RUN useradd --create-home --uid 10001 app
USER app

CMD ["python", "-m", "core.queue.worker", "--queue", "research", "--handler", "agents.research.agent:handle_job"]
