# Production Ready Claude Agent

> **Unofficial community project. Not affiliated with, endorsed by, or sponsored by Anthropic.**

A production-ready Claude Agent SDK template you can clone, run with one command, and ship. Instead of stitching together infrastructure around a demo agent, you start from a working foundation that already handles the hard parts of running agents in production, so you can focus on what your agent actually does.

## What's included

- **Queue and workers**: background job processing for agent tasks
- **Retries**: automatic retry handling for transient failures
- **Cost caps**: spending limits to keep usage under control
- **OpenTelemetry tracing**: end-to-end visibility into agent runs
- **Evals**: a harness for measuring agent quality
- **Permissions**: control over what tools and actions the agent can use
- **Docker Compose**: the whole stack runs locally with one command

## Setup

```bash
cp .env.example .env        # then set ANTHROPIC_API_KEY
docker compose up -d        # Postgres, Redis and Jaeger
uv sync                     # install dependencies
make migrate                # create the database schema (alembic upgrade head)
```

Run `make migrate` again whenever you pull new migrations. The Postgres integration tests run
only when `TEST_DATABASE_URL` is set, for example
`TEST_DATABASE_URL=postgresql://agent:agent@localhost:5432/agent uv run pytest`.
