# Production Ready Claude Agent

> **Unofficial community project. Not affiliated with, endorsed by, or sponsored by Anthropic.**

A production-ready Claude agent template you can clone, run with one command, and ship.

Most agent examples stop at the demo: a loop, a few tools, a print statement. This repo starts where those stop. The agent runs behind a queue, retries transient failures, stops when it hits its cost cap, emits OpenTelemetry traces, checks a permission policy before every tool call, and ships with a regression harness and CI. All of it runs locally with Docker Compose.

![Jaeger trace of one agent task](docs/images/jaeger-trace.png)

One task, four model calls, four tool calls, 12 seconds end to end. Almost all of it is waiting on the model. That is the kind of thing you only see with a trace.

## Quick start

You need Docker and an Anthropic API key.

```bash
git clone https://github.com/muktarsayedsaleh/Production-Ready-Claude-Agent.git
cd Production-Ready-Claude-Agent
cp .env.example .env        # put your ANTHROPIC_API_KEY in .env
make up                     # postgres, redis, jaeger, migrations, worker
make demo
```

`make demo` runs the bundled research agent against the sample docs, in-process inside the worker container:

```
Question: How does the retry backoff work, and what is the longest delay between attempts?

The runtime retries only transient errors, such as rate limits (HTTP 429),
overloaded servers (529), timeouts and dropped connections. It doesn't retry
bad requests (400) or authentication errors (401). It uses exponential
backoff, so the delay doubles after each failed attempt (1s, 2s, 4s, ...),
up to a cap of 30 seconds. It also uses full jitter ...

Sources:
  - retries-and-backoff.md

Steps:
  1. file_search(query='backoff')
  2. file_search(query='delay', max_results=50)
  3. file_search(query='retr', max_results=50)
  4. file_search(query='transient')

LLM calls: 4    Total cost: $0.0368
```

Then open Jaeger at http://localhost:16686, pick the `production-ready-claude-agent` service, and look at the trace.

To go through the queue instead, run `docker compose exec worker python -m agents.research "your question"` and watch the worker pick the job up in `make logs`.

Other targets: `make logs`, `make test`, `make lint`, `make evals`, `make down`.

If port 5432 or 6379 is already taken on your machine, set `POSTGRES_PORT` and `REDIS_PORT` in `.env` (for example 55432 and 56379), and update `DATABASE_URL` and `REDIS_URL` there to match.

The Postgres integration tests run only when `TEST_DATABASE_URL` is set, for example `TEST_DATABASE_URL=postgresql://agent:agent@localhost:5432/agent make test`. CI sets it for you.

## What you get

| Layer | Where | What it does |
|---|---|---|
| LLM client | `core/llm/` | One interface, a Claude implementation, per-call cost from a pricing table |
| Queue and workers | `core/queue/` | Redis queue, Postgres job records, a worker you run with `python -m core.queue.worker` |
| Runtime | `core/runtime/` | The agent loop: tool calls, step limit, idempotency keys, retries with backoff, cost cap per task |
| Tracing | `core/tracing/` | OpenTelemetry spans for each task, model call and tool call, exported to Jaeger |
| Permissions | `core/permissions/` | A YAML policy that allows, denies, or requires approval per tool |
| Example agent | `agents/research/` | A multi-step research agent with three local tools and no external API keys |
| Your agent | `agents/your-agent/` | A stub with a TODO, ready to replace the example |
| Evals | `evals/` | Regression cases in YAML and a runner. CI runs them with a scripted model to catch broken tools, policy rules and harness bugs; `make evals EVALS_LLM=real` checks answer quality against the real model |
| Migrations | `migrations/` | Alembic, applied automatically by the `migrate` service |

## Make it yours

1. Copy `agents/research/` to `agents/<your_agent>/`. Use underscores, not hyphens, so Python can import it.
2. Replace `prompt.md` with your system prompt.
3. Replace the tools in `tools/` with yours. A tool is a name, a description, a JSON schema, and a `run()` function.
4. Add your tools to `policies/default.yaml` and decide which ones need a human in the loop.
5. Write two or three cases in `evals/cases/` and run them with `make evals EVALS_LLM=real` before you ship a prompt or model change. CI runs the same cases with a scripted model, which catches broken tools and policy rules but not changes in answer quality.
6. Point the worker at your handler (change the `CMD` in the `Dockerfile`, or set `command:` for the `worker` service in `docker-compose.yml`) and run `make up`.

## Design decisions

**Agents run behind a queue, not behind an HTTP request.**
An agent call can take twenty seconds or two minutes, and nothing in the HTTP path is built for that. The load balancer times out, the client retries, and now the same task is running twice and you are paying for it twice. Putting the task on a queue and answering with a job id is the same idea as async work in any web app, except here the cost of getting it wrong is a model bill, not a slow page.

**Every task has a cost cap.**
A loop that keeps calling tools is the default failure mode of agents, not an edge case. The first runaway loop I saw cost about $200 overnight, for a task that should have cost cents. Companies will not ship something whose monthly cost is open ended, so each task gets a budget in dollars, the runtime adds up every call, and when the budget is gone the task stops with a clear error instead of quietly continuing.

**Schema lives in Alembic migrations, not in CREATE TABLE statements.**
The jobs table will change. Today an approval is just a status value; recording who approved a call, or what each job cost, means new columns, and there will be more after that. A system that serves real users cannot drop and recreate tables, so the schema is versioned in code and applied with `alembic upgrade head`, in CI and in production alike.

**Tracing uses OpenTelemetry, not just logs.**
Logs tell you something happened. A trace shows you the shape of the run: which model calls took the time, which tool was slow, where the retries went. OpenTelemetry is the standard that every backend reads, so the same spans you see in Jaeger on your laptop show up in Datadog, Grafana or Honeycomb in production without changing a line.

**Tools go through a permission policy with an approval gate.**
Models hallucinate, and a hallucinated tool call is an action, not a sentence. The policy file says which tools run freely, which are denied, and which pause the job until a human approves: anything that deletes, sends, or pays goes through the gate. It is cheap to add and it is the difference between an agent you can demo and one you can leave running.

## Roadmap

- Move the runtime onto the Claude Agent SDK, keeping the queue, cost caps, tracing and permission policy as the production layer around it.
- Approval UI (today an approval is a status in the jobs table and a `resume()` call)
- Per-tenant cost budgets on top of the per-task cap
- A second example agent built on this template
- Kubernetes manifests next to the Docker Compose file

## Contributing

Issues and pull requests are welcome. Keep commits small and the tests green: `make lint && make test && make evals`.

## License

MIT. See [LICENSE](LICENSE).
