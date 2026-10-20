# Load .env (if present) so DATABASE_URL and friends reach the commands below.
-include .env
export

DEMO_QUESTION ?= How does the retry backoff work, and what is the longest delay between attempts?
EVALS_LLM ?= fake

.PHONY: up down logs demo migrate test lint evals

up: ## Build the app image and start the whole stack in the background
	docker compose up --build --detach
	@echo "Stack is up. Jaeger UI: http://localhost:$${JAEGER_UI_PORT:-16686}"

down: ## Stop the stack (data in the postgres volume is kept)
	docker compose down

logs: ## Follow logs from all services
	docker compose logs --follow

demo: ## Ask the research agent a sample question inside the worker container
	@echo "Waiting for the worker to be running and healthy..."
	@for i in $$(seq 1 60); do \
		state=$$(docker compose ps worker --format '{{.State}} {{.Health}}' 2>/dev/null); \
		[ "$$state" = "running healthy" ] && break; \
		if [ $$i -eq 60 ]; then \
			echo "Worker not healthy after 120s (state: $${state:-not running}). Try 'make up' or 'make logs'."; \
			exit 1; \
		fi; \
		sleep 2; \
	done
	docker compose exec worker python -m agents.research --sync "$(DEMO_QUESTION)"

migrate: ## Apply database migrations from this machine
	uv run alembic upgrade head

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy core agents evals migrations

evals: ## Run the eval cases (EVALS_LLM=real to use the real model; costs money)
	uv run python -m evals.run --llm $(EVALS_LLM)
