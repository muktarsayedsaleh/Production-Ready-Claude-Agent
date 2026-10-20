# Load .env (if present) so DATABASE_URL and friends reach the commands below.
-include .env
export

.PHONY: migrate

migrate:
	uv run alembic upgrade head
