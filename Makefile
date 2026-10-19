.DEFAULT_GOAL := help
.PHONY: help setup sync docker-build check-uv check-docker

help:
	@printf '%s\n' 'Jaull setup (GNU Make)' '  make setup        Sync locked Python environment and build evaluator' '  make sync         uv sync --locked only' '  make docker-build Build jaull-quality-eval:gguf-v1 only' '  make help         Show this help';

setup: sync docker-build

check-uv:
	@command -v uv >/dev/null 2>&1 || { printf '%s\n' 'Error: uv is required. Install uv and add it to PATH.' >&2; exit 1; }

sync: check-uv
	uv sync --locked

check-docker:
	@command -v docker >/dev/null 2>&1 || { printf '%s\n' 'Error: Docker CLI is required.' >&2; exit 1; }
	@docker info >/dev/null 2>&1 || { printf '%s\n' 'Error: Docker daemon is unavailable. Start Docker and check permissions.' >&2; exit 1; }

docker-build: check-docker
	docker build --platform linux/amd64 --tag jaull-quality-eval:gguf-v1 pilot/quality_eval
