# Makefile for RHOAI MCP Server
# Podman-first container management and uv development

# =============================================================================
# Configuration
# =============================================================================

IMAGE_NAME ?= rhoai-mcp
IMAGE_TAG ?= latest
FULL_IMAGE := $(IMAGE_NAME):$(IMAGE_TAG)
CONTAINER_NAME ?= rhoai-mcp

# Container runtime detection (prefer podman if available)
CONTAINER_RUNTIME := $(shell command -v podman 2>/dev/null || command -v docker 2>/dev/null)

# Runtime configuration
PORT ?= 8000
KUBECONFIG ?= $(HOME)/.kube/config
LOG_LEVEL ?= INFO

# Build platform (force linux/amd64 for consistent builds across host architectures)
PLATFORM ?= linux/amd64

# Compose command detection (podman compose or docker compose)
COMPOSE_CMD := $(CONTAINER_RUNTIME) compose

# Podman-specific flags for user namespace mapping (allows reading host user files)
# This maps the current user to the container user for file permission compatibility
ifeq ($(findstring podman,$(CONTAINER_RUNTIME)),podman)
    USERNS_FLAGS := --userns=keep-id
    VOLUME_FLAGS := :ro,Z
else
    USERNS_FLAGS :=
    VOLUME_FLAGS := :ro
endif

.PHONY: help build build-no-cache run run-http run-stdio run-dev run-token stop logs shell clean info
.PHONY: dev install sync test lint format check typecheck eval eval-live eval-scenario eval-report eval-compare eval-trend eval-up eval-down

# =============================================================================
# Help
# =============================================================================

help: ## Show this help message
	@echo "RHOAI MCP Server - Development & Container Management"
	@echo ""
	@echo "Detected runtime: $(CONTAINER_RUNTIME)"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Development:"
	@grep -E '^(dev|install|sync|test|lint|format|check|typecheck):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Evaluation:"
	@grep -E '^(eval|eval-live|eval-scenario|eval-report|eval-compare|eval-trend):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "Container:"
	@grep -E '^(build|run|stop|logs|shell|clean|info|test-):.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Development (uv)
# =============================================================================

dev: install ## Setup development environment
	@echo "Development environment ready!"
	@echo "Run 'make test' to run tests"
	@echo "Run 'uv run rhoai-mcp --help' to run the server"

install: ## Install package in development mode
	uv sync

sync: ## Sync dependencies without installing dev packages
	uv sync --no-dev

test: ## Run all tests
	uv run pytest tests/ -v

test-unit: ## Run unit tests only (training domain)
	uv run pytest tests/training -v

test-integration: ## Run integration tests only
	uv run pytest tests/integration -v

lint: ## Run linter (ruff)
	uv run ruff check src/

format: ## Format code (ruff)
	uv run ruff format src/
	uv run ruff check --fix src/

typecheck: ## Run type checker (mypy)
	uv run mypy src/

check: lint typecheck ## Run all checks (lint + typecheck)

eval-up: build ## Start eval services (rhoai-mcp + llama-stack + LCS)
	@# Source .env.eval and derive INFERENCE_API_KEY from RHOAI_EVAL_EVAL_API_KEY if not set
	@if [ -f .env.eval ]; then set -a; . ./.env.eval; set +a; fi; \
	if [ -z "$$INFERENCE_API_KEY" ] && [ -n "$$RHOAI_EVAL_EVAL_API_KEY" ]; then \
		export INFERENCE_API_KEY="$$RHOAI_EVAL_EVAL_API_KEY"; \
	fi; \
	$(COMPOSE_CMD) -f docker-compose.eval.yml up -d
	@echo "Waiting for services to be healthy..."
	@timeout=120; elapsed=0; interval=5; \
	while [ $$elapsed -lt $$timeout ]; do \
		rhoai_ok=$$(curl -sf http://localhost:8000/health 2>/dev/null && echo "yes" || echo "no"); \
		lcs_ok=$$(curl -sf http://localhost:8443/readiness 2>/dev/null && echo "yes" || echo "no"); \
		if [ "$$rhoai_ok" = "yes" ] && [ "$$lcs_ok" = "yes" ]; then \
			echo "All services healthy!"; \
			break; \
		fi; \
		echo "Services not ready (rhoai=$$rhoai_ok, lcs=$$lcs_ok), waiting $${interval}s... ($$elapsed/$${timeout}s)"; \
		sleep $$interval; \
		elapsed=$$((elapsed + interval)); \
	done; \
	if [ "$$rhoai_ok" != "yes" ] || [ "$$lcs_ok" != "yes" ]; then \
		echo "Services did not become healthy within $${timeout}s"; \
		$(COMPOSE_CMD) -f docker-compose.eval.yml logs; \
		exit 1; \
	fi
	@# Register inference model with llama-stack (starter image starts with empty model list)
	@model=$${INFERENCE_MODEL:-gemini-2.5-flash}; \
	echo "Registering model $$model with llama-stack..."; \
	curl -sf -X POST http://localhost:8321/v1/models \
		-H "Content-Type: application/json" \
		-d "{\"model_id\": \"$$model\", \"provider_id\": \"gemini\", \"provider_model_id\": \"$$model\"}" \
		>/dev/null 2>&1 && echo "Model $$model registered." \
		|| echo "Model registration returned non-zero (may already exist)."

eval-down: ## Stop eval services
	$(COMPOSE_CMD) -f docker-compose.eval.yml down -v

eval: ## Run MCP evaluation tests (starts services, runs tests, stops services)
	$(MAKE) eval-up
	uv run --group eval pytest evals/ -v -m "eval and not live" --tb=short --reruns 2 --reruns-delay 5; \
	status=$$?; \
	$(MAKE) eval-down; \
	exit $$status

eval-live: ## Run all MCP evaluation tests including live cluster
	$(MAKE) eval-up
	uv run --group eval pytest evals/ -v -m "eval" --tb=short --reruns 2 --reruns-delay 5; \
	status=$$?; \
	$(MAKE) eval-down; \
	exit $$status

eval-scenario: ## Run a single eval scenario (usage: make eval-scenario SCENARIO=cluster_exploration)
ifndef SCENARIO
	$(error SCENARIO is required. Usage: make eval-scenario SCENARIO=cluster_exploration)
endif
	uv run --group eval pytest evals/scenarios/test_$(SCENARIO).py -v --tb=short

eval-report: ## Show latest eval run summary
	uv run --group eval python -m evals.reporting.cli summary

eval-compare: ## Compare eval scores across providers/models
	uv run --group eval python -m evals.reporting.cli compare

eval-trend: ## Show eval score trends over time
	uv run --group eval python -m evals.reporting.cli trend

# =============================================================================
# Build
# =============================================================================

build: ## Build the container image
	DOCKER_DEFAULT_PLATFORM=$(PLATFORM) $(CONTAINER_RUNTIME) build --platform=$(PLATFORM) -f Containerfile -t $(FULL_IMAGE) .

build-no-cache: ## Build the container image without cache
	DOCKER_DEFAULT_PLATFORM=$(PLATFORM) $(CONTAINER_RUNTIME) build --platform=$(PLATFORM) -f Containerfile --no-cache -t $(FULL_IMAGE) .

# =============================================================================
# Run (Container)
# =============================================================================

run: run-http ## Default: run with HTTP (SSE) transport

run-http: ## Run with HTTP (SSE) transport on port $(PORT)
	$(CONTAINER_RUNTIME) run --rm --name $(CONTAINER_NAME) \
		$(USERNS_FLAGS) \
		-p $(PORT):8000 \
		-v $(KUBECONFIG):/opt/app-root/src/kubeconfig/config$(VOLUME_FLAGS) \
		-e RHOAI_MCP_AUTH_MODE=kubeconfig \
		-e RHOAI_MCP_KUBECONFIG_PATH=/opt/app-root/src/kubeconfig/config \
		-e RHOAI_MCP_LOG_LEVEL=$(LOG_LEVEL) \
		$(FULL_IMAGE) --transport sse

run-streamable: ## Run with streamable-http transport
	$(CONTAINER_RUNTIME) run --rm --name $(CONTAINER_NAME) \
		$(USERNS_FLAGS) \
		-p $(PORT):8000 \
		-v $(KUBECONFIG):/opt/app-root/src/kubeconfig/config$(VOLUME_FLAGS) \
		-e RHOAI_MCP_AUTH_MODE=kubeconfig \
		-e RHOAI_MCP_KUBECONFIG_PATH=/opt/app-root/src/kubeconfig/config \
		-e RHOAI_MCP_LOG_LEVEL=$(LOG_LEVEL) \
		$(FULL_IMAGE) --transport streamable-http

run-stdio: ## Run with STDIO transport (interactive)
	$(CONTAINER_RUNTIME) run --rm -it --name $(CONTAINER_NAME) \
		$(USERNS_FLAGS) \
		-v $(KUBECONFIG):/opt/app-root/src/kubeconfig/config$(VOLUME_FLAGS) \
		-e RHOAI_MCP_AUTH_MODE=kubeconfig \
		-e RHOAI_MCP_KUBECONFIG_PATH=/opt/app-root/src/kubeconfig/config \
		-e RHOAI_MCP_LOG_LEVEL=$(LOG_LEVEL) \
		$(FULL_IMAGE) --transport stdio

run-dev: ## Run with debug logging and dangerous ops enabled
	$(CONTAINER_RUNTIME) run --rm --name $(CONTAINER_NAME) \
		$(USERNS_FLAGS) \
		-p $(PORT):8000 \
		-v $(KUBECONFIG):/opt/app-root/src/kubeconfig/config$(VOLUME_FLAGS) \
		-e RHOAI_MCP_AUTH_MODE=kubeconfig \
		-e RHOAI_MCP_KUBECONFIG_PATH=/opt/app-root/src/kubeconfig/config \
		-e RHOAI_MCP_LOG_LEVEL=DEBUG \
		-e RHOAI_MCP_ENABLE_DANGEROUS_OPERATIONS=true \
		$(FULL_IMAGE) --transport sse

run-token: ## Run with token auth (requires TOKEN and API_SERVER)
ifndef TOKEN
	$(error TOKEN is required. Usage: make run-token TOKEN=<token> API_SERVER=<url>)
endif
ifndef API_SERVER
	$(error API_SERVER is required. Usage: make run-token TOKEN=<token> API_SERVER=<url>)
endif
	$(CONTAINER_RUNTIME) run --rm --name $(CONTAINER_NAME) \
		-p $(PORT):8000 \
		-e RHOAI_MCP_AUTH_MODE=token \
		-e RHOAI_MCP_API_TOKEN=$(TOKEN) \
		-e RHOAI_MCP_API_SERVER=$(API_SERVER) \
		-e RHOAI_MCP_LOG_LEVEL=$(LOG_LEVEL) \
		$(FULL_IMAGE) --transport sse

run-background: ## Run in background (detached) with HTTP transport
	$(CONTAINER_RUNTIME) run -d --name $(CONTAINER_NAME) \
		$(USERNS_FLAGS) \
		-p $(PORT):8000 \
		-v $(KUBECONFIG):/opt/app-root/src/kubeconfig/config$(VOLUME_FLAGS) \
		-e RHOAI_MCP_AUTH_MODE=kubeconfig \
		-e RHOAI_MCP_KUBECONFIG_PATH=/opt/app-root/src/kubeconfig/config \
		-e RHOAI_MCP_LOG_LEVEL=$(LOG_LEVEL) \
		$(FULL_IMAGE) --transport sse

# =============================================================================
# Run (Local Development)
# =============================================================================

run-local: ## Run server locally (not in container)
	uv run rhoai-mcp --transport sse

run-local-stdio: ## Run server locally with stdio transport
	uv run rhoai-mcp --transport stdio

run-local-debug: ## Run server locally with debug logging
	RHOAI_MCP_LOG_LEVEL=DEBUG uv run rhoai-mcp --transport sse

# =============================================================================
# Management
# =============================================================================

stop: ## Stop the running container
	-$(CONTAINER_RUNTIME) stop $(CONTAINER_NAME) 2>/dev/null || true
	-$(CONTAINER_RUNTIME) rm $(CONTAINER_NAME) 2>/dev/null || true

logs: ## View container logs
	$(CONTAINER_RUNTIME) logs -f $(CONTAINER_NAME)

shell: ## Open a shell in the running container
	$(CONTAINER_RUNTIME) exec -it $(CONTAINER_NAME) /bin/bash

clean: stop ## Remove container and image
	-$(CONTAINER_RUNTIME) rmi $(FULL_IMAGE) 2>/dev/null || true

clean-dev: ## Clean development artifacts
	rm -rf .venv
	rm -rf dist
	rm -rf *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache -exec rm -rf {} + 2>/dev/null || true

# =============================================================================
# Testing
# =============================================================================

test-health: ## Test the health endpoint
	@curl -sf http://localhost:$(PORT)/health && echo " OK" || echo "FAILED"

test-build: build ## Verify the image builds and runs
	$(CONTAINER_RUNTIME) run --rm $(FULL_IMAGE) --version

test-plugins: ## Verify all plugins are discovered
	uv run python -c "from rhoai_mcp.server import RHOAIServer; s = RHOAIServer(); print('Plugins:', list(s._plugins.keys()))"

# =============================================================================
# Info
# =============================================================================

info: ## Show configuration
	@echo "IMAGE:     $(FULL_IMAGE)"
	@echo "CONTAINER: $(CONTAINER_NAME)"
	@echo "RUNTIME:   $(CONTAINER_RUNTIME)"
	@echo "PORT:      $(PORT)"
	@echo "KUBECONFIG: $(KUBECONFIG)"
