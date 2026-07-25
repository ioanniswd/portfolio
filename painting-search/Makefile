# ── Painting-search: local service management ─────────────────────────────────
# The search funnel needs two local services:
#   • Neo4j  (Docker)  — the colour knowledge graph
#   • Ollama (local)   — the LLM that parses a request into a SearchIntent
#
# Quick start:  make up  →  make search Q="warm and earthy"  →  make down
# See status any time with:  make status
# ──────────────────────────────────────────────────────────────────────────────

SHELL        := /bin/bash
OLLAMA_BIN   := $(shell command -v ollama 2>/dev/null || echo $(HOME)/.local/bin/ollama)
OLLAMA_HOST  := 127.0.0.1:11434
OLLAMA_MODEL := llama3.1:8b
NEO4J_NAME   := neo4j
NEO4J_DATA   := $(HOME)/neo4j/data
LOG_DIR      := logs

export OLLAMA_HOST

.DEFAULT_GOAL := help
.PHONY: help up down restart status neo4j-up neo4j-down ollama-up ollama-down ollama-pull search reset-graph ingest repopulate

help: ## Show this help
	@echo "Painting-search services:"
	@grep -E '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# ── All services ──────────────────────────────────────────────────────────────

up: neo4j-up ollama-up ## Start all services (Neo4j + Ollama)
	@echo "→ ready. try:  make search Q=\"predominantly gold, no red\""

down: ollama-down neo4j-down ## Stop all services

restart: down up ## Restart all services

status: ## Show service + model status
	@printf "Neo4j   : "; docker ps --filter name=^/$(NEO4J_NAME)$$ --filter status=running -q | grep -q . \
	  && echo "up   (bolt 7687, browser 7474)" || echo "down"
	@printf "Ollama  : "; curl -fsS -m2 http://$(OLLAMA_HOST)/api/tags >/dev/null 2>&1 \
	  && echo "up   ($(OLLAMA_HOST))" || echo "down"
	@printf "Model   : "; curl -fsS -m2 http://$(OLLAMA_HOST)/api/tags 2>/dev/null | grep -q '$(OLLAMA_MODEL)' \
	  && echo "$(OLLAMA_MODEL) ready" || echo "$(OLLAMA_MODEL) not pulled (run: make ollama-pull)"

# ── Neo4j ─────────────────────────────────────────────────────────────────────

neo4j-up: ## Start Neo4j (creates the container on first run; data persists)
	@docker start $(NEO4J_NAME) >/dev/null 2>&1 && echo "Neo4j: started" || \
	  { docker run -d --name $(NEO4J_NAME) -p 7474:7474 -p 7687:7687 \
	      -e NEO4J_AUTH=none -v $(NEO4J_DATA):/data neo4j:latest >/dev/null \
	    && echo "Neo4j: created + started"; }
	@printf "Neo4j: waiting for bolt :7687 "; \
	 for i in $$(seq 1 30); do \
	   (exec 3<>/dev/tcp/127.0.0.1/7687) 2>/dev/null && { echo "ready"; exit 0; }; \
	   printf "."; sleep 1; \
	 done; echo " timed out"

neo4j-down: ## Stop Neo4j (data is kept in the volume)
	@docker stop $(NEO4J_NAME) >/dev/null 2>&1 && echo "Neo4j: stopped" || echo "Neo4j: not running"

# ── Ollama ────────────────────────────────────────────────────────────────────

ollama-up: ## Start the Ollama server in the background
	@if curl -fsS -m2 http://$(OLLAMA_HOST)/api/tags >/dev/null 2>&1; then \
	   echo "Ollama: already running"; \
	 else \
	   test -x "$(OLLAMA_BIN)" || { echo "Ollama: binary not found at $(OLLAMA_BIN)"; exit 1; }; \
	   mkdir -p $(LOG_DIR); \
	   OLLAMA_HOST=$(OLLAMA_HOST) nohup "$(OLLAMA_BIN)" serve > $(LOG_DIR)/ollama.log 2>&1 & \
	   printf "Ollama: starting "; \
	   for i in $$(seq 1 30); do \
	     curl -fsS -m1 http://$(OLLAMA_HOST)/api/tags >/dev/null 2>&1 && { echo "up (log: $(LOG_DIR)/ollama.log)"; break; }; \
	     printf "."; sleep 1; \
	   done; \
	 fi

ollama-down: ## Stop the Ollama server
	@if pgrep -f "[o]llama serve" >/dev/null 2>&1; then \
	   pkill -f "[o]llama serve" && echo "Ollama: stopped"; \
	 else echo "Ollama: not running"; fi

ollama-pull: ## Download the LLM model (llama3.1:8b)
	@OLLAMA_HOST=$(OLLAMA_HOST) "$(OLLAMA_BIN)" pull $(OLLAMA_MODEL)

# ── Search ────────────────────────────────────────────────────────────────────

search: ## Run a query: make search Q="deep ocean water" [FLAGS="--show-intent --show-cypher"]
	@uv run python scripts/search.py $(FLAGS) $(Q)

# ── Data (graph population) ───────────────────────────────────────────────────

reset-graph: ## Delete all paintings (keeps colours + index) — run before a clean re-ingest
	@uv run python scripts/reset_graph.py

ingest: ## Extract colours from images/images/ into Neo4j
	@uv run python scripts/pipeline.py --images-dir images/images/

repopulate: reset-graph ingest ## Clean-slate re-ingest: reset the graph, then extract
	@echo "→ repopulate complete"
